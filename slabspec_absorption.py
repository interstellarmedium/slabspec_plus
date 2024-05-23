import numpy as np
import urllib
import pandas as pd

import sys

from astropy.table import Table
from astropy import units as un
from astropy.io import fits
from astropy.constants import c,h, k_B, G, M_sun, au, pc, u
from astropy.convolution import Gaussian1DKernel, convolve

from spectools_ir.utils import _check_hitran
from spectools_ir.utils import fwhm_to_sigma, sigma_to_fwhm, compute_thermal_velocity, extract_hitran_data
from spectools_ir.utils import  get_molecule_identifier, get_global_identifier, spec_convol, extract_hitran_from_par
from spectools_ir.slabspec.slabspec import _compute_partition_function
from spectools_ir.slabspec.helpers import _strip_superfluous_hitran_data, _convert_quantum_strings

def make_spec_absorption(molecule_name, n_col, temp, v, fwhm, continuum, wmax=40, wmin=1, res=1e-4, isotopologue_number=1, d_pc=1,
              aupmin=None, eupmax=None, vup=None, swmin=None, parfile=None):

    '''
    Create an IR spectrum for a two layer slab model to model profiles with emission and absorption
    Each layer has a given temperature, column density, and FWHM

    Parameters
    ---------
    molecule_name : string
        String identifier for molecule, for example, 'CO', or 'H2O'
    n_col : float
        Column density of absorbing layer, in m^-2
    temp : float
        Temperatures of absorbing layer, in K
    v : float
        velocity of absorbing layer in km/s
    fwhm : float, optional
        FWHM of local turbulent velocity in km/s.  Note this is NOT the global velocity distribution.
    continuum: float
        Background (stellar) continuum in Jy
    wmin : float, optional
        Minimum wavelength of output spectrum, in microns. Defaults to 1 micron.
    wmax : float, optional
        Maximum wavelength of output spectrum, in microns.  Defaults to 40 microns.
    isotopologue_number : float, optional
        Number representing isotopologue (1=most common, 2=next most common, etc.)
    d_pc : float, optional
        Distance to slab, in units of pc, for computing observed flux density.  Defaults to 1 pc.
    aupmin : float, optional
        Minimum Einstein-A coefficient for transitions
    swmin : float, optional
        Minimum line strength for transitions
    res : float, optional
        max resolution of spectrum, in microns.  Must be significantly higher than observed spectrum for correct calculation.
        Defaults to 1e-4.
    eupmax : float, optional
        Maximum energy of transitions to consider, in K
    vup : float, optional
        Optional parameter to restrict output to certain upper level vibrational states.  Only works if 'Vp' field is a single integer.

    Returns
    --------
    slabdict : dictionary
        Dictionary includes two astropy tables: 
          lineparams : line parameters from HITRAN, integrated line fluxes, peak tau
          spectrum : wavelength, flux, convolflux, tau
        and two dictionaries
          lines : wave_arr (in microns), flux_arr (in mks), velocity (in km/s) - for plotting individual lines
          modelparams : model parameters: column density, temperature, local velocity, convolution fwhm
    '''

# Test whether molecule is in HITRAN database.  If not, check for parfile and warn.
    database =_check_hitran(molecule_name)
    if((database == 'exomol') & (parfile is None)):
        print('This molecule is not in the HITRAN database.  You must provide a HITRAN-format parfile for this molecule.  Exiting.')
        sys.exit()
    if(database is None):
        print('This molecule is not covered by this code at this time.  Exiting.')
        sys.exit()

# Read HITRAN data
    if(parfile is not None):
        hitran_data = extract_hitran_from_par(parfile, aupmin=aupmin, eupmax=eupmax, isotopologue_number=isotopologue_number,
            vup=vup, wavemin=wmin, wavemax=wmax)
    else:  #parfile not provided.  Read using extract_hitran_data
        try:    
            hitran_data = extract_hitran_data(molecule_name, wmin,wmax, isotopologue_number=isotopologue_number,
                eupmax=eupmax, aupmin=aupmin, swmin=swmin, vup=vup)
        except:
            print("astroquery call to HITRAN failed. This can happen when your molecule does not have any lines in the requested wavelength region")
            sys.exit(1)
           
    wn0 = hitran_data['wn'] * 1e2        # now m-1
    aup = hitran_data['a']
    eup = (hitran_data['elower'] + hitran_data['wn']) * 1e2    # now m-1                                                             
    gup = hitran_data['gp']

# Compute partition function
    Q = _compute_partition_function(molecule_name, temp, isotopologue_number)
    
# Begin calculations
    sigma_v = fwhm / np.sqrt(8*np.log(2))                        # km/s
    afactor = ((aup * gup * n_col)/(8. * np.pi * Q *(wn0)**3))   # mks                                                                 
    wnfactor = h.value * c.value * wn0 / (k_B.value * temp)
    phia = 1. / (np.sqrt(2.0*np.pi) * sigma_v * 1e3)             # mks
    efactor2 = hitran_data['eup_k'] / temp
    efactor1 = hitran_data['elower'] * 1.e2 * h.value * c.value / (k_B.value * temp)
    tau = afactor * (np.exp(-efactor1) - np.exp(-efactor2)) * phia

    # velocity range for each line is -500 to 500 km/s
    # need to make much greater than fwhm so that the intensity really goes to zero at the edges
    # otherwise the interpolation from velocity to wavelength leaves a residual that adds up significantly over all lines
    dvel = 0.1        # km/s
    nvel = 10001
    vel = (np.arange(0, nvel) - nvel//2) * dvel   # km/s

    nbins = int((wmax - wmin) / res)
    totalwave = wmin + (wmax - wmin) * np.arange(nbins) / nbins
    totalflux = np.zeros(nbins)

    nlines = np.size(hitran_data)
    for i in np.arange(nlines):
        wave0 = 1e6 / wn0[i]
        tauv = tau[i] * np.exp(-(vel-v)**2 / (2 * sigma_v**2))
        flux = continuum * np.exp(-tauv)
        wave = wave0 * (1 + 1000 * vel / c.value)
        #print(f'{wave0:6.4f}, {np.max(tauv):5.3f}')

        # add up the contribution from this line without the continuum (otherwise it gets added multiple times)
        totalflux += np.interp(totalwave, wave, flux-continuum)

    # add the continuum back in and convert from intensity to flux in Jy
    totalflux = (continuum + totalflux)
    slabdict={}

# Line params
    hitran_data['tau_peak'] = tau
    hitran_data = _convert_quantum_strings(hitran_data)
    hitran_data = _strip_superfluous_hitran_data(hitran_data)
    slabdict['lineparams'] = hitran_data

# Spectrum
    spectrum_table = Table([totalwave, totalflux], names=('wave', 'flux'),  dtype=('f8', 'f8'))
    spectrum_table['wave'].unit = 'micron'
    spectrum_table['flux'].unit = 'Jy'
    slabdict['spectrum'] = spectrum_table

# Model params
    modelparams_table={'temp':temp*un.K, 'n_col':n_col/un.meter/un.meter, 'fwhm':fwhm*un.meter/un.s,
                       'continuum': continuum*un.Jy,
                       'res':res*un.micron, 
                       'd_pc':d_pc*un.parsec,
                       'isotopologue_number':isotopologue_number, 'molecule_name':molecule_name}
    slabdict['modelparams'] = modelparams_table

    return slabdict
