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

def make_spec_twolayer(molecule_name, n_col1, temp1, fwhm1, n_col2, temp2, fwhm2, area, continuum, wmax=40, wmin=1, res=1e-4, isotopologue_number=1, d_pc=1,
              aupmin=None, eupmax=None, vup=None, swmin=None, parfile=None):

    '''
    Create an IR spectrum for a two layer slab model to model profiles with emission and absorption
    Each layer has a given temperature, column density, and FWHM

    Parameters
    ---------
    molecule_name : string
        String identifier for molecule, for example, 'CO', or 'H2O'             
    n_col1, ncol2 : float
        Column densities, in m^-2
    temp1, temp2 : float
        Temperatures of each slab, in K
    fwhm1, fwhm2 : float, optional
        FWHM of local turbulent velocity in km/s.  Note this is NOT the global velocity distribution.
    area : float
        Area of slab model, in m^2 (same for both layers)
    continuum: float
        Background (stellar) continuum in Jy; the absorption may go below this value
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
          modelparams : model parameters: Area, column density, temperature, local velocity, convolution fwhm
    '''
    si2jy = 1e26                  # SI to Jy flux conversion factor

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
    q1 = _compute_partition_function(molecule_name, temp1, isotopologue_number)
    q2 = _compute_partition_function(molecule_name, temp2, isotopologue_number)
    
# Begin calculations                                                                                                       
    afactor1 = ((aup * gup * n_col1)/(8. * np.pi * q1 *(wn0)**3))   # mks                                                                 
    afactor2 = ((aup * gup * n_col2)/(8. * np.pi * q2 *(wn0)**3))   # mks                                                                 
    wnfactor1 = h.value * c.value * wn0 / (k_B.value * temp1)
    wnfactor2 = h.value * c.value * wn0 / (k_B.value * temp2)
    phia1 = 1. / (fwhm1 * np.sqrt(2.0*np.pi))
    phia2 = 1. / (fwhm2 * np.sqrt(2.0*np.pi))
    efactor1_2 = hitran_data['eup_k'] / temp1
    efactor1_1 = hitran_data['elower'] * 1.e2 * h.value * c.value / (k_B.value * temp1)
    efactor2_2 = hitran_data['eup_k'] / temp2
    efactor2_1 = hitran_data['elower'] * 1.e2 * h.value * c.value / (k_B.value * temp2)
    tau1 = afactor1 * (np.exp(-efactor1_1) - np.exp(-efactor1_2)) * phia1
    tau2 = afactor2 * (np.exp(-efactor2_1) - np.exp(-efactor2_2)) * phia2

    # velocity range for each line is -1000 to 1000 km/s
    # need to make much greater than fwhm so that the intensity really goes to zero at the edges
    # otherwise the interpolation from velocity to wavelength leaves a residual that adds up significantly over all lines
    dvel = 0.1        # km/s
    nvel = 20001
    vel = (np.arange(0, nvel) - nvel//2) * dvel   # km/s
    vel *= 1000                                   # m/s

    omega = area / (d_pc * pc.value)**2
    Intens0 = continuum / (omega * si2jy)

    nbins = int((wmax - wmin) / res)
    totalwave = wmin + (wmax - wmin) * np.arange(nbins) / nbins
    totalflux = np.zeros(nbins)

    nlines = np.size(hitran_data)
    for i in np.arange(nlines):
        wave0 = 1e6 / wn0[i]
        tau1v = tau1[i] * np.exp(-vel**2 / (2 * (1000*fwhm1/2.355)**2))
        tau2v = tau2[i] * np.exp(-vel**2 / (2 * (1000*fwhm2/2.355)**2))
        Intens1 = 2 * h.value * c.value * wn0[i]**3 / (np.exp(wnfactor1[i]) - 1) * (1 - np.exp(-tau1v))
        Intens2 = 2 * h.value * c.value * wn0[i]**3 / (np.exp(wnfactor2[i]) - 1) * (1 - np.exp(-tau2v))
        Intens = (Intens0 + Intens1) * np.exp(-tau2v) + Intens2
        wave = wave0 * (1 + vel / c.value)

        # add up the contribution from this line without the continuum (otherwise it gets added multiple times)
        totalflux += np.interp(totalwave, wave, Intens-Intens0)

    # add the continuum back in and convert from intensity to flux in Jy
    totalflux = (Intens0 + totalflux) * omega * si2jy
    slabdict={}

# Line params
    hitran_data['tau_peak1'] = tau1
    hitran_data['tau_peak2'] = tau2
    hitran_data = _convert_quantum_strings(hitran_data)
    hitran_data = _strip_superfluous_hitran_data(hitran_data)
    slabdict['lineparams'] = hitran_data

# Spectrum
    spectrum_table = Table([totalwave, totalflux], names=('wave', 'flux'),  dtype=('f8', 'f8'))
    spectrum_table['wave'].unit = 'micron'
    spectrum_table['flux'].unit = 'Jy'
    slabdict['spectrum'] = spectrum_table

# Model params
    modelparams_table={'area':area*un.meter*un.meter,
                       'temp1':temp1*un.K, 'n_col1':n_col1/un.meter/un.meter, 'fwhm1':fwhm1*un.meter/un.s,
                       'temp2':temp2*un.K, 'n_col2':n_col2/un.meter/un.meter, 'fwhm2':fwhm2*un.meter/un.s,
                       'continuum': continuum*un.Jy,
                       'res':res*un.micron, 
                       'd_pc':d_pc*un.parsec,
                       'isotopologue_number':isotopologue_number, 'molecule_name':molecule_name}
    slabdict['modelparams'] = modelparams_table

    return slabdict
