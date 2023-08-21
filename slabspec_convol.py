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

def make_spec_convol(molecule_name, n_col, temp, area, wmax=40, wmin=1, res=1e-4, vturb=None, phi=None, isotopologue_number=1, d_pc=1,
              aupmin=None, eupmax=None, vup=None, swmin=None, parfile=None):

    '''
    Create an IR spectrum for a slab model with given temperature, area, and column density

    Parameters
    ---------
    molecule_name : string
        String identifier for molecule, for example, 'CO', or 'H2O'             
    n_col : float
        Column density, in m^-2
    temp : float
        Temperature of slab model, in K
    area : float
        Area of slab model, in m^2
    wmin : float, optional
        Minimum wavelength of output spectrum, in microns. Defaults to 1 micron.
    wmax : float, optional
        Maximum wavelength of output spectrum, in microns.  Defaults to 40 microns.
    vturb : float, optional
        turbulent velocity in km/s.  Note this is NOT the global velocity distribution.
        If not set (default), the lines are broadened by the thermal speed of molecule given input temperature.
    phi : function, optional
        input method that defines the line profile as a function of velocity in m/s
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

# If no turbulent velocity is given, broaden the lines by the thermal velocity (in m/s)
    if(vturb is None):
        deltav = compute_thermal_velocity(molecule_name, temp)
    else:
        deltav = vturb * 1000

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
    q = _compute_partition_function(molecule_name,temp, isotopologue_number)
    
# Begin calculations                                                                                                       
    afactor = ((aup * gup * n_col)/(8. * np.pi * q *(wn0)**3))   # mks                                                                 
    efactor = h.value * c.value * eup / (k_B.value * temp)
    wnfactor = h.value * c.value * wn0 / (k_B.value * temp)
    phia = 1. / (deltav * np.sqrt(2.0*np.pi))
    efactor2 = hitran_data['eup_k'] / temp
    efactor1 = hitran_data['elower'] * 1.e2 * h.value * c.value / (k_B.value * temp)
    tau0 = afactor * (np.exp(-efactor1) - np.exp(-efactor2)) * phia      # Avoids numerical issues at low T
    w0 = 1e6 / wn0

    # velocity range for each line is -100 to 100 km/s
    dvel = 0.1        # km/s
    nvel = 2001
    vel = (np.arange(0, nvel) - nvel//2) * dvel   # km/s
    vel *= 1000                                   # m/s

    omega = area / (d_pc * pc.value)**2
    fthin = aup * gup * n_col * h.value * c.value * wn0 / (4 * np.pi * q) * np.exp(-efactor) * omega    # Energy/area/time, mks                   

    nbins = int((wmax - wmin) / res)
    totalwave = wmin + (wmax - wmin) * np.arange(nbins) / nbins
    totalflux = np.zeros(nbins)
    totalflux_convol = np.zeros(nbins)

# convolution line profile (user-defined function given as an input)
    phiv = phi(vel)
    phiv /= np.sum(phiv)

    nlines = np.size(hitran_data)
    for i in np.arange(nlines):
        wave0 = 1e6 / wn0[i]
        tau0v = tau0[i] * np.exp(-vel**2 / (2 * deltav**2))
        Intens = 2 * h.value * c.value * wn0[i]**3 / (np.exp(wnfactor[i]) - 1) * (1 - np.exp(-tau0v))
        Intens_convol = convolve(Intens, phiv)
        wave = wave0 * (1 + vel / c.value)
        totalflux += np.interp(totalwave, wave, Intens)
        totalflux_convol += np.interp(totalwave, wave, Intens_convol)

    totalflux *= omega * si2jy
    totalflux_convol *= omega * si2jy

    slabdict={}

# Line params
    hitran_data['tau_peak'] = tau0
    hitran_data['fthin'] = fthin
    hitran_data = _convert_quantum_strings(hitran_data)
    hitran_data = _strip_superfluous_hitran_data(hitran_data)
    slabdict['lineparams'] = hitran_data

# Spectrum
    spectrum_table = Table([totalwave, totalflux, totalflux_convol], names=('wave', 'flux', 'convolflux'),  dtype=('f8', 'f8', 'f8'))
    spectrum_table['wave'].unit = 'micron'
    spectrum_table['flux'].unit = 'Jy'
    spectrum_table['convolflux'].unit = 'Jy'
    slabdict['spectrum'] = spectrum_table

# Model params
    modelparams_table={'area':area*un.meter*un.meter,'temp':temp*un.K,'n_col':n_col/un.meter/un.meter, 'res':res*un.micron, 
                       'deltav':deltav*un.meter/un.s, 'd_pc':d_pc*un.parsec,
                       'isotopologue_number':isotopologue_number, 'molecule_name':molecule_name}
    slabdict['modelparams'] = modelparams_table

    return slabdict
