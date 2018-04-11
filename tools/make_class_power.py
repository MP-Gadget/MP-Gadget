"""This module creates a matter power spectrum using Classylss,
a python interface to the CLASS Boltzmann code.

See:
http://classylss.readthedocs.io/en/stable/
http://class-code.net/
It parses an MP-GenIC parameter file and generates a matter power spectrum
and transfer function at the initial redshift. It generates a second transfer
function at a slightly lower redshift to allow computing the growth function.
Files are saved where MP-GenIC expects to read them. Existing files will not be over-written.

Script should be compatible with python 2.7 and 3.

Cite CLASS paper:
 D. Blas, J. Lesgourgues, T. Tram, arXiv:1104.2933 [astro-ph.CO], JCAP 1107 (2011) 034
Call with:
    python make_class_power.py <MP-GenIC parameter file> <external power spectrum file>
    where the second external power spectrum file is optional and is a primordial power spectrum for CLASS."""

from __future__ import print_function
import sys
import math
import os.path
import numpy as np
import argparse
import classylss
import classylss.binding as CLASS
import configobj
import validate

GenICconfigspec = """
FileWithInputSpectrum = string(default='')
FileWithTransferFunction = string(default='')
FileWithFutureTransferFunction = string(default='')
Ngrid = integer(min=0)
BoxSize = float(min=0)
Omega0 = float(0,1)
OmegaLambda = float(0,1)
OmegaBaryon = float(0,1,default=0.0486)
HubbleParam = float(0,2)
Redshift = float(0,1100)
Sigma8 = float(default=-1)
InputPowerRedshift = float(default=-1)
InputFutureRedshift = float()
DifferentTransferFunctions = integer(0,1, default=1)
InputSpectrum_UnitLength_in_cm  = float(default=3.085678e24)
UnitLength_in_cm  = float(default=3.085678e21)
Omega_fld = float(0,1,default=0)
w0_fld = float(default=-1)
wa_fld = float(default=0)
MNue = float(min=0, default=0)
MNum = float(min=0, default=0)
MNut = float(min=0, default=0)
MWDM_Therm = float(min=0, default=0)
PrimordialIndex = float(default=0.971)
PrimordialAmp = float(default=2.215e-9)
CMBTemperature = float(default=2.7255)""".split('\n')

def _check_genic_config(config):
    """Check that the MP-GenIC config file is sensible for running CLASS on."""
    vtor = validate.Validator()
    config.validate(vtor)
    filekeys = ['FileWithInputSpectrum', ]
    if config['DifferentTransferFunctions'] == 1.:
        filekeys += ['FileWithTransferFunction', 'FileWithFutureTransferFunction']
    for ff in filekeys:
        if config[ff] == '':
            raise IOError("No savefile specified for ",ff)
        if os.path.exists(config[ff]):
            raise IOError("Refusing to write to existing file: ",config[ff])

    #Make sure MP-GenIC expects input in Mpc/h!
    iinMpc = config['InputSpectrum_UnitLength_in_cm']/3.085678e24
    if abs(iinMpc-1) > 1e-6:
        raise AssertionError("CLASS outputs power spectrum in Mpc/h units, MP-GenIC expects %.5f Mpc/h." % iinMpc)
    #Check unsupported configurations
    if config['MWDM_Therm'] > 0:
        raise ValueError("Warm dark matter power spectrum cutoff not yet supported.")
    if config['DifferentTransferFunctions'] == 1.:
        if config['InputPowerRedshift'] >= 0:
            raise ValueError("Rescaling with different transfer functions not supported.")

def _build_cosmology_params(config):
    """Build a correctly-named-for-class set of cosmology parameters from the MP-GenIC config file."""
    #Class takes omega_m h^2 as parameters
    h0 = config['HubbleParam']
    omeganu = (config['MNue'] + config['MNum'] + config['MNut'])/93.14/h0**2
    if config['OmegaBaryon'] < 0.001:
        config['OmegaBaryon'] = 0.0486
    ocdm = config['Omega0'] - config['OmegaBaryon'] - omeganu
    omegak = 1-config['OmegaLambda']-config['Omega0']
    gparams = {'h':config['HubbleParam'], 'Omega_cdm':ocdm,'Omega_b':config['OmegaBaryon'], 'Omega_k':omegak, 'n_s': config['PrimordialIndex'],'T_cmb':config["CMBTemperature"]}
    #One may specify either OmegaLambda or Omega_fld,
    #and the other is worked out by summing all matter to unity.
    #Specify Omega_fld even if we have Lambda, to avoid floating point.
    gparams['Omega_fld'] = config['Omega_fld']
    if config['Omega_fld'] > 0:
        gparams['w0_fld'] = config['w0_fld']
        gparams['wa_fld'] = config['wa_fld']
    #Set up massive neutrinos
    if omeganu > 0:
        gparams['m_ncdm'] = '%.2f,%.2f,%.2f' % (config['MNue'], config['MNum'], config['MNut'])
        gparams['N_ncdm'] = 3
        #Neutrino accuracy: Default pk_ref.pre has tol_ncdm_* = 1e-10,
        #which takes 45 minutes (!) on my laptop.
        #tol_ncdm_* = 1e-8 takes 20 minutes and is machine-accurate.
        #Default parameters are fast but off by 2%.
        #I chose 1e-5, which takes 6 minutes and is accurate to 1e-5
        gparams['tol_ncdm_newtonian'] = 1e-5
        gparams['tol_ncdm_synchronous'] = 1e-5
        gparams['tol_ncdm_bg'] = 1e-10
        gparams['l_max_ncdm'] = 50
    #Power spectrum amplitude
    if config['Sigma8'] > 0:
        gparams['sigma8'] = config['Sigma8']
    else:
        #Pivot scale is by default 0.05 1/Mpc! This number is NOT what is reported by Planck.
        gparams['A_s'] = config["PrimordialAmp"]
    return gparams

def make_class_power(paramfile, external_pk = None, extraz=None):
    """Main routine: parses a parameter file and makes a matter power spectrum.
    Will not over-write power spectra if already present.
    Options are loaded from the MP-GenIC parameter file.
    Supported:
        - Omega_fld and DE parameters.
        - Massive neutrinos.
        - Using Sigma8 to set the power spectrum scale.
        - Different transfer functions.
    Not supported:
        - Warm dark matter power spectra.
        - Rescaling with different transfer functions."""
    config = configobj.ConfigObj(infile=paramfile, configspec=GenICconfigspec, file_error=True)
    #Input sanitisation
    _check_genic_config(config)

    #Precision
    pre_params = {'tol_background_integration': 1e-9, 'tol_perturb_integration' : 1.e-7, 'tol_thermo_integration':1.e-5, 'k_per_decade_for_pk': 20,'k_per_decade_for_bao':  200, 'neglect_CMB_sources_below_visibility' : 1.e-30, 'transfer_neglect_late_source': 3000., 'l_max_g' : 50, 'l_max_ur':150}

    #Important! Densities are in synchronous gauge!
    pre_params['gauge'] = 'synchronous'

    gparams = _build_cosmology_params(config)
    pre_params.update(gparams)
    redshift = config['Redshift']
    if config['InputPowerRedshift'] >= 0:
        redshift = config['InputPowerRedshift']
    outputs = np.array([redshift, config['InputFutureRedshift']])
    if extraz is not None:
        outputs = np.concatenate([outputs, extraz])
    #Pass options for the power spectrum
    boxmpc = config['BoxSize'] / config['InputSpectrum_UnitLength_in_cm'] * config['UnitLength_in_cm']
    maxk = 2*math.pi/boxmpc*config['Ngrid']*16
    powerparams = {'output': 'dTk vTk mPk', 'P_k_max_h/Mpc' : maxk, "z_max_pk" : 1+np.max(outputs),'z_pk': outputs}
    pre_params.update(powerparams)

    #Specify an external primordial power spectrum
    if external_pk is not None:
        pre_params['P_k_ini'] = "external_pk"
        pre_params["command"] = "cat ",external_pk

    #Make the power spectra module
    engine = CLASS.ClassEngine(pre_params)
    powspec = CLASS.Spectra(engine)
    bg = CLASS.Background(engine)

    #Save directory
    sdir = os.path.split(paramfile)[0]
    #Get and save the transfer functions if needed
    trans = powspec.get_transfer(z=redshift)
    if config['DifferentTransferFunctions'] == 1.:
        tfile = os.path.join(sdir, config['FileWithTransferFunction'])
        save_transfer(trans, tfile, bg, redshift)
        transfut = powspec.get_transfer(z=config['InputFutureRedshift'])
        tfile = os.path.join(sdir, config['FileWithFutureTransferFunction'])
        save_transfer(transfut, tfile, bg, config['InputFutureRedshift'])
    #Get and save the matter power spectrum
    pk_lin = powspec.get_pklin(k=trans['k'], z=redshift)
    pkfile = os.path.join(sdir, config['FileWithInputSpectrum'])
    if os.path.exists(pkfile):
        raise IOError("Refusing to write to existing file: ",pkfile)
    np.savetxt(pkfile, np.vstack([trans['k'], pk_lin]).T)
    if extraz is not None:
        for red in extraz:
            trans = powspec.get_transfer(z=red)
            tfile = os.path.join(sdir, config['FileWithTransferFunction']+"-"+str(red))
            save_transfer(trans, tfile, bg, red)
            #Get and save the matter power spectrum
            pk_lin = powspec.get_pklin(k=trans['k'], z=red)
            pkfile = os.path.join(sdir, config['FileWithInputSpectrum']+"-"+str(red))
            if os.path.exists(pkfile):
                raise IOError("Refusing to write to existing file: ",pkfile)
            np.savetxt(pkfile, np.vstack([trans['k'], pk_lin]).T)

def save_transfer(transfer, transferfile, bg, redshift):
    """Save a transfer function. Note we save the CAMB FORMATTED transfer functions.
    These can be generated from CLASS by passing the 'format = camb' on the command line.
    The transfer functions differ by:
        T_CAMB(k) = -T_CLASS(k)/k^2 """
    if os.path.exists(transferfile):
        raise IOError("Refusing to write to existing file: ",transferfile)
    #This format matches the default output by CAMB and CLASS command lines.
    #Some entries may be zero sometimes
    kk = transfer['k']
    ftrans = np.zeros((np.size(transfer['k']), 13))
    ftrans[:,0] = kk
    ftrans[:,1] = -1*transfer['d_cdm']/kk**2
    ftrans[:,2] = -1*transfer['d_b']/kk**2
    ftrans[:,3] = -1*transfer['d_g']/kk**2
    ftrans[:,4] = -1*transfer['d_ur']/kk**2
    #This will fail if there are no massive neutrinos present
    try:
        #We use the most massive neutrino species, since these
        #are used for initialising the particle neutrinos.
        ftrans[:,5] = -1*transfer['d_ncdm[2]']/kk**2
    except ValueError:
        pass
    omegacdm = bg.Omega_cdm(redshift)
    omegab = bg.Omega_b(redshift)
    omeganu = bg.Omega_ncdm(redshift)
    #Note that the CLASS total transfer function apparently includes radiation!
    #We do not want this for the matter power: we want CDM + b + massive-neutrino.
    ftrans[:,6] = -1*(omegacdm *transfer['d_cdm'] + omegab * transfer['d_b'] + omeganu * ftrans[:,5])/(omeganu + omegacdm + omegab)/kk**2
    #The CDM+baryon weighted density.
    ftrans[:,7] = -1*(omegacdm *transfer['d_cdm'] + omegab * transfer['d_b'])/(omegacdm + omegab)/kk**2
    ftrans[:,8] = -1*transfer['d_tot']/kk**2
    #The velocity transfer functions: these are chosen to match CAMB output
    #Weyl potential
    ftrans[:,9] = (transfer['phi'] + transfer['psi'])/2
    #Velocity transfer: this is theta, but I don't know how to match CAMB.
#     hub = bg.hubble_function(redshift)
#     ftrans[:,10] = -transfer['t_cdm']/kk**2/hub
#     ftrans[:,11] = -transfer['t_b']/kk**2/hub
#     ftrans[:,12] = -transfer['t_b']/kk**2/hub + transfer['t_cdm']/kk**2/hub
    np.savetxt(transferfile, ftrans)

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('paramfile', type=str, help='genic paramfile')
    parser.add_argument('--extpk', type=str, help='optional external primordial power spectrum',required=False)
    parser.add_argument('--extraz', type=float,nargs='*', help='optional external primordial power spectrum',required=False)
    args = parser.parse_args()
    make_class_power(args.paramfile, args.extpk, args.extraz)
