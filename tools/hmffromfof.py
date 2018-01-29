# -*- coding: utf-8 -*-
"""This module prints a halo mass function from MP-Gadget's built in FOF tables.
        halo_mass_function: returns dn/dM in M_sun/Mpc^3
"""

from __future__ import division,print_function
import numpy as np
import bigfile


def HMFFromFOF(foftable, h0=False, bins='auto'):
    """Print a conventionally normalised halo mass function from the FOF tables.
    Units returned are:
    dn/dM (M_sun/Mpc^3) (comoving) Note no little-h!
    If h0 == True, units are dn/dM (h^4 M_sun/Mpc^3)
    bins specifies the number of evenly spaced bins if an integer,
    or one of the strings understood by numpy.histogram."""
    bf = bigfile.BigFile(foftable)
    #1 solar in g
    msun_in_g = 1.989e33
    #1 Mpc in cm
    Mpc_in_cm = 3.085678e+24
    #In units of 10^10 M_sun by default.
    try:
        imass_in_g = bf["Header"].attrs["UnitMass_in_g"]
    except KeyError:
        imass_in_g = 1.989e43
    #Length in units of kpc/h by default
    try:
        ilength_in_cm = bf["Header"].attrs["UnitLength_in_cm"]
    except KeyError:
        ilength_in_cm = 3.085678e+21
    hub = bf["Header"].attrs["HubbleParam"]
    box = bf["Header"].attrs["BoxSize"]
    #Convert to Mpc from kpc/h:
    box *= ilength_in_cm / hub / Mpc_in_cm
    masses = bf["FOFGroups/Mass"][:]
    #This is N(M) evenly spaced in log(M)
    NM, Mbins = np.histogram(np.log10(masses), bins=bins)
    #Convert Mbins to Msun
    Mbins = 10**Mbins
    Mbins *= (imass_in_g / msun_in_g)
    #Find dM:
    #This is dn/dM (Msun)
    dndm = NM/(Mbins[1:] - Mbins[:-1])
    Mcent = (Mbins[1:] + Mbins[:-1])/2.
    #Now divide by the volume:
    dndm /= box**3
    if h0:
        dndm /= hub**4
    return Mcent, dndm
