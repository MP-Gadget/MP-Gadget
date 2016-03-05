#ifndef COSMOLOGY_H
#define COSMOLOGY_H

#include <math.h>
#include "allvars.h"

/*With slightly relativistic massive neutrinos, for consistency we need to include radiation.
 * A note on normalisation (as of 08/02/2012):
 * CAMB appears to set Omega_Lambda + Omega_Matter+Omega_K = 1,
 * calculating Omega_K in the code and specifying Omega_Lambda and Omega_Matter in the paramfile.
 * This means that Omega_tot = 1+ Omega_r + Omega_g, effectively
 * making h0 (very) slightly larger than specified.
 */
#ifdef INCLUDE_RADIATION

#define  T_CMB0      2.7255	/* present-day CMB temperature, from Fixsen 2009. */
/*Stefan-Boltzmann constant in cgs units*/
#define STEFAN_BOLTZMANN 5.670373e-5
/* Omega_g = 4 \sigma_B T_{CMB}^4 8 \pi G / (3 c^3 H^2)*/
#define OMEGAG (4*STEFAN_BOLTZMANN*8*M_PI*GRAVITY/(3*C*C*C*HUBBLE*HUBBLE)*pow(T_CMB0,4)/All.HubbleParam/All.HubbleParam)
#if defined NEUTRINOS
    /*Neutrinos are massive and included elsewhere*/
    #define OMEGAR OMEGAG
#else
    /* Note there is a slight correction from 4/11
     * due to the neutrinos being slightly coupled at e+- annihilation.
     * See Mangano et al 2005 (hep-ph/0506164)
     *The correction is (3.046/3)^(1/4), for N_eff = 3.046 */
    #define TNU     (T_CMB0*pow(4/11.,1/3.)*1.00328)              /* Neutrino + antineutrino background temperature in Kelvin */
    /*Neutrinos are included in the radiation*/
    /*For massless neutrinos, rho_nu/rho_g = 7/8 (T_nu/T_cmb)^4 *N_eff, but we absorbed N_eff into T_nu above*/
    #define OMEGANU (OMEGAG*7/8.*pow(TNU/T_CMB0,4)*3)
    /*With massless neutrinos only, add the neutrinos to the radiation*/
    #define OMEGAR (OMEGAG+OMEGANU)
#endif
#else
        /*Default is no radiation*/
        #define OMEGAR 0.
#endif

/* For convenience define OMEGAK*/
#define OMEGAK (1-All.Omega0 - All.OmegaLambda)

/*Hubble function at scale factor a, in dimensions of All.Hubble*/
double hubble_function(double a);

#endif
