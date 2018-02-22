#ifndef COSMOLOGY_H
#define COSMOLOGY_H

#include <stddef.h>
#include "kspace-neutrinos/omega_nu_single.h"

typedef struct {
    double CMBTemperature;
    double Omega0;  /* matter density in units of the critical density (at z=0) */
    double OmegaCDM; /* CDM density, derived from Omega0 and OmegaBaryon */
    double OmegaG; /* Photon density, derived from T_CMB0 */
    double OmegaK; /* Curvature density, derived from Omega0 and OmegaLambda */
    double OmegaLambda;  /* vaccum energy density relative to crictical density (at z=0) */
    double Omega_fld; /*Energy density of dark energy fluid at z=0*/
    double w0_fld; /*Dark energy equation of state parameter*/
    double wa_fld; /*Dark energy equation of state evolution parameter*/
    double OmegaBaryon;  /* baryon density in units of the critical density (at z=0) */
    double HubbleParam;  /* little `h', i.e. Hubble constant in units of 100 km/s/Mpc. */
    double Hubble; /* 100 km/s/Mpc in whatever units we want*/
    int RadiationOn; /* flags whether to include the radiation density in the background */
    _omega_nu ONu;   /*Structure for storing massive neutrino densities*/
    double MNu[3]; /*Neutrino masses in eV*/
} Cosmology;

typedef struct {
    size_t bytesize;
    double normfactor;
    size_t size;
    struct {
        double k;
        double P;
    } table[];
} FunctionOfK;

void function_of_k_normalize_sigma(FunctionOfK * fk, double R, double sigma);

double function_of_k_eval(FunctionOfK * fk, double k);
double function_of_k_tophat_sigma(FunctionOfK * fk, double R);

/*Hubble function at scale factor a, in dimensions of CP.Hubble*/
double hubble_function(double a);
/* Linear theory growth factor between astart and aend. */
double GrowthFactor(double astart, double aend);
/*Note this is only used in GenIC*/
double F_Omega(double a);

/*Initialise the derived parts of the cosmology*/
void init_cosmology(Cosmology *CP, double TimeBegin, double box);
#endif
