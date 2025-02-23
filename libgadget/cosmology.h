#ifndef COSMOLOGY_H
#define COSMOLOGY_H

#include <stddef.h>
#include "omega_nu_single.h"
#include "utils/unitsystem.h"

typedef struct {
    double CMBTemperature;
    double Omega0;  /* matter density in units of the critical density (at z=0) */
    double OmegaCDM; /* CDM density, derived from Omega0 and OmegaBaryon */
    double OmegaG; /* Photon density, derived from T_CMB0 */
    double OmegaK; /* Curvature density, derived from Omega0 and OmegaLambda */
    double OmegaLambda;  /* vacuum energy density relative to crictical density (at z=0) */
    double Omega_fld; /*Energy density of dark energy fluid at z=0*/
    double w0_fld; /*Dark energy equation of state parameter*/
    double wa_fld; /*Dark energy equation of state evolution parameter*/
    double Omega_ur; /*Extra radiation density: either a sterile neutrino or other dark radiation*/
    double OmegaBaryon;  /* baryon density in units of the critical density (at z=0) */
    double HubbleParam;  /* little `h', i.e. Hubble constant in units of 100 km/s/Mpc. */
    double Hubble; /* 100 km/s/Mpc in internal units*/
    double UnitTime_in_s; /* Internal unit time in seconds*/
    double RhoCrit; /* critical density */
    int RadiationOn; /* flags whether to include the radiation density in the background */
    _omega_nu ONu;   /*Structure for storing massive neutrino densities*/
    double MNu[3]; /*Neutrino masses in eV*/
    int use_class_radiation_convention; /* Boolean that flags which omega_tot to use.
                                           CLASS uses Omega_tot = 1 = sum(Omega_i) but CAMB removes radiation,
                                           and has Omega_tot = 1 = sum(Omega_M + Omega_K + Omega_Lambda)*/
    int MassiveNuLinRespOn; /* Flags that massive neutrinos using the linear
                               response code of Ali-Haimoud & Bird 2013.*/
    int HybridNeutrinosOn; /* Flags that hybrid neutrinos are enabled */
    double HybridVcrit; /* Critical velocity switching between particle
                           and analytic solvers when hybrid neutrinos are on*/
    double HybridNuPartTime; /*!< Redshift at which hybrid neutrinos switch on*/
    double GravInternal;/*!< Gravity-constant in internal units */
    int ComovingIntegrationOn; /* If on then use comoving integration (default 1)*/
    int NonPeriodic; /*If on then remove the periodic boundary condition (default 0)*/
    double Redshift; /* Only use if ComovingIntegrationOn=0 !*/
} Cosmology;

typedef struct {
    size_t bytesize;
    double normfactor;
    size_t size;
    struct {
        double k;
        double Pk;
    } table[];
} FunctionOfK;

void function_of_k_normalize_sigma(FunctionOfK * fk, double R, double sigma);

double function_of_k_eval(FunctionOfK * fk, double k);
double function_of_k_tophat_sigma(FunctionOfK * fk, double R);

/*Hubble function at scale factor a, in dimensions of CP.Hubble*/
double hubble_function(const Cosmology * CP, double a);
/* Linear theory growth factor between astart and aend. */
double GrowthFactor(Cosmology * CP, double astart, double aend);
/*Note this is only used in GenIC*/
double F_Omega(Cosmology * CP, double a);
/* Returns 1 if the neutrino particles are 'tracers', not actively gravitating,
 * and 0 if they are actively gravitating particles.*/
int hybrid_nu_tracer(const Cosmology * CP, double atime);

/*Initialise the derived parts of the cosmology*/
void init_cosmology(Cosmology *CP, double TimeBegin, const struct UnitSystem units);

/* Check and print properties of the derived cosmology */
void check_units(const Cosmology * CP, const struct UnitSystem units);

#endif
