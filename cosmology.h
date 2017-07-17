#ifndef COSMOLOGY_H
#define COSMOLOGY_H
typedef struct {
    double CMBTemperature;
    double Omega0;  /* matter density in units of the critical density (at z=0) */
    double OmegaCDM; /* CDM density, derived from Omega0 and OmegaBaryon */
    double OmegaG; /* Photon density, derived from T_CMB0 */
    double OmegaK; /* Curvature density, derived from Omega0 and OmegaLambda */
    double OmegaNu0; /* Massless Neutrino density, derived from T_CMB0, useful only if there are no massive neutrino particles */
    double OmegaLambda;  /* vaccum energy density relative to crictical density (at z=0) */
    double OmegaBaryon;  /* baryon density in units of the critical density (at z=0) */
    double HubbleParam;  /* little `h', i.e. Hubble constant in units of 100 km/s/Mpc. */
    int RadiationOn; /* flags whether to include the radiation density in the background */
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

/*Hubble function at scale factor a, in dimensions of All.Hubble*/
double hubble_function(double a);
/* Linear theory growth factor normalized to D(a=1.0) = 1.0. */
double GrowthFactor(double a);
/*Note this is only used in GenIC*/
double F_Omega(double a);

/*Initialise the derived parts of the cosmology*/
void init_cosmology(void);
#endif
