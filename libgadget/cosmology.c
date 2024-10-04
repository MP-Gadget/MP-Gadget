#include <math.h>
#include <boost/numeric/odeint.hpp>
#include "cosmology.h"
#include "physconst.h"
#include "utils.h"
#include "timefac.h"

/*Stefan-Boltzmann constant in cgs units*/
#define  STEFAN_BOLTZMANN 5.670373e-5

static inline double OmegaFLD(const Cosmology * CP, const double a);

void init_cosmology(Cosmology * CP, const double TimeBegin, const struct UnitSystem units)
{
    CP->Hubble = HUBBLE * units.UnitTime_in_s;
    CP->UnitTime_in_s = units.UnitTime_in_s;
    CP->GravInternal = GRAVITY / pow(units.UnitLength_in_cm, 3) * units.UnitMass_in_g * pow(units.UnitTime_in_s, 2);

    CP->RhoCrit = 3.0 * CP->Hubble * CP->Hubble / (8.0 * M_PI * CP->GravInternal);  // in internal units

    /* Omega_g = 4 \sigma_B T_{CMB}^4 8 \pi G / (3 c^3 H^2) */

    CP->OmegaG = 4 * STEFAN_BOLTZMANN
                  * pow(CP->CMBTemperature, 4)
                  * (8 * M_PI * GRAVITY)
                  / (3*pow(LIGHTCGS, 3)*HUBBLE*HUBBLE)
                  / (CP->HubbleParam*CP->HubbleParam);

    init_omega_nu(&CP->ONu, CP->MNu, TimeBegin, CP->HubbleParam, CP->CMBTemperature);
    /*Initialise the hybrid neutrinos, after Omega_nu*/
    if(CP->HybridNeutrinosOn)
        init_hybrid_nu(&CP->ONu.hybnu, CP->MNu, CP->HybridVcrit, LIGHTCGS/1e5, CP->HybridNuPartTime, CP->ONu.kBtnu);

    CP->OmegaCDM = CP->Omega0 - CP->OmegaBaryon;
    /* Neutrinos will be included in Omega0, if massive.
     * This ensures that OmegaCDM contains only non-relativistic species.*/
    if(CP->MNu[0] + CP->MNu[1] + CP->MNu[2] > 0) {
        CP->OmegaCDM -= get_omega_nu(&CP->ONu, 1);
    }

    /*With slightly relativistic massive neutrinos, for consistency we need to include radiation.
     * A note on normalisation (as of 08/02/2012):
     * CAMB appears to set Omega_Lambda + Omega_Matter+Omega_K = 1,
     * calculating Omega_K in the code and specifying Omega_Lambda and Omega_Matter in the paramfile.
     * This means that Omega_tot = 1+ Omega_r + Omega_g, effectively
     * making h0 (very) slightly larger than specified, and the Universe is no longer flat!
     * CLASS just sets Omega_Lambda + Omega_Matter+Omega_K + Omega_g + Omega_ur = 1
     */
    CP->OmegaK = 1.0 - CP->Omega0 - CP->OmegaLambda - CP->Omega_fld;
    if(CP->use_class_radiation_convention) {
        CP->OmegaK = 1.0 - CP->OmegaCDM - CP->OmegaBaryon - CP->OmegaLambda - CP->Omega_fld - CP->Omega_ur - CP->OmegaG - get_omega_nu(&CP->ONu, 1);
    }
}

/* Returns 1 if the neutrino particles are 'tracers', not actively gravitating,
 * and 0 if they are actively gravitating particles.*/
int hybrid_nu_tracer(const Cosmology * CP, double atime)
{
    return CP->HybridNeutrinosOn && (atime <= CP->HybridNuPartTime);
}
/*Hubble function at scale factor a, in dimensions of CP.Hubble*/
double hubble_function(const Cosmology * CP, double a)
{

    double hubble_a;

    /* first do the terms in SQRT */
    hubble_a = CP->OmegaLambda;

    hubble_a += OmegaFLD(CP, a);
    hubble_a += CP->OmegaK / (a * a);
    hubble_a += (CP->OmegaCDM + CP->OmegaBaryon) / (a * a * a);

    if(CP->RadiationOn) {
        hubble_a += CP->OmegaG / (a * a * a * a);
        hubble_a += get_omega_nu(&CP->ONu, a);
    }
    hubble_a += CP->Omega_ur/(a*a*a*a);
    /* Now finish it up. */
    hubble_a = CP->Hubble * sqrt(hubble_a);
    return (hubble_a);
}

static double growth(Cosmology * CP, double a, double *dDda);

double GrowthFactor(Cosmology * CP, double astart, double aend)
{
    return growth(CP, astart, NULL) / growth(CP, aend, NULL);
}

int growth_ode(double a, const double yy[], double dyda[], void * params)
{
    Cosmology * CP = (Cosmology *) params;
    const double hub = hubble_function(CP, a)/CP->Hubble;
    dyda[0] = yy[1]/pow(a,3)/hub;
    /*Only use gravitating part*/
    /* Note: we do not include neutrinos
     * here as they are free-streaming at the initial time.
     * This is not right if our box is very large and thus overlaps
     * with their free-streaming scale. In that case the growth factor will be scale-dependent
     * and we need to numerically differentiate. In practice the box will either be larger
     * than the horizon, and so need radiation perturbations, or the neutrino
     * mass will be larger than current constraints allow, so we just warn for now.*/
    dyda[1] = yy[0] * 1.5 * a * (CP->OmegaCDM + CP->OmegaBaryon)/(a*a*a) / hub;
    return GSL_SUCCESS;
}

// Define the ODE system for the growth factor
void growth_ode(const std::vector<double> &yy, std::vector<double> &dyda, double a, void * params)
{
    Cosmology * CP = (Cosmology *) params;
    const double hub = hubble_function(CP, a) / CP->Hubble;

    dyda[0] = yy[1] / pow(a, 3) / hub;
    /*Only use gravitating part*/
    /* Note: we do not include neutrinos
     * here as they are free-streaming at the initial time.
     * This is not right if our box is very large and thus overlaps
     * with their free-streaming scale. In that case the growth factor will be scale-dependent
     * and we need to numerically differentiate. In practice the box will either be larger
     * than the horizon, and so need radiation perturbations, or the neutrino
     * mass will be larger than current constraints allow, so we just warn for now.*/
    dyda[1] = yy[0] * 1.5 * a * (CP->OmegaCDM + CP->OmegaBaryon) / (a * a * a) / hub;
}

/** The growth function is given as a 2nd order DE in Peacock 1999, Cosmological Physics.
 * D'' + a'/a D' - 1.5 * (a'/a)^2 D = 0
 * 1/a (a D')' - 1.5 (a'/a)^2 D
 * where ' is d/d tau = a^2 H d/da
 * Define F = a^3 H dD/da
 * and we have: dF/da = 1.5 a H D
 */

double growth(Cosmology *CP, double a, double *dDda)
{
    using namespace boost::numeric::odeint;

    // Define a default start time (scale factor)
    double curtime = 1e-5;

    // Adjust `curtime` if `a` is smaller than the default
    if (a < curtime) {
        curtime = a / 10.0;  // Ensure `curtime` is smaller than the target `a`
    }

    // Initial conditions for the growth factor
    std::vector<double> yinit(2);

    // Initial conditions at curtime: [D(curtime), D'(curtime)]
    yinit[0] = 1.5 * (CP->OmegaCDM + CP->OmegaBaryon) / (curtime * curtime);  
    yinit[1] = pow(curtime, 3) * hubble_function(CP, curtime) / CP->Hubble *
               1.5 * (CP->OmegaCDM + CP->OmegaBaryon) / (curtime * curtime * curtime); 

    // Include radiation if enabled
    if (CP->RadiationOn) {
        yinit[0] += CP->OmegaG / pow(curtime, 4) + get_omega_nu(&CP->ONu, curtime);
    }

    // Define the ODE system (as a lambda function)
    auto growth_system = [&CP](const std::vector<double> &yy, std::vector<double> &dyda, double a) {
        growth_ode(yy, dyda, a, CP);
    };

    // Use Boost's Runge-Kutta-Fehlberg (RKF45) adaptive step-size integrator
    runge_kutta_cash_karp54<std::vector<double>> stepper;
    double abs_error = 1e-8;
    double rel_error = 1e-8;
    double step_size = 1e-5;

    try {
        // Integrate the ODE from curtime (curtime) to the given `a`
        integrate_adaptive(make_controlled(abs_error, rel_error, stepper),
                           growth_system, yinit, curtime, a, step_size);
    } catch (...) {
        endrun(1, "Boost ODE solver failed during integration\n");
    }

    // If the derivative is needed, store it in dDda
    if (dDda) {
        *dDda = yinit[1] / pow(a, 3) / (hubble_function(CP, a) / CP->Hubble);
    }

    // Return the growth factor D(a)
    return yinit[0];
}
/*
 * This is the Zeldovich approximation prefactor,
 * f1 = d ln D1 / dlna = a / D (dD/da)
 */
double F_Omega(Cosmology * CP, double a)
{
    double dD1da=0;
    double D1 = growth(CP, a, &dD1da);
    return a / D1 * dD1da;
}

/*Dark energy density as a function of time:
 * OmegaFLD(a)  ~ exp(-3 int((1+w(a))/a da)^a_1
 * and w(a) = w0 + (1-a) wa*/
static inline double OmegaFLD(const Cosmology * CP, const double a)
{
    if(CP->Omega_fld == 0.)
        return 0;
    return CP->Omega_fld * pow(a, -3 * (1 + CP->w0_fld + CP->wa_fld))*exp(-3*CP->wa_fld*(1-a));
}

struct sigma2_params
{
    FunctionOfK * fk;
    double R;
};

static double sigma2_int(double k, void * p)
{
    struct sigma2_params * params = (struct sigma2_params *) p;
    FunctionOfK * fk = params->fk;
    const double R = params->R;
    double kr, kr3, kr2, w, x;

    kr = R * k;
    kr2 = kr * kr;
    kr3 = kr2 * kr;

    if(kr < 1e-8)
        return 0;

    w = 3 * (sin(kr) / kr3 - cos(kr) / kr2);
    x = 4 * M_PI * k * k * w * w * function_of_k_eval(fk, k);

    return x;
}

double function_of_k_eval(FunctionOfK * fk, double k)
{
    /* ignore the 0 mode */

    if(k == 0) return 1;

    int l = 0;
    int r = fk->size - 1;

    while(r - l > 1) {
        int m = (r + l) / 2;
        if(k < fk->table[m].k)
            r = m;
        else
            l = m;
    }
    double k2 = fk->table[r].k,
           k1 = fk->table[l].k;
    double p2 = fk->table[r].Pk,
           p1 = fk->table[l].Pk;

    if(l == r) {
        return fk->table[l].Pk;
    }

    if(p1 == 0 || p2 == 0 || k1 == 0 || k2 == 0) {
        /* if any of the p is zero, use linear interpolation */
        double p = (k - k1) * p2 + (k2 - k) * p1;
        p /= (k2 - k1);
        return p;
    } else {
        k = log(k);
        p1 = log(p1);
        p2 = log(p2);
        k1 = log(k1);
        k2 = log(k2);
        double p = (k - k1) * p2 + (k2 - k) * p1;
        p /= (k2 - k1);
        return exp(p);
    }
}

// Adapted function to use Tanh-Sinh adaptive integration
double function_of_k_tophat_sigma(FunctionOfK *fk, double R)
{
    // Create the parameter structure
    struct sigma2_params params = {fk, R};
    double abserr;  // To hold the estimated error

    // Define the integrand as a lambda function wrapping the original `sigma2_int`
    auto integrand = [&params](double k) -> double {
        return sigma2_int(k, (void*)&params);
    };

    // Perform the Tanh-Sinh adaptive integration
    double result = tanh_sinh_integrate_adaptive(integrand, 0, 500.0 / R, &abserr, 1e-4);

    // Return the square root of the result
    return sqrt(result);
}

void function_of_k_normalize_sigma(FunctionOfK * fk, double R, double sigma) {
    double old = function_of_k_tophat_sigma(fk, R);
    size_t i;
    for(i = 0; i < fk->size; i ++) {
        fk->table[i].Pk *= sigma / old;
    };
}

/*! Check and print properties of the cosmological unit system.
 */
void
check_units(const Cosmology * CP, const struct UnitSystem units)
{
    /* Detect cosmologies that are likely to be typos in the parameter files*/
    if(CP->HubbleParam < 0.1 || CP->HubbleParam > 10 ||
        CP->OmegaLambda < 0 || CP->OmegaBaryon < 0 || CP->OmegaG < 0 || CP->OmegaCDM < 0)
        endrun(5, "Bad cosmology: H0 = %g OL = %g Ob = %g Og = %g Ocdm = %g\n",
               CP->HubbleParam, CP->OmegaLambda, CP->OmegaBaryon, CP->OmegaG, CP->OmegaCDM);

    message(0, "Hubble (internal units) = %g\n", CP->Hubble);
    message(0, "G (internal units) = %g\n", CP->GravInternal);
    message(0, "UnitLength_in_cm = %g \n", units.UnitLength_in_cm);
    message(0, "UnitMass_in_g = %g \n", units.UnitMass_in_g);
    message(0, "UnitTime_in_s = %g \n", units.UnitTime_in_s);
    message(0, "UnitVelocity_in_cm_per_s = %g \n", units.UnitVelocity_in_cm_per_s);
    message(0, "UnitDensity_in_cgs = %g \n", units.UnitDensity_in_cgs);
    message(0, "UnitEnergy_in_cgs = %g \n", units.UnitEnergy_in_cgs);
    message(0, "Dark energy model: OmegaL = %g OmegaFLD = %g\n",CP->OmegaLambda, CP->Omega_fld);
    message(0, "Photon density OmegaG = %g\n",CP->OmegaG);
    if(!CP->MassiveNuLinRespOn)
        message(0, "Massless Neutrino density OmegaNu0 = %g\n",get_omega_nu(&CP->ONu, 1));
    message(0, "Curvature density OmegaK = %g\n",CP->OmegaK);
    if(CP->RadiationOn) {
        double OmegaTot = CP->OmegaG + CP->OmegaK + CP->OmegaCDM + CP->OmegaBaryon + CP->OmegaLambda + CP->Omega_ur;
        OmegaTot += get_omega_nu(&CP->ONu, 1);
        OmegaTot += OmegaFLD(CP, 1);
        if(CP->use_class_radiation_convention) {
            message(0, "Radiation is enabled in Hubble(a). "
               "Following CLASS convention: Omega_Tot - 1 = %g\n", OmegaTot - 1);
        }
        else {
            message(0, "Radiation is enabled in Hubble(a). "
               "Following CAMB convention: Omega_Tot - 1 = %g\n", OmegaTot - 1);
        }
    }
    message(0, "\n");
}
