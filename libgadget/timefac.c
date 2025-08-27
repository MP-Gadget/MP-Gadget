#include <mpi.h>
#include <math.h>
#include <gsl/gsl_integration.h>

#include "physconst.h"
#include "timefac.h"
#include "timebinmgr.h"

#define WORKSIZE 10000

/* Integrand for the drift table*/
static double drift_integ(double a, void *param)
{
  Cosmology * CP = (Cosmology *) param;
    if (!CP->ComovingIntegrationOn)
      return 1.0 / hubble_function(CP, a);
  double h = hubble_function(CP, a);
  return 1 / (h * a * a * a);
}

/* Integrand for the gravkick table*/
static double gravkick_integ(double a, void *param)
{
  Cosmology * CP = (Cosmology *) param;
  if (!CP->ComovingIntegrationOn)
      return 1.0 / hubble_function(CP, a);
  double h = hubble_function(CP, a);

  return 1 / (h * a * a);
}

/* Integrand for the hydrokick table.
 * Note this is the same function as drift.*/
static double hydrokick_integ(double a, void *param)
{

  Cosmology * CP = (Cosmology *) param;
  if (!CP->ComovingIntegrationOn)
      return 1.0 / hubble_function(CP, a);

  double h = hubble_function(CP, a);

  return 1 / (h * pow(a, 3 * GAMMA_MINUS1) * a);
}

/*Do the integral required to get a factor.*/
static double get_exact_factor(Cosmology * CP, inttime_t t0, inttime_t t1, double (*factor) (double, void *))
{
    double result, abserr;
    double a0, a1;
    if(t0 == t1)
        return 0;
    if (CP->ComovingIntegrationOn) {
        a0 = exp(loga_from_ti(t0));
        a1 = exp(loga_from_ti(t1));
    }
    else {
    /* In the Non-Comoving Case integrate wrt dloga
        integrand = dt = dloga / H(a) = dloga / H0
    */
        a0 = loga_from_ti(t0);
        a1 = loga_from_ti(t1);
    }
    gsl_function F;
    gsl_integration_workspace *workspace;
    workspace = gsl_integration_workspace_alloc(WORKSIZE);
    F.function = factor;
    F.params = CP;
    gsl_integration_qag(&F, a0, a1, 0, 1.0e-8, WORKSIZE, GSL_INTEG_GAUSS61, workspace, &result, &abserr);
    gsl_integration_workspace_free(workspace);
    return result;
}

/*Get the exact drift factor*/
double get_exact_drift_factor(Cosmology * CP, inttime_t ti0, inttime_t ti1)
{
    return get_exact_factor(CP, ti0, ti1, &drift_integ);
}

/*Get the exact drift factor*/
double get_exact_gravkick_factor(Cosmology * CP, inttime_t ti0, inttime_t ti1)
{
    return get_exact_factor(CP, ti0, ti1, &gravkick_integ);
}

double get_exact_hydrokick_factor(Cosmology * CP, inttime_t ti0, inttime_t ti1)
{
    return get_exact_factor(CP, ti0, ti1, &hydrokick_integ);
}

/* Integrand for comoving distance */
static double comoving_distance_integ(double a, void *param)
{
    Cosmology *CP = (Cosmology *) param;
    double h = hubble_function(CP, a);
    return 1. / (h * a * a);
}

/* Function to compute the comoving distance between two scale factors */
double compute_comoving_distance(Cosmology *CP, double a0, double a1, const double UnitVelocity_in_cm_per_s)
{
    double result, abserr;
    gsl_function F;
    gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(WORKSIZE);

    F.function = comoving_distance_integ;
    F.params = CP;

    // Using GSL to perform the integration
    gsl_integration_qag(&F, a0, a1, 0, 1.0e-8, WORKSIZE, GSL_INTEG_GAUSS61, workspace, &result, &abserr);
    gsl_integration_workspace_free(workspace);

    return (LIGHTCGS/UnitVelocity_in_cm_per_s) * result;
}
