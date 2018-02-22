/*Tests for the drift factor module.*/
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include <stdint.h>

#include "stub.h"
#include <libgadget/allvars.h>
#include <libgadget/timefac.h>

#define AMIN 0.005
#define AMAX 1.0
#define TIMEBINS 29

static double OmegaM;
/*Hubble function at scale factor a, in dimensions of All.Hubble*/
double hubble_function(double a)
{
    double hubble_a;
    /* lambda and matter */
    hubble_a = 1 - OmegaM;
    hubble_a += OmegaM / (a * a * a);
    /*Radiation*/
    hubble_a += 5.045e-5*(1+7./8.*pow(pow(4/11.,1/3.)*1.00328,4)*3) / (a * a * a * a);
    /* Hard-code default Gadget unit system. */
    hubble_a = 0.1 * sqrt(hubble_a);
    return (hubble_a);
}

double fac_integ(double a, void *param)
{
  double h;
  int ex = * (int *) param;

  h = hubble_function(a);

  return 1 / (h * pow(a,ex));
}

/*Get integer from real time*/
double loga_from_ti(int ti)
{
    double logDTime = (log(AMAX) - log(AMIN)) / (1 << TIMEBINS);
    return log(AMIN) + ti * logDTime;
}

/*Get integer from real time*/
static inline int get_ti(double aa)
{
    double logDTime = (log(AMAX) - log(AMIN)) / (1 << TIMEBINS);
    return (log(aa) - log(AMIN))/logDTime;
}

double exact_drift_factor(double a1, double a2, int exp)
{
    double result, abserr;
    gsl_function F;
    gsl_integration_workspace *workspace;
    workspace = gsl_integration_workspace_alloc(10000);
    F.function = &fac_integ;
    F.params = &exp;
    gsl_integration_qag(&F, a1,a2, 0, 1.0e-8, 10000, GSL_INTEG_GAUSS61, workspace, &result, &abserr);
    gsl_integration_workspace_free(workspace);
    return result;
}

void test_drift_factor(void ** state)
{
    /*Initialise the table: default values from z=200 to z=0*/
    OmegaM = 1.;
    init_drift_table(AMIN, AMAX);
    /* Check default scaling: for total matter domination
     * we should have a drift factor like 1/sqrt(a)*/
    assert_true(fabs(get_drift_factor(get_ti(0.8), get_ti(0.85)) + 2/0.1*(1/sqrt(0.85) - 1/sqrt(0.8))) < 5e-5);
    /*Test the kick table*/
    assert_true(fabs(get_gravkick_factor(get_ti(0.8), get_ti(0.85)) - 2/0.1*(sqrt(0.85) - sqrt(0.8))) < 5e-5);

    //Chosen so we get the same bin
    assert_true(fabs(get_drift_factor(get_ti(0.8), get_ti(0.8003)) + 2/0.1*(1/sqrt(0.8003) - 1/sqrt(0.8))) < 5e-6);
    //Now choose a more realistic cosmology
    OmegaM = 0.25;
    init_drift_table(AMIN, AMAX);
    /*Check late and early times*/
    assert_true(fabs(get_drift_factor(get_ti(0.95), get_ti(0.98)) - exact_drift_factor(0.95, 0.98,3)) < 5e-5);
    assert_true(fabs(get_drift_factor(get_ti(0.05), get_ti(0.06)) - exact_drift_factor(0.05, 0.06,3)) < 5e-5);
    /*Check boundary conditions*/
    double logDtime = (log(AMAX)-log(AMIN))/(1<<TIMEBINS);
    assert_true(fabs(get_drift_factor(((1<<TIMEBINS)-1), 1<<TIMEBINS) - exact_drift_factor(AMAX-logDtime, AMAX,3)) < 5e-5);
    assert_true(fabs(get_drift_factor(0, 1) - exact_drift_factor(1.0 - exp(log(AMAX)-log(AMIN))/(1<<TIMEBINS), 1.0,3)) < 5e-5);
    /*Gravkick*/
    assert_true(fabs(get_gravkick_factor(get_ti(0.8), get_ti(0.85)) - exact_drift_factor(0.8, 0.85, 2)) < 5e-5);
    assert_true(fabs(get_gravkick_factor(get_ti(0.05), get_ti(0.06)) - exact_drift_factor(0.05, 0.06, 2)) < 5e-5);

    /*Test the hydrokick table: always the same as drift*/
    assert_true(fabs(get_hydrokick_factor(get_ti(0.8), get_ti(0.85)) - get_drift_factor(get_ti(0.8), get_ti(0.85))) < 5e-5);

}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_drift_factor),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
