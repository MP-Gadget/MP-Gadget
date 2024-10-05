/*Tests for the drift factor module.*/
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../timefac.h"
#include <stdint.h>

#include "stub.h"
#include <libgadget/timefac.h>

#define AMIN 0.005
#define AMAX 1.0
#define LTIMEBINS 29

/*Hubble function at scale factor a, in dimensions of All.Hubble*/
double hubble_function(const Cosmology * CP, double a)
{
    double hubble_a;
    /* lambda and matter */
    hubble_a = 1 - CP->Omega0;
    hubble_a += CP->Omega0 / (a * a * a);
    /*Radiation*/
    hubble_a += 5.045e-5*(1+7./8.*pow(pow(4/11.,1/3.)*1.00328,4)*3) / (a * a * a * a);
    /* Hard-code default Gadget unit system. */
    hubble_a = 0.1 * sqrt(hubble_a);
    return (hubble_a);
}

struct fac_params
{
    Cosmology * CP;
    int exp;
};

double fac_integ(double a, void *param)
{
  double h;
  struct fac_params * ff = (struct fac_params *) param;

  h = hubble_function(ff->CP, a);

  return 1 / (h * pow(a,ff->exp));
}

/*Get integer from real time*/
double loga_from_ti(inttime_t ti)
{
    double logDTime = (log(AMAX) - log(AMIN)) / (1 << LTIMEBINS);
    return log(AMIN) + ti * logDTime;
}

/*Get integer from real time*/
static inline inttime_t get_ti(double aa)
{
    double logDTime = (log(AMAX) - log(AMIN)) / (1 << LTIMEBINS);
    return (log(aa) - log(AMIN))/logDTime;
}

double exact_drift_factor(Cosmology * CP, double a1, double a2, int exp)
{
    double result, abserr;
    
    struct fac_params ff = {CP, exp};
    auto integrand = [&ff](double a) {
        return fac_integ(a, (void*)&ff);
    };
    result = tanh_sinh_integrate_adaptive(integrand, a1, a2, &abserr, 1e-8);
    return result;
}

void test_drift_factor(void ** state)
{
    /*Initialise the table: default values from z=200 to z=0*/
    Cosmology CP;
    CP.Omega0 = 1.;
    /* Check default scaling: for total matter domination
     * we should have a drift factor like 1/sqrt(a)*/
    assert_true(fabs(get_exact_drift_factor(&CP, get_ti(0.8), get_ti(0.85)) + 2/0.1*(1/sqrt(0.85) - 1/sqrt(0.8))) < 5e-5);
    /*Test the kick table*/
    assert_true(fabs(get_exact_gravkick_factor(&CP, get_ti(0.8), get_ti(0.85)) - 2/0.1*(sqrt(0.85) - sqrt(0.8))) < 5e-5);

    //Chosen so we get the same bin
    assert_true(fabs(get_exact_drift_factor(&CP, get_ti(0.8), get_ti(0.8003)) + 2/0.1*(1/sqrt(0.8003) - 1/sqrt(0.8))) < 5e-6);
    //Now choose a more realistic cosmology
    CP.Omega0 = 0.25;
    /*Check late and early times*/
    assert_true(fabs(get_exact_drift_factor(&CP, get_ti(0.95), get_ti(0.98)) - exact_drift_factor(&CP, 0.95, 0.98,3)) < 5e-5);
    assert_true(fabs(get_exact_drift_factor(&CP, get_ti(0.05), get_ti(0.06)) - exact_drift_factor(&CP, 0.05, 0.06,3)) < 5e-5);
    /*Check boundary conditions*/
    double logDtime = (log(AMAX)-log(AMIN))/(1<<LTIMEBINS);
    assert_true(fabs(get_exact_drift_factor(&CP, ((1<<LTIMEBINS)-1), 1<<LTIMEBINS) - exact_drift_factor(&CP, AMAX-logDtime, AMAX,3)) < 5e-5);
    assert_true(fabs(get_exact_drift_factor(&CP, 0, 1) - exact_drift_factor(&CP, 1.0 - exp(log(AMAX)-log(AMIN))/(1<<LTIMEBINS), 1.0,3)) < 5e-5);
    /*Gravkick*/
    assert_true(fabs(get_exact_gravkick_factor(&CP, get_ti(0.8), get_ti(0.85)) - exact_drift_factor(&CP, 0.8, 0.85, 2)) < 5e-5);
    assert_true(fabs(get_exact_gravkick_factor(&CP, get_ti(0.05), get_ti(0.06)) - exact_drift_factor(&CP, 0.05, 0.06, 2)) < 5e-5);

    /*Test the hydrokick table: always the same as drift*/
    assert_true(fabs(get_exact_hydrokick_factor(&CP, get_ti(0.8), get_ti(0.85)) - get_exact_drift_factor(&CP, get_ti(0.8), get_ti(0.85))) < 5e-5);

}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_drift_factor),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
