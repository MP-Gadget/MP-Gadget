#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdio.h>
#include <math.h>
#include "stub.h"
#include "../omega_nu_single.h"
#include "../physconst.h"
#include "../timefac.h"

#define  T_CMB0      2.7255	/* present-day CMB temperature, from Fixsen 2009 */

/* A test case that checks initialisation. */
static void test_rho_nu_init(void **state) {
    (void) state; /* unused */
    double mnu = 0.06;
    _rho_nu_single rho_nu_tab;
    /*Initialise*/
    rho_nu_init(&rho_nu_tab, 0.01, mnu, BOLEVK*TNUCMB*T_CMB0);
    /*Check everything initialised ok*/
    assert_true(rho_nu_tab.mnu == mnu);
    assert_true(rho_nu_tab.acc);
    assert_true(rho_nu_tab.interp);
    assert_true(rho_nu_tab.loga);
    assert_true(rho_nu_tab.rhonu);
    /*Check that loga is correctly ordered (or interpolation won't work)*/
    int i;
    for(i=1; i<200; i++){
        assert_true(rho_nu_tab.loga[i] > rho_nu_tab.loga[i-1]);
    }
}
/*Check massless neutrinos work*/
#define STEFAN_BOLTZMANN 5.670373e-5
#define OMEGAR (4*STEFAN_BOLTZMANN*8*M_PI*GRAVITY/(3*LIGHTCGS*LIGHTCGS*LIGHTCGS*HUBBLE*HUBBLE*HubbleParam*HubbleParam)*pow(T_CMB0,4))


/* Check that the table gives the right answer. */
static void test_omega_nu_single(void **state) {
    (void) state; /* unused */
    double mnu = 0.5;
    double HubbleParam = 0.7;
    _omega_nu omnu;
    /*Initialise*/
    double MNu[3] = {mnu, mnu, 0};
    init_omega_nu(&omnu, MNu, 0.01, 0.7,T_CMB0);
    assert_true(omnu.RhoNuTab[0].mnu == mnu);
    /*This is the critical density at z=0:
     * we allow it to change by what is (at the time of writing) the uncertainty on G.*/
    assert_true(fabs(omnu.rhocrit - 1.8784e-29*HubbleParam*HubbleParam) < 5e-3*omnu.rhocrit);
    /*Check everything initialised ok*/
    double omnuz0 = omega_nu_single(&omnu, 1, 0);
    /*Check redshift scaling*/
    assert_true(fabs(omnuz0/pow(0.5,3) - omega_nu_single(&omnu, 0.5, 0)) < 5e-3*omnuz0);
    /*Check not just a^-3 scaling*/
    assert_true(omnuz0/pow(0.01,3) <  omega_nu_single(&omnu, 0.01, 0));
    /*Check that we have correctly accounted for neutrino decoupling*/
    assert_true(fabs(omnuz0 - mnu/93.14/HubbleParam/HubbleParam) < 1e-3*omnuz0);
    /*Check degenerate species works*/
    assert_true(omega_nu_single(&omnu, 0.1, 1) == omega_nu_single(&omnu, 0.1, 0));
    /*Check we get it right for massless neutrinos*/
    double omnunomassz0 = omega_nu_single(&omnu, 1, 2);
    assert_true(omnunomassz0 - OMEGAR*7./8.*pow(pow(4/11.,1/3.)*1.00381,4)< 1e-5*omnunomassz0);
    assert_true(omnunomassz0/pow(0.5,4) == omega_nu_single(&omnu, 0.5, 2));
    /*Check that we return something vaguely sensible for very early times*/
    assert_true(omega_nu_single(&omnu,1e-4,0) > omega_nu_single(&omnu, 1,0)/pow(1e-4,3));
}


double get_rho_nu_conversion();

/*Note q carries units of eV/c. kT/c has units of eV/c.
 * M_nu has units of eV  Here c=1. */
double rho_nu_int(double q, void * params);

double do_exact_rho_nu_integration(double a, double mnu, double rhocrit)
{
    auto integrand = [&param](double q) {
        return rho_nu_int(q, (void*) &param);
    };
    double abserr;
    double kTnu = BOLEVK*TNUCMB*T_CMB0;
    double param[2] = {mnu * a, kTnu};
    double result;
    result = tanh_sinh_integrate_adaptive(integrand, 0, 500*kTnu, &abserr, 1e-9);
    result*=get_rho_nu_conversion()/pow(a,4)/rhocrit;
    return result;
}

/*Check exact integration against the interpolation table*/
static void test_omega_nu_single_exact(void **state)
{
    double mnu = 0.05;
    double hubble = 0.7;
    int i;
    _omega_nu omnu;
    /*Initialise*/
    double MNu[3] = {mnu, mnu, mnu};
    init_omega_nu(&omnu, MNu, 0.01, hubble,T_CMB0);
    double omnuz0 = omega_nu_single(&omnu, 1, 0);
    double rhocrit = omnu.rhocrit;
    assert_true(fabs(1 - do_exact_rho_nu_integration(1, mnu, rhocrit)/omnuz0) < 1e-6);
    for(i=1; i< 123; i++) {
        double a = 0.01 + i/123.;
        omnuz0 = omega_nu_single(&omnu, a, 0);
        double omexact = do_exact_rho_nu_integration(a, mnu, rhocrit);
        if(fabs(omnuz0 - omexact) > 1e-6 * omnuz0)
            printf("a=%g %g %g %g\n",a, omnuz0, omexact, omnuz0/omexact-1);
        assert_true(fabs(1 - omexact/omnuz0) < 1e-6);
    }
}

static void test_omega_nu_init_degenerate(void **state) {
    /*Check we correctly initialise omega_nu with degenerate neutrinos*/
    _omega_nu omnu;
    /*Initialise*/
    double MNu[3] = {0.2,0.2,0.2};
    init_omega_nu(&omnu, MNu, 0.01, 0.7,T_CMB0);
    /*Check that we initialised the right number of arrays*/
    assert_int_equal(omnu.nu_degeneracies[0], 3);
    assert_int_equal(omnu.nu_degeneracies[1], 0);
    assert_true(omnu.RhoNuTab[0].loga);
    assert_false(omnu.RhoNuTab[1].loga);
}

static void test_omega_nu_init_nondeg(void **state) {
    /*Now check that it works with a non-degenerate set*/
    _omega_nu omnu;
    /*Initialise*/
    double MNu[3] = {0.2,0.1,0.3};
    int i;
    init_omega_nu(&omnu, MNu, 0.01, 0.7,T_CMB0);
    /*Check that we initialised the right number of arrays*/
    for(i=0; i<3; i++) {
        assert_int_equal(omnu.nu_degeneracies[i], 1);
        assert_true(omnu.RhoNuTab[i].loga);
    }
}

static void test_get_omega_nu(void **state) {
    /*Check that we get the right answer from get_omega_nu, in terms of rho_nu*/
    _omega_nu omnu;
    /*Initialise*/
    double MNu[3] = {0.2,0.1,0.3};
    init_omega_nu(&omnu, MNu, 0.01, 0.7,T_CMB0);
    double total =0;
    int i;
    for(i=0; i<3; i++) {
        total += omega_nu_single(&omnu, 0.5, i);
    }
    assert_true(fabs(get_omega_nu(&omnu, 0.5) - total) < 1e-6*total);
}

static void test_get_omegag(void **state) {
    /*Check that we get the right answer from get_omegag*/
    _omega_nu omnu;
    /*Initialise*/
    double MNu[3] = {0.2,0.1,0.3};
    const double HubbleParam = 0.7;
    init_omega_nu(&omnu, MNu, 0.01, HubbleParam,T_CMB0);
    const double omegag = OMEGAR/pow(0.5,4);
    assert_true(fabs(get_omegag(&omnu, 0.5)/omegag -1)< 1e-6);
}

/*Test integrate the fermi-dirac kernel between 0 and qc*/
static void test_nufrac_low(void **state)
{
    assert_true(nufrac_low(0) == 0);
    /*Mathematica integral: 1.*Integrate[x*x/(Exp[x] + 1), {x, 0, 0.5}]/(3*Zeta[3]/2)*/
    assert_true(fabs(nufrac_low(1)/0.0595634-1)< 1e-5);
    assert_true(fabs(nufrac_low(0.5)/0.00941738-1)< 1e-5);
}

static void test_hybrid_neutrinos(void **state)
{
    /*Check that we get the right answer from get_omegag*/
    _omega_nu omnu;
    /*Initialise*/
    double MNu[3] = {0.2,0.2,0.2};
    const double HubbleParam = 0.7;
    init_omega_nu(&omnu, MNu, 0.01, HubbleParam,T_CMB0);
    init_hybrid_nu(&omnu.hybnu, MNu, 700, 299792, 0.5,omnu.kBtnu);
    /*Check that the fraction of omega change over the jump*/
    double nufrac_part = nufrac_low(700/299792.*0.2/omnu.kBtnu);
    assert_true(fabs(particle_nu_fraction(&omnu.hybnu, 0.50001, 0)/nufrac_part -1) < 1e-5);
    assert_true(particle_nu_fraction(&omnu.hybnu, 0.49999, 0) == 0);
    assert_true(fabs(get_omega_nu_nopart(&omnu, 0.499999)*(1-nufrac_part)/get_omega_nu_nopart(&omnu, 0.500001)-1) < 1e-4);
    /*Ditto omega_nu_single*/
    assert_true(fabs(omega_nu_single(&omnu, 0.499999, 0)*(1-nufrac_part)/omega_nu_single(&omnu, 0.500001, 0)-1) < 1e-4);
}


int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_rho_nu_init),
        cmocka_unit_test(test_omega_nu_single),
        cmocka_unit_test(test_omega_nu_init_degenerate),
        cmocka_unit_test(test_omega_nu_init_nondeg),
        cmocka_unit_test(test_get_omega_nu),
        cmocka_unit_test(test_get_omegag),
        cmocka_unit_test(test_omega_nu_single_exact),
        cmocka_unit_test(test_nufrac_low),
        cmocka_unit_test(test_hybrid_neutrinos),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
