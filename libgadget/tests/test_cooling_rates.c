/*Tests for the cooling rates module, ported from python.*/

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <libgadget/physconst.h>
#include <libgadget/cooling_rates.h>
#include <libgadget/config.h>
#include "stub.h"

double recomb_alphaHp(double temp);

#define NEXACT 20
/*For hydrogen recombination we have an exact answer from Ferland et al 1992 (http://adsabs.harvard.edu/abs/1992ApJ...387...95F).
This function returns as an array these rates, for testing purposes.*/
//case B recombination rates for hydrogen from Ferland 92, final column of Table 1. For n >= 2.
static const double f92g2[NEXACT] = {5.758e-11, 2.909e-11, 1.440e-11, 6.971e-12,3.282e-12, 1.489e-12, 6.43e-13, 2.588e-13, 9.456e-14, 3.069e-14, 8.793e-15, 2.245e-15, 5.190e-16, 1.107e-16, 2.221e-17, 4.267e-18, 7.960e-19, 1.457e-19,2.636e-20, 4.737e-21};
//case B recombination rates for hydrogen from Ferland 92, second column of Table 1. For n == 1.
static const double f92n1[NEXACT] = {9.258e-12, 5.206e-12, 2.927e-12, 1.646e-12, 9.246e-13, 5.184e-13, 2.890e-13, 1.582e-13, 8.255e-14, 3.882e-14, 1.545e-14, 5.058e-15, 1.383e-15, 3.276e-16, 7.006e-17, 1.398e-17, 2.665e-18, 4.940e-19, 9.001e-20, 1.623e-20};
static const double tt[NEXACT] = {3.16227766e+00, 1.0e+01, 3.16227766e+01, 1.0e+02, 3.16227766e+02, 1.00e+03, 3.16227766e+03, 1.e+04, 3.16227766e+04, 1.e+05, 3.16227766e+05, 1.e+06, 3.16227766e+06, 1.0e+07, 3.16227766e+07, 1.0e+08, 3.16227766e+08, 1.0e+09, 3.16227766e+09, 1.0e+10};

/*Test the recombination rates*/
static void test_recomb_rates(void ** state)
{
    /*Test of recombination only: no need to load the photon background tables*/
    struct cooling_params coolpar;
    int i;
    coolpar.CMBTemperature = 2.7255;
    coolpar.PhotoIonizeFactor = 1;
    coolpar.SelfShieldingOn = 1;
    coolpar.fBar = 0.17;
    coolpar.PhotoIonizationOn = 1;
    coolpar.recomb = Verner96;
    coolpar.cooling = Sherwood;
    const char * TreeCool = GADGET_TESTDATA_ROOT "/examples/TREECOOL_ep_2018p";
    const char * MetalCool = "";

    init_cooling_rates(TreeCool, MetalCool, coolpar);
    for(i=0; i< NEXACT; i++) {
        assert_true(fabs(recomb_alphaHp(tt[i])/(f92g2[i]+f92n1[i])-1.) < 1e-2);
    }

    coolpar.recomb = Cen92;
    init_cooling_rates(TreeCool, MetalCool, coolpar);
    /*Cen rates are not very accurate.*/
    for(i=4; i< 12; i++) {
        assert_true(fabs(recomb_alphaHp(tt[i])/(f92g2[i]+f92n1[i])-1.) < 0.5);
    }
}

/* Simple tests for the rate network */
static void test_rate_network(void ** state)
{
    struct cooling_params coolpar;
    coolpar.CMBTemperature = 2.7255;
    coolpar.PhotoIonizeFactor = 1;
    coolpar.SelfShieldingOn = 1;
    coolpar.fBar = 0.17;
    coolpar.PhotoIonizationOn = 1;
    coolpar.recomb = Verner96;
    coolpar.cooling = Sherwood;
    coolpar.MinGasTemp = 100;
    const char * MetalCool = "";

    const char * TreeCool = GADGET_TESTDATA_ROOT "/examples/TREECOOL_ep_2018p";
    struct UVBG uvbg = get_global_UVBG(2);
    init_cooling_rates(TreeCool, MetalCool, coolpar);

    //Complete ionisation at low density
    double logt;
    assert_true( fabs(get_equilib_ne(1e-6, 200.*1e10, 0.24, &logt, &uvbg, 1) / (1e-6*0.76) - (1 + 2* 0.24/(1-0.24)/4)) < 3e-5);
    assert_true( fabs(get_equilib_ne(1e-6, 200.*1e10, 0.12, &logt, &uvbg, 1) / (1e-6*0.88) - (1 + 2* 0.12/(1-0.12)/4)) < 3e-5);
    assert_true( fabs(get_equilib_ne(1e-5, 200.*1e10, 0.24, &logt, &uvbg, 1) / (1e-5*0.76) - (1 + 2* 0.24/(1-0.24)/4)) < 3e-4);
    assert_true( fabs(get_equilib_ne(1e-4, 200.*1e10, 0.24, &logt, &uvbg, 1) / (1e-4*0.76) - (1 + 2* 0.24/(1-0.24)/4)) < 2e-3);

    double ne = 1.;
    double temp = get_temp(1e-4, 200.*1e10,0.24, &uvbg, &ne);
    assert_true(9500 < temp);
    assert_true(temp < 9510);
    //Roughly prop to internal energy when ionised
    assert_true(fabs(get_temp(1e-4, 400.*1e10,0.24, &uvbg, &ne) / get_temp(1e-4, 200.*1e10,0.24, &uvbg, &ne) - 2.) < 1e-3);

    assert_true(fabs(get_temp(1, 200.*1e10,0.24, &uvbg, &ne) - 14700) < 200);

    //Neutral fraction prop to density.
    double dens[3] = {1e-4, 1e-5, 1e-6};
    int i;
    for(i = 0; i < 3; i++) {
        assert_true(fabs(get_neutral_fraction(dens[i], 200.*1e10,0.24, &uvbg, &ne) / dens[i] - 0.3113) < 1e-3);
    }
    //Neutral (self-shielded) at high density:
    assert_true(get_neutral_fraction(1, 100.,0.24, &uvbg, &ne) > 0.95);
    assert_true(0.75 > get_neutral_fraction(0.1, 100.*1e10,0.24, &uvbg, &ne));
    assert_true(get_neutral_fraction(0.1, 100.*1e10,0.24, &uvbg, &ne) > 0.735);

    //Check self-shielding is working.
    coolpar.SelfShieldingOn = 0;
    init_cooling_rates(TreeCool, MetalCool, coolpar);

    assert_true( get_neutral_fraction(1, 100.*1e10,0.24, &uvbg, &ne) < 0.25);
    assert_true( get_neutral_fraction(0.1, 100.*1e10,0.24, &uvbg, &ne) <0.05);
}

/* This test checks that the heating and cooling rate is as expected.
 * In particular the physical density threshold is checked. */
static void test_heatingcooling_rate(void ** state)
{
    struct cooling_params coolpar;
    coolpar.CMBTemperature = 2.7255;
    coolpar.PhotoIonizeFactor = 1;
    coolpar.SelfShieldingOn = 0;
    coolpar.fBar = 0.17;
    coolpar.PhotoIonizationOn = 1;
    coolpar.recomb = Cen92;
    coolpar.cooling = KWH92;
    coolpar.HeliumHeatOn = 0;

    const char * TreeCool = GADGET_TESTDATA_ROOT "/examples/TREECOOL_ep_2018p";
    const char * MetalCool = "";
    init_cooling_rates(TreeCool, MetalCool, coolpar);

    /*unit system*/
    double HubbleParam = 0.697;
    double UnitDensity_in_cgs = 6.76991e-22;
    double UnitTime_in_s = 3.08568e+16;
    double UnitMass_in_g = 1.989e+43;
    double UnitLength_in_cm = 3.08568e+21;
    double UnitEnergy_in_cgs = UnitMass_in_g  * pow(UnitLength_in_cm, 2) / pow(UnitTime_in_s, 2);

    struct cooling_units coolunits;
    coolunits.CoolingOn = 1;
    coolunits.density_in_phys_cgs = UnitDensity_in_cgs * HubbleParam * HubbleParam;
    coolunits.uu_in_cgs = UnitEnergy_in_cgs / UnitMass_in_g;
    coolunits.tt_in_s = UnitTime_in_s / HubbleParam;

    /*Default values from sfr_eff.c. Some dependence on HubbleParam, so don't change it.*/
    double egyhot = 2104.92;
    egyhot *= coolunits.uu_in_cgs;

    /* convert to physical cgs units */
    double dens = 0.027755;
    dens *= coolunits.density_in_phys_cgs/PROTONMASS;
    double ne = 1.0;

    struct UVBG uvbg = {0};
    /* XXX: We set the threshold without metal cooling
     * and with zero ionization at z=0.
     * It probably make sense to set the parameters with
     * a metalicity dependence. */
    double LambdaNet = get_heatingcooling_rate(dens, egyhot, 1 - HYDROGEN_MASSFRAC, 0, 0, &uvbg, &ne);

    double tcool = egyhot / (- LambdaNet);

    /*Convert back to internal units*/
    tcool /= coolunits.tt_in_s;

    //message(1, "tcool = %g LambdaNet = %g ne=%g\n", tcool, LambdaNet, ne);
    /* This differs by 0.13% from the old cooling code number,
     * apparently just because of rounding errors. The excitation cooling
     * numbers from Cen are not accurate to better than 1% anyway, so don't worry about it*/
    assert_true(fabs(tcool / 4.68748e-06 - 1) < 1e-5);

    /*Now check that we get the desired cooling rate with a UVB*/
    uvbg = get_global_UVBG(0);

    assert_true(uvbg.epsHep > 0);
    assert_true(uvbg.gJHe0 > 0);

    dens /= 100;
    LambdaNet = get_heatingcooling_rate(dens, egyhot/10., 1 - HYDROGEN_MASSFRAC, 0, 0, &uvbg, &ne);
    //message(1, "LambdaNet = %g, uvbg=%g\n", LambdaNet, uvbg.epsHep);
    assert_true(fabs(LambdaNet / (-0.0406671) - 1) < 1e-5);

    LambdaNet = get_heatingcooling_rate(dens/2.5, egyhot/10., 1 - HYDROGEN_MASSFRAC, 0, 0, &uvbg, &ne);
    assert_true(LambdaNet > 0);
    /*Check self-shielding affects the cooling rates*/
    coolpar.SelfShieldingOn = 1;
    init_cooling_rates(TreeCool, MetalCool, coolpar);
    LambdaNet = get_heatingcooling_rate(dens*1.5, egyhot/10., 1 - HYDROGEN_MASSFRAC, 0, 0, &uvbg, &ne);
    //message(1, "LambdaNet = %g, uvbg=%g\n", LambdaNet, uvbg.epsHep);
    assert_false(LambdaNet > 0);
    assert_true(fabs(LambdaNet/ (-1.64682) - 1) < 1e-5);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_recomb_rates),
        cmocka_unit_test(test_rate_network),
        cmocka_unit_test(test_heatingcooling_rate),

    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
