/*Tests for the cosmology module, ported from N-GenIC.*/

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "stub.h"
#include <libgadget/config.h>
#include <libgenic/power.h>

int DifferentTransferFunctions = 1;

/*Dummy growth factor, tested elsewhere.*/
double GrowthFactor(double astart, double aend)
{
    return astart/aend;
}

/*Simple test without rescaling*/
static void
test_read_no_rescale(void ** state)
{
    /*Do setup*/
    struct power_params PowerP;
    /*Test without rescaling*/
    PowerP.InputPowerRedshift = -1;
    PowerP.Sigma8 = -1;
    PowerP.FileWithInputSpectrum = GADGET_TESTDATA_ROOT "/examples/camb_matterpow_99.dat";
    PowerP.FileWithTransferFunction = GADGET_TESTDATA_ROOT "/examples/camb_transfer_99.dat";
    PowerP.WhichSpectrum = 2;
    PowerP.SpectrumLengthScale = 1000;
    PowerP.PrimordialIndex = 1.0;
    int nentry = initialize_powerspectrum(0, 0.01, 3.085678e21, NULL, &PowerP);
    assert_int_equal(nentry, 335);
    /*Check that the tabulated power spectrum gives the right answer
     * First check ranges: these should both be out of range.
     * Should be the same k as in the file (but /10^3 for Mpc -> kpc)
     * Note that our PowerSpec function is 2pi^3 larger than that in S-GenIC.*/
    assert_true(PowerSpec(9.8e-9, 7) < 2e-30);
    assert_true(PowerSpec(2.1e-1, 7) < 2e-30);
    //Now check total power: k divided by 10^3,
    //Conversion for P(k) is 10^9/(2pi)^3
    assert_true(fabs(PowerSpec(0.11353e-01/1e3, 7) - 0.47803e+01*1e9) < 1e-5);
    assert_true(fabs(PowerSpec(1.0202/1e3, 7) - 0.11263E-01*1e9) < 1e-6);
    //Check that it gives reasonable results when interpolating
    int k;
    for (k = 1; k < 100; k++) {
        double newk = 0.10022E+01/1e3+ k*(0.10362E+01-0.10022E+01)/1e3/100;
        assert_true(PowerSpec(newk,7) < PowerSpec(0.10022E+01/1e3,7));
        assert_true(PowerSpec(newk,7) > PowerSpec(0.10362E+01/1e3,7));
    }
    //Now check transfer functions: ratio of total to species should be ratio of T_s to T_tot squared: large scales where T~ 1
    assert_true(fabs(PowerSpec(0.210658E-02/1e3,0)/PowerSpec(0.210658E-02/1e3,7)- pow(0.244763E+06/0.245082E+06,2)) < 1e-6);
    //CDM
    assert_true(fabs(PowerSpec(0.210658E-02/1e3,1)/PowerSpec(0.210658E-02/1e3,7)- pow(0.245146E+06/0.245082E+06,2)) < 1e-6);
    //Small scales where there are differences
    //T_tot=0.255697E+06
    //Baryons
    assert_true(fabs(PowerSpec(0.111030E+00/1e3,0)/PowerSpec(0.111030E+00/1e3,6)- pow(0.200504E+05/0.277947E+05,2)) < 1e-6);
    //CDM
    assert_true(fabs(PowerSpec(0.111030E+00/1e3,1)/PowerSpec(0.111030E+00/1e3,6)- pow(0.293336E+05/0.277947E+05,2)) < 1e-6);
}

/*Check normalising to a different sigma8 and redshift*/
static void
test_read_rescale_sigma8(void ** state)
{
    /*Do setup*/
    struct power_params PowerP;
    /* Test rescaling to an earlier time
     * (we still use the same z=99 power which should not be rescaled in a real simulation)*/
    PowerP.InputPowerRedshift = 0;
    PowerP.Sigma8 = -1;
    PowerP.FileWithInputSpectrum = GADGET_TESTDATA_ROOT "/examples/camb_matterpow_99.dat";
    PowerP.FileWithTransferFunction = GADGET_TESTDATA_ROOT "/examples/camb_transfer_99.dat";
    PowerP.WhichSpectrum = 2;
    PowerP.SpectrumLengthScale = 1000;
    PowerP.PrimordialIndex = 1.0;
    int nentry = initialize_powerspectrum(0, 0.01, 3.085678e21, NULL, &PowerP);
    assert_int_equal(nentry, 335);
    assert_true(fabs(PowerSpec(0.11353e-01/1e3, 7)*100*100 - 0.47803e+01*1e9) < 1e-5);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_read_no_rescale),
        cmocka_unit_test(test_read_rescale_sigma8)
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
