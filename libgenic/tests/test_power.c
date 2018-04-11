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
    PowerP.FileWithFutureTransferFunction = GADGET_TESTDATA_ROOT "/examples/camb_transfer_98.99.dat";
    PowerP.DifferentTransferFunctions = 1;
    PowerP.InputFutureRedshift = 98.99;
    PowerP.WhichSpectrum = 2;
    PowerP.SpectrumLengthScale = 1000;
    PowerP.PrimordialIndex = 1.0;
    int nentry = initialize_powerspectrum(0, 0.01, 3.085678e21, NULL, &PowerP);
    assert_int_equal(nentry, 335);
    /*Check that the tabulated power spectrum gives the right answer
     * First check ranges: these should both be out of range.
     * Should be the same k as in the file (but /10^3 for Mpc -> kpc)
     * Note that our PowerSpec function is 2pi^3 larger than that in S-GenIC.*/
    assert_true(DeltaSpec(9.8e-9, 7) < 2e-30);
    assert_true(DeltaSpec(300, 7) < 2e-30);
    //Now check total power: k divided by 10^3,
    //Conversion for P(k) is 10^9/(2pi)^3
    assert_true(fabs(pow(DeltaSpec(1.124995061548053968e-02/1e3, 7),2) / 4.745074933325402533/1e9 - 1) < 1e-5);
    assert_true(fabs(pow(DeltaSpec(1.010157135208153312e+00/1e3, 7),2) / 1.15292e-02/1e9 - 1) < 1e-5);
    //Check that it gives reasonable results when interpolating
    int k;
    for (k = 1; k < 100; k++) {
        double newk = 0.10022E+01/1e3+ k*(0.10362E+01-0.10022E+01)/1e3/100;
        assert_true(DeltaSpec(newk,7) < DeltaSpec(0.10022E+01/1e3,7));
        assert_true(DeltaSpec(newk,7) > DeltaSpec(0.10362E+01/1e3,7));
        assert_true(DeltaSpec(newk,0)/DeltaSpec(0.10362E+01/1e3,1) < 1);
    }
    //Now check transfer functions: ratio of total to species should be ratio of T_s to T_tot squared: large scales where T~ 1
    //CDM
    assert_true(fabs(DeltaSpec(2.005305808001081169e-03/1e3,1)/DeltaSpec(2.005305808001081169e-03/1e3,7)- 1.193460280018762132e+05/1.193185119820504624e+05) < 1e-5);
    //Small scales where there are differences
    //T_tot=0.255697E+06
    //Baryons
    assert_true(fabs(DeltaSpec(1.079260830861467901e-01/1e3,0)/DeltaSpec(1.079260830861467901e-01/1e3,6)- 9.735695830700024089e+03/1.394199788775037632e+04) < 1e-6);
    //CDM
    assert_true(fabs(DeltaSpec(1.079260830861467901e-01/1e3,1)/DeltaSpec(1.079260830861467901e-01/1e3,6)- 1.477251880454670209e+04/1.394199788775037632e+04) < 1e-6);
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
    PowerP.DifferentTransferFunctions = 0;
    PowerP.Sigma8 = -1;
    PowerP.FileWithInputSpectrum = GADGET_TESTDATA_ROOT "/examples/camb_matterpow_99.dat";
    PowerP.FileWithTransferFunction = GADGET_TESTDATA_ROOT "/examples/camb_transfer_99.dat";
    PowerP.FileWithFutureTransferFunction = GADGET_TESTDATA_ROOT "/examples/camb_transfer_98.99.dat";
    PowerP.WhichSpectrum = 2;
    PowerP.SpectrumLengthScale = 1000;
    PowerP.PrimordialIndex = 1.0;
    int nentry = initialize_powerspectrum(0, 0.01, 3.085678e21, NULL, &PowerP);
    assert_int_equal(nentry, 335);
    assert_true(fabs(pow(DeltaSpec(1.124995061548053968e-02/1e3, 7),2)* 100 * 100 /4.745074933325402533/1e9 - 1) < 1e-2);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_read_no_rescale),
        cmocka_unit_test(test_read_rescale_sigma8)
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
