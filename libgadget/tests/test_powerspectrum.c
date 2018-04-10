/* This file tests the power spectrum routines only.
 * The actual PM code is too complicated for now. */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <mpi.h>
#include "stub.h"

#include <libgadget/powerspectrum.h>

#define NUM_THREADS 4

/*Test the total powerspectrum on one processor only*/
static void test_total_powerspectrum(void **state) {
    (void) state;
    /*Check allocation*/
    int nmpi;
    struct _powerspectrum PowerSpectrum;
    MPI_Comm_size(MPI_COMM_WORLD, &nmpi);

    powerspectrum_alloc(&PowerSpectrum,15,NUM_THREADS);
    assert_true(PowerSpectrum.Nmodes);
    assert_true(PowerSpectrum.Power);
    assert_true(PowerSpectrum.kk);
    powerspectrum_zero(&PowerSpectrum);
    assert_true(PowerSpectrum.Nmodes[0] == 0);
    assert_true(PowerSpectrum.Nmodes[PowerSpectrum.size-1] == 0);

    //Construct input power (this would be done by the power spectrum routine in petapm)
    int ii, th;
    for(ii=0; ii<15; ii++) {
        for(th = 0; th < NUM_THREADS; th++) {
            PowerSpectrum.Nmodes[ii+PowerSpectrum.size*th] = ii;
            PowerSpectrum.Power[ii+PowerSpectrum.size*th] = ii*sin(ii)*sin(ii);
            PowerSpectrum.kk[ii+PowerSpectrum.size*th] = ii*ii;
        }
    }
    PowerSpectrum.Norm = 1;
    /*Now every thread and every MPI has the same data. Sum it.*/
    powerspectrum_sum(&PowerSpectrum,3.085678e24);

    /*Check summation was done correctly*/
    assert_true(PowerSpectrum.Nmodes[0] == 0);
    assert_true(PowerSpectrum.Nmodes[1] == NUM_THREADS*nmpi);
    assert_true(PowerSpectrum.Nmodes[14] == NUM_THREADS*nmpi*14);

    assert_true(PowerSpectrum.Power[0] == 0);
    assert_true(fabs(PowerSpectrum.Power[1] - sin(1)*sin(1)) < 1e-5);
    assert_true(fabs(PowerSpectrum.Power[13] - sin(13)*sin(13)) < 1e-5);
    assert_true(fabs(PowerSpectrum.kk[13] - 2 * M_PI *13) < 1e-5);
    assert_true(fabs(PowerSpectrum.kk[1] - 2 * M_PI ) < 1e-5);

}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_total_powerspectrum),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
