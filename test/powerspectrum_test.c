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
#include "../powerspectrum.h"

/*Dummy functions to keep linker happy.*/

void * mymalloc_fullinfo(const char * string, size_t size, const char *func, const char *file, int line)
{
    return malloc(size);
}

void myfree_fullinfo(void * ptr, const char *func, const char *file, int line)
{
    free(ptr);
}

/*End dummies*/

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
    assert_true(PowerSpectrum.P);
    assert_true(PowerSpectrum.k);
    powerspectrum_zero(&PowerSpectrum);
    assert_true(PowerSpectrum.Nmodes[0] == 0);
    assert_true(PowerSpectrum.Nmodes[PowerSpectrum.size-1] == 0);

    //Construct input power (this would be done by the power spectrum routine in petapm)
    for(int ii=0; ii<15; ii++) {
        for(int th = 0; th < NUM_THREADS; th++) {
            PowerSpectrum.Nmodes[ii+PowerSpectrum.size*th] = ii;
            PowerSpectrum.P[ii+PowerSpectrum.size*th] = ii*sin(ii)*sin(ii);
            PowerSpectrum.k[ii+PowerSpectrum.size*th] = ii*ii;
        }
    }
    PowerSpectrum.Norm = 1;
    /*Now every thread and every MPI has the same data. Sum it.*/
    powerspectrum_sum(&PowerSpectrum,3.085678e24);

    /*Check summation was done correctly*/
    assert_true(PowerSpectrum.Nmodes[0] == 0);
    assert_true(PowerSpectrum.Nmodes[1] == NUM_THREADS*nmpi);
    assert_true(PowerSpectrum.Nmodes[14] == NUM_THREADS*nmpi*14);

    assert_true(PowerSpectrum.P[0] == 0);
    assert_true(fabs(PowerSpectrum.P[1] - sin(1)*sin(1)) < 1e-5);
    assert_true(fabs(PowerSpectrum.P[13] - sin(13)*sin(13)) < 1e-5);
    assert_true(fabs(PowerSpectrum.k[13] - 2 * M_PI *13) < 1e-5);
    assert_true(fabs(PowerSpectrum.k[1] - 2 * M_PI ) < 1e-5);

}

static int setup_mpi(void **state) {
    int ac=1;
    char * str = "powerspectrum_test";
    char **av = &str;
    MPI_Init(&ac, &av);
    omp_set_num_threads(NUM_THREADS);
    return 0;
}

static int teardown_mpi(void **state) {
    MPI_Finalize();
    return 0;
}


int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_total_powerspectrum),
    };
    return cmocka_run_group_tests(tests, setup_mpi, teardown_mpi);
}
