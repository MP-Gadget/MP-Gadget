#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include "openmpsort.h"
#include "endrun.h"
#include "stub.h"

#if 0
static int checksorted(int * start, int nmemb) {
    int i;
    for(i = 1; i < nmemb; i ++) {
        if( start[i-1] > start[i]) {
            return i;
        }
    }
    return 0;
}
#endif
static int compare(const void *a, const void *b) {
        return ( *(int*)a - *(int*)b );
}

static void test_openmpsort(void ** state) {
    int i;
    // set up array to be sorted
    //a bad case is 87763 threads = 12
    int size = 87763;
    int *a = (int *) malloc(size * sizeof(int));

    srand48(8675309);
    for(i = 0; i < size; i++) 
        a[i] = (int) (size * drand48());

    double start, end;

    start = omp_get_wtime();
    qsort_openmp(a, size, sizeof(int), compare);
    end = omp_get_wtime();

    message(1,"parallel sort time = %g s %d threads\n",end-start, omp_get_max_threads());
    for(i=1; i<size; i++) {
        if( a[i-1]>a[i] )
            message(1,"BAD: %d %d %d\n",i,a[i-1],a[i]);
        assert_true(a[i-1] <= a[i]);
    }

    srand48(8675309);
    for(i = 0; i < size; i++) 
        a[i] = (int) (size * drand48());

    start = omp_get_wtime();
    qsort(a, size, sizeof(int), compare);
    end = omp_get_wtime();

    message(1, "serial sort time = %g s\n",end-start);

}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_openmpsort),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
