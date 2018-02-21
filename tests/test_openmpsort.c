#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

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

struct __data
{
    int dd[200];
};
/*This exists because large structs are sorted indirectly*/
static void test_openmpsort_struct(void ** state) {
    int i;
    // set up array to be sorted
    //a bad case is 87763 threads = 12
    int size = 187763;
    struct __data *a = (struct __data *) malloc(size * sizeof(struct __data));

    srand48(8675309);
    for(i = 0; i < size; i++)
        a[i].dd[0] = (int) (size * drand48());

    double start, end;

    start = omp_get_wtime();
    qsort_openmp(a, size, sizeof(struct __data), compare);
    end = omp_get_wtime();

    message(1,"parallel sort time = %g s %d threads\n",end-start, omp_get_max_threads());
    for(i=1; i<size; i++) {
        if( a[i-1].dd[0] >a[i].dd[0] )
            message(1,"BAD: %d %d %d\n",i,a[i-1].dd[0],a[i].dd[0]);
        assert_true(a[i-1].dd[0] <= a[i].dd[0]);
    }

    srand48(8675309);
    for(i = 0; i < size; i++)
        a[i].dd[0] = (int) (size * drand48());

    start = omp_get_wtime();
    qsort(a, size, sizeof(struct __data), compare);
    end = omp_get_wtime();

    message(1, "serial (glibc) sort time = %g s\n",end-start);

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
        cmocka_unit_test(test_openmpsort_struct),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
