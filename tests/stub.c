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
#include <cmocka.h>

#include <libgadget/utils.h>

#ifdef VALGRIND
#define allocator_init allocator_malloc_init
#endif

int
_cmocka_run_group_tests_mpi(const char * name, const struct CMUnitTest tests[], size_t size, void * p1, void * p2)
{
    MPI_Init(NULL, NULL);
    int NTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);

    init_endrun(1);

    if(NTask != 1) {
        setenv("CMOCKA_TEST_ABORT", "1", 1);
    }
    /* allocate some memory for MAIN and TEMP */

    allocator_init(A_MAIN, "MAIN", 650 * 1024 * 1024, 0, NULL);
    allocator_init(A_TEMP, "TEMP", 8 * 1024 * 1024, 0, A_MAIN);

    message(0, "GADGET_TESTDATA_ROOT : %s\n", GADGET_TESTDATA_ROOT);

    int rt = _cmocka_run_group_tests(name, tests, size, p1, p2);
    MPI_Finalize();
    return rt;
}

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

