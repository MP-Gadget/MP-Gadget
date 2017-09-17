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

int ThisTask;
int NTask;

int
_cmocka_run_group_tests_mpi(const char * name, const struct CMUnitTest tests[], int size, void * p1, void * p2)
{
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
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

