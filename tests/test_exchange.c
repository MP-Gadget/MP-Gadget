/*Simple test for the exchange function*/

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <gsl/gsl_rng.h>

#define qsort_openmp qsort

#include "exchange.h"
#include "slotsmanager.h"
#include "allvars.h"
/*Note this includes the garbage collection!
 * Should be tested separately.*/
#include "slotsmanager.c"
#include "stub.h"

struct particle_data *P;
struct global_data_all_processes All;
int NTask, ThisTask;
int NumPart;

#define NUMPART1 8
static int
setup_particles(void ** state)
{
    All.MaxPart = 1024;
    NumPart = NUMPART1 * 6;

    int newSlots[6] = {NUMPART1, NUMPART1, NUMPART1, NUMPART1, NUMPART1, NUMPART1};

    P = (struct particle_data *) mymalloc("P", All.MaxPart * sizeof(struct particle_data));
    memset(P, 0, sizeof(struct particle_data) * All.MaxPart);

    slots_init();

    slots_reserve(newSlots);

    int i;
    for(i = 0; i < NumPart; i ++) {
        P[i].ID = i + NumPart * ThisTask;
        P[i].Type = i / (NumPart / 6);
    }
    slots_setup_topology();

    slots_setup_id();

    return 0;
}

static int
teardown_particles(void **state)
{
    slots_free();
    myfree(P);
    return 0;
}


static int
test_exchange_layout_func(int i)
{
    return P[i].ID % NTask;
}

static void
test_exchange(void **state)
{
    setup_particles(state);
    int i;

    int fail = domain_exchange(&test_exchange_layout_func, 1);

    assert_false(fail);

#if 0

    int task;
    for(task = 0; task < NTask; task++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (task != ThisTask) continue;
        for(i = 0; i < NumPart; i ++) {
            printf("P[%d] = %ld\n", i, P[i].ID);
        }
    }
#endif

    slots_check_id_consistency();
    domain_test_id_uniqueness();

    for(i = 0; i < NumPart; i ++) {
        assert_true(P[i].ID % NTask == ThisTask);
    }

    teardown_particles(state);
    return;
}

static void
test_exchange_with_garbage(void **state)
{
    setup_particles(state);
    int i;

    P[0].IsGarbage = 1;

    int fail = domain_exchange(&test_exchange_layout_func, 1);

    assert_false(fail);

    slots_check_id_consistency();
    domain_test_id_uniqueness();

    for(i = 0; i < NumPart; i ++) {
        assert_true(P[i].ID % NTask == ThisTask);
    }

    teardown_particles(state);
    return;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_exchange),
        cmocka_unit_test(test_exchange_with_garbage)
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
