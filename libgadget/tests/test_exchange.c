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

#include <libgadget/exchange.h>
#include <libgadget/domain.h>
#include <libgadget/slotsmanager.h>
#include <libgadget/partmanager.h>
/*Note this includes the garbage collection!
 * Should be tested separately.*/
#include <libgadget/slotsmanager.c>
#include "stub.h"

double walltime_measure_full(char * name, char * file, int line) {
    return MPI_Wtime();
}

struct part_manager_type PartManager[1] = {{0}};
int NTask, ThisTask;
int TotNumPart;

#define NUMPART1 8
static int
setup_particles(int NType[6])
{
    MPI_Barrier(MPI_COMM_WORLD);
    PartManager->MaxPart = 1024;
    int ptype;
    PartManager->NumPart = 0;
    for(ptype = 0; ptype < 6; ptype ++) {
        PartManager->NumPart += NType[ptype];
    }

    P = (struct particle_data *) mymalloc("P", PartManager->MaxPart * sizeof(struct particle_data));
    memset(P, 0, sizeof(struct particle_data) * PartManager->MaxPart);

    slots_init(0.01 * PartManager->MaxPart, SlotsManager);
    slots_set_enabled(0, sizeof(struct sph_particle_data), SlotsManager);
    slots_set_enabled(4, sizeof(struct star_particle_data), SlotsManager);
    slots_set_enabled(5, sizeof(struct bh_particle_data), SlotsManager);


    slots_reserve(1, NType, SlotsManager);

    int i;

    ptype = 0;
    int itype = 0;
    for(i = 0; i < PartManager->NumPart; i ++) {
        P[i].ID = i + PartManager->NumPart * ThisTask;
        P[i].Type = ptype;
        itype ++;
        if(itype == NType[ptype]) {
            ptype++; itype = 0;
        }
    }
    slots_setup_topology(PartManager, SlotsManager);

    slots_setup_id(PartManager, SlotsManager);

    MPI_Allreduce(&PartManager->NumPart, &TotNumPart, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    return 0;
}

static int
teardown_particles(void **state)
{
    int TotNumPart2;

    MPI_Allreduce(&PartManager->NumPart, &TotNumPart2, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    assert_int_equal(TotNumPart2, TotNumPart);

    slots_free(SlotsManager);
    myfree(P);
    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
}


static int
test_exchange_layout_func(int i, const void * userdata)
{
    return P[i].ID % NTask;
}

static void
test_exchange(void **state)
{
    int newSlots[6] = {NUMPART1, NUMPART1, NUMPART1, NUMPART1, NUMPART1, NUMPART1};

    setup_particles(newSlots);

    int i;

    int fail = domain_exchange(&test_exchange_layout_func, NULL, 1, PartManager, SlotsManager,10000, MPI_COMM_WORLD);

    assert_all_true(!fail);

    slots_check_id_consistency(PartManager, SlotsManager);
    domain_test_id_uniqueness(PartManager);

    for(i = 0; i < PartManager->NumPart; i ++) {
        assert_true(P[i].ID % NTask == ThisTask);
    }

    teardown_particles(state);
    return;
}

static void
test_exchange_zero_slots(void **state)
{
    int newSlots[6] = {NUMPART1, 0, NUMPART1, 0, NUMPART1, 0};

    setup_particles(newSlots);

    int i;

    int fail = domain_exchange(&test_exchange_layout_func, NULL, 1, PartManager, SlotsManager, 10000, MPI_COMM_WORLD);

    assert_all_true(!fail);

    slots_check_id_consistency(PartManager, SlotsManager);
    domain_test_id_uniqueness(PartManager);

    for(i = 0; i < PartManager->NumPart; i ++) {
        assert_true (P[i].ID % NTask == ThisTask);
    }

    teardown_particles(state);
    return;
}

static void
test_exchange_with_garbage(void **state)
{
    int newSlots[6] = {NUMPART1, NUMPART1, NUMPART1, NUMPART1, NUMPART1, NUMPART1};

    setup_particles(newSlots);
    int i;

    slots_mark_garbage(0, PartManager, SlotsManager); /* watch out! this propogates the garbage flag to children */
    TotNumPart -= NTask;

    int fail = domain_exchange(&test_exchange_layout_func, NULL, 1, PartManager, SlotsManager, 10000, MPI_COMM_WORLD);

    assert_all_true(!fail);

    domain_test_id_uniqueness(PartManager);
    slots_check_id_consistency(PartManager, SlotsManager);

    for(i = 0; i < PartManager->NumPart; i ++) {
        assert_true (P[i].ID % NTask == ThisTask);
    }

    for(i = 0; i < PartManager->NumPart; i ++) {
        assert_true (P[i].IsGarbage == 0);
    }

    teardown_particles(state);
    return;
}

static int
test_exchange_layout_func_uneven(int i, const void * userdata)
{
    if(P[i].Type == 0) return 0;

    return P[i].ID % NTask;
}

static void
test_exchange_uneven(void **state)
{
    int newSlots[6] = {NUMPART1, NUMPART1, NUMPART1, NUMPART1, NUMPART1, NUMPART1};

    setup_particles(newSlots);
    int i;

    /* this will trigger a slot growth on slot type 0 due to the inbalance */
    int fail = domain_exchange(&test_exchange_layout_func_uneven, NULL, 1, PartManager, SlotsManager, 10000, MPI_COMM_WORLD);

    assert_all_true(!fail);

    if(ThisTask == 0) {
        /* the slot type must have grown automatically to handle the new particles. */
        assert_int_equal(SlotsManager->info[0].size, NUMPART1 * NTask);
    }

    slots_check_id_consistency(PartManager, SlotsManager);
    domain_test_id_uniqueness(PartManager);

    for(i = 0; i < PartManager->NumPart; i ++) {
        if(P[i].Type == 0) {
            assert_true (ThisTask == 0);
        } else {
            assert_true(P[i].ID % NTask == ThisTask);
        }
    }

    teardown_particles(state);
    return;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_exchange_with_garbage),
        cmocka_unit_test(test_exchange),
        cmocka_unit_test(test_exchange_zero_slots),
        cmocka_unit_test(test_exchange_uneven),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
