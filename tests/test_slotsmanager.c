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
#include "allvars.h"
#include "domain.h"
#include "slotsmanager.h"
#include "mymalloc.h"
#include "stub.h"

struct particle_data *P;
struct global_data_all_processes All;
int NTask, ThisTask;
int NumPart;

static int
setup_particles(void ** state)
{
    All.MaxPart = 1024;
    NumPart = 128 * 6;

    int newSlots[6] = {128, 128, 128, 128, 128, 128};

    P = (struct particle_data *) mymalloc("P", All.MaxPart * sizeof(struct particle_data));
    memset(P, 0, sizeof(struct particle_data) * All.MaxPart);

    slots_init();

    slots_reserve(newSlots);

    int i;
    for(i = 0; i < NumPart; i ++) {
        P[i].ID = i;
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

static void
test_slots_gc(void **state)
{
    setup_particles(state);
    int i;
    for(i = 0; i < 6; i ++) {
        slots_mark_garbage(128 * i);
    }
    slots_gc();
    assert_int_equal(NumPart, 127 * i);

    assert_int_equal(SlotsManager->info[0].size, 127);
    assert_int_equal(SlotsManager->info[4].size, 127);
    assert_int_equal(SlotsManager->info[5].size, 127);

    teardown_particles(state);
    return;
}

static void
test_slots_reserve(void **state)
{
    /* FIXME: these depends on the magic numbers in slots_reserve. After
     * moving those numbers to All.* we shall rework the code here. */
    setup_particles(state);

    int newSlots[6] = {128, 128, 128, 128, 128, 128};
    int oldSize[6];
    int ptype;
    for(ptype = 0; ptype < 6; ptype++) {
        oldSize[ptype] = SlotsManager->info[ptype].maxsize;
    }
    slots_reserve(newSlots);

    /* shall not increase max size*/
    for(ptype = 0; ptype < 6; ptype++) {
        assert_int_equal(oldSize[ptype], SlotsManager->info[ptype].maxsize);
    }

    for(ptype = 0; ptype < 6; ptype++) {
        newSlots[ptype] += 1;
    }

    /* shall not increase max size; because it is small difference */
    slots_reserve(newSlots);
    for(ptype = 0; ptype < 6; ptype++) {
        assert_int_equal(oldSize[ptype], SlotsManager->info[ptype].maxsize);
    }

    for(ptype = 0; ptype < 6; ptype++) {
        newSlots[ptype] += 128;
    }

    /* shall increase max size; because it large difference */
    slots_reserve(newSlots);

    for(ptype = 0; ptype < 6; ptype++) {
        assert_true(oldSize[ptype] < SlotsManager->info[ptype].maxsize);
    }

}

static void
test_slots_fork(void **state)
{
    setup_particles(state);
    int i;
    for(i = 0; i < 6; i ++) {
        slots_fork(128 * i, P[i * 128].Type);
    }

    assert_int_equal(NumPart, 129 * i);

    assert_int_equal(SlotsManager->info[0].size, 129);
    assert_int_equal(SlotsManager->info[4].size, 129);
    assert_int_equal(SlotsManager->info[5].size, 129);

    teardown_particles(state);
    return;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_slots_gc),
        cmocka_unit_test(test_slots_reserve),
        cmocka_unit_test(test_slots_fork),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
