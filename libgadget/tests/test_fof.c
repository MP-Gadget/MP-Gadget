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

#include <libgadget/fof.h>
#include <libgadget/walltime.h>
#include <libgadget/domain.h>
#include <libgadget/forcetree.h>
#include <libgadget/partmanager.h>
#include "stub.h"

static struct ClockTable CT;

#define NUMPART1 8
static int
setup_particles(int NumPart, double BoxSize)
{

    int ThisTask, NTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);

    particle_alloc_memory(PartManager, BoxSize, 1.5 * NumPart);
    PartManager->NumPart = NumPart;

    slots_init(0.01 * PartManager->MaxPart, SlotsManager);
    slots_set_enabled(0, sizeof(struct sph_particle_data), SlotsManager);
    slots_set_enabled(4, sizeof(struct star_particle_data), SlotsManager);
    slots_set_enabled(5, sizeof(struct bh_particle_data), SlotsManager);

    int64_t newSlots[6] = {128, 0, 0, 0, 128, 128};
    slots_reserve(1, newSlots, SlotsManager);
    int i;
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i ++) {
        P[i].ID = i + PartManager->NumPart * ThisTask;
        /* DM only*/
        P[i].Type = 1;
        P[i].Mass = 1;
        P[i].IsGarbage = 0;
        int j;
        for(j=0; j<3; j++) {
            P[i].Pos[j] = BoxSize * (j+1) * P[i].ID / (PartManager->NumPart * NTask);
            while(P[i].Pos[j] > BoxSize)
                P[i].Pos[j] -= BoxSize;
        }
    }
    fof_init(BoxSize/cbrt(PartManager->NumPart));
    /* TODO: Here create particles in some halo-like configuration*/
    return 0;
}

static void
test_fof(void **state)
{
    int NTask;
    walltime_init(&CT);

    struct DomainParams dp = {0};
    dp.DomainOverDecompositionFactor = 1;
    dp.DomainUseGlobalSorting = 0;
    dp.TopNodeAllocFactor = 1.;
    dp.SetAsideFactor = 1;
    set_domain_par(dp);
    set_fof_testpar(1, 0.2, 5);
    init_forcetree_params(0.7);

    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    int NumPart = 512*512 / NTask;
    /* 20000 kpc*/
    double BoxSize = 20000;
    setup_particles(NumPart, BoxSize);

    /* Build a tree and domain decomposition*/
    DomainDecomp ddecomp = {0};
    domain_decompose_full(&ddecomp, MPI_COMM_WORLD);

    FOFGroups fof = fof_fof(&ddecomp, 1, MPI_COMM_WORLD);

    /* Example assertion: this checks that the groups were allocated. */
    assert_all_true(fof.Group);
    assert_true(fof.TotNgroups == 1);
    /* Assert some more things about the particles,
     * maybe checking the halo properties*/

    fof_finish(&fof);
    domain_free(&ddecomp);
    slots_free(SlotsManager);
    myfree(P);
    return;
}

/* Runs fof_update_root_for_test(i, r) on a copy of Head and compares with expect. */
static void
check_update_root(const int * Head, const int * expect, int n, int i, int r)
{
    int h[8];
    int k;
    assert_true(n <= 8);
    memcpy(h, Head, sizeof(int) * n);
    fof_update_root_for_test(i, r, h);
    for(k = 0; k < n; k++) {
        if(h[k] != expect[k])
            message(1, "update_root(%d, %d): Head[%d] = %d, expected %d\n", i, r, k, h[k], expect[k]);
        assert_int_equal(h[k], expect[k]);
    }
}

/* update_root, the path compression after fofp_merge links two roots. */
static void
test_fof_update_root(void **state)
{
    /* (a) stale root: h1 = 1 was since linked below 0; the old code wrote {0, 1, 1} */
    {
        const int Head[3] = {0, 0, 1};
        check_update_root(Head, Head, 3, 1, 1);
    }
    /* (b) i == r, the root of its own tree: unchanged */
    {
        const int Head[4] = {0, 1, 1, 2};
        check_update_root(Head, Head, 4, 1, 1);
        const int One[1] = {0};
        check_update_root(One, One, 1, 0, 0);
    }
    /* (c) chain compression: the whole path points to 0, Head[0] untouched */
    {
        const int Head[5] = {0, 0, 1, 2, 3};
        const int expect[5] = {0, 0, 0, 0, 0};
        check_update_root(Head, expect, 5, 4, 0);
    }
    /* (d) node 3's parent 1 is already below r = 2: the writes stop after node 3 */
    {
        const int Head[5] = {0, 0, 1, 1, 3};
        const int expect[5] = {0, 0, 1, 2, 2};
        check_update_root(Head, expect, 5, 4, 2);
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_fof_update_root),
        cmocka_unit_test(test_fof),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
