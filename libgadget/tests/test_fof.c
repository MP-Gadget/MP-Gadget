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
    domain_decompose_full(&ddecomp);

    FOFGroups fof = fof_fof(&ddecomp, 1, MPI_COMM_WORLD);

    /* Example assertion: this checks that the groups were allocated. */
    assert_all_true(fof.Group);
    assert_true(fof.TotNgroups == 1);
    /* Assert some more things about the particles,
     * maybe checking the halo properties*/

    fof_finish(&fof);
    domain_free(&ddecomp);
    myfree(SlotsManager->Base);
    myfree(P);
    return;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_fof),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
