/*Simple test for the tree building functions*/

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <gsl/gsl_rng.h>

#include <libgadget/allvars.h>
#include <libgadget/partmanager.h>
#include <libgadget/slotsmanager.h>
#include <libgadget/utils/mymalloc.h>
#include <libgadget/density.h>
#include <libgadget/domain.h>
#include <libgadget/forcetree.h>
#include <libgadget/timestep.h>

#include "stub.h"

/* The true struct for the state variable*/
struct forcetree_testdata
{
    DomainDecomp ddecomp;
    gsl_rng * r;
};

static void set_init_hsml(ForceTree * Tree)
{
    int i;
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)
    {
        int no = force_get_father(i, Tree);

        while(10 * All.DesNumNgb * P[i].Mass > Tree->Nodes[no].u.d.mass)
        {
            int p = force_get_father(no, Tree);

            if(p < 0)
                break;

            no = p;
        }

        P[i].Hsml =
            pow(3.0 / (4 * M_PI) * All.DesNumNgb * P[i].Mass / (Tree->Nodes[no].u.d.mass),
                    1.0 / 3) * Tree->Nodes[no].len;
    }
}

/* Perform some simple checks on the densities*/
static void check_densities(void)
{
    int i;
    double maxHsml=P[0].Hsml, minHsml= P[0].Hsml;
    #pragma omp parallel for reduction(min:minHsml) reduction(max:maxHsml)
    for(i=0; i<PartManager->NumPart; i++) {
        assert_true(isfinite(P[i].Hsml));
        assert_true(isfinite(SPHP(i).Density));
        assert_true(SPHP(i).Density > 0);
        if(P[i].Hsml < minHsml)
            minHsml = P[i].Hsml;
        if(P[i].Hsml > maxHsml)
            maxHsml = P[i].Hsml;
    }
    assert_true(isfinite(minHsml));
    assert_true(minHsml >= All.MinGasHsml);
    assert_true(maxHsml <= All.BoxSize);

}

static void do_density_test(void ** state, const int numpart, double expectedhsml, double hsmlerr)
{
    int i, npbh=0;
    #pragma omp parallel for reduction(+: npbh)
    for(i=0; i<numpart; i++) {
        int j;
        P[i].Key = PEANO(P[i].Pos, All.BoxSize);
        P[i].Mass = 1;
        P[i].TimeBin = 0;
        P[i].GravCost = 1;
        P[i].Ti_drift = 0;
        P[i].Ti_kick = 0;
        for(j=0; j<3; j++)
            P[i].Vel[j] = 1.5;
        if(P[i].Type == 0) {
            SPHP(i).Entropy = 1;
            SPHP(i).DtEntropy = 0;
            SPHP(i).Density = 1;
        }
        if(P[i].Type == 5)
            npbh++;
    }

    SlotsManager->info[0].size = numpart-npbh;
    SlotsManager->info[5].size = npbh;
    PartManager->NumPart = numpart;
    ActiveParticles act = {0};
    act.NumActiveParticle = numpart;
    act.ActiveParticle = NULL;
    struct forcetree_testdata * data = * (struct forcetree_testdata **) state;
    DomainDecomp ddecomp = data->ddecomp;
    ddecomp.TopLeaves[0].topnode = PartManager->MaxPart;

    ForceTree tree = {0};
    force_tree_rebuild(&tree, &ddecomp, All.BoxSize, 0);
    set_init_hsml(&tree);
    /*Time doing the density finding*/
    double start, end;
    start = MPI_Wtime();
    /*Find the density*/
    density(&act, 1, 0, &tree);
    end = MPI_Wtime();
    double ms = (end - start)*1000;
    message(0, "Found densities in %.3g ms\n", ms);
    check_densities();

    double avghsml = 0;
    #pragma omp parallel for reduction(+:avghsml)
    for(i=0; i<numpart; i++) {
        avghsml += P[i].Hsml;
    }
    message(0, "Average Hsml: %g Expected %g +- %g\n",avghsml/numpart, expectedhsml, hsmlerr);
    assert_true(fabs(avghsml/numpart - expectedhsml) < hsmlerr);
    /* Make MaxNumNgbDeviation smaller and check we get a consistent result.*/
    double * Hsml = mymalloc("Hsml", numpart * sizeof(double));
    #pragma omp parallel for
    for(i=0; i<numpart; i++) {
        Hsml[i] = P[i].Hsml;
    }
    All.MaxNumNgbDeviation = 1;

    start = MPI_Wtime();
    /*Find the density*/
    density(&act, 1, 0, &tree);
    end = MPI_Wtime();
    ms = (end - start)*1000;
    message(0, "Found 1 dev densities in %.3g ms\n", ms);
    double diff = 0;
    #pragma omp parallel for reduction(max:diff)
    for(i=0; i<numpart; i++) {
        assert_true(fabs(Hsml[i]/P[i].Hsml-1) < All.MaxNumNgbDeviation / All.DesNumNgb);
        if(fabs(Hsml[i] - P[i].Hsml) > diff)
            diff = fabs(Hsml[i] - P[i].Hsml);
    }
    message(0, "Max diff between Hsml: %g\n",diff);
    myfree(Hsml);

    check_densities();
    force_tree_free(&tree);
}

static void test_density_flat(void ** state) {
    /*Set up the particle data*/
    int ncbrt = 32;
    int numpart = ncbrt*ncbrt*ncbrt;
    /* Create a regular grid of particles, 8x8x8, all of type 1,
     * in a box 8 kpc across.*/
    int i;
    #pragma omp parallel for
    for(i=0; i<numpart; i++) {
        P[i].Type = 0;
        P[i].PI = i;
        P[i].Hsml = 1.5*All.BoxSize/cbrt(numpart);
        P[i].Pos[0] = (All.BoxSize/ncbrt) * (i/ncbrt/ncbrt);
        P[i].Pos[1] = (All.BoxSize/ncbrt) * ((i/ncbrt) % ncbrt);
        P[i].Pos[2] = (All.BoxSize/ncbrt) * (i % ncbrt);
    }
    do_density_test(state, numpart, 0.508875, 1e-4);
}

static void test_density_close(void ** state) {
    /*Set up the particle data*/
    int ncbrt = 32;
    int numpart = ncbrt*ncbrt*ncbrt;
    double close = 500.;
    int i;
    /* A few particles scattered about the place so the tree is not sparse*/
    #pragma omp parallel for
    for(i=0; i<numpart/8; i++) {
        P[i].Type = 0;
        P[i].PI = i;
        P[i].Hsml = 4*All.BoxSize/cbrt(numpart/8);
        P[i].Pos[0] = (All.BoxSize/ncbrt) * (i/(ncbrt/2.)/(ncbrt/2.));
        P[i].Pos[1] = (All.BoxSize/ncbrt) * ((i*2/ncbrt) % (ncbrt/2));
        P[i].Pos[2] = (All.BoxSize/ncbrt) * (i % (ncbrt/2));
    }

    /* Create particles clustered in one place, all of type 0.*/
    #pragma omp parallel for
    for(i=numpart/4; i<numpart; i++) {
        P[i].Type = 0;
        P[i].PI = i;
        P[i].Hsml = 2*ncbrt/close;
        P[i].Pos[0] = 4.1 + (i/ncbrt/ncbrt)/close;
        P[i].Pos[1] = 4.1 + ((i/ncbrt) % ncbrt) /close;
        P[i].Pos[2] = 4.1 + (i % ncbrt)/close;
    }
    P[numpart-1].Type = 5;

    do_density_test(state, numpart, 0.127294, 1e-4);
}

void do_random_test(void **state, gsl_rng * r, const int numpart)
{
    /* Create a randomly space set of particles, 8x8x8, all of type 0. */
    int i;
    for(i=0; i<numpart/4; i++) {
        P[i].Type = 0;
        P[i].PI = i;
        P[i].Hsml = All.BoxSize/cbrt(numpart);

        int j;
        for(j=0; j<3; j++)
            P[i].Pos[j] = All.BoxSize * gsl_rng_uniform(r);
    }
    for(i=numpart/4; i<3*numpart/4; i++) {
        P[i].Type = 0;
        P[i].PI = i;
        P[i].Hsml = All.BoxSize/cbrt(numpart);
        int j;
        for(j=0; j<3; j++)
            P[i].Pos[j] = All.BoxSize/2 + All.BoxSize/8 * exp(pow(gsl_rng_uniform(r)-0.5,2));
    }
    for(i=3*numpart/4; i<numpart; i++) {
        P[i].Type = 0;
        P[i].PI = i;
        P[i].Hsml = All.BoxSize/cbrt(numpart);
        int j;
        for(j=0; j<3; j++)
            P[i].Pos[j] = All.BoxSize*0.1 + All.BoxSize/32 * exp(pow(gsl_rng_uniform(r)-0.5,2));
    }
    do_density_test(state, numpart,0.1908, 1e-3);
}

static void test_density_random(void ** state) {
    /*Set up the particle data*/
    int ncbrt = 32;
    struct forcetree_testdata * data = * (struct forcetree_testdata **) state;
    gsl_rng * r = (gsl_rng *) data->r;
    int numpart = ncbrt*ncbrt*ncbrt;
    /*Allocate tree*/
    /*Base pointer*/
    int i;
    for(i=0; i<2; i++) {
        do_random_test(state, r, numpart);
    }
}


/*Make a simple trivial domain for all data on a single processor*/
void trivial_domain(DomainDecomp * ddecomp)
{
    /* The whole tree goes into one topnode.
     * Set up just enough of the TopNode structure that
     * domain_get_topleaf works*/
    ddecomp->domain_allocated_flag = 1;
    ddecomp->NTopNodes = 1;
    ddecomp->NTopLeaves = 1;
    ddecomp->TopNodes = mymalloc("topnode", sizeof(struct topnode_data));
    ddecomp->TopNodes[0].Daughter = -1;
    ddecomp->TopNodes[0].Leaf = 0;
    ddecomp->TopLeaves = mymalloc("topleaf",sizeof(struct topleaf_data));
    ddecomp->TopLeaves[0].Task = 0;
    ddecomp->TopLeaves[0].topnode = 0;
    /*These are not used*/
    ddecomp->TopNodes[0].StartKey = 0;
    ddecomp->TopNodes[0].Shift = BITS_PER_DIMENSION * 3;
    /*To tell the code we are in serial*/
    ddecomp->Tasks = mymalloc("task",sizeof(struct task_data));
    ddecomp->Tasks[0].StartLeaf = 0;
    ddecomp->Tasks[0].EndLeaf = 1;
}

static int teardown_density(void **state) {
    slots_free_sph_scratch_data(SphP_scratch);
    struct forcetree_testdata * data = (struct forcetree_testdata * ) *state;
    myfree(data->ddecomp.Tasks);
    myfree(data->ddecomp.TopLeaves);
    myfree(data->ddecomp.TopNodes);
    free(data->r);
    myfree(data);
    return 0;
}

static int setup_density(void **state) {
    /*Set up the important parts of the All structure.*/
    All.DensityOn = 1;
    All.DensityResolutionEta = 1.;
    All.BlackHoleNgbFactor = 2;
    All.MaxNumNgbDeviation = 2;
    All.DesNumNgb = 35;
    All.DensityKernelType = DENSITY_KERNEL_CUBIC_SPLINE;
    All.BoxSize = 8;
    All.NumThreads = omp_get_max_threads();
    All.MinGasHsml = 0.006;
    All.BlackHoleMaxAccretionRadius = 99999.;
    /*Reserve space for the slots*/
    slots_init(0.01);
    slots_set_enabled(0, sizeof(struct sph_particle_data));
    int maxpart = pow(32,3);
    int atleast[6] = {0};
    atleast[0] = maxpart;
    atleast[5] = 2;
    particle_alloc_memory(maxpart);
    slots_reserve(1, atleast);
    slots_allocate_sph_scratch_data(0, maxpart);
    int i;
    for(i=0; i<6; i++)
        GravitySofteningTable[i] = 0.1 / 2.8;

    walltime_init(&All.CT);
    init_forcetree_params(2, GravitySofteningTable);
    /*Set up the top-level domain grid*/
    struct forcetree_testdata *data = mymalloc("data", sizeof(struct forcetree_testdata));
    trivial_domain(&data->ddecomp);
    data->r = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(data->r, 0);
    *state = (void *) data;
    return 0;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_density_flat),
        cmocka_unit_test(test_density_close),
        cmocka_unit_test(test_density_random),
    };
    return cmocka_run_group_tests_mpi(tests, setup_density, teardown_density);
}
