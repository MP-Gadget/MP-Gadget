/*Simple test for the tree building functions*/

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include <libgadget/forcetree.h>
#include <libgadget/partmanager.h>
#include <libgadget/domain.h>
#include <libgadget/walltime.h>

#include "stub.h"

/* The true struct for the state variable*/
struct forcetree_testdata
{
    DomainDecomp ddecomp;
    boost::random::mt19937 r;
};

#define NODECACHE_SIZE 100

/*This checks that the moments of the force tree in Nodes are valid:
 * that it the mass and flags are correct.*/
static int check_moments(const ForceTree * tb, const int numpart, const int nrealnode)
{
    double * oldmass = malloc(sizeof(double) * tb->numnodes);
    int i;

    for(i=tb->firstnode; i < tb->numnodes + tb->firstnode; i ++) {
        oldmass[i - tb->firstnode] = tb->Nodes[i].mom.mass;
    }

    for(i=0; i<numpart; i++)
    {
        int fnode = force_get_father(i, tb);
        /*Subtract mass so that nothing is left.*/
        assert_true(fnode >= tb->firstnode && fnode < tb->lastnode);
        while(fnode > 0) {
            tb->Nodes[fnode].mom.mass -= P[i].Mass;
            fnode = tb->Nodes[fnode].father;
            /*Validate father*/
            assert_true((fnode >= tb->firstnode && fnode < tb->lastnode) || fnode == -1);
        }
    }
    int node = tb->firstnode;
    int counter = 0;
    int sibcntr = 0;
    while(node >= 0) {
        /* Assert a real node*/
        assert_true(node >= -1 && node < tb->lastnode && node >= tb->firstnode);
        struct NODE * nop = &tb->Nodes[node];

        /*Check sibling*/
        assert_true(tb->Nodes[node].sibling >= -1 && tb->Nodes[node].sibling < tb->lastnode);
        int sib = tb->Nodes[node].sibling;
        int sfather = force_get_father(sib, tb);
        int father = force_get_father(node, tb);
        /* Our sibling should either be a true sibling, with the same father,
            * or should be the child of one of our ancestors*/
        if(sfather != father && sib != -1) {
            int ances = father;
            while(ances >= 0) {
                assert_true(ances >= tb->firstnode);
                ances = force_get_father(ances, tb);
                if(ances == sfather)
                    break;
            }
            assert_int_equal(ances, sfather);
/*                 printf("node %d ances %d sib %d next %d father %d sfather %d\n",node, ances, sib, nop->s.suns[0], father, sfather); */
        }
        else if(sib == -1)
            sibcntr++;

        if(!(tb->Nodes[node].mom.mass < 0.5 && tb->Nodes[node].mom.mass > -0.5)) {
            printf("node %d (%d) mass %g / %g TL %d DLM %d ITL %d\n",
                node, node - tb->firstnode, tb->Nodes[node].mom.mass, oldmass[node - tb->firstnode],
                tb->Nodes[node].f.TopLevel,
                tb->Nodes[node].f.DependsOnLocalMass,
                tb->Nodes[node].f.InternalTopLevel
            );
            /* something is wrong show the particles */
            if(tb->Nodes[node].f.ChildType == PARTICLE_NODE_TYPE)
                for(i = 0; i < nop->s.noccupied; i++) {
                    int nn = nop->s.suns[i];
                    printf("particles P[%d], Mass=%g\n", nn, P[nn].Mass);
                }
        }
        assert_true(tb->Nodes[node].mom.mass < 0.5 && tb->Nodes[node].mom.mass > -0.5);
        /*Check center of mass moments*/
        for(i=0; i<3; i++)
            assert_true(tb->Nodes[node].mom.cofm[i] <= PartManager->BoxSize && tb->Nodes[node].mom.cofm[i] >= 0);
        counter++;

        if(nop->f.ChildType == PARTICLE_NODE_TYPE)
            node = nop->sibling;
        else
            node = nop->s.suns[0];
    }
//     message(5, "count %d real %d\n", counter, nrealnode);
    assert_true(counter <= nrealnode);
    assert_true(sibcntr < counter/100);

    free(oldmass);
    return nrealnode;
}

/*This checks that the force tree in Nodes is valid:
 * that it contains every particle and that each parent
 * node contains particles within the right subnode.*/
static int check_tree(const ForceTree * tb, const int nnodes, const int numpart)
{
    const int firstnode = tb->firstnode;
    int tot_empty = 0, nrealnode = 0, sevens = 0;
    int i;
    for(i=firstnode; i<nnodes+firstnode; i++)
    {
        struct NODE * pNode = &(tb->Nodes[i]);
        /*Just reserved free space with nothing in it*/
        if(pNode->father < -1.5)
            continue;

        int j;
        /* Full of particles*/
        if(pNode->s.noccupied < 1<<16) {
            tot_empty += NMAXCHILD - pNode->s.noccupied;
            if(pNode->s.noccupied == 0)
                sevens++;
            for(j=0; j<pNode->s.noccupied; j++) {
                int child = pNode->s.suns[j];
                assert_true(child >= 0);
                assert_true(child < firstnode);
                P[child].PI += 1;
                assert_int_equal(force_get_father(child, tb), i);
            }
        }
        /* Node is full of other nodes*/
        else {
            for(j=0; j<8; j++) {
                /*Check children*/
                int child = pNode->s.suns[j];
                assert_true(child < firstnode+nnodes);
                assert_true(child >= firstnode);
                assert_true(fabs(tb->Nodes[child].len/pNode->len - 0.5) < 1e-4);
                int k;
                for(k=0; k<3; k++) {
                    if(j & (1<<k))
                        assert_true(tb->Nodes[child].center[k] > pNode->center[k]);
                    else
                        assert_true(tb->Nodes[child].center[k] <= pNode->center[k]);
                }
            }
        }
        nrealnode++;
    }

    for(i=0; i<numpart; i++)
    {
        assert_int_equal(P[i].PI, 1);
    }
    printf("Tree filling factor: %g on %d nodes (wasted: %d empty: %d)\n", tot_empty/(8.*nrealnode), nrealnode, nnodes - nrealnode, sevens);
    return nrealnode - sevens;
}

static void do_tree_test(const int numpart, ForceTree tb, DomainDecomp * ddecomp)
{
    /*Sort by peano key so this is more realistic*/
    int i;
    #pragma omp parallel for
    for(i=0; i<numpart; i++) {
        P[i].Mass = 1;
        P[i].PI = 0;
        P[i].IsGarbage = 0;
    }
    int maxnode = tb.lastnode - tb.firstnode;
    PartManager->MaxPart = numpart;
    PartManager->NumPart = numpart;
    slots_gc_sorted(PartManager, SlotsManager);
    assert_true(tb.Nodes != NULL);
    /*So we know which nodes we have initialised*/
    for(i=0; i< maxnode; i++)
        tb.Nodes_base[i].father = -2;
    /*Time creating the nodes*/
    double start, end;
    start = MPI_Wtime();
    ActiveParticles Act = init_empty_active_particles(PartManager);
    tb.mask = ALLMASK;
    force_tree_create_nodes(&tb, &Act, ALLMASK, ddecomp);
    assert_true(tb.numnodes < maxnode);
    end = MPI_Wtime();
    double ms = (end - start)*1000;
    printf("Number of nodes used: %d. Built tree in %.3g ms\n", tb.numnodes,ms);
    int nrealnode = check_tree(&tb, tb.numnodes, numpart);
    /* now compute the multipole moments recursively */
    start = MPI_Wtime();
    force_update_node_parallel(&tb, ddecomp);
    end = MPI_Wtime();
    ms = (end - start)*1000;
    printf("Updated moments in %.3g ms. Total mass: %g\n", ms, tb.Nodes[tb.firstnode].mom.mass);
    assert_true(fabs(tb.Nodes[tb.firstnode].mom.mass - numpart) < 0.5);
    check_moments(&tb, numpart, nrealnode);
}

/* Find the hmax value between a node and a particle.*/
static double compute_distance(int i, struct NODE * node)
{
    double hmax = 0;
    int j;
    for(j = 0; j < 3; j++) {
        /* Compute each direction independently and take the maximum.
            * This is the largest possible distance away from node center within a cube bounding hsml.
            * Note that because Pos - Center < len/2, the maximum value this can have is Hsml.*/
        hmax = DMAX(hmax, fabs(P[i].Pos[j] - node->center[j]) + P[i].Hsml - node->len/2.);
    }
    return hmax;
}

/* Test whether particle is inside node.
 * 0 if outside node, 1 if inside node. */
static double check_inside(int i, struct NODE * node)
{
    int j;
    for(j = 0; j < 3; j++)
        if (fabs(P[i].Pos[j] - node->center[j]) > node->len/2)
            return 0;
    return 1;
}

/* This checks that the hmax moment in Nodes is correct:
 * that is the distance between each particle and node center is larger than hmax.*/
static int check_hmax(const ForceTree * tb, const int numpart)
{
    int i;
    for(i=0; i<numpart; i++)
    {
        int j = tb->Father[i];
        while(j >= 0) {
            /* Test whether particle is in node*/
            assert_true(check_inside(i, &tb->Nodes[j]));
            /* Test whether hmax is set correctly*/
            assert_false(compute_distance(i, &tb->Nodes[j]) > tb->Nodes[j].mom.hmax+1e-5);
            assert_false(tb->Nodes[j].mom.hmax < 0);
            j = tb->Nodes[j].father;
        }
    }
    return 0;
}

static void do_tree_mask_hmax_update_test(const int numpart, ForceTree * tb, DomainDecomp * ddecomp)
{
    RandTable rnd = set_random_numbers(23, 8192);

    /*Sort by peano key so this is more realistic*/
    int i;
    #pragma omp parallel for
    for(i=0; i<numpart; i++) {
        P[i].Mass = 1;
        P[i].PI = 0;
        P[i].IsGarbage = 0;
        P[i].Type = 0;
        P[i].Hsml = PartManager->BoxSize/cbrt(numpart) * get_random_number(i, &rnd);
    }
    free_random_numbers(&rnd);
    PartManager->MaxPart = numpart;
    PartManager->NumPart = numpart;
    SlotsManager->info[0].enabled = 0;
    slots_gc_sorted(PartManager, SlotsManager);
    assert_true(tb->Nodes != NULL);
    /*Time creating the nodes*/
    double start, end;
    start = MPI_Wtime();
    ActiveParticles Act = init_empty_active_particles(PartManager);
    tb->mask = GASMASK;
    force_tree_create_nodes(tb, &Act, GASMASK, ddecomp);
    end = MPI_Wtime();
    double ms = (end - start)*1000;
    printf("Built gas tree in %.3g ms\n", ms);
    /* now compute the multipole moments recursively */
    start = MPI_Wtime();
    force_update_hmax(&Act, tb, ddecomp);
    end = MPI_Wtime();
    ms = (end - start)*1000;
    printf("Updated hmax in %.3g ms. Root hmax: %g\n", ms, tb->Nodes[tb->firstnode].mom.hmax);
    check_hmax(tb, numpart);
}

static void test_rebuild_flat(void ** state) {
    /*Set up the particle data*/
    int ncbrt = 128;
    int numpart = ncbrt*ncbrt*ncbrt;
    particle_alloc_memory(PartManager, 8, numpart);
    /* Create a regular grid of particles, 8x8x8, all of type 1,
     * in a box 8 kpc across.*/
    int i;
    #pragma omp parallel for
    for(i=0; i<numpart; i++) {
        P[i].Type = 1;
        P[i].Pos[0] = (PartManager->BoxSize/ncbrt) * (i/ncbrt/ncbrt);
        P[i].Pos[1] = (PartManager->BoxSize/ncbrt) * ((i/ncbrt) % ncbrt);
        P[i].Pos[2] = (PartManager->BoxSize/ncbrt) * (i % ncbrt);
    }
    PartManager->NumPart = numpart;
    /*Allocate tree*/
    /*Base pointer*/
    struct forcetree_testdata * data = * (struct forcetree_testdata **) state;
    DomainDecomp ddecomp = data->ddecomp;
    ddecomp.TopLeaves[0].treenode = numpart;
    ForceTree tb = force_treeallocate(0.7*numpart, numpart, &ddecomp, 1, 0);
    /* So unused memory has Father < 0*/
    for(i = tb.firstnode; i < tb.lastnode; i++)
        tb.Nodes[i].father = -10;

    do_tree_test(numpart, tb, &ddecomp);
    force_tree_free(&tb);
    tb = force_treeallocate(0.7*numpart, numpart, &ddecomp, 1, 0);
    do_tree_mask_hmax_update_test(numpart, &tb, &ddecomp);
    assert_true(tb.Nodes[tb.firstnode].mom.hmax >= 0.0584);
    force_tree_free(&tb);
    myfree(PartManager->Base);
}

static void test_rebuild_close(void ** state) {
    /*Set up the particle data*/
    int ncbrt = 128;
    int numpart = ncbrt*ncbrt*ncbrt;
    double close = 5000;
    particle_alloc_memory(PartManager, 8, numpart);
    /* Create particles clustered in one place, all of type 1.*/
    int i;
    #pragma omp parallel for
    for(i=0; i<numpart; i++) {
        P[i].Type = 1;
        P[i].Pos[0] = 4. + (i/ncbrt/ncbrt)/close;
        P[i].Pos[1] = 4. + ((i/ncbrt) % ncbrt) /close;
        P[i].Pos[2] = 4. + (i % ncbrt)/close;
    }
    PartManager->NumPart = numpart;
    struct forcetree_testdata * data = * (struct forcetree_testdata **) state;
    DomainDecomp ddecomp = data->ddecomp;
    ddecomp.TopLeaves[0].treenode = numpart;
    ForceTree tb = force_treeallocate(0.7*numpart, numpart, &ddecomp, 1, 0);
    do_tree_test(numpart, tb, &ddecomp);
    force_tree_free(&tb);
    tb = force_treeallocate(0.7*numpart, numpart, &ddecomp, 1, 0);
    do_tree_mask_hmax_update_test(numpart, &tb, &ddecomp);
    force_tree_free(&tb);
    myfree(PartManager->Base);
}

void do_random_test(boost::random::mt19937 & r, const int numpart, const ForceTree tb, DomainDecomp * ddecomp)
{
    boost::random::uniform_real_distribution<double> dist(0, 1);
    /* Create a regular grid of particles, 8x8x8, all of type 1,
     * in a box 8 kpc across.*/
    int i;
    for(i=0; i<numpart/4; i++) {
        P[i].Type = 1;
        int j;
        for(j=0; j<3; j++)
            P[i].Pos[j] = PartManager->BoxSize * dist(r);
    }
    for(i=numpart/4; i<3*numpart/4; i++) {
        P[i].Type = 1;
        int j;
        for(j=0; j<3; j++)
            P[i].Pos[j] = PartManager->BoxSize/2 + PartManager->BoxSize/8 * exp(pow(dist(r)-0.5,2));
    }
    for(i=3*numpart/4; i<numpart; i++) {
        P[i].Type = 1;
        int j;
        for(j=0; j<3; j++)
            P[i].Pos[j] = PartManager->BoxSize*0.1 + PartManager->BoxSize/32 * exp(pow(dist(r)-0.5,2));
    }
    PartManager->NumPart = numpart;
    do_tree_test(numpart, tb, ddecomp);
}

static void test_rebuild_random(void ** state) {
    /*Set up the particle data*/
    int ncbrt = 64;
    struct forcetree_testdata * data = * (struct forcetree_testdata **) state;
    DomainDecomp ddecomp = data->ddecomp;
    boost::random::mt19937 & r = data->r;
    int numpart = ncbrt*ncbrt*ncbrt;
    particle_alloc_memory(PartManager, 8, numpart);
    /*Allocate tree*/
    /*Base pointer*/
    ddecomp.TopLeaves[0].treenode = numpart;
    ForceTree tb = force_treeallocate(0.7*numpart, numpart, &ddecomp, 1, 0);
    assert_true(tb.Nodes != NULL);
    int i;
    for(i=0; i<2; i++) {
        do_random_test(r, numpart, tb, &ddecomp);
    }
    force_tree_free(&tb);
    tb = force_treeallocate(0.7*numpart, numpart, &ddecomp, 1, 0);
    do_tree_mask_hmax_update_test(numpart, &tb, &ddecomp);
    force_tree_free(&tb);
    myfree(PartManager->Base);
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
    ddecomp->TopNodes = malloc(sizeof(struct topnode_data));
    ddecomp->TopNodes[0].Daughter = -1;
    ddecomp->TopNodes[0].Leaf = 0;
    ddecomp->TopLeaves = malloc(sizeof(struct topleaf_data));
    ddecomp->TopLeaves[0].Task = 0;
    ddecomp->TopLeaves[0].treenode = 0;
    /*These are not used*/
    ddecomp->TopNodes[0].StartKey = 0;
    ddecomp->TopNodes[0].Shift = BITS_PER_DIMENSION * 3;
    /*To tell the code we are in serial*/
    ddecomp->Tasks = malloc(sizeof(struct task_data));
    ddecomp->Tasks[0].StartLeaf = 0;
    ddecomp->Tasks[0].EndLeaf = 1;
}

static struct ClockTable Clocks;

static int setup_tree(void **state) {
    /*Set up the important parts of the All structure.*/
    /*Particles should not be outside this*/
    memset(PartManager, 0, sizeof(PartManager[0]));
    memset(SlotsManager, 0, sizeof(SlotsManager[0]));
    PartManager->BoxSize = 8;
    init_forcetree_params(0.5);
    /*Set up the top-level domain grid*/
    struct forcetree_testdata *data = malloc(sizeof(struct forcetree_testdata));
    trivial_domain(&data->ddecomp);
    data->r = boost::random::mt19937(0);
    *state = (void *) data;
    walltime_init(&Clocks);
    return 0;
}

static int teardown_tree(void **state) {
    struct forcetree_testdata * data = (struct forcetree_testdata * ) *state;
    free(data->ddecomp.TopNodes);
    free(data->ddecomp.TopLeaves);
    free(data->ddecomp.Tasks);
    free(data->r);
    free(data);
    return 0;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_rebuild_flat),
        cmocka_unit_test(test_rebuild_close),
        cmocka_unit_test(test_rebuild_random),
    };
    return cmocka_run_group_tests_mpi(tests, setup_tree, teardown_tree);
}
