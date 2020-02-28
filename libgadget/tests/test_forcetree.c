/*Simple test for the tree building functions*/

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <gsl/gsl_rng.h>

#include <libgadget/forcetree.h>
#include <libgadget/partmanager.h>
#include <libgadget/domain.h>


#include "stub.h"

/*Defined in forcetree.c*/
/*Next three are not static as tested.*/
int
force_tree_create_nodes(const ForceTree tb, const int npart, DomainDecomp * ddecomp, const double BoxSize);

ForceTree
force_treeallocate(int maxnodes, int maxpart, DomainDecomp * ddecomp);

int
force_update_node_parallel(const ForceTree * tree, const int HybridNuGrav);

/*Particle data.*/
struct part_manager_type PartManager[1] = {{0}};
double GravitySofteningTable[6];
double BoxSize;

/* The true struct for the state variable*/
struct forcetree_testdata
{
    DomainDecomp ddecomp;
    gsl_rng * r;
};

/*Dummy versions of functions that implement only what we need for the tests:
 * most of these are used in the non-tested globally accessible parts of forcetree.c and
 * so not executed by our tests anyway.*/

void dump_snapshot() { }

double walltime_measure_full(char * name, char * file, int line) {
    return MPI_Wtime();
}

/*End dummies*/

static int
order_by_type_and_key(const void *a, const void *b)
{
    const struct particle_data * pa  = (const struct particle_data *) a;
    const struct particle_data * pb  = (const struct particle_data *) b;

    if(pa->Type < pb->Type)
        return -1;
    if(pa->Type > pb->Type)
        return +1;
    if(pa->Key < pb->Key)
        return -1;
    if(pa->Key > pb->Key)
        return +1;

    return 0;
}

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
/*                 printf("node %d ances %d sib %d next %d father %d sfather %d\n",node, ances, sib, nop->nextnode, father, sfather); */
        }
        else if(sib == -1)
            sibcntr++;

        if(!(tb->Nodes[node].mom.mass < 0.5 && tb->Nodes[node].mom.mass > -0.5)) {
            printf("node %d (%d) mass %g / %g TL %d DLM %d MS %g ITL %d\n",
                node, node - tb->firstnode, tb->Nodes[node].mom.mass, oldmass[node - tb->firstnode],
                tb->Nodes[node].f.TopLevel,
                tb->Nodes[node].f.DependsOnLocalMass,
                tb->Nodes[node].mom.MaxSoftening,
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
            assert_true(tb->Nodes[node].mom.cofm[i] <= BoxSize && tb->Nodes[node].mom.cofm[i] >= 0);
        counter++;

        if(nop->f.ChildType == PARTICLE_NODE_TYPE)
            node = nop->sibling;
        else
            node = nop->nextnode;
    }
    assert_int_equal(counter, nrealnode);
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
    assert_true(nnodes - nrealnode < omp_get_max_threads()*NODECACHE_SIZE);

    for(i=0; i<numpart; i++)
    {
        assert_int_equal(P[i].PI, 1);
    }
    printf("Tree filling factor: %g on %d nodes (wasted: %d empty: %d)\n", tot_empty/(8.*nrealnode), nrealnode, nnodes - nrealnode, sevens);
    return nrealnode - sevens;
}

static void do_tree_test(const int numpart, const ForceTree tb, DomainDecomp * ddecomp)
{
    /*Sort by peano key so this is more realistic*/
    int i;
    #pragma omp parallel for
    for(i=0; i<numpart; i++) {
        P[i].Key = PEANO(P[i].Pos, BoxSize);
        P[i].Mass = 1;
        P[i].PI = 0;
        P[i].IsGarbage = 0;
    }
    qsort(P, numpart, sizeof(struct particle_data), order_by_type_and_key);
    int maxnode = numpart;
    PartManager->MaxPart = numpart;
    assert_true(tb.Nodes != NULL);
    /*So we know which nodes we have initialised*/
    for(i=0; i< tb.numnodes+1; i++)
        tb.Nodes_base[i].father = -2;
    /*Time creating the nodes*/
    double start, end;
    start = MPI_Wtime();
    int nodes = force_tree_create_nodes(tb, numpart, ddecomp, BoxSize);
    assert_true(nodes < maxnode);
    end = MPI_Wtime();
    double ms = (end - start)*1000;
    printf("Number of nodes used: %d. Built tree in %.3g ms\n", nodes,ms);
    int nrealnode = check_tree(&tb, nodes, numpart);
    /* now compute the multipole moments recursively */
    start = MPI_Wtime();
    force_update_node_parallel(&tb, 0);
    end = MPI_Wtime();
    ms = (end - start)*1000;
    printf("Updated moments in %.3g ms. Total mass: %g\n", ms, tb.Nodes[numpart].mom.mass);
    assert_true(fabs(tb.Nodes[numpart].mom.mass - numpart) < 0.5);
    check_moments(&tb, numpart, nrealnode);
}

static void test_rebuild_flat(void ** state) {
    /*Set up the particle data*/
    int ncbrt = 128;
    int numpart = ncbrt*ncbrt*ncbrt;
    P = malloc(numpart*sizeof(struct particle_data));
    /* Create a regular grid of particles, 8x8x8, all of type 1,
     * in a box 8 kpc across.*/
    int i;
    #pragma omp parallel for
    for(i=0; i<numpart; i++) {
        P[i].Type = 1;
        P[i].Pos[0] = (BoxSize/ncbrt) * (i/ncbrt/ncbrt);
        P[i].Pos[1] = (BoxSize/ncbrt) * ((i/ncbrt) % ncbrt);
        P[i].Pos[2] = (BoxSize/ncbrt) * (i % ncbrt);
    }
    /*Allocate tree*/
    /*Base pointer*/
    struct forcetree_testdata * data = * (struct forcetree_testdata **) state;
    DomainDecomp ddecomp = data->ddecomp;
    ddecomp.TopLeaves[0].topnode = numpart;
    ForceTree tb = force_treeallocate(numpart, numpart, &ddecomp);
    do_tree_test(numpart, tb, &ddecomp);
    force_tree_free(&tb);
    free(P);
}

static void test_rebuild_close(void ** state) {
    /*Set up the particle data*/
    int ncbrt = 128;
    int numpart = ncbrt*ncbrt*ncbrt;
    double close = 5000;
    P = malloc(numpart*sizeof(struct particle_data));
    /* Create particles clustered in one place, all of type 1.*/
    int i;
    #pragma omp parallel for
    for(i=0; i<numpart; i++) {
        P[i].Type = 1;
        P[i].Pos[0] = 4. + (i/ncbrt/ncbrt)/close;
        P[i].Pos[1] = 4. + ((i/ncbrt) % ncbrt) /close;
        P[i].Pos[2] = 4. + (i % ncbrt)/close;
    }
    struct forcetree_testdata * data = * (struct forcetree_testdata **) state;
    DomainDecomp ddecomp = data->ddecomp;
    ddecomp.TopLeaves[0].topnode = numpart;
    ForceTree tb = force_treeallocate(numpart, numpart, &ddecomp);
    do_tree_test(numpart, tb, &ddecomp);
    force_tree_free(&tb);
    free(P);
}

void do_random_test(gsl_rng * r, const int numpart, const ForceTree tb, DomainDecomp * ddecomp)
{
    /* Create a regular grid of particles, 8x8x8, all of type 1,
     * in a box 8 kpc across.*/
    int i;
    for(i=0; i<numpart/4; i++) {
        P[i].Type = 1;
        int j;
        for(j=0; j<3; j++)
            P[i].Pos[j] = BoxSize * gsl_rng_uniform(r);
    }
    for(i=numpart/4; i<3*numpart/4; i++) {
        P[i].Type = 1;
        int j;
        for(j=0; j<3; j++)
            P[i].Pos[j] = BoxSize/2 + BoxSize/8 * exp(pow(gsl_rng_uniform(r)-0.5,2));
    }
    for(i=3*numpart/4; i<numpart; i++) {
        P[i].Type = 1;
        int j;
        for(j=0; j<3; j++)
            P[i].Pos[j] = BoxSize*0.1 + BoxSize/32 * exp(pow(gsl_rng_uniform(r)-0.5,2));
    }
    do_tree_test(numpart, tb, ddecomp);
}

static void test_rebuild_random(void ** state) {
    /*Set up the particle data*/
    int ncbrt = 64;
    struct forcetree_testdata * data = * (struct forcetree_testdata **) state;
    DomainDecomp ddecomp = data->ddecomp;
    gsl_rng * r = (gsl_rng *) data->r;
    int numpart = ncbrt*ncbrt*ncbrt;
    /*Allocate tree*/
    /*Base pointer*/
    ddecomp.TopLeaves[0].topnode = numpart;
    ForceTree tb = force_treeallocate(numpart, numpart, &ddecomp);
    assert_true(tb.Nodes != NULL);
    P = malloc(numpart*sizeof(struct particle_data));
    int i;
    for(i=0; i<2; i++) {
        do_random_test(r, numpart, tb, &ddecomp);
    }
    force_tree_free(&tb);
    free(P);
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
    ddecomp->TopLeaves[0].topnode = 0;
    /*These are not used*/
    ddecomp->TopNodes[0].StartKey = 0;
    ddecomp->TopNodes[0].Shift = BITS_PER_DIMENSION * 3;
    /*To tell the code we are in serial*/
    ddecomp->Tasks = malloc(sizeof(struct task_data));
    ddecomp->Tasks[0].StartLeaf = 0;
    ddecomp->Tasks[0].EndLeaf = 1;
}

static int setup_tree(void **state) {
    /*Set up the important parts of the All structure.*/
    /*Particles should not be outside this*/
    BoxSize = 8;
    int i;
    for(i=0; i<6; i++)
        GravitySofteningTable[i] = 0.1 / 2.8;

    init_forcetree_params(2);
    /*Set up the top-level domain grid*/
    struct forcetree_testdata *data = malloc(sizeof(struct forcetree_testdata));
    trivial_domain(&data->ddecomp);
    data->r = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(data->r, 0);
    *state = (void *) data;
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
