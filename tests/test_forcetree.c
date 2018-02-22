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

#include "stub.h"

#include <libgadget/allvars.h>
#include <libgadget/forcetree.h>
#include <libgadget/partmanager.h>
#include <libgadget/domain.h>

/*Defined in forcetree.c*/
int
force_tree_create_nodes(const struct TreeBuilder tb, const int npart);

struct TreeBuilder
force_treeallocate(int maxnodes, int maxpart, int first_node_offset);

int
force_update_node_parallel(const struct TreeBuilder tb);

/*Used data from All and domain*/
struct part_manager_type PartManager[1] = {{0}};
struct global_data_all_processes All;

int MaxTopNodes, NTopNodes, NTopLeaves, NTask, ThisTask;
struct topleaf_data *TopLeaves;
struct topnode_data *TopNodes;
struct task_data *Tasks;
int NTask, ThisTask;
double GravitySofteningTable[6];

/*Dummy versions of functions that implement only what we need for the tests:
 * most of these are used in the non-tested globally accessible parts of forcetree.c and
 * so not executed by our tests anyway.*/

/*This function determines the TopLeaves entry for the given key.*/
inline int
domain_get_topleaf(const peano_t key) {
    int no=0;
    while(TopNodes[no].Daughter >= 0)
        no = TopNodes[no].Daughter + ((key - TopNodes[no].StartKey) >> (TopNodes[no].Shift - 3));
    no = TopNodes[no].Leaf;
    return no;
}

void dump_snapshot() { }

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

int force_get_father(int no, int firstnode)
{
    if(no >= firstnode)
        return Nodes[no].father;
    else
        return Father[no];
}

/*This checks that the moments of the force tree in Nodes are valid:
 * that it the mass and flags are correct.*/
static int check_moments(const struct TreeBuilder tb, const int numpart, const int nrealnode)
{
    double * oldmass = malloc(sizeof(double) * MaxNodes);
    int i;

    for(i=tb.firstnode; i < tb.lastnode; i ++) {
        oldmass[i - tb.firstnode] = Nodes[i].u.d.mass;
    }

    for(i=0; i<numpart; i++)
    {
        int fnode = Father[i];
        /*Subtract mass so that nothing is left.*/
        assert_true(fnode >= tb.firstnode && fnode < tb.lastnode);
        while(fnode > 0) {
            Nodes[fnode].u.d.mass -= P[i].Mass;
            fnode = Nodes[fnode].father;
            /*Validate father*/
            assert_true((fnode >= tb.firstnode && fnode < tb.lastnode) || fnode == -1);
        }
    }
    int node = tb.firstnode;
    int counter = 0;
    int sibcntr = 0;
    while(node >= 0) {
        assert_true(node >= -1 && node < tb.lastnode);
        int next = force_get_next_node(node,tb);
        /*If a real node*/
        if(node >= tb.firstnode) {
            /*Check sibling*/
            assert_true(Nodes[node].u.d.sibling >= -1 && Nodes[node].u.d.sibling < tb.lastnode);
            int sib = Nodes[node].u.d.sibling;
            int sfather = force_get_father(sib, tb.firstnode);
            int father = force_get_father(node, tb.firstnode);
            /* Our sibling should either be a true sibling, with the same father,
             * or should be the child of one of our ancestors*/
            if(sfather != father && sib != -1) {
                int ances = father;
                while(ances >= 0) {
                    assert_true(ances >= tb.firstnode);
                    ances = force_get_father(ances, tb.firstnode);
                    if(ances == sfather)
                        break;
                }
                assert_int_equal(ances, sfather);
/*                 printf("node %d ances %d sib %d next %d father %d sfather %d\n",node, ances, sib, force_get_next_node(node, tb), father, sfather); */
            }
            else if(sib == -1)
                sibcntr++;

            if(!(Nodes[node].u.d.mass < 0.5 && Nodes[node].u.d.mass > -0.5)) {
                printf("node %d (%d) mass %g / %g TL %d DLM %d MS %g MSN %d ITL %d\n", 
                    node, node - tb.firstnode, Nodes[node].u.d.mass, oldmass[node - tb.firstnode],
                    Nodes[node].f.TopLevel,
                    Nodes[node].f.DependsOnLocalMass,
                    Nodes[node].u.d.MaxSoftening,
                    Nodes[node].f.MixedSofteningsInNode,
                    Nodes[node].f.InternalTopLevel
                    );
                int nn = force_get_next_node(node, tb);
                while(nn < tb.firstnode) { /* something is wrong show the particles */
                    printf("particles P[%d], Mass=%g\n", nn, P[nn].Mass);
                    nn = force_get_next_node(nn, tb);
                }
            }
            assert_true(Nodes[node].u.d.mass < 0.5 && Nodes[node].u.d.mass > -0.5);
            /*Check center of mass moments*/
            for(i=0; i<3; i++)
                assert_true(Nodes[node].u.d.s[i] <= All.BoxSize && Nodes[node].u.d.s[i] >= 0);
            counter++;
        }
        node = next;
    }
    assert_int_equal(counter, nrealnode);
    assert(sibcntr < counter/100);

    free(oldmass);
    return nrealnode;
}

/*This checks that the force tree in Nodes is valid:
 * that it contains every particle and that each parent
 * node contains particles within the right subnode.*/
static int check_tree(const struct TreeBuilder tb, const int nnodes, const int numpart)
{
    const int firstnode = tb.firstnode;
    int tot_empty = 0, nrealnode = 0, sevens = 0;
    int i;
    for(i=firstnode; i<nnodes+firstnode; i++)
    {
        struct NODE * pNode = &Nodes[i];
        int empty = 0;
        /*Just reserved free space with nothing in it*/
        if(pNode->father < -1.5)
            continue;

        int j;
        for(j=0; j<8; j++) {
            /*Check children*/
            int child = pNode->u.suns[j];
            if(child == -1) {
                empty++;
                continue;
            }
            assert_true(child < firstnode+nnodes);
            assert_true(child >= 0);
            /*If an internal node*/
            if(child > firstnode) {
                assert_true(fabs(Nodes[child].len/pNode->len - 0.5) < 1e-4);
                int k;
                for(k=0; k<3; k++) {
                    if(j & (1<<k))
                        assert_true(Nodes[child].center[k] > pNode->center[k]);
                    else
                        assert_true(Nodes[child].center[k] <= pNode->center[k]);
                }
            }
            /*Particle*/
            else {
                /* if the first particle suffers, then all particles on the list
                 * must be suffering from particle-coupling */
                do {
                    P[child].PI += 1;
                    if(Nextnode[child] > -1) {
                        assert_int_equal(Father[child], Father[Nextnode[child]]);
                    }
                    /*Check in right quadrant*/
                    int k;
                    for(k=0; k<3; k++) {
                        if(j & (1<<k)) {
                            assert_true(P[child].Pos[k] > pNode->center[k]);
                        }
                        else
                            assert_true(P[child].Pos[k] <= pNode->center[k]);
                    }
                    child = force_get_next_node(child, tb);
                } while(child > -1);
            }
        }
        /*All nodes should have at least one thing in them:
         * maybe particles or other nodes.*/
        if(empty > 6)
            sevens++;
        assert_true(empty <= 7);
        tot_empty += empty;
        nrealnode++;
    }
    assert_true(nnodes - nrealnode < omp_get_max_threads()*NODECACHE_SIZE);

    for(i=0; i<numpart; i++)
    {
        assert_int_equal(P[i].PI, 1);
    }
    printf("Tree filling factor: %g on %d nodes (wasted: %d seven empty: %d)\n", tot_empty/(8.*nrealnode), nrealnode, nnodes - nrealnode, sevens);
    return nrealnode;
}

static void do_tree_test(const int numpart, const struct TreeBuilder tb)
{
    /*Sort by peano key so this is more realistic*/
    int i;
    #pragma omp parallel for
    for(i=0; i<numpart; i++) {
        P[i].Key = PEANO(P[i].Pos, All.BoxSize);
        P[i].Mass = 1;
    }
    qsort(P, numpart, sizeof(struct particle_data), order_by_type_and_key);
    int maxnode = numpart;
    PartManager->MaxPart = numpart;
    MaxNodes = numpart;
    assert_true(Nodes != NULL);
    /*So we know which nodes we have initialised*/
    for(i=0; i< MaxNodes+1; i++)
        Nodes_base[i].father = -2;
    /*Time creating the nodes*/
    double start, end;
    start = MPI_Wtime();
    int nodes = force_tree_create_nodes(tb, numpart);
    assert_true(nodes < maxnode);
    end = MPI_Wtime();
    double ms = (end - start)*1000;
    printf("Number of nodes used: %d. Built tree in %.3g ms\n", nodes,ms);
    int nrealnode = check_tree(tb, nodes, numpart);
    /* now compute the multipole moments recursively */
    start = MPI_Wtime();
    int tail = force_update_node_parallel(tb);
    force_set_next_node(tail, -1, tb);
/*     assert_true(tail < nodes); */
    end = MPI_Wtime();
    ms = (end - start)*1000;
    printf("Updated moments in %.3g ms. Total mass: %g\n", ms, Nodes[numpart].u.d.mass);
    assert_true(fabs(Nodes[numpart].u.d.mass - numpart) < 0.5);
    check_moments(tb, numpart, nrealnode);
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
        P[i].Pos[0] = (All.BoxSize/ncbrt) * (i/ncbrt/ncbrt);
        P[i].Pos[1] = (All.BoxSize/ncbrt) * ((i/ncbrt) % ncbrt);
        P[i].Pos[2] = (All.BoxSize/ncbrt) * (i % ncbrt);
    }
    /*Allocate tree*/
    /*Base pointer*/
    TopLeaves[0].topnode = numpart;
    struct TreeBuilder tb = force_treeallocate(numpart, numpart, numpart);
    do_tree_test(numpart, tb);
    force_tree_free();
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
    struct TreeBuilder tb = force_treeallocate(numpart, numpart, numpart);
    do_tree_test(numpart, tb);
    force_tree_free();
    free(P);
}

void do_random_test(gsl_rng * r, const int numpart, const int maxnode, const struct TreeBuilder tb)
{
    /* Create a regular grid of particles, 8x8x8, all of type 1,
     * in a box 8 kpc across.*/
    int i;
    for(i=0; i<numpart/4; i++) {
        P[i].Type = 1;
        P[i].PI = 0;
        int j;
        for(j=0; j<3; j++)
            P[i].Pos[j] = All.BoxSize * gsl_rng_uniform(r);
    }
    for(i=numpart/4; i<3*numpart/4; i++) {
        P[i].Type = 1;
        P[i].PI = 0;
        int j;
        for(j=0; j<3; j++)
            P[i].Pos[j] = All.BoxSize/2 + All.BoxSize/8 * exp(pow(gsl_rng_uniform(r)-0.5,2));
    }
    for(i=3*numpart/4; i<numpart; i++) {
        P[i].Type = 1;
        P[i].PI = 0;
        int j;
        for(j=0; j<3; j++)
            P[i].Pos[j] = All.BoxSize*0.1 + All.BoxSize/32 * exp(pow(gsl_rng_uniform(r)-0.5,2));
    }
    do_tree_test(numpart, tb);
}

static void test_rebuild_random(void ** state) {
    /*Set up the particle data*/
    int ncbrt = 64;
    gsl_rng * r = (gsl_rng *) *state;
    int numpart = ncbrt*ncbrt*ncbrt;
    /*Allocate tree*/
    /*Base pointer*/
    TopLeaves[0].topnode = numpart;
    int maxnode = numpart;
    struct TreeBuilder tb = force_treeallocate(numpart, numpart, numpart);
    assert_true(Nodes != NULL);
    P = malloc(numpart*sizeof(struct particle_data));
    int i;
    for(i=0; i<2; i++) {
        do_random_test(r, numpart, maxnode, tb);
    }
    force_tree_free();
    free(P);
}

static int setup_tree(void **state) {
    /*Set up the important parts of the All structure.*/
    /*Particles should not be outside this*/
    All.BoxSize = 8;
    All.NumThreads = omp_get_max_threads();
    int i;
    for(i=0; i<6; i++)
        GravitySofteningTable[i] = 0.1 / 2.8;
    /*Set up the top-level domain grid*/
    /* The whole tree goes into one topnode.
     * Set up just enough of the TopNode structure that
     * domain_get_topleaf works*/
    MaxTopNodes = 1;
    NTopNodes = NTopLeaves = 1;
    TopNodes = malloc(sizeof(struct topnode_data));
    TopNodes[0].Daughter = -1;
    TopNodes[0].Leaf = 0;
    TopLeaves = malloc(sizeof(struct topleaf_data));
    TopLeaves[0].Task = 0;
    TopLeaves[0].topnode = 0;
    /*These are not used*/
    TopNodes[0].StartKey = 0;
    TopNodes[0].Shift = BITS_PER_DIMENSION * 3;
    /*To tell the code we are in serial*/
    ThisTask = 0;
    NTask = 1;
    Tasks = malloc(sizeof(struct task_data));
    Tasks[0].StartLeaf = 0;
    Tasks[0].EndLeaf = 1;
    gsl_rng * r = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(r, 0);
    *state = (void *) r;
    return 0;
}

static int teardown_tree(void **state) {
    free(TopNodes);
    free(TopLeaves);
    free(Tasks);
    free(*state);
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
