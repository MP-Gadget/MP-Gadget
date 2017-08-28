/*Simple test for the tree building functions*/

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <stdio.h>
#include <gsl/gsl_rng.h>
#include "../forcetree.h"
#include "../allvars.h"
#include "../domain.h"
#include "../peano.c"
#include "stub.h"

/*Defined in forcetree.c*/
int
force_tree_build_single(const int firstnode, const int lastnode, const int npart);

size_t
force_treeallocate(int maxnodes, int maxpart, int first_node_offset);

/*Used data from All and domain*/
struct particle_data *P;
struct global_data_all_processes All;

int MaxTopNodes, NTopNodes, NTopLeaves, NTask, ThisTask;
struct topleaf_data *TopLeaves;
struct topnode_data *TopNodes;
struct task_data *Tasks;
size_t AllocatedBytes;
int NTask, ThisTask;
int NumPart;

/*Dummy versions of functions that implement only what we need for the tests:
 * most of these are used in the non-tested globally accessible parts of forcetree.c and
 * so not executed by our tests anyway.*/
/* this function determines the TopLeaves entry for the given key, and returns the level of the
 * node in terms of `shift`. */
int
domain_get_topleaf_with_shift(const peano_t key, int * shift) {
    int no=0;
    while(TopNodes[no].Daughter >= 0) {
        no = TopNodes[no].Daughter + ((key - TopNodes[no].StartKey) >> (TopNodes[no].Shift - 3));
    }
    *shift = TopNodes[no].Shift;
    no = TopNodes[no].Leaf;
    return no;
}

void savepositions(int n, int i){}

double walltime_measure_full(char * name, char * file, int line)
{
    return 0;
}

double get_random_number(MyIDType id)
{
    return 0;
}

/*End dummies*/

static void test_rebuild_flat(void ** state) {
    /*Set up the particle data*/
    int ncbrt = 128;
    int numpart = ncbrt*ncbrt*ncbrt;
    P = malloc(numpart*sizeof(struct particle_data));
    /* Create a regular grid of particles, 8x8x8, all of type 1,
     * in a box 8 kpc across.*/
    for(int i=0; i<numpart; i++) {
        P[i].Type = 1;
        P[i].Pos[0] = (All.BoxSize/ncbrt) * (i/ncbrt/ncbrt);
        P[i].Pos[1] = (All.BoxSize/ncbrt) * ((i/ncbrt) % ncbrt);
        P[i].Pos[2] = (All.BoxSize/ncbrt) * (i % ncbrt);
    }
    /*Allocate tree*/
    /*Base pointer*/
    TopLeaves[0].topnode = numpart;
    int maxnode = numpart;
    size_t alloc = force_treeallocate(maxnode, numpart, numpart);
    assert_true(alloc > 0);
    assert_true(Nodes);
    int nodes = force_tree_build_single(numpart, numpart + maxnode, numpart);
    printf("Number of nodes used: %d (allocated %ld)\n", nodes,alloc);
    assert_true(nodes > 0);
    free(P);
    force_tree_free();
}

static void test_rebuild_close(void ** state) {
    /*Set up the particle data*/
    int ncbrt = 128;
    int numpart = ncbrt*ncbrt*ncbrt;
    double close = 5000;
    P = malloc(numpart*sizeof(struct particle_data));
    /* Create particles clustered in one place, all of type 1.*/
    for(int i=0; i<numpart; i++) {
        P[i].Type = 1;
        P[i].Pos[0] = 4. + (i/ncbrt/ncbrt)/close;
        P[i].Pos[1] = 4. + ((i/ncbrt) % ncbrt) /close;
        P[i].Pos[2] = 4. + (i % ncbrt)/close;
    }
    /*Allocate tree*/
    /*Base pointer*/
    TopLeaves[0].topnode = numpart;
    int maxnode = numpart;
    size_t alloc = force_treeallocate(maxnode, numpart, numpart);
    assert_true(Nodes);
    int nodes = force_tree_build_single(numpart, numpart+maxnode, numpart);
    printf("Number of nodes used: %d (allocated %ld)\n", nodes,alloc);
    assert_true(nodes > 0);
    free(P);
    force_tree_free();
}

int do_random_test(gsl_rng * r, const int numpart, const int maxnode)
{
    /* Create a regular grid of particles, 8x8x8, all of type 1,
     * in a box 8 kpc across.*/
    for(int i=0; i<numpart/4; i++) {
        P[i].Type = 1;
        for(int j=0; j<3; j++)
            P[i].Pos[j] = All.BoxSize * gsl_rng_uniform(r);
    }
    for(int i=numpart/4; i<3*numpart/4; i++) {
        P[i].Type = 1;
        for(int j=0; j<3; j++)
            P[i].Pos[j] = All.BoxSize/2 + All.BoxSize/8 * exp(pow(gsl_rng_uniform(r)-0.5,2));
    }
    for(int i=3*numpart/4; i<numpart; i++) {
        P[i].Type = 1;
        for(int j=0; j<3; j++)
            P[i].Pos[j] = All.BoxSize*0.1 + All.BoxSize/32 * exp(pow(gsl_rng_uniform(r)-0.5,2));
    }
    int nodes = force_tree_build_single(numpart, numpart + maxnode, numpart);
    return nodes;
}

static void test_rebuild_random(void ** state) {
    /*Set up the particle data*/
    int ncbrt = 128;
    gsl_rng * r = (gsl_rng *) *state;
    int numpart = ncbrt*ncbrt*ncbrt;
    /*Allocate tree*/
    /*Base pointer*/
    TopLeaves[0].topnode = numpart;
    int maxnode = numpart;
    size_t alloc = force_treeallocate(maxnode, numpart, numpart);
    assert_true(alloc > 0);
    assert_true(Nodes);
    P = malloc(numpart*sizeof(struct particle_data));
    for(int i=0; i<2; i++) {
        int nodes = do_random_test(r, numpart, maxnode);
        printf("Random %d used: %d nodes\n", i, nodes);
        assert_true(nodes > 0);
    }
    free(P);
    force_tree_free();
}
static int setup_tree(void **state) {
    /*Set up the important parts of the All structure.*/
    All.NoTreeRnd = 1;
    /*Particles should not be outside this*/
    All.BoxSize = 8;
    for(int i=0; i<6; i++)
        All.ForceSoftening[i] = 0.001;
    /*Set up the top-level domain grid*/
    /* The whole tree goes into one topnode.
     * Set up just enough of the TopNode structure that
     * domain_get_topleaf_with_shift works*/
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
    AllocatedBytes = 0;
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
