/*Simple test for the exchange function*/

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <gsl/gsl_rng.h>

#define qsort_openmp qsort

#include "exchange.h"
#include "allvars.h"
/*Note this includes the garbage collection!
 * Should be tested separately.*/
#include "garbage.c"
#include "stub.h"

/*Used data from All and domain*/
struct particle_data *P;
struct sph_particle_data *SphP;
struct star_particle_data *StarP;
struct bh_particle_data *BhP;
/*This can be removed when the slot data is moved to slot-manager*/
struct global_data_all_processes All;
int NumPart;
int N_sph_slots, N_star_slots, N_bh_slots;

/*Dummies*/
double walltime_measure_full(char * name, char * file, int line) {
    return MPI_Wtime();
}
int force_tree_allocated() {
    return 0;
}

/*Dummy: used only in domain_fork_particle, which is not tested here.*/

int *ActiveParticle;
int NumActiveParticle;
int * Nextnode;
int * Father;
int is_timebin_active(int i, inttime_t current) {
    return 0;
}


/*Simple layout function: this needs to return
 *an exactly even division of the particles.*/
int layoutfunc(int p)
{
    return (NTask * P[p].Pos[2])/ All.BoxSize;
}

static void do_exchange_test(int numpart, int garbage)
{
    int i;
    All.MaxPart = numpart;
    /*Time to do exchange*/
    double start, end;
    start = MPI_Wtime();
    int fail = domain_exchange(&layoutfunc, 1);
    assert_false(fail);
    end = MPI_Wtime();
    double ms = (end - start)*1000;
    message(0,"Exchange in %.3g ms\n", ms);
    for(i=0; i<NumPart; i++) {
//         message(1, "i = %d, pos = %g %g %g\n",i, P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
        assert_int_equal(layoutfunc(i), ThisTask);
        /*Check first N/NTask elements are from this processor*/
        if(i < numpart/NTask - garbage)
            assert_int_equal(P[i].ID/All.MaxPart, ThisTask);
        else
            assert_int_not_equal(P[i].ID/All.MaxPart, ThisTask);
        /*Check there was no corruption*/
        assert_true(P[i].Pos[0] >= 0 && P[i].Pos[0] < All.BoxSize);
        assert_true(P[i].Pos[1] >= 0 && P[i].Pos[1] < All.BoxSize);
        assert_true(P[i].Pos[2] >= 0 && P[i].Pos[2] < All.BoxSize);
    }
}

static void exc_onlydm(void ** state) {
    /*Set up the particle data*/
    All.BoxSize = 8;
    int ncbrt = 96;
    All.MaxPart = ncbrt*ncbrt*ncbrt;
    NumPart = ncbrt*ncbrt*ncbrt;
    All.MaxPartSph = 0;
    All.MaxPartBh = 0;
    All.MaxPartStar = 0;
    P = calloc(All.MaxPart, sizeof(struct particle_data));
    /* Create a regular grid of particles, 8x8x8, all of type 1,
     * in a box 8 kpc across.*/
    int i;
    #pragma omp parallel for
    for(i=0; i<All.MaxPart; i++) {
        P[i].Type = 1;
        P[i].ID = ThisTask*All.MaxPart + i;
        P[i].Pos[0] = (All.BoxSize/ncbrt/NTask) * (ThisTask + (i/ncbrt/ncbrt));
        P[i].Pos[1] = (All.BoxSize/ncbrt) * ((i/ncbrt) % ncbrt);
        P[i].Pos[2] = (All.BoxSize/ncbrt) * (i % ncbrt);
    }
    do_exchange_test(All.MaxPart, 0);
    assert_int_equal(NumPart, All.MaxPart);
    free(P);
}

static void exc_sph(void ** state) {
    /*Set up the particle data*/
    All.BoxSize = 8;
    int ncbrt = 96;
    All.MaxPart = ncbrt*ncbrt*ncbrt;
    NumPart = ncbrt*ncbrt*ncbrt;
    N_sph_slots = All.MaxPart/2;
    /*This is different from MaxPart to make sure we test*/
    All.MaxPartSph = All.MaxPart/2;
    All.MaxPartBh = 0;
    All.MaxPartStar = 0;
    P = calloc(All.MaxPart, sizeof(struct particle_data));
    SphP = calloc(All.MaxPart/2,sizeof(struct sph_particle_data));
    /* Create a regular grid of particles, 8x8x8, all of type 1,
     * in a box 8 kpc across.*/
    int i;
    #pragma omp parallel for
    for(i=0; i<All.MaxPart/2; i++) {
        P[i].Type = 1;
        P[i].ID = ThisTask*All.MaxPart + i;
        P[i].Pos[0] = (All.BoxSize/ncbrt/NTask) * (ThisTask + (i/ncbrt/ncbrt));
        P[i].Pos[1] = (All.BoxSize/ncbrt) * ((i/ncbrt) % ncbrt);
        P[i].Pos[2] = (All.BoxSize/ncbrt) * (i % ncbrt);
    }
    /* This means that the SPH particles will all end up on different processors.*/
    #pragma omp parallel for
    for(i=All.MaxPart/2; i<All.MaxPart; i++) {
        P[i].Type = 0;
        P[i].ID = ThisTask*All.MaxPart + i;
        P[i].Pos[0] = (All.BoxSize/ncbrt/NTask) * (ThisTask + (i/ncbrt/ncbrt));
        P[i].Pos[1] = (All.BoxSize/ncbrt) * ((i/ncbrt) % ncbrt);
        P[i].Pos[2] = (All.BoxSize/ncbrt) * (i % ncbrt);
        P[i].PI = i - All.MaxPart/2;
        SPHP(i).base.ID = P[i].ID;
    }

    do_exchange_test(All.MaxPart,0);
    for(i=0; i<NumPart; i++) {
        assert_true(P[i].ID % All.MaxPart < All.MaxPart);
        if(P[i].Type != 1)
            assert_int_equal(P[i].ID , SPHP(i).base.ID);
    }
    assert_int_equal(NumPart, All.MaxPart);
    assert_int_equal(N_sph_slots, All.MaxPart/2);

    free(P);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(exc_onlydm),
        cmocka_unit_test(exc_sph),
//         cmocka_unit_test(exchange_all),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
