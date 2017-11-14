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

#include "exchange.h"
#include "allvars.h"
/*Note this includes the garbage collection!
 * Should be tested separately.*/
#include "garbage.c"
#include "stub.h"
#include "mymalloc.h"

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

static void do_exchange_test()
{
    int i;
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
    for(i=0; i<All.MaxPart; i++) {
        P[i].Type = 1;
        P[i].ID = ThisTask*All.MaxPart + i;
        P[i].Pos[0] = (All.BoxSize/ncbrt/NTask) * (ThisTask + (i/ncbrt/ncbrt));
        P[i].Pos[1] = (All.BoxSize/ncbrt) * ((i/ncbrt) % ncbrt);
        P[i].Pos[2] = (All.BoxSize/ncbrt) * (i % ncbrt);
    }
    do_exchange_test();
    for(i=0; i<NumPart; i++) {
        /*Check first N/NTask elements are from this processor*/
        if(i < NumPart/NTask)
            assert_int_equal(P[i].ID/All.MaxPart, ThisTask);
        else
            assert_int_not_equal(P[i].ID/All.MaxPart, ThisTask);
    }
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
    N_star_slots = 0;
    N_bh_slots = 0;
    /*This is different from MaxPart to make sure we test*/
    All.MaxPartSph = All.MaxPart/2;
    All.MaxPartBh = 0;
    All.MaxPartStar = 0;
    P = calloc(All.MaxPart, sizeof(struct particle_data));
    SphP = calloc(All.MaxPart/2,sizeof(struct sph_particle_data));
    /* Create a regular grid of particles, 8x8x8, all of type 1,
     * in a box 8 kpc across.*/
    int i;
    for(i=0; i<NumPart/2; i++) {
        P[i].Type = 1;
        P[i].ID = ThisTask*NumPart + i;
        P[i].Pos[0] = (All.BoxSize/ncbrt/NTask) * (ThisTask + (i/ncbrt/ncbrt));
        P[i].Pos[1] = (All.BoxSize/ncbrt) * ((i/ncbrt) % ncbrt);
        P[i].Pos[2] = (All.BoxSize/ncbrt) * (i % ncbrt);
    }
    /* This means that the SPH particles will all end up on different processors.*/
    for(i=NumPart/2; i<NumPart; i++) {
        P[i].Type = 0;
        P[i].ID = ThisTask*NumPart + i;
        P[i].Pos[0] = (All.BoxSize/ncbrt/NTask) * (ThisTask + (i/ncbrt/ncbrt));
        P[i].Pos[1] = (All.BoxSize/ncbrt) * ((i/ncbrt) % ncbrt);
        P[i].Pos[2] = (All.BoxSize/ncbrt) * (i % ncbrt);
        P[i].PI = i - All.MaxPart/2;
        SPHP(i).base.ID = P[i].ID;
    }

    do_exchange_test();
    for(i=0; i<NumPart; i++) {
        assert_true(P[i].ID % All.MaxPart < All.MaxPart);
        if(P[i].Type != 1)
            assert_int_equal(P[i].ID , SPHP(i).base.ID);
        /*Check first N/NTask elements are from this processor*/
        if(i < NumPart/NTask)
            assert_int_equal(P[i].ID/All.MaxPart, ThisTask);
        else
            assert_int_not_equal(P[i].ID/All.MaxPart, ThisTask);
    }
    assert_int_equal(NumPart, All.MaxPart);
    assert_int_equal(N_sph_slots, All.MaxPart/2);

    free(P);
    free(SphP);
}

static void exc_garbage(void ** state) {
    /*Set up the particle data*/
    All.BoxSize = 8;
    int ncbrt = 64;
    All.MaxPart = ncbrt*ncbrt*ncbrt+NTask*2;
    NumPart = ncbrt*ncbrt*ncbrt;
    N_sph_slots = All.MaxPart;
    N_star_slots = 0;
    N_bh_slots = 0;
    /*This is different from MaxPart to make sure we test*/
    All.MaxPartSph = All.MaxPart;
    All.MaxPartBh = 0;
    All.MaxPartStar = 0;
    P = calloc(All.MaxPart, sizeof(struct particle_data));
    SphP = calloc(All.MaxPart,sizeof(struct sph_particle_data));
    /* Create a regular grid of particles, 8x8x8, all of type 1,
     * in a box 8 kpc across.*/
    int i;
    /* This means that the SPH particles will all end up on different processors.
     Not parallel so we know reliably which one is last.*/
    for(i=0; i<NumPart; i++) {
        P[i].Type = 0;
        P[i].ID = ThisTask*NumPart + i;
        P[i].Pos[0] = (All.BoxSize/ncbrt/NTask) * (ThisTask + (i/ncbrt/ncbrt));
        P[i].Pos[1] = (All.BoxSize/ncbrt) * ((i/ncbrt) % ncbrt);
        P[i].Pos[2] = (All.BoxSize/ncbrt) * (i % ncbrt);
        P[i].PI = i;
        SPHP(i).base.ID = P[i].ID;
    }
    //Mark a particle as garbage*/
    P[NumPart-1].IsGarbage = 1;
    P[NumPart-2].IsGarbage = 1;

    do_exchange_test();
    for(i=0; i<NumPart; i++) {
        assert_false(P[i].IsGarbage);
        /*Make sure the garbage particle was really removed*/
        assert_true(P[i].ID % (ncbrt*ncbrt*ncbrt) < (ncbrt*ncbrt*ncbrt)-2);
        assert_int_equal(P[i].ID , SPHP(i).base.ID);
    }
    free(P);
    free(SphP);
}

static void allslot_test(int realloc) {
    /*Set up the particle data*/
    All.BoxSize = 8;
    int ncbrt = 32;
    All.MaxPart = NTask*ncbrt*ncbrt*ncbrt;
    NumPart = ncbrt*ncbrt*ncbrt;
    N_sph_slots = 0.6*NumPart+20;
    N_star_slots = 0.3*NumPart+15;
    N_bh_slots = NumPart - N_sph_slots - N_star_slots;
    /*This is different from MaxPart to make sure we test*/
    if(realloc) {
        All.MaxPartSph = N_sph_slots;
        All.MaxPartBh = N_bh_slots;
        All.MaxPartStar = N_star_slots;
    }
    else {
        All.MaxPartSph = NumPart;
        All.MaxPartBh = NumPart;
        All.MaxPartStar = NumPart;
    }
    P = calloc(All.MaxPart, sizeof(struct particle_data));
    size_t bytes = All.MaxPartSph * sizeof(struct sph_particle_data) +
        All.MaxPartStar * sizeof(struct star_particle_data) + All.MaxPartBh * sizeof(struct bh_particle_data);
    void * secondary_data = mymalloc("SecondaryP", bytes);
    /*Ordering: SPH, black holes, then stars*/
    SphP = (struct sph_particle_data *) secondary_data;
    BhP = (struct bh_particle_data *) (SphP + All.MaxPartSph);
    StarP = (struct star_particle_data *) (BhP + All.MaxPartBh);
    memset(SphP, 0, All.MaxPartSph*sizeof(struct sph_particle_data));
    memset(StarP, 0, All.MaxPartStar*sizeof(struct star_particle_data));
    memset(BhP, 0, All.MaxPartBh*sizeof(struct bh_particle_data));
    /* Create a regular grid of particles, 8x8x8, all of type 1,
     * in a box 8 kpc across.*/
    int i;
    int last_sph = 0, last_bh = 0, last_star = 0;
    for(i=0; i<NumPart; i++) {
        P[i].ID = ThisTask*NumPart + i;
        P[i].Pos[0] = (All.BoxSize/ncbrt/NTask) * (ThisTask + (i/ncbrt/ncbrt));
        P[i].Pos[1] = (All.BoxSize/ncbrt) * ((i/ncbrt) % ncbrt);
        if(realloc)
            P[i].Pos[2] = 0.5*All.BoxSize + (All.BoxSize/NTask) * (((double) (i % ncbrt))/ncbrt);
        else
            P[i].Pos[2] = All.BoxSize * (((double) (i % ncbrt))/ncbrt);
        if(i < N_sph_slots) {
            P[i].Type = 0;
            P[i].PI = last_sph++;
            SPHP(i).base.ID = P[i].ID;
        } else if(i >= N_sph_slots && i < N_sph_slots + N_star_slots) {
            P[i].Type = 4;
            P[i].PI = last_star++;
            STARP(i).base.ID = P[i].ID;
        } else {
            P[i].Type = 5;
            P[i].PI = last_bh++;
            BHP(i).base.ID = P[i].ID;
        }
    }
    do_exchange_test();
    for(i=0; i<NumPart; i++) {
        assert_true(P[i].ID % (ncbrt*ncbrt*ncbrt) < (ncbrt*ncbrt*ncbrt));
        if(P[i].Type == 0) {
            assert_int_equal(P[i].ID , SPHP(i).base.ID);
        }
        if(P[i].Type == 4) {
            assert_int_equal(P[i].ID , STARP(i).base.ID);
        }
        if(P[i].Type == 5) {
            assert_int_equal(P[i].ID , BHP(i).base.ID);
        }
    }
    free(P);
    myfree(secondary_data);
}

static void exc_allslot(void ** state) {
    allslot_test(0);
}

static void exc_realloc(void ** state) {
    allslot_test(1);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(exc_onlydm),
        cmocka_unit_test(exc_sph),
        cmocka_unit_test(exc_garbage),
        cmocka_unit_test(exc_allslot),
        cmocka_unit_test(exc_realloc)
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
