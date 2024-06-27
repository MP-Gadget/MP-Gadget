#ifndef GRAVSHORT_H
#define GRAVSHORT_H

#include "partmanager.h"
#include "treewalk.h"
#include "gravity.h"

typedef struct {
    TreeWalkNgbIterBase base;
} TreeWalkNgbIterGravShort;

typedef struct
{
    TreeWalkQueryBase base;
    MyFloat OldAcc;
} TreeWalkQueryGravShort;

typedef struct {
    TreeWalkResultBase base;
    MyFloat Acc[3];
    MyFloat Potential;
} TreeWalkResultGravShort;

struct GravShortPriv {
    /* Size of a PM cell, in internal units. Box / Nmesh */
    double cellsize;
    /* How many PM cells do we go
     * before we stop calculating the tree?*/
    double Rcut;
    /* Newton's constant in internal units*/
    double G;
    inttime_t Ti_Current;
    /* Matter density in internal units.
     * rho_0 = Omega0 * rho_crit
     * rho_crit = 3 H^2 /(8 pi G).
     * This is (rho_0)^(1/3) ,
     * Note: should account for
     * massive neutrinos, but doesn't. */
    double cbrtrho0;
    /* Pointer to the place to store accelerations*/
    MyFloat (*Accel)[3];
};

#define GRAV_GET_PRIV(tw) ((struct GravShortPriv *) ((tw)->priv))

static void
grav_short_postprocess(int i, TreeWalk * tw)
{
    double G = GRAV_GET_PRIV(tw)->G;
    GRAV_GET_PRIV(tw)->Accel[i][0] *= G;
    GRAV_GET_PRIV(tw)->Accel[i][1] *= G;
    GRAV_GET_PRIV(tw)->Accel[i][2] *= G;

    if(tw->tree->full_particle_tree_flag) {
        /* On a PM step, update the stored full tree grav accel for the next PM step.
         * Needs to be done here so internal treewalk iterations don't get a partial acceleration.*/
        P[i].FullTreeGravAccel[0] = GRAV_GET_PRIV(tw)->Accel[i][0];
        P[i].FullTreeGravAccel[1] = GRAV_GET_PRIV(tw)->Accel[i][1];
        P[i].FullTreeGravAccel[2] = GRAV_GET_PRIV(tw)->Accel[i][2];
        /* calculate the potential */
        P[i].Potential += P[i].Mass / (FORCE_SOFTENING() / 2.8);
        /* remove self-potential */
        P[i].Potential -= 2.8372975 * pow(P[i].Mass, 2.0 / 3) * GRAV_GET_PRIV(tw)->cbrtrho0;
        P[i].Potential *= G;
    }
}

/*Compute the absolute magnitude of the acceleration for a particle.*/
static MyFloat
grav_get_abs_accel(struct particle_data * PP, const double G)
{
    double aold=0;
    int j;
    for(j = 0; j < 3; j++) {
       double ax = PP->FullTreeGravAccel[j] + PP->GravPM[j];
       aold += ax*ax;
    }
    return sqrt(aold) / G;
}

static void
grav_short_copy(int place, TreeWalkQueryGravShort * input, TreeWalk * tw)
{
    input->OldAcc = grav_get_abs_accel(&P[place], GRAV_GET_PRIV(tw)->G);
}

static void
grav_short_reduce(int place, TreeWalkResultGravShort * result, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    TREEWALK_REDUCE(GRAV_GET_PRIV(tw)->Accel[place][0], result->Acc[0]);
    TREEWALK_REDUCE(GRAV_GET_PRIV(tw)->Accel[place][1], result->Acc[1]);
    TREEWALK_REDUCE(GRAV_GET_PRIV(tw)->Accel[place][2], result->Acc[2]);
    if(tw->tree->full_particle_tree_flag)
        TREEWALK_REDUCE(P[place].Potential, result->Potential);
}

#endif
