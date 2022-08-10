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
    /*Used for adaptive gravitational softening*/
    MyFloat Soft;
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
    /* Desired accuracy of the tree force in units of the old acceleration.*/
    double ErrTolForceAcc;
    /* If > 0, use the Barnes-Hut opening angle.
     * If < 0, use the acceleration condition. */
    int TreeUseBH;
    /* Barnes-Hut opening angle to use.*/
    double BHOpeningAngle;
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
    /* Whether to calculate the short-range gravitational potential from
     * the particles in the current force tree.
     * Note that in practice when for hierarchical gravity only active particles
     * are in the tree and so this is only useful on steps where all particles are active.*/
    int CalcPotential;
    /* (Optional) pointer to the place to store accelerations, if it is not P->GravAccel*/
    MyFloat (*Accel)[3];
};

#define GRAV_GET_PRIV(tw) ((struct GravShortPriv *) ((tw)->priv))

static void
grav_short_postprocess(int i, TreeWalk * tw)
{
    double G = GRAV_GET_PRIV(tw)->G;
    MyFloat *GravAccel = NULL;
    if(GRAV_GET_PRIV(tw)->Accel)
        GravAccel = GRAV_GET_PRIV(tw)->Accel[i];
    else
        GravAccel = P[i].FullTreeGravAccel;
    GravAccel[0] *= G;
    GravAccel[1] *= G;
    GravAccel[2] *= G;
    /* calculate the potential */
    /* remove self-potential */
    if(GRAV_GET_PRIV(tw)->CalcPotential) {
        P[i].Potential += P[i].Mass / (FORCE_SOFTENING(i, P[i].Type) / 2.8);
        P[i].Potential -= 2.8372975 * pow(P[i].Mass, 2.0 / 3) * GRAV_GET_PRIV(tw)->cbrtrho0;
        P[i].Potential *= G;
    }
}

static void
grav_short_copy(int place, TreeWalkQueryGravShort * input, TreeWalk * tw)
{
    input->Soft = FORCE_SOFTENING(place, P[place].Type);
    input->OldAcc = P[place].OldAcc;
}
static void
grav_short_reduce(int place, TreeWalkResultGravShort * result, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    MyFloat * GravAccel = NULL;
    if(GRAV_GET_PRIV(tw)->Accel)
        GravAccel = GRAV_GET_PRIV(tw)->Accel[place];
    else
        GravAccel = P[place].FullTreeGravAccel;
    int k;
    for(k = 0; k < 3; k++)
        TREEWALK_REDUCE(GravAccel[k], result->Acc[k]);
    if(GRAV_GET_PRIV(tw)->CalcPotential)
        TREEWALK_REDUCE(P[place].Potential, result->Potential);
}

#endif
