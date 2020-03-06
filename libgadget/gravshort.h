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
    int Type;
    /*Used for adaptive gravitational softening*/
    MyFloat Soft;
    MyFloat OldAcc;
} TreeWalkQueryGravShort;

typedef struct {
    TreeWalkResultBase base;
    MyFloat Acc[3];
    MyFloat Potential;
    int Ninteractions;
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
    /* Which particle type should we exclude from
     * the tree calculation. */
    int FastParticleType;
    /* Are neutrinos tracers? If so, exclude them from the tree force*/
    int NeutrinoTracer;
    /* Newton's constant in internal units*/
    double G;
    /* Matter density in internal units.
     * rho_0 = Omega0 * rho_crit
     * rho_crit = 3 H^2 /(8 pi G).
     * This is (rho_0)^(1/3) ,
     * Note: should account for
     * massive neutrinos, but doesn't. */
    double cbrtrho0;
};

#define GRAV_GET_PRIV(tw) ((struct GravShortPriv *) ((tw)->priv))

static void
grav_short_postprocess(int i, TreeWalk * tw)
{
    double G = GRAV_GET_PRIV(tw)->G;
    P[i].GravAccel[0] *= G;
    P[i].GravAccel[1] *= G;
    P[i].GravAccel[2] *= G;
    /* calculate the potential */
    /* remove self-potential */
    P[i].Potential += P[i].Mass / (FORCE_SOFTENING(i) / 2.8);

    P[i].Potential -= 2.8372975 * pow(P[i].Mass, 2.0 / 3) * GRAV_GET_PRIV(tw)->cbrtrho0;

    P[i].Potential *= G;
}

static void
grav_short_copy(int place, TreeWalkQueryGravShort * input, TreeWalk * tw)
{
    input->Type = P[place].Type;
    input->Soft = FORCE_SOFTENING(place);
    /*Compute old acceleration before we over-write things*/
    double aold=0;
    int i;
    for(i = 0; i < 3; i++) {
       double ax = P[place].GravAccel[i] + P[place].GravPM[i];
       aold += ax*ax;
    }

    input->OldAcc = sqrt(aold)/GRAV_GET_PRIV(tw)->G;

}
static void
grav_short_reduce(int place, TreeWalkResultGravShort * result, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    int k;
    for(k = 0; k < 3; k++)
        TREEWALK_REDUCE(P[place].GravAccel[k], result->Acc[k]);

    TREEWALK_REDUCE(P[place].Potential, result->Potential);
}

#endif
