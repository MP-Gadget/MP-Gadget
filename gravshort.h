#ifndef GRAVSHORT_H
#define GRAVSHORT_H

#include "partmanager.h"
#include "treewalk.h"

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

static int
grav_short_haswork(int i, TreeWalk * tw)
{
    return 1; /* gravity applies to all particles. Including Tracer particles to enhance numerical stability. */
}

static void
grav_short_postprocess(int i, TreeWalk * tw)
{
    P[i].GravAccel[0] *= All.G;
    P[i].GravAccel[1] *= All.G;
    P[i].GravAccel[2] *= All.G;
    /* calculate the potential */
    /* remove self-potential */
    P[i].Potential += P[i].Mass / (FORCE_SOFTENING(i) / 2.8);

    P[i].Potential -= 2.8372975 * pow(P[i].Mass, 2.0 / 3) *
        pow(All.CP.Omega0 * 3 * All.CP.Hubble * All.CP.Hubble / (8 * M_PI * All.G), 1.0 / 3);

    P[i].Potential *= All.G;
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

    input->OldAcc = sqrt(aold)/All.G;

}
static void
grav_short_reduce(int place, TreeWalkResultGravShort * result, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    int k;
    for(k = 0; k < 3; k++)
        TREEWALK_REDUCE(P[place].GravAccel[k], result->Acc[k]);

    TREEWALK_REDUCE(P[place].GravCost, result->Ninteractions);
    TREEWALK_REDUCE(P[place].Potential, result->Potential);
}

int grav_apply_short_range_window(double r, double * fac, double * facpot);

#endif
