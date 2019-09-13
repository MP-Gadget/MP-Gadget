#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "allvars.h"
#include "drift.h"
#include "timefac.h"
#include "timestep.h"
#include "slotsmanager.h"
#include "partmanager.h"
#include "utils.h"

#define MAXHSML 30000.0


static void real_drift_particle(int i, inttime_t ti1, const double ddrift);

/* Updates a single particle to the current drift time*/
void drift_particle(int i, inttime_t ti1, struct SpinLocks * spin) {
    if(P[i].Ti_drift == ti1) return;

    lock_spinlock(i, spin);
    inttime_t ti0 = P[i].Ti_drift;
    if(ti0 != ti1) {
        const double ddrift = get_drift_factor(ti0, ti1);
        real_drift_particle(i, ti1, ddrift);
#pragma omp flush
    }
    unlock_spinlock(i, spin);
}

static void real_drift_particle(int i, inttime_t ti1, const double ddrift)
{
    int j;
    if(P[i].IsGarbage) {
        P[i].Ti_drift = ti1;
        return;
    }
    inttime_t ti0 = P[i].Ti_drift;
    if(ti1 < ti0) {
        endrun(12, "i=%d ti0=%d ti1=%d\n", i, ti0, ti1);
    }


#ifdef LIGHTCONE
    double oldpos[3];
    for(j = 0; j < 3; j++) {
        oldpos[j] = P[i].Pos[j];
    }
#endif

    /* Jumping of BH */
    if(P[i].Type == 5) {
        int k;
        if (BHP(i).JumpToMinPot) {
            for(k = 0; k < 3; k++) {
                double dx = NEAREST(P[i].Pos[k] - BHP(i).MinPotPos[k]);
                if(dx > 0.1 * All.BoxSize) {
                    endrun(1, "Drifting blackhole very far, from %g %g %g to %g %g %g id = %ld. Likely due to the time step is too sparse.\n",
                        P[i].Pos[0],
                        P[i].Pos[1],
                        P[i].Pos[2],
                        BHP(i).MinPotPos[0],
                        BHP(i).MinPotPos[1],
                        BHP(i).MinPotPos[2], P[i].ID);
                }
                P[i].Pos[k] = BHP(i).MinPotPos[k];
            }
        }
        BHP(i).JumpToMinPot = 0;
    }

    for(j = 0; j < 3; j++) {
        P[i].Pos[j] += P[i].Vel[j] * ddrift;
    }

#ifdef LIGHTCONE
    lightcone_cross(i, oldpos);
#endif

    for(j = 0; j < 3; j ++) {
        while(P[i].Pos[j] > All.BoxSize) P[i].Pos[j] -= All.BoxSize;
        while(P[i].Pos[j] <= 0) P[i].Pos[j] += All.BoxSize;
    }
    /* avoid recomputing them during layout and force tree build.*/
    P[i].Key = PEANO(P[i].Pos, All.BoxSize);

    if(P[i].Type == 0)
    {
        /* This accounts for adiabatic density changes,
         * and is a good predictor for most of the gas.*/
        double densdriftfac = exp(-SPHP(i).DivVel * ddrift);
        SPHP(i).Density *= densdriftfac;
        if(All.DensityIndependentSphOn)
            SPHP(i).EgyWtDensity *= densdriftfac;

        //      P[i].Hsml *= exp(0.333333333333 * SPHP(i).DivVel * ddrift);
        //---This was added
        double fac = exp(0.333333333333 * SPHP(i).DivVel * ddrift);
        if(fac > 1.25)
            fac = 1.25;
        P[i].Hsml *= fac;
        if(P[i].Hsml > MAXHSML)
        {
            message(1, "warning: we reached Hsml=%g for ID=%lu\n", P[i].Hsml, P[i].ID);
            P[i].Hsml = MAXHSML;
        }
        //---This was added

        if(P[i].Hsml < All.MinGasHsml)
            P[i].Hsml = All.MinGasHsml;
    }

    P[i].Ti_drift = ti1;
}

void drift_all_particles(inttime_t ti1)
{
    int i;
    walltime_measure("/Misc");

    const inttime_t ti0 = P[0].Ti_drift;
    const double ddrift = get_exact_drift_factor(ti0, ti1);

#pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++) {
#ifdef DEBUG
        if(P[i].Ti_drift != ti0)
            endrun(10, "Drift time mismatch: (ids = %ld %ld) %d != %d\n",P[0].ID, P[i].ID, ti0,  P[i].Ti_drift);
#endif
        real_drift_particle(i, ti1, ddrift);
    }

    walltime_measure("/Drift/All");
}
