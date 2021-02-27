#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "drift.h"
#include "walltime.h"
#include "timefac.h"
#include "timestep.h"
#include "slotsmanager.h"
#include "partmanager.h"
#include "utils.h"

/* Drifts an individual particle to time ti1, by a drift factor ddrift.
 * The final argument is a random shift vector applied uniformly to all particles before periodic wrapping.
 * The box is periodic, so this does not affect real physics, but it avoids correlated errors
 * in the tree expansion. The shift is relative to the current position. On most timesteps this is zero,
 * signifying no change in the coordinate frame. On PM steps a random offset is generated, and the routine
 * receives a shift vector removing the previous random shift and adding a new one.
 * This function also updates the velocity and updates the density according to an adiabatic factor.
 */
void real_drift_particle(int i, inttime_t dti, const double ddrift, const double BoxSize, const double random_shift[3])
{
    int j;
    if(P[i].IsGarbage || P[i].Swallowed) {
        /* Keep the random shift updated so the
         * physical position of swallowed particles remains unchanged.*/
        for(j = 0; j < 3; j++) {
            P[i].Pos[j] += random_shift[j];
            while(P[i].Pos[j] > BoxSize) P[i].Pos[j] -= BoxSize;
            while(P[i].Pos[j] <= 0) P[i].Pos[j] += BoxSize;
        }
        /* Swallowed particles still need a peano key.*/
        if(P[i].Swallowed)
            P[i].Key = PEANO(P[i].Pos, BoxSize);
        return;
    }

    /* Jumping of BH */
    if(P[i].Type == 5) {
        int k;
        if (BHP(i).JumpToMinPot) {
            for(k = 0; k < 3; k++) {
                double dx = NEAREST(P[i].Pos[k] - BHP(i).MinPotPos[k], BoxSize);
                if(dx > 0.1 * BoxSize) {
                    endrun(1, "Drifting blackhole very far, from %g %g %g to %g %g %g id = %ld. Likely due to the time step is too sparse.\n",
                        P[i].Pos[0],
                        P[i].Pos[1],
                        P[i].Pos[2],
                        BHP(i).MinPotPos[0],
                        BHP(i).MinPotPos[1],
                        BHP(i).MinPotPos[2], P[i].ID);
                }
                P[i].Pos[k] = BHP(i).MinPotPos[k];
                P[i].Vel[k] = BHP(i).MinPotVel[k];
            }
        }
        BHP(i).JumpToMinPot = 0;
    }

    for(j = 0; j < 3; j++) {
        P[i].Pos[j] += P[i].Vel[j] * ddrift + random_shift[j];
    }

    for(j = 0; j < 3; j ++) {
        while(P[i].Pos[j] > BoxSize) P[i].Pos[j] -= BoxSize;
        while(P[i].Pos[j] <= 0) P[i].Pos[j] += BoxSize;
    }
    /* avoid recomputing them during layout and force tree build.*/
    P[i].Key = PEANO(P[i].Pos, BoxSize);

    if(P[i].Type == 0)
    {
        /* This accounts for adiabatic density changes,
         * and is a good predictor for most of the gas.*/
        double densdriftfac = exp(-SPHP(i).DivVel * ddrift);
        SPHP(i).Density *= densdriftfac;
        /* Only does something when Pressure-entropy is enabled*/
        SPHP(i).EgyWtDensity *= densdriftfac;

        //      P[i].Hsml *= exp(0.333333333333 * SPHP(i).DivVel * ddrift);
        //---This was added
        double fac = exp(0.333333333333 * SPHP(i).DivVel * ddrift);
        inttime_t ti_step = dti_from_timebin(P[i].TimeBin);

        if(fac > 1.25)
            fac = 1.25;

        /* During deep timestep hierarchies the
         * expansion factor may get out of control,
         * so don't let it do that.*/
        if(ti_step <= 8*(dti))
            P[i].Hsml *= fac;
        /* Cap the Hsml: if DivVel is large for a particle with a long timestep
         * (most likely a wind particle) Hsml can very rarely run away*/
        const double Maxhsml = BoxSize /2.;
        if(P[i].Hsml > Maxhsml)
            P[i].Hsml = Maxhsml;
    }
}

/* Update all particles to the current time, shifting them by a random vector.*/
void drift_all_particles(inttime_t ti0, inttime_t ti1, const double BoxSize, Cosmology * CP, const double random_shift[3])
{
    int i;
    walltime_measure("/Misc");
    if(ti1 < ti0) {
        endrun(12, "Trying to reverse time: ti0=%d ti1=%d\n", ti0, ti1);
    }
    const double ddrift = get_exact_drift_factor(CP, ti0, ti1);

#pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++) {
#ifdef DEBUG
        if(P[i].Ti_drift != ti0)
            endrun(10, "Drift time mismatch: (ids = %ld %ld) %d != %d\n",P[0].ID, P[i].ID, ti0,  P[i].Ti_drift);
#endif
        real_drift_particle(i, ti1-ti0, ddrift, BoxSize, random_shift);
        P[i].Ti_drift = ti1;
    }

    walltime_measure("/Drift/All");
}
