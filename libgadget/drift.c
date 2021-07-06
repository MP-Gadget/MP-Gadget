#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "drift.h"
#include "blackhole.h"
#include "walltime.h"
#include "timefac.h"
#include "timestep.h"
#include "utils.h"

/* Drifts an individual particle to time ti1, by a drift factor ddrift.
 * The final argument is a random shift vector applied uniformly to all particles before periodic wrapping.
 * The box is periodic, so this does not affect real physics, but it avoids correlated errors
 * in the tree expansion. The shift is relative to the current position. On most timesteps this is zero,
 * signifying no change in the coordinate frame. On PM steps a random offset is generated, and the routine
 * receives a shift vector removing the previous random shift and adding a new one.
 * This function also updates the velocity and updates the density according to an adiabatic factor.
 */
void real_drift_particle(struct particle_data * pp, struct slots_manager_type * sman, const double ddrift, const double BoxSize, const double random_shift[3])
{
    int j;
    if(pp->IsGarbage || pp->Swallowed) {
        /* Keep the random shift updated so the
         * physical position of swallowed particles remains unchanged.*/
        for(j = 0; j < 3; j++) {
            pp->Pos[j] += random_shift[j];
            while(pp->Pos[j] > BoxSize) pp->Pos[j] -= BoxSize;
            while(pp->Pos[j] <= 0) pp->Pos[j] += BoxSize;
        }
        return;
    }

    /* Jumping of BH */
    if(BHGetRepositionEnabled() && pp->Type == 5) {
        int k;
        int pi = pp->PI;
        struct bh_particle_data * BH = (struct bh_particle_data *) sman->info[5].ptr;
        if (BH[pi].JumpToMinPot) {
            for(k = 0; k < 3; k++) {
                double dx = NEAREST(pp->Pos[k] - BH[pi].MinPotPos[k], BoxSize);
                if(dx > 0.1 * BoxSize) {
                    endrun(1, "Drifting blackhole very far, from %g %g %g to %g %g %g id = %ld. Likely due to the time step is too sparse.\n",
                        pp->Pos[0],
                        pp->Pos[1],
                        pp->Pos[2],
                        BH[pi].MinPotPos[0],
                        BH[pi].MinPotPos[1],
                        BH[pi].MinPotPos[2], pp->ID);
                }
                pp->Pos[k] = BH[pi].MinPotPos[k];
                pp->Vel[k] = BH[pi].MinPotVel[k];
            }
        }
        BH[pi].JumpToMinPot = 0;
    }
    else if(pp->Type == 0)
    {
        /* DtHsml is 1/3 DivVel * Hsml evaluated at the last active timestep for this particle.
         * This predicts Hsml during the current timestep in the way used in Gadget-4, more accurate
         * than the Gadget-2 prediction which could run away in deep timesteps. */
        pp->Hsml += pp->DtHsml * ddrift;
        if(pp->Hsml <= 0)
            endrun(5, "Part id %ld has bad Hsml %g with DtHsml %g vel %g %g %g\n",
                   pp->ID, pp->Hsml, pp->DtHsml, pp->Vel[0], pp->Vel[1], pp->Vel[2]);
        /* Cap the Hsml just in case: if DivVel is large for a particle with a long timestep
         * at one point Hsml could rarely run away.*/
        const double Maxhsml = BoxSize /2.;
        if(pp->Hsml > Maxhsml)
            pp->Hsml = Maxhsml;
    }

    for(j = 0; j < 3; j++) {
        pp->Pos[j] += pp->Vel[j] * ddrift + random_shift[j];
    }

    for(j = 0; j < 3; j ++) {
        while(pp->Pos[j] > BoxSize) pp->Pos[j] -= BoxSize;
        while(pp->Pos[j] <= 0) pp->Pos[j] += BoxSize;
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
        if(PartManager->Base[i].Ti_drift != ti0)
            endrun(10, "Drift time mismatch: (ids = %ld %ld) %d != %d\n",PartManager->Base[0].ID, PartManager->Base[i].ID, ti0,  PartManager->Base[i].Ti_drift);
#endif
        real_drift_particle(&PartManager->Base[i], SlotsManager, ddrift, BoxSize, random_shift);
        PartManager->Base[i].Ti_drift = ti1;
    }

    walltime_measure("/Drift/All");
}
