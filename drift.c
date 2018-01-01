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
#include "endrun.h"

static int drift_particle_full(int i, inttime_t ti1, int blocking);

static void real_drift_particle(int i, inttime_t ti1);
void lock_particle(int i) {
    pthread_spin_lock(&P[i].SpinLock);
}
void unlock_particle(int i) {
    pthread_spin_unlock(&P[i].SpinLock);
}
void drift_particle(int i, inttime_t ti1) {
    drift_particle_full(i, ti1, 1);
}
int drift_particle_full(int i, inttime_t ti1, int blocking) {
    if(P[i].Ti_drift == ti1) return 0 ;

#ifdef OPENMP_USE_SPINLOCK
    int lockstate;
    if (blocking) {
        lockstate = pthread_spin_lock(&P[i].SpinLock);
    } else {
        lockstate = pthread_spin_trylock(&P[i].SpinLock);
    }
    if(0 == lockstate) {
        if(P[i].Ti_drift != ti1) {
            real_drift_particle(i, ti1);
#pragma omp flush
        }
        pthread_spin_unlock(&P[i].SpinLock);
        return 0;
    } else {
        if(blocking) {
            endrun(99999, "This shall not happen. Why?");
            return -1;
        } else {
            return -1;
        }
    }

#else
    /* do not use SpinLock */
#pragma omp critical (_driftparticle_)
    {
        if(P[i].Ti_drift != ti1) {
            real_drift_particle(i, ti1);
        }
    }
    return 0;
#endif
}


/*Evolve the entropy variable, according to DtEntropy computed elsewhere.
 Input: dloga, Current values of Entropy, DtEntropy, and a factor
 which is the timestep of the particle divided by the current timestep.
 This implements an entropy floor and an entropy change floor.*/
static inline MyFloat
drift_entropy(const int i, const double dloga)
{
    const MyFloat CurEntropy = SPHP(i).Entropy;
    const MyFloat DtEntropy = SPHP(i).DtEntropy;
    /*New entropy variable*/
    MyFloat Entropy = CurEntropy + DtEntropy * dloga;

    /* Implement an entropy floor*/
    const double minentropy = All.MinEgySpec * GAMMA_MINUS1 / pow(SPHP(i).EOMDensity * All.cf.a3inv, GAMMA_MINUS1);
    if(Entropy < minentropy) {
        Entropy = minentropy;
        SPHP(i).DtEntropy = 0;
    }

    return Entropy;
}


static void real_drift_particle(int i, inttime_t ti1)
{
    int j, ti0;
    double ddrift;

    if(P[i].Ti_drift == ti1) return;


    ti0 = P[i].Ti_drift;

    if(ti1 < ti0)
    {
        endrun(12, "i=%d ti0=%d ti1=%d\n", i, ti0, ti1);
    }

    if(ti1 == ti0)
        return;

    ddrift = get_drift_factor(ti0, ti1);

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
                P[i].Vel[k] = BHP(i).MinPotVel[k];
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
        SPHP(i).Density *= exp(-SPHP(i).DivVel * ddrift);
#ifdef DENSITY_INDEPENDENT_SPH
        SPHP(i).EgyWtDensity *= exp(-SPHP(i).DivVel * ddrift);
#endif
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
        /*Evolve entropy at drift time: evolved dlog a. */
        double dloga = dloga_from_dti(ti1 - ti0);
        SPHP(i).Entropy = drift_entropy(i, dloga);
        P[i].Ti_drift = ti1;
    }

    P[i].Ti_drift = ti1;
}

void drift_active_particles(inttime_t ti1)
{
    int i;
    walltime_measure("/Misc");

#pragma omp parallel for
    for(i = 0; i < NumActiveParticle; i++)
        real_drift_particle(ActiveParticle[i], ti1);

    walltime_measure("/Drift/Active");
}

void drift_all_particles(inttime_t ti1)
{
    int i;
    walltime_measure("/Misc");

#pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)
        real_drift_particle(i, ti1);

    walltime_measure("/Drift/All");
}
