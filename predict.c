#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "allvars.h"
#include "proto.h"
#include "endrun.h"

void reconstruct_timebins(void)
{
    int i, n, prev, bin;

    for(bin = 0; bin < TIMEBINS; bin++)
    {
        TimeBinCount[bin] = 0;
        TimeBinCountSph[bin] = 0;
        FirstInTimeBin[bin] = -1;
        LastInTimeBin[bin] = -1;
#ifdef BLACK_HOLES
        Local_BH_mass = 0;
        Local_BH_dynamicalmass = 0;
        Local_BH_Mdot = 0;
        Local_BH_Medd = 0;
#endif
    }

    for(i = 0; i < NumPart; i++)
    {
        int bin = P[i].TimeBin;

        if(TimeBinCount[bin] > 0)
        {
            PrevInTimeBin[i] = LastInTimeBin[bin];
            NextInTimeBin[i] = -1;
            NextInTimeBin[LastInTimeBin[bin]] = i;
            LastInTimeBin[bin] = i;
        }
        else
        {
            FirstInTimeBin[bin] = LastInTimeBin[bin] = i;
            PrevInTimeBin[i] = NextInTimeBin[i] = -1;
        }
        TimeBinCount[bin]++;
        if(P[i].Type == 0)
            TimeBinCountSph[bin]++;

#if BLACK_HOLES
        if(P[i].Type == 5)
        {
            Local_BH_mass += BHP(i).Mass;
            Local_BH_dynamicalmass += P[i].Mass;
            Local_BH_Mdot += BHP(i).Mdot;
            Local_BH_Medd += BHP(i).Mdot / BHP(i).Mass;
        }
#endif
    }

    FirstActiveParticle = -1;

    for(n = 0, prev = -1; n < TIMEBINS; n++)
    {
        if(TimeBinActive[n])
            for(i = FirstInTimeBin[n]; i >= 0; i = NextInTimeBin[i])
            {
                if(prev == -1)
                    FirstActiveParticle = i;

                if(prev >= 0)
                    NextActiveParticle[prev] = i;

                prev = i;
            }
    }

    if(prev >= 0)
        NextActiveParticle[prev] = -1;

}

static void real_drift_particle(int i, int time1);
void lock_particle_if_not(int i, MyIDType id) {
    if(P[i].ID == id) return;
    pthread_spin_lock(&P[i].SpinLock);
}
void lock_particle(int i) {
    pthread_spin_lock(&P[i].SpinLock);
}
void unlock_particle_if_not(int i, MyIDType id) {
    if(P[i].ID == id) return;
    pthread_spin_unlock(&P[i].SpinLock);
}
void unlock_particle(int i) {
    pthread_spin_unlock(&P[i].SpinLock);
}
void drift_particle(int i, int time1) {
    drift_particle_full(i, time1, 1);
}
int drift_particle_full(int i, int time1, int blocking) {
    if(P[i].Ti_current == time1) return 0 ;

#pragma omp atomic
    TotalParticleDrifts ++;

#ifdef OPENMP_USE_SPINLOCK
    int lockstate;
    if (blocking) {
        lockstate = pthread_spin_lock(&P[i].SpinLock);
    } else {
        lockstate = pthread_spin_trylock(&P[i].SpinLock);
    }
    if(0 == lockstate) {
        if(P[i].Ti_current != time1) {
            real_drift_particle(i, time1);
#pragma omp flush
        } else {
#pragma omp atomic
            BlockedParticleDrifts ++;
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
        if(P[i].Ti_current != time1) {
            real_drift_particle(i, time1);
        } else {
            BlockedParticleDrifts ++;
        }
    }
    return 0;
#endif
}

static void real_drift_particle(int i, int time1)
{
    int j, time0, dt_step;
    double dt_drift, dt_gravkick, dt_hydrokick, dt_entr;

    if(P[i].Ti_current == time1) return;


    time0 = P[i].Ti_current;

    if(time1 < time0)
    {
        endrun(12, "i=%d time0=%d time1=%d\n", i, time0, time1);
    }

    if(time1 == time0)
        return;

    dt_drift = get_drift_factor(time0, time1);
    dt_gravkick = get_gravkick_factor(time0, time1);
    dt_hydrokick = get_hydrokick_factor(time0, time1);

#ifdef LIGHTCONE
    double oldpos[3];
    for(j = 0; j < 3; j++) {
        oldpos[j] = P[i].Pos[j];
    }
#endif

    for(j = 0; j < 3; j++) {
        P[i].Pos[j] += P[i].Vel[j] * dt_drift;
    }
    if(P[i].Type == 5) {
        int k;

        for(k = 0; k < 3; k++) {
            double dx = NEAREST(P[i].Pos[k] - BHP(i).MinPotPos[k]);
            if(dx > 0.1 * All.BoxSize) {
                endrun(1, "Drifing blackhole from %g %g %g to %g %g %g id = %ld\n",
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

#ifdef LIGHTCONE
    lightcone_cross(i, oldpos);
#endif

    if(P[i].Type == 0)
    {
        for(j = 0; j < 3; j++)
            SPHP(i).VelPred[j] +=
                (P[i].GravAccel[j] + P[i].GravPM[j]) * dt_gravkick + SPHP(i).HydroAccel[j] * dt_hydrokick;

        SPHP(i).Density *= exp(-SPHP(i).DivVel * dt_drift);
        //      P[i].Hsml *= exp(0.333333333333 * SPHP(i).DivVel * dt_drift);
        //---This was added
        double fac = exp(0.333333333333 * SPHP(i).DivVel * dt_drift);
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

        dt_step = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0);
        dt_entr = (time1 - (P[i].Ti_begstep + dt_step / 2)) * All.Timebase_interval;

#ifdef DENSITY_INDEPENDENT_SPH
    SPHP(i).EgyWtDensity *= exp(-SPHP(i).DivVel * dt_drift);
    SPHP(i).EntVarPred = pow(SPHP(i).Entropy + SPHP(i).DtEntropy * dt_entr, 1/GAMMA);
#endif
    SPHP(i).Pressure = (SPHP(i).Entropy + SPHP(i).DtEntropy * dt_entr) * pow(SPHP(i).EOMDensity, GAMMA);

    }

    P[i].Ti_current = time1;
}



void move_particles(int time1)
{
    int i;
    walltime_measure("/Misc");

#pragma omp parallel for
    for(i = 0; i < NumPart; i++)
        real_drift_particle(i, time1);

    walltime_measure("/Drift");
}



/*! This function makes sure that all particle coordinates (Pos) are
 *  periodically mapped onto the interval [0, BoxSize].  After this function
 *  has been called, a new domain decomposition should be done, which will
 *  also force a new tree construction.
 */
void do_box_wrapping(void)
{
    int i;
    double boxsize[3];

    int j;
    for(j = 0; j < 3; j++)
        boxsize[j] = All.BoxSize;

#pragma omp parallel for
    for(i = 0; i < NumPart; i++) {
        int k;
        for(k = 0; k < 3; k++)
        {
            while(P[i].Pos[k] < 0)
                P[i].Pos[k] += boxsize[k];

            while(P[i].Pos[k] >= boxsize[k])
                P[i].Pos[k] -= boxsize[k];
        }
    }
}



