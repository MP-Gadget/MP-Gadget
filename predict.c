#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "allvars.h"
#include "proto.h"
void reconstruct_timebins(void)
{
    int i, n, prev, bin;
    int64_t glob_sum1, glob_sum2;

    for(bin = 0; bin < TIMEBINS; bin++)
    {
        TimeBinCount[bin] = 0;
        TimeBinCountSph[bin] = 0;
        FirstInTimeBin[bin] = -1;
        LastInTimeBin[bin] = -1;
#ifdef SFR
        TimeBinSfr[bin] = 0;
#endif
#ifdef BLACK_HOLES
        TimeBin_BH_mass[bin] = 0;
        TimeBin_BH_dynamicalmass[bin] = 0;
        TimeBin_BH_Mdot[bin] = 0;
        TimeBin_BH_Medd[bin] = 0;
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

#ifdef SFR
        if(P[i].Type == 0)
            TimeBinSfr[bin] += SPHP(i).Sfr;
#endif
#if BLACK_HOLES
        if(P[i].Type == 5)
        {
            TimeBin_BH_mass[bin] += BHP(i).Mass;
            TimeBin_BH_dynamicalmass[bin] += P[i].Mass;
            TimeBin_BH_Mdot[bin] += BHP(i).Mdot;
            TimeBin_BH_Medd[bin] += BHP(i).Mdot / BHP(i).Mass;
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

    sumup_large_ints(1, &NumForceUpdate, &glob_sum1);

    for(i = FirstActiveParticle, NumForceUpdate = 0; i >= 0; i = NextActiveParticle[i])
    {
        NumForceUpdate++;
        if(i >= NumPart)
        {
            printf("Bummer i=%d\n", i);
            endrun(12);
        }

    }

    sumup_large_ints(1, &NumForceUpdate, &glob_sum2);

    if(ThisTask == 0)
    {
        printf("sum1=%d%9d sum2=%d%9d\n",
                (int) (glob_sum1 / 1000000000), (int) (glob_sum1 % 1000000000),
                (int) (glob_sum2 / 1000000000), (int) (glob_sum2 % 1000000000));
    }

    if(glob_sum1 != glob_sum2 && All.NumCurrentTiStep > 0)
        endrun(121);
}

static void real_drift_particle(int i, int time1);
void lock_particle_if_not(int i, MyIDType id) {
    if(P[i].ID == id) return;
    pthread_spin_lock(&P[i].SpinLock);
}
void unlock_particle_if_not(int i, MyIDType id) {
    if(P[i].ID == id) return;
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
            endrun(99999);
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
        printf("i=%d time0=%d time1=%d\n", i, time0, time1);
        endrun(12);
    }

    if(time1 == time0)
        return;

    if(All.ComovingIntegrationOn)
    {
        dt_drift = get_drift_factor(time0, time1);
        dt_gravkick = get_gravkick_factor(time0, time1);
        dt_hydrokick = get_hydrokick_factor(time0, time1);
    }
    else
    {
        dt_drift = dt_gravkick = dt_hydrokick = (time1 - time0) * All.Timebase_interval;
    }

#ifdef LIGHTCONE
    double oldpos[3];
    for(j = 0; j < 3; j++) {
        oldpos[j] = P[i].Pos[j];
    }
#endif

    for(j = 0; j < 3; j++) {
        P[i].Pos[j] += P[i].Vel[j] * dt_drift;
    }
#ifdef BH_REPOSITION_ON_POTMIN
#define BHPOTVALUEINIT 1.0e30
    if(P[i].Type == 5) {
        int k;
        if(BHP(i).MinPot < 0.5 * BHPOTVALUEINIT)
            for(k = 0; k < 3; k++) {
                P[i].Pos[k] = BHP(i).MinPotPos[k];
                P[i].Vel[k] = BHP(i).MinPotVel[k];
            }
    }
#endif

#ifdef LIGHTCONE
    lightcone_cross(i, oldpos);
#endif


#ifndef HPM
    if(P[i].Type == 0)
    {
#ifdef PETAPM
        for(j = 0; j < 3; j++)
            SPHP(i).VelPred[j] +=
                (P[i].GravAccel[j] + P[i].GravPM[j]) * dt_gravkick + SPHP(i).HydroAccel[j] * dt_hydrokick;
#else
        for(j = 0; j < 3; j++)
            SPHP(i).VelPred[j] += P[i].GravAccel[j] * dt_gravkick + SPHP(i).HydroAccel[j] * dt_hydrokick;
#endif

        SPHP(i).Density *= exp(-SPHP(i).DivVel * dt_drift);
        //      P[i].Hsml *= exp(0.333333333333 * SPHP(i).DivVel * dt_drift);
        //---This was added
        double fac = exp(0.333333333333 * SPHP(i).DivVel * dt_drift);
        if(fac > 1.25)
            fac = 1.25;
        P[i].Hsml *= fac;
        if(P[i].Hsml > MAXHSML)
        {
            printf("warning: On Task=%d: we reached Hsml=%g for ID=%lu\n",
                    ThisTask, P[i].Hsml, P[i].ID);
            P[i].Hsml = MAXHSML;
        }
        //---This was added

        if(P[i].Hsml < All.MinGasHsml)
            P[i].Hsml = All.MinGasHsml;

#ifndef WAKEUP
        dt_step = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0);
#else
        dt_step = P[i].dt_step;
#endif
        dt_entr = (time1 - (P[i].Ti_begstep + dt_step / 2)) * All.Timebase_interval;

#ifndef EOS_DEGENERATE
#ifndef MHM
#ifndef SOFTEREQS

#ifndef TRADITIONAL_SPH_FORMULATION
#ifdef DENSITY_INDEPENDENT_SPH
        SPHP(i).EgyWtDensity *= exp(-SPHP(i).DivVel * dt_drift);
        SPHP(i).EntVarPred = pow(SPHP(i).Entropy + SPHP(i).DtEntropy * dt_entr, 1/GAMMA);
#endif
        SPHP(i).Pressure = (SPHP(i).Entropy + SPHP(i).DtEntropy * dt_entr) * pow(SPHP(i).EOMDensity, GAMMA);
#else
        SPHP(i).Pressure = GAMMA_MINUS1 * (SPHP(i).Entropy + SPHP(i).DtEntropy * dt_entr) * SPHP(i).d.Density;
#endif

#endif
#else
        /* Here we use an isothermal equation of state */
        SPHP(i).Pressure = All.cf.fac_egy * GAMMA_MINUS1 * SPHP(i).d.Density * All.InitGasU;
        SPHP(i).Entropy = SPHP(i).Pressure / pow(SPHP(i).d.Density, GAMMA);
#endif
#else
        /* call tabulated eos with physical units */
        eos_calc_egiven_v(SPHP(i).d.Density * All.UnitDensity_in_cgs, SPHP(i).xnuc, SPHP(i).dxnuc,
                dt_entr * All.UnitTime_in_s, SPHP(i).Entropy, SPHP(i).DtEntropy, &SPHP(i).temp,
                &SPHP(i).Pressure, &SPHP(i).dpdr);
        SPHP(i).Pressure /= All.UnitPressure_in_cgs;
#endif

#if defined(MAGNETIC) && ( !defined(EULERPOTENTIALS) || !defined(VECT_POTENTIAL))
#ifdef DIVBCLEANING_DEDNER
        SPHP(i).PhiPred += SPHP(i).DtPhi * dt_entr;

#endif
        for(j = 0; j < 3; j++)
            SPHP(i).BPred[j] += SPHP(i).DtB[j] * dt_entr;
#endif
#ifdef EULER_DISSIPATION
        SPHP(i).EulerA += SPHP(i).DtEulerA * dt_entr;
        SPHP(i).EulerB += SPHP(i).DtEulerB * dt_entr;
#endif
#ifdef VECT_POTENTIAL
        SPHP(i).APred[0] += SPHP(i).DtA[0] * dt_entr;
        SPHP(i).APred[1] += SPHP(i).DtA[1] * dt_entr;
        SPHP(i).APred[2] += SPHP(i).DtA[2] * dt_entr;
#endif

    }
#endif /* end of HPM */

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
#ifdef PERIODIC
void do_box_wrapping(void)
{
    int i;
    double boxsize[3];

    int j;
    for(j = 0; j < 3; j++)
        boxsize[j] = All.BoxSize;

#ifdef LONG_X
    boxsize[0] *= LONG_X;
#endif
#ifdef LONG_Y
    boxsize[1] *= LONG_Y;
#endif
#ifdef LONG_Z
    boxsize[2] *= LONG_Z;
#endif

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
#endif



/*

#ifdef XXLINFO
#ifdef MAGNETIC
double MeanB_part = 0, MeanB_sum;

#ifdef TRACEDIVB
double MaxDivB_part = 0, MaxDivB_all;
#endif
#endif
#ifdef TIME_DEP_ART_VISC
double MeanAlpha_part = 0, MeanAlpha_sum;
#endif
#endif



#ifdef XXLINFO
if(Flag_FullStep == 1)
{
#ifdef MAGNETIC
MeanB_part += sqrt(SPHP(i).BPred[0] * SPHP(i).BPred[0] +
SPHP(i).BPred[1] * SPHP(i).BPred[1] + SPHP(i).BPred[2] * SPHP(i).BPred[2]);
#ifdef TRACEDIVB
MaxDivB_part = DMAX(MaxDivB, fabs(SPHP(i).divB));
#endif
#endif
#ifdef TIME_DEP_ART_VISC
MeanAlpha_part += SPHP(i).alpha;
#endif
}
#endif


#ifdef XXLINFO
if(Flag_FullStep == 1)
{
#ifdef MAGNETIC
MPI_Reduce(&MeanB_part, &MeanB_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
if(ThisTask == 0)
MeanB = MeanB_sum / All.TotN_sph;
#ifdef TRACEDIVB
MPI_Reduce(&MaxDivB_part, &MaxDivB_all, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
if(ThisTask == 0)
MaxDivB = MaxDivB_all;
#endif
#endif
#ifdef TIME_DEP_ART_VISC
MPI_Reduce(&MeanAlpha_part, &MeanAlpha_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
if(ThisTask == 0)
MeanAlpha = MeanAlpha_sum / All.TotN_sph;
#endif
}
#endif
*/
