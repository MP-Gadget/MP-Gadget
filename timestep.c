#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "allvars.h"
#include "proto.h"
#ifdef COSMIC_RAYS
#include "cosmic_rays.h"
#endif

/*! \file timestep.c 
 *  \brief routines for 'kicking' particles in
 *  momentum space and assigning new timesteps
 */

static double fac1, fac2, fac3, hubble_a, atime, a3inv;
static double dt_displacement = 0;

#ifdef PMGRID
static double dt_gravkickA, dt_gravkickB;
#endif

/*! This function advances the system in momentum space, i.e. it does apply the 'kick' operation after the
 *  forces have been computed. Additionally, it assigns new timesteps to particles. At start-up, a
 *  half-timestep is carried out, as well as at the end of the simulation. In between, the half-step kick that
 *  ends the previous timestep and the half-step kick for the new timestep are combined into one operation.
 */
void advance_and_find_timesteps(void)
{
    int i, ti_step, ti_step_old, ti_min, tend, tstart, bin, binold, prev, next;
    double aphys;
    int badstepsizecount = 0;
    int badstepsizecount_global = 0;

#ifdef PMGRID
    int j, dt_step;
    double dt_gravkick, dt_hydrokick;

#ifdef DISTORTIONTENSORPS
    /* for the distortion 'velocity part', so only the lower two 3x3 submatrices will be != 0 */
    MyDouble dv_distortion_tensorps[6][6];
    int j1, j2;
#endif
#endif
#ifdef MAKEGLASS
    double disp, dispmax, globmax, dmean, fac, disp2sum, globdisp2sum;
#endif
#ifdef WAKEUP
    int n, k, dt_bin, ti_next_for_bin, ti_next_kick, ti_next_kick_global, max_time_bin_active;
#ifndef PMGRID
    int dt_step;
#endif  
    int time0, time1_old, time1_new;
    double dt_entr;

    long long ntot;
#endif

    CPU_Step[CPU_MISC] += measure_time();

    if(All.ComovingIntegrationOn)
    {
        fac1 = 1 / (All.Time * All.Time);
        fac2 = 1 / pow(All.Time, 3 * GAMMA - 2);
        fac3 = pow(All.Time, 3 * (1 - GAMMA) / 2.0);
        hubble_a = hubble_function(All.Time);
        a3inv = 1 / (All.Time * All.Time * All.Time);
        atime = All.Time;
    }
    else
        fac1 = fac2 = fac3 = hubble_a = a3inv = atime = 1;

    if(Flag_FullStep || dt_displacement == 0)
        find_dt_displacement_constraint(hubble_a * atime * atime);

#ifdef PMGRID
    if(All.ComovingIntegrationOn)
        dt_gravkickB = get_gravkick_factor(All.PM_Ti_begstep, All.Ti_Current) -
            get_gravkick_factor(All.PM_Ti_begstep, (All.PM_Ti_begstep + All.PM_Ti_endstep) / 2);
    else
        dt_gravkickB = (All.Ti_Current - (All.PM_Ti_begstep + All.PM_Ti_endstep) / 2) * All.Timebase_interval;
#endif

#ifdef MAKEGLASS
    for(i = 0, dispmax = 0, disp2sum = 0; i < NumPart; i++)
    {
        for(j = 0; j < 3; j++)
        {
            P[i].g.GravAccel[j] *= -1;
#ifdef PMGRID
            P[i].GravPM[j] *= -1;
            P[i].g.GravAccel[j] += P[i].GravPM[j];
            P[i].GravPM[j] = 0;
#endif
        }

        disp = sqrt(P[i].g.GravAccel[0] * P[i].g.GravAccel[0] +
                P[i].g.GravAccel[1] * P[i].g.GravAccel[1] + P[i].g.GravAccel[2] * P[i].g.GravAccel[2]);

        disp *= 2.0 / (3 * All.Hubble * All.Hubble);

        disp2sum += disp * disp;

        if(disp > dispmax)
            dispmax = disp;
    }

    MPI_Allreduce(&dispmax, &globmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&disp2sum, &globdisp2sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    dmean = pow(P[0].Mass / (All.Omega0 * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.G)), 1.0 / 3);

    if(globmax > dmean)
        fac = dmean / globmax;
    else
        fac = 1.0;

    if(ThisTask == 0)
    {
        printf("\nglass-making:  dmean= %g  global disp-maximum= %g  rms= %g\n\n",
                dmean, globmax, sqrt(globdisp2sum / All.TotNumPart));
        fflush(stdout);
    }

    for(i = 0, dispmax = 0; i < NumPart; i++)
    {
        for(j = 0; j < 3; j++)
        {
            P[i].Vel[j] = 0;
            P[i].Pos[j] += fac * P[i].g.GravAccel[j] * 2.0 / (3 * All.Hubble * All.Hubble);
            P[i].g.GravAccel[j] = 0;
        }
    }
#endif




    All.DoDynamicUpdate = ShouldWeDoDynamicUpdate();

    /* Now assign new timesteps and kick */

    if(All.DoDynamicUpdate)
    {
        GlobFlag++;
        DomainNumChanged = 0;
        DomainList = (int *) mymalloc("DomainList", NTopleaves * sizeof(int));
        if(ThisTask == 0)
            printf("kicks will prepare for dynamic update of tree\n");
    }

#ifdef FORCE_EQUAL_TIMESTEPS
    for(i = FirstActiveParticle, ti_min = TIMEBASE; i >= 0; i = NextActiveParticle[i])
    {
        ti_step = get_timestep(i, &aphys, 0);

        if(ti_step < ti_min)
            ti_min = ti_step;
    }

    int ti_min_glob;

    MPI_Allreduce(&ti_min, &ti_min_glob, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif

#ifdef RELAXOBJECT
    if(All.Time < 0.2 * All.TimeMax)
    {
        All.RelaxFac = 1. / All.RelaxBaseFac;
    }
    else if(All.Time > 0.8 * All.TimeMax)
    {
        All.RelaxFac = 0.;
    }
    else
    {
        All.RelaxFac =
            1. / (All.RelaxBaseFac * pow(10., (All.Time - 0.2 * All.TimeMax) / (0.6 * All.TimeMax) * 3.));
    }
#endif

    badstepsizecount = 0;
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
#ifdef FORCE_EQUAL_TIMESTEPS
        ti_step = ti_min_glob;
#else
        ti_step = get_timestep(i, &aphys, 0);
#endif

        /* make it a power 2 subdivision */
        ti_min = TIMEBASE;
        while(ti_min > ti_step)
            ti_min >>= 1;
        ti_step = ti_min;

        bin = get_timestep_bin(ti_step);
        if(bin == -1) {
            printf("time-step of integer size 1 not allowed, id = %llu, debugging info follows. %d\n", P[i].ID, ti_step);
            badstepsizecount++;
        }
        binold = P[i].TimeBin;

        if(bin > binold)		/* timestep wants to increase */
        {
            while(TimeBinActive[bin] == 0 && bin > binold)	/* make sure the new step is synchronized */
                bin--;

            ti_step = bin ? (1 << bin) : 0;
        }

        if(All.Ti_Current >= TIMEBASE)	/* we here finish the last timestep. */
        {
            ti_step = 0;
            bin = 0;
        }

        if((TIMEBASE - All.Ti_Current) < ti_step)	/* check that we don't run beyond the end */
        {
            fprintf(stderr, "\n @ /* should not happen */ \n");
            endrun(888);		/* should not happen */
            fprintf(stderr, "\n @ /* should not happen */ \n");
            ti_step = TIMEBASE - All.Ti_Current;
            ti_min = TIMEBASE;
            while(ti_min > ti_step)
                ti_min >>= 1;
            ti_step = ti_min;
        }

        if(bin != binold)
        {
            TimeBinCount[binold]--;
            if(P[i].Type == 0)
            {
                TimeBinCountSph[binold]--;
#ifdef SFR
                TimeBinSfr[binold] -= SPHP(i).Sfr;
                TimeBinSfr[bin] += SPHP(i).Sfr;
#endif
            }

#ifdef BLACK_HOLES
            if(P[i].Type == 5)
            {
                TimeBin_BH_mass[binold] -= BHP(i).Mass;
                TimeBin_BH_dynamicalmass[binold] -= P[i].Mass;
                TimeBin_BH_Mdot[binold] -= BHP(i).Mdot;
                if(BHP(i).Mass > 0)
                    TimeBin_BH_Medd[binold] -= BHP(i).Mdot / BHP(i).Mass;
                TimeBin_BH_mass[bin] += BHP(i).Mass;
                TimeBin_BH_dynamicalmass[bin] += P[i].Mass;
                TimeBin_BH_Mdot[bin] += BHP(i).Mdot;
                if(BHP(i).Mass > 0)
                    TimeBin_BH_Medd[bin] += BHP(i).Mdot / BHP(i).Mass;
            }
#endif

            prev = PrevInTimeBin[i];
            next = NextInTimeBin[i];

            if(FirstInTimeBin[binold] == i)
                FirstInTimeBin[binold] = next;
            if(LastInTimeBin[binold] == i)
                LastInTimeBin[binold] = prev;
            if(prev >= 0)
                NextInTimeBin[prev] = next;
            if(next >= 0)
                PrevInTimeBin[next] = prev;

            if(TimeBinCount[bin] > 0)
            {
                PrevInTimeBin[i] = LastInTimeBin[bin];
                NextInTimeBin[LastInTimeBin[bin]] = i;
                NextInTimeBin[i] = -1;
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

            P[i].TimeBin = bin;
        }

#ifndef WAKEUP
        ti_step_old = binold ? (1 << binold) : 0;
#else
        ti_step_old = P[i].dt_step;
#endif

        tstart = P[i].Ti_begstep + ti_step_old / 2;	/* midpoint of old step */
        tend = P[i].Ti_begstep + ti_step_old + ti_step / 2;	/* midpoint of new step */

        P[i].Ti_begstep += ti_step_old;
#ifdef WAKEUP
        P[i].dt_step = ti_step;
#endif

        do_the_kick(i, tstart, tend, P[i].Ti_begstep);
    }
    MPI_Allreduce(&badstepsizecount, &badstepsizecount_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if(badstepsizecount_global) {
        if (ThisTask == 0)
            printf("bad timestep spotted terminating and saving snapshot as %d\n", All.SnapshotFileCount);
        DumpFlag = 1;
        All.NumCurrentTiStep = 0;
        savepositions(999999);
        MPI_Barrier(MPI_COMM_WORLD);
        endrun(1231134);
    }
    if(All.DoDynamicUpdate)
    {
        force_finish_kick_nodes();
        myfree(DomainList);
    }


#ifdef CONDUCTION
    if(All.Conduction_Ti_endstep == All.Ti_Current)
    {
        ti_step = TIMEBASE;
        while(ti_step > (All.MaxSizeConductionStep / All.Timebase_interval))
            ti_step >>= 1;
        while(ti_step > (All.MaxSizeTimestep / All.Timebase_interval))
            ti_step >>= 1;

        if(ti_step > (All.Conduction_Ti_endstep - All.Conduction_Ti_begstep))	/* PM-timestep wants to increase */
        {
            /* we only increase if an integer number of steps will bring us to the end */
            if(((TIMEBASE - All.Conduction_Ti_endstep) % ti_step) > 0)
                ti_step = All.Conduction_Ti_endstep - All.Conduction_Ti_begstep;	/* leave at old step */
        }

        if(All.Ti_Current == TIMEBASE)	/* we here finish the last timestep. */
            ti_step = 0;

        All.Conduction_Ti_begstep = All.Conduction_Ti_endstep;
        All.Conduction_Ti_endstep = All.Conduction_Ti_begstep + ti_step;
    }
#endif

#ifdef CR_DIFFUSION
    if(All.CR_Diffusion_Ti_endstep == All.Ti_Current)
    {
        if(All.CR_Diffusion_Ti_endstep < All.Ti_Current)
            endrun(1231);

        ti_step = TIMEBASE;
        while(ti_step > (All.CR_DiffusionMaxSizeTimestep / All.Timebase_interval))
            ti_step >>= 1;
        while(ti_step > (All.MaxSizeTimestep / All.Timebase_interval))
            ti_step >>= 1;

        if(ti_step > (All.CR_Diffusion_Ti_endstep - All.CR_Diffusion_Ti_begstep))	/* PM-timestep wants to increase */
        {
            /* we only increase if an integer number of steps will bring us to the end */
            if(((TIMEBASE - All.CR_Diffusion_Ti_endstep) % ti_step) > 0)
                ti_step = All.CR_Diffusion_Ti_endstep - All.CR_Diffusion_Ti_begstep;	/* leave at old step */
        }

        if(All.Ti_Current == TIMEBASE)	/* we here finish the last timestep. */
            ti_step = 0;

        All.CR_Diffusion_Ti_begstep = All.CR_Diffusion_Ti_endstep;
        All.CR_Diffusion_Ti_endstep = All.CR_Diffusion_Ti_begstep + ti_step;
    }
#endif


#ifdef PMGRID
    if(All.PM_Ti_endstep == All.Ti_Current)	/* need to do long-range kick */
    {
        All.DoDynamicUpdate = 0;

        ti_step = TIMEBASE;
        while(ti_step > (dt_displacement / All.Timebase_interval))
            ti_step >>= 1;

        if(ti_step > (All.PM_Ti_endstep - All.PM_Ti_begstep))	/* PM-timestep wants to increase */
        {
            /* we only increase if an integer number of steps will bring us to the end */
            if(((TIMEBASE - All.PM_Ti_endstep) % ti_step) > 0)
                ti_step = All.PM_Ti_endstep - All.PM_Ti_begstep;	/* leave at old step */
        }

        if(All.Ti_Current == TIMEBASE)	/* we here finish the last timestep. */
            ti_step = 0;

        tstart = (All.PM_Ti_begstep + All.PM_Ti_endstep) / 2;
        tend = All.PM_Ti_endstep + ti_step / 2;

        if(All.ComovingIntegrationOn)
            dt_gravkick = get_gravkick_factor(tstart, tend);
        else
            dt_gravkick = (tend - tstart) * All.Timebase_interval;

        All.PM_Ti_begstep = All.PM_Ti_endstep;
        All.PM_Ti_endstep = All.PM_Ti_begstep + ti_step;

        if(All.ComovingIntegrationOn)
            dt_gravkickB = -get_gravkick_factor(All.PM_Ti_begstep, (All.PM_Ti_begstep + All.PM_Ti_endstep) / 2);
        else
            dt_gravkickB =
                -((All.PM_Ti_begstep + All.PM_Ti_endstep) / 2 - All.PM_Ti_begstep) * All.Timebase_interval;

        for(i = 0; i < NumPart; i++)
        {
            for(j = 0; j < 3; j++)	/* do the kick */
                P[i].Vel[j] += P[i].GravPM[j] * dt_gravkick;

#ifdef DISTORTIONTENSORPS
            /* add long range tidal forces calculated on mesh */
            /* now we do the distortiontensor kick */
            for(j1 = 0; j1 < 3; j1++)
                for(j2 = 0; j2 < 3; j2++)
                {
                    dv_distortion_tensorps[j1 + 3][j2] = 0.0;
                    dv_distortion_tensorps[j1 + 3][j2 + 3] = 0.0;

                    /* the 'acceleration' is given by the product of tidaltensor and distortiontensor */
                    for(j = 0; j < 3; j++)
                    {
                        dv_distortion_tensorps[j1 + 3][j2] +=
                            P[i].tidal_tensorpsPM[j1][j] * P[i].distortion_tensorps[j][j2];
                        dv_distortion_tensorps[j1 + 3][j2 + 3] +=
                            P[i].tidal_tensorpsPM[j1][j] * P[i].distortion_tensorps[j][j2 + 3];
                    }
                    dv_distortion_tensorps[j1 + 3][j2] *= dt_gravkick;
                    dv_distortion_tensorps[j1 + 3][j2 + 3] *= dt_gravkick;
                    /* add it to the distortiontensor 'velocities' */
                    P[i].distortion_tensorps[j1 + 3][j2] += dv_distortion_tensorps[j1 + 3][j2];
                    P[i].distortion_tensorps[j1 + 3][j2 + 3] += dv_distortion_tensorps[j1 + 3][j2 + 3];
                }
#endif



            if(P[i].Type == 0)
            {
#ifndef WAKEUP
                dt_step = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0);
#else
                dt_step = P[i].dt_step;
#endif

                if(All.ComovingIntegrationOn)
                {
                    dt_gravkickA = get_gravkick_factor(P[i].Ti_begstep, All.Ti_Current) -
                        get_gravkick_factor(P[i].Ti_begstep, P[i].Ti_begstep + dt_step / 2);
                    dt_hydrokick = get_hydrokick_factor(P[i].Ti_begstep, All.Ti_Current) -
                        get_hydrokick_factor(P[i].Ti_begstep, P[i].Ti_begstep + dt_step / 2);
                }
                else
                    dt_gravkickA = dt_hydrokick =
                        (All.Ti_Current - (P[i].Ti_begstep + dt_step / 2)) * All.Timebase_interval;

                for(j = 0; j < 3; j++)
                    SPHP(i).VelPred[j] = P[i].Vel[j]
                        + P[i].g.GravAccel[j] * dt_gravkickA
                        + SPHP(i).a.HydroAccel[j] * dt_hydrokick + P[i].GravPM[j] * dt_gravkickB;
            }
        }
    }
#endif

#ifdef WAKEUP
    /* find the next kick time */
    for(n = 0, ti_next_kick = TIMEBASE; n < TIMEBINS; n++)
    {
        if(TimeBinCount[n])
        {
            if(n > 0)
            {
                dt_bin = (1 << n);
                ti_next_for_bin = (All.Ti_Current / dt_bin) * dt_bin + dt_bin;	/* next kick time for this timebin */
            }
            else
            {
                dt_bin = 0;
                ti_next_for_bin = All.Ti_Current;
            }

            if(ti_next_for_bin < ti_next_kick)
                ti_next_kick = ti_next_for_bin;
        }
    }

    MPI_Allreduce(&ti_next_kick, &ti_next_kick_global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if(ThisTask == 0)
        printf("predicting next timestep: %g\n", (ti_next_kick_global - All.Ti_Current) * All.Timebase_interval);

    max_time_bin_active = 0;
    /* get the highest bin, that is active next time */
    for(n = 0; n < TIMEBINS; n++)
    {
        dt_bin = (1 << n);

        if((ti_next_kick_global % dt_bin) == 0)
            max_time_bin_active = n;
    }

    /* move the particle on the highest bin, that is active in the next timestep and that is lower than its last timebin */
    bin = 0;
    for(n = 0; n < TIMEBINS; n++)
    {
        if(TimeBinCount[n] > 0)
        {
            bin = n;
            break;
        }
    }
    n = 0;

    for(i = 0; i < NumPart; i++)
    {
        if(P[i].Type != 0)
            continue;

        if(!SPHP(i).wakeup)
            continue;

        binold = P[i].TimeBin;
        if(TimeBinActive[binold])
            continue;

        bin = max_time_bin_active < binold ? max_time_bin_active : binold;

        if(bin != binold)
        {
            TimeBinCount[binold]--;
            if(P[i].Type == 0)
                TimeBinCountSph[binold]--;

            prev = PrevInTimeBin[i];
            next = NextInTimeBin[i];

            if(FirstInTimeBin[binold] == i)
                FirstInTimeBin[binold] = next;
            if(LastInTimeBin[binold] == i)
                LastInTimeBin[binold] = prev;
            if(prev >= 0)
                NextInTimeBin[prev] = next;
            if(next >= 0)
                PrevInTimeBin[next] = prev;

            if(TimeBinCount[bin] > 0)
            {
                PrevInTimeBin[i] = LastInTimeBin[bin];
                NextInTimeBin[LastInTimeBin[bin]] = i;
                NextInTimeBin[i] = -1;
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

            P[i].TimeBin = bin;

            if(TimeBinActive[bin])
                NumForceUpdate++;

            /* correct quantities predicted for a longer timestep */
            ti_step_old = P[i].dt_step;
            dt_step = ti_next_kick_global - P[i].Ti_begstep;
            P[i].dt_step = dt_step;
            /*
               dt_entr = (-ti_step_old / 2 + dt_step / 2) * All.Timebase_interval;
               */
            time0 = P[i].Ti_begstep;
            time1_old = P[i].Ti_begstep + ti_step_old;
            time1_new = P[i].Ti_begstep + dt_step;

            /* This part has still to be adapted ...
#ifdef PMGRID
if(All.ComovingIntegrationOn)
dt_gravkickB = get_gravkick_factor(All.PM_Ti_begstep, All.Ti_Current) -
get_gravkick_factor(All.PM_Ti_begstep, (All.PM_Ti_begstep + All.PM_Ti_endstep) / 2);
else
dt_gravkickB = (All.Ti_Current - (All.PM_Ti_begstep + All.PM_Ti_endstep) / 2) * All.Timebase_interval;
#endif
*/
            if(All.ComovingIntegrationOn)
            {
                dt_entr = dt_gravkick = dt_hydrokick = (-(time1_old - time0) / 2
                        + (time1_new - time0) / 2) * All.Timebase_interval;
                dt_gravkick = -get_gravkick_factor(time0, time1_old) / 2
                    + get_gravkick_factor(time0, time1_new) / 2;
                dt_hydrokick = -get_hydrokick_factor(time0, time1_old) / 2
                    + get_hydrokick_factor(time0, time1_new) / 2;
            }
            else
            {
                dt_entr = dt_gravkick = dt_hydrokick = (-(time1_old - time0) / 2
                        + (time1_new - time0) / 2) * All.Timebase_interval;
            }

            /* This may now work in comoving runs */
            /* WARNING: this velocity correction is inconsistent, 
             * as the position of the particle was calculated with a "wrong" velocity before  */
            for(k = 0; k < 3; k++)
            {
                P[i].Vel[k] += P[i].g.GravAccel[k] * dt_gravkick;
            }

            for(k = 0; k < 3; k++)
            {
                P[i].Vel[k] += SPHP(i).a.HydroAccel[k] * dt_hydrokick;
            }
#if defined(MAGNETIC) && !defined(EULERPOTENTIALS) && !defined(VECT_POTENTIAL)
            for(k = 0; k < 3; k++)
            {
                SPHP(i).B[k] += SPHP(i).DtB[k] * dt_entr;
                SPHP(i).BPred[k] = SPHP(i).B[k] - SPHP(i).DtB[k] * dt_entr;
            }
#endif
#if !defined(EOS_DEGENERATE)
            SPHP(i).Entropy += SPHP(i).e.DtEntropy * dt_entr;
#else
            SPHP(i).Entropy += SPHP(i).e.DtEntropy * dt_entr * All.UnitTime_in_s;
#endif

#ifdef NUCLEAR_NETWORK
            for(k = 0; k < EOS_NSPECIES; k++)
            {
                SPHP(i).xnuc[k] += SPHP(i).dxnuc[k] * dt_entr * All.UnitTime_in_s;
            }
            network_normalize(SPHP(i).xnuc, &SPHP(i).Entropy);
#endif

            n++;
        }
    }

    sumup_large_ints(1, &n, &ntot);
    if(ThisTask == 0)
        printf("%d%09d particles woken up.\n", (int) (ntot / 1000000000), (int) (ntot % 1000000000));
#endif

    CPU_Step[CPU_TIMELINE] += measure_time();
}



void do_the_kick(int i, int tstart, int tend, int tcurrent)
{
    int j;
    MyFloat dv[3];
    double minentropy;
    double dt_entr, dt_gravkick, dt_hydrokick, dt_gravkick2, dt_hydrokick2, dt_entr2;
#ifdef CHEMISTRY
    int ifunc, mode;
    double a_start, a_end;
#endif

#ifdef DISTORTIONTENSORPS
    /* for the distortion 'velocity part', so only the lower two 3x3 submatrices will be != 0 */
    MyDouble dv_distortion_tensorps[6][6];
    int j1, j2;
#endif

#ifdef MAX_GAS_VEL
    double vv,velfac;
#endif

    if(All.ComovingIntegrationOn)
    {
        dt_entr = (tend - tstart) * All.Timebase_interval;
        dt_entr2 = (tend - tcurrent) * All.Timebase_interval;
        dt_gravkick = get_gravkick_factor(tstart, tend);
        dt_hydrokick = get_hydrokick_factor(tstart, tend);
        dt_gravkick2 = get_gravkick_factor(tcurrent, tend);
        dt_hydrokick2 = get_hydrokick_factor(tcurrent, tend);
#ifdef MAX_GAS_VEL
        velfac = 1 / sqrt(1 / (All.Time * All.Time * All.Time));
#endif
    }
    else
    {
        dt_entr = dt_gravkick = dt_hydrokick = (tend - tstart) * All.Timebase_interval;
        dt_gravkick2 = dt_hydrokick2 = dt_entr2 = (tend - tcurrent) * All.Timebase_interval;
#ifdef MAX_GAS_VEL
        velfac = 1;
#endif
    }


    /* do the kick */

    for(j = 0; j < 3; j++)
    {
        dv[j] = P[i].g.GravAccel[j] * dt_gravkick;
#ifdef RELAXOBJECT
        dv[j] -= P[i].Vel[j] * All.RelaxFac * dt_gravkick;
#endif
        P[i].Vel[j] += dv[j];
    }

#ifdef DISTORTIONTENSORPS
    /* now we do the distortiontensor kick */
    for(j1 = 0; j1 < 3; j1++)
        for(j2 = 0; j2 < 3; j2++)
        {
            dv_distortion_tensorps[j1 + 3][j2] = 0.0;
            dv_distortion_tensorps[j1 + 3][j2 + 3] = 0.0;

            /* the 'acceleration' is given by the product of tidaltensor and distortiontensor */
            for(j = 0; j < 3; j++)
            {
                dv_distortion_tensorps[j1 + 3][j2] +=
                    P[i].tidal_tensorps[j1][j] * P[i].distortion_tensorps[j][j2];
                dv_distortion_tensorps[j1 + 3][j2 + 3] +=
                    P[i].tidal_tensorps[j1][j] * P[i].distortion_tensorps[j][j2 + 3];
            }
            dv_distortion_tensorps[j1 + 3][j2] *= dt_gravkick;
            dv_distortion_tensorps[j1 + 3][j2 + 3] *= dt_gravkick;

            /* add it to the distortiontensor 'velocities' */
            P[i].distortion_tensorps[j1 + 3][j2] += dv_distortion_tensorps[j1 + 3][j2];
            P[i].distortion_tensorps[j1 + 3][j2 + 3] += dv_distortion_tensorps[j1 + 3][j2 + 3];
        }
#endif

    if(P[i].Type == 0)		/* SPH stuff */
    {
        for(j = 0; j < 3; j++)
        {
            //        SPHP(i).a.HydroAccel[0] = 0.0;
            dv[j] += SPHP(i).a.HydroAccel[j] * dt_hydrokick;
            P[i].Vel[j] += SPHP(i).a.HydroAccel[j] * dt_hydrokick;

            SPHP(i).VelPred[j] =
                P[i].Vel[j] - dt_gravkick2 * P[i].g.GravAccel[j] - dt_hydrokick2 * SPHP(i).a.HydroAccel[j];
#ifdef PMGRID
            SPHP(i).VelPred[j] += P[i].GravPM[j] * dt_gravkickB;
#endif
#if defined(MAGNETIC) && !defined(EULERPOTENTIALS) && !defined(VECT_POTENTIAL)
            SPHP(i).B[j] += SPHP(i).DtB[j] * dt_entr;
            SPHP(i).BPred[j] = SPHP(i).B[j] - SPHP(i).DtB[j] * dt_entr2;
#endif
#ifdef VECT_POTENTIAL
            SPHP(i).A[j] += SPHP(i).DtA[j] * dt_entr;
            SPHP(i).APred[j] = SPHP(i).A[j] - SPHP(i).DtA[j] * dt_entr2;
#endif
        }

#ifdef MAX_GAS_VEL
        vv=0;
        for(j=0; j < 3; j++)
            vv += P[i].Vel[j] * P[i].Vel[j];
        vv = sqrt(vv);
        if(vv > MAX_GAS_VEL * velfac)
            for(j=0;j < 3; j++)
            {
                P[i].Vel[j] *= MAX_GAS_VEL * velfac / vv;
                SPHP(i).VelPred[j] =
                    P[i].Vel[j] - dt_gravkick2 * P[i].g.GravAccel[j] - dt_hydrokick2 * SPHP(i).a.HydroAccel[j];
#ifdef PMGRID
                SPHP(i).VelPred[j] += P[i].GravPM[j] * dt_gravkickB;
#endif
            }
#endif

#if defined(MAGNETIC) && defined(DIVBCLEANING_DEDNER)
        SPHP(i).Phi += SPHP(i).DtPhi * dt_entr;
        SPHP(i).PhiPred = SPHP(i).Phi - SPHP(i).DtPhi * dt_entr2;
#endif
#ifdef TIME_DEP_ART_VISC
        SPHP(i).alpha += SPHP(i).Dtalpha * dt_entr;
        SPHP(i).alpha = DMIN(SPHP(i).alpha, All.ArtBulkViscConst);
        if(SPHP(i).alpha < All.AlphaMin)
            SPHP(i).alpha = All.AlphaMin;
#endif
#ifdef TIME_DEP_MAGN_DISP
        SPHP(i).Balpha += SPHP(i).DtBalpha * dt_entr;
        SPHP(i).Balpha = DMIN(SPHP(i).Balpha, All.ArtMagDispConst);
        if(SPHP(i).Balpha < All.ArtMagDispMin)
            SPHP(i).Balpha = All.ArtMagDispMin;
#endif
        /* In case of cooling, we prevent that the entropy (and
           hence temperature decreases by more than a factor 0.5 */

        if(SPHP(i).e.DtEntropy * dt_entr > -0.5 * SPHP(i).Entropy)
#if !defined(EOS_DEGENERATE)
            SPHP(i).Entropy += SPHP(i).e.DtEntropy * dt_entr;
#else
        SPHP(i).Entropy += SPHP(i).e.DtEntropy * dt_entr * All.UnitTime_in_s;
#endif
        else
            SPHP(i).Entropy *= 0.5;

#ifdef NUCLEAR_NETWORK
        for(j = 0; j < EOS_NSPECIES; j++)
        {
            SPHP(i).xnuc[j] += SPHP(i).dxnuc[j] * dt_entr * All.UnitTime_in_s;
        }
        network_normalize(SPHP(i).xnuc, &SPHP(i).Entropy);
#endif

#ifdef CHEMISTRY
        /* update the chemical abundances for the new density and temperature */
        a_start = All.TimeBegin * exp(tstart * All.Timebase_interval);
        a_end = All.TimeBegin * exp(tend * All.Timebase_interval);

        /* time in cosmic expansion parameter */
        ifunc = compute_abundances(mode = 1, i, a_start, a_end);
#endif

        if(All.MinEgySpec)
        {
#ifndef TRADITIONAL_SPH_FORMULATION
            minentropy = All.MinEgySpec * GAMMA_MINUS1 / pow(SPHP(i).EOMDensity * a3inv, GAMMA_MINUS1);
#else
            minentropy = All.MinEgySpec;
#endif
            if(SPHP(i).Entropy < minentropy)
            {
                SPHP(i).Entropy = minentropy;
                SPHP(i).e.DtEntropy = 0;
            }
        }

        /* In case the timestep increases in the new step, we
           make sure that we do not 'overcool'. */
#ifndef WAKEUP
        dt_entr = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) / 2 * All.Timebase_interval;
#else
        dt_entr = P[i].dt_step / 2 * All.Timebase_interval;
#endif
        if(SPHP(i).Entropy + SPHP(i).e.DtEntropy * dt_entr < 0.5 * SPHP(i).Entropy)
            SPHP(i).e.DtEntropy = -0.5 * SPHP(i).Entropy / dt_entr;
    }

    if(All.DoDynamicUpdate)
        force_kick_node(i, dv);
}



/*! This function normally (for flag==0) returns the maximum allowed timestep of a particle, expressed in
 *  terms of the integer mapping that is used to represent the total simulated timespan. The physical
 *  acceleration is returned in aphys. The latter is used in conjunction with the PSEUDOSYMMETRIC integration
 *  option, which also makes of the second function of get_timestep. When it is called with a finite timestep
 *  for flag, it returns the physical acceleration that would lead to this timestep, assuming timestep
 *  criterion 0.
 */
int get_timestep(int p,		/*!< particle index */
        double *aphys,	/*!< acceleration (physical units) */
        int flag	/*!< either 0 for normal operation, or finite timestep to get corresponding
                      aphys */ )
{
    double ax, ay, az, ac;
    double csnd = 0, dt = 0, dt_courant = 0;
    int ti_step;
    double dt_viscous = 0;
#ifdef CHEMCOOL
    double hubble_param;

    if(All.ComovingIntegrationOn)
        hubble_param = All.HubbleParam;
    else
        hubble_param = 1.0;
#endif

#ifdef BLACK_HOLES
    double dt_accr;
    double dt_limiter;

#ifdef UNIFIED_FEEDBACK
    double meddington = 0;
#endif
#endif

#ifdef NS_TIMESTEP
    double dt_NS = 0;
#endif

#ifdef NONEQUILIBRIUM
    double dt_cool, dt_elec;
#endif

#ifdef COSMIC_RAYS
    int CRpop;
#endif

#ifdef NUCLEAR_NETWORK
    double dt_network, dt_species;
    int k;
#endif

    if(flag <= 0)
    {
        ax = fac1 * P[p].g.GravAccel[0];
        ay = fac1 * P[p].g.GravAccel[1];
        az = fac1 * P[p].g.GravAccel[2];

#ifdef PMGRID
        ax += fac1 * P[p].GravPM[0];
        ay += fac1 * P[p].GravPM[1];
        az += fac1 * P[p].GravPM[2];
#endif

        if(P[p].Type == 0)
        {
            ax += fac2 * SPHP(p).a.HydroAccel[0];
            ay += fac2 * SPHP(p).a.HydroAccel[1];
            az += fac2 * SPHP(p).a.HydroAccel[2];
        }

        ac = sqrt(ax * ax + ay * ay + az * az);	/* this is now the physical acceleration */
        *aphys = ac;
    }
    else
        ac = *aphys;

    if(ac == 0)
        ac = 1.0e-30;


    switch (All.TypeOfTimestepCriterion)
    {
        case 0:
            if(flag > 0)
            {
                dt = flag * All.Timebase_interval;

                dt /= hubble_a;	/* convert dloga to physical timestep  */

                ac = 2 * All.ErrTolIntAccuracy * atime * All.SofteningTable[P[p].Type] / (dt * dt);
                *aphys = ac;
                return flag;
            }
            dt = sqrt(2 * All.ErrTolIntAccuracy * atime * All.SofteningTable[P[p].Type] / ac);
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
#ifdef ADAPTIVE_GRAVSOFT_FORGAS_HSML
            if(P[p].Type == 0)
                dt = sqrt(2 * All.ErrTolIntAccuracy * atime * P[p].Hsml / 2.8 / ac);
#else
            if(P[p].Type == 0)
                dt =
                    sqrt(2 * All.ErrTolIntAccuracy * atime * All.SofteningTable[P[p].Type] *
                            pow(P[p].Mass / All.ReferenceGasMass, 1.0 / 3) / ac);
#endif
#endif
            break;

        default:
            fprintf(stderr, "\n !!!2@@@!!! \n");
            endrun(888);
            fprintf(stderr, "\n !!!2@@@!!! \n");
            break;
    }


    if(P[p].Type == 0)
    {
        csnd = sqrt(GAMMA * SPHP(p).Pressure / SPHP(p).EOMDensity);

#ifdef ALTERNATIVE_VISCOUS_TIMESTEP

        if(All.ComovingIntegrationOn)
            dt_courant = All.CourantFac * All.Time * DMAX(P[p].Hsml, All.SofteningTable[0]) / (fac3 * csnd);
        else
            dt_courant = All.CourantFac * DMAX(P[p].Hsml, All.SofteningTable[0]) / csnd;

        if(dt_courant > 2 * All.CourantFac * SPHP(p).MinViscousDt)
            dt_courant = 2 * All.CourantFac * SPHP(p).MinViscousDt;
#else
        if(All.ComovingIntegrationOn)
            dt_courant = 2 * All.CourantFac * All.Time * P[p].Hsml / (fac3 * SPHP(p).MaxSignalVel);
        else
            dt_courant = 2 * All.CourantFac * P[p].Hsml / SPHP(p).MaxSignalVel;
#endif

        if(dt_courant < dt)
            dt = dt_courant;

#ifdef MYFALSE
        dt_viscous = All.CourantFac * SPHP(p).MaxViscStep / hubble_a;	/* to convert dloga to physical dt */

        if(dt_viscous < dt)
            dt = dt_viscous;
#endif

#ifdef NS_TIMESTEP
        if(fabs(SPHP(p).ViscEntropyChange))
        {
            dt_NS = VISC_TIMESTEP_PARAMETER * SPHP(p).Entropy / SPHP(p).ViscEntropyChange / hubble_a;

            if(dt_NS < dt)
                dt = dt_NS;
        }
#endif


#ifdef NUCLEAR_NETWORK
        if(SPHP(p).temp > 1e7)
        {
            /* check if the new timestep blows up our abundances */
            dt_network = dt * All.UnitTime_in_s;
            for(k = 0; k < EOS_NSPECIES; k++)
            {
                if(SPHP(p).dxnuc[k] > 0)
                {
                    dt_species = (1.0 - SPHP(p).xnuc[k]) / SPHP(p).dxnuc[k];
                    if(dt_species < dt_network)
                        dt_network = dt_species;
                }
                else if(SPHP(p).dxnuc[k] < 0)
                {
                    dt_species = (0.0 - SPHP(p).xnuc[k]) / SPHP(p).dxnuc[k];
                    if(dt_species < dt_network)
                        dt_network = dt_species;
                }

            }

            dt_network /= All.UnitTime_in_s;
            if(dt_network < dt)
                dt = dt_network;
        }
#endif

#ifdef CHEMCOOL
        dt_courant = do_chemcool(p, dt_courant * All.UnitTime_in_s / hubble_param);

        if(dt_courant < dt)
            dt = dt_courant;
#endif
    }

#ifdef BLACK_HOLES
    if(P[p].Type == 5)
    {
        if(BHP(p).Mdot > 0 && BHP(p).Mass > 0)
        {
            dt_accr = 0.25 * BHP(p).Mass / BHP(p).Mdot;
            if(dt_accr < dt)
                dt = dt_accr;
        }
        if(BHP(p).TimeBinLimit > 0) {
            dt_limiter = 0.5 * (1L << BHP(p).TimeBinLimit) * All.Timebase_interval / hubble_a;
            if (dt_limiter < dt) dt = dt_limiter;
        }
    }
#endif

#ifdef BH_BUBBLES
    if(P[p].Type == 5)
    {
        if(BHP(p).Mdot > 0 && BHP(p).Mass > 0)
        {
#ifdef UNIFIED_FEEDBACK
            meddington = (4 * M_PI * GRAVITY * C * PROTONMASS /
                    (0.1 * C * C * THOMPSON)) * BHP(p).Mass * All.UnitTime_in_s;
            if(BHP(p).Mdot < All.RadioThreshold * meddington)
#endif
                dt_accr = (All.BlackHoleRadioTriggeringFactor - 1) * BHP(p).Mass / BHP(p).Mdot;
            if(dt_accr < dt)
                dt = dt_accr;
        }
    }
#endif

#ifdef NONEQUILIBRIUM
    /* another criterion given by the local cooling time */

    if(P[p].Type == 0)
    {
        dt_cool = fabs(SPHP(p).t_cool);	/* still in yrs */
        dt_cool *= SEC_PER_YEAR;	/* in seconds */
        dt_cool /= All.UnitTime_in_s;
        dt_cool *= All.HubbleParam;	/* internal units */

        dt_cool = All.Epsilon * dt_cool;


#ifndef UM_CONTINUE      
        if(dt_cool > 0 && dt_cool < dt)
            dt = dt_cool;
#else
        if(dt_cool > 0 && dt_cool < dt && SPHP(p).DelayTime  < 0)
            dt = dt_cool;
#endif


        /* yet another criterion given by the electron number density change */

        dt_elec = fabs(SPHP(p).t_elec);	/* still in yrs */
        dt_elec *= SEC_PER_YEAR;	/* in seconds */
        dt_elec /= All.UnitTime_in_s;
        dt_elec *= All.HubbleParam;	/* internal units */


        dt_elec = All.Epsilon * dt_elec;

#ifndef UM_CONTINUE            
        if(dt_elec > 0 && dt_elec < dt)
            dt = dt_elec;
#else
        if(dt_elec > 0 && dt_elec < dt && SPHP(p).DelayTime  < 0)
            dt = dt_elec;
#endif      
    }
#endif

    /* convert the physical timestep to dloga if needed. Note: If comoving integration has not been selected,
       hubble_a=1.
       */
    dt *= hubble_a;

#ifdef ONLY_PM
    dt = All.MaxSizeTimestep;
#endif



    if(dt >= All.MaxSizeTimestep)
        dt = All.MaxSizeTimestep;


    if(dt >= dt_displacement)
        dt = dt_displacement;


#ifdef CONDUCTION
    if(P[p].Type == 0)
        if(dt >= All.MaxSizeConductionStep)
            dt = All.MaxSizeConductionStep;
#endif
#ifdef CR_DIFFUSION
    if(P[p].Type == 0)
        if(dt >= All.CR_DiffusionMaxSizeTimestep)
            dt = All.CR_DiffusionMaxSizeTimestep;
#endif

    ti_step = (int) (dt / All.Timebase_interval);

    if(!(ti_step > 1 && ti_step < TIMEBASE))
    {
        printf("\nError: A timestep of size zero was assigned on the integer timeline!\n"
                "We better stop.\n"
                "Task=%d type %d Part-ID=%llu dt=%g dtc=%g dtv=%g dtdis=%g tibase=%g ti_step=%d ac=%g xyz=(%g|%g|%g) tree=(%g|%g|%g), dt0=%g, ErrTolIntAccuracy=%g\n\n",
                ThisTask, P[p].Type, (MyIDType)P[p].ID, dt, dt_courant, dt_viscous, dt_displacement,
                All.Timebase_interval, ti_step, ac,
                P[p].Pos[0], P[p].Pos[1], P[p].Pos[2], P[p].g.GravAccel[0], P[p].g.GravAccel[1],
                P[p].g.GravAccel[2],
                sqrt(2 * All.ErrTolIntAccuracy * atime * All.SofteningTable[P[p].Type] / ac) * hubble_a, All.ErrTolIntAccuracy
              );
#ifdef PMGRID
        printf("pm_force=(%g|%g|%g)\n", P[p].GravPM[0], P[p].GravPM[1], P[p].GravPM[2]);
#endif
        if(P[p].Type == 0)
            printf("hydro-frc=(%g|%g|%g) dens=%g hsml=%g numngb=%g\n", SPHP(p).a.HydroAccel[0], SPHP(p).a.HydroAccel[1],
                    SPHP(p).a.HydroAccel[2], SPHP(p).d.Density, P[p].Hsml, P[p].n.NumNgb);
#ifdef DENSITY_INDEPENDENT_SPH
        if(P[p].Type == 0)
            printf("egyrho=%g entvarpred=%g dhsmlegydensityfactor=%g Entropy=%g, dtEntropy=%g, Pressure=%g\n", SPHP(p).EgyWtDensity, SPHP(p).EntVarPred,
                    SPHP(p).DhsmlEgyDensityFactor, SPHP(p).Entropy, SPHP(p).e.DtEntropy, SPHP(p).Pressure);
#endif
#ifdef SFR
        if(P[p].Type == 0) {
            printf("sfr = %g\n" , SPHP(p).Sfr);
        }
#endif
#if defined(BH_THERMALFEEDBACK) || defined(BH_KINETICFEEDBACK)
        if(P[p].Type == 0) {
            printf("injected_energy = %g\n" , SPHP(p).i.Injected_BH_Energy);
        }
#endif
#ifdef COSMIC_RAYS
        if(P[p].Type == 0)
            for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
                printf("Cosmic Ray Properties: C0: %g -- q0  : %g -- P  : %g\n"
                        "                       Rho: %g\n\n",
                        SPHP(p).CR_C0[CRpop], SPHP(p).CR_q0[CRpop], CR_Particle_Pressure(SphP + p, CRpop),
                        SPHP(p).d.Density);
#endif

        fflush(stdout);
        //endrun(818);
    }

    return ti_step;
}


/*! This function computes an upper limit ('dt_displacement') to the global timestep of the system based on
 *  the rms velocities of particles. For cosmological simulations, the criterion used is that the rms
 *  displacement should be at most a fraction MaxRMSDisplacementFac of the mean particle separation. Note that
 *  the latter is estimated using the assigned particle masses, separately for each particle type. If comoving
 *  integration is not used, the function imposes no constraint on the timestep.
 */
void find_dt_displacement_constraint(double hfac /*!<  should be  a^2*H(a)  */ )
{
    int i, type;
    int count[6];
    long long count_sum[6];
    double v[6], v_sum[6], mim[6], min_mass[6];
    double dt, dmean, asmth = 0;

    dt_displacement = All.MaxSizeTimestep;

    if(All.ComovingIntegrationOn)
    {
        for(type = 0; type < 6; type++)
        {
            count[type] = 0;
            v[type] = 0;
            mim[type] = 1.0e30;
        }

        for(i = 0; i < NumPart; i++)
        {
            v[P[i].Type] += P[i].Vel[0] * P[i].Vel[0] + P[i].Vel[1] * P[i].Vel[1] + P[i].Vel[2] * P[i].Vel[2];
            if(P[i].Mass > 0)
            {
                if(mim[P[i].Type] > P[i].Mass)
                    mim[P[i].Type] = P[i].Mass;
            }
            count[P[i].Type]++;
        }

        MPI_Allreduce(v, v_sum, 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(mim, min_mass, 6, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        sumup_large_ints(6, count, count_sum);

#ifdef SFR
        /* add star and gas particles together to treat them on equal footing, using the original gas particle
           spacing. */
        v_sum[0] += v_sum[4];
        count_sum[0] += count_sum[4];
        v_sum[4] = v_sum[0];
        count_sum[4] = count_sum[0];
#ifdef BLACK_HOLES
        v_sum[0] += v_sum[5];
        count_sum[0] += count_sum[5];
        v_sum[5] = v_sum[0];
        count_sum[5] = count_sum[0];
        min_mass[5] = min_mass[0];
#endif
#endif

        for(type = 0; type < 6; type++)
        {
            if(count_sum[type] > 0)
            {
                if(type == 0 || (type == 4 && All.StarformationOn))
                    dmean =
                        pow(min_mass[type] / (All.OmegaBaryon * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.G)),
                                1.0 / 3);
                else
                    dmean =
                        pow(min_mass[type] /
                                ((All.Omega0 - All.OmegaBaryon) * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.G)),
                                1.0 / 3);

#ifdef BLACK_HOLES
                if(type == 5)
                    dmean =
                        pow(min_mass[type] / (All.OmegaBaryon * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.G)),
                                1.0 / 3);
#endif
                dt = All.MaxRMSDisplacementFac * hfac * dmean / sqrt(v_sum[type] / count_sum[type]);

#ifdef PMGRID
                asmth = All.Asmth[0];
#ifdef PLACEHIGHRESREGION
                if(((1 << type) & (PLACEHIGHRESREGION)))
                    asmth = All.Asmth[1];
#endif
                if(asmth < dmean)
                    dt = All.MaxRMSDisplacementFac * hfac * asmth / sqrt(v_sum[type] / count_sum[type]);
#endif

                if(ThisTask == 0)
                    printf("type=%d  dmean=%g asmth=%g minmass=%g a=%g  sqrt(<p^2>)=%g  dlogmax=%g\n",
                            type, dmean, asmth, min_mass[type], All.Time, sqrt(v_sum[type] / count_sum[type]), dt);


#ifdef NEUTRINOS
                if(type != 2)	/* don't constrain the step to the neutrinos */
#endif
                    if(dt < dt_displacement)
                        dt_displacement = dt;
            }
        }

        if(ThisTask == 0)
            printf("displacement time constraint: %g  (%g)\n", dt_displacement, All.MaxSizeTimestep);
    }
}

int get_timestep_bin(int ti_step)
{
    int bin = -1;

    if(ti_step == 0)
        return 0;

    if(ti_step == 1)
    {
        return -1;
    }

    while(ti_step)
    {
        bin++;
        ti_step >>= 1;
    }

    return bin;
}
