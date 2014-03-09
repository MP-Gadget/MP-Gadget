#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <ctype.h>

#include "allvars.h"
#include "proto.h"

/*! \file run.c
 *  \brief  iterates over timesteps, main loop
 */

/*! This routine contains the main simulation loop that iterates over
 * single timesteps. The loop terminates when the cpu-time limit is
 * reached, when a `stop' file is found in the output directory, or
 * when the simulation ends because we arrived at TimeMax.
 */
void run(void)
{
    FILE *fd;

#if defined(RADIATIVE_RATES) || defined(RADIATION)
    int ifunc;
#endif
    int stopflag = 0;
    char buf[200], stopfname[200], contfname[200];


    sprintf(stopfname, "%sstop", All.OutputDir);
    sprintf(contfname, "%scont", All.OutputDir);
    unlink(contfname);


    CPU_Step[CPU_MISC] += measure_time();

#ifdef DENSITY_BASED_SNAPS
    All.nh_next = 10.0;
#endif


    do				/* main loop */
    {

        find_next_sync_point_and_drift();	/* find next synchronization point and drift particles to this time.
                                             * If needed, this function will also write an output file
                                             * at the desired time.
                                             */

        every_timestep_stuff();	/* write some info to log-files */

#if defined(RADIATIVE_RATES) || defined(RADIATION)
        ifunc = init_rad(All.Time);
#endif

#ifdef COOLING
        IonizeParams();		/* set UV background for the current time */
#endif

#ifdef COMPUTE_POTENTIAL_ENERGY
        if((All.Time - All.TimeLastStatistics) >= All.TimeBetStatistics)
            All.NumForcesSinceLastDomainDecomp = (int64_t) (1 + All.TotNumPart * All.TreeDomainUpdateFrequency);
#endif

        domain_Decomposition();	/* do domain decomposition if needed */


        compute_accelerations(0);	/* compute accelerations for 
                                     * the particles that are to be advanced  
                                     */

#ifdef SINKS
        do_sinks();
#endif

#ifdef INVARIANCETEST
        compare_partitions();
#endif

#ifdef SCF_HYBRID
        SCF_do_center_of_mass_correction(0.75,10.0*SCF_HQ_A, 0.01, 1000);
#endif
        /* check whether we want a full energy statistics */
        if((All.Time - All.TimeLastStatistics) >= All.TimeBetStatistics)
        {
#ifdef COMPUTE_POTENTIAL_ENERGY
            compute_potential();
#endif
            energy_statistics();	/* compute and output energy statistics */

#ifdef SCFPOTENTIAL
            SCF_write(0);
#endif
            All.TimeLastStatistics += All.TimeBetStatistics;
        }

        advance_and_find_timesteps();	/* 'kick' active particles in
                                         * momentum space and compute new
                                         * timesteps for them
                                         */

        write_cpu_log();		/* produce some CPU usage info */

        All.NumCurrentTiStep++;

        /* Check whether we need to interrupt the run */
        if(ThisTask == 0)
        {
            /* Is the stop-file present? If yes, interrupt the run. */
            if((fd = fopen(stopfname, "r")))
            {
                fclose(fd);
                stopflag = 1;
                unlink(stopfname);
            }

            /* are we running out of CPU-time ? If yes, interrupt run. */
            if(CPUThisRun > 0.85 * All.TimeLimitCPU)
            {
                printf("reaching time-limit. stopping.\n");
                stopflag = 2;
            }
        }

        MPI_Bcast(&stopflag, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if(stopflag)
        {
            restart(0);		/* write restart file */
            MPI_Barrier(MPI_COMM_WORLD);

            if(stopflag == 2 && ThisTask == 0)
            {
                if((fd = fopen(contfname, "w")))
                    fclose(fd);
            }

            if(stopflag == 2 && All.ResubmitOn && ThisTask == 0)
            {
                close_outputfiles();
                sprintf(buf, "%s", All.ResubmitCommand);
#ifndef NOCALLSOFSYSTEM
                int ret;

                ret = system(buf);
#endif
            }
            return;
        }

        /* is it time to write a regular restart-file? (for security) */
        if(ThisTask == 0)
        {
            if((CPUThisRun - All.TimeLastRestartFile) >= All.CpuTimeBetRestartFile)
            {
                All.TimeLastRestartFile = CPUThisRun;
                stopflag = 3;
            }
            else
                stopflag = 0;
        }

        MPI_Bcast(&stopflag, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if(stopflag == 3)
        {
            restart(0);		/* write an occasional restart file */
            stopflag = 0;
        }

        report_memory_usage(&HighMark_run, "RUN");
    }
    while(All.Ti_Current < TIMEBASE && All.Time <= All.TimeMax);
#ifndef SNAP_SET_TG
    restart(0);

    savepositions(All.SnapshotFileCount++);
#endif	
    /* write a last snapshot
     * file at final time (will
     * be overwritten if
     * All.TimeMax is increased
     * and the run is continued)
     */
#if defined(CHEMISTRY) || defined (UM_CHEMISTRY)
    if(ThisTask == 0)
    {
        printf("Initial abundances: \n");
        printf("HI=%g, HII=%g, HeI=%g, HeII=%g, HeIII=%g \n",
                SPHP(1).HI, SPHP(1).HII, SPHP(1).HeI, SPHP(1).HeII, SPHP(1).HeIII);

        printf("HM=%g, H2I=%g, H2II=%g, elec=%g, %d\n",
                SPHP(1).HM, SPHP(1).H2I, SPHP(1).H2II, SPHP(1).elec, P[1].ID);

#if defined (UM_CHEMISTRY) && defined (UM_HD_COOLING)
        printf("HD=%g, DI=%g, DII=%g ", SPHP(1).HD, SPHP(1).DI, SPHP(1).DII);
        printf("HeHII=%g",SPHP(1).HeHII);
#endif

        printf("\nx=%g, y=%g, z=%g, vx=%g, vy=%g, vz=%g, density=%g, entropy=%g\n",
                P[N_sph - 1].Pos[0], P[N_sph - 1].Pos[1], P[N_sph - 1].Pos[2], P[N_sph - 1].Vel[0],
                P[N_sph - 1].Vel[1], P[N_sph - 1].Vel[2], SPHP(N_sph - 1).d.Density, SPHP(N_sph - 1).Entropy);
    }

#endif


#ifdef SCFPOTENTIAL
    if(ThisTask == 0)
    { 
        printf("Free SCF...\n");
        fflush(stdout);   
    }
    SCF_free();
    if(ThisTask == 0)
    { 
        printf("done.\n");
        fflush(stdout);  
    }
#endif

}


/*! This function finds the next synchronization point of the system
 * (i.e. the earliest point of time any of the particles needs a force
 * computation), and drifts the system to this point of time.  If the
 * system dirfts over the desired time of a snapshot file, the
 * function will drift to this moment, generate an output, and then
 * resume the drift.
 */
void find_next_sync_point_and_drift(void)
{
    int n, i, prev, dt_bin, ti_next_for_bin, ti_next_kick, ti_next_kick_global;
    int64_t numforces2;
    double timeold;

    timeold = All.Time;

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

#ifdef SNAP_SET_TG
    double a3inv, hubble_param2, nh, nh_max, nh_glob_max;

    if(All.ComovingIntegrationOn)
    {
        a3inv = 1.0 / (All.Time * All.Time * All.Time);
        hubble_param2 = All.HubbleParam * All.HubbleParam;
    }
    else
        a3inv = hubble_param2 = 1.0;

    for(i = 0, nh_max = 0; i < N_sph; i++)
    {
        nh = HYDROGEN_MASSFRAC * SPHP(i).d.Density * All.UnitDensity_in_cgs * a3inv * hubble_param2 / PROTONMASS;

        if(nh > nh_max)
            nh_max = nh;
    }

    MPI_Allreduce(&nh_max, &nh_glob_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if(ThisTask == 0)
        printf("\nNH_MAX = %g\n\n", nh_glob_max);
#endif

#ifdef NSTEPS_BASED_SNAPS
    if((All.NumCurrentTiStep + 2) % All.SnapNumFac == 0)
        savepositions(All.SnapshotFileCount++);
#else

#ifdef DENSITY_BASED_SNAPS
    if(nh_glob_max > All.nh_next)
    {
        All.nh_next *= pow(10.0, 0.25);

        if(ThisTask == 0)
            printf("nh_next = %g\n", All.nh_next);

        savepositions(All.SnapshotFileCount++);
    }
#else
    while(ti_next_kick_global >= All.Ti_nextoutput && All.Ti_nextoutput >= 0)
    {
        All.Ti_Current = All.Ti_nextoutput;

        if(All.ComovingIntegrationOn)
            All.Time = All.TimeBegin * exp(All.Ti_Current * All.Timebase_interval);
        else
            All.Time = All.TimeBegin + All.Ti_Current * All.Timebase_interval;
#ifdef TIMEDEPGRAV
        All.G = All.Gini * dGfak(All.Time);
#endif

        move_particles(All.Ti_nextoutput);

        CPU_Step[CPU_DRIFT] += measure_time();

#ifdef OUTPUTPOTENTIAL
#if !defined(EVALPOTENTIAL) || (defined(EVALPOTENTIAL) && defined(RECOMPUTE_POTENTIAL_ON_OUTPUT))
        All.NumForcesSinceLastDomainDecomp = (int64_t) (1 + All.TotNumPart * All.TreeDomainUpdateFrequency);
        domain_Decomposition();
        compute_potential();
#endif
#endif


        savepositions(All.SnapshotFileCount++);	/* write snapshot file */

        All.Ti_nextoutput = find_next_outputtime(All.Ti_nextoutput + 1);
    }
#endif
#endif

    All.Ti_Current = ti_next_kick_global;

    if(All.ComovingIntegrationOn)
        All.Time = All.TimeBegin * exp(All.Ti_Current * All.Timebase_interval);
    else
        All.Time = All.TimeBegin + All.Ti_Current * All.Timebase_interval;
#ifdef TIMEDEPGRAV
    All.G = All.Gini * dGfak(All.Time);
#endif

    All.TimeStep = All.Time - timeold;


    /* mark the bins that will be active */
    for(n = 1, TimeBinActive[0] = 1, NumForceUpdate = TimeBinCount[0]; n < TIMEBINS; n++)
    {
        dt_bin = (1 << n);

        if((ti_next_kick_global % dt_bin) == 0)
        {
            TimeBinActive[n] = 1;
            NumForceUpdate += TimeBinCount[n];
        }
        else
            TimeBinActive[n] = 0;
    }

    sumup_large_ints(1, &NumForceUpdate, &GlobNumForceUpdate);

    if(GlobNumForceUpdate == All.TotNumPart)
        Flag_FullStep = 1;
    else
        Flag_FullStep = 0;

    All.NumForcesSinceLastDomainDecomp += GlobNumForceUpdate;


    FirstActiveParticle = -1;

    for(n = 0, prev = -1; n < TIMEBINS; n++)
    {
        if(TimeBinActive[n])
        {
            for(i = FirstInTimeBin[n]; i >= 0; i = NextInTimeBin[i])
            {
                if(prev == -1)
                    FirstActiveParticle = i;

                if(prev >= 0)
                    NextActiveParticle[prev] = i;

                prev = i;
            }
        }
    }

    if(prev >= 0)
        NextActiveParticle[prev] = -1;


#ifdef PERMUTATAION_OPTIMIZATION
    generate_permutation_in_active_list();
#endif

    /* drift the active particles, others will be drifted on the fly if needed */

    for(i = FirstActiveParticle, NumForceUpdate = 0; i >= 0; i = NextActiveParticle[i])
    {
        drift_particle(i, All.Ti_Current);

        NumForceUpdate++;
    }

    sumup_large_ints(1, &NumForceUpdate, &numforces2);
    if(GlobNumForceUpdate != numforces2)
    {
        printf("terrible\n");
        endrun(2);
    }


    CPU_Step[CPU_DRIFT] += measure_time();
}


#ifdef PERMUTATAION_OPTIMIZATION

#define CHUNKSIZE 1000

struct permut_data
{
    double rnd;
    int seg;
};


void generate_permutation_in_active_list(void)
{
    int i, count, nseg, maxseg, last_particle;
    int *first_list, *last_list;
    int64_t idsum_old, idsum_new;
    struct permut_data *permut;

    for(i = FirstActiveParticle, idsum_old = 0; i >= 0; i = NextActiveParticle[i])
        idsum_old += P[i].ID;

    maxseg = NumPart / CHUNKSIZE + 1;

    first_list = (int *) mymalloc("first_list", maxseg * sizeof(int));
    last_list = (int *) mymalloc("last_list", maxseg * sizeof(int));

    nseg = 0;
    first_list[nseg] = FirstActiveParticle;

    for(i = last_particle = FirstActiveParticle, count = 0; i >= 0; i = NextActiveParticle[i])
    {
        last_particle = i;
        count++;

        if((count % 1000) == 0 && NextActiveParticle[i] >= 0)
        {
            last_list[nseg] = last_particle;
            nseg++;
            first_list[nseg] = NextActiveParticle[i];
        }
    }

    last_list[nseg] = last_particle;
    nseg++;

    permut = (struct permut_data *) mymalloc("permut", nseg * sizeof(struct permut_data));

    for(i = 0; i < nseg; i++)
    {
        permut[i].rnd = get_random_number(i);
        permut[i].seg = i;
    }

    qsort(permut, nseg, sizeof(struct permut_data), permut_data_compare);

    FirstActiveParticle = first_list[permut[0].seg];

    for(i = 0; i < nseg; i++)
    {
        if(i == nseg - 1)
        {
            if(last_list[permut[i].seg] >= 0)
                NextActiveParticle[last_list[permut[i].seg]] = -1;
        }
        else
            NextActiveParticle[last_list[permut[i].seg]] = first_list[permut[i + 1].seg];
    }

    for(i = FirstActiveParticle, idsum_new = 0; i >= 0; i = NextActiveParticle[i])
        idsum_new += P[i].ID;

    if(idsum_old != idsum_new)
        endrun(12199991);

    myfree(permut);
    myfree(last_list);
    myfree(first_list);
}

int permut_data_compare(const void *a, const void *b)
{
    if(((struct permut_data *) a)->rnd < (((struct permut_data *) b)->rnd))
        return -1;

    if(((struct permut_data *) a)->rnd > (((struct permut_data *) b)->rnd))
        return +1;

    return 0;
}

#endif


int ShouldWeDoDynamicUpdate(void)
{
    int n, num, dt_bin, ti_next_for_bin, ti_next_kick, ti_next_kick_global;
    int64_t numforces;


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

    /* count the particles that will be active */
    for(n = 1, num = TimeBinCount[0]; n < TIMEBINS; n++)
    {
        dt_bin = (1 << n);

        if((ti_next_kick_global % dt_bin) == 0)
            num += TimeBinCount[n];
    }

    sumup_large_ints(1, &num, &numforces);

    if(ThisTask == 0)
        printf("I'm guessing %d%09d particles to be active in the next step\n",
                (int) (numforces / 1000000000), (int) (numforces % 1000000000));

    if((All.NumForcesSinceLastDomainDecomp + numforces) >= All.TreeDomainUpdateFrequency * All.TotNumPart)
        return 0;
    else
        return 1;
}



/*! this function returns the next output time that is equal or larger to
 *  ti_curr
 */
int find_next_outputtime(int ti_curr)
{
    int i, ti, ti_next, iter = 0;
    double next, time;

    DumpFlag = 1;
    ti_next = -1;


    if(All.OutputListOn)
    {
        for(i = 0; i < All.OutputListLength; i++)
        {
            time = All.OutputListTimes[i];

            if(time >= All.TimeBegin && time <= All.TimeMax)
            {
                if(All.ComovingIntegrationOn)
                    ti = (int) (log(time / All.TimeBegin) / All.Timebase_interval);
                else
                    ti = (int) ((time - All.TimeBegin) / All.Timebase_interval);

                if(ti >= ti_curr)
                {
                    if(ti_next == -1)
                    {
                        ti_next = ti;
                        DumpFlag = All.OutputListFlag[i];
                    }

                    if(ti_next > ti)
                    {
                        ti_next = ti;
                        DumpFlag = All.OutputListFlag[i];
                    }
                }
            }
        }
    }
    else
    {
        if(All.ComovingIntegrationOn)
        {
            if(All.TimeBetSnapshot <= 1.0)
            {
                printf("TimeBetSnapshot > 1.0 required for your simulation.\n");
                endrun(13123);
            }
        }
        else
        {
            if(All.TimeBetSnapshot <= 0.0)
            {
                printf("TimeBetSnapshot > 0.0 required for your simulation.\n");
                endrun(13123);
            }
        }
#ifdef SNAP_SET_TG
        if(All.ComovingIntegrationOn)
            time = All.TimeBegin * All.TimeBetSnapshot;
        else
            time = All.TimeBegin + All.TimeBetSnapshot;
#else
        time = All.TimeOfFirstSnapshot;
#endif
        iter = 0;

        while(time < All.TimeBegin)
        {
            if(All.ComovingIntegrationOn)
                time *= All.TimeBetSnapshot;
            else
                time += All.TimeBetSnapshot;

            iter++;

            if(iter > 1000000)
            {
                printf("Can't determine next output time.\n");
                endrun(110);
            }
        }
#ifdef DYN_TIME_BASED_SNAPS
        double rho, t_ff, ff_frac = 0.2;

        rho = nh_glob_max * PROTONMASS / HYDROGEN_MASSFRAC;

        t_ff = sqrt(3.0 * M_PI / 32.0 / GRAVITY / rho);

        if(All.ComovingIntegrationOn)
        {
            time = pow(3.0 / 2.0 * HUBBLE * All.HubbleParam * sqrt(All.Omega0) * ff_frac * t_ff + pow(All.Time, 3.0 / 2.0), 2.0 / 3.0);

            ti_next = (int) (log(time / All.TimeBegin) / All.Timebase_interval);
        }
        else
        {
            time = All.Time + ff_frac * t_ff / All.UnitTime_in_s;

            ti_next = (int) ((time - All.TimeBegin) / All.Timebase_interval);
        }
#else
        while(time <= All.TimeMax)
        {
            if(All.ComovingIntegrationOn)
                ti = (int) (log(time / All.TimeBegin) / All.Timebase_interval);
            else
                ti = (int) ((time - All.TimeBegin) / All.Timebase_interval);

            if(ti >= ti_curr)
            {
                ti_next = ti;
                break;
            }

            if(All.ComovingIntegrationOn)
                time *= All.TimeBetSnapshot;
            else
                time += All.TimeBetSnapshot;

            iter++;

            if(iter > 1000000)
            {
                printf("Can't determine next output time.\n");
                endrun(111);
            }
        }
#endif
    }


    if(ti_next == -1)
    {
        ti_next = 2 * TIMEBASE;	/* this will prevent any further output */

        if(ThisTask == 0)
            printf("\nThere is no valid time for a further snapshot file.\n");
    }
    else
    {
        if(All.ComovingIntegrationOn)
            next = All.TimeBegin * exp(ti_next * All.Timebase_interval);
        else
            next = All.TimeBegin + ti_next * All.Timebase_interval;

        if(ThisTask == 0)
            printf("\nSetting next time for snapshot file to Time_next= %g  (DumpFlag=%d)\n\n", next, DumpFlag);

    }

    return ti_next;
}




/*! This routine writes one line for every timestep to two log-files.
 * In FdInfo, we just list the timesteps that have been done, while in
 * FdCPU the cumulative cpu-time consumption in various parts of the
 * code is stored.
 */
void every_timestep_stuff(void)
{
    double z;
    int i;
    int64_t tot, tot_sph;
    int64_t tot_count[TIMEBINS];
    int64_t tot_count_sph[TIMEBINS];

    sumup_large_ints(TIMEBINS, TimeBinCount, tot_count);
    sumup_large_ints(TIMEBINS, TimeBinCountSph, tot_count_sph);

    if(ThisTask == 0)
    {
        if(All.ComovingIntegrationOn)
        {
            z = 1.0 / (All.Time) - 1;
            fprintf(FdInfo, "\nBegin Step %d, Time: %g, Redshift: %g, Nf = %d%09d, Systemstep: %g, Dloga: %g\n",
                    All.NumCurrentTiStep, All.Time, z,
                    (int) (GlobNumForceUpdate / 1000000000), (int) (GlobNumForceUpdate % 1000000000),
                    All.TimeStep, log(All.Time) - log(All.Time - All.TimeStep));
            printf("\nBegin Step %d, Time: %g, Redshift: %g, Systemstep: %g, Dloga: %g\n", All.NumCurrentTiStep,
                    All.Time, z, All.TimeStep, log(All.Time) - log(All.Time - All.TimeStep));

#if defined (CHEMISTRY) || defined (UM_CHEMISTRY)
            printf("Abundances  elec: %g, HM: %g, H2I: %g, H2II: %g\n",
                    SPHP(1).elec, SPHP(1).HM, SPHP(1).H2I, SPHP(1).H2II);
            printf("Abundances  HI: %g, HII: %g, HeI: %g, HeII: %g, HeIII: %g\n",
                    SPHP(1).HI, SPHP(1).HII, SPHP(1).HeI, SPHP(1).HeII, SPHP(1).HeIII);
#endif

#if defined (UM_CHEMISTRY) && defined (UM_HD_COOLING)
            printf("Abundances HD: %g,  DI: %g,  DII: %g\n", SPHP(1).HD, SPHP(1).DI, SPHP(1).DII);
            printf("Abundances HeHII: %g",SPHP(1).HeHII);
#endif

            fflush(FdInfo);
        }
        else
        {
            fprintf(FdInfo, "\nBegin Step %d, Time: %g, Nf = %d%09d, Systemstep: %g\n", All.NumCurrentTiStep,
                    All.Time, (int) (GlobNumForceUpdate / 1000000000), (int) (GlobNumForceUpdate % 1000000000),
                    All.TimeStep);
            printf("\nBegin Step %d, Time: %g, Systemstep: %g\n", All.NumCurrentTiStep, All.Time, All.TimeStep);
            fflush(FdInfo);
        }

        printf("Occupied timebins: non-sph         sph       dt\n");
        for(i = TIMEBINS - 1, tot = tot_sph = 0; i >= 0; i--)
            if(tot_count_sph[i] > 0 || tot_count[i] > 0)
            {
                printf(" %c  bin=%2d     %2d%09d %2d%09d   %6g\n",
                        TimeBinActive[i] ? 'X' : ' ',
                        i,
                        (int) ((tot_count[i] - tot_count_sph[i]) / 1000000000),
                        (int) ((tot_count[i] - tot_count_sph[i]) % 1000000000),
                        (int) (tot_count_sph[i] / 1000000000), (int) (tot_count_sph[i] % 1000000000),
                        i > 0 ? (1 << i) * All.Timebase_interval : 0.0);
                if(TimeBinActive[i])
                {
                    tot += tot_count[i];
                    tot_sph += tot_count_sph[i];
                }
            }
        printf("               -----------------------------------\n");
#ifdef PMGRID
        if(All.PM_Ti_endstep == All.Ti_Current)
            printf("PM-Step. Total:%2d%09d %2d%09d    Sum:%2d%09d\n",
                    (int) ((tot - tot_sph) / 1000000000), (int) ((tot - tot_sph) % 1000000000),
                    (int) (tot_sph / 1000000000), (int) (tot_sph % 1000000000),
                    (int) (tot / 1000000000), (int) (tot % 1000000000));
        else
#endif
            printf("Total active:  %2d%09d %2d%09d    Sum:%2d%09d\n",
                    (int) ((tot - tot_sph) / 1000000000), (int) ((tot - tot_sph) % 1000000000),
                    (int) (tot_sph / 1000000000), (int) (tot_sph % 1000000000),
                    (int) (tot / 1000000000), (int) (tot % 1000000000));

#ifdef CHEMISTRY
        printf("Abundances elec: %g, HM: %g, H2I: %g, H2II: %g\n",
                SPHP(1).elec, SPHP(1).HM, SPHP(1).H2I, SPHP(1).H2II);
#endif

#ifdef XXLINFO
        if(Flag_FullStep == 1)
        {
            fprintf(FdXXL, "%d %g ", All.NumCurrentTiStep, All.Time);
#ifdef MAGNETIC
            fprintf(FdXXL, "%e ", MeanB);
#ifdef TRACEDIVB
            fprintf(FdXXL, "%e ", MaxDivB);
#endif
#endif
#ifdef TIME_DEP_ART_VISC
            fprintf(FdXXL, "%f ", MeanAlpha);
#endif
            fprintf(FdXXL, "\n");
            fflush(FdXXL);
        }
#endif

#ifdef DARKENERGY
        if(All.ComovingIntegrationOn == 1)
        {
            double hubble_a;

            hubble_a = hubble_function(All.Time);
            fprintf(FdDE, "%d %g %e ", All.NumCurrentTiStep, All.Time, hubble_a);
#ifndef TIMEDEPDE
            fprintf(FdDE, "%e ", All.DarkEnergyParam);
#else
            fprintf(FdDE, "%e %e ", get_wa(All.Time), DarkEnergy_a(All.Time));
#endif
#ifdef TIMEDEPGRAV
            fprintf(FdDE, "%e %e", dHfak(All.Time), dGfak(All.Time));
#endif
            fprintf(FdDE, "\n");
            fflush(FdDE);
        }
#endif
    }

    set_random_numbers();
}



void write_cpu_log(void)
{
    double max_CPU_Step[CPU_PARTS], avg_CPU_Step[CPU_PARTS], t0, t1, tsum;
    int i;

    CPU_Step[CPU_MISC] += measure_time();

    All.Cadj_Cpu += CPU_Step[CPU_TREEWALK1] + CPU_Step[CPU_TREEWALK2];

    for(i = 1, CPU_Step[0] = 0; i < CPU_PARTS; i++)
        CPU_Step[0] += CPU_Step[i];

    MPI_Reduce(CPU_Step, max_CPU_Step, CPU_PARTS, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(CPU_Step, avg_CPU_Step, CPU_PARTS, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    int64_t totBlockedPD = -1, totBlockedND = -1;
    int64_t totTotalPD = -1, totTotalND = -1;

#ifdef _OPENMP
    MPI_Reduce(&BlockedParticleDrifts, &totBlockedPD, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&BlockedNodeDrifts, &totBlockedND, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&TotalParticleDrifts, &totTotalPD, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&TotalNodeDrifts, &totTotalND, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
#endif

    if(ThisTask == 0)
    {
        for(i = 0; i < CPU_PARTS; i++)
            avg_CPU_Step[i] /= NTask;

        put_symbol(0.0, 1.0, '#');

        for(i = 1, tsum = 0.0; i < CPU_PARTS; i++)
        {
            if(max_CPU_Step[i] > 0)
            {
                t0 = tsum;
                t1 = tsum + avg_CPU_Step[i] * (avg_CPU_Step[i] / max_CPU_Step[i]);
                put_symbol(t0 / avg_CPU_Step[0], t1 / avg_CPU_Step[0], CPU_Symbol[i]);
                tsum += t1 - t0;

                t0 = tsum;
                t1 = tsum + avg_CPU_Step[i] * ((max_CPU_Step[i] - avg_CPU_Step[i]) / max_CPU_Step[i]);
                put_symbol(t0 / avg_CPU_Step[0], t1 / avg_CPU_Step[0], CPU_SymbolImbalance[i]);
                tsum += t1 - t0;
            }
        }

        put_symbol(tsum / max_CPU_Step[0], 1.0, '-');

        fprintf(FdBalance, "Step=%7d  sec=%10.3f  Nf=%2d%09d  %s\n", All.NumCurrentTiStep, max_CPU_Step[0],
                (int) (GlobNumForceUpdate / 1000000000), (int) (GlobNumForceUpdate % 1000000000), CPU_String);
        fflush(FdBalance);
    }

    CPUThisRun += CPU_Step[0];

    for(i = 0; i < CPU_PARTS; i++)
        CPU_Step[i] = 0;

    if(ThisTask == 0)
    {
        for(i = 0; i < CPU_PARTS; i++)
            All.CPU_Sum[i] += avg_CPU_Step[i];

        fprintf(FdCPU, "Step %d, Time: %g, MPIs: %d Threads %d\n", All.NumCurrentTiStep, All.Time, NTask, All.NumThreads);
#ifdef _OPENMP
        fprintf(FdCPU, "Blocked Drifts (Particle Node): %ld %ld\n", totBlockedPD, totBlockedND);
        fprintf(FdCPU, "Total Drifts (Particle Node): %ld %ld\n", totTotalPD, totTotalND);
#endif
        fprintf(FdCPU,
                "total         %10.2f  %5.1f%%\n"
                "treegrav      %10.2f  %5.1f%%\n"
                "   treebuild  %10.2f  %5.1f%%\n"
                "   treeupdate %10.2f  %5.1f%%\n"
                "   treewalk   %10.2f  %5.1f%%\n"
                "   treecomm   %10.2f  %5.1f%%\n"
                "   treeimbal  %10.2f  %5.1f%%\n"
                "pmgrav        %10.2f  %5.1f%%\n"
                "sph           %10.2f  %5.1f%%\n"
                "   density    %10.2f  %5.1f%%\n"
                "   denscomm   %10.2f  %5.1f%%\n"
                "   densimbal  %10.2f  %5.1f%%\n"
                "   hydrofrc   %10.2f  %5.1f%%\n"
                "   hydcomm    %10.2f  %5.1f%%\n"
                "   hydmisc    %10.2f  %5.1f%%\n"
                "   hydnetwork %10.2f  %5.1f%%\n"
                "   hydimbal   %10.2f  %5.1f%%\n"
                "   hmaxupdate %10.2f  %5.1f%%\n"
                "domain        %10.2f  %5.1f%%\n"
                "potential     %10.2f  %5.1f%%\n"
                "predict       %10.2f  %5.1f%%\n"
                "kicks         %10.2f  %5.1f%%\n"
                "i/o           %10.2f  %5.1f%%\n"
                "peano         %10.2f  %5.1f%%\n"
                "sfrcool       %10.2f  %5.1f%%\n"
                "blackholes    %10.2f  %5.1f%%\n"
                "fof/subfind   %10.2f  %5.1f%%\n"
                "smoothing     %10.2f  %5.1f%%\n"
                "hotngbs       %10.2f  %5.1f%%\n"
                "weights_hot   %10.2f  %5.1f%%\n"
                "enrich_hot    %10.2f  %5.1f%%\n"
                "weights_cold  %10.2f  %5.1f%%\n"
                "enrich_cold   %10.2f  %5.1f%%\n"
                "cs_misc       %10.2f  %5.1f%%\n"
                "misc          %10.2f  %5.1f%%\n",
            All.CPU_Sum[CPU_ALL], 100.0,
            All.CPU_Sum[CPU_TREEWALK1] + All.CPU_Sum[CPU_TREEWALK2]
                + All.CPU_Sum[CPU_TREESEND] + All.CPU_Sum[CPU_TREERECV]
                + All.CPU_Sum[CPU_TREEWAIT1] + All.CPU_Sum[CPU_TREEWAIT2]
                + All.CPU_Sum[CPU_TREEBUILD] + All.CPU_Sum[CPU_TREEUPDATE]
                + All.CPU_Sum[CPU_TREEMISC],
            (All.CPU_Sum[CPU_TREEWALK1] + All.CPU_Sum[CPU_TREEWALK2]
             + All.CPU_Sum[CPU_TREESEND] + All.CPU_Sum[CPU_TREERECV]
             + All.CPU_Sum[CPU_TREEWAIT1] + All.CPU_Sum[CPU_TREEWAIT2]
             + All.CPU_Sum[CPU_TREEBUILD] + All.CPU_Sum[CPU_TREEUPDATE]
             + All.CPU_Sum[CPU_TREEMISC]) / All.CPU_Sum[CPU_ALL] * 100,
            All.CPU_Sum[CPU_TREEBUILD],
            (All.CPU_Sum[CPU_TREEBUILD]) / All.CPU_Sum[CPU_ALL] * 100,
            All.CPU_Sum[CPU_TREEUPDATE],
            (All.CPU_Sum[CPU_TREEUPDATE]) / All.CPU_Sum[CPU_ALL] * 100,
            All.CPU_Sum[CPU_TREEWALK1] + All.CPU_Sum[CPU_TREEWALK2],
            (All.CPU_Sum[CPU_TREEWALK1] + All.CPU_Sum[CPU_TREEWALK2]) / All.CPU_Sum[CPU_ALL] * 100,
            All.CPU_Sum[CPU_TREESEND] + All.CPU_Sum[CPU_TREERECV],
            (All.CPU_Sum[CPU_TREESEND] + All.CPU_Sum[CPU_TREERECV]) / All.CPU_Sum[CPU_ALL] * 100,
            All.CPU_Sum[CPU_TREEWAIT1] + All.CPU_Sum[CPU_TREEWAIT2],
            (All.CPU_Sum[CPU_TREEWAIT1] + All.CPU_Sum[CPU_TREEWAIT2]) / All.CPU_Sum[CPU_ALL] * 100,
            All.CPU_Sum[CPU_MESH],
            (All.CPU_Sum[CPU_MESH]) / All.CPU_Sum[CPU_ALL] * 100,
            All.CPU_Sum[CPU_DENSCOMPUTE] + All.CPU_Sum[CPU_DENSWAIT]
                + All.CPU_Sum[CPU_DENSCOMM] + All.CPU_Sum[CPU_DENSMISC]
                + All.CPU_Sum[CPU_HYDCOMPUTE] + All.CPU_Sum[CPU_HYDWAIT] + All.CPU_Sum[CPU_TREEHMAXUPDATE]
                + All.CPU_Sum[CPU_HYDCOMM] + All.CPU_Sum[CPU_HYDMISC] + All.CPU_Sum[CPU_HYDNETWORK],
            (All.CPU_Sum[CPU_DENSCOMPUTE] + All.CPU_Sum[CPU_DENSWAIT]
             + All.CPU_Sum[CPU_DENSCOMM] + All.CPU_Sum[CPU_DENSMISC]
             + All.CPU_Sum[CPU_HYDCOMPUTE] + All.CPU_Sum[CPU_HYDWAIT] + All.CPU_Sum[CPU_TREEHMAXUPDATE]
             + All.CPU_Sum[CPU_HYDCOMM] + All.CPU_Sum[CPU_HYDMISC] +
             All.CPU_Sum[CPU_HYDNETWORK]) / All.CPU_Sum[CPU_ALL] * 100, All.CPU_Sum[CPU_DENSCOMPUTE],
            (All.CPU_Sum[CPU_DENSCOMPUTE]) / All.CPU_Sum[CPU_ALL] * 100, All.CPU_Sum[CPU_DENSCOMM],
            (All.CPU_Sum[CPU_DENSCOMM]) / All.CPU_Sum[CPU_ALL] * 100, All.CPU_Sum[CPU_DENSWAIT],
            (All.CPU_Sum[CPU_DENSWAIT]) / All.CPU_Sum[CPU_ALL] * 100, All.CPU_Sum[CPU_HYDCOMPUTE],
            (All.CPU_Sum[CPU_HYDCOMPUTE]) / All.CPU_Sum[CPU_ALL] * 100, All.CPU_Sum[CPU_HYDCOMM],
            (All.CPU_Sum[CPU_HYDCOMM]) / All.CPU_Sum[CPU_ALL] * 100, All.CPU_Sum[CPU_HYDMISC],
            (All.CPU_Sum[CPU_HYDMISC]) / All.CPU_Sum[CPU_ALL] * 100, All.CPU_Sum[CPU_HYDNETWORK],
            (All.CPU_Sum[CPU_HYDNETWORK]) / All.CPU_Sum[CPU_ALL] * 100, All.CPU_Sum[CPU_HYDWAIT],
            (All.CPU_Sum[CPU_HYDWAIT]) / All.CPU_Sum[CPU_ALL] * 100, All.CPU_Sum[CPU_TREEHMAXUPDATE],
            (All.CPU_Sum[CPU_TREEHMAXUPDATE]) / All.CPU_Sum[CPU_ALL] * 100, All.CPU_Sum[CPU_DOMAIN],
            (All.CPU_Sum[CPU_DOMAIN]) / All.CPU_Sum[CPU_ALL] * 100, All.CPU_Sum[CPU_POTENTIAL],
            (All.CPU_Sum[CPU_POTENTIAL]) / All.CPU_Sum[CPU_ALL] * 100, All.CPU_Sum[CPU_DRIFT],
            (All.CPU_Sum[CPU_DRIFT]) / All.CPU_Sum[CPU_ALL] * 100, All.CPU_Sum[CPU_TIMELINE],
            (All.CPU_Sum[CPU_TIMELINE]) / All.CPU_Sum[CPU_ALL] * 100, All.CPU_Sum[CPU_SNAPSHOT],
            (All.CPU_Sum[CPU_SNAPSHOT]) / All.CPU_Sum[CPU_ALL] * 100, All.CPU_Sum[CPU_PEANO],
            (All.CPU_Sum[CPU_PEANO]) / All.CPU_Sum[CPU_ALL] * 100, All.CPU_Sum[CPU_COOLINGSFR],
            (All.CPU_Sum[CPU_COOLINGSFR]) / All.CPU_Sum[CPU_ALL] * 100, All.CPU_Sum[CPU_BLACKHOLES],
            (All.CPU_Sum[CPU_BLACKHOLES]) / All.CPU_Sum[CPU_ALL] * 100, All.CPU_Sum[CPU_FOF],
            (All.CPU_Sum[CPU_FOF]) / All.CPU_Sum[CPU_ALL] * 100,
            All.CPU_Sum[CPU_SMTHCOMPUTE] + All.CPU_Sum[CPU_SMTHWAIT] + All.CPU_Sum[CPU_SMTHCOMM] +
                All.CPU_Sum[CPU_SMTHMISC],
            (All.CPU_Sum[CPU_SMTHCOMPUTE] + All.CPU_Sum[CPU_SMTHWAIT] + All.CPU_Sum[CPU_SMTHCOMM] +
             All.CPU_Sum[CPU_SMTHMISC]) / All.CPU_Sum[CPU_ALL] * 100, All.CPU_Sum[CPU_HOTNGBS],
            (All.CPU_Sum[CPU_HOTNGBS]) / All.CPU_Sum[CPU_ALL] * 100, All.CPU_Sum[CPU_WEIGHTS_HOT],
            (All.CPU_Sum[CPU_WEIGHTS_HOT]) / All.CPU_Sum[CPU_ALL] * 100, All.CPU_Sum[CPU_ENRICH_HOT],
            (All.CPU_Sum[CPU_ENRICH_HOT]) / All.CPU_Sum[CPU_ALL] * 100, All.CPU_Sum[CPU_WEIGHTS_COLD],
            (All.CPU_Sum[CPU_WEIGHTS_COLD]) / All.CPU_Sum[CPU_ALL] * 100, All.CPU_Sum[CPU_ENRICH_COLD],
            (All.CPU_Sum[CPU_ENRICH_COLD]) / All.CPU_Sum[CPU_ALL] * 100, All.CPU_Sum[CPU_CSMISC],
            (All.CPU_Sum[CPU_CSMISC]) / All.CPU_Sum[CPU_ALL] * 100, All.CPU_Sum[CPU_MISC],
            (All.CPU_Sum[CPU_MISC]) / All.CPU_Sum[CPU_ALL] * 100);
        fprintf(FdCPU, "\n");
        fflush(FdCPU);
    }
}


void put_symbol(double t0, double t1, char c)
{
    int i, j;

    i = (int) (t0 * CPU_STRING_LEN + 0.5);
    j = (int) (t1 * CPU_STRING_LEN);

    if(i < 0)
        i = 0;
    if(j < 0)
        j = 0;
    if(i >= CPU_STRING_LEN)
        i = CPU_STRING_LEN;
    if(j >= CPU_STRING_LEN)
        j = CPU_STRING_LEN;

    while(i <= j)
        CPU_String[i++] = c;

    CPU_String[CPU_STRING_LEN] = 0;
}


/*! This routine first calls a computation of various global
 * quantities of the particle distribution, and then writes some
 * statistics about the energies in the various particle components to
 * the file FdEnergy.
 */
void energy_statistics(void)
{
    compute_global_quantities_of_system();

    if(ThisTask == 0)
    {
        fprintf(FdEnergy,
                "%g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g\n",
                All.Time, SysState.EnergyInt, SysState.EnergyPot, SysState.EnergyKin, SysState.EnergyIntComp[0],
                SysState.EnergyPotComp[0], SysState.EnergyKinComp[0], SysState.EnergyIntComp[1],
                SysState.EnergyPotComp[1], SysState.EnergyKinComp[1], SysState.EnergyIntComp[2],
                SysState.EnergyPotComp[2], SysState.EnergyKinComp[2], SysState.EnergyIntComp[3],
                SysState.EnergyPotComp[3], SysState.EnergyKinComp[3], SysState.EnergyIntComp[4],
                SysState.EnergyPotComp[4], SysState.EnergyKinComp[4], SysState.EnergyIntComp[5],
                SysState.EnergyPotComp[5], SysState.EnergyKinComp[5], SysState.MassComp[0],
                SysState.MassComp[1], SysState.MassComp[2], SysState.MassComp[3], SysState.MassComp[4],
                SysState.MassComp[5]);

        fflush(FdEnergy);
    }
}
