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
static int human_interaction();
int stopflag = 0;
void run(void)
{
    FILE *fd;

#if defined(RADIATIVE_RATES) || defined(RADIATION)
    int ifunc;
#endif
#ifdef OLD_RESTART
    char buf[200], stopfname[200], contfname[200];


    sprintf(stopfname, "%sstop", All.OutputDir);
    sprintf(contfname, "%scont", All.OutputDir);
    unlink(contfname);

#endif
    walltime_measure("/Misc");

#ifdef DENSITY_BASED_SNAPS
    All.nh_next = 10.0;
#endif


    write_cpu_log();		/* produce some CPU usage info */

    do				/* main loop */
    {

        find_next_sync_point_and_drift();	/* find next synchronization point and drift particles to this time.
                                             * If needed, this function will also write an output file
                                             * at the desired time.
                                             */

        if(stopflag == 1 || stopflag == 2) {
            /* OK snapshot file is written, lets quit */
            return;
        }
        every_timestep_stuff();	/* write some info to log-files */

#if defined(RADIATIVE_RATES) || defined(RADIATION)
        ifunc = init_rad(All.Time);
#endif

#ifdef COOLING
        IonizeParams();		/* set UV background for the current time */
#endif


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

        stopflag = human_interaction();
        if(stopflag != 0) {
            All.Ti_nextoutput = All.Ti_Current;
            /* next loop will write a new snapshot file */
        }
#ifdef OLD_RESTART
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
            if(All.CT.ElapsedTime > 0.85 * All.TimeLimitCPU)
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

            return;
        }

        /* is it time to write a regular restart-file? (for security) */
        if(ThisTask == 0)
        {
            if((All.CT.ElapsedTime - All.TimeLastRestartFile) >= All.CpuTimeBetRestartFile)
            {
                All.TimeLastRestartFile = All.CT.ElapsedTime;
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
#endif /* old restart*/
        report_memory_usage("RUN");
    }
    while(All.Ti_Current < TIMEBASE && All.Time <= All.TimeMax);
#ifndef SNAP_SET_TG
    restart(0);

    savepositions(All.SnapshotFileCount++, 0);
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
static int human_interaction() {
        /* Check whether we need to interrupt the run */
    int stopflag = 0;
    char stopfname[4096], contfname[4096];
    char restartfname[4096];
    char ioctlfname[4096];

    sprintf(stopfname, "%s/stop", All.OutputDir);
    sprintf(restartfname, "%s/restart", All.OutputDir);
    sprintf(contfname, "%s/cont", All.OutputDir);
    sprintf(ioctlfname, "%s/ioctl", All.OutputDir);

    if(ThisTask == 0)
    {
        FILE * fd;
        if((fd = fopen(ioctlfname, "r"))) {
             /* there is an ioctl file, parse it and update
              * All.NumFilesPerSnapshot
              * All.NumFilesPerPIG
              * All.NumWritersPerSnapshot
              * All.NumWritersPerPig
              */
            size_t n = 0;
            char * line = NULL;
            int NumFilesPerSnapshot = -1;
            int NumFilesPerPIG = -1;
            int NumWritersPerSnapshot = -1;
            int NumWritersPerPIG = -1;
            while(-1 != getline(&line, &n, fd)) {
                sscanf(line, "NumFilesPerSnapshot %d", &NumFilesPerSnapshot);
                sscanf(line, "NumWritersPerPIG %d", &NumWritersPerPIG);
                sscanf(line, "NumFilesPerPIG %d", &NumFilesPerPIG);
                sscanf(line, "NumWritersPerSnapshot %d", &NumWritersPerSnapshot);
            }
            free(line);
            int changed = 0;
            if(NumFilesPerSnapshot > 0 && 
                NumFilesPerSnapshot != All.NumFilesPerSnapshot) {
                All.NumFilesPerSnapshot = NumFilesPerSnapshot;
                changed = 1;
            }
            if(NumWritersPerSnapshot > 0) {
                if(All.NumWritersPerSnapshot > NTask) {
                    All.NumWritersPerSnapshot = NTask;
                }
                if(NumWritersPerSnapshot != All.NumWritersPerSnapshot) {
                    All.NumWritersPerSnapshot = NumWritersPerSnapshot;
                    changed = 1;
                }
            }
            if(NumFilesPerPIG > 0 && 
                NumFilesPerPIG != All.NumFilesPerPIG) {
                All.NumFilesPerPIG = NumFilesPerPIG;
                changed = 1;
            }
            if(NumWritersPerPIG > 0) {
                if(All.NumWritersPerPIG > NTask) {
                    All.NumWritersPerPIG = NTask;
                }
                if(NumWritersPerPIG != All.NumWritersPerPIG) {
                    All.NumWritersPerPIG = NumWritersPerPIG;
                    changed = 1;
                }
            }
            if(changed) {
                printf("New IO parameter recieved from %s:\n"
                       "NumFilesPerSnapshot %d\n"
                       "NumFilesPerPIG      %d\n"
                       "NumWritersPerSnapshot %d\n"
                       "NumWritersPerPIG     %d\n",
                    ioctlfname,
                    All.NumFilesPerSnapshot,
                    All.NumFilesPerPIG,
                    All.NumWritersPerSnapshot,
                    All.NumWritersPerPIG);
            }
            fclose(fd);
        }
        /* Is the stop-file present? If yes, interrupt the run. */
        if((fd = fopen(stopfname, "r")))
        {
            printf("human controlled stopping.\n");
            fclose(fd);
            stopflag = 1;
            unlink(stopfname);
        }

        /* are we running out of CPU-time ? If yes, interrupt run. */
        if(All.CT.ElapsedTime > 0.85 * All.TimeLimitCPU) {
            printf("reaching time-limit. stopping.\n");
            stopflag = 2;
        }
        if((fd = fopen(restartfname, "r")))
        {
            printf("human controlled snapshot.\n");
            fclose(fd);
            stopflag = 3;
            unlink(restartfname);
        }
        if((All.CT.ElapsedTime - All.TimeLastRestartFile) >= All.CpuTimeBetRestartFile) {
            All.TimeLastRestartFile = All.CT.ElapsedTime;
            printf("time to write a snapshot for restarting\n");
            stopflag = 3;
        }
    }

    MPI_Bcast(&stopflag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    return stopflag;
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
        savepositions(All.SnapshotFileCount++, 0);
#else

#ifdef DENSITY_BASED_SNAPS
    if(nh_glob_max > All.nh_next)
    {
        All.nh_next *= pow(10.0, 0.25);

        if(ThisTask == 0)
            printf("nh_next = %g\n", All.nh_next);

        savepositions(All.SnapshotFileCount++, 0);
    }
#else
    while(ti_next_kick_global >= All.Ti_nextoutput && All.Ti_nextoutput >= 0)
    {
        All.Ti_Current = All.Ti_nextoutput;

        double nexttime;
        if(All.ComovingIntegrationOn)
            nexttime = All.TimeBegin * exp(All.Ti_Current * All.Timebase_interval);
        else
            nexttime = All.TimeBegin + All.Ti_Current * All.Timebase_interval;

        set_global_time(nexttime);

#ifdef TIMEDEPGRAV
        All.G = All.Gini * dGfak(All.Time);
#endif

        move_particles(All.Ti_nextoutput);

        savepositions(All.SnapshotFileCount++, stopflag);	/* write snapshot file */

        All.Ti_nextoutput = find_next_outputtime(All.Ti_nextoutput + 1);
    }
#endif
#endif

    All.Ti_Current = ti_next_kick_global;

    double nexttime;
    if(All.ComovingIntegrationOn)
        nexttime = All.TimeBegin * exp(All.Ti_Current * All.Timebase_interval);
    else
        nexttime = All.TimeBegin + All.Ti_Current * All.Timebase_interval;

    set_global_time(nexttime);

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

    if(GlobNumForceUpdate >= All.TotNumPart)
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

    walltime_measure("/Misc");
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

    walltime_measure("/Drift");
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

    /* let's update Tot counts in one place tot variables;
     * at this point there can still be holes in SphP
     * because rearrange_particle_squence is not called yet.
     * but anywaysTotN_sph variables are not well defined and
     * not used any places but printing.
     *
     * we shall just say they we sync these variables right after gravity
     * calculation in every timestep.
     * */

    sumup_large_ints(1, &NumPart, &All.TotNumPart);
    sumup_large_ints(1, &N_sph, &All.TotN_sph);
    sumup_large_ints(1, &N_bh, &All.TotN_bh);

    if(ThisTask == 0)
    {
        char buf[1024];
        
        char extra[1024] = {0};
#ifdef PETAPM
        if(All.PM_Ti_endstep == All.Ti_Current)
            strcat(extra, "PM-Step");
#endif

        if(All.ComovingIntegrationOn)
        {
            z = 1.0 / (All.Time) - 1;
            sprintf(buf, "\nBegin Step %d, Time: %g, Redshift: %g, Nf = %014ld, Systemstep: %g, Dloga: %g, status: %s\n",
                        All.NumCurrentTiStep, All.Time, z,
                        GlobNumForceUpdate,
                        All.TimeStep, log(All.Time) - log(All.Time - All.TimeStep), 
                        extra);
        } else {
            sprintf(buf , "\nBegin Step %d, Time: %g, Nf = %014ld, Systemstep: %g, status:%s\n", All.NumCurrentTiStep,
                    All.Time, GlobNumForceUpdate,
                    All.TimeStep,
                    extra);
        }
        fprintf(FdInfo, "%s", buf);
        printf("%s", buf);

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

        printf("TotNumPart: %014ld SPH %014ld BH %014ld\n",
                All.TotNumPart, All.TotN_sph, All.TotN_bh);
        printf("Occupied timebins: non-sph         sph       dt\n");
        for(i = TIMEBINS - 1, tot = tot_sph = 0; i >= 0; i--)
            if(tot_count_sph[i] > 0 || tot_count[i] > 0)
            {
                printf(" %c  bin=%2d     %014ld %014ld   %6g\n",
                        TimeBinActive[i] ? 'X' : ' ',
                        i,
                        (tot_count[i] - tot_count_sph[i]),
                        tot_count_sph[i], 
                        i > 0 ? (1 << i) * All.Timebase_interval : 0.0);
                if(TimeBinActive[i])
                {
                    tot += tot_count[i];
                    tot_sph += tot_count_sph[i];
                }
            }
        printf("               -----------------------------------\n");
        printf("Total:%014ld %014ld    Sum:%014ld\n",
            (tot - tot_sph),
            (tot_sph), 
            (tot));

#ifdef CHEMISTRY
        printf("Abundances elec: %g, HM: %g, H2I: %g, H2II: %g\n",
                SPHP(1).elec, SPHP(1).HM, SPHP(1).H2I, SPHP(1).H2II);
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
    All.Cadj_Cpu += walltime_get_time("/Tree/Walk1") + walltime_get_time("/Tree/Walk2");

    int64_t totBlockedPD = -1, totBlockedND = -1;
    int64_t totTotalPD = -1, totTotalND = -1;

#ifdef _OPENMP
    MPI_Reduce(&BlockedParticleDrifts, &totBlockedPD, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&BlockedNodeDrifts, &totBlockedND, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&TotalParticleDrifts, &totTotalPD, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&TotalNodeDrifts, &totTotalND, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
#endif

    walltime_summary(0, MPI_COMM_WORLD);

    if(ThisTask == 0)
    {

        put_symbol(0.0, 1.0, '#');

        fprintf(FdBalance, "Step=%7d  sec=%10.3f  Nf=%014ld  %s\n", All.NumCurrentTiStep, walltime_step_max("/"),
                GlobNumForceUpdate, CPU_String);
        fflush(FdBalance);
    }


    if(ThisTask == 0)
    {

        fprintf(FdCPU, "Step %d, Time: %g, MPIs: %d Threads: %d Elapsed: %g\n", All.NumCurrentTiStep, All.Time, NTask, All.NumThreads, All.CT.ElapsedTime);
#ifdef _OPENMP
        fprintf(FdCPU, "Blocked Drifts (Particle Node): %ld %ld\n", totBlockedPD, totBlockedND);
        fprintf(FdCPU, "Total Drifts (Particle Node): %ld %ld\n", totTotalPD, totTotalND);
#endif
        fflush(FdCPU);
    }
    walltime_report(FdCPU, 0, MPI_COMM_WORLD);
    if(ThisTask == 0) {
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
