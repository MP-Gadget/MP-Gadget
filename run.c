#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <ctype.h>

#include "allvars.h"
#include "proto.h"
#include "domain.h"
#include "cooling.h"
#include "mymalloc.h"
#include "endrun.h"
#include "timestep.h"
#include "system.h"
#include "drift.h"
#include "forcetree.h"
#include "blackhole.h"
#include "sfr_eff.h"

/*! \file run.c
 *  \brief  iterates over timesteps, main loop
 */

/*! This routine contains the main simulation loop that iterates over
 * single timesteps. The loop terminates when the cpu-time limit is
 * reached, when a `stop' file is found in the output directory, or
 * when the simulation ends because we arrived at TimeMax.
 */
enum ActionType {
    NO_ACTION = 0,
    STOP = 1,
    TIMEOUT = 2,
    AUTO_CHECKPOINT = 3,
    CHECKPOINT = 4,
    TERMINATE = 5,
    IOCTL = 6,
};
static enum ActionType human_interaction();
static int find_next_sync_point(int ti_nextoutput);
static void compute_accelerations(void);
static void update_IO_params(const char * ioctlfname);
static void every_timestep_stuff(int NumForces);
static void write_cpu_log(void);


void run(void)
{
    enum ActionType action = NO_ACTION;

    walltime_measure("/Misc");

    write_cpu_log(); /* produce some CPU usage info */

    do /* main loop */
    {
        /* find next synchronization point and the timebins active during this timestep.
         * If needed, this function will also write an output file
         * at the desired time.
         */
        All.Ti_Current = find_next_sync_point(All.Ti_nextoutput);

        /* Sync positions of all particles */
        drift_all_particles(All.Ti_Current);

        /* drift and domain decomposition */

        /* at first step this is a noop */
        if(All.PM_Ti_endstep == All.Ti_Current) {
            /* full decomposition rebuilds the tree */
            domain_decompose_full();
        } else {
            /* FIXME: add a parameter for domain_decompose_incremental */
            /* currently we drift all particles every step */
            /* If it is not a PM step, do a shorter version
             * of the domain decomp which just moves and exchanges drifted (active) particles.*/
            domain_maintain();
        }

        int NumForces = update_active_timebins(All.Ti_Current);

        rebuild_activelist();

        every_timestep_stuff(NumForces);	/* write some info to log-files */

        /* force */
        compute_accelerations();	/* compute accelerations for
                                     * the particles that are to be advanced
                                     */

        /* kick */
        advance_and_find_timesteps();	/* 'kick' active particles in
                                         * momentum space and compute new
                                         * timesteps for them
                                         */

        /*If this timestep is after the last snapshot time, write a snapshot.
         * No need to do a domain decomposition as we already did one since
         * the last move in compute_accelerations().
         * This is after advance_and_find_timesteps so the acceleration
         * is included in the kick.*/
        if(All.Ti_Current >= All.Ti_nextoutput)
        {
            /*Save snapshot*/
            savepositions(All.SnapshotFileCount++, action == NO_ACTION);	/* write snapshot file */
            All.Ti_nextoutput = find_next_outputtime(All.Ti_nextoutput + 1);
        }
        write_cpu_log();		/* produce some CPU usage info */

        if(action == STOP || action == TIMEOUT) {
            /* OK snapshot file is written, lets quit */
            return;
        }

        All.NumCurrentTiStep++;

        report_memory_usage("RUN");

        action = human_interaction();
        switch(action) {
            case STOP:
                message(0, "human controlled stop with checkpoint.\n");
                All.Ti_nextoutput = All.Ti_Current;
                /* Note there is an error involved in doing this:
                 * part of the SPH VelPred array is computed using
                 * the length of a PM step, and will have been
                 * slightly incorrect for this timestep. But we want a stop NOW.*/
                All.PM_Ti_endstep = All.Ti_Current;
                /* next loop will write a new snapshot file; break is for switch */
                break;
            case TIMEOUT:
                message(0, "stopping due to TimeLimitCPU.\n");
                All.Ti_nextoutput = All.Ti_Current;
                All.PM_Ti_endstep = All.Ti_Current;
                /* next loop will write a new snapshot file */
                break;

            case AUTO_CHECKPOINT:
                message(0, "auto checkpoint due to TimeBetSnapshot.\n");
                All.Ti_nextoutput = All.PM_Ti_endstep;
                /* will write a new snapshot file next time the PM step finishes*/
                break;

            case CHECKPOINT:
                message(0, "human controlled checkpoint.\n");
                All.Ti_nextoutput = All.PM_Ti_endstep;
                /* will write a new snapshot file next time the PM step finishes*/
                break;

            case TERMINATE:
                message(0, "human controlled termination.\n");
                /* no snapshot, at termination, directly end the loop */
                return;

            case IOCTL:
                break;

            case NO_ACTION:
                break;
        }

    }
    while(All.Ti_Current < TIMEBASE);
}

static void
update_IO_params(const char * ioctlfname)
{
    if(ThisTask == 0) {
        FILE * fd = fopen(ioctlfname, "r");
         /* there is an ioctl file, parse it and update
          * All.NumPartPerFile
          * All.NumWriters
          */
        size_t n = 0;
        char * line = NULL;
        while(-1 != getline(&line, &n, fd)) {
            sscanf(line, "BytesPerFile %lu", &All.IO.BytesPerFile);
            sscanf(line, "NumWriters %d", &All.IO.NumWriters);
        }
        free(line);
        fclose(fd);
    }

    MPI_Bcast(&All.IO, sizeof(All.IO), MPI_BYTE, 0, MPI_COMM_WORLD);
    message(0, "New IO parameter recieved from %s:"
               "NumPartPerfile %d"
               "NumWriters %d\n",
            ioctlfname,
            All.IO.BytesPerFile,
            All.IO.NumWriters);
}

static enum ActionType
human_interaction()
{
        /* Check whether we need to interrupt the run */
    enum ActionType action = NO_ACTION;
    char stopfname[4096], termfname[4096];
    char restartfname[4096];
    char ioctlfname[4096];

    sprintf(stopfname, "%s/stop", All.OutputDir);
    sprintf(restartfname, "%s/checkpoint", All.OutputDir);
    sprintf(termfname, "%s/terminate", All.OutputDir);
    sprintf(ioctlfname, "%s/ioctl", All.OutputDir);
    /*Last IO time*/
    double iotime = 0.02*All.TimeLimitCPU;
    double EstimatedTimeToPMEnd = 0.2 * All.TimeLimitCPU;

    int nwritten = All.SnapshotFileCount - All.InitSnapshotCount;

    if(nwritten > 0)
        iotime = walltime_get("/Snapshot/Write",CLOCK_ACCU_MAX)/nwritten;
    if (NumActiveParticle > 0)
        EstimatedTimeToPMEnd = fmin(EstimatedTimeToPMEnd, All.CT.StepTime*NumPart/NumActiveParticle);

    message(0, "iotime = %g, Est = %g\n", iotime, EstimatedTimeToPMEnd);
    if(ThisTask == 0)
    {
        FILE * fd;
        if((fd = fopen(ioctlfname, "r"))) {
            action = IOCTL;
            update_IO_params(ioctlfname);
            fclose(fd);
        }
        /* Is the stop-file present? If yes, interrupt the run. */
        if((fd = fopen(stopfname, "r")))
        {
            action = STOP;
            fclose(fd);
            unlink(stopfname);
        }
        /* Is the terminate-file present? If yes, interrupt the run. */
        if((fd = fopen(termfname, "r")))
        {
            action = TERMINATE;
            fclose(fd);
            unlink(termfname);
        }

        /* are we running out of CPU-time ? If yes, interrupt run. */
        if(All.CT.ElapsedTime + 4*(iotime+EstimatedTimeToPMEnd) > All.TimeLimitCPU) {
            action = TIMEOUT;
        }

        if((All.CT.ElapsedTime - All.TimeLastRestartFile) >= All.CpuTimeBetRestartFile) {
            action = AUTO_CHECKPOINT;
            All.TimeLastRestartFile = All.CT.ElapsedTime;
        }

        if((fd = fopen(restartfname, "r")))
        {
            action = CHECKPOINT;
            fclose(fd);
            unlink(restartfname);
        }
    }

    MPI_Bcast(&action, sizeof(action), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&All.TimeLastRestartFile, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return action;
}

/*! This function finds the next synchronization point of the system
 * (i.e. the earliest point of time any of the particles needs a force
 * computation), and drifts the system to this point of time.  If the
 * system drifts over the desired time of a snapshot file, the
 * function will drift to this moment, generate an output, and then
 * resume the drift.
 */
int find_next_sync_point(int ti_nextoutput)
{
    int n, ti_next_kick_global;
    int ti_next_kick = TIMEBASE;
    const double timeold = All.Time;
    /*This repopulates all timebins on the first timestep*/
    if(TimeBinCount[0])
        ti_next_kick = All.Ti_Current;

    /* find the next kick time */
    for(n = 1; n < TIMEBINS; n++)
    {
        if(!TimeBinCount[n])
            continue;
	    /* next kick time for this timebin */
        const int dt_bin = (1 << n);
        const int ti_next_for_bin = (All.Ti_Current / dt_bin) * dt_bin + dt_bin;
        if(ti_next_for_bin < ti_next_kick)
            ti_next_kick = ti_next_for_bin;
    }

    /*All processors sync timesteps*/
    MPI_Allreduce(&ti_next_kick, &ti_next_kick_global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    All.Ti_Current = ti_next_kick_global;
    /*Convert back to floating point time*/
    double nexttime = All.TimeInit * exp(All.Ti_Current * All.Timebase_interval);

    set_global_time(nexttime);

    All.TimeStep = All.Time - timeold;

    walltime_measure("/Misc");

    return All.Ti_Current;
}

/*! this function returns the next output time that is equal or larger to
 *  ti_curr
 */
int find_next_outputtime(int ti_curr)
{
    int i, ti_next=-1;

    for(i = 0; i < All.OutputListLength; i++)
    {
        const double time = All.OutputListTimes[i];

        if(time >= All.TimeInit && time <= All.TimeMax)
        {
            const int ti = (int) (log(time / All.TimeInit) / All.Timebase_interval);

            if(ti >= ti_curr)
            {
                ti_next = ti;
                break;
            }
        }
    }
    if(ti_next == -1)
    {
        /* Next output is at TimeMax*/
        ti_next = TIMEBASE;
    }
    const double next = All.TimeInit * exp(ti_next * All.Timebase_interval);
    message(0, "Setting next time for snapshot file to Time_next= %g \n", next);
    return ti_next;
}

/*! This routine computes the accelerations for all active particles.  First, the gravitational forces are
 * computed. This also reconstructs the tree, if needed, otherwise the drift/kick operations have updated the
 * tree to make it fullu usable at the current time.
 *
 * If gas particles are presented, the `interior' of the local domain is determined. This region is guaranteed
 * to contain only particles local to the processor. This information will be used to reduce communication in
 * the hydro part.  The density for active SPH particles is computed next. If the number of neighbours should
 * be outside the allowed bounds, it will be readjusted by the function ensure_neighbours(), and for those
 * particle, the densities are recomputed accordingly. Finally, the hydrodynamical forces are added.
 */
void compute_accelerations(void)
{
    message(0, "Start force computation...\n");

    walltime_measure("/Misc");

    if(All.PM_Ti_endstep == All.Ti_Current)
    {
        gravpm_force();

        /* compute and output energy statistics if desired. */
        if(All.OutputEnergyDebug)
            energy_statistics();

        /*Update the displacement timestep*/
        All.MaxTiStepDisplacement = find_dti_displacement_constraint();
    }

    grav_short_tree();		/* computes gravity accel. */

    if(All.Ti_Current == 0)
        grav_short_tree();		/* For the first timestep, we redo it
                             * to allow usage of relative opening
                             * criterion for consistent accuracy.
                             */

    if(NTotal[0] > 0)
    {
        /***** density *****/
        message(0, "Start density computation...\n");

        density();		/* computes density, and pressure */

        /***** update smoothing lengths in tree *****/
        force_update_hmax(ActiveParticle, NumActiveParticle);

        /***** hydro forces *****/
        message(0, "Start hydro-force computation...\n");

        hydro_force(BINMASK_ALL);		/* adds hydrodynamical accelerations  and computes du/dt  */

#ifdef BLACK_HOLES
        /***** black hole accretion and feedback *****/
        blackhole();
#endif

/**** radiative cooling and star formation *****/
#ifdef SFR
        cooling_and_starformation();
#else
        cooling_only();
#endif

    }
    message(0, "force computation done.\n");
}


/*! This routine writes one line for every timestep.
 * FdCPU the cumulative cpu-time consumption in various parts of the
 * code is stored.
 */
void every_timestep_stuff(int NumForce)
{
    double z;
    int i;
    int64_t tot = 0, tot_type[6] = {0};
    int64_t tot_count[TIMEBINS] = {0};
    int64_t tot_count_type[6][TIMEBINS] = {0};
    int64_t tot_num_force = 0;

    sumup_large_ints(TIMEBINS, TimeBinCount, tot_count);
    for(i = 0; i < 6; i ++) {
        sumup_large_ints(TIMEBINS, TimeBinCountType[i], tot_count_type[i]);
    }
    sumup_large_ints(1, &NumForce, &tot_num_force);

    /* let's update Tot counts in one place tot variables;
     * at this point there can still be holes in SphP
     * because rearrange_particle_squence is not called yet.
     * but anywaysTotN_sph variables are not well defined and
     * not used any places but printing.
     *
     * we shall just say they we sync these variables right after gravity
     * calculation in every timestep.
     * */

    char extra[1024] = {0};

    if(All.PM_Ti_endstep == All.Ti_Current)
        strcat(extra, "PM-Step");

    z = 1.0 / (All.Time) - 1;
    message(0, "Begin Step %d, Time: %g, Redshift: %g, Nf = %014ld, Systemstep: %g, Dloga: %g, status: %s\n",
                All.NumCurrentTiStep, All.Time, z, tot_num_force,
                All.TimeStep, log(All.Time) - log(All.Time - All.TimeStep),
                extra);

    message(0, "TotNumPart: %013ld SPH %013ld BH %010ld STAR %013ld \n",
                TotNumPart, NTotal[0], NTotal[5], NTotal[4]);
    message(0,     "Occupied:   % 12ld % 12ld % 12ld % 12ld % 12ld % 12ld dt\n", 0L, 1L, 2L, 3L, 4L, 5L);

    for(i = TIMEBINS - 1;  i >= 0; i--) {
        if(tot_count[i] == 0) continue;
        message(0, " %c bin=%2d % 12ld % 12ld % 12ld % 12ld % 12ld % 12ld %6g\n",
                is_timebin_active(i) ? 'X' : ' ',
                i,
                tot_count_type[0][i],
                tot_count_type[1][i],
                tot_count_type[2][i],
                tot_count_type[3][i],
                tot_count_type[4][i],
                tot_count_type[5][i],
                get_dloga_for_bin(i));

        if(is_timebin_active(i))
        {
            tot += tot_count[i];
            int ptype;
            for(ptype = 0; ptype < 6; ptype ++) {
                tot_type[ptype] += tot_count_type[ptype][i];
            }
        }
    }
    message(0,     "               -----------------------------------\n");
    message(0,     "Total:  % 12ld % 12ld % 12ld % 12ld % 12ld % 12ld  Sum:% 14ld\n",
        tot_type[0], tot_type[1], tot_type[2], tot_type[3], tot_type[4], tot_type[5], tot);

    set_random_numbers();
}



void write_cpu_log(void)
{
    int64_t totBlockedPD = -1;
    int64_t totTotalPD = -1;

#ifdef _OPENMP
    MPI_Reduce(&BlockedParticleDrifts, &totBlockedPD, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&TotalParticleDrifts, &totTotalPD, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
#endif

    walltime_summary(0, MPI_COMM_WORLD);

    if(ThisTask == 0)
    {

        fprintf(FdCPU, "Step %d, Time: %g, MPIs: %d Threads: %d Elapsed: %g\n", All.NumCurrentTiStep, All.Time, NTask, All.NumThreads, All.CT.ElapsedTime);
#ifdef _OPENMP
        fprintf(FdCPU, "Blocked Particle Drifts: %ld\n", totBlockedPD);
        fprintf(FdCPU, "Total Particle Drifts: %ld\n", totTotalPD);
#endif
        fflush(FdCPU);
    }
    walltime_report(FdCPU, 0, MPI_COMM_WORLD);
    if(ThisTask == 0) {
        fflush(FdCPU);
    }
}
