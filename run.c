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
static enum ActionType human_interaction(double lastPM, double TimeLastOutput);
static int should_we_timeout(double TimelastPM);
static void compute_accelerations(int is_PM);
static void update_IO_params(const char * ioctlfname);
static void every_timestep_stuff(int NumForces, int NumCurrentTiStep);
static void write_cpu_log(int NumCurrentTiStep);

void run(void)
{
    /*Number of timesteps performed this run*/
    int NumCurrentTiStep = 0;

    /*To compute the wall time between PM steps and decide when to timeout.*/
    double lastPM = All.CT.ElapsedTime;
    double TimeLastOutput = 0;

    walltime_measure("/Misc");

    write_cpu_log(NumCurrentTiStep); /* produce some CPU usage info */

    /* find the first output time. */
    SyncPoint * next_sync = find_next_sync_point(All.Ti_Current);

    while(next_sync) /* main loop */
    {
        /* find next synchronization point and the timebins active during this timestep.
         * If needed, this function will also write an output file
         * at the desired time.
         */
        All.Ti_Current = find_next_kick(All.Ti_Current);

        /*Convert back to floating point time*/
        set_global_time(exp(loga_from_ti(All.Ti_Current)));

        int is_PM = is_PM_timestep(All.Ti_Current);

        SyncPoint * current_sync = find_current_sync_point(All.Ti_Current);
        next_sync = find_next_sync_point(All.Ti_Current);

        enum ActionType action = NO_ACTION;

        if(is_PM) {
            action = human_interaction(lastPM, TimeLastOutput);
            switch(action) {
                case STOP:
                    message(0, "human controlled stop with checkpoint at next PM.\n");

                    /* Write when the PM timestep completes*/
                    current_sync = get_pm_sync_point(All.Ti_Current);
                    next_sync = NULL;
                    break;

                case TIMEOUT:
                    message(0, "Stopping due to TimeLimitCPU.\n");
                    current_sync = get_pm_sync_point(All.Ti_Current);
                    next_sync = NULL;
                    break;

                case AUTO_CHECKPOINT:
                    message(0, "Auto checkpoint due to AutoSnapshotTime.\n");
                    current_sync = get_pm_sync_point(All.Ti_Current);
                    break;

                case CHECKPOINT:
                    message(0, "human controlled checkpoint at next PM.\n");
                    current_sync = get_pm_sync_point(All.Ti_Current);
                    break;

                case TERMINATE:
                    message(0, "human controlled termination.\n");
                    /* no snapshot, at termination, directly end the loop */
                    return;

                case IOCTL:
                case NO_ACTION:
                    break;
            }
        }

        int WillOutput = is_PM && current_sync && current_sync->write_snapshot;

        /* Sync positions of all particles */
        drift_all_particles(All.Ti_Current);

        /* drift and domain decomposition */

        /* at first step this is a noop */
        if(is_PM) {
            lastPM = All.CT.ElapsedTime;
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

        every_timestep_stuff(NumForces, NumCurrentTiStep);	/* write some info to log-files */

        /* update force to Ti_Current */
        compute_accelerations(is_PM);

        /* Update velocity to Ti_Current; this synchonizes TiKick and TiDrift for the active particles */

        if(is_PM) {
            apply_PM_half_kick();
        }

        apply_half_kick();

        /* assign new timesteps to the active particles, now that we know they have synched TiKick and TiDrift */
        find_timesteps();

        /* If this timestep is after the last snapshot time, write a snapshot.
         * No need to do a domain decomposition as we already did one since
         * the last move in compute_accelerations().
         *
         * Also watch out WillOutput is only true on is_PM; to ensure the PM kick is done
         * and included in the velocity. This is the only chance where all variables are
         * synchonized in a consistent state in a K(KDDK)^mK scheme.
         */

        if(WillOutput)
        {
            /*Save snapshot*/
            savepositions(All.SnapshotFileCount++, current_sync->write_fof);	/* write snapshot file */

            TimeLastOutput = All.CT.ElapsedTime;
        }

        /*Do the extra half-kick we avoided for a snapshot.*/

        /* Update velocity to the new step, with the newly computed step size */
        apply_half_kick();

        if(is_PM) {
            apply_PM_half_kick();
        }

        write_cpu_log(NumCurrentTiStep);		/* produce some CPU usage info */

        NumCurrentTiStep++;

        report_memory_usage("RUN");

    }
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

/* lastPMlength is the walltime in seconds between the last two PM steps.
 * It is used to decide when we are going to timeout*/
static enum ActionType
human_interaction(double TimeLastPM, double TimeLastOut)
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
    /*How long since the last checkpoint?*/
    if(All.AutoSnapshotTime > 0 && All.CT.ElapsedTime - TimeLastOut >= All.AutoSnapshotTime) {
        action = AUTO_CHECKPOINT;
    }

    if(ThisTask == 0)
    {
        FILE * fd;
        if((fd = fopen(ioctlfname, "r"))) {
            action = IOCTL;
            update_IO_params(ioctlfname);
            fclose(fd);
        }

        if((fd = fopen(restartfname, "r")))
        {
            action = CHECKPOINT;
            fclose(fd);
            unlink(restartfname);
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

    }
    /*Will we run out of time by the next PM step?*/
    if(should_we_timeout(TimeLastPM)) {
        action = TIMEOUT;
    }

    MPI_Bcast(&action, sizeof(action), MPI_BYTE, 0, MPI_COMM_WORLD);

    return action;
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
void compute_accelerations(int is_PM)
{
    message(0, "Start force computation...\n");

    walltime_measure("/Misc");

    if(is_PM)
    {
        gravpm_force();

        /* compute and output energy statistics if desired. */
        if(All.OutputEnergyDebug)
            energy_statistics();
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

        hydro_force();		/* adds hydrodynamical accelerations  and computes du/dt  */

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

int should_we_timeout(double TimeLastPM)
{
    /*Last IO time*/
    double iotime = 0.02*All.TimeLimitCPU;

    int nwritten = All.SnapshotFileCount - All.InitSnapshotCount;
    if(nwritten > 0)
        iotime = walltime_get("/Snapshot/Write",CLOCK_ACCU_MAX)/nwritten;

    double curTime = All.CT.ElapsedTime;
/*     message(0, "iotime = %g, lastPM = %g\n", iotime, lastPMlength); */
    /* are we running out of CPU-time ? If yes, interrupt run. */
    if(curTime + 4*(iotime + curTime - TimeLastPM) > All.TimeLimitCPU) {
        return 1;
    }
    return 0;
}

/*! This routine writes one line for every timestep.
 * FdCPU the cumulative cpu-time consumption in various parts of the
 * code is stored.
 */
void every_timestep_stuff(int NumForce, int NumCurrentTiStep)
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

    if(is_PM_timestep(All.Ti_Current))
        strcat(extra, "PM-Step");

    z = 1.0 / (All.Time) - 1;
    message(0, "Begin Step %d, Time: %g, Redshift: %g, Nf = %014ld, Systemstep: %g, Dloga: %g, status: %s\n",
                NumCurrentTiStep, All.Time, z, tot_num_force,
                All.TimeStep, log(All.Time) - log(All.Time - All.TimeStep),
                extra);

    int64_t TotNumPart = 0;
    for(i = 0; i < 6; i ++) TotNumPart += NTotal[i];

    message(0, "TotNumPart: %013ld SPH %013ld BH %010ld STAR %013ld \n",
                TotNumPart, NTotal[0], NTotal[5], NTotal[4]);
    message(0,     "Occupied: % 12ld % 12ld % 12ld % 12ld % 12ld % 12ld dt\n", 0L, 1L, 2L, 3L, 4L, 5L);

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
    message(0,     "Total:    % 12ld % 12ld % 12ld % 12ld % 12ld % 12ld  Sum:% 14ld\n",
        tot_type[0], tot_type[1], tot_type[2], tot_type[3], tot_type[4], tot_type[5], tot);

    set_random_numbers();
}

void write_cpu_log(int NumCurrentTiStep)
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

        fprintf(FdCPU, "Step %d, Time: %g, MPIs: %d Threads: %d Elapsed: %g\n", NumCurrentTiStep, All.Time, NTask, All.NumThreads, All.CT.ElapsedTime);
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
