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
#include "slotsmanager.h"
#include "fof.h"

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
static void compute_accelerations(int is_PM, int FirstStep);
static void update_IO_params(const char * ioctlfname);
static void write_cpu_log(int NumCurrentTiStep);

void run(void)
{
    /*Number of timesteps performed this run*/
    int NumCurrentTiStep = 0;
    /*Minimum occupied timebin. Initially (but never again) zero*/
    int minTimeBin = 0;

    /*To compute the wall time between PM steps and decide when to timeout.*/
    double lastPM = All.CT.ElapsedTime;
    double TimeLastOutput = 0;

    walltime_measure("/Misc");

    write_cpu_log(NumCurrentTiStep); /* produce some CPU usage info */

    while(1) /* main loop */
    {
        /* Find next synchronization point and the timebins active during this timestep.
         *
         * Note that on startup, P[i].TimeBin == 0 for all particles,
         * all bins except the zeroth are inactive and so we return 0 from this function.
         * This ensures we run the force calculation for the first timestep.
         */
        All.Ti_Current = find_next_kick(All.Ti_Current, minTimeBin);

        /*Convert back to floating point time*/
        set_global_time(exp(loga_from_ti(All.Ti_Current)));
        /*1.0 check for rate setting in sfr_eff.c*/
        if(NumCurrentTiStep > 0 && All.TimeStep < 0)
            endrun(1, "Negative timestep: %g New Time: %g!\n", All.TimeStep, All.Time);

        int is_PM = is_PM_timestep(All.Ti_Current);

        SyncPoint * next_sync; /* if we are out of planned sync points, terminate */
        SyncPoint * planned_sync; /* NULL; if the step is not a planned sync point. */
        SyncPoint * unplanned_sync; /* begin and end of a PM step; not planned in advance */

        next_sync = find_next_sync_point(All.Ti_Current);
        planned_sync = find_current_sync_point(All.Ti_Current);
        unplanned_sync = NULL;

        enum ActionType action = NO_ACTION;

        if(is_PM) {
            unplanned_sync = make_unplanned_sync_point(All.Ti_Current);
            action = human_interaction(lastPM, TimeLastOutput);

            switch(action) {
                case STOP:
                    message(0, "human controlled stop with checkpoint at next PM.\n");
                    /* Write when the PM timestep completes*/
                    unplanned_sync->write_snapshot = 1;
                    unplanned_sync->write_fof = 0;
                    next_sync = NULL; /* will terminate */
                    break;

                case TIMEOUT:
                    message(0, "Stopping due to TimeLimitCPU.\n");
                    unplanned_sync->write_snapshot = 1;
                    unplanned_sync->write_fof = 0;

                    next_sync = NULL; /* will terminate */
                    break;

                case AUTO_CHECKPOINT:
                    message(0, "Auto checkpoint due to AutoSnapshotTime.\n");
                    unplanned_sync->write_snapshot = 1;
                    unplanned_sync->write_fof = 0;
                    break;

                case CHECKPOINT:
                    message(0, "human controlled checkpoint at next PM.\n");
                    unplanned_sync->write_snapshot = 1;
                    unplanned_sync->write_fof = 0;
                    break;

                case TERMINATE:
                    message(0, "human controlled termination.\n");
                    /* FIXME: this shall occur every step; but it means we need
                     * two versions human-interaction routines.*/

                    /* no snapshot, at termination, directly end the loop */
                    return;

                case IOCTL:
                case NO_ACTION:
                    unplanned_sync->write_snapshot = 0;
                    unplanned_sync->write_fof = 0;
                    break;
            }
        }
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

        rebuild_activelist(All.Ti_Current);

        print_timebin_statistics(NumCurrentTiStep);

        set_random_numbers();

        /* update force to Ti_Current */
        compute_accelerations(is_PM, NumCurrentTiStep == 0);

        /* Update velocity to Ti_Current; this synchonizes TiKick and TiDrift for the active particles */

        if(is_PM) {
            apply_PM_half_kick();
        }

        apply_half_kick();

        /* If a snapshot is requested, write it.
         * savepositions is responsible to maintain a valid domain and tree after it is called.
         *
         * We only attempt to output on sync points. This is the only chance where all variables are
         * synchonized in a consistent state in a K(KDDK)^mK scheme.
         */

        int WriteSnapshot = planned_sync && planned_sync->write_snapshot;
        WriteSnapshot |= unplanned_sync && unplanned_sync->write_snapshot;
        int WriteFOF = planned_sync && planned_sync->write_fof;
        WriteFOF |= unplanned_sync && unplanned_sync->write_fof;

        if(WriteSnapshot || WriteFOF) {
            int snapnum = All.SnapshotFileCount++;

            /* The accel may have created garbage -- collect them before checkpointing!
             * Tree will be auto-rebuilt if gc collected particles,
             * but we should rebuild the active list.*/
            int compact[6] = {0};
            if(slots_gc(compact))
                rebuild_activelist(All.Ti_Current);

            if(WriteSnapshot)
            {
                /* Save snapshot and fof. */
                /* FIXME: this doesn't allow saving fof without the snapshot yet. do it after allocator is merged */

                /* write snapshot of particles */
                savepositions(snapnum);

                TimeLastOutput = All.CT.ElapsedTime;
            }

            if(WriteFOF) {
                message(0, "computing group catalogue...\n");

                fof_fof();
                fof_save_groups(snapnum);
                fof_finish();

                message(0, "done with group catalogue.\n");
            }
        }
        write_cpu_log(NumCurrentTiStep);		/* produce some CPU usage info */

        NumCurrentTiStep++;

        report_memory_usage("RUN");

        if(!next_sync) {
            /* out of sync points, the run has finally finished! Yay.*/
            break;
        }

        /* more steps to go. */

        /* assign new timesteps to the active particles, now that we know they have synched TiKick and TiDrift */
        find_timesteps(&minTimeBin);

        /* Update velocity to the new step, with the newly computed step size */
        apply_half_kick();

        if(is_PM) {
            apply_PM_half_kick();
        }
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
void compute_accelerations(int is_PM, int FirstStep)
{
    message(0, "Start force computation...\n");

    walltime_measure("/Misc");

    /* The opening criterion for the gravtree
     * uses the *total* gravitational acceleration
     * from the last timestep, GravPM+GravAccel.
     * So we must compute GravAccel for this timestep
     * before gravpm_force() writes the PM acc. for
     * this timestep to GravPM. Note initially both
     * are zero and so the tree is opened maximally
     * on the first timestep.*/
    grav_short_tree();

    /* We use the total gravitational acc.
     * to open the tree and total acc for the timestep.
     * Note that any of (GravAccel, GravPM,
     * HydroAccel) may change much faster than
     * the total acc.
     * We do the same as Gadget-2, but one could
     * instead use short-range tree acc. only
     * for opening angle or short-range timesteps,
     * or include hydro in the opening angle.*/

    if(is_PM)
    {
        gravpm_force();

        /* compute and output energy statistics if desired. */
        if(All.OutputEnergyDebug)
            energy_statistics();
    }

    /* For the first timestep, we do tree force twice
     * to allow usage of relative opening
     * criterion for consistent accuracy.
     * This happens after PM because we want to
     * use the total acceleration for tree opening.
     */
    if(FirstStep)
        grav_short_tree();

    if(All.NTotalInit[0] > 0)
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
