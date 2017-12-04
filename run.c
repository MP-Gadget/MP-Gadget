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
#include "hci.h"

/*! \file run.c
 *  \brief  iterates over timesteps, main loop
 */

/*! This routine contains the main simulation loop that iterates over
 * single timesteps. The loop terminates when the cpu-time limit is
 * reached, when a `stop' file is found in the output directory, or
 * when the simulation ends because we arrived at TimeMax.
 */
static void compute_accelerations(int is_PM, int FirstStep, int GasEnabled);
static void write_cpu_log(int NumCurrentTiStep);

void run(void)
{
    /*Number of timesteps performed this run*/
    int NumCurrentTiStep = 0;
    /*Minimum occupied timebin. Initially (but never again) zero*/
    int minTimeBin = 0;
    /*Is gas physics enabled?*/
    int GasEnabled = All.NTotalInit[0];

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

        next_sync = find_next_sync_point(All.Ti_Current);
        planned_sync = find_current_sync_point(All.Ti_Current);

        HCIAction action[1];

        hci_action_init(action); /* init to no action */

        int stop = 0;

        if(is_PM) {
            /* query other requests only on PM step. */
            stop = hci_query(HCI_DEFAULT_MANAGER, action);

            if(action->type == HCI_TERMINATE) {
                endrun(0, "Human triggered termination.\n");
            }
        }
        /* Sync positions of all particles */
        drift_all_particles(All.Ti_Current);

        /* drift and domain decomposition */

        /* at first step this is a noop */
        if(is_PM) {
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
        compute_accelerations(is_PM, NumCurrentTiStep == 0, GasEnabled);

        /* Update velocity to Ti_Current; this synchonizes TiKick and TiDrift for the active particles */

        if(is_PM) {
            apply_PM_half_kick();
        }

        apply_half_kick();

        /* If a snapshot is requested, write it.
         * write_checkpoint is responsible to maintain a valid domain and tree after it is called.
         *
         * We only attempt to output on sync points. This is the only chance where all variables are
         * synchonized in a consistent state in a K(KDDK)^mK scheme.
         */

        int WriteSnapshot = 0;
        int WriteFOF = 0;

        if(planned_sync) {
            WriteSnapshot |= planned_sync->write_snapshot;
            WriteFOF |= planned_sync->write_fof;

        }

        if(is_PM) { /* the if here is unnecessary but to signify checkpointing occurs only at PM steps. */
            WriteSnapshot |= action->write_snapshot;
        }

        if(WriteSnapshot) {
            /* The accel may have created garbage -- collect them before writing a snapshot.
             * If we do collect, rebuild tree and active list.*/
            int compact[6] = {0};

            if(slots_gc(compact)) {
                force_tree_rebuild();
                rebuild_activelist(All.Ti_Current);
            }
        }

        write_checkpoint(WriteSnapshot, WriteFOF);

        write_cpu_log(NumCurrentTiStep);		/* produce some CPU usage info */

        NumCurrentTiStep++;

        report_memory_usage("RUN");

        if(!next_sync || stop) {
            /* out of sync points, or a requested stop, the run has finally finished! Yay.*/
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
void compute_accelerations(int is_PM, int FirstStep, int GasEnabled)
{
    message(0, "Begin force computation.\n");

    walltime_measure("/Misc");

    /* We do this first so that the density is up to date for
     * adaptive gravitational softenings. */
    if(GasEnabled > 0)
    {
        /***** density *****/
        message(0, "Start density computation...\n");

        density();		/* computes density, and pressure */

        /***** update smoothing lengths in tree *****/
        force_update_hmax(ActiveParticle, NumActiveParticle);
        /***** hydro forces *****/
        message(0, "Start hydro-force computation...\n");

        hydro_force();		/* adds hydrodynamical accelerations  and computes du/dt  */
    }

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

    /* Note this must be after gravaccel and hydro,
     * because new star particles are not in the tree,
     * so mass conservation would be broken.*/
    if(GasEnabled)
    {
#ifdef BLACK_HOLES
        /* Black hole accretion and feedback */
        blackhole();
#endif
        /**** radiative cooling and star formation *****/
#ifdef SFR
        cooling_and_starformation();
#else
        cooling_only();
#endif
    }
    message(0, "Forces computed.\n");
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
