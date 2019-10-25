#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <ctype.h>

#include "utils.h"

#include "allvars.h"
#include "gravity.h"
#include "density.h"
#include "domain.h"
#include "run.h"
#include "init.h"
#include "cooling.h"
#include "checkpoint.h"
#include "petaio.h"
#include "petapm.h"
#include "timestep.h"
#include "drift.h"
#include "forcetree.h"
#include "blackhole.h"
#include "hydra.h"
#include "sfr_eff.h"
#include "slotsmanager.h"
#include "hci.h"
#include "fof.h"

void energy_statistics(void); /* stats.c only used here */

/*! \file run.c
 *  \brief  iterates over timesteps, main loop
 */

/*! This routine contains the main simulation loop that iterates over
 * single timesteps. The loop terminates when the cpu-time limit is
 * reached, when a `stop' file is found in the output directory, or
 * when the simulation ends because we arrived at TimeMax.
 */
static void compute_accelerations(const ActiveParticles * act, int is_PM, PetaPM * pm, int FirstStep, int GasEnabled, int HybridNuGrav, ForceTree * tree, DomainDecomp * ddecomp);
static void write_cpu_log(int NumCurrentTiStep);

/* Updates the global storing the current random offset of the particles,
 * and stores the relative offset from the last random offset in rel_random_shift*/
static void update_random_offset(double * rel_random_shift);

/*! \file begrun.c
 *  \brief initial set-up of a simulation run
 *
 *  This file contains various functions to initialize a simulation run. In
 *  particular, the parameterfile is read in and parsed, the initial
 *  conditions or restart files are read, and global variables are initialized
 *  to their proper values.
 */


static void set_units();
static void set_softenings();

static void
open_outputfiles(int RestartSnapNum);

static void
close_outputfiles(void);

/*! This function performs the initial set-up of the simulation. First, the
 *  parameterfile is set, then routines for setting units, reading
 *  ICs/restart-files are called, auxialiary memory is allocated, etc.
 */
void begrun(int RestartSnapNum)
{
    hci_init(HCI_DEFAULT_MANAGER, All.OutputDir, All.TimeLimitCPU, All.AutoSnapshotTime);

    petapm_module_init(All.NumThreads);
    petaio_init();
    walltime_init(&All.CT);

    petaio_read_header(RestartSnapNum);

    slots_init(All.SlotsIncreaseFactor);
    /* Enable the slots: stars and BHs are allocated if there are some,
     * or if some will form*/
    if(All.NTotalInit[0] > 0)
        slots_set_enabled(0, sizeof(struct sph_particle_data));
    if(All.StarformationOn || All.NTotalInit[4] > 0)
        slots_set_enabled(4, sizeof(struct star_particle_data));
    if(All.BlackHoleOn || All.NTotalInit[5] > 0)
        slots_set_enabled(5, sizeof(struct bh_particle_data));

    set_softenings();
    set_units();

#ifdef DEBUG
    char * pidfile = fastpm_strdup_printf("%s/%s", All.OutputDir, "PIDs.txt");

    MPIU_write_pids(pidfile);
    myfree(pidfile);

    enable_core_dumps_and_fpu_exceptions();
#endif

    init_forcetree_params(All.FastParticleType, GravitySofteningTable);

    init_cooling_and_star_formation();

    gravshort_fill_ntab(All.ShortRangeForceWindowType, All.Asmth);

    set_random_numbers(All.RandomSeed);

#ifdef LIGHTCONE
    lightcone_init(All.Time);
#endif

    open_outputfiles(RestartSnapNum);
}

void
run(int RestartSnapNum)
{
    /*Number of timesteps performed this run*/
    int NumCurrentTiStep = 0;
    /*Minimum occupied timebin. Initially (but never again) zero*/
    int minTimeBin = 0;
    /*Is gas physics enabled?*/
    int GasEnabled = All.NTotalInit[0] > 0;

    PetaPM pm = {0};
    gravpm_init_periodic(&pm, All.BoxSize, All.Asmth, All.Nmesh, All.G);

    DomainDecomp ddecomp[1] = {0};
    init(RestartSnapNum, ddecomp);          /* ... read in initial model */

    /* Stored scale factor of the next black hole seeding check*/
    double TimeNextSeedingCheck = All.Time;

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
            /* query HCI requests only on PM step; where kick and drifts are synced */
            stop = hci_query(HCI_DEFAULT_MANAGER, action);

            if(action->type == HCI_TERMINATE) {
                endrun(0, "Human triggered termination.\n");
            }
        }

        double rel_random_shift[3] = {0};
        if(NumCurrentTiStep > 0 && is_PM  && All.RandomParticleOffset > 0) {
            update_random_offset(rel_random_shift);
        }
        /* Sync positions of all particles */
        drift_all_particles(All.Ti_Current, rel_random_shift);

        /* drift and ddecomp decomposition */

        /* at first step this is a noop */
        if(is_PM) {
            /* full decomposition rebuilds the tree */
            domain_decompose_full(ddecomp);
        } else {
            /* FIXME: add a parameter for ddecomp_decompose_incremental */
            /* currently we drift all particles every step */
            /* If it is not a PM step, do a shorter version
             * of the ddecomp decomp which just exchanges particles.*/
            domain_maintain(ddecomp);
        }

        ActiveParticles Act = {0};
        rebuild_activelist(&Act, All.Ti_Current, NumCurrentTiStep);

        set_random_numbers(All.RandomSeed + All.Ti_Current);

        /* Are the particle neutrinos gravitating this timestep?
         * If so we need to add them to the tree.*/
        int HybridNuGrav = All.HybridNeutrinosOn && All.Time <= All.HybridNuPartTime;

        /* Need to rebuild the force tree because all TopLeaves are out of date.*/
        ForceTree Tree = {0};
        force_tree_rebuild(&Tree, ddecomp, All.BoxSize, HybridNuGrav);

        /*Allocate the extra SPH data for transient SPH particle properties.*/
        if(GasEnabled)
            slots_allocate_sph_scratch_data(sfr_need_to_compute_sph_grad_rho(), SlotsManager->info[0].size);
        /* update force to Ti_Current */
        compute_accelerations(&Act, is_PM, &pm, NumCurrentTiStep == 0, GasEnabled, HybridNuGrav, &Tree, ddecomp);

        /* Note this must be after gravaccel and hydro,
         * because new star particles are not in the tree,
         * so mass conservation would be broken.*/
        if(GasEnabled)
        {
            /* Black hole accretion and feedback */
            blackhole(&Act, &Tree);
            /* this will find new black hole seed halos */
            if(All.BlackHoleOn && All.Time >= TimeNextSeedingCheck)
            {
                /* Seeding */
                FOFGroups fof = fof_fof(&Tree, All.BoxSize, All.BlackHoleOn, MPI_COMM_WORLD);
                fof_seed(&fof, MPI_COMM_WORLD);
                fof_finish(&fof);
                TimeNextSeedingCheck = All.Time * All.TimeBetweenSeedingSearch;
            }

            /**** radiative cooling and star formation *****/
            cooling_and_starformation(&Act, &Tree);

            /* Scratch data cannot be used checkpoint because FOF does an exchange.*/
            slots_free_sph_scratch_data(SphP_scratch);
        }

        /* Update velocity to Ti_Current; this synchonizes TiKick and TiDrift for the active particles */

        if(is_PM) {
            apply_PM_half_kick();
        }

        apply_half_kick(&Act);

        /* If a snapshot is requested, write it.
         * write_checkpoint is responsible to maintain a valid ddecomp and tree after it is called.
         *
         * We only attempt to output on sync points. This is the only chance where all variables are
         * synchronized in a consistent state in a K(KDDK)^mK scheme.
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
             * If we do collect, rebuild tree and reset active list size.*/
            int compact[6] = {0};

            if(slots_gc(compact)) {
                force_tree_rebuild(&Tree, ddecomp, All.BoxSize, HybridNuGrav);
                Act.NumActiveParticle = PartManager->NumPart;
            }
        }

        write_checkpoint(WriteSnapshot, WriteFOF, &Tree);

        write_cpu_log(NumCurrentTiStep);    /* produce some CPU usage info */

        NumCurrentTiStep++;

        report_memory_usage("RUN");

        /*Note FoF may free the tree too*/
        force_tree_free(&Tree);

        if(!next_sync || stop) {
            /* out of sync points, or a requested stop, the run has finally finished! Yay.*/
            break;
        }

        /* more steps to go. */

        /* assign new timesteps to the active particles,
         * now that we know they have synched TiKick and TiDrift,
         * and advance the PM timestep.*/
        minTimeBin = find_timesteps(&Act, All.Ti_Current);

        /* Update velocity to the new step, with the newly computed step size */
        apply_half_kick(&Act);

        if(is_PM) {
            apply_PM_half_kick();
        }

        /* We can now free the active list: the new step have new active particles*/
        free_activelist(&Act);
    }

    close_outputfiles();
}

/*! This routine computes the accelerations for all active particles.  First, the gravitational forces are
 * computed. This also reconstructs the tree, if needed, otherwise the drift/kick operations have updated the
 * tree to make it fully usable at the current time.
 *
 * If gas particles are presented, the `interior' of the local ddecomp is determined. This region is guaranteed
 * to contain only particles local to the processor. This information will be used to reduce communication in
 * the hydro part.  The density for active SPH particles is computed next. If the number of neighbours should
 * be outside the allowed bounds, it will be readjusted by the function ensure_neighbours(), and for those
 * particle, the densities are recomputed accordingly. Finally, the hydrodynamical forces are added.
 */
void compute_accelerations(const ActiveParticles * act, int is_PM, PetaPM * pm, int FirstStep, int GasEnabled, int HybridNuGrav, ForceTree * tree, DomainDecomp * ddecomp)
{
    message(0, "Begin force computation.\n");

    walltime_measure("/Misc");

    /* density() happens before gravity because it also initializes the predicted variables.
     * This ensures that prediction consistently uses the grav and hydro accel from the
     * timestep before this one, which matches Gadget-2/3. It was tested to make a small difference,
     * since prediction is only really used for artificial viscosity.
     *
     * Doing it first also means the density is up to date for
     * adaptive gravitational softenings. */
    if(GasEnabled)
    {
        /***** density *****/
        message(0, "Start density computation...\n");

        density(act, 1, All.DensityIndependentSphOn, tree);  /* computes density, and pressure */

        /***** update smoothing lengths in tree *****/
        force_update_hmax(act->ActiveParticle, act->NumActiveParticle, tree, ddecomp);
        /***** hydro forces *****/
        MPIU_Barrier(MPI_COMM_WORLD);
        message(0, "Start hydro-force computation...\n");

        hydro_force(act, tree);		/* adds hydrodynamical accelerations  and computes du/dt  */
    }

    /* The opening criterion for the gravtree
     * uses the *total* gravitational acceleration
     * from the last timestep, GravPM+GravAccel.
     * So we must compute GravAccel for this timestep
     * before gravpm_force() writes the PM acc. for
     * this timestep to GravPM. Note initially both
     * are zero and so the tree is opened maximally
     * on the first timestep.*/
    const int NeutrinoTracer =  All.HybridNeutrinosOn && (All.Time <= All.HybridNuPartTime);
    const double rho0 = All.CP.Omega0 * 3 * All.CP.Hubble * All.CP.Hubble / (8 * M_PI * All.G);

    if(All.TreeGravOn)
        grav_short_tree(act, pm, tree, rho0, NeutrinoTracer, All.FastParticleType);

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
        gravpm_force(pm, tree);

        /*Rebuild the force tree we freed in gravpm to save memory*/
        force_tree_rebuild(tree, ddecomp, All.BoxSize, HybridNuGrav);

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
    if(FirstStep && All.TreeGravOn)
        grav_short_tree(act, pm, tree, rho0, NeutrinoTracer, All.FastParticleType);

    MPIU_Barrier(MPI_COMM_WORLD);
    message(0, "Forces computed.\n");
}

void write_cpu_log(int NumCurrentTiStep)
{
    walltime_summary(0, MPI_COMM_WORLD);

    int NTask, ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);

    if(ThisTask == 0)
    {
        fprintf(FdCPU, "Step %d, Time: %g, MPIs: %d Threads: %d Elapsed: %g\n", NumCurrentTiStep, All.Time, NTask, All.NumThreads, All.CT.ElapsedTime);
        fflush(FdCPU);
    }
    walltime_report(FdCPU, 0, MPI_COMM_WORLD);
    if(ThisTask == 0) {
        fflush(FdCPU);
    }
}

/* We operate in a situation where the particles are in a coordinate frame
 * offset slightly from the ICs (to avoid correlated tree errors).
 * This function updates the global variable containing that offset, and
 * stores the relative shift from the last offset in the rel_random_shift output
 * array. */
static void
update_random_offset(double * rel_random_shift)
{
    int i;
    for (i = 0; i < 3; i++) {
        /* Note random number table is duplicated across processors*/
        double rr = get_random_number(i);
        /* Upstream Gadget uses a random fraction of the box, but since all we need
         * is to adjust the tree openings, and the tree force is zero anyway on the
         * scale of a few PM grid cells, this seems enough.*/
        rr *= All.RandomParticleOffset * All.BoxSize / All.Nmesh;
        /* Subtract the old random shift first.*/
        rel_random_shift[i] = rr - All.CurrentParticleOffset[i];
        All.CurrentParticleOffset[i] = rr;
    }
    message(0, "Internal particle offset is now %g %g %g\n", All.CurrentParticleOffset[0], All.CurrentParticleOffset[1], All.CurrentParticleOffset[2]);
#ifdef DEBUG
    /* Check explicitly that the vector is the same on all processors*/
    double test_random_shift[3] = {0};
    for (i = 0; i < 3; i++)
        test_random_shift[i] = All.CurrentParticleOffset[i];
    MPI_Bcast(test_random_shift, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (i = 0; i < 3; i++)
        if(test_random_shift[i] != All.CurrentParticleOffset[i])
            endrun(44, "Random shift %d is %g != %g on task 0!\n", i, test_random_shift[i], All.CurrentParticleOffset[i]);
#endif
}

/*!  This function opens various log-files that report on the status and
 *   performance of the simulstion. On restart from restart-files
 *   (start-option 1), the code will append to these files.
 */
static void
open_outputfiles(int RestartSnapNum)
{
    const char mode[3]="a+";
    char * buf;
    char * postfix;

    if(ThisTask != 0) {
        /* only the root processors writes to the log files */
        return;
    }

    if(RestartSnapNum != -1) {
        postfix = fastpm_strdup_printf("-R%03d", RestartSnapNum);
    } else {
        postfix = fastpm_strdup_printf("%s", "");
    }

    buf = fastpm_strdup_printf("%s/%s%s", All.OutputDir, All.CpuFile, postfix);
    fastpm_path_ensure_dirname(buf);
    if(!(FdCPU = fopen(buf, mode)))
        endrun(1, "error in opening file '%s'\n", buf);
    myfree(buf);

    if(All.OutputEnergyDebug) {
        buf = fastpm_strdup_printf("%s/%s%s", All.OutputDir, All.EnergyFile, postfix);
        fastpm_path_ensure_dirname(buf);
        if(!(FdEnergy = fopen(buf, mode)))
            endrun(1, "error in opening file '%s'\n", buf);
        myfree(buf);
    }

    buf = fastpm_strdup_printf("%s/%s%s", All.OutputDir, "sfr.txt", postfix);
    fastpm_path_ensure_dirname(buf);
    if(!(FdSfr = fopen(buf, mode)))
        endrun(1, "error in opening file '%s'\n", buf);
    myfree(buf);

    if(All.BlackHoleOn) {
        buf = fastpm_strdup_printf("%s/%s%s", All.OutputDir, "blackholes.txt", postfix);
        fastpm_path_ensure_dirname(buf);
        if(!(FdBlackHoles = fopen(buf, mode)))
            endrun(1, "error in opening file '%s'\n", buf);
        myfree(buf);
    }

}


/*!  This function closes the global log-files.
*/
static void
close_outputfiles(void)
{

    if(ThisTask != 0)		/* only the root processors writes to the log files */
        return;

    fclose(FdCPU);
    if(All.OutputEnergyDebug)
        fclose(FdEnergy);

    fclose(FdSfr);

    if(All.BlackHoleOn)
        fclose(FdBlackHoles);
}

/*! Computes conversion factors between internal code units and the
 *  cgs-system.
 */
static void
set_units(void)
{
    All.UnitTime_in_s = All.UnitLength_in_cm / All.UnitVelocity_in_cm_per_s;
    All.UnitTime_in_Megayears = All.UnitTime_in_s / SEC_PER_MEGAYEAR;

    All.G = GRAVITY / pow(All.UnitLength_in_cm, 3) * All.UnitMass_in_g * pow(All.UnitTime_in_s, 2);

    All.UnitDensity_in_cgs = All.UnitMass_in_g / pow(All.UnitLength_in_cm, 3);
    All.UnitEnergy_in_cgs = All.UnitMass_in_g * pow(All.UnitLength_in_cm, 2) / pow(All.UnitTime_in_s, 2);

    /* convert some physical input parameters to internal units */

    All.CP.Hubble = HUBBLE * All.UnitTime_in_s;
    init_cosmology(&All.CP, All.TimeIC);

    if(All.InitGasTemp < 0)
        All.InitGasTemp = All.CP.CMBTemperature / All.TimeInit;
    /*Initialise the hybrid neutrinos, after Omega_nu*/
    if(All.HybridNeutrinosOn)
        init_hybrid_nu(&All.CP.ONu.hybnu, All.CP.MNu, All.HybridVcrit, LIGHTCGS/1e5, All.HybridNuPartTime, All.CP.ONu.kBtnu);

    message(0, "Hubble (internal units) = %g\n", All.CP.Hubble);
    message(0, "G (internal units) = %g\n", All.G);
    message(0, "UnitLengh_in_cm = %g \n", All.UnitLength_in_cm);
    message(0, "UnitMass_in_g = %g \n", All.UnitMass_in_g);
    message(0, "UnitTime_in_s = %g \n", All.UnitTime_in_s);
    message(0, "UnitVelocity_in_cm_per_s = %g \n", All.UnitVelocity_in_cm_per_s);
    message(0, "UnitDensity_in_cgs = %g \n", All.UnitDensity_in_cgs);
    message(0, "UnitEnergy_in_cgs = %g \n", All.UnitEnergy_in_cgs);
    message(0, "Photon density OmegaG = %g\n",All.CP.OmegaG);
    if(!All.MassiveNuLinRespOn)
        message(0, "Massless Neutrino density OmegaNu0 = %g\n",get_omega_nu(&All.CP.ONu, 1));
    message(0, "Curvature density OmegaK = %g\n",All.CP.OmegaK);
    if(All.CP.RadiationOn) {
        /* note that this value is inaccurate if there is massive neutrino. */
        double OmegaTot = All.CP.OmegaG + All.CP.OmegaK + All.CP.Omega0 + All.CP.OmegaLambda;
        if(!All.MassiveNuLinRespOn)
            OmegaTot += get_omega_nu(&All.CP.ONu, 1);
        message(0, "Radiation is enabled in Hubble(a). "
               "Following CAMB convention: Omega_Tot - 1 = %g\n", OmegaTot - 1);
    }
    message(0, "\n");
}

/*! This function sets the (comoving) softening length of all particle
 *  types in the table All.SofteningTable[...].  We check that the physical
 *  softening length is bounded by the Softening-MaxPhys values.
 */
static void
set_softenings()
{
    int i;

    for(i = 0; i < 6; i ++)
        GravitySofteningTable[i] = All.GravitySoftening * All.MeanSeparation[1];

    /* 0: Gas is collisional */
    GravitySofteningTable[0] = All.GravitySofteningGas * All.MeanSeparation[1];

    All.MinGasHsml = All.MinGasHsmlFractional * GravitySofteningTable[1];

    for(i = 0; i < 6; i ++) {
        message(0, "GravitySoftening[%d] = %g\n", i, GravitySofteningTable[i]);
    }
}

