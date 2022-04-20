#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <ctype.h>
#include <omp.h>

#include "utils.h"

#include "walltime.h"
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
#include "metal_return.h"
#include "slotsmanager.h"
#include "hci.h"
#include "fof.h"
#include "cooling_qso_lightup.h"
#include "lightcone.h"
#include "timefac.h"
#include "neutrinos_lra.h"
#include "stats.h"

static struct ClockTable Clocks;

/*! \file run.c
 *  \brief  iterates over timesteps, main loop
 */

/*! This structure contains parameters local to the run module.*/
static struct run_params
{
    double SlotsIncreaseFactor; /* !< What percentage to increase the slot allocation by when requested*/
    int OutputDebugFields;      /* Flag whether to include a lot of debug output in snapshots*/

    double RandomParticleOffset; /* If > 0, a random shift of max RandomParticleOffset * BoxSize is applied to every particle
                                  * every time a full domain decomposition is done. The box is periodic and the offset
                                  * is subtracted on output, so this only affects the internal gravity solver.
                                  * The purpose of this is to avoid correlated errors in the tree code, which occur when
                                  * the tree opening conditions are similar in every timestep and accumulate over a
                                  * long period of time. Upstream Arepo says this substantially improves momentum conservation,
                                  * and it has the side-effect of guarding against periodicity bugs.
                                  */
    /* Cosmology */
    Cosmology CP;

    /* Code options */
    int CoolingOn;  /* if cooling is enabled */
    int HydroOn;  /*  if hydro force is enabled */
    int DensityOn;  /*  if SPH density computation is enabled */
    int TreeGravOn;     /* tree gravity force is enabled*/

    int BlackHoleOn;  /* if black holes are enabled */
    int StarformationOn;  /* if star formation is enabled */
    int MetalReturnOn; /* If late return of metals from AGB stars is enabled*/
    int LightconeOn;    /* Enable the light cone module,
                           which writes a list of particles to a file as they cross a light cone*/
    int HierarchicalGravity; /* Enable the hierarchical gravity mode. */

    int MaxDomainTimeBinDepth; /* We should redo domain decompositions every timestep, after the timestep hierarchy gets deeper than this.
                                  Essentially forces a domain decompositon every 2^MaxDomainTimeBinDepth timesteps.*/
    int FastParticleType; /*!< flags a particle species to exclude timestep calculations.*/
    /* parameters determining output frequency */
    double PairwiseActiveFraction; /* Fraction of particles active for which we do a pairwise computation instead of a tree*/

    /* parameters determining output frequency */
    double AutoSnapshotTime;    /*!< cpu-time between regularly generated snapshots. */
    double TimeBetweenSeedingSearch; /*Factor to multiply TimeInit by to find the next seeding check.*/

    double TimeMax;			/*!< marks the point of time until the simulation is to be evolved */

    int Nmesh;

    /* variables that keep track of cumulative CPU consumption */

    double TimeLimitCPU;

    /*! The scale of the short-range/long-range force split in units of FFT-mesh cells */
    double Asmth;
    enum ShortRangeForceWindowType ShortRangeForceWindowType;

    /* some filenames */
    char OutputDir[100],
         FOFFileBase[100];

    int SnapshotWithFOF; /*Flag that doing FOF for snapshot outputs is on*/

    int RandomSeed; /*Initial seed for the random number table*/
} All;

/*Set the global parameters*/
void
set_all_global_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        /* Start reading the values */
        param_get_string2(ps, "OutputDir", All.OutputDir, sizeof(All.OutputDir));
        param_get_string2(ps, "FOFFileBase", All.FOFFileBase, sizeof(All.FOFFileBase));

        All.CP.CMBTemperature = param_get_double(ps, "CMBTemperature");
        All.CP.RadiationOn = param_get_int(ps, "RadiationOn");
        All.CP.Omega0 = param_get_double(ps, "Omega0");
        All.CP.OmegaBaryon = param_get_double(ps, "OmegaBaryon");
        All.CP.OmegaLambda = param_get_double(ps, "OmegaLambda");
        All.CP.Omega_fld = param_get_double(ps, "Omega_fld");
        if(All.CP.OmegaLambda > 0 && All.CP.Omega_fld > 0)
            endrun(0, "Cannot have OmegaLambda and Omega_fld (evolving dark energy) at the same time!\n");
        All.CP.w0_fld = param_get_double(ps,"w0_fld");
        All.CP.wa_fld = param_get_double(ps,"wa_fld");
        All.CP.Omega_ur = param_get_double(ps, "Omega_ur");
        All.CP.HubbleParam = param_get_double(ps, "HubbleParam");

        All.OutputDebugFields = param_get_int(ps, "OutputDebugFields");

        All.TimeMax = param_get_double(ps, "TimeMax");
        All.Asmth = param_get_double(ps, "Asmth");
        All.ShortRangeForceWindowType = (enum ShortRangeForceWindowType) param_get_enum(ps, "ShortRangeForceWindowType");
        All.Nmesh = param_get_int(ps, "Nmesh");

        All.CoolingOn = param_get_int(ps, "CoolingOn");
        All.HydroOn = param_get_int(ps, "HydroOn");
        All.DensityOn = param_get_int(ps, "DensityOn");
        All.TreeGravOn = param_get_int(ps, "TreeGravOn");
        All.LightconeOn = param_get_int(ps, "LightconeOn");
        All.HierarchicalGravity = param_get_int(ps, "HierarchicalGravityOn");
        All.FastParticleType = param_get_int(ps, "FastParticleType");
        All.PairwiseActiveFraction = param_get_double(ps, "PairwiseActiveFraction");
        All.TimeLimitCPU = param_get_double(ps, "TimeLimitCPU");
        All.AutoSnapshotTime = param_get_double(ps, "AutoSnapshotTime");
        All.TimeBetweenSeedingSearch = param_get_double(ps, "TimeBetweenSeedingSearch");
        All.RandomParticleOffset = param_get_double(ps, "RandomParticleOffset");
        /* Convert to a fraction of the box, from a fraction of a PM mesh cell*/
        All.RandomParticleOffset /= All.Nmesh;

        All.SlotsIncreaseFactor = param_get_double(ps, "SlotsIncreaseFactor");

        All.SnapshotWithFOF = param_get_int(ps, "SnapshotWithFOF");

        All.RandomSeed = param_get_int(ps, "RandomSeed");

        All.BlackHoleOn = param_get_int(ps, "BlackHoleOn");

        All.StarformationOn = param_get_int(ps, "StarformationOn");
        All.MetalReturnOn = param_get_int(ps, "MetalReturnOn");
        All.MaxDomainTimeBinDepth = param_get_int(ps, "MaxDomainTimeBinDepth");

        /*Massive neutrino parameters*/
        All.CP.MassiveNuLinRespOn = param_get_int(ps, "MassiveNuLinRespOn");
        All.CP.HybridNeutrinosOn = param_get_int(ps, "HybridNeutrinosOn");
        All.CP.MNu[0] = param_get_double(ps, "MNue");
        All.CP.MNu[1] = param_get_double(ps, "MNum");
        All.CP.MNu[2] = param_get_double(ps, "MNut");
        All.CP.HybridVcrit = param_get_double(ps, "Vcrit");
        All.CP.HybridNuPartTime = param_get_double(ps, "NuPartTime");
        if(All.CP.MassiveNuLinRespOn && !All.CP.RadiationOn)
            endrun(2, "You have enabled (kspace) massive neutrinos without radiation, but this will give an inconsistent cosmology!\n");
        /*End massive neutrino parameters*/

        if(All.StarformationOn != 0 && All.CoolingOn == 0)
        {
                endrun(1, "You try to use the code with star formation enabled,\n"
                          "but you did not switch on cooling.\nThis mode is not supported.\n");
        }
    }
    MPI_Bcast(&All, sizeof(All), MPI_BYTE, 0, MPI_COMM_WORLD);
}

/*! This function performs the initial set-up of the simulation. First, the
 *  parameterfile is set, then routines for setting units, reading
 *  ICs/restart-files are called, auxialiary memory is allocated, etc.
 */
inttime_t
begrun(const int RestartFlag, int RestartSnapNum, struct header_data * head)
{
    if(RestartFlag == 1) {
        RestartSnapNum = find_last_snapnum(All.OutputDir);
        message(0, "Last Snapshot number is %d.\n", RestartSnapNum);
    }

    petapm_module_init(omp_get_max_threads());
    petaio_init();
    walltime_init(&Clocks);

    *head = petaio_read_header(RestartSnapNum, All.OutputDir, &All.CP);
    /*Set Nmesh to triple the mean grid spacing of the dark matter by default.*/
    if(All.Nmesh  < 0)
        All.Nmesh = 3*pow(2, (int)(log(head->NTotal[1])/3./log(2)) );
    if(head->neutrinonk <= 0)
        head->neutrinonk = All.Nmesh;

    slots_init(All.SlotsIncreaseFactor * PartManager->MaxPart, SlotsManager);
    /* Enable the slots: stars and BHs are allocated if there are some,
     * or if some will form*/
    if(head->NTotalInit[0] > 0)
        slots_set_enabled(0, sizeof(struct sph_particle_data), SlotsManager);
    if(All.StarformationOn || head->NTotalInit[4] > 0)
        slots_set_enabled(4, sizeof(struct star_particle_data), SlotsManager);
    if(All.BlackHoleOn || head->NTotalInit[5] > 0)
        slots_set_enabled(5, sizeof(struct bh_particle_data), SlotsManager);

    const struct UnitSystem units = get_unitsystem(head->UnitLength_in_cm, head->UnitMass_in_g, head->UnitVelocity_in_cm_per_s);
    /* convert some physical input parameters to internal units */
    init_cosmology(&All.CP, head->TimeIC, units);

    check_units(&All.CP, units);

#ifdef DEBUG
    char * pidfile = fastpm_strdup_printf("%s/%s", All.OutputDir, "PIDs.txt");

    MPIU_write_pids(pidfile);
    myfree(pidfile);
#endif

    init_forcetree_params(All.FastParticleType);

    init_cooling_and_star_formation(All.CoolingOn, All.StarformationOn, &All.CP, head->MassTable[0], head->BoxSize, units);

    gravshort_fill_ntab(All.ShortRangeForceWindowType, All.Asmth);

    set_random_numbers(All.RandomSeed);

    if(All.LightconeOn)
        lightcone_init(&All.CP, head->TimeSnapshot, head->UnitLength_in_cm, All.OutputDir);

    init_timeline(RestartSnapNum, All.TimeMax, head, All.SnapshotWithFOF);

    /* Get the nk and do allocation. */
    if(All.CP.MassiveNuLinRespOn)
        init_neutrinos_lra(head->neutrinonk, head->TimeIC, All.TimeMax, All.CP.Omega0, &All.CP.ONu, All.CP.UnitTime_in_s, CM_PER_MPC);

    /* ... read initial model and initialise the times*/
    inttime_t ti_init = init(RestartSnapNum, All.OutputDir, head, &All.CP);

    if(RestartSnapNum < 0) {
        DomainDecomp ddecomp[1] = {0};
        domain_decompose_full(ddecomp); /* do initial domain decomposition (gives equal numbers of particles) so density() is safe*/
        /* On first run, generate smoothing lengths and set initial entropies based on CMB temperature*/
        setup_smoothinglengths(RestartSnapNum, ddecomp, &All.CP, All.BlackHoleOn, get_MinEgySpec(), units.UnitInternalEnergy_in_cgs, ti_init, head->TimeSnapshot, head->NTotalInit[0]);
        domain_free(ddecomp);
    }
    else
        /* When we restart, validate the SPH properties of the particles.
         * This also allows us to increase MinEgySpec on a restart if we choose.*/
        check_density_entropy(&All.CP, get_MinEgySpec(), head->TimeSnapshot);

    return ti_init;
}

/* Small function to decide - collectively - whether to use pairwise gravity this step*/
static int
use_pairwise_gravity(ActiveParticles * Act, struct part_manager_type * PartManager)
{
    /* Find total number of active particles*/
    int64_t total_active, total_particle;
    MPI_Allreduce(&Act->NumActiveParticle, &total_active, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&PartManager->NumPart, &total_particle, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

    /* Since the pairwise step is O(N^2) and tree is O(NlogN) we should scale the condition like O(N)*/
    return total_active < All.PairwiseActiveFraction * total_particle;
}

#ifdef DEBUG
static void
check_kick_drift_times(struct part_manager_type * PartManager, inttime_t ti_current)
{
    int i;
    int bad = 0;
    #pragma omp parallel for reduction(+: bad)
    for(i = 0; i < PartManager->NumPart; i++) {
        const struct particle_data * pp = &PartManager->Base[i];
        if(pp->IsGarbage || pp->Swallowed)
            continue;
        if ( ((pp->Type == 0 || pp->Type == 5) && is_timebin_active(pp->TimeBinHydro, ti_current) && pp->Ti_drift != pp->Ti_kick_hydro) ||
           (is_timebin_active(pp->TimeBinGravity, ti_current) && pp->Ti_drift != pp->Ti_kick_grav) ) {
            message(1, "Bad timestep sync: Particle id %ld type %d hydro timebin: %d grav timebin: %d drift %x kick_hydro %x kick_grav %x\n", pp->ID, pp->Type, pp->TimeBinHydro, pp->TimeBinGravity, pp->Ti_drift, pp->Ti_kick_hydro, pp->Ti_kick_grav);
            bad++;
        }
    }
    if(bad)
        endrun(7, "Poor timestep sync for %d particles\n", bad);
}
#endif

/*! This routine contains the main simulation loop that iterates over
 * single timesteps. The loop terminates when the cpu-time limit is
 * reached, when a `stop' file is found in the output directory, or
 * when the simulation ends because we arrived at TimeMax.
 */
void
run(const int RestartSnapNum, const inttime_t ti_init, const struct header_data * header)
{
    /*Number of timesteps performed this run*/
    int NumCurrentTiStep = 0;
    /*Is gas physics enabled?*/
    int GasEnabled = SlotsManager->info[0].enabled;

    HCIManager HCI_DEFAULT_MANAGER[1] = {0};
    hci_init(HCI_DEFAULT_MANAGER, All.OutputDir, All.TimeLimitCPU, All.AutoSnapshotTime, All.SnapshotWithFOF);

    const struct UnitSystem units = get_unitsystem(header->UnitLength_in_cm, header->UnitMass_in_g, header->UnitVelocity_in_cm_per_s);

    int SnapshotFileCount = RestartSnapNum;

    const double MinEgySpec = get_MinEgySpec();

    PetaPM pm = {0};
    gravpm_init_periodic(&pm, PartManager->BoxSize, All.Asmth, All.Nmesh, All.CP.GravInternal);

    DomainDecomp ddecomp[1] = {0};

    /* Stored scale factor of the next black hole seeding check*/
    double TimeNextSeedingCheck = header->TimeSnapshot;

    walltime_measure("/Misc");
    struct OutputFD fds;
    open_outputfiles(RestartSnapNum, &fds, All.OutputDir, All.BlackHoleOn, All.StarformationOn);

    write_cpu_log(NumCurrentTiStep, header->TimeSnapshot, fds.FdCPU, Clocks.ElapsedTime); /* produce some CPU usage info */

    DriftKickTimes times = init_driftkicktime(ti_init);

    double atime = get_atime(times.Ti_Current);

    while(1) /* main loop */
    {
        /* Find next synchronization point and the timebins active during this timestep.
         *
         * Note that on startup, P[i].TimeBin == 0 for all particles,
         * all bins except the zeroth are inactive and so we return 0 from this function.
         * This ensures we run the force calculation for the first timestep.
         */
        inttime_t Ti_Next = find_next_kick(times.Ti_Current, times.mintimebin);
        inttime_t Ti_Last = times.Ti_Current;

        times.Ti_Current = Ti_Next;

        /*Convert back to floating point time*/
        double newatime = get_atime(times.Ti_Current);
        if(newatime < atime)
            endrun(1, "Negative timestep: %g New Time: %g Old time %g!\n", newatime - atime, newatime, atime);
        atime = newatime;

        /* Compute the list of particles that cross a lightcone and write it to disc.*/
        if(All.LightconeOn)
            lightcone_compute(atime, PartManager->BoxSize, &All.CP, Ti_Last, Ti_Next);

        int is_PM = is_PM_timestep(&times);

        SyncPoint * next_sync; /* if we are out of planned sync points, terminate */
        SyncPoint * planned_sync; /* NULL; if the step is not a planned sync point. */

        next_sync = find_next_sync_point(times.Ti_Current);
        planned_sync = find_current_sync_point(times.Ti_Current);

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
            update_random_offset(PartManager, rel_random_shift, All.RandomParticleOffset);
        }

        int extradomain = is_timebin_active(times.mintimebin + All.MaxDomainTimeBinDepth, times.Ti_Current);
        /* drift and ddecomp decomposition */
        /* at first step this is a noop */
        if(extradomain || is_PM) {
            /* Sync positions of all particles */
            drift_all_particles(Ti_Last, times.Ti_Current, &All.CP, rel_random_shift);
            /* full decomposition rebuilds the domain, needs keys.*/
            domain_decompose_full(ddecomp);
        } else {
            /* FIXME: add a parameter for ddecomp_decompose_incremental */
            /* currently we drift all particles every step */
            /* If it is not a PM step, do a shorter version
             * of the ddecomp decomp which just exchanges particles.*/
            struct DriftData drift;
            drift.CP = &All.CP;
            drift.ti0 = Ti_Last;
            drift.ti1 = times.Ti_Current;
            domain_maintain(ddecomp, &drift);
        }
        update_lastactive_drift(&times);


        ActiveParticles Act = {0};
        rebuild_activelist(&Act, &times, NumCurrentTiStep, atime);

        set_random_numbers(All.RandomSeed + times.Ti_Current);

        /* Are the particle neutrinos gravitating this timestep?
         * If so we need to add them to the tree.*/
        int HybridNuTracer = hybrid_nu_tracer(&All.CP, atime);

        /* Collective: total number of active particles must be small enough*/
        int pairwisestep = use_pairwise_gravity(&Act, PartManager);

        MyFloat * GradRho = NULL;
        if(sfr_need_to_compute_sph_grad_rho())
            GradRho = (MyFloat *) mymalloc2("SPH_GradRho", sizeof(MyFloat) * 3 * SlotsManager->info[0].size);

        /* density() happens before gravity because it also initializes the predicted variables.
        * This ensures that prediction consistently uses the grav and hydro accel from the
        * timestep before this one, which matches Gadget-2/3. It was tested to make a small difference,
        * since prediction is only really used for artificial viscosity.
        *
        * Doing it first also means the density is up to date for
        * adaptive gravitational softenings. */
        if(GasEnabled)
        {
            ForceTree gasTree = {0};
            /* Just gas, no moments*/
            force_tree_rebuild_mask(&gasTree, ddecomp, GASMASK, HybridNuTracer, All.OutputDir);
            walltime_measure("/SPH/Build");

            /*Allocate the memory for predicted SPH data.*/
            struct sph_pred_data sph_predicted = slots_allocate_sph_pred_data(SlotsManager->info[0].size);

            if(All.DensityOn)
                density(&Act, 1, DensityIndependentSphOn(), All.BlackHoleOn, MinEgySpec, times, &All.CP, &sph_predicted, GradRho, &gasTree);  /* computes density, and pressure */

            /***** update smoothing lengths in tree *****/
            force_update_hmax(Act.ActiveParticle, Act.NumActiveParticle, &gasTree, ddecomp);
            /***** hydro forces *****/
            MPIU_Barrier(MPI_COMM_WORLD);

            /* adds hydrodynamical accelerations and computes du/dt  */
            if(All.HydroOn) {
                /* In Gadget-4 this is optionally split into two, with the pressure force
                 * computed on either side of the cooling term. Volker Springel confirms that
                 * he has never encountered a simulation where this matters in practice, probably because
                 * it would only be important in very dissipative environments where the SPH noise is fairly large
                 * and there is no opportunity for errors to build up.*/
                hydro_force(&Act, atime, &sph_predicted, MinEgySpec, times, &All.CP, &gasTree);
            }
            /* Scratch data cannot be used checkpoint because FOF does an exchange.*/
            slots_free_sph_pred_data(&sph_predicted);
            force_tree_free(&gasTree);

            /* Hydro half-kick after hydro force, as not done with the gravity.*/
            if(All.HierarchicalGravity)
                apply_hydro_half_kick(&Act, &All.CP, &times, atime, MinEgySpec);
        }

        /* The opening criterion for the gravtree
        * uses the *total* gravitational acceleration
        * from the last timestep, GravPM+GravAccel.
        * So we must compute GravAccel for this timestep
        * before gravpm_force() writes the PM acc. for
        * this timestep to GravPM. Note initially both
        * are zero and so the tree is opened maximally
        * on the first timestep.*/
        const double rho0 = All.CP.Omega0 * 3 * All.CP.Hubble * All.CP.Hubble / (8 * M_PI * All.CP.GravInternal);

        /* Build the force tree for tree gravity.*/
        ForceTree Tree = {0};
        int anygravactive = MPIU_Any(Act.NumActiveGravity, MPI_COMM_WORLD);

        /* Gravitational acceleration here*/
        if(All.HierarchicalGravity) {
            hierarchical_gravity_accelerations(&Act, &pm, ddecomp, &times, HybridNuTracer, All.FastParticleType, &All.CP, All.OutputDir);
        }
        else {
            ActiveParticles allpart = {0};
            allpart.NumActiveParticle = PartManager->NumPart;
            /* We need a tree if we will do a short-range gravity treewalk.
            * We also need one for PM so we can do the indexing.
            * So this condition is true and the next false only if !TreeGravOn.*/
            if(is_PM || (All.TreeGravOn && anygravactive))
                force_tree_rebuild(&Tree, ddecomp, &allpart, HybridNuTracer, !pairwisestep && All.TreeGravOn, All.OutputDir);

            if(All.TreeGravOn && (is_PM || anygravactive)) {
                /* Do a short range pairwise only step if desired*/
                if(pairwisestep) {
                    struct gravshort_tree_params gtp = get_gravshort_treepar();
                    grav_short_pair(&Act, &pm, &Tree, gtp.Rcut, rho0, HybridNuTracer, All.FastParticleType);
                }
                else
                    grav_short_tree(&Act, &pm, &Tree, rho0, HybridNuTracer, All.FastParticleType, times.Ti_Current);
            }
        }

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
            ActiveParticles allpart = {0};
            allpart.NumActiveParticle = PartManager->NumPart;

            if(is_PM && All.HierarchicalGravity)
                force_tree_rebuild(&Tree, ddecomp, &allpart, HybridNuTracer, !pairwisestep && All.TreeGravOn, All.OutputDir);

            gravpm_force(&pm, &Tree, &All.CP, atime, units.UnitLength_in_cm, All.OutputDir, header->TimeIC, All.FastParticleType);

            /* compute and output energy statistics if desired. */
            if(fds.FdEnergy)
                energy_statistics(fds.FdEnergy, atime, PartManager);
        }

        if(!All.HierarchicalGravity){
            /* Do both short-range gravity and hydro kicks.
             * Need a scale factor for entropy and velocity limiters.
             * For hierarchical gravity the short-range kick is done above.
             * Synchronises TiKick and TiDrift for the active particles. */
            apply_half_kick(&Act, &All.CP, &times, atime, MinEgySpec);
        }

        /* Sets Ti_Kick in the times structure.*/
        update_kick_times(&times);

        if(is_PM) {
            apply_PM_half_kick(&All.CP, &times);
            /* Now we have computed the total acceleration, set old acc for the next PM step.*/
            grav_set_oldaccs(All.CP.GravInternal);
        }

        MPIU_Barrier(MPI_COMM_WORLD);
        message(0, "Forces computed.\n");

        /* Cooling and extra physics show up as a source term in the evolution equations.
         * Formally you can write the structure of the partial differential equations:
           dU/dt +  div(F) = S
         * where the cooling, BH and SFR are the source term S.
         * The extra physics is done after the kick, using a Strang split operator.
         * Gadget3/Arepo tries to follow the general operator splitting ansatz (often called Strang splitting).
         * Here you alternate the evolution under the operator generating the time evolution of the homogenous system (ie, without S)
         * with the operator generating the time evolution under the source function alone.
         * This means to advance the full system by dt, you first evolve dU/dt = S
         * by dt, and then dU/dt +  div(F) = 0 by dt.
         * [Actually, for second-order convergence in time, you should rather evolve S for dt/2, then the homogenous part for dt, and then S again for dt/2.]

         * The operator-split approach offers a number of practical advantages when the source function is stiff.
         * You can, for example, solve dU/dt = S in a robust and stable fashion with an implict solver, whereas the Gadget2 approach is severely challenged
         * and either requires an artficial clipping of the maximum allowed cooling rate, or a severe reduction of the timestep, otherwise the
         * predicted entropy due to cooling someting during the timestep can become severely wrong. Also, the source term approach can be easily
         * used to treat effectively instantaneous injections of energy (like from BHs), which is again hard to properly incorporate in the
         * time-integration approach where you want to have a "full" dU/dt all times. (Volker Springel 2020).
         */
        if(GasEnabled)
        {
            /* Do this before sfr and bh so the gas hsml always contains DesNumNgb neighbours.*/
            if(All.MetalReturnOn) {
                double AvgGasMass = All.CP.OmegaBaryon * 3 * All.CP.Hubble * All.CP.Hubble / (8 * M_PI * All.CP.GravInternal) * pow(PartManager->BoxSize, 3) / header->NTotalInit[0];
                metal_return(&Act, ddecomp, &All.CP, atime, AvgGasMass);
            }

            /* this will find new black hole seed halos.
             * Note: the FOF code does not know about garbage particles,
             * so ensure we do not have garbage present when we call this.
             * Also a good idea to only run it on a PM step.
             * This does not break the tree because the new black holes do not move or change mass, just type.
             * It does not matter that the velocities are half a step off because they are not used in the FoF code.*/
            if (is_PM && ((All.BlackHoleOn && atime >= TimeNextSeedingCheck) ||
                (during_helium_reionization(1/atime - 1) && need_change_helium_ionization_fraction(atime)))) {

                /* Seeding: builds its own tree.*/
                FOFGroups fof = fof_fof(ddecomp, 0, MPI_COMM_WORLD);
                if(All.BlackHoleOn && atime >= TimeNextSeedingCheck) {
                    fof_seed(&fof, &Act, atime, MPI_COMM_WORLD);
                    TimeNextSeedingCheck = atime * All.TimeBetweenSeedingSearch;
                }

                if(during_helium_reionization(1/atime - 1)) {
                    /* Helium reionization by switching on quasar bubbles*/
                    do_heiii_reionization(atime, &fof, ddecomp, &All.CP, units.UnitInternalEnergy_in_cgs, fds.FdHelium);
                }
                fof_finish(&fof);
            }

            /* Note that the tree here may be freed, if we are not a gravity-active timestep,
             * or if we are a PM step.*/
            /* If we didn't build a tree for gravity, we need to build one in BH or in winds.
             * The BH tree needs stars for DF, gas + BH for accretion and swallowing and technically
             * needs DM for DF and repositioning, although it doesn't do much there. It is needed if any BHs
             * are active (ie, not for the shortest timestep).
             * The wind tree is needed if any new stars are formed and needs DM and gas (for the default wind model).
             */
            /* Black hole accretion and feedback */
            if(All.BlackHoleOn)
                blackhole(&Act, atime, &All.CP, &Tree, ddecomp, units, fds.FdBlackHoles, fds.FdBlackholeDetails);

            /**** radiative cooling and star formation *****/
            if(All.CoolingOn)
                cooling_and_starformation(&Act, atime, get_dloga_for_bin(times.mintimebin, times.Ti_Current), &Tree, ddecomp, &All.CP, GradRho, fds.FdSfr);
        }
        /* We don't need this timestep's tree anymore.*/
        force_tree_free(&Tree);

        if(GradRho) {
            myfree(GradRho);
            GradRho = NULL;
        }

        /* If a snapshot is requested, write it.         *
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
            WriteFOF |= action->write_fof;
        }
        if(WriteSnapshot || WriteFOF) {
            /* Get a new snapshot*/
            SnapshotFileCount++;
            /* The accel may have created garbage -- collect them before writing a snapshot.
             * If we do collect, reset active list size.*/
            int compact[6] = {0};
            if(slots_gc(compact, PartManager, SlotsManager))
                Act.NumActiveParticle = PartManager->NumPart;
        }
        FOFGroups fof = {0};
        if(WriteFOF) {
            /* Compute FOF and assign GrNr so it can be written in checkpoint.*/
            fof = fof_fof(ddecomp, 1, MPI_COMM_WORLD);
        }

        /* WriteFOF just reminds the checkpoint code to save GroupID*/
        if(WriteSnapshot)
            write_checkpoint(SnapshotFileCount, WriteFOF, All.MetalReturnOn, atime, &All.CP, All.OutputDir, All.OutputDebugFields);

        /* Save FOF tables after checkpoint so that if there is a FOF save bug we have particle tables available to debug it*/
        if(WriteFOF) {
            fof_save_groups(&fof, All.OutputDir, All.FOFFileBase, SnapshotFileCount, &All.CP, atime, header->MassTable, All.MetalReturnOn, All.BlackHoleOn, MPI_COMM_WORLD);
            fof_finish(&fof);
        }

#ifdef DEBUG
        check_kick_drift_times(PartManager, times.Ti_Current);
#endif
        write_cpu_log(NumCurrentTiStep, atime, fds.FdCPU, Clocks.ElapsedTime);    /* produce some CPU usage info */

        report_memory_usage("RUN");

        if(!next_sync || stop) {
            /* out of sync points, or a requested stop, the run has finally finished! Yay.*/
            if(action->type == HCI_TIMEOUT)
                message(0, "Stopping: not enough time for another PM step before TimeLimitCPU is reached.\n");
            break;
        }

        /* more steps to go. */

        /* assign new timesteps to the active particles,
         * now that we know they have synched TiKick and TiDrift,
         * and advance the PM timestep.*/
        int badtimestep=0;
        if(!All.HierarchicalGravity) {
            const double asmth = pm.Asmth * PartManager->BoxSize / pm.Nmesh;
            badtimestep = find_timesteps(&Act, &times, atime, All.FastParticleType, &All.CP, asmth, NumCurrentTiStep == 0);
            /* Update velocity and ti_kick to the new step, with the newly computed step size. Unsyncs ti_kick and ti_drift.
             * Both hydro and gravity are kicked.*/
            apply_half_kick(&Act, &All.CP, &times, atime, MinEgySpec);
        } else {
            /* This finds the gravity timesteps, computes the gravitational forces
             * and kicks the particles on the gravitational timeline.
             * Note this is separated from the first force computation because
             * each timebin has a force done individually and we do not store the acceleration hierarchy.
             * This does mean we double the cost of the force evaluations.*/
            badtimestep = hierarchical_gravity_and_timesteps(&Act, &pm, ddecomp, &times, atime, HybridNuTracer, All.FastParticleType, &All.CP, All.OutputDir);
            if(GasEnabled) {
                /* Find hydro timesteps and apply the hydro kick, unsyncing the drift and kick times. */
                badtimestep += find_hydro_timesteps(&Act, &times, atime, &All.CP, NumCurrentTiStep == 0);
                /* If there is no hydro kick to do we still need to update the kick times.*/
                apply_hydro_half_kick(&Act, &All.CP, &times, atime, MinEgySpec);
            }
        }

        /* Set ti_kick in the time structure*/
        update_kick_times(&times);

        if(badtimestep) {
            message(0, "bad timestep spotted: terminating and saving snapshot.\n");
            dump_snapshot("TIMESTEP-DUMP", atime, &All.CP, All.OutputDir);
            endrun(0, "Ending due to bad timestep");
        }

        if(is_PM) {
            apply_PM_half_kick(&All.CP, &times);
        }

        /* We can now free the active list: the new step have new active particles*/
        free_activelist(&Act);

        NumCurrentTiStep++;
    }

    close_outputfiles(&fds);
}

/* Run various checks on the gravity code. Check that the short-range/long-range force split is working.*/
void
runtests(const int RestartSnapNum, const inttime_t Ti_Current, const struct header_data * header)
{
    run_gravity_test(RestartSnapNum, &All.CP, All.Asmth, All.Nmesh, All.FastParticleType, Ti_Current, All.OutputDir, header);
}

void
runfof(const int RestartSnapNum, const inttime_t Ti_Current, const struct header_data * header)
{
    PetaPM pm = {0};
    gravpm_init_periodic(&pm, PartManager->BoxSize, All.Asmth, All.Nmesh, All.CP.GravInternal);
    DomainDecomp ddecomp[1] = {0};
    /* ... read in initial model */

    domain_decompose_full(ddecomp);	/* do initial domain decomposition (gives equal numbers of particles) */

    DriftKickTimes times = init_driftkicktime(Ti_Current);
    /*FoF needs a tree*/
    int HybridNuGrav = hybrid_nu_tracer(&All.CP, header->TimeSnapshot);
    /* Regenerate the star formation rate for the FOF table.*/
    if(All.StarformationOn) {
        ActiveParticles Act = {0};
        Act.NumActiveParticle = PartManager->NumPart;
        MyFloat * GradRho = NULL;
        if(sfr_need_to_compute_sph_grad_rho()) {
            ForceTree gasTree = {0};
            GradRho = (MyFloat *) mymalloc2("SPH_GradRho", sizeof(MyFloat) * 3 * SlotsManager->info[0].size);
            /*Allocate the memory for predicted SPH data.*/
            struct sph_pred_data sph_predicted = slots_allocate_sph_pred_data(SlotsManager->info[0].size);
            force_tree_rebuild(&gasTree, ddecomp, &Act, HybridNuGrav, 0, All.OutputDir);
            /* computes GradRho with a treewalk. No hsml update as we are reading from a snapshot.*/
            density(&Act, 0, 0, All.BlackHoleOn, get_MinEgySpec(), times, &All.CP, &sph_predicted, GradRho, &gasTree);
            force_tree_free(&gasTree);
            slots_free_sph_pred_data(&sph_predicted);
        }
        ForceTree Tree = {0};
        cooling_and_starformation(&Act, header->TimeSnapshot, 0, &Tree, ddecomp, &All.CP, GradRho, NULL);
        if(GradRho)
            myfree(GradRho);
    }
    FOFGroups fof = fof_fof(ddecomp, 1, MPI_COMM_WORLD);
    fof_save_groups(&fof, All.OutputDir, All.FOFFileBase, RestartSnapNum, &All.CP, header->TimeSnapshot, header->MassTable, All.MetalReturnOn, All.BlackHoleOn, MPI_COMM_WORLD);
    fof_finish(&fof);
}

void
runpower(const struct header_data * header)
{
    PetaPM pm = {0};
    gravpm_init_periodic(&pm, PartManager->BoxSize, All.Asmth, All.Nmesh, All.CP.GravInternal);
    DomainDecomp ddecomp[1] = {0};
    /* ... read in initial model */
    domain_decompose_full(ddecomp);	/* do initial domain decomposition (gives equal numbers of particles) */
    ActiveParticles Act = {0};
    Act.NumActiveParticle = PartManager->NumPart;

    /*PM needs a tree*/
    ForceTree Tree = {0};
    int HybridNuGrav = hybrid_nu_tracer(&All.CP, header->TimeSnapshot);
    force_tree_rebuild(&Tree, ddecomp, &Act, HybridNuGrav, 1, All.OutputDir);
    gravpm_force(&pm, &Tree, &All.CP, header->TimeSnapshot, header->UnitLength_in_cm, All.OutputDir, header->TimeSnapshot, All.FastParticleType);
    force_tree_free(&Tree);
}
