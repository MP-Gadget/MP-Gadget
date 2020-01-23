#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <gsl/gsl_sf_gamma.h>

#include "utils.h"

#include "allvars.h"
#include "cooling.h"
#include "forcetree.h"
#include "density.h"

#include "timefac.h"
#include "petaio.h"
#include "domain.h"
#include "walltime.h"
#include "slotsmanager.h"
#include "hydra.h"
#include "sfr_eff.h"
#include "exchange.h"
#include "fof.h"
#include "timestep.h"
#include "timebinmgr.h"
#include "cosmology.h"

/*! \file init.c
 *  \brief code for initialisation of a simulation from initial conditions
 */

/*! This structure contains data which is the SAME for all tasks (mostly code parameters read from the
 * parameter file).  Holding this data in a structure is convenient for writing/reading the restart file, and
 * it allows the introduction of new global variables in a simple way. The only thing to do is to introduce
 * them into this structure.
 */
struct global_data_all_processes All;

/*Set the parameters of the hydro module*/
void
set_init_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    if(ThisTask == 0) {
        /* Start reading the values */
        param_get_string2(ps, "InitCondFile", All.InitCondFile, sizeof(All.InitCondFile));
        param_get_string2(ps, "OutputDir", All.OutputDir, sizeof(All.OutputDir));
        param_get_string2(ps, "SnapshotFileBase", All.SnapshotFileBase, sizeof(All.SnapshotFileBase));
        param_get_string2(ps, "FOFFileBase", All.FOFFileBase, sizeof(All.FOFFileBase));
        param_get_string2(ps, "EnergyFile", All.EnergyFile, sizeof(All.EnergyFile));
        All.OutputEnergyDebug = param_get_int(ps, "EnergyFile");
        param_get_string2(ps, "CpuFile", All.CpuFile, sizeof(All.CpuFile));

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

        All.OutputPotential = param_get_int(ps, "OutputPotential");
        All.OutputHeliumFractions = param_get_int(ps, "OutputHeliumFractions");
        All.OutputDebugFields = param_get_int(ps, "OutputDebugFields");

        All.TimeMax = param_get_double(ps, "TimeMax");
        All.Asmth = param_get_double(ps, "Asmth");
        All.ShortRangeForceWindowType = param_get_enum(ps, "ShortRangeForceWindowType");
        All.Nmesh = param_get_int(ps, "Nmesh");

        All.HydroCostFactor = param_get_double(ps, "HydroCostFactor");

        All.CoolingOn = param_get_int(ps, "CoolingOn");
        All.HydroOn = param_get_int(ps, "HydroOn");
        All.DensityOn = param_get_int(ps, "DensityOn");
        All.TreeGravOn = param_get_int(ps, "TreeGravOn");
        All.FastParticleType = param_get_int(ps, "FastParticleType");
        All.TimeLimitCPU = param_get_double(ps, "TimeLimitCPU");
        All.AutoSnapshotTime = param_get_double(ps, "AutoSnapshotTime");
        All.TimeBetweenSeedingSearch = param_get_double(ps, "TimeBetweenSeedingSearch");
        All.RandomParticleOffset = param_get_double(ps, "RandomParticleOffset");

        All.GravitySoftening = param_get_double(ps, "GravitySoftening");
        All.GravitySofteningGas = param_get_double(ps, "GravitySofteningGas");

        All.PartAllocFactor = param_get_double(ps, "PartAllocFactor");
        All.SlotsIncreaseFactor = param_get_double(ps, "SlotsIncreaseFactor");

        All.SnapshotWithFOF = param_get_int(ps, "SnapshotWithFOF");

        All.RandomSeed = param_get_int(ps, "RandomSeed");

        All.BlackHoleOn = param_get_int(ps, "BlackHoleOn");
        All.WriteBlackHoleDetails = param_get_int(ps,"WriteBlackHoleDetails");

        All.StarformationOn = param_get_int(ps, "StarformationOn");
        All.WindOn = param_get_int(ps, "WindOn");

        All.InitGasTemp = param_get_double(ps, "InitGasTemp");

        /*Massive neutrino parameters*/
        All.MassiveNuLinRespOn = param_get_int(ps, "MassiveNuLinRespOn");
        All.HybridNeutrinosOn = param_get_int(ps, "HybridNeutrinosOn");
        All.CP.MNu[0] = param_get_double(ps, "MNue");
        All.CP.MNu[1] = param_get_double(ps, "MNum");
        All.CP.MNu[2] = param_get_double(ps, "MNut");
        All.HybridVcrit = param_get_double(ps, "Vcrit");
        All.HybridNuPartTime = param_get_double(ps, "NuPartTime");
        if(All.MassiveNuLinRespOn && !All.CP.RadiationOn)
            endrun(2, "You have enabled (kspace) massive neutrinos without radiation, but this will give an inconsistent cosmology!\n");
        /*End massive neutrino parameters*/

        if(All.StarformationOn == 0)
        {
            if(All.WindOn == 1) {
                endrun(1, "You try to use the code with wind enabled,\n"
                          "but you did not switch on starformation.\nThis mode is not supported.\n");
            }
        } else {
            if(All.CoolingOn == 0)
            {
                endrun(1, "You try to use the code with star formation enabled,\n"
                          "but you did not switch on cooling.\nThis mode is not supported.\n");
            }
        }
    }
    MPI_Bcast(&All, sizeof(All), MPI_BYTE, 0, MPI_COMM_WORLD);
}

static void check_omega(void);
static void check_positions(void);

static void
setup_smoothinglengths(int RestartSnapNum, DomainDecomp * ddecomp);

/*! This function reads the initial conditions, and allocates storage for the
 *  tree(s). Various variables of the particle data are initialised and An
 *  intial domain decomposition is performed. If SPH particles are present,
 *  the initial SPH smoothing lengths are determined.
 */
void init(int RestartSnapNum, DomainDecomp * ddecomp)
{
    int i, j;

    /*Add TimeInit and TimeMax to the output list*/
    if (RestartSnapNum < 0) {
        /* allow a first snapshot at IC time; */
        setup_sync_points(All.TimeIC, 0.0);
    } else {
        /* skip dumping the exactly same snapshot */
        setup_sync_points(All.TimeIC, All.TimeInit);
        /* If TimeInit is not in a sensible place on the integer timeline
         * (can happen if the outputs changed since it was written)
         * start the integer timeline anew from TimeInit */
        inttime_t ti_init = ti_from_loga(log(All.TimeInit)) % TIMEBASE;
        if(round_down_power_of_two(ti_init) != ti_init) {
            message(0,"Resetting integer timeline (as %x != %x) to current snapshot\n",ti_init, round_down_power_of_two(ti_init));
            setup_sync_points(All.TimeInit, All.TimeInit);
        }
    }

    init_timebins(All.TimeInit);

    /* Important to set the global time before reading in the snapshot time as it affects the GT funcs for IO. */
    set_global_time(All.Ti_Current);

    init_drift_table(&All.CP, All.TimeInit, All.TimeMax);

    /*Read the snapshot*/
    petaio_read_snapshot(RestartSnapNum, MPI_COMM_WORLD);

    domain_test_id_uniqueness(PartManager);

    check_omega();

    check_positions();

    /* As the above will mostly take place
     * on Task 0, there will be a lot of imbalance*/
    MPIU_Barrier(MPI_COMM_WORLD);

    fof_init(All.MeanSeparation[1]);

    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)	/* initialize sph_properties */
    {
        P[i].GravCost = 1;
        P[i].Ti_drift = P[i].Ti_kick = All.Ti_Current;

        if(All.BlackHoleOn && RestartSnapNum == -1 && P[i].Type == 5 )
        {
            /* Note: Gadget-3 sets this to the seed black hole mass.*/
            BHP(i).Mass = P[i].Mass;

            /* Touch up potentially zero BH smoothing lengths, since they have historically not been saved in the snapshots.
             * Anything non-zero would work, but since BH tends to be in high density region,
             *  use a small number */
            if(P[i].Hsml == 0)
                P[i].Hsml = 0.01 * All.MeanSeparation[0];
        }
        P[i].Key = PEANO(P[i].Pos, All.BoxSize);

        if(P[i].Type != 0) continue;

        for(j = 0; j < 3; j++)
        {
            SPHP(i).HydroAccel[j] = 0;
        }

        SPHP(i).DtEntropy = 0;

        if(RestartSnapNum == -1)
        {
            SPHP(i).Density = -1;
            SPHP(i).EgyWtDensity = -1;
            SPHP(i).DhsmlEgyDensityFactor = -1;
            SPHP(i).Entropy = -1;
            SPHP(i).Ne = 1.0;
            SPHP(i).DivVel = 0;
            SPHP(i).CurlVel = 0;
            SPHP(i).DelayTime = 0;
            SPHP(i).Metallicity = 0;
            SPHP(i).Sfr = 0;
            SPHP(i).MaxSignalVel = 0;
        }
    }

    walltime_measure("/Init");

    domain_decompose_full(ddecomp);	/* do initial domain decomposition (gives equal numbers of particles) */

    if(All.DensityOn)
        setup_smoothinglengths(RestartSnapNum, ddecomp);
}


/*! This routine computes the mass content of the box and compares it to the
 * specified value of Omega-matter.  If discrepant, the run is terminated.
 */
void check_omega(void)
{
    double mass = 0, masstot, omega;
    int i;

    #pragma omp parallel for reduction(+: mass)
    for(i = 0; i < PartManager->NumPart; i++)
        mass += P[i].Mass;

    MPI_Allreduce(&mass, &masstot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    omega =
        masstot / (All.BoxSize * All.BoxSize * All.BoxSize) / (3 * All.CP.Hubble * All.CP.Hubble / (8 * M_PI * All.G));

    /*Add the density for analytically follows massive neutrinos*/
    if(All.MassiveNuLinRespOn)
        omega += get_omega_nu_nopart(&All.CP.ONu, 1);
    if(fabs(omega - All.CP.Omega0) > 1.0e-3)
    {
        endrun(0, "The mass content accounts only for Omega=%g,\nbut you specified Omega=%g in the parameterfile.\n",
                omega, All.CP.Omega0);
    }
}

/*! This routine checks that the initial positions of the particles are within the box.
 * If not, there is likely a bug in the IC generator and we abort.
 */
void check_positions(void)
{
    int i;
    #pragma omp parallel for
    for(i=0; i< PartManager->NumPart; i++){
        int j;
        for(j=0; j<3; j++) {
            if(P[i].Pos[j] < 0 || P[i].Pos[j] > All.BoxSize || !isfinite(P[i].Pos[j]))
                endrun(0,"Particle %d is outside the box (L=%g) at (%g %g %g)\n",i,All.BoxSize, P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
        }
    }
}

/* Initialize the entropy variable in Pressure-Entropy Sph.
 * Initialization of the entropy variable is a little trickier in this version of SPH,
 * since we need to make sure it 'talks to' the density appropriately */
static void
setup_density_indep_entropy(const ActiveParticles * act, ForceTree * Tree, double u_init, double a3)
{
    int j;
    int stop = 0;

    message(0, "Converting u -> entropy, with density split sph\n");

    /* This gives better convergence than initializing EgyWtDensity before Density is known*/
    #pragma omp parallel for
    for(j = 0; j < SlotsManager->info[0].size; j++)
        SphP[j].EgyWtDensity = SphP[j].Density;

    MyFloat * olddensity = (MyFloat *)mymalloc("olddensity ", SlotsManager->info[0].size * sizeof(MyFloat));
    for(j = 0; j < 100; j++)
    {
        int i;
        /* since ICs give energies, not entropies, need to iterate get this initialized correctly */
        #pragma omp parallel for
        for(i = 0; i < SlotsManager->info[0].size; i++) {
            SphP[i].Entropy = GAMMA_MINUS1 * u_init / pow(SphP[i].EgyWtDensity / a3 , GAMMA_MINUS1);
            olddensity[i] = SphP[i].EgyWtDensity;
        }
        /* Update the EgyWtDensity*/
        density(act, 0, DensityIndependentSphOn(), All.BlackHoleOn, All.HydroCostFactor, 0,  All.Time, Tree);
        if(stop)
            break;

        double maxdiff = 0;
        #pragma omp parallel for reduction(max: maxdiff)
        for(i = 0; i < SlotsManager->info[0].size; i++) {
            double value = fabs(SphP[i].EgyWtDensity - olddensity[i]) / SphP[i].EgyWtDensity;
            maxdiff = DMAX(maxdiff,value);
        }
        MPI_Allreduce(MPI_IN_PLACE, &maxdiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        message(0, "iteration %d, max relative change in EgyWtDensity = %g \n", j, maxdiff);

        /* If maxdiff is small, do one more iteration and stop*/
        if(maxdiff < 1e-3)
            stop = 1;

    }
    myfree(olddensity);
}

/*! This function is used to find an initial smoothing length for each SPH
 *  particle. It guarantees that the number of neighbours will be between
 *  desired_ngb-MAXDEV and desired_ngb+MAXDEV. For simplicity, a first guess
 *  of the smoothing length is provided to the function density(), which will
 *  then iterate if needed to find the right smoothing length.
 */
static void
setup_smoothinglengths(int RestartSnapNum, DomainDecomp * ddecomp)
{
    int i;
    const double a3 = All.Time * All.Time * All.Time;

    int64_t tot_sph, tot_bh;
    sumup_large_ints(1, &SlotsManager->info[0].size, &tot_sph);
    sumup_large_ints(1, &SlotsManager->info[5].size, &tot_bh);

    /* Do nothing if we are a pure DM run*/
    if(tot_sph + tot_bh == 0)
        return;

    ForceTree Tree = {0};
    force_tree_rebuild(&Tree, ddecomp, All.BoxSize, 0, All.OutputDir);

    if(RestartSnapNum == -1)
    {
        /* quick hack to adjust for the baryon fraction
         * only this fraction of mass is of that type.
         * this won't work for non-dm non baryon;
         * ideally each node shall have separate count of
         * ptypes of each type.
         *
         * Eventually the iteration will fix this. */
        const double massfactor = All.CP.OmegaBaryon / All.CP.Omega0;
        const double DesNumNgb = GetNumNgb(GetDensityKernelType());

        #pragma omp parallel for
        for(i = 0; i < PartManager->NumPart; i++)
        {
            /* These initial smoothing lengths are only used for SPH.
             * BH is set elsewhere. */
            if(P[i].Type != 0)
                continue;

            int no = force_get_father(i, &Tree);

            while(10 * DesNumNgb * P[i].Mass > massfactor * Tree.Nodes[no].mom.mass)
            {
                int p = force_get_father(no, &Tree);

                if(p < 0)
                    break;

                no = p;
            }

            P[i].Hsml =
                pow(3.0 / (4 * M_PI) * DesNumNgb * P[i].Mass / (massfactor * Tree.Nodes[no].mom.mass),
                        1.0 / 3) * Tree.Nodes[no].len;

            /* recover from a poor initial guess */
            if(P[i].Hsml > 500.0 * All.MeanSeparation[0])
                P[i].Hsml = All.MeanSeparation[0];
        }
    }
    /* When we restart, validate the SPH properties of the particles.
     * This also allows us to increase MinEgySpec on a restart if we choose.*/
    else
    {
        int bad = 0;
        double meanbar = All.CP.OmegaBaryon * 3 * HUBBLE * All.CP.HubbleParam * HUBBLE * All.CP.HubbleParam/ (8 * M_PI * GRAVITY);
        #pragma omp parallel for reduction(+: bad)
        for(i = 0; i < SlotsManager->info[0].size; i++) {
            /* This allows us to continue gracefully if
             * there was some kind of bug in the run that output the snapshot.
             * density() below will fix this up.*/
            if(SphP[i].Density <= 0 || !isfinite(SphP[i].Density)) {
                SphP[i].Density = meanbar;
                bad++;
            }
            if(SphP[i].EgyWtDensity <= 0 || !isfinite(SphP[i].EgyWtDensity)) {
                SphP[i].EgyWtDensity = SphP[i].Density;
            }
            double minent = GAMMA_MINUS1 * All.MinEgySpec / pow(SphP[i].Density / a3 , GAMMA_MINUS1);
            if(SphP[i].Entropy < minent)
                SphP[i].Entropy = minent;
        }
        MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if(bad > 0)
            message(0, "Detected bad densities in %d particles on disc\n",bad);
    }

    /*Allocate the extra SPH data for transient SPH particle properties.*/
    slots_allocate_sph_scratch_data(0, SlotsManager->info[0].size, &SlotsManager->sph_scratch);

        /*At the first time step all particles should be active*/
    ActiveParticles act = {0};
    act.ActiveParticle = NULL;
    act.NumActiveParticle = PartManager->NumPart;

    density(&act, 1, 0, All.BlackHoleOn, All.HydroCostFactor, 0,  All.Time, &Tree);

    /* for clean IC with U input only, we need to iterate to find entrpoy */
    if(RestartSnapNum == -1)
    {
        double u_init = (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.InitGasTemp;
        u_init *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;	/* unit conversion */

        double molecular_weight;
        if(All.InitGasTemp > 1.0e4)	/* assuming FULL ionization */
            molecular_weight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));
        else				/* assuming NEUTRAL GAS */
            molecular_weight = 4 / (1 + 3 * HYDROGEN_MASSFRAC);

        u_init /= molecular_weight;
        if(u_init < All.MinEgySpec)
            u_init = All.MinEgySpec;
        /* snapshot already has EgyWtDensity; hope it is read in correctly.
         * (need a test on this!) */
        if(DensityIndependentSphOn()) {
            setup_density_indep_entropy(&act, &Tree, u_init, a3);
        }
        else {
           /*Initialize to initial energy*/
            #pragma omp parallel for
            for(i = 0; i < SlotsManager->info[0].size; i++)
                SphP[i].Entropy = GAMMA_MINUS1 * u_init / pow(SphP[i].Density / a3 , GAMMA_MINUS1);
        }
    }
    slots_free_sph_scratch_data(SphP_scratch);
    force_tree_free(&Tree);
}
