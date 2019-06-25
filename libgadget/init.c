#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <gsl/gsl_sf_gamma.h>

#include <mpsort.h>

#include "utils.h"

#include "allvars.h"
#include "cooling.h"
#include "forcetree.h"
#include "density.h"

#include "timefac.h"
#include "petaio.h"
#include "domain.h"
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
    set_global_time(exp(loga_from_ti(All.Ti_Current)));

    init_drift_table(All.TimeInit, All.TimeMax);

    /*Read the snapshot*/
    petaio_read_snapshot(RestartSnapNum, MPI_COMM_WORLD);

    domain_test_id_uniqueness();

    check_omega();

    check_positions();

    /* As the above will mostly take place
     * on Task 0, there will be a lot of imbalance*/
    MPIU_Barrier(MPI_COMM_WORLD);

    fof_init(All.MeanSeparation[1]);

    All.SnapshotFileCount = RestartSnapNum + 1;
    All.InitSnapshotCount = RestartSnapNum + 1;

    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)	/* initialize sph_properties */
    {
        P[i].GravCost = 1;
        P[i].Ti_drift = P[i].Ti_kick = All.Ti_Current;

        if(All.BlackHoleOn && RestartSnapNum == -1 && P[i].Type == 5 )
        {
            /* Note: Gadget-3 sets this to the seed black hole mass.*/
            BHP(i).Mass = P[i].Mass;
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
            SPHP(i).Entropy = -1;
            SPHP(i).Ne = 1.0;
            SPHP(i).DivVel = 0;
        }
        SPHP(i).DelayTime = 0;
    }

    walltime_measure("/Init");

    domain_decompose_full(ddecomp);	/* do initial domain decomposition (gives equal numbers of particles) */

    /*At the first time step all particles should be active*/
    ActiveParticle = NULL;
    NumActiveParticle = PartManager->NumPart;

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
            if(P[i].Pos[j] < 0 || P[i].Pos[j] > All.BoxSize)
                endrun(0,"Particle %d is outside the box (L=%g) at (%g %g %g)\n",i,All.BoxSize, P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
        }
    }
}

/* Initialize the entropy variable in Pressure-Entropy Sph.
 * Initialization of the entropy variable is a little trickier in this version of SPH,
 * since we need to make sure it 'talks to' the density appropriately */
static void
setup_density_indep_entropy(ForceTree * Tree, double u_init, double a3)
{
    message(0, "Converting u -> entropy, with density split sph\n");

    int j;
    double * olddensity = (double *)mymalloc("olddensity ", PartManager->NumPart * sizeof(double));
    for(j = 0; j < 100; j++)
    {
        int i;
        /* since ICs give energies, not entropies, need to iterate get this initialized correctly */
        #pragma omp parallel for
        for(i = 0; i < PartManager->NumPart; i++)
        {
            if(P[i].Type == 0) {
                SPHP(i).Entropy = GAMMA_MINUS1 * u_init / pow(SPHP(i).EgyWtDensity / a3 , GAMMA_MINUS1);
                olddensity[i] = SPHP(i).EgyWtDensity;
            }
        }
        density_update(Tree);
        double badness = 0;

        #pragma omp parallel for reduction(max: badness)
        for(i = 0; i < PartManager->NumPart; i++) {
            if(P[i].Type == 0) {
                if(SPHP(i).EgyWtDensity <= 0)
                    continue;
                double value = fabs(SPHP(i).EgyWtDensity - olddensity[i]) / SPHP(i).EgyWtDensity;
                badness = DMAX(badness,value);
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &badness, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        message(0, "iteration %03d, max relative difference = %g \n", j, badness);

        if(badness < 1e-3) break;
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

    ForceTree Tree = {0};
    force_tree_rebuild(&Tree, ddecomp, All.BoxSize, 0);

    if(RestartSnapNum == -1)
    {
#pragma omp parallel for
        for(i = 0; i < PartManager->NumPart; i++)
        {
            int no = force_get_father(i, &Tree);
            /* Don't need smoothing lengths for DM particles*/
            if(P[i].Type != 0 && P[i].Type != 4 && P[i].Type != 5)
                continue;
            /* quick hack to adjust for the baryon fraction
             * only this fraction of mass is of that type.
             * this won't work for non-dm non baryon;
             * ideally each node shall have separate count of
             * ptypes of each type.
             *
             * Eventually the iteration will fix this. */
            double massfactor;
            if(P[i].Type == 0) {
                massfactor = 0.04 / 0.26;
            } else {
                massfactor = 1.0 - 0.04 / 0.26;
            }
            while(10 * All.DesNumNgb * P[i].Mass > massfactor * Tree.Nodes[no].u.d.mass)
            {
                int p = force_get_father(no, &Tree);

                if(p < 0)
                    break;

                no = p;
            }

            P[i].Hsml =
                pow(3.0 / (4 * M_PI) * All.DesNumNgb * P[i].Mass / (massfactor * Tree.Nodes[no].u.d.mass),
                        1.0 / 3) * Tree.Nodes[no].len;

            /* recover from a poor initial guess */
            if(P[i].Hsml > 500.0 * All.MeanSeparation[0])
                P[i].Hsml = All.MeanSeparation[0];
        }
    }

    /* FIXME: move this inside the condition above and
      * save BHs in the snapshots to avoid this; */
    for(i = 0; i < PartManager->NumPart; i++)
        if(P[i].Type == 5) {
            /* Anything non-zero would work, but since BH tends to be in high density region,
             *  use a small number */
            P[i].Hsml = 0.01 * All.MeanSeparation[0];
            BHP(i).TimeBinLimit = -1;
        }

    /*Allocate the extra SPH data for transient SPH particle properties.*/
    slots_allocate_sph_scratch_data(sfr_need_to_compute_sph_grad_rho(), SlotsManager->info[0].size);

    density(&Tree);

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

        if(All.DensityIndependentSphOn) {
            setup_density_indep_entropy(&Tree, u_init, a3);
        }

        /*Initialize to initial energy*/
        #pragma omp parallel for
        for(i = 0; i < PartManager->NumPart; i++) {
            if(P[i].Type == 0) {
                SPHP(i).Entropy = GAMMA_MINUS1 * u_init / pow(SPH_EOMDensity(i)/a3 , GAMMA_MINUS1);
            }
        }
    }
    /* snapshot already has EgyWtDensity; hope it is read in correctly.
     * (need a test on this!) */
    if(All.DensityIndependentSphOn)
        density_update(&Tree);

    slots_free_sph_scratch_data(SphP_scratch);
    force_tree_free(&Tree);
}
