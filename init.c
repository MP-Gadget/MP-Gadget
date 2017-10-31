#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <gsl/gsl_sf_gamma.h>

#include "allvars.h"
#include "proto.h"
#include "cooling.h"
#include "forcetree.h"

#include "timefac.h"
#include "petaio.h"
#include "domain.h"
#include "slotsmanager.h"
#include "mpsort.h"
#include "mymalloc.h"
#include "fof.h"
#include "endrun.h"
#include "timestep.h"
#include "timebinmgr.h"

/*! \file init.c
 *  \brief code for initialisation of a simulation from initial conditions
 */

static void check_omega(void);
static void check_positions(void);

static void
setup_smoothinglengths(int RestartSnapNum);

/*! This function reads the initial conditions, and allocates storage for the
 *  tree(s). Various variables of the particle data are initialised and An
 *  intial domain decomposition is performed. If SPH particles are present,
 *  the initial SPH smoothing lengths are determined.
 */
void init(int RestartSnapNum)
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

    init_timebins(log(All.TimeInit));

    /* Important to set the global time before reading in the snapshot time as it affects the GT funcs for IO. */
    set_global_time(exp(loga_from_ti(All.Ti_Current)));

    init_drift_table(All.TimeInit, All.TimeMax);

    /*Read the snapshot*/
    petaio_read_snapshot(RestartSnapNum);

    /* this ensures the initial BhP array is consistent */
    domain_garbage_collection();

    domain_test_id_uniqueness();

    check_omega();

    check_positions();

    fof_init();

    All.SnapshotFileCount = RestartSnapNum + 1;
    All.InitSnapshotCount = RestartSnapNum + 1;

    All.TreeAllocFactor = 0.7;

    #pragma omp parallel for
    for(i = 0; i < NumPart; i++)	/* initialize sph_properties */
    {
        P[i].GravCost = 1;
        P[i].Ti_drift = P[i].Ti_kick = All.Ti_Current;

        P[i].IsGarbage = 0;
        P[i].IsNewParticle = 0;
        P[i].Swallowed = 0;

#ifdef BLACK_HOLES
        if(RestartSnapNum == -1 && P[i].Type == 5 )
        {
            BHP(i).Mass = All.SeedBlackHoleMass;
        }
#endif
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
#ifdef DENSITY_INDEPENDENT_SPH
            SPHP(i).EgyWtDensity = -1;
#endif
            SPHP(i).Ne = 1.0;
            SPHP(i).DivVel = 0;
        }
#ifdef SFR
        SPHP(i).DelayTime = 0;
        SPHP(i).Sfr = 0;
#endif

#ifdef BLACK_HOLES
        SPHP(i).Injected_BH_Energy = 0;
#endif
    }

    domain_decompose_full();	/* do initial domain decomposition (gives equal numbers of particles) */

    rebuild_activelist(0);

    setup_smoothinglengths(RestartSnapNum);
}


/*! This routine computes the mass content of the box and compares it to the
 * specified value of Omega-matter.  If discrepant, the run is terminated.
 */
void check_omega(void)
{
    double mass = 0, masstot, omega;
    int i;

    for(i = 0; i < NumPart; i++)
        mass += P[i].Mass;

    MPI_Allreduce(&mass, &masstot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    omega =
        masstot / (All.BoxSize * All.BoxSize * All.BoxSize) / (3 * All.CP.Hubble * All.CP.Hubble / (8 * M_PI * All.G));

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
    int i,j;
    for(i=0; i< NumPart; i++){
        for(j=0; j<3; j++) {
            if(P[i].Pos[j] < 0 || P[i].Pos[j] > All.BoxSize)
                endrun(0,"Particle %d is outside the box (L=%g) at (%g %g %g)\n",i,All.BoxSize, P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
        }
    }
}

/*! This function is used to find an initial smoothing length for each SPH
 *  particle. It guarantees that the number of neighbours will be between
 *  desired_ngb-MAXDEV and desired_ngb+MAXDEV. For simplicity, a first guess
 *  of the smoothing length is provided to the function density(), which will
 *  then iterate if needed to find the right smoothing length.
 */
static void
setup_smoothinglengths(int RestartSnapNum)
{
    int i;

    if(RestartSnapNum == -1)
    {
#pragma omp parallel for
        for(i = 0; i < NumPart; i++)
        {
            int no = Father[i];
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
            while(10 * All.DesNumNgb * P[i].Mass > massfactor * Nodes[no].u.d.mass)
            {
                int p = Nodes[no].father;

                if(p < 0)
                    break;

                no = p;
            }

            P[i].Hsml =
                pow(3.0 / (4 * M_PI) * All.DesNumNgb * P[i].Mass / (massfactor * Nodes[no].u.d.mass),
                        1.0 / 3) * Nodes[no].len;

            /* recover from a poor initial guess */
            if(P[i].Hsml > 500.0 * All.MeanSeparation[0])
                P[i].Hsml = All.MeanSeparation[0];
        }
    }

#ifdef BLACK_HOLES
    /* FIXME: move this inside the condition above and
      * save BHs in the snapshots to avoid this; */
    for(i = 0; i < NumPart; i++)
        if(P[i].Type == 5) {
            /* Anything non-zero would work, but since BH tends to be in high density region, 
             *  use a small number */
            P[i].Hsml = 0.01 * All.MeanSeparation[0];
            BHP(i).TimeBinLimit = -1;
        }
#endif

    density();

    /* for clean IC with U input only, we need to iterate to find entrpoy */
    if(RestartSnapNum == -1)
    {
        const double a3 = All.Time * All.Time * All.Time;

        double u_init = (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.InitGasTemp;
        u_init *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;	/* unit conversion */

        double molecular_weight;
        if(All.InitGasTemp > 1.0e4)	/* assuming FULL ionization */
            molecular_weight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));
        else				/* assuming NEUTRAL GAS */
            molecular_weight = 4 / (1 + 3 * HYDROGEN_MASSFRAC);

        u_init /= molecular_weight;

#ifdef DENSITY_INDEPENDENT_SPH
        for(i = 0; i < NumPart; i++)
        {
            if(P[i].Type == 0)
            /* start the iteration from mass density */
            SPHP(i).EgyWtDensity = SPHP(i).Density;
        }


        /* initialization of the entropy variable is a little trickier in this version of SPH,
           since we need to make sure it 'talks to' the density appropriately */
        message(0, "Converting u -> entropy, with density split sph\n");

        int j;
        double badness;
        double * olddensity = (double *)mymalloc("olddensity ", NumPart * sizeof(double));
        for(j=0;j<100;j++)
        {/* since ICs give energies, not entropies, need to iterate get this initialized correctly */
#pragma omp parallel for
            for(i = 0; i < NumPart; i++)
            {
                if(P[i].Type == 0) {
                    SPHP(i).Entropy = GAMMA_MINUS1 * u_init / pow(SPHP(i).EgyWtDensity / a3 , GAMMA_MINUS1);
                    olddensity[i] = SPHP(i).EgyWtDensity;
                }
            }
            density_update();
            badness = 0;

#pragma omp parallel private(i)
            {
                double mybadness = 0;
#pragma omp for
                for(i = 0; i < NumPart; i++) {
                    if(P[i].Type == 0) {
                        if(!(SPHP(i).EgyWtDensity > 0)) continue;
                        double value = fabs(SPHP(i).EgyWtDensity - olddensity[i]) / SPHP(i).EgyWtDensity;
                        if(value > mybadness) mybadness = value;
                    }
                }
#pragma omp critical
                {
                    if(mybadness > badness) {
                        badness = mybadness;
                    }
                }
            }
            MPI_Allreduce(MPI_IN_PLACE, &badness, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

            message(0, "iteration %03d, max relative difference = %g \n", j, badness);

            if(badness < 1e-3) break;
        }
        myfree(olddensity);
#endif //DENSITY_INDEPENDENT_SPH
#pragma omp parallel for
        for(i = 0; i < NumPart; i++) {
            if(P[i].Type == 0) {
                /* EgyWtDensity stabilized, now we convert from energy to entropy*/
                SPHP(i).Entropy = GAMMA_MINUS1 * u_init / pow(SPHP(i).EOMDensity/a3 , GAMMA_MINUS1);
            }
        }
    }

#ifdef DENSITY_INDEPENDENT_SPH
    density_update();
#endif //DENSITY_INDEPENDENT_SPH
}
