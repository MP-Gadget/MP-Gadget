#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <gsl/gsl_sf_gamma.h>

#include "allvars.h"
#include "proto.h"
#include "cooling.h"
#include "forcetree.h"

#include "petaio.h"
#include "domain.h"
#include "mpsort.h"
#include "mymalloc.h"
#include "fof.h"
#include "endrun.h"

/*! \file init.c
 *  \brief code for initialisation of a simulation from initial conditions
 */

static void
setup_smoothinglengths(int RestartSnapNum);

/*! This function reads the initial conditions, and allocates storage for the
 *  tree(s). Various variables of the particle data are initialised and An
 *  intial domain decomposition is performed. If SPH particles are present,
 *  the inial SPH smoothing lengths are determined.
 */
void init(int RestartSnapNum)
{
    int i, j;

    switch(RestartSnapNum) {
        case -1:
            petaio_read_ic();
            break;
        default:
            petaio_read_snapshot(RestartSnapNum);
            break;
    }

    /* this ensures the initial BhP array is consistent */
    domain_garbage_collection();

    domain_test_id_uniqueness();

    check_omega();

    fof_init();

    All.NumCurrentTiStep = 0;	/* setup some counters */
    All.SnapshotFileCount = 0;
    All.SnapshotFileCount = RestartSnapNum + 1;

    All.TotNumOfForces = 0;
    All.NumForcesSinceLastDomainDecomp = 0;

    All.TreeAllocFactor = 0.7;

    for(i = 0; i < NumPart; i++)	/*  start-up initialization */
    {
        P[i].GravCost = 1;

#ifdef BLACK_HOLES
        P[i].Swallowed = 0;
        if(RestartSnapNum == -1 && P[i].Type == 5 )
        {
            BHP(i).Mass = All.SeedBlackHoleMass;
        }
#endif
    }

    for(i = 0; i < TIMEBINS; i++)
        TimeBinActive[i] = 1;

    reconstruct_timebins();

    All.PM_Ti_endstep = All.PM_Ti_begstep = 0;

    for(i = 0; i < NumPart; i++)	/* initialize sph_properties */
    {
        if(P[i].Type != 0) continue;
        for(j = 0; j < 3; j++)
        {
            SPHP(i).VelPred[j] = P[i].Vel[j];
            SPHP(i).HydroAccel[j] = 0;
        }

        SPHP(i).DtEntropy = 0;

        if(RestartSnapNum == -1)
        {
            SPHP(i).Density = -1;
#ifdef DENSITY_INDEPENDENT_SPH
            SPHP(i).EgyWtDensity = -1;
            SPHP(i).EntVarPred = -1;
#endif
#ifdef VOLUME_CORRECTION
            SPHP(i).DensityOld = 1;
#endif
            SPHP(i).Ne = 1.0;
            SPHP(i).DivVel = 0;
        }
#ifdef WINDS
        SPHP(i).DelayTime = 0;
#endif
#ifdef SFR
        SPHP(i).Sfr = 0;
#endif

#ifdef BLACK_HOLES
        SPHP(i).Injected_BH_Energy = 0;
#endif
#ifdef TWODIMS
        SPHP(i).VelPred[2] = 0;
        SPHP(i).HydroAccel[2] = 0;
#endif
#ifdef ONEDIM
        SPHP(i).VelPred[1] = SPHP(i).VelPred[2] = 0;
        SPHP(i).HydroAccel[1] =SPHP(i).HydroAccel[2] = 0;
#endif
    }


    Flag_FullStep = 1;		/* to ensure that Peano-Hilbert order is done */

    domain_Decomposition();	/* do initial domain decomposition (gives equal numbers of particles) */

    force_tree_rebuild();

    All.Ti_Current = 0;

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
        masstot / (All.BoxSize * All.BoxSize * All.BoxSize) / (3 * All.Hubble * All.Hubble / (8 * M_PI * All.G));

    if(fabs(omega - All.CP.Omega0) > 1.0e-3)
    {
        endrun(0, "The mass content accounts only for Omega=%g,\nbut you specified Omega=%g in the parameterfile.\n",
                omega, All.CP.Omega0);
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
            int no, p;
            no = Father[i];
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
                p = Nodes[no].u.d.father;

                if(p < 0)
                    break;

                no = p;
            }

            P[i].Hsml =
                pow(3.0 / (4 * M_PI) * All.DesNumNgb * P[i].Mass / (massfactor * Nodes[no].u.d.mass),
                        1.0 / 3) * Nodes[no].len;
            if(All.SofteningTable[0] != 0 && P[i].Hsml > 500.0 * All.SofteningTable[0])
                P[i].Hsml = All.SofteningTable[0];
        }
    }

#ifdef BLACK_HOLES
    for(i = 0; i < NumPart; i++)
        if(P[i].Type == 5) {
            P[i].Hsml = All.SofteningTable[5];
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

        for(i = 0; i < NumPart; i++) {
            if(P[i].Type == 0) SPHP(i).Entropy = u_init;
        }

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
                    double entropy = GAMMA_MINUS1 * SPHP(i).Entropy / pow(SPHP(i).EgyWtDensity / a3 , GAMMA_MINUS1);
                    SPHP(i).EntVarPred = pow(entropy, 1/GAMMA);
                    olddensity[i] = SPHP(i).EgyWtDensity;
                }
            }
            density();
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
                SPHP(i).Entropy = GAMMA_MINUS1 * SPHP(i).Entropy / pow(SPHP(i).EOMDensity/a3 , GAMMA_MINUS1);
            }
        }
    }

#ifdef DENSITY_INDEPENDENT_SPH
    /* snapshot already has Entropy and EgyWtDensity;
     * hope it is read in correctly. (need a test
     * on this!) */
    /* regardless we initalize EntVarPred. This may be unnecessary*/
    for(i = 0; i < NumPart; i++) {
        if(P[i].Type == 0) {
            SPHP(i).EntVarPred = pow(SPHP(i).Entropy, 1./GAMMA);
        }
    }
    density();
#endif //DENSITY_INDEPENDENT_SPH
}
