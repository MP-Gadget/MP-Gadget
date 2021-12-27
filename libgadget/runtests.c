#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <ctype.h>

#include "utils.h"

#include "allvars.h"
#include "init.h"
#include "partmanager.h"
#include "forcetree.h"
#include "cooling.h"
#include "gravity.h"
#include "petaio.h"
#include "domain.h"
#include "timestep.h"
#include "fof.h"
#include "sfr_eff.h"
#include "density.h"
#include "hydra.h"

char * GDB_format_particle(int i);

SIMPLE_GETTER(GTGravAccel, GravAccel[0], float, 3, struct particle_data)
SIMPLE_GETTER(GTGravPM, GravPM[0], float, 3, struct particle_data)

void register_extra_blocks(struct IOTable * IOTable)
{
    int ptype;
    for(ptype = 0; ptype < 6; ptype++) {
        IO_REG_WRONLY(GravAccel,       "f4", 3, ptype, IOTable);
        IO_REG_WRONLY(GravPM,       "f4", 3, ptype, IOTable);
    }
}

double copy_and_mean_accn(double (* PairAccn)[3])
{
    int i;
    double meanacc = 0;
    #pragma omp parallel for reduction(+: meanacc)
    for(i = 0; i < PartManager->NumPart; i++)
    {
        int k;
        for(k=0; k<3; k++) {
            PairAccn[i][k] = P[i].GravPM[k] + P[i].GravAccel[k];
            meanacc += fabs(PairAccn[i][k]);
        }
    }
    int64_t tot_npart;
    MPI_Allreduce(&PartManager->NumPart, &tot_npart, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &meanacc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    meanacc/= (tot_npart*3.);
    return meanacc;
}

void check_accns(double * meanerr_tot, double * maxerr_tot, double (*PairAccn)[3], double meanacc)
{
    double meanerr=0, maxerr=-1;
    int i;
    /* This checks that the short-range force accuracy is being correctly estimated.*/
    #pragma omp parallel for reduction(+: meanerr) reduction(max:maxerr)
    for(i = 0; i < PartManager->NumPart; i++)
    {
        int k;
        for(k=0; k<3; k++) {
            double err = 0;
            if(PairAccn[i][k] != 0)
                err = fabs((PairAccn[i][k] - (P[i].GravPM[k] + P[i].GravAccel[k]))/meanacc);
            meanerr += err;
            if(maxerr < err)
                maxerr = err;
        }
    }
    MPI_Allreduce(&meanerr, meanerr_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&maxerr, maxerr_tot, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    int64_t tot_npart;
    MPI_Allreduce(&PartManager->NumPart, &tot_npart, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    *meanerr_tot/= (tot_npart*3.);
}

/* Run various checks on the gravity code. Check that the short-range/long-range force split is working.*/
void runtests(int RestartSnapNum)
{
    PetaPM pm = {0};
    gravpm_init_periodic(&pm, All.BoxSize, All.Asmth, All.Nmesh, All.G);
    DomainDecomp ddecomp[1] = {0};
    /* So we can run a test on the final snapshot*/
    All.TimeMax = All.TimeInit * 1.1;

    inttime_t Ti_Current = init(RestartSnapNum, All.TimeIC, All.TimeInit, All.TimeMax, &All.CP, All.SnapshotWithFOF, All.MassiveNuLinRespOn, All.G, All.MassTable, All.NTotalInit);

    domain_decompose_full(ddecomp);	/* do initial domain decomposition (gives equal numbers of particles) */

    if(All.DensityOn) {
        double uu_in_cgs = All.UnitEnergy_in_cgs / All.UnitMass_in_g;
        setup_smoothinglengths(RestartSnapNum, ddecomp, &All.CP, All.BlackHoleOn, All.MinEgySpec, uu_in_cgs, Ti_Current, All.TimeInit, All.NTotalInit[0]);
    }

    struct IOTable IOTable = {0};
    register_io_blocks(&IOTable, 0);
    register_extra_blocks(&IOTable);

    int NTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    ActiveParticles Act = {0};
    DriftKickTimes times = init_driftkicktime(Ti_Current);
    rebuild_activelist(&Act, &times, 0, All.Time);

    ForceTree Tree = {0};
    force_tree_rebuild(&Tree, ddecomp, 1, 1, All.OutputDir);
    gravpm_force(&pm, &Tree, &All.CP, All.Time, All.UnitLength_in_cm, All.OutputDir, All.MassiveNuLinRespOn, All.TimeIC, All.HybridNeutrinosOn, All.FastParticleType, All.BlackHoleOn);
    force_tree_rebuild(&Tree, ddecomp, 1, 1, All.OutputDir);

    struct gravshort_tree_params origtreeacc = get_gravshort_treepar();
    struct gravshort_tree_params treeacc = origtreeacc;
    const double rho0 = All.CP.Omega0 * 3 * All.CP.Hubble * All.CP.Hubble / (8 * M_PI * All.G);
    grav_short_pair(&Act, &pm, &Tree, treeacc.Rcut, rho0, 0, All.FastParticleType);

    double (* PairAccn)[3] = mymalloc2("PairAccns", 3*sizeof(double) * PartManager->NumPart);

    double meanacc = copy_and_mean_accn(PairAccn);
    message(0, "GravShort Pairs %s\n", GDB_format_particle(0));
    petaio_save_snapshot(&IOTable, 0, All.Time, "%s/PART-pairs-%03d", All.OutputDir, RestartSnapNum);

    treeacc.ErrTolForceAcc = 0;
    set_gravshort_treepar(treeacc);
    grav_short_tree(&Act, &pm, &Tree, rho0, 0, All.FastParticleType);

    /* This checks fully opened tree force against pair force*/
    double meanerr, maxerr;
    check_accns(&meanerr,&maxerr,PairAccn, meanacc);
    message(0, "Force error, open tree vs pairwise. max : %g mean: %g forcetol: %g\n", maxerr, meanerr, treeacc.ErrTolForceAcc);

    if(maxerr > 0.1)
        endrun(2, "Fully open tree force does not agree with pairwise calculation! maxerr %g > 0.1!\n", maxerr);

    message(0, "GravShort Tree %s\n", GDB_format_particle(0));
    petaio_save_snapshot(&IOTable, 0, All.Time, "%s/PART-tree-open-%03d", All.OutputDir, RestartSnapNum);

    /* This checks tree force against tree force with zero error (which always opens).*/
    copy_and_mean_accn(PairAccn);

    treeacc = origtreeacc;
    set_gravshort_treepar(treeacc);
    /* Code automatically sets the UseTreeBH parameter.*/
    grav_short_tree(&Act, &pm, &Tree, rho0, 0, All.FastParticleType);
    grav_short_tree(&Act, &pm, &Tree, rho0, 0, All.FastParticleType);

    petaio_save_snapshot(&IOTable, 0, All.Time, "%s/PART-tree-%03d", All.OutputDir, RestartSnapNum);

    check_accns(&meanerr,&maxerr,PairAccn, meanacc);
    message(0, "Force error, open tree vs tree.: %g mean: %g forcetol: %g\n", maxerr, meanerr, treeacc.ErrTolForceAcc);

    if(meanerr > treeacc.ErrTolForceAcc* 1.2)
        endrun(2, "Average force error is underestimated: %g > 1.2 * %g!\n", meanerr, treeacc.ErrTolForceAcc);

    copy_and_mean_accn(PairAccn);
    /* This checks the tree against a larger Rcut.*/
    treeacc.Rcut = 9.5;
    set_gravshort_treepar(treeacc);
    grav_short_tree(&Act, &pm, &Tree, rho0, 0, All.FastParticleType);
    grav_short_tree(&Act, &pm, &Tree, rho0, 0, All.FastParticleType);
    petaio_save_snapshot(&IOTable, 0, All.Time, "%s/PART-tree-rcut-%03d", All.OutputDir, RestartSnapNum);

    check_accns(&meanerr,&maxerr,PairAccn, meanacc);
    message(0, "Force error, tree vs rcut.: %g mean: %g Rcut = %g\n", maxerr, meanerr, treeacc.Rcut);

    if(maxerr > 0.2)
        endrun(2, "Rcut decreased below desired value, error too large %g\n", maxerr);

    /* This checks the tree against a box with a smaller Nmesh.*/
    treeacc = origtreeacc;
    force_tree_free(&Tree);
    gravpm_init_periodic(&pm, All.BoxSize, All.Asmth, All.Nmesh/2., All.G);
    force_tree_rebuild(&Tree, ddecomp, 1, 1, All.OutputDir);
    gravpm_force(&pm, &Tree, &All.CP, All.Time, All.UnitLength_in_cm, All.OutputDir, All.MassiveNuLinRespOn, All.TimeIC, All.HybridNeutrinosOn, All.FastParticleType, All.BlackHoleOn);
    force_tree_rebuild(&Tree, ddecomp, 1, 1, All.OutputDir);
    set_gravshort_treepar(treeacc);
    grav_short_tree(&Act, &pm, &Tree, rho0, 0, All.FastParticleType);
    grav_short_tree(&Act, &pm, &Tree, rho0, 0, All.FastParticleType);
    petaio_save_snapshot(&IOTable, 0, All.Time, "%s/PART-tree-nmesh2-%03d", All.OutputDir, RestartSnapNum);

    check_accns(&meanerr, &maxerr, PairAccn, meanacc);
    message(0, "Force error, nmesh %d vs %d: %g mean: %g \n", All.Nmesh, All.Nmesh/2, maxerr, meanerr);

    if(maxerr > 0.5 || meanerr > 0.05)
        endrun(2, "Nmesh sensitivity worse, something may be wrong\n");

    myfree(PairAccn);
    force_tree_free(&Tree);
    destroy_io_blocks(&IOTable);
    petapm_destroy(&pm);
}

void
runfof(int RestartSnapNum)
{
    PetaPM pm = {0};
    gravpm_init_periodic(&pm, All.BoxSize, All.Asmth, All.Nmesh, All.G);
    DomainDecomp ddecomp[1] = {0};
    /* ... read in initial model */

    inttime_t Ti_Current = init(RestartSnapNum, All.TimeIC, All.TimeInit, All.TimeMax, &All.CP, All.SnapshotWithFOF, All.MassiveNuLinRespOn, All.G, All.MassTable, All.NTotalInit);

    domain_decompose_full(ddecomp);	/* do initial domain decomposition (gives equal numbers of particles) */

    if(All.DensityOn) {
        double uu_in_cgs = All.UnitEnergy_in_cgs / All.UnitMass_in_g;
        setup_smoothinglengths(RestartSnapNum, ddecomp, &All.CP, All.BlackHoleOn, All.MinEgySpec, uu_in_cgs, Ti_Current, All.TimeInit, All.NTotalInit[0]);
    }

    DriftKickTimes times = init_driftkicktime(Ti_Current);
    /*FoF needs a tree*/
    int HybridNuGrav = All.HybridNeutrinosOn && All.Time <= All.HybridNuPartTime;
    /* Regenerate the star formation rate for the FOF table.*/
    if(All.StarformationOn) {
        ActiveParticles Act = {0};
        Act.NumActiveParticle = PartManager->NumPart;
        MyFloat * GradRho = NULL;
        if(sfr_need_to_compute_sph_grad_rho()) {
            ForceTree gasTree = {0};
            GradRho = mymalloc2("SPH_GradRho", sizeof(MyFloat) * 3 * SlotsManager->info[0].size);
            /*Allocate the memory for predicted SPH data.*/
            struct sph_pred_data sph_predicted = slots_allocate_sph_pred_data(SlotsManager->info[0].size);
            force_tree_rebuild(&gasTree, ddecomp, HybridNuGrav, 0, All.OutputDir);
            /* computes GradRho with a treewalk. No hsml update as done in init().*/
            density(&Act, 0, 0, All.BlackHoleOn, All.MinEgySpec, times, &All.CP, &sph_predicted, GradRho, &gasTree);
            force_tree_free(&gasTree);
            slots_free_sph_pred_data(&sph_predicted);
        }
        ForceTree Tree = {0};
        cooling_and_starformation(&Act, All.Time, 0, &Tree, GradRho, NULL);
        if(GradRho)
            myfree(GradRho);
    }
    FOFGroups fof = fof_fof(ddecomp, 1, MPI_COMM_WORLD);
    fof_save_groups(&fof, All.OutputDir, All.FOFFileBase, RestartSnapNum, All.PartAllocFactor, All.StarformationOn, All.BlackHoleOn, MPI_COMM_WORLD);
    fof_finish(&fof);
}

void
runpower(int RestartSnapNum)
{
    PetaPM pm = {0};
    gravpm_init_periodic(&pm, All.BoxSize, All.Asmth, All.Nmesh, All.G);
    DomainDecomp ddecomp[1] = {0};
    /* ... read in initial model */

    inttime_t Ti_Current = init(RestartSnapNum, All.TimeIC, All.TimeInit, All.TimeMax, &All.CP, All.SnapshotWithFOF, All.MassiveNuLinRespOn, All.G, All.MassTable, All.NTotalInit);

    domain_decompose_full(ddecomp);	/* do initial domain decomposition (gives equal numbers of particles) */

    if(All.DensityOn) {
        double uu_in_cgs = All.UnitEnergy_in_cgs / All.UnitMass_in_g;
        setup_smoothinglengths(RestartSnapNum, ddecomp, &All.CP, All.BlackHoleOn, All.MinEgySpec, uu_in_cgs, Ti_Current, All.TimeInit, All.NTotalInit[0]);
    }

    /*PM needs a tree*/
    ForceTree Tree = {0};
    int HybridNuGrav = All.HybridNeutrinosOn && All.Time <= All.HybridNuPartTime;
    force_tree_rebuild(&Tree, ddecomp, HybridNuGrav, 1, All.OutputDir);
    gravpm_force(&pm, &Tree, &All.CP, All.Time, All.UnitLength_in_cm, All.OutputDir, All.MassiveNuLinRespOn, All.TimeIC, All.HybridNeutrinosOn, All.FastParticleType, 1);
    force_tree_free(&Tree);
}
