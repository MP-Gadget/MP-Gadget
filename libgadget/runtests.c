#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <ctype.h>

#include "utils.h"

#include "allvars.h"
#include "partmanager.h"
#include "forcetree.h"
#include "cooling.h"
#include "gravity.h"
#include "petaio.h"
#include "domain.h"
#include "timestep.h"
#include "fof.h"

char * GDB_format_particle(int i);

SIMPLE_PROPERTY(GravAccel, P[i].GravAccel[0], float, 3)
SIMPLE_PROPERTY(GravPM, P[i].GravPM[0], float, 3)

void register_extra_blocks(struct IOTable * IOTable)
{
    int ptype;
    for(ptype = 0; ptype < 6; ptype++) {
        IO_REG(GravAccel,       "f4", 3, ptype, IOTable);
        IO_REG(GravPM,       "f4", 3, ptype, IOTable);
    }
}

/* Run various checks on the gravity code. Check that the short-range/long-range force split is working.*/
void runtests(DomainDecomp * ddecomp)
{
    struct IOTable IOTable = {0};
    register_io_blocks(&IOTable);
    register_extra_blocks(&IOTable);

    int NTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    rebuild_activelist(All.Ti_Current, 0);

    ForceTree Tree = {0};
    force_tree_rebuild(&Tree, ddecomp, All.BoxSize, 1);
    gravpm_force(&Tree);
    force_tree_rebuild(&Tree, ddecomp, All.BoxSize, 1);

    struct TreeAccParams treeacc = All.treeacc;
    const double rho0 = All.CP.Omega0 * 3 * All.CP.Hubble * All.CP.Hubble / (8 * M_PI * All.G);
    grav_short_pair(&Tree, All.G, All.BoxSize, All.Nmesh, All.Asmth, rho0, 0, All.FastParticleType, All.treeacc);

    double (* PairAccn)[3] = mymalloc("PairAccns", 3*sizeof(double) * PartManager->NumPart);

    int i;
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)
    {
        int k;
        for(k=0; k<3; k++)
            PairAccn[i][k] = P[i].GravPM[k] + P[i].GravAccel[k];
    }
    message(0, "GravShort Pairs %s\n", GDB_format_particle(0));
    petaio_save_snapshot(&IOTable, "%s/PART-pairs-%03d-mpi", All.OutputDir, NTask);

    treeacc.TreeUseBH = 1;
    grav_short_tree(&Tree, All.G, All.BoxSize, All.Nmesh, All.Asmth, rho0, 0, All.FastParticleType, treeacc);
    treeacc.TreeUseBH = 0;
    grav_short_tree(&Tree, All.G, All.BoxSize, All.Nmesh, All.Asmth, rho0, 0, All.FastParticleType, treeacc);

    message(0, "GravShort Tree %s\n", GDB_format_particle(0));
    petaio_save_snapshot(&IOTable, "%s/PART-tree-%03d-mpi", All.OutputDir, NTask);


    double meanerr = 0, maxerr=-1;
    /* This checks that the short-range force accuracy is being correctly estimated.*/
    #pragma omp parallel for reduction(+: meanerr) reduction(max:maxerr)
    for(i = 0; i < PartManager->NumPart; i++)
    {
        int k;
        for(k=0; k<3; k++) {
            double err = 0;
            if(PairAccn[i][k] != 0)
                err = fabs((PairAccn[i][k] - (P[i].GravPM[k] + P[i].GravAccel[k]))/(PairAccn[i][k]));
            meanerr += err;
            if(maxerr < err)
                maxerr = err;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &meanerr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxerr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    int64_t tot_npart;
    sumup_large_ints(1, &PartManager->NumPart, &tot_npart);

    meanerr/= tot_npart;
    myfree(PairAccn);
    force_tree_free(&Tree);

    destroy_io_blocks(&IOTable);

    message(0, "Max rel force error (tree vs pairwise): %g mean: %g forcetol: %g\n", maxerr, meanerr, treeacc.ErrTolForceAcc);


}

void runfof(int RestartSnapNum, DomainDecomp * ddecomp)
{
    ForceTree Tree = {0};
    /*FoF needs a tree*/
    int HybridNuGrav = All.HybridNeutrinosOn && All.Time <= All.HybridNuPartTime;
    force_tree_rebuild(&Tree, ddecomp, All.BoxSize, HybridNuGrav);
    fof_fof(&Tree, All.BoxSize, All.BlackHoleOn, MPI_COMM_WORLD);
    force_tree_free(&Tree);
    fof_save_groups(RestartSnapNum, MPI_COMM_WORLD);
    fof_finish();
}
