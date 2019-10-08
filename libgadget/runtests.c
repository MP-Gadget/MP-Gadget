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

void runtests(DomainDecomp * ddecomp)
{
    struct IOTable IOTable = {0};
    register_io_blocks(&IOTable);
    register_extra_blocks(&IOTable);

    /* this produces a very imbalanced load to trigger Issue 86 */
    if(ThisTask == 0) {
        P[0].GravCost = 1e10;
        P[PartManager->NumPart - 1].GravCost = 1e10;
    }

    int NTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    rebuild_activelist(All.Ti_Current, 0);

    ForceTree Tree = {0};
    force_tree_rebuild(&Tree, ddecomp, All.BoxSize, 1);
    gravpm_force(&Tree);
    force_tree_rebuild(&Tree, ddecomp, All.BoxSize, 1);

    const double rho0 = All.CP.Omega0 * 3 * All.CP.Hubble * All.CP.Hubble / (8 * M_PI * All.G);
    grav_short_pair(&Tree, All.G, All.BoxSize, All.Nmesh, All.Asmth, rho0, 0, All.FastParticleType, All.treeacc);

    message(0, "GravShort Pairs %s\n", GDB_format_particle(0));
    petaio_save_snapshot(&IOTable, "%s/PART-pairs-%03d-mpi", All.OutputDir, NTask);

    grav_short_tree(&Tree, All.G, All.BoxSize, All.Nmesh, All.Asmth, rho0, 0, All.FastParticleType, All.treeacc);
    grav_short_tree(&Tree, All.G, All.BoxSize, All.Nmesh, All.Asmth, rho0, 0, All.FastParticleType, All.treeacc);
    grav_short_tree(&Tree, All.G, All.BoxSize, All.Nmesh, All.Asmth, rho0, 0, All.FastParticleType, All.treeacc);
    grav_short_tree(&Tree, All.G, All.BoxSize, All.Nmesh, All.Asmth, rho0, 0, All.FastParticleType, All.treeacc);
    message(0, "GravShort Tree %s\n", GDB_format_particle(0));
    force_tree_free(&Tree);

    petaio_save_snapshot(&IOTable, "%s/PART-tree-%03d-mpi", All.OutputDir, NTask);

    destroy_io_blocks(&IOTable);
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
