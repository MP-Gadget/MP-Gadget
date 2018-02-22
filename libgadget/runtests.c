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

void grav_short_pair(void);

char * GDB_format_particle(int i);

SIMPLE_PROPERTY(GravAccel, P[i].GravAccel[0], float, 3)
SIMPLE_PROPERTY(GravPM, P[i].GravPM[0], float, 3)

void runtests()
{

    int ptype;
    for(ptype = 0; ptype < 6; ptype++) {
        IO_REG(GravAccel,       "f4", 3, ptype);
        IO_REG(GravPM,       "f4", 3, ptype);
    }

    gravpm_force();

    /* this produces a very imbalanced load to trigger Issue 86 */
    if(ThisTask == 0) {
        P[0].GravCost = 1e10;
        P[PartManager->NumPart - 1].GravCost = 1e10;
    }

    domain_decompose_full();	/* do domain decomposition */
    rebuild_activelist(All.Ti_Current);

    grav_short_pair();
    message(0, "GravShort Pairs %s\n", GDB_format_particle(0));
    petaio_save_snapshot("%s/PART-pairs-%03d-mpi", All.OutputDir, NTask);

    grav_short_tree();  /* computes gravity accel. */
    grav_short_tree();  /* computes gravity accel. */
    grav_short_tree();  /* computes gravity accel. */
    grav_short_tree();  /* computes gravity accel. */
    message(0, "GravShort Tree %s\n", GDB_format_particle(0));

    petaio_save_snapshot("%s/PART-tree-%03d-mpi", All.OutputDir, NTask);

}
