#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <ctype.h>

#include "allvars.h"
#include "forcetree.h"
#include "proto.h"
#include "cooling.h"
#include "mymalloc.h"
#include "endrun.h"
#include "petaio.h"

char * GDB_format_particle(int i);

SIMPLE_PROPERTY(GravAccel, P[i].GravAccel[0], float, 3)
SIMPLE_PROPERTY(GravPM, P[i].GravPM[0], float, 3)
SIMPLE_PROPERTY(OldAcc, P[i].OldAcc, float, 1)

void runtests()
{

    int ptype;
    for(ptype = 0; ptype < 6; ptype++) {
        IO_REG(GravAccel,       "f4", 3, ptype);
        IO_REG(GravPM,       "f4", 3, ptype);
        IO_REG(OldAcc,       "f4", 1, ptype);
    }
    
    long_range_force();
    domain_Decomposition();	/* do domain decomposition */
    force_treebuild_simple();

    grav_short_pair();
    message(0, "GravShort Pairs %s\n", GDB_format_particle(0));
    petaio_save_snapshot("%s/PART-pairs-%03d-mpi", All.OutputDir, NTask);

    grav_short_tree();  /* computes gravity accel. */
    grav_short_tree();  /* computes gravity accel. */
    grav_short_tree();  /* computes gravity accel. */
    grav_short_tree();  /* computes gravity accel. */
    message(0, "GravShort Tree %s\n", GDB_format_particle(0));

    petaio_save_snapshot("%s/PART-tree-%03d-mpi", All.OutputDir, NTask);

    grav_short_tree_old();
    grav_short_tree_old();
    grav_short_tree_old();
    grav_short_tree_old();
    message(0, "GravTree old %s\n", GDB_format_particle(0));
    petaio_save_snapshot("%s/PART-oldtree-%03d-mpi", All.OutputDir, NTask);
}
