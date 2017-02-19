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

void runtests()
{

    int ptype;
    for(ptype = 0; ptype < 6; ptype++)
        IO_REG(GravAccel,       "f4", 3, ptype);
    
    // long_range_force();
    domain_Decomposition();	/* do domain decomposition */
    force_treebuild_simple();
    message(0, "Before GravTree %s\n", GDB_format_particle(0));
    grav_short_tree();  /* computes gravity accel. */
    message(0, "After GravTree:1 %s\n", GDB_format_particle(0));
    grav_short_tree();  /* For the first timestep, we redo it
                         * to allow usage of relative opening
                         * criterion for consistent accuracy.
                         */
    message(0, "After GravTree:2 %s\n", GDB_format_particle(0));

    petaio_save_snapshot(999001);
    grav_short_pair();

    message(0, "After GravShort %s\n", GDB_format_particle(0));
    petaio_save_snapshot(999002);
}
