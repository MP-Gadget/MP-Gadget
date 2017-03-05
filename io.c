#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "allvars.h"
#include "densitykernel.h"
#include "proto.h"
#include "petaio.h"
#include "domain.h"
#include "fof.h"
#include "endrun.h"

/*! \file io.c
 *  \brief Output of a snapshot file to disk.
 */

/*! This function writes a snapshot of the particle distribution to one or
 * several files using Gadget's default file format.  If
 * NumFilesPerSnapshot>1, the snapshot is distributed into several files,
 * which are written simultaneously. Each file contains data from a group of
 * processors of size roughly NTask/NumFilesPerSnapshot.
 */
/* with_fof != 0 regular snapshot, do fof and write it out */
void savepositions(int num, int with_fof)
{
    walltime_measure("/Misc");

    rearrange_particle_sequence();
    /* ensures that new tree will be constructed */
    All.NumForcesSinceLastDomainDecomp = (int64_t) (1 + All.TreeDomainUpdateFrequency * All.TotNumPart);

    walltime_measure("/Snapshot/Misc");

    petaio_save_snapshot("%s/PART_%03d", All.OutputDir, num);

    walltime_measure("/Snapshot/Write");

    /* regular snapshot, do fof and write it out */
    if(All.SnapshotWithFOF && with_fof != 0) {
        message(0, "computing group catalogue...\n");

        fof_fof(num);

        message(0, "done with group catalogue.\n");
        walltime_measure("/Snapshot/WriteFOF");
    }

    if(ThisTask == 0) {
        char buf[1024];
        sprintf(buf, "%s/LastSnapshotNum.txt", All.OutputDir);
        FILE * fd = fopen(buf, "a");
        fprintf(fd, "Time %g Redshift %g Ti_current %d Snapnumber %03d\n", All.Time, 1 / All.Time - 1, All.Ti_Current, num);
        fclose(fd);

    }
}
