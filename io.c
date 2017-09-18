#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "allvars.h"
#include "petaio.h"
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

    petaio_save_snapshot("%s/PART_%03d", All.OutputDir, num);

    walltime_measure("/Snapshot/Write");

    /* regular snapshot, do fof and write it out */
    if(All.SnapshotWithFOF && with_fof != 0) {
        message(0, "computing group catalogue...\n");

        fof_fof(num);

        message(0, "done with group catalogue.\n");
        walltime_measure("/Snapshot/WriteFOF");
    }

    walltime_measure("/Domain/Misc");

    if(ThisTask == 0) {
        char buf[1024];
        sprintf(buf, "%s/Snapshots.txt", All.OutputDir);
        FILE * fd = fopen(buf, "a");
        fprintf(fd, "%03d %g\n", num, All.Time);
        fclose(fd);
    }
}

int
find_last_snapnum()
{
    /* FIXME: this is very fragile; should be fine */
    int snapnumber = -1;
    if(ThisTask == 0) {
        char buf[1024];
        sprintf(buf, "%s/Snapshots.txt", All.OutputDir);
        FILE * fd = fopen(buf, "r");
        if(fd == NULL) {
            snapnumber = -1;
        } else {
            double time;
            char ch;
            int line = 0;
            while (!feof(fd)) {
                int n = fscanf(fd, "%d %lg%c", &snapnumber, &time, &ch);
                message(1, "n = %d\n", n);
                if (n == 3 && ch == '\n') {
                    line ++;
                    continue;
                }
                if (n == -1 && feof(fd)) {
                    continue;
                }
                endrun(1, "Failed to parse %s:%d for the last snap shot number.\n", buf, line);
            }
            fclose(fd);
        }
    }

    MPI_Bcast(&snapnumber, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return snapnumber;
}
