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
#include "checkpoint.h"
#include "fof.h"

#include "utils.h"

/*! \file io.c
 *  \brief Output of a snapshot file to disk.
 *
 *  This file delegates the functions to petaio and fof.
 */

static void
write_snapshot(int num);

void
write_checkpoint(int WriteSnapshot, int WriteFOF)
{
    if(!WriteSnapshot && !WriteFOF) return;

    int snapnum = All.SnapshotFileCount++;

    if(WriteSnapshot)
    {
        /* write snapshot of particles */
        write_snapshot(snapnum);
    }

    if(WriteFOF) {
        /* Compute and save FOF*/
        message(0, "computing group catalogue...\n");

        fof_fof();
        fof_save_groups(snapnum);
        fof_finish();

        message(0, "done with group catalogue.\n");
    }
}

void
dump_snapshot()
{
    petaio_save_snapshot("%s/CRASH-DUMP", All.OutputDir);
}

static void
write_snapshot(int num)
{
    walltime_measure("/Misc");

    petaio_save_snapshot("%s/%s_%03d", All.OutputDir, All.SnapshotFileBase, num);

    walltime_measure("/Snapshot/Write");

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
