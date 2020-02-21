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
#include "walltime.h"
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
write_checkpoint(int WriteSnapshot, int WriteFOF, ForceTree * tree)
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

        FOFGroups fof = fof_fof(tree, MPI_COMM_WORLD);
        /* Tree is invalid now because of the exchange in FoF.*/
        force_tree_free(tree);
        fof_save_groups(&fof, snapnum, MPI_COMM_WORLD);
        fof_finish(&fof);

        message(0, "done with group catalogue.\n");
    }
}

void
dump_snapshot(const char * dump)
{
    struct IOTable IOTable = {0};
    register_io_blocks(&IOTable);
    register_debug_io_blocks(&IOTable);
    petaio_save_snapshot(&IOTable, 1, "%s/%s", All.OutputDir, dump);
    destroy_io_blocks(&IOTable);
}

static void
write_snapshot(int num)
{
    walltime_measure("/Misc");
    struct IOTable IOTable = {0};
    register_io_blocks(&IOTable);
    if(All.OutputDebugFields)
        register_debug_io_blocks(&IOTable);
    petaio_save_snapshot(&IOTable, 1, "%s/%s_%03d", All.OutputDir, All.SnapshotFileBase, num);

    destroy_io_blocks(&IOTable);
    walltime_measure("/Snapshot/Write");

    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        char * buf = fastpm_strdup_printf("%s/Snapshots.txt", All.OutputDir);
        FILE * fd = fopen(buf, "a");
        fprintf(fd, "%03d %g\n", num, All.Time);
        fclose(fd);
        myfree(buf);
    }
}

int
find_last_snapnum(const char * OutputDir)
{
    /* FIXME: this is very fragile; should be fine */
    int snapnumber = -1;
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        char * buf = fastpm_strdup_printf("%s/Snapshots.txt", OutputDir);
        FILE * fd = fopen(buf, "r");
        if(fd == NULL) {
            snapnumber = -1;
        } else {
            double time;
            char ch;
            int line = 0;
            while (!feof(fd)) {
                int n = fscanf(fd, "%d %lg%c", &snapnumber, &time, &ch);
//                 message(1, "n = %d\n", n);
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
        myfree(buf);
    }

    MPI_Bcast(&snapnumber, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return snapnumber;
}
