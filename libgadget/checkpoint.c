#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

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

void
write_checkpoint(int snapnum, int WriteSnapshot, int WriteFOF, double Time, const char * OutputDir, const char * SnapshotFileBase, const int OutputDebugFields, ForceTree * tree)
{
    if(!WriteSnapshot && !WriteFOF) return;

    walltime_measure("/Misc");
    if(WriteSnapshot)
    {
        /* write snapshot of particles */
        struct IOTable IOTable = {0};
        register_io_blocks(&IOTable, WriteFOF);
        if(OutputDebugFields)
            register_debug_io_blocks(&IOTable);
        petaio_save_snapshot(&IOTable, 1, "%s/%s_%03d", OutputDir, SnapshotFileBase, snapnum);

        destroy_io_blocks(&IOTable);
        walltime_measure("/Snapshot/Write");

        int ThisTask;
        MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
        if(ThisTask == 0) {
            char * buf = fastpm_strdup_printf("%s/Snapshots.txt", OutputDir);
            FILE * fd = fopen(buf, "a");
            fprintf(fd, "%03d %g\n", snapnum, Time);
            fclose(fd);
            myfree(buf);
        }
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
dump_snapshot(const char * dump, const char * OutputDir)
{
    struct IOTable IOTable = {0};
    register_io_blocks(&IOTable, 0);
    register_debug_io_blocks(&IOTable);
    petaio_save_snapshot(&IOTable, 1, "%s/%s", OutputDir, dump);
    destroy_io_blocks(&IOTable);
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
