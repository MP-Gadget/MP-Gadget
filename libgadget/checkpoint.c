#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "petaio.h"
#include "checkpoint.h"
#include "walltime.h"
#include "utils/string.h"
#include "utils/endrun.h"
#include "utils/mymalloc.h"

/*! \file io.c
 *  \brief Output of a snapshot file to disk.
 *
 *  This file delegates the functions to petaio and fof.
 */

void
write_checkpoint(int snapnum, int WriteGroupID, int MetalReturnOn, double Time, const Cosmology * CP, const char * OutputDir, const int OutputDebugFields)
{
    /* write snapshot of particles */
    struct IOTable IOTable = {0};
    register_io_blocks(&IOTable, WriteGroupID, MetalReturnOn);
    if(OutputDebugFields)
        register_debug_io_blocks(&IOTable);
    char * fname = petaio_get_snapshot_fname(snapnum, OutputDir);
    petaio_save_snapshot(fname, &IOTable, 1, Time, CP);
    myfree(fname);

    destroy_io_blocks(&IOTable);
    walltime_measure("/WriteSnapshot");

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

void
dump_snapshot(const char * dump, const double Time, const Cosmology * CP, const char * OutputDir)
{
    struct IOTable IOTable = {0};
    register_io_blocks(&IOTable, 0, 1);
    register_debug_io_blocks(&IOTable);
    char * fname = fastpm_strdup_printf("%s/%s", OutputDir, dump);
    petaio_save_snapshot(fname, &IOTable, 1, Time, CP);
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
