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

/*! \file io.c
 *  \brief Output of a snapshot file to disk.
 */

/*! This function writes a snapshot of the particle distribution to one or
 * several files using Gadget's default file format.  If
 * NumFilesPerSnapshot>1, the snapshot is distributed into several files,
 * which are written simultaneously. Each file contains data from a group of
 * processors of size roughly NTask/NumFilesPerSnapshot.
 */
/* reason == 0 regular snapshot, do fof and write it out */
/* reason != 0 checkpoints, do not run fof */
void savepositions(int num, int reason)
{
    walltime_measure("/Misc");

#if defined(SFR) || defined(BLACK_HOLES)
    rearrange_particle_sequence();
    /* ensures that new tree will be constructed */
    All.NumForcesSinceLastDomainDecomp = (int64_t) (1 + All.TreeDomainUpdateFrequency * All.TotNumPart);
#endif

    walltime_measure("/Snapshot/Misc");
    petaio_save_snapshot(num);
    walltime_measure("/Snapshot/Write");

#ifdef FOF
    /* regular snapshot, do fof and write it out */
    if(reason == 0) {
        if(ThisTask == 0)
            printf("\ncomputing group catalogue...\n");

        fof_fof(num);

        if(ThisTask == 0)
            printf("done with group catalogue.\n");
    }
    walltime_measure("/Snapshot/WriteFOF");
#endif

    if(ThisTask == 0) {
        char buf[1024];
        sprintf(buf, "%s/LastSnapshotNum.txt", All.OutputDir);
        FILE * fd = fopen(buf, "a");
        fprintf(fd, "Time %g Redshift %g Ti_current %d Snapnumber %03d\n", All.Time, 1 / All.Time - 1, All.Ti_Current, num);
        fclose(fd);

    }
}

size_t my_fwrite(void *ptr, size_t size, size_t nmemb, FILE * stream)
{
    size_t nwritten;

    if(size * nmemb > 0)
    {
        if((nwritten = fwrite(ptr, size, nmemb, stream)) != nmemb)
        {
            printf("I/O error (fwrite) on task=%d has occured: %s\n", ThisTask, strerror(errno));
            fflush(stdout);
            endrun(777);
        }
    }
    else
        nwritten = 0;

    return nwritten;
}

size_t my_fread(void *ptr, size_t size, size_t nmemb, FILE * stream)
{
    size_t nread;

    if(size * nmemb == 0)
        return 0;

    if((nread = fread(ptr, size, nmemb, stream)) != nmemb)
    {
        if(feof(stream))
            printf("I/O error (fread) on task=%d has occured: end of file\n", ThisTask);
        else
            printf("I/O error (fread) on task=%d has occured: %s\n", ThisTask, strerror(errno));
        fflush(stdout);
        endrun(778);
    }
    return nread;
}

                

