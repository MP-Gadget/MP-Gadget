#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "endrun.h"

/* Watch out:
 *
 * On some versions of OpenMPI with CPU frequency scaling we see negative time
 * due to a bug in OpenMPI https://github.com/open-mpi/ompi/issues/3003
 *
 * But they have fixed it.
 */

static double _timestart = -1;
/*  This function aborts the simulation.
 *
 *  if where > 0, the error is uncollective.
 *  if where <= 0, the error is 'collective',  only the root rank prints the error.
 */

void endrun(int where, const char * fmt, ...)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(_timestart < 0) _timestart = MPI_Wtime();

    va_list va;
    char buf[4096];
    va_start(va, fmt);
    vsprintf(buf, fmt, va);
    va_end(va);
    if(where > 0) {
        printf("[ %09.2f ] Task %d: %s", MPI_Wtime() - _timestart, ThisTask, buf);
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, where);
    } else {
        if(ThisTask == 0) {
            printf("[ %09.2f ] %s", MPI_Wtime() - _timestart, buf);
            fflush(stdout);
        }
        MPI_Abort(MPI_COMM_WORLD, where);
    }
}

/*  This function writes a message.
 *
 *  if where > 0, the message is uncollective.
 *  if where <= 0, the message is 'collective', only the root rank prints the message. A barrier is applied.
 */

void message(int where, const char * fmt, ...)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    if(_timestart < 0) _timestart = MPI_Wtime();

    va_list va;
    char buf[4096];
    va_start(va, fmt);
    vsprintf(buf, fmt, va);
    va_end(va);
    /* FIXME: deal with \n in the buf. */
    if(where > 0) {
        printf("[ %09.2f ] Task %d: %s", MPI_Wtime() - _timestart, ThisTask, buf);
        fflush(stdout);
    } else {
        MPI_Barrier(MPI_COMM_WORLD);
        if(ThisTask == 0) {
            printf("[ %09.2f ] %s", MPI_Wtime() - _timestart, buf);
            fflush(stdout);
        }
    }
}
