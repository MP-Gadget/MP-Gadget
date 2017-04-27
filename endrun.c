#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "endrun.h"

static double _timestart = -1;
/*  This function aborts the simulation.
 *
 *  if ierr > 0, the error is uncollective.
 *  if ierr <= 0, the error is 'collective',  only the root rank prints the error.
 */

void endrun(int ierr, const char * fmt, ...)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(_timestart < 0) _timestart = MPI_Wtime();

    va_list va;
    char buf[4096];
    va_start(va, fmt);
    vsprintf(buf, fmt, va);
    va_end(va);
    if(ierr > 0) {
        printf("[ %09.2f ] Task %d: %s", MPI_Wtime() - _timestart, ThisTask, buf);
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, ierr);
    } else {
        if(ThisTask == 0) {
            printf("[ %09.2f ] %s", MPI_Wtime() - _timestart, buf);
            fflush(stdout);
        }
        MPI_Abort(MPI_COMM_WORLD, ierr);
    }
}

/*  This function writes a message.
 *
 *  if ierr > 0, the message is uncollective.
 *  if ierr <= 0, the message is 'collective', only the root rank prints the message. A barrier is applied.
 */

void message(int ierr, const char * fmt, ...)
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
    if(ierr > 0) {
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
