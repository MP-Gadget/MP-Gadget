#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "allvars.h"
#include "endrun.h"

/*  This function aborts the simulations. If a single processors
 *  wants an immediate termination,  the function needs to be 
 *  called with ierr>0. A bunch of MPI-error messages will also
 *  appear in this case.
 *  For ierr=0, MPI is gracefully cleaned up, but this requires
 *  that all processors call endrun().
 */

void endrun(int ierr, const char * fmt, ...)
{
    va_list va;
    char buf[4096];
    va_start(va, fmt);
    vsprintf(buf, fmt, va);
    va_end(va);
    printf("Task %d: Error (%d) %s\n", ThisTask, ierr, buf);
    fflush(stdout);
    BREAKPOINT;
    MPI_Abort(MPI_COMM_WORLD, ierr);
}
