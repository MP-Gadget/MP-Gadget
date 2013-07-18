#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <signal.h>
#include <unistd.h>

#include "allvars.h"
#include "proto.h"

/*  This function aborts the simulations. If a single processors
 *  wants an immediate termination,  the function needs to be 
 *  called with ierr>0. A bunch of MPI-error messages will also
 *  appear in this case.
 *  For ierr=0, MPI is gracefully cleaned up, but this requires
 *  that all processors call endrun().
 */

void endrun(int ierr)
{
  if(ierr)
    {
      printf("task %d: endrun called with an error level of %d\n\n\n", ThisTask, ierr);
      fflush(stdout);
      MPI_Abort(MPI_COMM_WORLD, ierr);
      exit(0);
    }

  MPI_Finalize();
  exit(0);
}

