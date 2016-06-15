#include <math.h>
#include <stdlib.h>
#include <strings.h>
#include <stdio.h>
#include <mpi.h>

#include "allvars.h"
#include "proto.h"
#include "walltime.h"
#include "mymalloc.h"

static struct ClockTable CT;

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
  MPI_Comm_size(MPI_COMM_WORLD, &NTask);

  if(argc < 2)
    {
      if(ThisTask == 0)
	{
	  fprintf(stdout, "\nParameters are missing.\n");
	  fprintf(stdout, "Call with <ParameterFile>\n\n");
	}
      MPI_Finalize();
      exit(0);
    }

  /* fixme: make this a mpi bcast */
  read_parameterfile(argv[1]);

  walltime_init(&CT);
  mymalloc_init(MaxMemoryPerCore * 1024 * 1024);

  set_units();

  initialize_powerspectrum();

  initialize_ffts();

  MPI_Barrier(MPI_COMM_WORLD);

  displacement_fields();

  MPI_Barrier(MPI_COMM_WORLD);

  write_particle_data();
  if(NumPart)
    free(P);

  free_ffts();


  walltime_summary(0, MPI_COMM_WORLD);
  walltime_report(stdout, 0, MPI_COMM_WORLD);
  if(ThisTask == 0)
    {
      printf("\nIC's generated.\n\n");
      printf("Initial scale factor = %g\n", InitTime);
      printf("\n");
    }

  MPI_Barrier(MPI_COMM_WORLD);
  print_spec();

  MPI_Finalize();		/* clean up & finalize MPI */
  exit(0);
}




int FatalError(int errnum)
{
  printf("FatalError called with number=%d\n", errnum);
  fflush(stdout);
  MPI_Abort(MPI_COMM_WORLD, errnum);
  exit(0);
}

double periodic_wrap(double x)
{
  while(x >= Box)
    x -= Box;

  while(x < 0)
    x += Box;

  return x;
}

