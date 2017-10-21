#include <math.h>
#include <stdlib.h>
#include <strings.h>
#include <stdio.h>
#include <mpi.h>

#include "genic/allvars.h"
#include "genic/proto.h"
#include "walltime.h"
#include "mymalloc.h"
#include "endrun.h"

static struct ClockTable CT;
void print_spec(void);

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
  mymalloc_init(MaxMemSizePerNode);

  initialize_powerspectrum(ThisTask, InitTime, UnitLength_in_cm, &CP, &PowerP);

  initialize_ffts();

  MPI_Barrier(MPI_COMM_WORLD);

  displacement_fields();

  MPI_Barrier(MPI_COMM_WORLD);

  write_particle_data();
  if(NumPart)
    myfree(P);

  free_ffts();


  walltime_summary(0, MPI_COMM_WORLD);
  walltime_report(stdout, 0, MPI_COMM_WORLD);

      message(0, "IC's generated.\n");
      message(0, "Initial scale factor = %g\n", InitTime);

  MPI_Barrier(MPI_COMM_WORLD);
  print_spec();

  MPI_Finalize();		/* clean up & finalize MPI */
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

void print_spec(void)
{
  if(ThisTask == 0)
    {
      double k, po, dl, kstart, kend, DDD;
      char buf[1000];
      FILE *fd;

      sprintf(buf, "%s/inputspec_%s.txt", OutputDir, FileBase);
      
      fd = fopen(buf, "w");
      if (fd == NULL) {
          printf("Failed to create powerspec file at:%s\n", buf);
        return;
      }
      DDD = GrowthFactor(InitTime, 1.0);

      fprintf(fd, "%12g %12g\n", 1/InitTime-1, DDD);	/* print actual starting redshift and 
							   linear growth factor for this cosmology */
      kstart = 2 * M_PI / (1000.0 * (3.085678e24 / UnitLength_in_cm));	/* 1000 Mpc/h */
      kend = 2 * M_PI / (0.001 * (3.085678e24 / UnitLength_in_cm));	/* 0.001 Mpc/h */

      printf("kstart=%lg kend=%lg\n",kstart,kend);

      for(k = kstart; k < kend; k *= 1.025)
	  {
	    po = PowerSpec(k, 7);
	    dl = 4.0 * M_PI * k * k * k * po;
	    fprintf(fd, "%12g %12g\n", k, dl);
	  }
      fclose(fd);
    }
}
