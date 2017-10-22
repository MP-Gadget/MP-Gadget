#include <math.h>
#include <stdlib.h>
#include <strings.h>
#include <stdio.h>
#include <mpi.h>

#include "bigfile-mpi.h"
#include "genic/allvars.h"
#include "genic/proto.h"
#include "walltime.h"
#include "mymalloc.h"
#include "endrun.h"
#include "petapm.h"

static struct ClockTable CT;
void print_spec(void);

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
  MPI_Comm_size(MPI_COMM_WORLD, &NTask);

  if(argc < 2)
    endrun(0,"Please pass a parameter file.\n");

  /* fixme: make this a mpi bcast */
  read_parameterfile(argv[1]);

  walltime_init(&CT);
  mymalloc_init(MaxMemSizePerNode);
  const double meanspacing = Box / Ngrid;
  const double shift_gas = -0.5 * (CP.Omega0 - CP.OmegaBaryon) / CP.Omega0 * meanspacing;
  const double shift_dm = +0.5 * CP.OmegaBaryon / CP.Omega0 * meanspacing;

  initialize_powerspectrum(ThisTask, InitTime, UnitLength_in_cm, &CP, &PowerP);
  petapm_init(Box, Nmesh, 1);
  setup_grid(ProduceGas * shift_dm);

  /*Write the header*/
  char buf[4096];
  snprintf(buf, 4096, "%s/%s", OutputDir, FileBase);
  BigFile bf;
  if(0 != big_file_mpi_create(&bf, buf, MPI_COMM_WORLD)) {
      endrun(0, "%s\n", big_file_get_error_message());
  }
  saveheader(&bf, Ngrid*Ngrid*Ngrid);

  /*First compute and write CDM*/
  displacement_fields(1);
  write_particle_data(1, &bf);
  /*Now write gas if required*/
  if(ProduceGas) {
    /* If we have different transfer functions
     * we need new displacements.*/
    if(PowerP.DifferentTransferFunctions) {
        free_ffts();
        setup_grid(shift_gas);
        displacement_fields(0);
    }
    /*Otherwise we can just translate the particles*/
    else
        shift_particles(shift_gas - shift_dm, Ngrid*Ngrid*Ngrid);
    write_particle_data(0, &bf);
  }

  free_ffts();

  walltime_summary(0, MPI_COMM_WORLD);
  walltime_report(stdout, 0, MPI_COMM_WORLD);

  message(0, "IC's generated.\n");
  message(0, "Initial scale factor = %g\n", InitTime);

  MPI_Barrier(MPI_COMM_WORLD);
  print_spec();

  MPI_Finalize();		/* clean up & finalize MPI */
  return 0;
}

void print_spec(void)
{
  if(ThisTask == 0)
    {
      double k, kstart, kend, DDD;
      char buf[1000];
      FILE *fd;

      sprintf(buf, "%s/inputspec_%s.txt", OutputDir, FileBase);
      
      fd = fopen(buf, "w");
      if (fd == NULL) {
        message(1, "Failed to create powerspec file at:%s\n", buf);
        return;
      }
      DDD = GrowthFactor(InitTime, 1.0);

      fprintf(fd, "# %12g %12g\n", 1/InitTime-1, DDD);
      /* print actual starting redshift and linear growth factor for this cosmology */
      kstart = 2 * M_PI / (2*Box * (3.085678e24 / UnitLength_in_cm));	/* 2x box size Mpc/h */
      kend = 2 * M_PI / (Box/(8*Ngrid) * (3.085678e24 / UnitLength_in_cm));	/* 1/8 mean spacing Mpc/h */

      message(1,"kstart=%lg kend=%lg\n",kstart,kend);

      for(k = kstart; k < kend; k *= 1.025)
	  {
	    double po = PowerSpec(k, 7);
	    fprintf(fd, "%12g %12g\n", k, po);
	  }
      fclose(fd);
    }
}
