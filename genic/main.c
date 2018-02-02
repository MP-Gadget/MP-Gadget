#include <math.h>
#include <stdlib.h>
#include <strings.h>
#include <stdio.h>
#include <mpi.h>

#include "bigfile-mpi.h"
#include "genic/allvars.h"
#include "genic/proto.h"
#include "genic/thermal.h"
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
  init_endrun();

  if(argc < 2)
    endrun(0,"Please pass a parameter file.\n");

  /* fixme: make this a mpi bcast */
  read_parameterfile(argv[1]);

  mymalloc_init(MaxMemSizePerNode);

  walltime_init(&CT);

  int64_t TotNumPart = (int64_t) Ngrid*Ngrid*Ngrid;

  init_cosmology(&CP, InitTime, Box * UnitLength_in_cm);

  initialize_powerspectrum(ThisTask, InitTime, UnitLength_in_cm, &CP, &PowerP);
  petapm_init(Box, Nmesh, 1);
  /*Initialise particle spacings*/
  const double meanspacing = Box / Ngrid;
  const double shift_gas = -ProduceGas * 0.5 * (CP.Omega0 - CP.OmegaBaryon) / CP.Omega0 * meanspacing;
  double shift_dm = ProduceGas * 0.5 * CP.OmegaBaryon / CP.Omega0 * meanspacing;
  double shift_nu = 0;
  if(!ProduceGas && NGridNu > 0) {
      double OmegaNu = get_omega_nu(&CP.ONu, 1);
      shift_nu = -0.5 * (CP.Omega0 - OmegaNu) / CP.Omega0 * meanspacing;
      shift_dm = 0.5 * OmegaNu / CP.Omega0 * meanspacing;
  }
  setup_grid(ProduceGas * shift_dm, 0, Ngrid);

  /*Write the header*/
  char buf[4096];
  snprintf(buf, 4096, "%s/%s", OutputDir, FileBase);
  BigFile bf;
  if(0 != big_file_mpi_create(&bf, buf, MPI_COMM_WORLD)) {
      endrun(0, "%s\n", big_file_get_error_message());
  }
  /*Massive neutrinos*/

  const int64_t TotNu = (int64_t) NGridNu*NGridNu*NGridNu;
  double total_nufrac = 0;
  struct thermalvel nu_therm;
  if(TotNu > 0) {
    const double kBMNu = 3*CP.ONu.kBtnu / (CP.MNu[0]+CP.MNu[1]+CP.MNu[2]);
    double v_th = NU_V0(InitTime, kBMNu, UnitVelocity_in_cm_per_s);
    if(!UsePeculiarVelocity)
        v_th /= sqrt(InitTime);
    total_nufrac = init_thermalvel(&nu_therm, v_th, Max_nuvel/v_th, 0);
    message(0,"F-D velocity scale: %g. Max particle vel: %g. Fraction of mass in particles: %g\n",v_th*sqrt(InitTime), Max_nuvel*sqrt(InitTime), total_nufrac);
  }
  saveheader(&bf, TotNumPart, TotNu, total_nufrac);
  /*Use 'total' (CDM + baryon) transfer function
   * unless DifferentTransferFunctions are on.
   */
  int DMType = 3, GasType = 3, NuType = 3;
  if(ProduceGas && DifferentTransferFunctions) {
      NuType = 2;
      DMType = 1;
      GasType = 0;
  }

  /*First compute and write CDM*/
  displacement_fields(DMType);
  write_particle_data(1, &bf);
  free_ffts();

  /*Add a thermal velocity to WDM particles*/
  if(WDM_therm_mass > 0){
      int i;
      double v_th = WDM_V0(InitTime, WDM_therm_mass, CP.Omega0 - CP.OmegaBaryon - get_omega_nu(&CP.ONu, 1), CP.HubbleParam, UnitVelocity_in_cm_per_s);
      if(!UsePeculiarVelocity)
         v_th /= sqrt(InitTime);
      struct thermalvel WDM;
      init_thermalvel(&WDM, v_th, 10000/v_th, 0);
      for(i = 0; i < NumPart; i++)
          add_thermal_speeds(&WDM, P[i].ID, P[i].Vel);
  }

  /*Now make the gas if required*/
  if(ProduceGas) {
    setup_grid(shift_gas, TotNumPart, Ngrid);
    displacement_fields(GasType);
    write_particle_data(0, &bf);
    free_ffts();
  }
  /*Now add random velocity neutrino particles*/
  if(NGridNu > 0) {
      int i;
      setup_grid(shift_nu, 2*TotNumPart, NGridNu);
      displacement_fields(NuType);
      for(i = 0; i < NumPart; i++)
          add_thermal_speeds(&nu_therm, P[i].ID, P[i].Vel);
      write_particle_data(2,&bf);
      free_ffts();
  }

  big_file_mpi_close(&bf, MPI_COMM_WORLD);

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
	    double po = PowerSpec(k, -1);
	    fprintf(fd, "%12g %12g\n", k, po);
	  }
      fclose(fd);
    }
}
