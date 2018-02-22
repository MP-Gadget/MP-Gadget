#include <math.h>
#include <stdlib.h>
#include <strings.h>
#include <stdio.h>
#include <mpi.h>

#include <bigfile-mpi.h>
#include <libgenic/allvars.h>
#include <libgenic/proto.h>
#include <libgenic/thermal.h>
#include <libgadget/walltime.h>
#include <libgadget/petapm.h>
#include <libgadget/utils.h>

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

  mymalloc_init(All.MaxMemSizePerNode);

  walltime_init(&All.CT);

  int64_t TotNumPart = (int64_t) All2.Ngrid*All2.Ngrid*All2.Ngrid;

  init_cosmology(&All.CP, All.TimeIC, All.BoxSize * All.UnitLength_in_cm);

  initialize_powerspectrum(ThisTask, All.TimeIC, All.UnitLength_in_cm, &All.CP, &All2.PowerP);
  petapm_init(All.BoxSize, All.Nmesh, 1);
  /*Initialise particle spacings*/
  const double meanspacing = All.BoxSize / All2.Ngrid;
  const double shift_gas = -All2.ProduceGas * 0.5 * (All.CP.Omega0 - All.CP.OmegaBaryon) / All.CP.Omega0 * meanspacing;
  double shift_dm = All2.ProduceGas * 0.5 * All.CP.OmegaBaryon / All.CP.Omega0 * meanspacing;
  double shift_nu = 0;
  if(!All2.ProduceGas && All2.NGridNu > 0) {
      double OmegaNu = get_omega_nu(&All.CP.ONu, 1);
      shift_nu = -0.5 * (All.CP.Omega0 - OmegaNu) / All.CP.Omega0 * meanspacing;
      shift_dm = 0.5 * OmegaNu / All.CP.Omega0 * meanspacing;
  }
  setup_grid(All2.ProduceGas * shift_dm, 0, All2.Ngrid);

  /*Write the header*/
  char buf[4096];
  snprintf(buf, 4096, "%s/%s", All.OutputDir, All.InitCondFile);
  BigFile bf;
  if(0 != big_file_mpi_create(&bf, buf, MPI_COMM_WORLD)) {
      endrun(0, "%s\n", big_file_get_error_message());
  }
  /*Massive neutrinos*/

  const int64_t TotNu = (int64_t) All2.NGridNu*All2.NGridNu*All2.NGridNu;
  double total_nufrac = 0;
  struct thermalvel nu_therm;
  if(TotNu > 0) {
    const double kBMNu = 3*All.CP.ONu.kBtnu / (All.CP.MNu[0]+All.CP.MNu[1]+All.CP.MNu[2]);
    double v_th = NU_V0(All.TimeIC, kBMNu, All.UnitVelocity_in_cm_per_s);
    if(!All.IO.UsePeculiarVelocity)
        v_th /= sqrt(All.TimeIC);
    total_nufrac = init_thermalvel(&nu_therm, v_th, All2.Max_nuvel/v_th, 0);
    message(0,"F-D velocity scale: %g. Max particle vel: %g. Fraction of mass in particles: %g\n",v_th*sqrt(All.TimeIC), All2.Max_nuvel*sqrt(All.TimeIC), total_nufrac);
  }
  saveheader(&bf, TotNumPart, TotNu, total_nufrac);
  /*Use 'total' (CDM + baryon) transfer function
   * unless DifferentTransferFunctions are on.
   */
  int DMType = 3, GasType = 3, NuType = 3;
  if(All2.ProduceGas && All2.DifferentTransferFunctions) {
      NuType = 2;
      DMType = 1;
      GasType = 0;
  }

  /*First compute and write CDM*/
  displacement_fields(DMType);
  write_particle_data(1, &bf);
  free_ffts();

  /*Add a thermal velocity to WDM particles*/
  if(All2.WDM_therm_mass > 0){
      int i;
      double v_th = WDM_V0(All.TimeIC, All2.WDM_therm_mass, All.CP.Omega0 - All.CP.OmegaBaryon - get_omega_nu(&All.CP.ONu, 1), All.CP.HubbleParam, All.UnitVelocity_in_cm_per_s);
      if(!All.IO.UsePeculiarVelocity)
         v_th /= sqrt(All.TimeIC);
      struct thermalvel WDM;
      init_thermalvel(&WDM, v_th, 10000/v_th, 0);
      unsigned int * seedtable = init_rng(All2.Seed+1,All2.Ngrid);
      gsl_rng * g_rng = gsl_rng_alloc(gsl_rng_ranlxd1);
      /*Seed the random number table with the Id.*/
      gsl_rng_set(g_rng, seedtable[0]);

      for(i = 0; i < NumPart; i++) {
           /*Find the slab, and reseed if it has zero z rank*/
           if((ICP[i].ID -1) % All2.Ngrid == 0) {
                /*Seed the random number table with x,y index.*/
                gsl_rng_set(g_rng, seedtable[(ICP[i].ID-1) / All2.Ngrid]);
           }
           add_thermal_speeds(&WDM, g_rng, ICP[i].Vel);
      }
      gsl_rng_free(g_rng);
      myfree(seedtable);
  }

  /*Now make the gas if required*/
  if(All2.ProduceGas) {
    setup_grid(shift_gas, TotNumPart, All2.Ngrid);
    displacement_fields(GasType);
    write_particle_data(0, &bf);
    free_ffts();
  }
  /*Now add random velocity neutrino particles*/
  if(All2.NGridNu > 0) {
      int i;
      setup_grid(shift_nu, 2*TotNumPart, All2.NGridNu);
      displacement_fields(NuType);
      unsigned int * seedtable = init_rng(All2.Seed+2,All2.Ngrid);
      gsl_rng * g_rng = gsl_rng_alloc(gsl_rng_ranlxd1);
      /*Just in case*/
      gsl_rng_set(g_rng, seedtable[0]);
      for(i = 0; i < NumPart; i++) {
           /*Find the slab, and reseed if it has zero z rank*/
           if((ICP[i].ID -1 - 2*TotNumPart) % All2.Ngrid == 0) {
                /*Seed the random number table with x,y index.*/
                gsl_rng_set(g_rng, seedtable[(ICP[i].ID-1 - 2*TotNumPart) / All2.Ngrid]);
           }
           add_thermal_speeds(&nu_therm, g_rng, ICP[i].Vel);
      }
      gsl_rng_free(g_rng);
      myfree(seedtable);

      write_particle_data(2,&bf);
      free_ffts();
  }

  big_file_mpi_close(&bf, MPI_COMM_WORLD);

  walltime_summary(0, MPI_COMM_WORLD);
  walltime_report(stdout, 0, MPI_COMM_WORLD);

  message(0, "IC's generated.\n");
  message(0, "Initial scale factor = %g\n", All.TimeIC);

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

      sprintf(buf, "%s/inputspec_%s.txt", All.OutputDir, All.InitCondFile);

      fd = fopen(buf, "w");
      if (fd == NULL) {
        message(1, "Failed to create powerspec file at:%s\n", buf);
        return;
      }
      DDD = GrowthFactor(All.TimeIC, 1.0);

      fprintf(fd, "# %12g %12g\n", 1/All.TimeIC-1, DDD);
      /* print actual starting redshift and linear growth factor for this cosmology */
      kstart = 2 * M_PI / (2*All.BoxSize * (3.085678e24 / All.UnitLength_in_cm));	/* 2x box size Mpc/h */
      kend = 2 * M_PI / (All.BoxSize/(8*All2.Ngrid) * (3.085678e24 / All.UnitLength_in_cm));	/* 1/8 mean spacing Mpc/h */

      message(1,"kstart=%lg kend=%lg\n",kstart,kend);

      for(k = kstart; k < kend; k *= 1.025)
	  {
	    double po = PowerSpec(k, -1);
	    fprintf(fd, "%12g %12g\n", k, po);
	  }
      fclose(fd);
    }
}
