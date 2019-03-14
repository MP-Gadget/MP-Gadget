#include <math.h>
#include <stdlib.h>
#include <strings.h>
#include <stdio.h>
#include <mpi.h>
#include <omp.h>

#include <bigfile-mpi.h>
#include <libgenic/allvars.h>
#include <libgenic/proto.h>
#include <libgenic/thermal.h>
#include <libgadget/walltime.h>
#include <libgadget/petapm.h>
#include <libgadget/utils.h>

#define GLASS_SEED_HASH(seed) ((seed) * 9999721L)

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

  init_cosmology(&All.CP, All.TimeIC);

  init_powerspectrum(ThisTask, All.TimeIC, All.UnitLength_in_cm, &All.CP, &All2.PowerP);
  All.NumThreads = omp_get_max_threads();

  petapm_init(All.BoxSize, All.Nmesh, All.NumThreads);
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

  /*Save the transfer functions*/
  save_all_transfer_tables(&bf, ThisTask);

  /*Use 'total' (CDM + baryon) transfer function
   * unless DifferentTransferFunctions are on.
   */
  enum TransferType DMType = DELTA_CB, GasType = DELTA_CB, NuType = DELTA_NU;
  if(All2.ProduceGas && All2.PowerP.DifferentTransferFunctions) {
      DMType = DELTA_CDM;
      GasType = DELTA_BAR;
  }

  /*First compute and write CDM*/

  if(!All2.MakeGlass) {
      setup_grid(All2.ProduceGas * shift_dm, All2.Ngrid);
  } else {
      setup_glass(All2.ProduceGas * shift_dm, All2.Ngrid, GLASS_SEED_HASH(All2.Seed));
  }

  displacement_fields(DMType);

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
           if(i % All2.Ngrid == 0) {
                uint64_t id = id_offset_from_index(i, All2.Ngrid);
                /*Seed the random number table with x,y index.*/
                gsl_rng_set(g_rng, seedtable[id / All2.Ngrid]);
           }
           add_thermal_speeds(&WDM, g_rng, ICP[i].Vel);
      }
      gsl_rng_free(g_rng);
      myfree(seedtable);
  }

  write_particle_data(1, &bf, 0, All2.Ngrid);
  free_ffts();

  /*Now make the gas if required*/
  if(All2.ProduceGas) {

    if(!All2.MakeGlass) {
        setup_grid(shift_gas, All2.Ngrid);
    } else {
        setup_glass(shift_gas, All2.Ngrid, GLASS_SEED_HASH(All2.Seed + 1));
    }

    displacement_fields(GasType);
    write_particle_data(0, &bf, TotNumPart, All2.Ngrid);
    free_ffts();
  }
  /*Now add random velocity neutrino particles*/
  if(All2.NGridNu > 0) {
      int i;
      if(!All2.MakeGlass) {
        setup_grid(shift_nu, All2.NGridNu);
      } else {
        setup_glass(shift_nu, All2.Ngrid, GLASS_SEED_HASH(All2.Seed + 2));
      }

      displacement_fields(NuType);
      unsigned int * seedtable = init_rng(All2.Seed+2,All2.Ngrid);
      gsl_rng * g_rng = gsl_rng_alloc(gsl_rng_ranlxd1);
      /*Just in case*/
      gsl_rng_set(g_rng, seedtable[0]);
      for(i = 0; i < NumPart; i++) {
           /*Find the slab, and reseed if it has zero z rank*/
           if(i % All2.Ngrid == 0) {
                uint64_t id = id_offset_from_index(i, All2.Ngrid);
                /*Seed the random number table with x,y index.*/
                gsl_rng_set(g_rng, seedtable[id / All2.Ngrid]);
           }
           add_thermal_speeds(&nu_therm, g_rng, ICP[i].Vel);
      }
      gsl_rng_free(g_rng);
      myfree(seedtable);

      write_particle_data(2,&bf, 2*TotNumPart, All2.NGridNu);
      free_ffts();
  }

  big_file_mpi_close(&bf, MPI_COMM_WORLD);

  walltime_summary(0, MPI_COMM_WORLD);
  walltime_report(stdout, 0, MPI_COMM_WORLD);

  message(0, "IC's generated.\n");
  message(0, "Initial scale factor = %g\n", All.TimeIC);

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
      kstart = 2 * M_PI / (2*All.BoxSize * (CM_PER_MPC / All.UnitLength_in_cm));	/* 2x box size Mpc/h */
      kend = 2 * M_PI / (All.BoxSize/(8*All2.Ngrid) * (CM_PER_MPC / All.UnitLength_in_cm));	/* 1/8 mean spacing Mpc/h */

      message(1,"kstart=%lg kend=%lg\n",kstart,kend);

      for(k = kstart; k < kend; k *= 1.025)
	  {
	    double po = pow(DeltaSpec(k, DELTA_TOT),2);
	    fprintf(fd, "%12g %12g\n", k, po);
	  }
      fclose(fd);
    }
}
