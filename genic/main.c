#include <math.h>
#include <stdlib.h>
#include <strings.h>
#include <stdio.h>
#include <mpi.h>
#include <omp.h>

#include <bigfile-mpi.h>
#include <libgenic/allvars.h>
#include <libgenic/proto.h>
#include <libgadget/walltime.h>
#include <libgadget/physconst.h>
#include <libgadget/petapm.h>
#include <libgadget/utils.h>
#include <libgadget/partmanager.h>
#include <libgadget/utils/unitsystem.h>

#define GLASS_SEED_HASH(seed) ((seed) * 9999721L)

static void print_spec(int ThisTask, const int Ngrid, struct genic_config All2, Cosmology * CP);

int main(int argc, char **argv)
{
  int thread_provided, ThisTask;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_provided);
  if(thread_provided != MPI_THREAD_FUNNELED)
    message(1, "MPI_Init_thread returned %d != MPI_THREAD_FUNNELED\n", thread_provided);

  if(argc < 2)
    endrun(0,"Please pass a parameter file.\n");

  tamalloc_init();


  /* Genic Specific configuration structure*/
  struct genic_config All2 = {0};

  Cosmology CP ={0};
  int ShowBacktrace;
  double MaxMemSizePerNode;
  read_parameterfile(argv[1], &All2, &ShowBacktrace, &MaxMemSizePerNode, &CP);
  All2.units = get_unitsystem(All2.units.UnitLength_in_cm, All2.units.UnitMass_in_g, All2.units.UnitVelocity_in_cm_per_s);

  mymalloc_init(MaxMemSizePerNode);

  init_endrun(ShowBacktrace);

  struct ClockTable Clocks;
  walltime_init(&Clocks);

  int64_t TotNumPart = (int64_t) All2.Ngrid*All2.Ngrid*All2.Ngrid;
  int64_t TotNumPartGas = (int64_t) All2.ProduceGas*All2.NgridGas*All2.NgridGas*All2.NgridGas;

  init_cosmology(&CP, All2.TimeIC, All2.units);

  MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
  init_powerspectrum(ThisTask, All2.TimeIC, All2.units.UnitLength_in_cm, &CP, &All2.PowerP);

  petapm_module_init(omp_get_max_threads());

  /*Initialise particle spacings*/
  const double meanspacing = All2.BoxSize / DMAX(All2.Ngrid, All2.NgridGas);
  double shift_gas = -All2.ProduceGas * 0.5 * (CP.Omega0 - CP.OmegaBaryon) / CP.Omega0 * meanspacing;
  double shift_dm = All2.ProduceGas * 0.5 * CP.OmegaBaryon / CP.Omega0 * meanspacing;

  if(All2.PrePosGridCenter){
      shift_dm += 0.5 * meanspacing;
      shift_gas += 0.5 * meanspacing;
  }

  /*Write the header*/
  char buf[4096];
  snprintf(buf, 4096, "%s/%s", All2.OutputDir, All2.InitCondFile);
  BigFile bf;
  if(0 != big_file_mpi_create(&bf, buf, MPI_COMM_WORLD)) {
      endrun(0, "%s\n", big_file_get_error_message());
  }
  /*Massive neutrinos*/

  const int64_t TotNu = (int64_t) All2.NGridNu*All2.NGridNu*All2.NGridNu;
  double total_nufrac = 0;
  saveheader(&bf, TotNumPart, TotNumPartGas, TotNu, total_nufrac, All2.BoxSize, &CP, All2);

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
  PetaPM pm[1];

  double UnitTime_in_s = All2.units.UnitLength_in_cm / All2.units.UnitVelocity_in_cm_per_s;
  double Grav = GRAVITY / pow(All2.units.UnitLength_in_cm, 3) * All2.units.UnitMass_in_g * pow(UnitTime_in_s, 2);

  petapm_init(pm, All2.BoxSize, 0, All2.Nmesh, Grav, MPI_COMM_WORLD);

  /*First compute and write CDM*/
  double mass[6] = {0};
  /*Can neglect neutrinos since this only matters for the glass force.*/
  compute_mass(mass, TotNumPart, TotNumPartGas, 0, 0, All2.BoxSize, &CP, All2);
  /*Not used*/
  IDGenerator idgen_cdm[1];
  IDGenerator idgen_gas[1];

  idgen_init(idgen_cdm, pm, All2.Ngrid, All2.BoxSize);
  idgen_init(idgen_gas, pm, All2.NgridGas, All2.BoxSize);

  int NumPartCDM = idgen_cdm->NumPart;
  int NumPartGas = idgen_gas->NumPart;

  /*Space for both CDM and baryons*/
  struct ic_part_data * ICP = (struct ic_part_data *) mymalloc("PartTable", (NumPartCDM + All2.ProduceGas * NumPartGas)*sizeof(struct ic_part_data));

  /* If we have incoherent glass files, we need to store both the particle tables
   * to ensure that there are no close particle pairs*/
  /*Make the table for the CDM*/
  if(!All2.MakeGlassCDM) {
      setup_grid(idgen_cdm, shift_dm, mass[1], ICP);
  } else {
      setup_glass(idgen_cdm, pm, 0, GLASS_SEED_HASH(All2.Seed), mass[1], ICP, All2.units.UnitLength_in_cm, All2.OutputDir);
  }

  /*Make the table for the baryons if we need, using the second half of the memory.*/
  if(All2.ProduceGas) {
    if(!All2.MakeGlassGas) {
        setup_grid(idgen_gas, shift_gas, mass[0], ICP+NumPartCDM);
    } else {
        setup_glass(idgen_gas, pm, 0, GLASS_SEED_HASH(All2.Seed + 1), mass[0], ICP+NumPartCDM, All2.units.UnitLength_in_cm, All2.OutputDir);
    }
    /*Do coherent glass evolution to avoid close pairs*/
    if(All2.MakeGlassGas || All2.MakeGlassCDM)
        glass_evolve(pm, 14, "powerspectrum-glass-tot", ICP, NumPartCDM+NumPartGas, All2.units.UnitLength_in_cm, All2.OutputDir);
  }

  /*Write initial positions into ICP struct (for CDM and gas)*/
  int j,k;
  for(j=0; j<NumPartCDM+NumPartGas; j++)
      for(k=0; k<3; k++)
          ICP[j].PrePos[k] = ICP[j].Pos[k];

  if(NumPartCDM > 0) {
    displacement_fields(pm, DMType, ICP, NumPartCDM, &CP, All2);
    write_particle_data(idgen_cdm, 1, &bf, 0, All2.SavePrePos, All2.NumFiles, All2.NumWriters, ICP);
  }

  /*Now make the gas if required*/
  if(All2.ProduceGas) {
    displacement_fields(pm, GasType, ICP+NumPartCDM, NumPartGas, &CP, All2);
    write_particle_data(idgen_gas, 0, &bf, TotNumPart, All2.SavePrePos, All2.NumFiles, All2.NumWriters, ICP+NumPartCDM);
  }
  myfree(ICP);

  petapm_destroy(pm);
  big_file_mpi_close(&bf, MPI_COMM_WORLD);

  walltime_summary(0, MPI_COMM_WORLD);
  walltime_report(stdout, 0, MPI_COMM_WORLD);

  message(0, "IC's generated.\n");
  message(0, "Initial scale factor = %g\n", All2.TimeIC);

  print_spec(ThisTask, All2.Ngrid, All2, &CP);

  MPI_Finalize();		/* clean up & finalize MPI */
  return 0;
}

void print_spec(int ThisTask, const int Ngrid, struct genic_config All2, Cosmology * CP)
{
  if(ThisTask == 0)
    {
      double k, kstart, kend, DDD;
      char buf[1000];
      FILE *fd;

      sprintf(buf, "%s/inputspec_%s.txt", All2.OutputDir, All2.InitCondFile);

      fd = fopen(buf, "w");
      if (fd == NULL) {
        message(1, "Failed to create powerspec file at:%s\n", buf);
        return;
      }
      DDD = GrowthFactor(CP, All2.TimeIC, 1.0);

      fprintf(fd, "# %12g %12g\n", 1/All2.TimeIC-1, DDD);
      /* print actual starting redshift and linear growth factor for this cosmology */
      kstart = 2 * M_PI / (2*All2.BoxSize * (CM_PER_MPC / All2.units.UnitLength_in_cm));	/* 2x box size Mpc/h */
      kend = 2 * M_PI / (All2.BoxSize/(8*Ngrid) * (CM_PER_MPC / All2.units.UnitLength_in_cm));	/* 1/8 mean spacing Mpc/h */

      message(1,"kstart=%lg kend=%lg\n",kstart,kend);

      for(k = kstart; k < kend; k *= 1.025)
	  {
	    double po = pow(DeltaSpec(k, DELTA_TOT),2);
	    fprintf(fd, "%12g %12g\n", k, po);
	  }
      fclose(fd);
    }
}

