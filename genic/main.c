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
#include <libgadget/partmanager.h>

#define GLASS_SEED_HASH(seed) ((seed) * 9999721L)

void print_spec(void);

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
  if(argc < 2)
    endrun(0,"Please pass a parameter file.\n");

  tamalloc_init();

  read_parameterfile(argv[1]);

  mymalloc_init(All.MaxMemSizePerNode);

  init_endrun(All.ShowBacktrace);

  walltime_init(&All.CT);

  int64_t TotNumPart = (int64_t) All2.Ngrid*All2.Ngrid*All2.Ngrid;
  int64_t TotNumPartGas = (int64_t) All2.ProduceGas*All2.NgridGas*All2.NgridGas*All2.NgridGas;

  init_cosmology(&All.CP, All.TimeIC);

  init_powerspectrum(ThisTask, All.TimeIC, All.UnitLength_in_cm, &All.CP, &All2.PowerP);
  All.NumThreads = omp_get_max_threads();

  petapm_module_init(All.NumThreads);

  /*Initialise particle spacings*/
  const double meanspacing = All.BoxSize / DMAX(All2.Ngrid, All2.NgridGas);
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
  saveheader(&bf, TotNumPart, TotNumPartGas, TotNu, total_nufrac);

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

  petapm_init(pm, All.BoxSize, All.Asmth, All.Nmesh, All.G, MPI_COMM_WORLD);

  /*First compute and write CDM*/
  double mass[6] = {0};
  /*Can neglect neutrinos since this only matters for the glass force.*/
  compute_mass(mass, TotNumPart, TotNumPartGas, 0, 0);
  /*Not used*/
  IDGenerator idgen_cdm[1];
  IDGenerator idgen_gas[1];

  idgen_init(idgen_cdm, pm, All2.Ngrid, All.BoxSize);
  idgen_init(idgen_gas, pm, All2.NgridGas, All.BoxSize);

  int NumPartCDM = idgen_cdm->NumPart;
  int NumPartGas = idgen_gas->NumPart;

  /*Space for both CDM and baryons*/
  struct ic_part_data * ICP = (struct ic_part_data *) mymalloc("PartTable", (NumPartCDM + All2.ProduceGas * NumPartGas)*sizeof(struct ic_part_data));

  /* If we have incoherent glass files, we need to store both the particle tables
   * to ensure that there are no close particle pairs*/
  /*Make the table for the CDM*/
  if(!All2.MakeGlassCDM) {
      setup_grid(idgen_cdm, All2.ProduceGas * shift_dm, mass[1], ICP);
  } else {
      setup_glass(idgen_cdm, pm, 0, GLASS_SEED_HASH(All2.Seed), mass[1], ICP);
  }

  /*Make the table for the baryons if we need, using the second half of the memory.*/
  if(All2.ProduceGas) {
    if(!All2.MakeGlassGas) {
        setup_grid(idgen_gas, shift_gas, mass[0], ICP+NumPartCDM);
    } else {
        setup_glass(idgen_gas, pm, 0, GLASS_SEED_HASH(All2.Seed + 1), mass[0], ICP+NumPartCDM);
    }
    /*Do coherent glass evolution to avoid close pairs*/
    if(All2.MakeGlassGas || All2.MakeGlassCDM)
        glass_evolve(pm, 14, "powerspectrum-glass-tot", ICP, NumPartCDM+NumPartGas);
  }

  /*Write initial positions into ICP struct (for CDM and gas)*/
  int j,k;
  for(j=0; j<NumPartCDM+NumPartGas; j++)
      for(k=0; k<3; k++)
          ICP[j].PrePos[k] = ICP[j].Pos[k];

  if(NumPartCDM > 0) {
    displacement_fields(pm, DMType, ICP, NumPartCDM);

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

        for(i = 0; i < NumPartCDM; i++) {
             /*Find the slab, and reseed if it has zero z rank*/
             if(i % All2.Ngrid == 0) {
                  uint64_t id = idgen_create_id_from_index(idgen_cdm, i);
                  /*Seed the random number table with x,y index.*/
                  gsl_rng_set(g_rng, seedtable[id / All2.Ngrid]);
             }
             add_thermal_speeds(&WDM, g_rng, ICP[i].Vel);
        }
        gsl_rng_free(g_rng);
        myfree(seedtable);
    }

    write_particle_data(idgen_cdm, 1, &bf, 0, ICP);
  }

  /*Now make the gas if required*/
  if(All2.ProduceGas) {
    displacement_fields(pm, GasType, ICP+NumPartCDM, NumPartGas);
    write_particle_data(idgen_gas, 0, &bf, TotNumPart, ICP+NumPartCDM);
  }
  myfree(ICP);

  /*Now add random velocity neutrino particles*/
  if(All2.NGridNu > 0) {
      int i;
      IDGenerator idgen_nu[1];
      idgen_init(idgen_nu, pm, All2.NGridNu, All.BoxSize);

      int NumPartNu = idgen_nu->NumPart;
      ICP = (struct ic_part_data *) mymalloc("PartTable", NumPartNu*sizeof(struct ic_part_data));

      NumPartNu = setup_grid(idgen_nu, shift_nu, mass[2], ICP);

	  /*Write initial positions into ICP struct (for neutrinos)*/
	  for(j=0; j<NumPartNu; j++)
		  for(k=0; k<3; k++)
		      ICP[j].PrePos[k] = ICP[j].Pos[k];

      displacement_fields(pm, NuType, ICP, NumPartNu);
      unsigned int * seedtable = init_rng(All2.Seed+2,All2.NGridNu);
      gsl_rng * g_rng = gsl_rng_alloc(gsl_rng_ranlxd1);
      /*Just in case*/
      gsl_rng_set(g_rng, seedtable[0]);
      for(i = 0; i < NumPartNu; i++) {
           /*Find the slab, and reseed if it has zero z rank*/
           if(i % All2.NGridNu == 0) {
                uint64_t id = idgen_create_id_from_index(idgen_nu, i);
                /*Seed the random number table with x,y index.*/
                gsl_rng_set(g_rng, seedtable[id / All2.NGridNu]);
           }
           add_thermal_speeds(&nu_therm, g_rng, ICP[i].Vel);
      }
      gsl_rng_free(g_rng);
      myfree(seedtable);

      write_particle_data(idgen_nu, 2, &bf, TotNumPart+TotNumPartGas, ICP);
      myfree(ICP);
  }

  petapm_destroy(pm);
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
      DDD = GrowthFactor(&All.CP, All.TimeIC, 1.0);

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

