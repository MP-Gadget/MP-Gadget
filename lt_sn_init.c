
void setup_SF_related(int);

int calculate_effective_yields(double inf, double sup, int mySFi)
{
  double abserr, result;

  int i;

  int nonZero = 0;

  F.function = &zmRSnII;

  SD.ZArray = IIZbins[SD.Yset];
  SD.MArray = IIMbins[SD.Yset];
  SD.Mdim = IIMbins_dim[SD.Yset];
  SD.Zdim = IIZbins_dim[SD.Yset];

  for(SD.Zbin = 0; SD.Zbin < IIZbins_dim[SD.Yset]; SD.Zbin++)
    {
      if(IIZbins_dim[SD.Yset] > 1)
	SD.Zstar = IIZbins[SD.Yset][SD.Zbin];
      else
	SD.Zstar = 0;
      for(i = 0; i < LT_NMet; i++)
	{
	  SD.Y = SnIIYields[SD.Yset][i];
	  if((gsl_status =
	      gsl_integration_qag(&F, inf, sup, 1e-6, 1e-4, gsl_ws_dim, qag_INT_KEY, w, &result, &abserr)))
	    {
	      if(ThisTask == 0)
		printf("  >> Task %i, gsl integration error %i in calculating effective yields"
		       " [%9.7g - %9.7g] : %9.7g %9.7g\n", ThisTask, gsl_status, inf, sup, result, abserr);
	      fflush(stdout);
	      endrun(LT_ERROR_INTEGRATION_ERROR);
	    }

	  if((SnII_ShortLiv_Yields[mySFi][i][SD.Zbin] = result) > 0)
	    nonZero++;
	}
    }

  if(ThisTask == 0)
    printf("\n");

  return nonZero;
}

double calculate_FactorSN(double m_inf, double m_sup, void *params)
{
  double abserr, result;

  F.function = &ejectaSnII;
  F.params = params;

  if((gsl_status =
      gsl_integration_qag(&F, m_inf, m_sup, 1e-4, 1e-3, gsl_ws_dim, qag_INT_KEY, w, &result, &abserr)))
    {
      if(ThisTask == 0)
	printf
	  ("  >> Task %i, gsl integration error %i in calculating FactorSN [%9.7g - %9.7g] : %9.7g %9.7g\n",
	   ThisTask, gsl_status, m_inf, m_sup, result, abserr);
      fflush(stdout);
      endrun(LT_ERROR_INTEGRATION_ERROR);
    }
  return result;
}






void init_SN(void)
{
  int i, j, guess_on_Nsteps;

  double astep, meanweight;

  char buf[200], buffer[200], mode[2];

#if defined (UM_CHEMISTRY) && defined (UM_METAL_COOLING)
  um_FillEl_mu = 40.0;
#endif

  /* : ....................... : */
  if(ThisTask == 0)		/* :[ open some output file ]: */
    {
      if(RestartFlag == 0)
	strcpy(mode, "w");
      else
	strcpy(mode, "a");

      sprintf(buf, "%s%s", All.OutputDir, "sn_init.txt");	/*   this will list all the fundamental data */
      if((FdSnInit = fopen(buf, mode)) == 0x0)	/*   about supernonave                       */
	{
	  printf("error in opening file '%s'\n", buf);
	  endrun(1);
	}

      if(strcmp(mode, "a") == 0x0)
	fprintf(FdSnInit, "========================================\n" "restarting from a= %g\n\n", All.Time);

#ifdef LT_SEv_INFO_DETAILS
      sprintf(buf, "%s%s", All.OutputDir, "sn_details.dat");	/*   this file will contain all the details */
      /*   about all the sn evolution steps       */
      if((FdSnDetails = fopen(buf, mode)) == 0x0)
	{
	  printf("error in opening file '%s'\n", buf);
	  endrun(1);
	}
#endif
      sprintf(buf, "%s%s", All.OutputDir, "warnings.txt");	/*   this file will store all warnings      */
      if((FdWarn = fopen(buf, mode)) == 0x0)
	{
	  printf("error in opening file '%s'\n", buf);
	  endrun(1);
	}
    }

  UnitMassFact = All.UnitMass_in_g / SOLAR_MASS;

  /* : ................................ : */
  /* :[ allocate integration workspace ]: */
  w = gsl_integration_workspace_alloc(gsl_ws_dim);
  old_error_handler = gsl_set_error_handler(&fsolver_error_handler);
  /* : ................... : */
  /* :[ start the cooling ]: */

  All.Time = All.TimeBegin;
  InitCool();
  /* : ............................... :    */
  /* :[ start the IMF  and SF StartUp ]:    */
  allocate_IMF_integration_space();	/*    allocate memory for IMF integration */
  load_SFs_IMFs();		/*    load IMF and SF files               */
  All.Generations = 0;

  for(i = 0; i < IMFs_dim; i++)	/*    normalize IMFs by mass if they are not */
    {
      if(IMFs[i].NParams > 0 || IMFs[i].timedep)
	IMFs[i].getp(i, IMFs[i].Params, All.Time);	/*    get parameters for a (time-dependent) IMF */

      IMFs[i].Atot = 1.0 / IntegrateIMF_byMass(IMFs[i].Mm, IMFs[i].MU, &IMFs[i], INC_BH);
      if(IMFs[i].NSlopes == 1)
	IMFs[i].A[0] = IMFs[i].Atot;
    }

  for(i = 0; i < SFs_dim; i++)
    if(SFs[i].Generations > All.Generations)
      All.Generations = SFs[i].Generations;	/*    set All.Generations to the maximum */

  if(ThisTask == 0)
    for(i = 0; i < SFs_dim; i++)
      if(All.Generations % SFs[i].Generations != 0)
	{
	  printf("!!!! max # of generations %d is not an integer multiple of the # of gen. for SF %d\n",
		 All.Generations, i);
	  endrun(99119911);
	}

  for(All.StarBits = 0; All.Generations > (1 << All.StarBits); All.StarBits++);

  if(ThisTask == 0)		/*    write infos about IMFs */
    {
      for(i = 0; i < IMFs_dim; i++)
	write_IMF_info(i, stdout);
      for(i = 0; i < IMFs_dim; i++)
	write_IMF_info(i, FdSnInit);
      for(i = 0; i < IMFs_dim; i++)
	print_IMF(i, All.IMFfilename);

      for(i = 0; i < SFs_dim; i++)
	write_SF_info(i, stdout);
      for(i = 0; i < SFs_dim; i++)
	write_SF_info(i, FdSnInit);
    }
  /* : .................................. : */
  /* :[ initialize the packing stricture ]: */

#ifdef LT_TRACK_CONTRIBUTES
  init_packing();
#endif

  if(strcasecmp(All.SnIaDataFile, "none") == 0 || All.Ia_Nset_ofYields == 0)
    UseSnIa = 0;
  else
    UseSnIa = 1;

  if(strcasecmp(All.SnIIDataFile, "none") == 0 || All.II_Nset_ofYields == 0)
    UseSnII = 0;
  else
    UseSnII = 1;

  if(strcasecmp(All.AGBDataFile, "none") == 0 || All.AGB_Nset_ofYields == 0)
    UseAGB = 0;
  else
    UseAGB = 1;

  /* : ............................................... : */
  /* :[ initialize time scales for stellar evolutions ]: */
  initialize_star_lifetimes();

  if(ThisTask == 0)
    {
      fprintf(FdSnInit, "[stellar evolution initialization - lifetimes]");
      fprintf(FdSnInit, "\nstellar lifetimes: mean (%5.4g Msun) = %g Gyrs\n", All.Mup, All.mean_lifetime);
      for(i = 0; i < IMFs_dim; i++)
	fprintf(FdSnInit, "   IMF %3d         inf (%5.4g Msun) = %g Gyrs\n"
		"                   sup (%5.4g Msun) = %g Gyrs\n",
		i, IMFs[i].MU, IMFs[i].inf_lifetime, min(IMFs[i].Mm, All.MBms), IMFs[i].sup_lifetime);
    }
  /* : ................ : */
  /* :[ metals StartUp ]: */
  read_metals();


  /* : ............ : */
  /* :[ Sn StartUp ]: */

  /*    reading yields */

  ReadYields(1, 1, 1);
  /* : ............ : */
  /* :[ SF StartUp ]: */


  /* Set some SF-related Units (old set_units_sfr) */
  meanweight = 4 / (1 + 3 * HYDROGEN_MASSFRAC);	/* note: assuming NEUTRAL GAS */

  All.EgySpecCold = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.TempClouds;
  All.EgySpecCold *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;

  meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));	/* note: assuming FULL ionization */

  All.OverDensThresh =
    All.CritOverDensity * All.OmegaBaryon * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.G);

#ifdef INTERNAL_CRIT_DENSITY
  All.PhysDensThresh = All.CritPhysDensity;
#else
  All.PhysDensThresh = All.CritPhysDensity * PROTONMASS / HYDROGEN_MASSFRAC / All.UnitDensity_in_cgs /
    (All.HubbleParam * All.HubbleParam);
#endif
#ifdef LT_STARBURSTS
  if(All.StarBurstCondition == SB_DENSITY)
    All.SB_Density_Thresh *= PROTONMASS / HYDROGEN_MASSFRAC / All.UnitDensity_in_cgs /
      (All.HubbleParam * All.HubbleParam);
#endif

  /* ----------- */

  SnII_ShortLiv_Yields = (double ***) mymalloc("SnII_ShortLiv_Yields", sizeof(double **) * SFs_dim);

  for(i = 0; i < SFs_dim; i++)
    {
      if(All.PhysDensThresh > 0 && SFs[i].PhysDensThresh[0] == 0)
	SFs[i].PhysDensThresh[0] = All.PhysDensThresh;
      else
	/* Phys Dens Thresh is supposed to be in hydrogen number density cm^-3 */
	SFs[i].PhysDensThresh[0] *= PROTONMASS / HYDROGEN_MASSFRAC / All.UnitDensity_in_cgs /
	  (All.HubbleParam * All.HubbleParam);
      if(All.egyIRA_ThMass > 0)
	SFs[i].egyShortLiv_MassTh = All.egyIRA_ThMass;
      if(All.metIRA_ThMass > 0)
	SFs[i].metShortLiv_MassTh = All.metIRA_ThMass;
      if(All.WindEfficiency > 0)
	SFs[i].WindEfficiency = All.WindEfficiency;
      if(All.WindEnergyFraction > 0)
	SFs[i].WindEnergyFraction = All.WindEnergyFraction;

      setup_SF_related(i);

      /* ................::................... */

      SFs[i].ShortLiv_MassTh = max(SFs[i].metShortLiv_MassTh, SFs[i].egyShortLiv_MassTh);
      SFs[i].ShortLiv_TimeTh = min(SFs[i].metShortLiv_TimeTh, SFs[i].egyShortLiv_TimeTh);

    }

  /* ...........: initialize stepping :...................... */

  ShortLiv_Nsteps = (int *) mymalloc("ShortLiv_Nsteps", SFs_dim * sizeof(int));

  LongLiv_Nsteps = (int *) mymalloc("LongLiv_Nsteps", SFs_dim * sizeof(int));

  Nsteps = (int *) mymalloc("Nsteps", SFs_dim * sizeof(int));
  SNtimesteps = (double ***) mymalloc("SNtimesteps", SFs_dim * sizeof(double *));

  for(i = 0; i < SFs_dim; i++)
    {
      if(UseSnII)
	guess_on_Nsteps = (int) (1.3 / All.SnII_Step_Prec);
      else
	guess_on_Nsteps = 0;


      if(!IMFs[SFs[i].IMFi].timedep)
	ShortLiv_Nsteps[i] = guess_on_Nsteps;
      else
	ShortLiv_Nsteps[i] = 0;

      if(UseSnIa || UseAGB)
	guess_on_Nsteps = (int) (1.3 / All.LLv_Step_Prec);
      else
	guess_on_Nsteps = 0;

      if(!IMFs[SFs[i].IMFi].timedep)
	LongLiv_Nsteps[i] = guess_on_Nsteps;
      else
	LongLiv_Nsteps[i] = 0;


      Nsteps[i] = ShortLiv_Nsteps[i] + LongLiv_Nsteps[i];
      sprintf(buffer, "SNtimesteps_imf_%02d", i);
      SNtimesteps[i] = (double **) mymalloc(buffer, 2 * sizeof(double *));

      sprintf(buffer, "SNtimesteps_timebins_imf_%02d", i);
      SNtimesteps[i][0] = (double *) mymalloc(buffer, Nsteps[i] * sizeof(double));
      memset(SNtimesteps[i][0], 0, Nsteps[i] * sizeof(double));

      sprintf(buffer, "SNtimesteps_timedelta_imf_%02d", i);
      SNtimesteps[i][1] = (double *) mymalloc(buffer, Nsteps[i] * sizeof(double));
      memset(SNtimesteps[i][1], 0, Nsteps[i] * sizeof(double));

      build_SN_Stepping(i);

      /* ...........: write the SNtimestep table :................... */

      if(ThisTask == 0)
	{
	  fprintf(FdSnInit, "\n[time stepping :: %d]\n" "%10s\t%20s\n", i, "time (Gyr)", "timestep (Gyr)");
	  for(j = 0; j < Nsteps[i]; j++)
	    fprintf(FdSnInit, "%8.5lg \t%8.5lg\n", SNtimesteps[i][0][j], SNtimesteps[i][1][j]);
	  fflush(FdSnInit);
	}
    }

  /* ...........: build arrays to interpolate a vs t :................... */

  accel = gsl_interp_accel_alloc();
  spline = gsl_spline_alloc(gsl_interp_cspline, TIME_INTERP_SIZE);

  astep = pow(10, log10(1.0 / 0.001) / TIME_INTERP_SIZE);
  aarray[0] = 0.001;
  tarray[0] = 0;
  cosmic_time = get_age(0.001);

  for(i = 1; i < TIME_INTERP_SIZE; i++)
    {
      aarray[i] = aarray[i - 1] * astep;
      tarray[i] = cosmic_time - get_age(aarray[i]);
    }
  gsl_spline_init(spline, tarray, aarray, TIME_INTERP_SIZE);

  /* */

  if(ThisTask == 0)
    fflush(stdout);

  /* ....... */

#ifdef LT_SEvDbg
  do_spread_dbg_list = (int *) mymalloc("do_spread_dbg_list", NTask * sizeof(int));
#endif

#ifdef LT_SEv_INFO
  Zmass = (double *) mymalloc("Zmass", SFs_dim * (SPECIES * 2) * LT_NMet * 4 * sizeof(double));
  memset(Zmass, 0, SFs_dim * (SPECIES * 2) * LT_NMet * 4 * sizeof(double));
  tot_Zmass = Zmass + SFs_dim * (SPECIES * 2) * LT_NMet;
  SNdata = tot_Zmass + SFs_dim * (SPECIES * 2) * LT_NMet;
  tot_SNdata = SNdata + SFs_dim * (SPECIES * 2) * LT_NMet;
#endif

  sfrrates = (double *) mymalloc("sfrrates", sizeof(double) * SFs_dim);
  totsfrrates = (double *) mymalloc("ttosfrrates", sizeof(double) * SFs_dim);

  sum_sm = (double *) mymalloc("sum_sm", SFs_dim * 4 * sizeof(double));
  memset(sum_sm, 0, SFs_dim * sizeof(double));

  total_sm = sum_sm + SFs_dim;
  sum_mass_stars = total_sm + SFs_dim;
  total_sum_mass_stars = sum_mass_stars + SFs_dim;

  return;
}

int bin_time_interval(int, double, double, double, double, double *, double *);

int build_SN_Stepping(int mySFi)
{
  double f0, time, end_time;

  int IMFi;

  int myNsteps = 0;

  IMFi = SFs[mySFi].IMFi;

  if(UseSnII)
    {
      /* set-up things for SnII */

      /* calculates the total number of expected stars below 8 Solar masses
         in the adopted model this is related to the total number of SnIa
       */
      f0 = IntegrateIMF_byNum(max(IMFs[IMFi].Mm, All.Mup), IMFs[IMFi].MU, &IMFs[IMFi], INC_BH);

      if((f0 > 0)
	 && (max(max(IMFs[IMFi].Mm, All.Mup), IMFs[IMFi].notBH_ranges.inf[IMFs[IMFi].N_notBH_ranges - 1]) <
	     SFs[mySFi].ShortLiv_MassTh))
	{
	  /* set the final time */
	  end_time = min(All.mean_lifetime, IMFs[IMFi].sup_lifetime);

	  /* first time */
	  SNtimesteps[mySFi][0][0] = 0;

	  /* first delta_t */
	  /* note that ShortLiv_TimeTh is set to inf_lifetime in case no threshold is used */
	  SNtimesteps[mySFi][1][0] = SFs[mySFi].ShortLiv_TimeTh;

	  /* start time */
	  time = SFs[mySFi].ShortLiv_TimeTh;

	  ShortLiv_Nsteps[mySFi] = bin_time_interval(IMFi, f0, time, end_time, All.SnII_Step_Prec,
						     SNtimesteps[mySFi][0], SNtimesteps[mySFi][1]);

	  myNsteps = ShortLiv_Nsteps[mySFi];
	}
      else
	ShortLiv_Nsteps[mySFi] = 0;
    }
  else
    {
      ShortLiv_Nsteps[mySFi] = 0;
      SNtimesteps[mySFi][0][0] = 0;
      SNtimesteps[mySFi][1][0] = 0;
      ShortLiv_Nsteps[mySFi] = 0;
    }


  if(UseSnIa || UseAGB)
    {

      /* set-up things for SnIa and AGB stars */

      /* calculates the total number of expected stars below 8 Solar masses
         in the adopted model this is related to the total number of SnIa
       */
      f0 = IntegrateIMF_byNum(All.MBms, All.Mup, &IMFs[IMFi], INC_BH);

      if(f0 > 0)
	{
	  /* set the final time */
	  end_time = IMFs[IMFi].sup_lifetime;

	  /* first time */
	  SNtimesteps[mySFi][0][ShortLiv_Nsteps[mySFi]] =
	    SNtimesteps[mySFi][0][ShortLiv_Nsteps[mySFi] - 1] + SNtimesteps[mySFi][1][ShortLiv_Nsteps[mySFi] -
										      1];

	  /* first delta_t */
	  SNtimesteps[mySFi][1][ShortLiv_Nsteps[mySFi]] =
	    All.mean_lifetime - SNtimesteps[mySFi][0][ShortLiv_Nsteps[mySFi]];

	  /* start time */
	  time = All.mean_lifetime;

	  LongLiv_Nsteps[mySFi] = bin_time_interval(IMFi, f0, time, end_time, All.LLv_Step_Prec,
						    &SNtimesteps[mySFi][0][ShortLiv_Nsteps[mySFi] - 1],
						    &SNtimesteps[mySFi][1][ShortLiv_Nsteps[mySFi] - 1]);
	  myNsteps += LongLiv_Nsteps[mySFi];
	  SNtimesteps[mySFi][0][myNsteps] =
	    SNtimesteps[mySFi][0][myNsteps - 1] + SNtimesteps[mySFi][1][myNsteps - 1];

	}
      else
	LongLiv_Nsteps[mySFi] = 0;
    }
  else
    LongLiv_Nsteps[mySFi] = 0;

  Nsteps[mySFi] = myNsteps;
  SNtimesteps[mySFi][0][Nsteps[mySFi] + 1] = 100.0;	/* last forever */


  return Nsteps[mySFi];
}

int bin_time_interval(int IMFi, double f0, double time, double end_time, double Step_Prec, double *times,
		      double *delta_times)
{
  double m, delta_time;

  double Left, Right, timed_frac;

  double next_notBH_sup, next_notBH_inf;

  int BHi, i, limit;



  for(m = dying_mass(time), BHi = 0;
      (BHi < IMFs[IMFi].N_notBH_ranges) && (m <= IMFs[IMFi].notBH_ranges.inf[BHi]); BHi++)
    ;
  next_notBH_sup = lifetime(IMFs[IMFi].notBH_ranges.inf[BHi]);
  next_notBH_inf = lifetime(IMFs[IMFi].notBH_ranges.sup[BHi]);

  i = 1;
  /* initial guess for delta_time */
  modf(log10(time) + 9, &delta_time);
  delta_time = pow(10, delta_time) / 1e9;

  /* cycle to calculate time steps */
  while(time < end_time)
    {
      timed_frac = 1;
      Left = Right = limit = 0;
      while((timed_frac / f0 < Step_Prec * 0.9 || timed_frac / f0 > Step_Prec * 1.1) && !limit)
	{
	  if(time + delta_time > end_time)
	    delta_time = end_time - time;
	  if(time + delta_time > next_notBH_sup)
	    delta_time = next_notBH_sup - time;
	  timed_frac =
	    IntegrateIMF_byNum(dying_mass(time + delta_time), dying_mass(time), &IMFs[IMFi], INC_BH);

	  if(timed_frac / f0 < Step_Prec * 0.9)
	    {
	      if(!(limit = (time + delta_time == end_time)))
		{
		  Left = max(Left, delta_time);
		  if(Right == 0)
		    delta_time *= 2;
		  else
		    delta_time = (Left + Right) / 2;
		}
	    }
	  if(timed_frac / f0 > Step_Prec * 1.1)
	    {
	      if(Right == 0)
		Right = delta_time;
	      else
		Right = min(Right, delta_time);
	      if(Left == 0)
		delta_time /= 2;
	      else
		delta_time = (Left + Right) / 2;
	    }
	}
      if(timed_frac / f0 < Step_Prec / 2 && limit)
	delta_times[i - 1] += delta_time;
      else
	{
	  times[i] = time;
	  delta_times[i] = delta_time;
	  i++;
	}
      if((time += delta_time) == next_notBH_sup)
	{
	  if(BHi < IMFs[IMFi].N_notBH_ranges)
	    {
	      BHi++;
	      delta_time = lifetime(IMFs[IMFi].notBH_ranges.sup[BHi]) - next_notBH_sup;
	      next_notBH_sup = lifetime(IMFs[IMFi].notBH_ranges.inf[BHi]);
	      next_notBH_inf = lifetime(IMFs[IMFi].notBH_ranges.sup[BHi]);
	    }
	  else
	    {
	      limit = 1;
	      time = end_time;
	    }
	}

    }
  return i;
}


void get_Egy_and_Beta(double ThMass, double *Egy, double *Beta, SF_Type * SFp)
{
  double NumFrac_inIRA;

  IMF_Type *myIMFp;

  myIMFp = &IMFs[SFp->IMFi];

  *Beta = calculate_FactorSN(ThMass, myIMFp->MU, myIMFp);
  NumFrac_inIRA = IntegrateIMF_byMass(ThMass, myIMFp->MU, myIMFp, EXC_BH);
  *Egy = IntegrateIMF_byEgy(SFp->egyShortLiv_MassTh, IMFp->MU, myIMFp) / SOLAR_MASS;
  /**Egy = (All.SnIIEgy * All.UnitEnergy_in_cgs) * NumFrac_inIRA / SOLAR_MASS;*/
  *Egy *= All.UnitMass_in_g;
  *Egy *= (1 - *Beta) / *Beta;

  return;
}



void calculate_ShortLiving_related(SF_Type * SFp, int loud)
{
  double num_snII;

  int set;

  IMFp = (IMF_Type *) & IMFs[SFp->IMFi];

  if(ThisTask == 0 && loud)
    printf(":: energy IRA is active in the range [%9.7g - %9.7g]Msun <> [%9.7g - %9.7g]Gyr\n",
	   IMFp->MU, SFp->egyShortLiv_MassTh, IMFp->inf_lifetime, SFp->egyShortLiv_TimeTh);

  /* mass fraction involved in SnII */
  SFp->MassFrac_inIRA = IntegrateIMF_byMass(SFp->egyShortLiv_MassTh, IMFp->MU, IMFp, INC_BH);
  /* expected number of snII per initial stellar population mass in units of solar masses IN IRA RANGE */
  SFp->NumFrac_inIRA = IntegrateIMF_byNum(SFp->egyShortLiv_MassTh, IMFp->MU, IMFp, EXC_BH);
  /* expected number of stars per initial stellar population mass in units of solar masses */
  num_snII = IntegrateIMF_byNum(max(All.Mup, IMFp->Mm), IMFp->MU, IMFp, EXC_BH);
  /* erg/g per solar masses of initial stellar population mass, due to IRA snII */
  /*  ~7.4x10^48 erg/(s * Msun) for all Sn (8-100Msun) for a Salp. IMF */
  SFp->IRA_erg_per_g =
    IntegrateIMF_byEgy(SFp->egyShortLiv_MassTh, IMFp->MU, IMFp) * All.UnitEnergy_in_cgs / SOLAR_MASS;
  /*  IMFp->IRA_erg_per_g = (All.SnIIEgy * All.UnitEnergy_in_cgs) * IMFp->NumFrac_inIRA / SOLAR_MASS; */
  /*  erg/g per solar masses of initial stellar population mass, due to ALL IRA snII */
  SFp->TOT_erg_per_g =
    IntegrateIMF_byEgy(max(All.Mup, IMFp->Mm), IMFp->MU, IMFp) * All.UnitEnergy_in_cgs / SOLAR_MASS;
  /*  IMFp->TOT_erg_per_g = (All.SnIIEgy * All.UnitEnergy_in_cgs) * num_snII / SOLAR_MASS; */

  set = IMFp->YSet;

  SD.Zbin = 0;
  SD.MArray = IIMbins[set];
  SD.Mdim = IIMbins_dim[set];
  SD.Y = SnIIEj[set];

  SFp->FactorSN = calculate_FactorSN(SFp->egyShortLiv_MassTh, IMFp->MU, IMFp);	/* restored by Short-Living stars */
  SFp->totFactorSN = calculate_FactorSN(All.Mup, IMFp->MU, IMFp);

  if(UseAGB)
    {
      SD.Zbin = 0;
      SD.MArray = AGBMbins[set];
      SD.Mdim = AGBMbins_dim[set];
      SD.Y = AGBEj[set];
      SFp->totResFrac = SFp->totFactorSN + calculate_FactorSN(IMFp->Mm, All.Mup, IMFp);
    }
  else
    SFp->totResFrac = SFp->totFactorSN;

  if(ThisTask == 0 && loud)
    {
      printf("    %.3lg%% of mass in egy Short-Living tail\n"
	     "    energy per g due to Short-Living snII is %9.7g erg/g\n",
	     SFp->MassFrac_inIRA * 100, SFp->IRA_erg_per_g);

      printf("   restored fraction from short-living stars.. (beta parameter) ");
      printf(" : %.8g\n", SFp->FactorSN);

      fprintf(FdSnInit,
	      "\n[ Energy from Short-Living Stars]\n"
	      "range is %6.3g - %6.3g Msun [%6.3g - %6.3g Gyr]\n"
	      "Mass fraction in tail is         : %.3lg%%\n"
	      "Number fraction in tail is       : %.3lg%%\n"
	      "energy per g due to S-L  snII is : %.8g erg/g (%.3lg%% of the total budget)\n"
	      "restored mass by Short-Liv stars (`beta' used in SF computation) is    :",
	      IMFp->MU, SFp->egyShortLiv_MassTh, IMFp->inf_lifetime, SFp->egyShortLiv_TimeTh,
	      SFp->MassFrac_inIRA * 100,
	      SFp->NumFrac_inIRA / num_snII * 100,
	      SFp->IRA_erg_per_g, SFp->IRA_erg_per_g / SFp->TOT_erg_per_g * 100);
      fprintf(FdSnInit, " %.8g", SFp->FactorSN);
      fprintf(FdSnInit, "\n");
      fflush(FdSnInit);
    }

  /* convert to code units */
  SFp->IRA_erg_per_g *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;
  SFp->TOT_erg_per_g *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;

  SFp->EgySpecSN = SFp->IRA_erg_per_g * (1 - SFp->FactorSN) / SFp->FactorSN;
}





void setup_SF_related(int mySFi)
 /*
  *             |         RHOth indep. of Z               |       RHOth dep. on Z
  *  ==========================================================================================================
  *             |                                         |                                                  |
  *             | A calc. RHOth using Mth @ a given       | F calculate RHOth using the given Mth for each Z |
  *             |   Z using metal cooling                 |                                                  |
  *             |                                         |                                                  |
  *  RHOth from |   ................................      |                                                  |
  *  parameters |                                         |                                                  |
  *             | B calc. RHOth using Mth not using       |                                                  |
  *             |   the metal cooling                     |                                                  |
  *  ==========================================================================================================
  *             |                                         |                                                  |
  *             | C calc. Mth @ a given metallicity       | G calculate Mth @                                |
  *             |   keep Mth for all Z                    |   a given Z                                      |
  *  RHOth set  |   ...............................       |         \                                        |
  *  to a given | D calc. Mth not using the metal cool    |           --> keep Mth for all Z --> calc. RHOth |
  *  value      | --------------------------------------  |         /                                        |
  *             |                                         | H set Mth to a  given value                      |
  *             | E set Mth to a given value for all Z    |                                                  |
  * ===========================================================================================================
  *
  */
{
  int IMFi, j, k, filecount, set;

  double num_snII;

  double myPhysDensTh, myFEVP, m;

  double left, right;

  double feedbackenergyinergs;

  double *dummies, dummy, frac;

  int PATH;

  int exist_eff_model_file;

  char buffer[200];

  IMFi = SFs[mySFi].IMFi;
  IMFp = &IMFs[IMFi];

  if(SFs[mySFi].MaxSfrTimescale_rescale_by_densityth > 1)
    SFs[mySFi].MaxSfrTimescale /= 1.2 * sqrt(SFs[mySFi].MaxSfrTimescale_rescale_by_densityth);
  if(SFs[mySFi].MaxSfrTimescale_rescale_by_densityth < 1
     && SFs[mySFi].MaxSfrTimescale_rescale_by_densityth > 0)
    SFs[mySFi].MaxSfrTimescale *= 0.8 * sqrt(SFs[mySFi].MaxSfrTimescale_rescale_by_densityth);


  if(SFs[mySFi].egyShortLiv_MassTh > 0)
    /*
     *  Mth has been specified, check that it stays
     *  within boundaries.
     *  then translate it in a time threshold and calculate
     *  beta and egyspecSN.
     */
    {
      /* checks that the defined mass th does not fall in a BH mass range */
      j = not_in_BHrange(IMFi, &SFs[mySFi].egyShortLiv_MassTh);
      if(j == 0)
	{
	  if(ThisTask == 0)
	    {
	      fprintf(FdWarn, "incorrect IRA limit set by your parameters: out of range\n"
		      "  your inf IRA mass is %9.7g vs [%9.7g : %9.7g] Msun limits for the IMF\n"
		      "  better to stop here\n", SFs[mySFi].egyShortLiv_MassTh, IMFs[IMFi].Mm, IMFs[IMFi].MU);
	      fflush(FdWarn);
	      printf("incorrect IRA limit set by your parameters: out of range\n"
		     "  your inf IRA mass is %9.7g vs [%9.7g : %9.7g] Msun limits for the IMF\n"
		     "  better to stop here\n", SFs[mySFi].egyShortLiv_MassTh, IMFs[IMFi].Mm, IMFs[IMFi].MU);
	      fflush(stdout);
	    }
	  endrun(19001);
	}
      else if(j < 0)
	{
	  j *= -1;

	  if(IMFs[IMFi].N_notBH_ranges > j)
	    SFs[mySFi].metShortLiv_MassTh = IMFs[IMFi].notBH_ranges.inf[j];
	  else
	    SFs[mySFi].metShortLiv_MassTh = IMFs[IMFi].notBH_ranges.sup[j - 1];

	  if(ThisTask == 0)
	    {
	      fprintf(FdWarn, "The Short-Living mass limit has been set to %g in order to match"
		      " the non-BH range %d\n", SFs[mySFi].egyShortLiv_MassTh, j);
	      fflush(FdWarn);
	      printf("The Short-Living mass limit has been set to %g in order to match"
		     " the non-BH range %d\n", SFs[mySFi].egyShortLiv_MassTh, j);
	      fflush(stdout);
	    }
	}
      /* set the time th accordingly */
      SFs[mySFi].egyShortLiv_TimeTh = lifetime(SFs[mySFi].egyShortLiv_MassTh);

      calculate_ShortLiving_related((SF_Type *) & SFs[mySFi], 1);
    }

  if(SFs[mySFi].SFTh_Zdep == 1)
    /*
     * if metallicity dependence is on, check whether the
     * thresholds' file does exist
     */
    {
#ifdef LT_METAL_COOLING
      if(ThisTask == 0)
	{
	  exist_eff_model_file = 0;
	  if(RestartFlag == 2)
	    {
	      filecount = atoi(All.InitCondFile + strlen(All.InitCondFile) - 3);
	      exist_eff_model_file = read_eff_model(filecount, mySFi);
	    }
	  if(exist_eff_model_file == 0)
	    /* if restartflag is 2 but the eff model file for the snapshot is not
	     * present (j = 0), the code try again searching for the initial file */
	    exist_eff_model_file = read_eff_model(-1, mySFi);

	  if(exist_eff_model_file == 0)
	    {
	      reading_thresholds_for_thermal_instability();
	      for(j = 0; j < ZBins; j++)
		{
		  ThInst_onset[j] = pow(10, ThInst_onset[j]);
		  SFs[mySFi].FEVP[j] = All.TempSupernova / ThInst_onset[j];
		}
	    }
	}

      /* in case that the threshold file does not exist, the array will however be set to zero */
      MPI_Bcast((void *) &exist_eff_model_file, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast((void *) &SFs[mySFi].PhysDensThresh[0], ZBins * sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD);
      MPI_Bcast((void *) &SFs[mySFi].FEVP[0], ZBins * sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD);
      MPI_Bcast((void *) ThInst_onset, ZBins * sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
    }
  else
    {
      ThInst_onset[0] = 1e5;
      //      SFs[mySFi].FEVP = &All.FactorEVP;
      SFs[mySFi].FEVP[0] = All.TempSupernova / ThInst_onset[0];
    }


  PATH = (SFs[mySFi].SFTh_Zdep == 1);

  PATH |= ((SFs[mySFi].egyShortLiv_MassTh > 0) << 1);

  PATH |= ((SFs[mySFi].PhysDensThresh[0] > 0) << 2);

  if(PATH < 2)
    {
      printf
	("neither the Mass Threshold for the energy release nor the Density Threshold for the Star Formation have been specified..\n"
	 "I do feel difficult to continue. Better to think about a bit more..\n");
      endrun(LT_ERR_NOT_ENOUGH_PARAM_FOR_SF);
    }

  if(PATH < 4)
    /*
     * egyShortLiv_MassTh is specified
     * PhysDensThresh is not specified
     */
    {
      if(SFs[mySFi].SFTh_Zdep == 1)	/* PATH = 3 */
	{
	  /*
	   * metal dependence is on but the thresholds' file does not
	   * exist.
	   */
	  MPI_Barrier(MPI_COMM_WORLD);
	  init_clouds_cm(2, SFs[mySFi].PhysDensThresh, SFs[mySFi].FEVP,
			 SFs[mySFi].EgySpecSN, SFs[mySFi].FactorSN, SFs[mySFi].MaxSfrTimescale,
			 ZBins, &CoolZvalue[0]);
	}
      else			/* PATH = 2 */
	{
	  init_clouds(2, SFs[mySFi].EgySpecSN, SFs[mySFi].FactorSN, SFs[mySFi].MaxSfrTimescale,
		      SFs[mySFi].referenceZ_toset_SF_DensTh, SFs[mySFi].PhysDensThresh, SFs[mySFi].FEVP);
	}

      PATH = 8;
    }

  if(PATH < 6)
    /*
     * egyShortLiv_MassTh is not specified
     * PhysDensThresh is specified
     */
    {
      /* initial guess for Mth */
      SFs[mySFi].egyShortLiv_MassTh = All.Mup * 1.1;

      set = IMFs[IMFi].YSet;
      SD.Zbin = 0;
      SD.MArray = IIMbins[set];
      SD.Mdim = IIMbins_dim[set];
      SD.Y = SnIIEj[set];

      get_Egy_and_Beta(SFs[mySFi].egyShortLiv_TimeTh, &SFs[mySFi].EgySpecSN, &SFs[mySFi].FactorSN,
		       &SFs[mySFi]);
      init_clouds(2, SFs[mySFi].EgySpecSN, SFs[mySFi].FactorSN, SFs[mySFi].MaxSfrTimescale,
		  SFs[mySFi].referenceZ_toset_SF_DensTh, &myPhysDensTh, &myFEVP);

      while((fabs(myPhysDensTh - SFs[mySFi].PhysDensThresh[0]) / SFs[mySFi].PhysDensThresh[0] > 1e-2) &&
	    SFs[mySFi].egyShortLiv_MassTh > All.Mup &&
	    SFs[IMFi].egyShortLiv_MassTh < IMFs[IMFi].notBH_ranges.sup[0])
	/* note: because of using get_Egy_an_Beta, this calculations automatically
	 * exclude BH ranges
	 */
	{
	  if((myPhysDensTh - SFs[mySFi].PhysDensThresh[0]) < 0)
	    {
	      left = SFs[mySFi].egyShortLiv_TimeTh;
	      if(right == 0)
		SFs[mySFi].egyShortLiv_MassTh *= 0.8;
	      else
		SFs[mySFi].egyShortLiv_MassTh = (SFs[mySFi].egyShortLiv_MassTh + right) / 2;
	    }
	  else
	    {
	      right = SFs[mySFi].egyShortLiv_MassTh;
	      if(left == 0)
		SFs[mySFi].egyShortLiv_MassTh *= 1.2;
	      else
		SFs[mySFi].egyShortLiv_MassTh = (SFs[mySFi].egyShortLiv_MassTh + left) / 2;
	    }

	  if(SFs[mySFi].egyShortLiv_MassTh < All.Mup)
	    SFs[mySFi].egyShortLiv_MassTh = All.Mup;
	  else if(SFs[mySFi].egyShortLiv_MassTh > IMFs[IMFi].notBH_ranges.sup[0])
	    /* this prevent to fall in the last BH range: if it extend up to MU,
	     * we would obtain zero energy. If the energy is still not sufficient,
	     * the cycle will interrupt anyway due to the 3rd head condition */
	    SFs[mySFi].egyShortLiv_MassTh = IMFs[IMFi].notBH_ranges.sup[0];

	  get_Egy_and_Beta(SFs[mySFi].egyShortLiv_MassTh, &SFs[mySFi].EgySpecSN, &SFs[mySFi].FactorSN,
			   &SFs[mySFi]);

	  init_clouds(2, SFs[mySFi].EgySpecSN, SFs[mySFi].FactorSN, SFs[mySFi].MaxSfrTimescale,
		      SFs[mySFi].referenceZ_toset_SF_DensTh, &myPhysDensTh, &myFEVP);
	}

      /* checks that found mass does not fall in a BH mass range */
      for(j = 0;
	  (j < IMFs[IMFi].N_notBH_ranges) &&
	  (SFs[mySFi].egyShortLiv_MassTh <= IMFs[IMFi].notBH_ranges.inf[j]); j++)
	;
      if(j < IMFs[IMFi].N_notBH_ranges)
	if(SFs[mySFi].egyShortLiv_MassTh > IMFs[IMFi].notBH_ranges.sup[j])
	  /* this case can arise only when at least 1 BH interval lives
	   * between 2 nom-BH intervals: then, j shold be > 0.
	   */
	  SFs[mySFi].egyShortLiv_MassTh = IMFs[IMFi].notBH_ranges.sup[j - 1];


      if((myPhysDensTh - SFs[mySFi].PhysDensThresh[0]) / SFs[mySFi].PhysDensThresh[0] > 1e-2 && ThisTask == 0)
	{
	  if(SFs[mySFi].egyShortLiv_MassTh == All.Mup)
	    {
	      fprintf(FdWarn, "  . warning: the needed Mass Threshold for a star to be\n"
		      "             short--living would be lower that %5.3f Msun.\n"
		      "             we set it to %5.3f Msun; pls check sfrrate.txt\n", All.Mup, All.Mup);
	      fflush(stdout);
	    }
	  else if(SFs[mySFi].egyShortLiv_MassTh == IMFs[IMFi].MU)
	    {
	      fprintf(FdWarn, "  . warning: the needed Mass Threshold for a star to be\n"
		      "             short--living would be larger that %5.3f Msun.\n"
		      "             we set it to %5.3f Msun; pls check sfrrate.txt\n",
		      IMFs[IMFi].MU, IMFs[IMFi].MU);
	      fflush(stdout);
	    }
	  else
	    {
	      fprintf(FdWarn, "  . warning: Mass Threshold for a star to be short-living is\n"
		      "             %5.3f Msun; this would set a physical density threshold for\n"
		      "             the star formation different than that you set in paramfile.\n"
		      "             pls, check sfrrate.txt\n", SFs[mySFi].egyShortLiv_MassTh);
	      fflush(stdout);
	    }
	}

      calculate_ShortLiving_related(&SFs[mySFi], 1);

      PATH |= 2;
    }


  if(PATH == 7)
    /* if PATH == 6 nothing is left to be done */
    /*
     * egyShortLiv_MassTh is specified
     * PhysDensThresh is specified
     */
    {

      if(SFs[mySFi].SFTh_Zdep == 1 &&	/* PATH = 7 */
	 exist_eff_model_file == 0)	/* assume that the variable has been set */
	{
	  dummies = (double *) mymalloc("dummies", ZBins * sizeof(double));
	  dummy = SFs[mySFi].PhysDensThresh[0];

	  /* calculate expected thresholds for all metallicities, given the mass EgyMassTh */
	  MPI_Barrier(MPI_COMM_WORLD);
	  init_clouds_cm(2, dummies, SFs[mySFi].FEVP,
			 SFs[mySFi].EgySpecSN, SFs[mySFi].FactorSN, SFs[mySFi].MaxSfrTimescale,
			 ZBins, &CoolZvalue[0]);

	  /* now apply relative variations due to metallicities to the specified rho_threshold */

	  if(ZBins > 1)
	    {
	      for(j = ZBins - 1; (j > 0) && (SFs[mySFi].referenceZ_toset_SF_DensTh < CoolZvalue[j]); j--)
		;

	      if(j == ZBins - 1)
		j--;

	      if(SFs[mySFi].referenceZ_toset_SF_DensTh > CoolZvalue[j] * (1 + 0.01) ||
		 SFs[mySFi].referenceZ_toset_SF_DensTh < CoolZvalue[j] * (1 - 0.01))
		frac = (SFs[mySFi].referenceZ_toset_SF_DensTh - CoolZvalue[j]) *
		  (dummies[j + 1] - dummies[j]) / (CoolZvalue[j + 1] - CoolZvalue[j]);
	      else
		frac = dummy / dummies[j];

	      for(j = 0; j < ZBins; j++)
		{
		  if(SFs[mySFi].referenceZ_toset_SF_DensTh > CoolZvalue[j] * (1 + 0.01) ||
		     SFs[mySFi].referenceZ_toset_SF_DensTh < CoolZvalue[j] * (1 - 0.01))
		    SFs[mySFi].PhysDensThresh[j] = dummies[j] * frac;
		  else
		    SFs[mySFi].PhysDensThresh[j] = dummy;
		}

	    }
	  myfree(dummies);
	}
    }

  if(ThisTask == 0)
    printf("determining Kennicutt law%c..\n", (SFs[mySFi].SFTh_Zdep ? 's' : ' '));
  fflush(stdout);


  if(SFs[mySFi].SFTh_Zdep == 0)
    {
#ifndef LT_WIND_VELOCITY
#ifdef LT_ALL_SNII_EGY_INWINDS
      feedbackenergyinergs =
	SFs[mySFi].TOT_erg_per_g / All.UnitMass_in_g * (All.UnitEnergy_in_cgs * SOLAR_MASS);
      SFs[mySFi].WindEnergy =
	sqrt(2 * SFs[mySFi].WindEnergyFraction * SFs[mySFi].TOT_erg_per_g / SFs[mySFi].WindEfficiency);
#else
      feedbackenergyinergs =
	SFs[mySFi].IRA_erg_per_g / All.UnitMass_in_g * (All.UnitEnergy_in_cgs * SOLAR_MASS);
      SFs[mySFi].WindEnergy =
	sqrt(2 * SFs[mySFi].WindEnergyFraction * SFs[mySFi].IRA_erg_per_g / SFs[mySFi].WindEfficiency);
#endif
#else
      feedbackenergyinergs = All.FeedbackEnergy / All.UnitMass_in_g * (All.UnitEnergy_in_cgs * SOLAR_MASS);
      SFs[mySFi].WindEnergy = LT_WIND_VELOCITY;
#ifdef LT_ALL_SNII_EGY_INWINDS
      SFs[mySFi].WindEnergyFraction =
	SFs[mySFi].WindEnergy * SFs[mySFi].WindEnergy * SFs[mySFi].WindEfficiency / (2 *
										     SFs
										     [mySFi].TOT_erg_per_g);
#else
      SFs[mySFi].WindEnergyFraction =
	SFs[mySFi].WindEnergy * SFs[mySFi].WindEnergy * SFs[mySFi].WindEfficiency / (2 *
										     SFs
										     [mySFi].IRA_erg_per_g);
#endif
#endif

      //SFs[mySFi].FEVP = &All.FactorEVP;
      if(ThisTask == 0)
	{
	  sfrrate_filenum = mySFi;
	  init_clouds(0, SFs[mySFi].EgySpecSN, SFs[mySFi].FactorSN, SFs[mySFi].MaxSfrTimescale,
		      SFs[mySFi].referenceZ_toset_SF_DensTh, SFs[mySFi].PhysDensThresh, SFs[mySFi].FEVP);

	  printf("Energy Fraction in Winds= %g\n", SFs[mySFi].WindEnergyFraction);
	  printf("Winds Mass load efficiency= %g\n", SFs[mySFi].WindEfficiency);
	  printf("PhysDensThresh= %g (internal units)\n", SFs[mySFi].PhysDensThresh[0]);

	  fprintf(FdSnInit, "\n[Winds]\nEnergy Fraction in Winds= %g \n", SFs[mySFi].WindEnergyFraction);
	  fprintf(FdSnInit, "Winds Mass load efficiency= %g\n", SFs[mySFi].WindEfficiency);
	  fprintf(FdSnInit, "\n[SF Rho Threshold]\nPhysDensThresh= %g (internal units)\n",
		  SFs[mySFi].PhysDensThresh[0]);
	}
    }
  else
    {
      init_clouds_cm(0, SFs[mySFi].PhysDensThresh, SFs[mySFi].FEVP,
		     SFs[mySFi].EgySpecSN, SFs[mySFi].FactorSN, SFs[mySFi].MaxSfrTimescale,
		     ZBins, &CoolZvalue[0]);
      if(ThisTask == 0)
	write_eff_model(-1, mySFi);
    }

  MPI_Barrier(MPI_COMM_WORLD);


  if(ThisTask == 0)
    {
      printf("Feedback energy per formed solar mass in stars= %g  ergs\n"
	     "Wind Velocity= %g Km/s\n", feedbackenergyinergs, SFs[mySFi].WindEnergy);

      fprintf(FdSnInit, "Feedback energy per formed solar mass in stars= %g  ergs\n"
	      "Wind Velocity= %g Km/s\n", feedbackenergyinergs, SFs[mySFi].WindEnergy);
    }

  m = max(IMFs[IMFi].Mm, All.Mup);

  if(SFs[mySFi].metShortLiv_MassTh <= m)
    SFs[mySFi].metShortLiv_MassTh = m;
  else if(SFs[mySFi].metShortLiv_MassTh > IMFs[IMFi].MU)
    SFs[mySFi].metShortLiv_MassTh = IMFs[IMFi].MU;

  /* checks that the defined mass th does not fall in a BH mass range */
  for(j = 0;
      (j < IMFs[IMFi].N_notBH_ranges) &&
      (SFs[mySFi].metShortLiv_MassTh <= IMFs[IMFi].notBH_ranges.inf[j]); j++)
    ;

  if(j < IMFs[IMFi].N_notBH_ranges)
    if(SFs[mySFi].metShortLiv_MassTh > IMFs[IMFi].notBH_ranges.sup[j])
      {
	if(j == 0)
	  /* if the mass th falls ini the upper BH range, and that BH range reaches
	   *  MU, then it's impossible to correctly adjust the mass th
	   */
	  {
	    if(ThisTask == 0)
	      {
		printf("The Short-Living mass limit for metal release that you defined is not\n"
		       "compatible with the BH ranges that you defined. Pls check all that in\n"
		       "param file and in IMF file\n");
		endrun(LT_ERR_MISMATCH_ShL_TH_BHranges);
	      }
	  }
	else
	  {
	    SFs[mySFi].metShortLiv_MassTh = IMFs[IMFi].notBH_ranges.inf[j - 1];
	    fprintf(FdWarn,
		    "The Short-Living mass limit for metal release has been set to %g in order to match"
		    " with the non-BH range %d\n", SFs[mySFi].egyShortLiv_MassTh, j);
	  }
      }

  SFs[mySFi].metShortLiv_TimeTh = lifetime(SFs[mySFi].metShortLiv_MassTh);

  if(ThisTask == 0)
    {
      if(SFs[mySFi].metShortLiv_MassTh < IMFs[IMFi].MU)
	printf(":: metal IRA is active in the range [%9.7g - %9.7g]Msun <> [%9.7g - %9.7g]Gyr\n",
	       IMFs[IMFi].MU, SFs[mySFi].metShortLiv_MassTh, IMFs[IMFi].inf_lifetime,
	       SFs[mySFi].metShortLiv_TimeTh);
      else
	printf(":: metal IRA is not active\n");
    }

  if(SFs[mySFi].metShortLiv_MassTh < IMFs[IMFi].MU)
    {
      /* mass fraction involved in SnII */
      SFs[mySFi].MassFrac_inIRA =
	IntegrateIMF_byMass(SFs[mySFi].metShortLiv_MassTh, IMFs[IMFi].MU, &IMFs[IMFi], INC_BH);
      /* expected number of SnII per initial stellar population mass in units of solar masses IN IRA RANGE */
      SFs[mySFi].NumFrac_inIRA =
	IntegrateIMF_byNum(SFs[mySFi].metShortLiv_MassTh, IMFs[IMFi].MU, &IMFs[IMFi], EXC_BH);
      /* expected number of SnII per initial stellar population mass in units of solar masses */
      num_snII = IntegrateIMF_byNum(max(All.Mup, IMFs[IMFi].Mm), IMFs[IMFi].MU, &IMFs[IMFi], EXC_BH);

      if(ThisTask == 0)
	printf("    %.3lg%% of mass in metal IRA tail\n", SFs[mySFi].MassFrac_inIRA * 100);

      /* ...........: calculates the restored fraction :................... */
      if(ThisTask == 0)
	printf("   metal IRA, calculating restored fraction..\n");



/* ...........: calculates the effective yields :................... */

      if(ThisTask == 0)
	printf(":: calculating effective yields for Short-Living Stars..\n");


      SD.Yset = IMFs[IMFi].YSet;

      sprintf(buffer, "SnII_ShortLiv_Yields_%2d", mySFi);
      SnII_ShortLiv_Yields[mySFi] = (double **) mymalloc(buffer, LT_NMet * sizeof(double *));
      for(k = 0; k < LT_NMet; k++)
	{
	  sprintf(buffer, "SnII_ShortLiv_Yields_%2d_%d", mySFi, k);
	  SnII_ShortLiv_Yields[mySFi][k] = (double *) mymalloc(buffer, IIZbins_dim[SD.Yset] * sizeof(double));
	}

/* #if defined (UM_CHEMISTRY) && defined (UM_METAL_COOLING) */
/*       IIShLv_AvgFillNDens[IMFi] = (double*)calloc(IIZbins_dim[SD.Yset], sizeof(double)); */
/* #endif */

      SFs[mySFi].nonZeroIRA = calculate_effective_yields(SFs[mySFi].metShortLiv_MassTh, IMFs[IMFi].MU, IMFi);

      for(j = 0; j < IIZbins_dim[SD.Yset]; j++)
	{
	  if(j == 0)
	    {
	      SD.Zbin = j;
	      SD.MArray = IIMbins[SD.Yset];
	      SD.Mdim = IIMbins_dim[SD.Yset];
	      SD.Y = SnIIYields[SD.Yset][FillEl];
	      SFs[mySFi].metFactorSN = calculate_FactorSN(SFs[mySFi].metShortLiv_MassTh, IMFs[IMFi].MU, &IMFs[IMFi]);	/* restored by IRA SnII */
	    }

	  if(ThisTask == 0)
	    {
	      fprintf(FdSnInit, "\n[Effective Yields for Short-Living stars :: set %03d - Z %8.6g]\n",
		      SD.Yset, (IIZbins_dim[SD.Yset] > 1) ? IIZbins[SD.Yset][j] : 0);
	      fprintf(FdSnInit, " %3s  %6s   %10s %10s\n", "IMF", "Z", "name", "Yield");
	      printf(" %3s  %6s   %10s %10s\n", "IMF", "Z", "name", "Yield");
	      for(k = 0; k < LT_NMet; k++)
		{
		  printf(" [%3d][%8.6g]   %10s %10.7lg\n",
			 IMFi, (IIZbins_dim[SD.Yset] > 1) ? IIZbins[SD.Yset][j] : 0,
			 MetNames[k], SnII_ShortLiv_Yields[mySFi][k][j]);
		  fprintf(FdSnInit, " [%3d][%8.6g] %10s %10.7lg\n",
			  SD.Yset, (IIZbins_dim[SD.Yset] > 1) ? IIZbins[SD.Yset][j] : 0,
			  MetNames[k], SnII_ShortLiv_Yields[mySFi][k][j]);
		}
	    }
	}
    }
  else
    {
      SFs[mySFi].MassFrac_inIRA = 0;
      SFs[mySFi].NumFrac_inIRA = 0;
      SFs[mySFi].nonZeroIRA = 0;
      num_snII = 0;
      SnII_ShortLiv_Yields[mySFi] = 0x0;
      SFs[mySFi].metFactorSN = 0;
      if(ThisTask == 0)
	{
	  printf(":: No metals are supposed to be promptly ejected..\n");
	  fprintf(FdSnInit, "\n[Effective Yields for IRA part]\n"
		  "No metals are supposed to be promptly ejected..\n");
	}
    }

  if(ThisTask == 0)
    printf("\n\n");


}
