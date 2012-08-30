
double get_SnIa_product(int i, int Yset, double *metals, double *energy, double prec_evolve_time,
			double delta_time)
{
  int I, Zbin, Mbin;

  double Zstar, ejecta, result, abserr;

  double next_evolve_time;

  int j;

  /* does not care about the size of the integration time; you must care about that elsewhere */

  I = P[i].MetID;

  next_evolve_time = prec_evolve_time + delta_time;

  F.function = &nRSnIa;
  F.params = NULL;


  if((gsl_status =
      gsl_integration_qag(&F, prec_evolve_time, next_evolve_time, 1, 1e-5, 1000, qag_INT_KEY, w, &result,
			  &abserr)))
    {
      printf("  >> Task %i, gsl integration error %i in Sn Ia integration [%.6g - %.6g]: %.6g %.6g\n",
	     ThisTask, gsl_status, prec_evolve_time, next_evolve_time, result, abserr);
      fflush(stdout);
      endrun(LT_ERROR_INTEGRATION_ERROR);
    }

  if(result < 0 && result > -Z_ROUND_OFF_ERROR)
    {
      printf(" > probable round-off error in integrating Ia num [%d][%d][%u] %g\n",
	     ThisTask, i, P[i].ID, result);
      fflush(stdout);
      result = 0;
    }

  /* calculate number of SnIa explosions: MetP[I].iMass * UnitMassFact
   * is the initial mass population in unit of Msun
   */

  result *= MetP[I].iMass;	/* result is in code units */

  if(result == 0)
    {
      for(j = 0; j < LT_NMet; j++)
	metals[j] = 0;
/* #if defined (UM_CHEMISTRY) && defined (UM_METAL_COOLING) */
/*       metals[LT_NMet] = 0; */
/* #endif */
      *energy = 0;
      return 0;
    }

  /* ----------------------------------------------------------------- */

#ifdef LT_SEv_INFO
  if(spreading_on)
    ADD3d(SN_INUM, SNdata, (SFp - SFs), SPEC_snIa, SN_num, (result * UnitMassFact / All.HubbleParam))	/* sum-up number of SnIa explosions */
      else
    ADD3d(SN_INUM, SNdata, (SFp - SFs), (SPEC_snIa + 1), SN_num, (result * UnitMassFact / All.HubbleParam));
#endif

  Zstar = get_metallicity(i, -1);
  for(Zbin = IaZbins_dim[Yset] - 1; Zstar < IaZbins[Yset][Zbin] && Zbin > 0; Zbin--)
    ;

  if(IaMbins_dim[Yset] > 1)
    {
      /*
       * !!!!!!!!! note: not supported yet !!!!!!!!!
       * the following line is just for completeness
       */
      for(Mbin = IaMbins_dim[Yset] - 1; MetP[I].iMass < IaMbins[Yset][Mbin] && Mbin > 0; Mbin--)
	;
    }
  else
    Mbin = 0;


  for(j = ejecta = 0; j < LT_NMet; j++)
    {
      /* gives mass of jth element in unit of Msun */
      metals[j] = result * IaY(Yset, j, Zbin, Mbin);

#ifdef LT_SEv_INFO
      if(spreading_on)
	ADD3d(LT_NMet, Zmass, (SFp - SFs), SPEC_snIa, j, (metals[j] * UnitMassFact))
	else
	ADD3d(LT_NMet, Zmass, (SFp - SFs), (SPEC_snIa + 1), j, (metals[j] * UnitMassFact));
#endif
      ejecta += metals[j];
    }

/* #if defined (UM_CHEMISTRY) && defined (UM_METAL_COOLING) */
/*   metals[LT_NMet] = result * Ia_AvgFillNDens[Yset][0]; */
/* #endif */

  *energy = result * All.SnIaEgy * UnitMassFact / All.HubbleParam;	/* erg in code units */

#ifdef LT_SEv_INFO
  if(spreading_on)
    ADD3d(SN_INUM, SNdata, (SFp - SFs), SPEC_snIa, SN_egy, (*energy * All.UnitEnergy_in_cgs))
    else
    ADD3d(SN_INUM, SNdata, (SFp - SFs), (SPEC_snIa + 1), SN_egy, (*energy * All.UnitEnergy_in_cgs));
#endif

  /* return ejected mass (1.4 Msun) */
  /*result /= UnitMassFact;
     return result * SnIaEjectedMass;  return ejected mass */

  return ejecta;
}

double INLINE_FUNC inner_integrand(double m, void *params)
{
  double m2 = m * m;

  double (*phi) (double, void *);

  phi = IMFp->IMFfunc_byNum;

  return (phi(m, 0x0) / (m2 * m2));
}


double INLINE_FUNC nRSnIa(double t, void *params)
{

  /* calculates Sn type Ia explosion rate in #/yr    */
  /* at some time t, for a SSP of 1Msun.             */
  /* in snIa_mass return the total mass in SnIa.     */
  /* in Rejecta return the rate of ejecta in Msun/yr */
  /* t is in Gyr */

  gsl_function localF;

  double result, err;

  size_t n_eval;

  double m2, Mb_inf, Mb_sup, fact1, fact2;

  int i, sbin1, sbin2;

  double (*phi) (double, void *);

  phi = IMFp->IMFfunc_byNum;


  m2 = dying_mass(t);

  if((m2 < All.MBms) || (m2 > All.Mup))
    return 0;

  Mb_inf = max(2 * m2, All.MBm);
  Mb_sup = 0.5 * All.MBM + m2;


  if(IMFp->type == power_law)
    {
      if(IMFp->NSlopes == 1)
	sbin1 = sbin2 = 0;
      else
	{
	  sbin1 = get_IMF_SlopeBin(Mb_sup);
	  sbin2 = get_IMF_SlopeBin(Mb_inf);
	}

      if(sbin1 == sbin2)
	{
	  fact2 = 1.0 / (3 + IMFp->Slopes.slopes[sbin1]);
	  result = ((fact2 * pow(Mb_inf, -(3 + IMFp->Slopes.slopes[sbin1]))) -
		    (fact2 * pow(Mb_sup, -(3 + IMFp->Slopes.slopes[sbin1])))) * IMFp->A[sbin1];
	}
      else
	{
	  fact1 = Mb_inf;
	  for(i = sbin1, result = 0; i <= sbin2; i++)
	    {
	      if(i > sbin1)
		Mb_sup = IMFp->Slopes.masses[i];

	      if(i < sbin2)
		Mb_sup = IMFp->Slopes.masses[i + 1];
	      else
		Mb_sup = fact1;

	      fact2 = 1.0 / (3 + IMFp->Slopes.slopes[i]);
	      result += ((fact2 * pow(Mb_inf, -(3 + IMFp->Slopes.slopes[i]))) -
			 (fact2 * pow(Mb_sup, -(3 + IMFp->Slopes.slopes[i])))) * IMFp->A[i];
	    }
	}
    }
  else
    {
      localF.function = &inner_integrand;
      localF.params = 0x0;

      if((gsl_status = gsl_integration_qng(&localF, Mb_inf, Mb_sup, 1, 1e-5, &result, &err, &n_eval)))
	{
	  printf("  >> Task %i, gsl integration error %i in Sn (k) [%.6g - %.6g]: %.6g %.6g\n",
		 ThisTask, gsl_status, Mb_inf, Mb_sup, result, err);
	  fflush(stdout);
	}
      result *= IMFp->A[0];	/* if NSlopes > 1 this should appear inside the inner_integrand function */
    }

  fact1 = All.BinFrac * 24 * m2 * m2 * (-dm_dt(m2, t));
  return (fact1 * result);
}

double get_AGB_product(int i, int Yset, double *metals, double inf_mass, double sup_mass, double *numsn)
{
  int I;

  double NonMetalMass, HMass, Abund;

  double Zstar, err, result, ejecta, tot_num;

  int j;

  /* does not care about the size of the integration time; you must care about that elsewhere */

  I = P[i].MetID;
  *numsn = 0;

  for(j = 0; j < LT_NMet; j++)
    metals[j] = 0;

  if(inf_mass > All.Mup)
    return 0;
  if(sup_mass > All.Mup)
    sup_mass = All.Mup;

  tot_num = IntegrateIMF_byNum(inf_mass, sup_mass, IMFp, EXC_BH);

  if(tot_num < 0 && tot_num > -Z_ROUND_OFF_ERROR)
    {
      printf(" > probable round-off error in integrating AGB num [%d][%d][%u] %g\n",
	     ThisTask, i, P[i].ID, tot_num);
      fflush(stdout);
      result = 0;
    }
  *numsn = tot_num * MetP[I].iMass;
  tot_num *= (MetP[I].iMass * UnitMassFact) / All.HubbleParam * (1 - All.BinFrac);

  if(tot_num == 0)
    return 0;

#ifdef LT_SEv_INFO
  if(spreading_on)
    ADD3d(SN_INUM, SNdata, (SFp - SFs), SPEC_agb, SN_num, tot_num)	/* sum-up number */
      else
    ADD3d(SN_INUM, SNdata, (SFp - SFs), (SPEC_agb + 1), SN_num, tot_num);
#endif

  Zstar = get_metallicity(i, -1);
  for(SD.Zbin = AGBZbins_dim[Yset] - 1; Zstar < AGBZbins[Yset][SD.Zbin] && SD.Zbin > 0; SD.Zbin--)
    ;

  SD.Zstar = Zstar;
  SD.Zdim = AGBZbins_dim[Yset];
  SD.ZArray = AGBZbins[Yset];
  SD.Mdim = AGBMbins_dim[Yset];
  SD.MArray = AGBMbins[Yset];

  ejecta = 0;
  F.function = &zmRSnII;
  F.params = 0x0;

  if(NonProcOn_AGB[Yset])
    {
      if((Abund = get_metalmass(MetP[I].Metals)) > 0)
	{
	  NonMetalMass = MetP[I].iMass - Abund;
	  HMass = NonMetalMass - MetP[I].Metals[Hel];
	  Abund /= NonMetalMass;
	}
    }

  for(j = 0; j < LT_NMet; j++)
    {
      SD.Y = AGBYields[Yset][j];

      if((gsl_status =
	  gsl_integration_qag(&F, inf_mass, sup_mass, 1e-6, 1e-4, gsl_ws_dim, qag_INT_KEY, w, &result, &err)))
	{
	  printf("  >> Task %i, gsl integration error %i in AGB z integration [%.6g - %.6g] : %.6g %.6g\n",
		 ThisTask, gsl_status, inf_mass, sup_mass, result, err);
	  fflush(stdout);
	  endrun(LT_ERROR_INTEGRATION_ERROR);
	}

      if(result < 0)
	{
	  printf(" $$$$$ AGB [%d][%d] %g %g %g %d\n", ThisTask, i, inf_mass, sup_mass, result, SD.Zbin);
	  fflush(stdout);
	}

      if(result < 0 && result > -Z_ROUND_OFF_ERROR)
	{
	  printf(" > probable round-off error in integrating AGB [%d][%d][%u][%d] %g\n",
		 ThisTask, i, P[i].ID, j, result);
	  fflush(stdout);
	  result = 0;
	}

      metals[j] = result * MetP[I].iMass * (1 - All.BinFrac);

      ejecta += (metals[j] = result * MetP[I].iMass * (1 - All.BinFrac));
    }

  if(NonProcOn_AGB[Yset] && Abund > 0)
    {
      /*
         say Mh is the "apparent" hydrogen mass, as stored by far in metals[Hyd]; if non processed metals should
         be accounted, this mass is the sum of the actual H mass, mh, and of the nonproc metals, mz:
         Mh = mh + mz
         where mz is
         mz = mh * Z.
         Then,
         Mh = mh (1+ Z)
         and
         mh = Mh / (1+ Z)
         mz = Mh * Z / (1+ Z)
       */

      for(j = 0; j < LT_NMet; j++)
	if(j != Hel && j != Hyd)
	  metals[j] += MetP[I].Metals[j] * (metals[Hyd] + metals[Hel]) / NonMetalMass / (1 + Abund);

      metals[Hel] /= (1 + Abund);
      metals[Hyd] /= (1 + Abund);
    }


  for(j = 0; j < LT_NMet; j++)
    {
#ifdef LT_SEv_INFO
      if(spreading_on)
	ADD3d(LT_NMet, Zmass, (SFp - SFs), SPEC_agb, j, (metals[j] * UnitMassFact))
	else
	ADD3d(LT_NMet, Zmass, (SFp - SFs), (SPEC_agb + 1), j, (metals[j] * UnitMassFact));
#endif

/* #if defined (UM_CHEMISTRY) && defined (UM_METAL_COOLING) */
/* 	  if(j == FillEl) */
/* 	    { */
/* 	      SD.Y = AGB_AvgFillNDens[Yset]; */
/* 	      if( (gsl_status = */
/* 		   gsl_integration_qag(&F, inf_mass, sup_mass, 1e-6, 1e-4, gsl_ws_dim, qag_INT_KEY, w, &result, &err)) ) */
/* 		{ */
/* 		  printf("  >> Task %i, gsl integration error %i in SnII z integration [%.6g - %.6g] : %.6g %.6g\n", */
/* 			 ThisTask, gsl_status, inf_mass, sup_mass, result, err); */
/* 		  fflush(stdout); */
/* 		  endrun(LT_ERROR_INTEGRATION_ERROR); */
/* 		} */

/* 	      if(result < 0 && result > -Z_ROUND_OFF_ERROR) */
/* 		{ */
/* 		  printf(" > probable round-off error in integrating SnII [%d][%d][%u] %g\n", */
/* 			 ThisTask, i, P[i].ID, result); */
/* 		  fflush(stdout); */
/* 		  result = 0; */
/* 		} */

/* 	      metals[LT_NMet] += result * MetP[i].iMass; */
/* 	    } */
/* #endif */
    }

  return ejecta;
}

double get_SnII_product(int i, int Yset, double *metals, double *energy, double inf_mass, double sup_mass,
			double *numsn)
{
  int I;

  double NonMetalMass, HMass, Abund;

  double Zstar, err, result, ejecta;

  double sup_mass_store, tot_num;

  int j;

  /* does not care about the size of the integration time; you must care about that elsewhere */

  I = P[i].MetID;

  *energy = 0;
  ejecta = 0;
  result = 0;
  *numsn = 0;
  for(j = 0; j < LT_NMet; j++)
    metals[j] = 0;

  if(sup_mass < All.Mup)
    return 0;
  if(inf_mass < All.Mup)
    inf_mass = All.Mup;

  tot_num = IntegrateIMF_byNum(inf_mass, sup_mass, IMFp, EXC_BH);
  *numsn = tot_num * MetP[I].iMass;
  tot_num *= (MetP[I].iMass * UnitMassFact) / All.HubbleParam;

  sup_mass_store = sup_mass;

  if(inf_mass < SFp->egyShortLiv_MassTh)
    {
      if(sup_mass_store > SFp->egyShortLiv_MassTh)
	sup_mass_store = SFp->egyShortLiv_MassTh;

      *energy = IntegrateIMF_byEgy(inf_mass, sup_mass_store, IMFp);

      if(result < 0 && result > -Z_ROUND_OFF_ERROR)
	{
	  printf(" > probable round-off error in integrating SnII num [%d][%d][%u] %g\n",
		 ThisTask, i, P[i].ID, result);
	  fflush(stdout);
	  result = 0;
	}

      *energy *= (MetP[I].iMass * UnitMassFact) / All.HubbleParam;	/* *agefact */
    }

#ifdef LT_SEv_INFO
  if(spreading_on)
    {
    ADD3d(SN_INUM, SNdata, (SFp - SFs), SPEC_snII, SN_num, tot_num)
	ADD3d(SN_INUM, SNdata, (SFp - SFs), SPEC_snII, SN_egy, (*energy * All.UnitEnergy_in_cgs))}
  else
    {
    ADD3d(SN_INUM, SNdata, (SFp - SFs), (SPEC_snII + 1), SN_num, tot_num)
	ADD3d(SN_INUM, SNdata, (SFp - SFs), (SPEC_snII + 1), SN_egy, (*energy * All.UnitEnergy_in_cgs))}
#endif

  /*
   * integrate metal production and H&He ejection
   *
   *    -> metals
   */

  if(inf_mass < SFp->metShortLiv_MassTh)
    {
      if(sup_mass > SFp->metShortLiv_MassTh)
	sup_mass = SFp->metShortLiv_MassTh;

      Zstar = get_metallicity(i, -1);
      for(SD.Zbin = IIZbins_dim[Yset] - 1; Zstar < IIZbins[Yset][SD.Zbin] && SD.Zbin > 0; SD.Zbin--)
	;

      SD.Zstar = Zstar;
      SD.Zdim = IIZbins_dim[Yset];
      SD.ZArray = IIZbins[Yset];
      SD.Mdim = IIMbins_dim[Yset];
      SD.MArray = IIMbins[Yset];

      F.function = &zmRSnII;
      F.params = 0x0;

      if(NonProcOn_II[Yset])
	{
	  if((Abund = get_metalmass(MetP[I].Metals)) > 0)
	    {
	      NonMetalMass = MetP[I].iMass - Abund;
	      HMass = NonMetalMass - MetP[I].Metals[Hel];
	      Abund /= NonMetalMass;
	    }
	}

      for(j = 0; j < LT_NMet; j++)
	{
	  SD.Y = SnIIYields[Yset][j];

	  if((gsl_status =
	      gsl_integration_qag(&F, inf_mass, sup_mass, 1e-6, 1e-4, gsl_ws_dim, qag_INT_KEY, w, &result,
				  &err)))
	    {
	      printf
		("  >> Task %i, gsl integration error %i in SnII z integration [%.6g - %.6g] : %.6g %.6g\n",
		 ThisTask, gsl_status, inf_mass, sup_mass, result, err);
	      fflush(stdout);
	      endrun(LT_ERROR_INTEGRATION_ERROR);
	    }

	  if(result < 0 && result > -Z_ROUND_OFF_ERROR)
	    {
	      printf(" > probable round-off error in integrating SnII [%d][%d][%u] %g\n",
		     ThisTask, i, P[i].ID, result);
	      fflush(stdout);
	      result = 0;
	    }

	  /* give metal mass in Msun in code units */
	  ejecta += (metals[j] = result * MetP[I].iMass);	/* *agefact */
	}

      if(NonProcOn_II[Yset] && Abund > 0)
	{
	  for(j = 0; j < LT_NMet; j++)
	    if(j != Hel && j != Hyd)
	      metals[j] += MetP[I].Metals[j] * (metals[Hyd] + metals[Hel]) / NonMetalMass / (1 + Abund);

	  metals[Hel] /= (1 + Abund);
	  metals[Hyd] /= (1 + Abund);
	}


      for(j = 0; j < LT_NMet; j++)
	{
#ifdef LT_SEv_INFO
	  if(spreading_on)
	    ADD3d(LT_NMet, Zmass, (SFp - SFs), SPEC_snII, j, (metals[j] * UnitMassFact))
	    else
	    ADD3d(LT_NMet, Zmass, (SFp - SFs), (SPEC_snII + 1), j, (metals[j] * UnitMassFact));;
#endif

/* #if defined (UM_CHEMISTRY) && defined (UM_METAL_COOLING) */
/* 	  if(j == FillEl) */
/* 	    { */
/* 	      SD.Y = II_AvgFillNDens[Yset]; */
/* 	      if( (gsl_status = */
/* 		   gsl_integration_qag(&F, inf_mass, sup_mass, 1e-6, 1e-4, gsl_ws_dim, qag_INT_KEY, w, &result, &err)) ) */
/* 		{ */
/* 		  printf("  >> Task %i, gsl integration error %i in SnII z integration [%.6g - %.6g] : %.6g %.6g\n", */
/* 			 ThisTask, gsl_status, inf_mass, sup_mass, result, err); */
/* 		  fflush(stdout); */
/* 		  endrun(LT_ERROR_INTEGRATION_ERROR); */
/* 		} */

/* 	      if(result < 0 && result > -Z_ROUND_OFF_ERROR) */
/* 		{ */
/* 		  printf(" > probable round-off error in integrating SnII [%d][%d][%u] %g\n", */
/* 			 ThisTask, i, P[i].ID, result); */
/* 		  fflush(stdout); */
/* 		  result = 0; */
/* 		} */

/* 	      metals[LT_NMet] += result * MetP[i].iMass; */
/* 	    } */
/* #endif */

	}
    }

  return ejecta;
}


double INLINE_FUNC nRSnII(double time, void *p)
{
  /* calculates how many stars are dying */
  /* at some time t, for a SSP of 1Msun. */
  /* t is in Gyr */

  double m;

  double (*phi) (double, void *);

  phi = IMFp->IMFfunc_byNum;

  if((time >= IMFp->inf_lifetime) &&	/* Mm Msun < m < MUP Msun */
     (time <= IMFp->sup_lifetime))
    {
      m = dying_mass(time);
      return (phi(m, p) * (-dm_dt(m, time)));
    }
  else
    return 0;
}


double INLINE_FUNC mRSnII(double time, void *p)
{
  /* calculates how many stars are dying */
  /* at some time t, for a SSP of 1Msun. */
  /* t is in Gyr                         */
  /* results is in Msun/Gyr              */

  int BHi;

  double m, fact;

  double (*phi) (double, void *);

  phi = IMFp->IMFfunc_byNum;

  if((time >= IMFp->inf_lifetime) &&	/* Mm Msun < m < MUP Msun */
     (time <= IMFp->sup_lifetime))
    {
      m = dying_mass(time);

      for(BHi = 0; (BHi < IMFp->N_notBH_ranges); BHi++)
	if(m <= IMFp->notBH_ranges.sup[BHi] && m >= IMFp->notBH_ranges.inf[BHi])
	  break;

      if(BHi == IMFp->N_notBH_ranges)
	return 0;

      fact = phi(m, p) * (-dm_dt(m, time));
      return fact * m;
    }
  else
    return 0;
}


double INLINE_FUNC zmRSnII(double m, void *p)
{
  /* calculates Sn type II explosion rate in Msun/Gyr */
  /* at some time t, for a SSP of 1Msun.              */
  /* t is in Gyr */

  int ir, BHi, zone;

  double y, t, u;

  double (*phi) (double, void *);

  for(BHi = 0; (BHi < IMFp->N_notBH_ranges); BHi++)
    if(m <= IMFp->notBH_ranges.sup[BHi] && m >= IMFp->notBH_ranges.inf[BHi])
      break;

  if(BHi == IMFp->N_notBH_ranges)
    return 0;

  phi = IMFp->IMFfunc_byNum;

  if((m >= IMFp->Mm) &&		/* Mm Msun < m < MU Msun */
     (m <= IMFp->MU))
    {
      for(ir = SD.Mdim - 1; m < SD.MArray[ir] && ir > 0; ir--)
	;

      if(SD.Zdim > 1)
	{
	  zone = 0;
	  zone |= (SD.Zstar >= SD.ZArray[SD.Zdim - 1]);
	  zone |= ((m >= SD.MArray[SD.Mdim - 1]) << 1);
	  zone |= ((SD.Zstar < SD.ZArray[0]) << 2);
	  zone |= ((m < SD.MArray[0]) << 3);
	}
      else
	{
	  zone = 16;
	  zone |= (m >= SD.MArray[SD.Mdim - 1]);
	  zone |= ((m < SD.MArray[0]) << 1);
	}

      switch (zone)
	{
	case 0:
	  t = (m - SD.MArray[ir]) / (SD.MArray[ir + 1] - SD.MArray[ir]);
	  u = (SD.Zstar - SD.ZArray[SD.Zbin]) / (SD.ZArray[SD.Zbin + 1] - SD.ZArray[SD.Zbin]);
	  y = (1 - t) * (1 - u) * SD.Y[SD.Zbin * SD.Mdim + ir] +
	    t * (1 - u) * SD.Y[SD.Zbin * SD.Mdim + ir + 1] +
	    t * u * SD.Y[(SD.Zbin + 1) * SD.Mdim + ir + 1] + (1 - t) * u * SD.Y[(SD.Zbin + 1) * SD.Mdim + ir];
	  break;
	case 2:
	case 8:
	  u = (SD.Zstar - SD.ZArray[SD.Zbin]) / (SD.ZArray[SD.Zbin + 1] - SD.ZArray[SD.Zbin]);
	  y = (1 - u) * SD.Y[SD.Zbin * SD.Mdim + ir] + u * SD.Y[(SD.Zbin + 1) * SD.Mdim + ir];
	  if(zone == 8)
	    y *= (m / SD.MArray[0]);
	  break;
	case 1:
	case 4:
	  t = (m - SD.MArray[ir]) / (SD.MArray[ir + 1] - SD.MArray[ir]);
	  y = (1 - t) * SD.Y[SD.Zbin * SD.Mdim + ir] + t * SD.Y[SD.Zbin * SD.Mdim + ir + 1];
	  break;
	case 9:
	case 12:
	  y = SD.Y[SD.Zbin * SD.Mdim + ir] * (m / SD.MArray[0]);
	  break;
	case 3:
	case 6:
	  y = SD.Y[SD.Zbin * SD.Mdim + ir];
	  break;
	case 16:
	  t = (m - SD.MArray[ir]) / (SD.MArray[ir + 1] - SD.MArray[ir]);
	  y = (1 - t) * SD.Y[SD.Zbin * SD.Mdim + ir] + t * SD.Y[SD.Zbin * SD.Mdim + ir + 1];
	  break;
	case 17:
	  if(SD.ExtrDir & 1 && !SD.ExtrDir & 1 << 31)
	    {
	      t = (m - SD.MArray[SD.Mdim - 1]) / (SD.ExtrMArray[0] - SD.MArray[SD.Mdim - 1]);
	      y = (1 - t) * SD.Y[SD.Zbin * SD.Mdim + SD.Mdim - 1] + t * SD.ExtrY[SD.ExtrZbin * SD.ExtrMdim];
	    }
	  else
	    y = SD.Y[SD.Zbin * SD.Mdim + SD.Mdim - 1];
	  break;
	case 18:
	  if(SD.ExtrDir & 1 && SD.ExtrDir & 1 << 31)
	    {
	      t = (m - SD.ExtrMArray[SD.ExtrMdim - 1]) / (SD.MArray[0] - SD.ExtrMArray[SD.ExtrMdim - 1]);
	      y = (1 - t) * SD.ExtrY[SD.ExtrZbin * SD.ExtrMdim + SD.ExtrMdim - 1] +
		t * SD.Y[SD.Zbin * SD.Mdim];
	    }
	  else
	    y = SD.Y[SD.Zbin * SD.Mdim] * (m / SD.MArray[0]);
	  break;
	}

      return y * phi(m, p);
    }
  else
    return 0;
}


double INLINE_FUNC ejectaSnII(double m, void *p)
{
  /* calculates Sn type II restored mass in Msun */
  /* for a star of mass m */

  int ir, BHi;

  double t, mr;

  double (*phi) (double, void *);

  for(BHi = 0; (BHi < IMFp->N_notBH_ranges); BHi++)
    if(m <= IMFp->notBH_ranges.sup[BHi] && m >= IMFp->notBH_ranges.inf[BHi])
      break;

  if(BHi == IMFp->N_notBH_ranges)
    return 0;

  phi = IMFp->IMFfunc_byNum;

  if(m >= IMFp->Mm && m <= IMFp->MU)
    {
      if(getindex((double *) &SD.MArray[0], 0, SD.Mdim - 1, &m, &ir) < 0)
	{
	  mr = SD.Y[SD.Zbin * SD.Mdim + ir];
	  if(m < SD.MArray[0])
	    mr = SD.Y[SD.Zbin * SD.Mdim] * (m / SD.MArray[0]);
	}
      else
	{
	  t = (m - SD.MArray[ir]) / (SD.MArray[ir + 1] - SD.MArray[ir]);
	  mr = (1 - t) * SD.Y[SD.Zbin * SD.Mdim + ir] + t * SD.Y[SD.Zbin * SD.Mdim + ir + 1];
	}
      return mr * phi(m, p);
    }
  else
    return 0;
}

double INLINE_FUNC ztRSnII(double time, void *p)
{
  /* return the production rate in mass for elements at time t */

  int ir, BHi, zone;

  double m, t, u, y;

  /*
   * linear interpolation in 2 dim:
   *   y1 = y[j][k]
   *   y2 = y[j+1][k]
   *   y3 = y[j+1][k+1]
   *   y4 = y[j][k+1]
   *
   *   y(x1,x2) = (1-t)*(1-u)*y1 + t*(1-u)*y2 + t*u*y3 + (1-t)*u * y4
   *
   */


  if((time >= IMFp->inf_lifetime) &&	/* Mm Msun < m < MUP Msun */
     (time <= IMFp->sup_lifetime))
    {
      m = dying_mass(time);

      for(BHi = 0; (BHi < IMFp->N_notBH_ranges); BHi++)
	if(m <= IMFp->notBH_ranges.sup[BHi] && m >= IMFp->notBH_ranges.inf[BHi])
	  break;

      if(BHi == IMFp->N_notBH_ranges)
	return 0;

      for(ir = SD.Mdim - 1; m < SD.MArray[ir] && ir > 0; ir--)
	;

      if(SD.Zdim > 1)
	{
	  zone = 0;
	  zone |= (SD.Zstar >= SD.ZArray[SD.Zdim - 1]);
	  zone |= ((m >= SD.MArray[SD.Mdim - 1]) << 1);
	  zone |= ((SD.Zstar < SD.ZArray[0]) << 2);
	  zone |= ((m < SD.MArray[0]) << 3);
	}
      else
	{
	  zone = 16;
	  zone |= (m >= SD.MArray[SD.Mdim - 1]);
	  zone |= ((m < SD.MArray[0]) << 1);
	}

      switch (zone)
	{
	case 0:
	  t = (m - SD.MArray[ir]) / (SD.MArray[ir + 1] - SD.MArray[ir]);
	  u = (SD.Zstar - SD.ZArray[SD.Zbin]) / (SD.ZArray[SD.Zbin + 1] - SD.ZArray[SD.Zbin]);
	  y = (1 - t) * (1 - u) * SD.Y[SD.Zbin * SD.Mdim + ir] +
	    t * (1 - u) * SD.Y[SD.Zbin * SD.Mdim + ir + 1] +
	    t * u * SD.Y[(SD.Zbin + 1) * SD.Mdim + ir + 1] + (1 - t) * u * SD.Y[(SD.Zbin + 1) * SD.Mdim + ir];
	  break;
	case 2:
	case 8:
	  u = (SD.Zstar - SD.ZArray[SD.Zbin]) / (SD.ZArray[SD.Zbin + 1] - SD.ZArray[SD.Zbin]);
	  y = (1 - u) * SD.Y[SD.Zbin * SD.Mdim + ir] + u * SD.Y[(SD.Zbin + 1) * SD.Mdim + ir];
	  if(zone == 8)
	    y *= (m / SD.MArray[0]);
	  break;
	case 1:
	case 4:
	  t = (m - SD.MArray[ir]) / (SD.MArray[ir + 1] - SD.MArray[ir]);
	  y = (1 - t) * SD.Y[SD.Zbin * SD.Mdim + ir] + t * SD.Y[SD.Zbin * SD.Mdim + ir + 1];
	  break;
	case 9:
	case 12:
	  y = SD.Y[SD.Zbin * SD.Mdim + ir] * (m / SD.MArray[0]);
	  break;
	case 3:
	case 6:
	  y = SD.Y[SD.Zbin * SD.Mdim + ir];
	  break;
	case 16:
	  t = (m - SD.MArray[ir]) / (SD.MArray[ir + 1] - SD.MArray[ir]);
	  y = (1 - t) * SD.Y[SD.Zbin * SD.Mdim + ir] + t * SD.Y[SD.Zbin * SD.Mdim + ir + 1];
	  break;
	case 17:
	  if(SD.ExtrDir & 1 && !SD.ExtrDir & 1 << 31)
	    {
	      t = (m - SD.MArray[SD.Mdim - 1]) / (SD.ExtrMArray[0] - SD.MArray[SD.Mdim - 1]);
	      y = (1 - t) * SD.Y[SD.Zbin * SD.Mdim + SD.Mdim - 1] + t * SD.ExtrY[SD.ExtrZbin * SD.ExtrMdim];
	    }
	  else
	    y = SD.Y[SD.Zbin * SD.Mdim + SD.Mdim - 1];
	  break;
	case 18:
	  if(SD.ExtrDir & 1 && SD.ExtrDir & 1 << 31)
	    {
	      t = (m - SD.ExtrMArray[SD.ExtrMdim - 1]) / (SD.MArray[0] - SD.ExtrMArray[SD.ExtrMdim - 1]);
	      y = (1 - t) * SD.ExtrY[SD.ExtrZbin * SD.ExtrMdim + SD.ExtrMdim - 1] +
		t * SD.Y[SD.Zbin * SD.Mdim];
	    }
	  else
	    y = SD.Y[SD.Zbin * SD.Mdim] * (m / SD.MArray[0]);
	  break;
	}
      if(p != 0x0)
	{
	  printf(" [ %g %d %d %d %g %g %g %g %g\n", time, zone, SD.Zbin, ir, t, y, m, SD.MArray[ir],
		 SD.Y[SD.Zbin * SD.Mdim + ir]);
	  fflush(stdout);
	}

      return nRSnII(time, p) * y;
    }
  else
    return 0;
}
