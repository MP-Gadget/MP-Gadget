#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "allvars.h"
#include "proto.h"

/*! \file c_metals.c 
 *  
 *  This file contains some of the routines for the
 *  chemical model, feedback and decoupling. 
 */

#ifdef CS_MODEL
#include "cs_metals.h"



/* input/output */
#if defined(CS_SNI) || defined(CS_SNII)
FILE *FdSN;
#endif
#ifdef CS_FEEDBACK
FILE *FdPromotion;

#ifdef CS_TESTS
FILE *FdEgyTest;
FILE *FdPromTest;
FILE *FdSNTest;
#endif
#endif


/* global variables */

int Flag_phase;
double InternalEnergy;
double Energy_cooling;

#ifdef CS_SNII
double Raiteri_COEFF_1;
double Raiteri_COEFF_2;
double Raiteri_COEFF_3;
double Raiteri_COEFF_4;
double Raiteri_COEFF_5;
#endif

#ifdef CS_FEEDBACK
double SN_Energy;

#ifdef CS_TESTS
double Energy_promotion;
double Energy_feedback, Energy_reservoir;
#endif
#endif


/* other  variables internal to this file */
int *Nlines, cont_sm;
double **logLambda_i;
double **logT_i;
double **yield1, **yield2, **yield3, **yield4, **yield5;
double metal, sm2, FeHgas;
double **yy;

float **nsimfww;
double XH, yhelium;
int numenrich;


/* This function reads the cooling tables from 
      Sutherland & Dopita 1993 */

FILE *fpcool;

void cs_read_coolrate_table(void)
{
  int i, itab;


  Nlines = (int *) calloc(Nab, sizeof(int));
  logT_i = (double **) calloc(Nab, sizeof(double *));
  logLambda_i = (double **) calloc(Nab, sizeof(double *));


  for(itab = 0; itab <= 7; itab++)	/* 7 is the number of cooling tables */
    {
      if(itab == 0)
	fpcool = fopen("tablas_cooling/mzero.cie", "r");	/*  primordial abundance */
      else if(itab == 1)
	fpcool = fopen("tablas_cooling/m-30.cie", "r");
      else if(itab == 2)
	fpcool = fopen("tablas_cooling/m-20.cie", "r");
      else if(itab == 3)
	fpcool = fopen("tablas_cooling/m-15.cie", "r");
      else if(itab == 4)
	fpcool = fopen("tablas_cooling/m-10.cie", "r");
      else if(itab == 5)
	fpcool = fopen("tablas_cooling/m-05.cie", "r");
      else if(itab == 6)
	fpcool = fopen("tablas_cooling/m-00.cie", "r");
      else if(itab == 7)
	fpcool = fopen("tablas_cooling/m+05.cie", "r");

      if(fpcool == NULL)
	{
	  printf("can't open Cooling Table %d\n", itab);
	  endrun(2);
	}

      fscanf(fpcool, "%d\n", Nlines + itab);
      *(logT_i + itab) = (double *) malloc(*(Nlines + itab) * sizeof(double));
      *(logLambda_i + itab) = (double *) malloc(*(Nlines + itab) * sizeof(double));
      for(i = 0; i < (*(Nlines + itab)); i++)
	fscanf(fpcool, "%lf %*f %*f %*f %*f %lf %*f %*f %*f %*f %*f %*f\n", *(logT_i + itab) + i,
	       *(logLambda_i + itab) + i);

    }
}


/**********************************************************************************************/
/**********************************************************************************************/

/* This function selects the correct cooling rate for a given temperature
and abundance */

double cs_get_Lambda_SD(double logT, double abund)
{
  double fhi = 0., flow = 0., t = 0.;
  double Lambda = 0., logLambda = 0.;
  int itab = 0, j = 0;

  if(abund <= -3.5)
    itab = 0;
  if(abund > -3.5 && abund <= -2.5)
    itab = 1;
  if(abund > -2.5 && abund <= -1.75)
    itab = 2;
  if(abund > -1.75 && abund <= -1.25)
    itab = 3;
  if(abund > -1.25 && abund <= -0.75)
    itab = 4;
  if(abund > -0.75 && abund <= -0.25)
    itab = 5;
  if(abund > -0.25 && abund <= 0.25)
    itab = 6;
  if(abund > 0.25)
    itab = 7;

  if(logT < *(*(logT_i + itab) + 0))
    Lambda = 0;
  else
    {
      if(logT > *(*(logT_i + itab) + *(Nlines + itab) - 1))
	Lambda = pow(10.0, *(*(logLambda_i + itab) + *(Nlines + itab) - 1));
      else
	{
	  /* Interpolation to get the correct Cooling Rate */

	  t = (logT - *(*(logT_i + itab) + 0)) / delta;
	  j = (int) t;

	  fhi = t - j;
	  flow = 1 - fhi;

	  logLambda = flow * (*(*(logLambda_i + itab) + j)) + fhi * (*(*(logLambda_i + itab) + j + 1));
	  Lambda = pow(10.0, logLambda);
	}
    }
  return Lambda;
}


/**********************************************************************************************/
/**********************************************************************************************/

/* This function reads the SNII yields tables from Woosley & Weaver 1995 
   Chemical elements: 3He, 12C, 24Mg, 16O, 56Fe, 28Si, H, 14N, 20Ne, 32S, 40Ca, 62Zn, 56N
   As 56N decays into 56Fe in a very short time-scale, we add its contribution to
   56Fe. Finally we take into account 12 elements */


#ifdef CS_SNII

FILE *fpyield;
void cs_read_yield_table(void)
{
  int i, j, itab;

  yield1 = (double **) calloc(Nelements, sizeof(double *));
  yield2 = (double **) calloc(Nelements, sizeof(double *));
  yield3 = (double **) calloc(Nelements, sizeof(double *));
  yield4 = (double **) calloc(Nelements, sizeof(double *));
  yield5 = (double **) calloc(Nelements, sizeof(double *));

  for(i = 0; i < Nelements; i++)
    {
      *(yield1 + i) = (double *) calloc(Nmass, sizeof(double));
      *(yield2 + i) = (double *) calloc(Nmass, sizeof(double));
      *(yield3 + i) = (double *) calloc(Nmass, sizeof(double));
      *(yield4 + i) = (double *) calloc(Nmass, sizeof(double));
      *(yield5 + i) = (double *) calloc(Nmass, sizeof(double));
    }

  /* this is used as the general one */
  yy = (double **) calloc(Nelements, sizeof(double *));
  for(i = 0; i < Nelements; i++)
    *(yy + i) = (double *) calloc(Nmass, sizeof(double));

  for(itab = 1; itab <= 5; itab++)
    {

      if(itab == 1)
	fpyield = fopen("yieldsww/yield1.dat-ww", "r");	/* primordial abundance */
      else if(itab == 2)
	fpyield = fopen("yieldsww/yield2.dat-ww", "r");
      else if(itab == 3)
	fpyield = fopen("yieldsww/yield3.dat-ww", "r");
      else if(itab == 4)
	fpyield = fopen("yieldsww/yield4.dat-ww", "r");
      else if(itab == 5)
	fpyield = fopen("yieldsww/yield5.dat-ww", "r");

      if(fpyield == NULL)
	{
	  printf("can't open Yield Table %d \n", itab);
	  endrun(11);
	}

      for(i = 0; i < Nelements; i++)
	for(j = 0; j < Nmass; j++)
	  {
	    fscanf(fpyield, "   %lf\n", (*(yy + i) + j));
	    if(i == 12)
	      *(*(yy + 4) + j) += *(*(yy + 12) + j);

	    *(*(yy + 4) + j) /= 2;	/* Correction of Fe yield */
	  }


      for(i = 0; i < Nelements; i++)
	for(j = 0; j < Nmass; j++)
	  {
	    if(itab == 1)
	      *(*(yield1 + i) + j) = *(*(yy + i) + j);
	    else if(itab == 2)
	      *(*(yield2 + i) + j) = *(*(yy + i) + j);
	    else if(itab == 3)
	      *(*(yield3 + i) + j) = *(*(yy + i) + j);
	    else if(itab == 4)
	      *(*(yield4 + i) + j) = *(*(yy + i) + j);
	    else if(itab == 5)
	      *(*(yield5 + i) + j) = *(*(yy + i) + j);
	  }

      fclose(fpyield);
    }
}



/**********************************************************************************************/
/**********************************************************************************************/

/* This function selects the correct  SNII yields for a given metallicity and
   stellar mass. */

double cs_SNII_yields(int index)
{
#define  Zsol 0.02

  int i, j, ik;
  int kmetal = 0;
  double yield[12][10];
  double delta_metals = 0;
  double metals_in_element[12];
  double check = 0.;
  double metal = 0;

# ifdef CS_FEEDBACK
  double sn_number = 0;
# endif

  for(i = 0; i < Nelements - 1; i++)	/* Now we have 12 elements */
    for(j = 0; j < Nmass; j++)
      yield[i][j] = 0;

  metal = 0;
  for(ik = 1; ik < Nelements - 1; ik++)	/* all chemical elements but H & He */
    if(ik != 6)
      metal += P[index].Zm[ik];

  metal /= (P[index].Mass * Zsol);	/* metallicity in solar units */

  if(metal < 0.)
    {
      printf("ERROR: Negative metallicity %g\n", metal);
      endrun(330);
    }

  for(i = 0; i < Nelements - 1; i++)
    check += P[index].ZmReservoir[i];
  if(check > 0.)
#ifdef CS_FEEDBACK
    if(Flag_phase == 1)
#endif
      {
	printf("ERROR Part=%d enters SNII_yields with Reservoir=%g\n", P[index].ID, check);
	endrun(331);
      }

  if(metal < 0.0001)
    {
      kmetal = 0;
      for(i = 0; i < Nelements - 1; i++)
	for(j = 0; j < Nmass; j++)
	  yield[i][j] = *(*(yield1 + i) + j);
    }

  if(metal >= 0.0001 && metal < 0.01)
    {
      kmetal = 1;
      for(i = 0; i < Nelements - 1; i++)
	for(j = 0; j < Nmass; j++)
	  yield[i][j] = *(*(yield2 + i) + j);
    }

  if(metal >= 0.01 && metal < 0.1)
    {
      kmetal = 2;
      for(i = 0; i < Nelements - 1; i++)
	for(j = 0; j < Nmass; j++)
	  yield[i][j] = *(*(yield3 + i) + j);
    }

  if(metal >= 0.1 && metal < 1)
    {
      kmetal = 3;
      for(i = 0; i < Nelements - 1; i++)
	for(j = 0; j < Nmass; j++)
	  yield[i][j] = *(*(yield4 + i) + j);
    }

  if(metal >= 1)
    {
      kmetal = 4;
      for(i = 0; i < Nelements - 1; i++)
	for(j = 0; j < Nmass; j++)
	  yield[i][j] = *(*(yield5 + i) + j);
    }

  /* Calculation of the production of metals by the stellar mass considered. */

  for(i = 0; i < Nelements - 1; i++)	/* Note: 56N has already decayed into 56Fe */
    {
      metals_in_element[i] = 0;

      for(j = 0; j < Nmass; j++)
	{
	  metals_in_element[i] += P[index].Mass * (*(*(nsimfww + kmetal) + j)) * (yield[i][j]);
#ifdef CS_FEEDBACK
	  if(i == 0)
	    sn_number += (*(*(nsimfww + kmetal) + j));
#endif
	}
      P[index].ZmReservoir[i] = metals_in_element[i];

      delta_metals += metals_in_element[i];	/* total mass in metals to be ejected */
    }

  for(i = 0; i < Nelements - 1; i++)
    P[index].Zm[i] *= (1 - delta_metals / P[index].Mass);

  P[index].Mass -= delta_metals;


#ifdef CS_FEEDBACK
  if(P[index].EnergySN != 0.)
    {
      printf("ERROR - Star = %d comes with EnergySN = %g\n", index, P[index].EnergySN);
      endrun(23487686);
    }

  /* Note: when a gas particle with a non-zero energy reservoir is transformed into a star  
     this reservoir energy is dumped into the EnergySNCold reservoir to be distributed.
     Then, all stars should enter here with a zero-energy reservoir */
  P[index].EnergySN = sn_number * P[index].Mass * SN_Energy / SOLAR_MASS * All.UnitMass_in_g / All.HubbleParam;	/*conversion to internal units. IMF comes in units of 1M_sun */
#endif

  return delta_metals;
}



/**********************************************************************************************/
/**********************************************************************************************/


/* This function construct the Initial Mass Function compatible with yield tables.  */
void cs_imf(void)
{
  int i;

  /* Salpeter IMF */
#define pend 1.35		/* Slope  */
#define cnorm 0.1706		/* Normalization */

  int *Nlines2;			/* Number of files of each table */

  Nlines2 = (int *) calloc(5, sizeof(int));
  *(Nlines2 + 0) = 10;
  *(Nlines2 + 1) = 10;
  *(Nlines2 + 2) = 10;
  *(Nlines2 + 3) = 10;
  *(Nlines2 + 4) = 12;

  nsimfww = (float **) calloc(5, sizeof(float *));
  for(i = 0; i < 5; i++)
    *(nsimfww + i) = (float *) calloc(*(Nlines2 + i), sizeof(float));

  /*  Z=0  in solar units */
  *(*(nsimfww + 0) + 0) = cnorm / (1 - pend) * (pow(11, (1 - pend)) - pow(10, (1 - pend))) / 10.5;
  *(*(nsimfww + 0) + 1) = cnorm / (1 - pend) * (pow(13, (1 - pend)) - pow(11, (1 - pend))) / 12.0;
  *(*(nsimfww + 0) + 2) = cnorm / (1 - pend) * (pow(14, (1 - pend)) - pow(13, (1 - pend))) / 13.5;
  *(*(nsimfww + 0) + 3) = cnorm / (1 - pend) * (pow(16, (1 - pend)) - pow(14, (1 - pend))) / 15.0;
  *(*(nsimfww + 0) + 4) = cnorm / (1 - pend) * (pow(18, (1 - pend)) - pow(16, (1 - pend))) / 17.0;
  *(*(nsimfww + 0) + 5) = cnorm / (1 - pend) * (pow(23, (1 - pend)) - pow(18, (1 - pend))) / 20.5;
  *(*(nsimfww + 0) + 6) = cnorm / (1 - pend) * (pow(24, (1 - pend)) - pow(23, (1 - pend))) / 23.5;
  *(*(nsimfww + 0) + 7) = cnorm / (1 - pend) * (pow(28, (1 - pend)) - pow(24, (1 - pend))) / 26.0;
  *(*(nsimfww + 0) + 8) = cnorm / (1 - pend) * (pow(38, (1 - pend)) - pow(28, (1 - pend))) / 32.0;
  *(*(nsimfww + 0) + 9) = cnorm / (1 - pend) * (pow(40, (1 - pend)) - pow(38, (1 - pend))) / 39.0;


  /*  Z=0.0001 in solar units  */
  *(*(nsimfww + 1) + 0) = cnorm / (1 - pend) * (pow(11, (1 - pend)) - pow(10, (1 - pend))) / 10.5;
  *(*(nsimfww + 1) + 1) = cnorm / (1 - pend) * (pow(13, (1 - pend)) - pow(11, (1 - pend))) / 12.0;
  *(*(nsimfww + 1) + 2) = cnorm / (1 - pend) * (pow(16, (1 - pend)) - pow(13, (1 - pend))) / 14.5;
  *(*(nsimfww + 1) + 3) = cnorm / (1 - pend) * (pow(18, (1 - pend)) - pow(16, (1 - pend))) / 17.0;
  *(*(nsimfww + 1) + 4) = cnorm / (1 - pend) * (pow(20, (1 - pend)) - pow(18, (1 - pend))) / 19.0;
  *(*(nsimfww + 1) + 5) = cnorm / (1 - pend) * (pow(23, (1 - pend)) - pow(20, (1 - pend))) / 21.5;
  *(*(nsimfww + 1) + 6) = cnorm / (1 - pend) * (pow(26, (1 - pend)) - pow(23, (1 - pend))) / 24.5;
  *(*(nsimfww + 1) + 7) = cnorm / (1 - pend) * (pow(36, (1 - pend)) - pow(26, (1 - pend))) / 31.0;
  *(*(nsimfww + 1) + 8) = cnorm / (1 - pend) * (pow(38, (1 - pend)) - pow(36, (1 - pend))) / 37.0;
  *(*(nsimfww + 1) + 9) = cnorm / (1 - pend) * (pow(40, (1 - pend)) - pow(38, (1 - pend))) / 39.0;


  /*  Z=0.01 in solar units */
  *(*(nsimfww + 2) + 0) = cnorm / (1 - pend) * (pow(11, (1 - pend)) - pow(10, (1 - pend))) / 10.5;
  *(*(nsimfww + 2) + 1) = cnorm / (1 - pend) * (pow(13, (1 - pend)) - pow(11, (1 - pend))) / 12.0;
  *(*(nsimfww + 2) + 2) = cnorm / (1 - pend) * (pow(16, (1 - pend)) - pow(13, (1 - pend))) / 14.5;
  *(*(nsimfww + 2) + 3) = cnorm / (1 - pend) * (pow(18, (1 - pend)) - pow(16, (1 - pend))) / 17.0;
  *(*(nsimfww + 2) + 4) = cnorm / (1 - pend) * (pow(20, (1 - pend)) - pow(18, (1 - pend))) / 19.0;
  *(*(nsimfww + 2) + 5) = cnorm / (1 - pend) * (pow(23, (1 - pend)) - pow(20, (1 - pend))) / 21.5;
  *(*(nsimfww + 2) + 6) = cnorm / (1 - pend) * (pow(31, (1 - pend)) - pow(23, (1 - pend))) / 27.0;
  *(*(nsimfww + 2) + 7) = cnorm / (1 - pend) * (pow(35, (1 - pend)) - pow(31, (1 - pend))) / 33.0;
  *(*(nsimfww + 2) + 8) = cnorm / (1 - pend) * (pow(38, (1 - pend)) - pow(35, (1 - pend))) / 36.5;
  *(*(nsimfww + 2) + 9) = cnorm / (1 - pend) * (pow(40, (1 - pend)) - pow(38, (1 - pend))) / 39.0;


  /*  Z=0.1 in solar units */
  *(*(nsimfww + 3) + 0) = cnorm / (1 - pend) * (pow(11, (1 - pend)) - pow(10, (1 - pend))) / 10.5;
  *(*(nsimfww + 3) + 1) = cnorm / (1 - pend) * (pow(13, (1 - pend)) - pow(11, (1 - pend))) / 12.0;
  *(*(nsimfww + 3) + 2) = cnorm / (1 - pend) * (pow(16, (1 - pend)) - pow(13, (1 - pend))) / 14.5;
  *(*(nsimfww + 3) + 3) = cnorm / (1 - pend) * (pow(18, (1 - pend)) - pow(16, (1 - pend))) / 17.0;
  *(*(nsimfww + 3) + 4) = cnorm / (1 - pend) * (pow(20, (1 - pend)) - pow(18, (1 - pend))) / 19.0;
  *(*(nsimfww + 3) + 5) = cnorm / (1 - pend) * (pow(23, (1 - pend)) - pow(20, (1 - pend))) / 21.5;
  *(*(nsimfww + 3) + 6) = cnorm / (1 - pend) * (pow(31, (1 - pend)) - pow(23, (1 - pend))) / 27.0;
  *(*(nsimfww + 3) + 7) = cnorm / (1 - pend) * (pow(35, (1 - pend)) - pow(31, (1 - pend))) / 33.0;
  *(*(nsimfww + 3) + 8) = cnorm / (1 - pend) * (pow(38, (1 - pend)) - pow(35, (1 - pend))) / 36.5;
  *(*(nsimfww + 3) + 9) = cnorm / (1 - pend) * (pow(40, (1 - pend)) - pow(38, (1 - pend))) / 39.0;


  /*  Z=1 in solar units */
  *(*(nsimfww + 4) + 0) = cnorm / (1 - pend) * (pow(10, (1 - pend)) - pow(9, (1 - pend))) / 9.5;
  *(*(nsimfww + 4) + 1) = cnorm / (1 - pend) * (pow(11, (1 - pend)) - pow(10, (1 - pend))) / 10.5;
  *(*(nsimfww + 4) + 2) = cnorm / (1 - pend) * (pow(13, (1 - pend)) - pow(11, (1 - pend))) / 12.0;
  *(*(nsimfww + 4) + 3) = cnorm / (1 - pend) * (pow(16, (1 - pend)) - pow(13, (1 - pend))) / 14.5;
  *(*(nsimfww + 4) + 4) = cnorm / (1 - pend) * (pow(17, (1 - pend)) - pow(16, (1 - pend))) / 16.5;
  *(*(nsimfww + 4) + 5) = cnorm / (1 - pend) * (pow(18, (1 - pend)) - pow(17, (1 - pend))) / 17.5;
  *(*(nsimfww + 4) + 6) = cnorm / (1 - pend) * (pow(20, (1 - pend)) - pow(18, (1 - pend))) / 19.0;
  *(*(nsimfww + 4) + 7) = cnorm / (1 - pend) * (pow(23, (1 - pend)) - pow(20, (1 - pend))) / 21.5;
  *(*(nsimfww + 4) + 8) = cnorm / (1 - pend) * (pow(27, (1 - pend)) - pow(23, (1 - pend))) / 25.0;
  *(*(nsimfww + 4) + 9) = cnorm / (1 - pend) * (pow(32, (1 - pend)) - pow(27, (1 - pend))) / 29.5;
  *(*(nsimfww + 4) + 10) = cnorm / (1 - pend) * (pow(36, (1 - pend)) - pow(32, (1 - pend))) / 34.0;
  *(*(nsimfww + 4) + 11) = cnorm / (1 - pend) * (pow(40, (1 - pend)) - pow(36, (1 - pend))) / 38.0;

}
#endif /* end of  CS_SNII */

#endif /* end of CS_MODEL */
