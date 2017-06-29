#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>

#include "driftfac.h"
#include "cosmology.h"
#include "endrun.h"


#define  GAMMA_MINUS1  (2.0/3.0)
#define DRIFT_TABLE_LENGTH  1000	/*!< length of the lookup table used to hold the drift and kick factors */

static double logTimeInit;
static double logTimeMax;
static double logDTime;

/*! table for the cosmological drift factors */
static double DriftTable[DRIFT_TABLE_LENGTH];

/*! table for the cosmological kick factor for gravitational forces */
static double GravKickTable[DRIFT_TABLE_LENGTH];

/*! table for the cosmological kick factor for hydrodynmical forces */
static double HydroKickTable[DRIFT_TABLE_LENGTH];


double drift_integ(double a, void *param)
{
  double h;

  h = hubble_function(a);

  return 1 / (h * a * a * a);
}

double gravkick_integ(double a, void *param)
{
  double h;

  h = hubble_function(a);

  return 1 / (h * a * a);
}


double hydrokick_integ(double a, void *param)
{
  double h;

  h = hubble_function(a);

  return 1 / (h * pow(a, 3 * GAMMA_MINUS1) * a);
}

void init_drift_table(double timeBegin, double timeMax, int timebase)
{
    logDTime = (log(timeMax) - log(timeBegin)) / timebase;

#define WORKSIZE 100000
  int i;
  double result, abserr;

  gsl_function F;
  gsl_integration_workspace *workspace;

  logTimeInit = log(timeBegin);
  logTimeMax = log(timeMax);
  if(logTimeMax <=logTimeInit)
      endrun(1,"Error: Invalid drift table range: (%d->%d)\n", timeBegin, timeMax);
  if(timebase <= 0)
      endrun(1,"Error: Invalid timebase: %d\n", timebase);

  workspace = gsl_integration_workspace_alloc(WORKSIZE);

  for(i = 0; i < DRIFT_TABLE_LENGTH; i++)
    {
      F.function = &drift_integ;
      gsl_integration_qag(&F, exp(logTimeInit),
			  exp(logTimeInit + ((logTimeMax - logTimeInit) / DRIFT_TABLE_LENGTH) * (i + 1)), 0,
			  1.0e-8, WORKSIZE, GSL_INTEG_GAUSS41, workspace, &result, &abserr);
      DriftTable[i] = result;


      F.function = &gravkick_integ;
      gsl_integration_qag(&F, exp(logTimeInit),
			  exp(logTimeInit + ((logTimeMax - logTimeInit) / DRIFT_TABLE_LENGTH) * (i + 1)), 0,
			  1.0e-8, WORKSIZE, GSL_INTEG_GAUSS41, workspace, &result, &abserr);
      GravKickTable[i] = result;


      F.function = &hydrokick_integ;
      gsl_integration_qag(&F, exp(logTimeInit),
			  exp(logTimeInit + ((logTimeMax - logTimeInit) / DRIFT_TABLE_LENGTH) * (i + 1)), 0,
			  1.0e-8, WORKSIZE, GSL_INTEG_GAUSS41, workspace, &result, &abserr);
      HydroKickTable[i] = result;

    }
  gsl_integration_workspace_free(workspace);
}

/*Find which bin in the table we are looking up.
 * Pointer argument gives the full floating point value for interpolation.*/
int find_bin_number(int time0, double *rem)
{
  double a1 = logTimeInit + time0 * logDTime;
  double u1;
  int i1;
  u1 = (a1 - logTimeInit) / (logTimeMax - logTimeInit) * DRIFT_TABLE_LENGTH;
  i1 = (int) u1;
  /*Bound u1*/
  if(i1 >= DRIFT_TABLE_LENGTH)
    i1 = DRIFT_TABLE_LENGTH - 1;
  if(i1 <=1)
      i1=1;
  *rem = u1;
  return i1;
}


/*! This function integrates the cosmological prefactor for a drift
 *   step between time0 and time1. The value returned is
 *  \f[ \int_{a_0}^{a_1} \frac{{\rm d}a}{H(a)}
 *  \f]
 *  
 *  A lookup-table is used for reasons of speed. 
 */
static int df_last_time0 = -1, df_last_time1 = -1;
static double df_last_value;
#pragma omp threadprivate(df_last_time0, df_last_time1, df_last_value)
double get_drift_factor(int time0, int time1)
{
  double df1, df2, u1, u2;
  int i1, i2;
  if(time0 == df_last_time0 && time1 == df_last_time1)
    return df_last_value;

  /* note: will only be called for cosmological integration */

  i1 = find_bin_number(time0, &u1);
  if(i1 <= 1)
    df1 = u1 * DriftTable[0];
  else
    df1 = DriftTable[i1 - 1] + (DriftTable[i1] - DriftTable[i1 - 1]) * (u1 - i1);

  i2 = find_bin_number(time1, &u2);
  if(i2 <= 1)
    df2 = u2 * DriftTable[0];
  else
    df2 = DriftTable[i2 - 1] + (DriftTable[i2] - DriftTable[i2 - 1]) * (u2 - i2);

  df_last_time0 = time0;
  df_last_time1 = time1;

  return df_last_value = (df2 - df1);
}

static int gk_last_time0 = -1, gk_last_time1 = -1;
static double gk_last_value;
#pragma omp threadprivate(gk_last_time0, gk_last_time1, gk_last_value)
double get_gravkick_factor(int time0, int time1)
{
  double df1, df2, u1, u2;
  int i1, i2;

  if(time0 == gk_last_time0 && time1 == gk_last_time1)
    return gk_last_value;

  /* note: will only be called for cosmological integration */
  i1 = find_bin_number(time0, &u1);
  if(i1 <= 1)
    df1 = u1 * GravKickTable[0];
  else
    df1 = GravKickTable[i1 - 1] + (GravKickTable[i1] - GravKickTable[i1 - 1]) * (u1 - i1);

  i2 = find_bin_number(time1, &u2);
  if(i2 <= 1)
    df2 = u2 * GravKickTable[0];
  else
    df2 = GravKickTable[i2 - 1] + (GravKickTable[i2] - GravKickTable[i2 - 1]) * (u2 - i2);

  gk_last_time0 = time0;
  gk_last_time1 = time1;

  return gk_last_value = (df2 - df1);
}

static int hk_last_time0 = -1, hk_last_time1 = -1;
static double hk_last_value;
#pragma omp threadprivate(hk_last_time0, hk_last_time1, hk_last_value)
double get_hydrokick_factor(int time0, int time1)
{
  double df1, df2,u1,u2;
  int i1, i2;

  if(time0 == hk_last_time0 && time1 == hk_last_time1)
    return hk_last_value;

  /* note: will only be called for cosmological integration */

  i1 = find_bin_number(time0, &u1);
  if(i1 <= 1)
    df1 = u1 * HydroKickTable[0];
  else
    df1 = HydroKickTable[i1 - 1] + (HydroKickTable[i1] - HydroKickTable[i1 - 1]) * (u1 - i1);

  i2 = find_bin_number(time1, &u2);
  if(i2 <= 1)
    df2 = u2 * HydroKickTable[0];
  else
    df2 = HydroKickTable[i2 - 1] + (HydroKickTable[i2] - HydroKickTable[i2 - 1]) * (u2 - i2);

  hk_last_time0 = time0;
  hk_last_time1 = time1;

  return hk_last_value = (df2 - df1);
}
