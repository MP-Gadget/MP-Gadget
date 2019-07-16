#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>

#include "allvars.h"
#include "timefac.h"
#include "timebinmgr.h"
#include "cosmology.h"
#include "utils.h"

#define WORKSIZE 10000
/*!< length of the lookup table used to hold the drift and kick factors */
#define DRIFT_TABLE_LENGTH 2000

static double logTimeInit;
static double logTimeMax;

/*! table for the cosmological drift factors */
static double * DriftTable;

/*! table for the cosmological kick factor for gravitational forces */
static double * GravKickTable;

/*! table for the cosmological kick factor for hydrodynmical forces */
static double * HydroKickTable;

/* Simple single-value cache for the drift table,
 * which is the same for all particles*/
static inttime_t df_last_ti0 = -1, df_last_ti1 = -1;
static double df_last_value;
#pragma omp threadprivate(df_last_ti0, df_last_ti1, df_last_value)

static inttime_t hk_last_ti0 = -1, hk_last_ti1 = -1;
static double hk_last_value;
#pragma omp threadprivate(hk_last_ti0, hk_last_ti1, hk_last_value)

static inttime_t gk_last_ti0 = -1, gk_last_ti1 = -1;
static double gk_last_value;
#pragma omp threadprivate(gk_last_ti0, gk_last_ti1, gk_last_value)

/* Integrand for the drift table*/
static double drift_integ(double a, void *param)
{
  double h;
  h = hubble_function(a);
  return 1 / (h * a * a * a);
}

/* Integrand for the gravkick table*/
static double gravkick_integ(double a, void *param)
{
  double h;

  h = hubble_function(a);

  return 1 / (h * a * a);
}

/* Integrand for the hydrokick table.
 * Note this is the same function as drift.*/
static double hydrokick_integ(double a, void *param)
{
  double h;

  h = hubble_function(a);

  return 1 / (h * pow(a, 3 * GAMMA_MINUS1) * a);
}

/*Do the integral required to get a factor.*/
static double get_exact_factor(inttime_t t0, inttime_t t1, double (*factor) (double, void *))
{
    double result, abserr;
    if(t0 == t1)
        return 0;
    double a0 = exp(loga_from_ti(t0));
    double a1 = exp(loga_from_ti(t1));
    gsl_function F;
    gsl_integration_workspace *workspace;
    workspace = gsl_integration_workspace_alloc(WORKSIZE);
    F.function = factor;
    gsl_integration_qag(&F, a0, a1, 0, 1.0e-8, WORKSIZE, GSL_INTEG_GAUSS61, workspace, &result, &abserr);
    gsl_integration_workspace_free(workspace);
    return result;
}

/*Get the exact drift factor*/
double get_exact_drift_factor(inttime_t ti0, inttime_t ti1)
{
    return get_exact_factor(ti0, ti1, &drift_integ);
}

void init_drift_table(double timeBegin, double timeMax)
{
  int i;
  double result, abserr;

  GravKickTable = mymalloc("gravkicktable", sizeof(double) * DRIFT_TABLE_LENGTH);
  HydroKickTable = mymalloc("hydrokicktable", sizeof(double) * DRIFT_TABLE_LENGTH);
  //DriftTable = mymalloc("drifttable", sizeof(double) * DRIFT_TABLE_LENGTH);

  gsl_function F;
  gsl_integration_workspace *workspace;

  logTimeInit = log(timeBegin);
  logTimeMax = log(timeMax);
  if(logTimeMax <=logTimeInit)
      endrun(1,"Error: Invalid drift table range: (%d->%d)\n", timeBegin, timeMax);

  workspace = gsl_integration_workspace_alloc(WORKSIZE);

  for(i = 0; i < DRIFT_TABLE_LENGTH; i++)
    {
        /*F.function = &drift_integ;
        gsl_integration_qag(&F, exp(logTimeInit),
            exp(logTimeInit + ((logTimeMax - logTimeInit) / DRIFT_TABLE_LENGTH) * (i + 1)), 0,
                1.0e-8, WORKSIZE, GSL_INTEG_GAUSS41, workspace, &result, &abserr);
        DriftTable[i] = result;*/

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
  df_last_ti0 = df_last_ti1 = gk_last_ti0 = gk_last_ti1 = hk_last_ti0 = hk_last_ti1 = -1;
}

/*Find which bin in the table we are looking up.
 * Pointer argument gives the full floating point value for interpolation.*/
static int find_bin_number(inttime_t ti0, double *rem)
{
  double a1 = loga_from_ti(ti0);
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


/*! This function looks up a cosmological prefactor in
 * a drift/kick table for a step between ti0 and ti1.
 * The value returned is
 *  \f[ \int_{a_0}^{a_1} \frac{{\rm d}a}{H(a) a^3}
 *  \f]
 */
static double get_cached_kick_factor(const inttime_t ti0, const inttime_t ti1, const double * table)
{
  double df1, df2, u1, u2;
  int i1, i2;

  /* note: will only be called for cosmological integration */
  i1 = find_bin_number(ti0, &u1);
  if(i1 <= 1)
    df1 = u1 * table[0];
  else
    df1 = table[i1 - 1] + (table[i1] - table[i1 - 1]) * (u1 - i1);

  i2 = find_bin_number(ti1, &u2);
  if(i2 <= 1)
    df2 = u2 * table[0];
  else
    df2 = table[i2 - 1] + (table[i2] - table[i2 - 1]) * (u2 - i2);

  return df2 - df1;
}

/*! This function integrates the cosmological prefactor for a drift
 *   step between ti0 and ti1. The value returned is
 *  \f[ \int_{a_0}^{a_1} \frac{{\rm d}a}{H(a) a^3}
 *  \f]
 *
 *  A cached value is used for reasons of speed.
 */
double get_drift_factor(inttime_t ti0, inttime_t ti1)
{
  if(ti0 == df_last_ti0 && ti1 == df_last_ti1)
    return df_last_value;

  df_last_ti0 = ti0;
  df_last_ti1 = ti1;

  if(!DriftTable)
    endrun(1, "Drift table not allocated!\n");
  df_last_value = get_cached_kick_factor(ti0, ti1, DriftTable);

  return df_last_value;
}

/*! This function looks up a cosmological prefactor in
 * a drift/kick table for a step between ti0 and ti1.
 * The value returned is
 *  \f[ \int_{a_0}^{a_1} \frac{{\rm d}a}{H(a) a^3}
 *  \f]
 */
double get_gravkick_factor(inttime_t ti0, inttime_t ti1)
{
  if(ti0 == ti1)
      return 0;
  if(ti0 == gk_last_ti0 && ti1 == gk_last_ti1)
    return gk_last_value;

  gk_last_ti0 = ti0;
  gk_last_ti1 = ti1;

  gk_last_value = get_cached_kick_factor(ti0, ti1, GravKickTable);
  return gk_last_value;
}

double get_hydrokick_factor(inttime_t ti0, inttime_t ti1)
{
  if(ti0 == ti1)
      return 0;
  if(ti0 == hk_last_ti0 && ti1 == hk_last_ti1)
    return hk_last_value;

  hk_last_ti0 = ti0;
  hk_last_ti1 = ti1;

  hk_last_value = get_cached_kick_factor(ti0, ti1, HydroKickTable);

  return hk_last_value;
}
