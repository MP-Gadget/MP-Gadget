#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "allvars.h"
#include "proto.h"

#include "cooling.h"

#if defined(RT_COOLING_PHOTOHEATING)

static double c_light;
static double HeatH, LambdaH;
static double HeatHe, LambdaHe;

/* rate1 : photoheating for a blackbody spectrum */
/* rate2 : recombination cooling rate */
/* rate3 : collisional ionization cooling rate */
/* rate4 : collisional excitation cooling rate */
/* rate5 : Bremsstrahlung cooling rate */

double radtransfer_cooling_photoheating(int i, double dt)
{
  double a3inv;
  double du;

  if(All.ComovingIntegrationOn)
    a3inv = 1 / (All.Time * All.Time * All.Time);
  else
    a3inv = 1.0;

  rt_get_rates(i);
  
  du = (HeatH - LambdaH) / (SphP[i].d.Density * a3inv);
  du += (HeatHe - LambdaHe) / (SphP[i].d.Density * a3inv);

  return du * dt;
}

void rt_get_rates(int i)
{
  int j;
  double temp, molecular_weight;
  double e1, sigma1;
  double eV_to_erg = 1.60184e-12;
  double dt, a3inv;
  double nH, nHe;
  double rate1, rate2, rate3, rate4, rate5;
  double rateHe1, rateHe2, rateHe3, rateHe4, rateHe5;
  double de1, de2, de3, de4, de5;
  double deHe1, deHe2, deHe3, deHe4, deHe5;
  double E, dE;
  double x, y;

  if(All.ComovingIntegrationOn)
    a3inv = 1 / All.Time / All.Time / All.Time;
  else
    a3inv = 1;

  dt = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval;
  c_light = C / All.UnitVelocity_in_cm_per_s;

  nH = (HYDROGEN_MASSFRAC * SphP[i].d.Density * a3inv) / (PROTONMASS / All.UnitMass_in_g * All.HubbleParam);	//physical
  nHe = ((1.0 - HYDROGEN_MASSFRAC) * SphP[i].d.Density * a3inv) / (4.0 * PROTONMASS / All.UnitMass_in_g * All.HubbleParam);  

  molecular_weight = 4 / (1 + 3 * HYDROGEN_MASSFRAC + 4 * HYDROGEN_MASSFRAC * SphP[i].n_elec);

  temp = (SphP[i].Entropy + SphP[i].e.DtEntropy * dt) * pow(SphP[i].d.Density * a3inv, GAMMA_MINUS1) *
    (molecular_weight * PROTONMASS / All.UnitMass_in_g * All.HubbleParam) /
    (BOLTZMANN / All.UnitEnergy_in_cgs * All.HubbleParam);

  dE = (end_E - start_E) / (float)N_BINS;

  de1 = 0.0;
  deHe1 = 0.0;
  for(j = 0; j < N_BINS; j++)
    {
      x = SphP[i].nHI * nH * rt_sigma_HI[j] / 
        (SphP[i].nHI * nH * rt_sigma_HI[j] + SphP[i].nHeI * nHe * rt_sigma_HeI[j] + SphP[i].nHeII * nHe * rt_sigma_HeII[j]);
      
      y = SphP[i].nHeI * nHe * rt_sigma_HeI[j] / 
	(SphP[i].nHI * nH * rt_sigma_HI[j] + SphP[i].nHeI * nHe * rt_sigma_HeI[j] + SphP[i].nHeII * nHe * rt_sigma_HeII[j]);
      
      E = start_E + (j + 0.5) * dE;
      
      /*photoheating */
      sigma1 = rt_sigma_HI[j];
      e1 = E * eV_to_erg / All.UnitEnergy_in_cgs * All.HubbleParam;
      rate1 = c_light * e1 * sigma1;
      de1 += SphP[i].nHI * nH * x * SphP[i].n_gamma[j] / P[i].Mass * a3inv * rate1;

      sigma1 = rt_sigma_HeI[j];
      e1 = E * eV_to_erg / All.UnitEnergy_in_cgs * All.HubbleParam;
      rateHe1 = c_light * e1 * sigma1;
      deHe1 += SphP[i].nHeI * nHe * y * SphP[i].n_gamma[j] / P[i].Mass * a3inv * rateHe1;

      sigma1 = rt_sigma_HeII[j];
      e1 = E * eV_to_erg / All.UnitEnergy_in_cgs * All.HubbleParam;
      rateHe1 = c_light * e1 * sigma1;
      deHe1 += SphP[i].nHeI * nHe * (1.0 - x - y) * SphP[i].n_gamma[j] / P[i].Mass * a3inv * rateHe1;
    }
  
  /* all rates in erg cm^3 s^-1 in code units */
  /* recombination cooling rate */
  rate2 = 8.7e-27 * pow(temp, 0.5) * pow(temp / 1e3, -0.2) / (1.0 + pow(temp / 1e6, 0.7));
  rate2 *= All.UnitTime_in_s / All.UnitLength_in_cm / All.UnitLength_in_cm / All.UnitLength_in_cm;
  rate2 *= All.HubbleParam * All.HubbleParam;
  rate2 /= All.UnitEnergy_in_cgs / All.HubbleParam;
  de2 = SphP[i].nHII * nH * SphP[i].n_elec * nH * rate2;

  /* collisional ionization cooling rate */
  rate3 = 1.27e-21 * pow(temp, 0.5) * exp(-157809.1 / temp) / (1.0 + pow(temp / 1e5, 0.5));
  rate3 *= All.UnitTime_in_s / All.UnitLength_in_cm / All.UnitLength_in_cm / All.UnitLength_in_cm;
  rate3 *= All.HubbleParam * All.HubbleParam;
  rate3 /= All.UnitEnergy_in_cgs / All.HubbleParam;
  de3 = SphP[i].nHI * nH * SphP[i].n_elec * nH * rate3;

  /* collisional excitation cooling rate */
  rate4 = 7.5e-19 / (1.0 + pow(temp / 1e5, 0.5)) * exp(-118348 / temp);
  rate4 *= All.UnitTime_in_s / All.UnitLength_in_cm / All.UnitLength_in_cm / All.UnitLength_in_cm;
  rate4 *= All.HubbleParam * All.HubbleParam;
  rate4 /= All.UnitEnergy_in_cgs / All.HubbleParam;
  de4 = SphP[i].nHI * nH * SphP[i].n_elec * nH * rate4;

  /* Bremsstrahlung cooling rate */
  rate5 = 1.42e-27 * pow(temp, 0.5);
  rate5 *= All.UnitTime_in_s / All.UnitLength_in_cm / All.UnitLength_in_cm / All.UnitLength_in_cm;
  rate5 *= All.HubbleParam * All.HubbleParam;
  rate5 /= All.UnitEnergy_in_cgs / All.HubbleParam;
  de5 = SphP[i].nHII * nH * SphP[i].n_elec * nH * rate5;

  HeatH = de1;
  LambdaH = de2 + de3 + de4 + de5;

  /* recombination cooling rate */
  rateHe2 = 1.55e-26 * pow(temp, 0.3647);
  rateHe2 *= All.UnitTime_in_s / All.UnitLength_in_cm / All.UnitLength_in_cm / All.UnitLength_in_cm;
  rateHe2 *= All.HubbleParam * All.HubbleParam;
  rateHe2 /= All.UnitEnergy_in_cgs / All.HubbleParam;
  deHe2 = SphP[i].nHeII * nHe * SphP[i].n_elec * nH * rateHe2;

  rateHe2 = 3.48e-26 * pow(temp, 0.5) * pow(temp / 1e3, -0.2) / (1.0 + pow(temp / 1e6, 0.7));
  rateHe2 *= All.UnitTime_in_s / All.UnitLength_in_cm / All.UnitLength_in_cm / All.UnitLength_in_cm;
  rateHe2 *= All.HubbleParam * All.HubbleParam;
  rateHe2 /= All.UnitEnergy_in_cgs / All.HubbleParam;
  deHe2 += SphP[i].nHeIII * nHe * SphP[i].n_elec * nH * rateHe2;

  /* collisional ionization cooling rate */
  rateHe3 = 9.38e-22 * pow(temp, 0.5) * exp(-285335.4 / temp) / (1.0 + pow(temp / 1e5, 0.5));
  rateHe3 *= All.UnitTime_in_s / All.UnitLength_in_cm / All.UnitLength_in_cm / All.UnitLength_in_cm;
  rateHe3 *= All.HubbleParam * All.HubbleParam;
  rateHe3 /= All.UnitEnergy_in_cgs / All.HubbleParam;
  deHe3 = SphP[i].nHeI * nHe * SphP[i].n_elec * nH * rateHe3;

  rateHe3 = 4.95e-22 * pow(temp, 0.5) * exp(-631515 / temp) / (1.0 + pow(temp / 1e5, 0.5));
  rateHe3 *= All.UnitTime_in_s / All.UnitLength_in_cm / All.UnitLength_in_cm / All.UnitLength_in_cm;
  rateHe3 *= All.HubbleParam * All.HubbleParam;
  rateHe3 /= All.UnitEnergy_in_cgs / All.HubbleParam;
  deHe3 += SphP[i].nHeII * nHe * SphP[i].n_elec * nH * rateHe3;

  /* collisional excitation cooling rate */
  rateHe4 = 5.54e-17 * pow(temp, -0.397) / (1.0 + pow(temp / 1e5, 0.5)) * exp(-473638 / temp);
  rateHe4 *= All.UnitTime_in_s / All.UnitLength_in_cm / All.UnitLength_in_cm / All.UnitLength_in_cm;
  rateHe4 *= All.HubbleParam * All.HubbleParam;
  rateHe4 /= All.UnitEnergy_in_cgs / All.HubbleParam;
  deHe4 = SphP[i].nHeII * nHe * SphP[i].n_elec * nH * rateHe4;

  rateHe4 = 9.10e-27 * pow(temp, -0.1687) / (1.0 + pow(temp / 1e5, 0.5)) * exp(-13179 / temp);
  rateHe4 *= All.UnitTime_in_s / All.UnitLength_in_cm / All.UnitLength_in_cm / All.UnitLength_in_cm;
  rateHe4 *= 1.0 / All.UnitLength_in_cm / All.UnitLength_in_cm / All.UnitLength_in_cm;
  rateHe4 *= All.HubbleParam * All.HubbleParam * All.HubbleParam * All.HubbleParam * All.HubbleParam;
  rateHe4 /= All.UnitEnergy_in_cgs / All.HubbleParam;
  deHe4 += SphP[i].nHeII * nHe * SphP[i].n_elec * nH * SphP[i].n_elec * nH * rateHe4;

  /* Bremsstrahlung cooling rate */
  rateHe5 = 1.42e-27 * pow(temp, 0.5);
  rateHe5 *= All.UnitTime_in_s / All.UnitLength_in_cm / All.UnitLength_in_cm / All.UnitLength_in_cm;
  rateHe5 *= All.HubbleParam * All.HubbleParam;
  rateHe5 /= All.UnitEnergy_in_cgs / All.HubbleParam;
  deHe5 =
    (SphP[i].nHeII * nHe * SphP[i].n_elec * nH + 4.0 * SphP[i].nHeIII * nHe * SphP[i].n_elec * nH) * rateHe5;

  HeatHe = deHe1;
  LambdaHe = deHe2 + deHe3 + deHe4 + deHe5;
}

#endif
