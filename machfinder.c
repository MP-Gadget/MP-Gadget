#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "allvars.h"
#include "proto.h"
#ifdef COSMIC_RAYS
#include "cosmic_rays.h"
#endif

#ifdef MACHNUM

#include "machfinder.h"

#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_sf_gamma.h>

#ifndef DEBUG
#define NDEBUG
#endif
#include <assert.h>

extern double hubble_a, atime, hubble_a2, fac_mu, fac_vsic_fix, a3inv, fac_egy;

/* Part used for the Mach number finder (by Christoph Pfrommer): */


static double Mestimate;

inline double SQR(double a)
{
  return a * a;
}

inline double mycube(double A)
{
  return (A * A * A);
}

inline double myfmin(double A, double B)
{
  if(A < B)
    {
      return A;
    }
  else
    {
      return B;
    }
}

/*********************************************************/
/**************    1D Mach number finder    **************/
/*********************************************************/

/* --- function EntropieJump --- */
double EntropyJump(double M)
{
  double f;

  f = (1 + GAMMA * (2 * M * M - 1)) / (GAMMA + 1) *
    pow(((GAMMA - 1) * M * M + 2) / ((GAMMA + 1) * M * M), GAMMA);
  return (f);
}

/* --- end of function EntropieJump --- */

/* --- function DEntropieJumpDM: derivative of EntropieJump with respect to to M --- */
double DEntropyJumpDM(double M)
{
  double f;

  f = 4 * (GAMMA - 1) * GAMMA * SQR(M * M - 1) / (2 * (GAMMA + 1) * M + (SQR(GAMMA) - 1) * M * M * M) *
    pow(((GAMMA - 1) * M * M + 2) / ((GAMMA + 1) * M * M), GAMMA);
  return (f);
}

/* --- end of function DEntropieJumpDM --- */

/* --- function AuxMach --- */
void AuxMach(double M, double *f, double *df)
{
  double DelS;

  DelS = EntropyJump(M);

  *f = (DelS - 1) * M;
  *df = DEntropyJumpDM(M) * M + DelS - 1;
}

/* --- end of function AuxMach --- */

/* Semianalytic recalibration of Mach number acording to Shock tube simulations.
 * Possible reason: Overshooting of SPH particles and thus incorrect estimation of
 * preshock quantities (rho, c_s, ...)!
 */

/* --- function MachNumberCalibration --- */
double MachNumberCalibration(double M)
{
  if(M > 3.)
    {
      M = (0.0903273 * pow(M, 1.34114) + 1.66215 * exp(-M / 3.)) * M;
    }

  return (M);
}

/* --- end of function MachNumberCalibration --- */

#ifndef CS_MODEL

/*  function MachNumber: computes the Mach number per SPH particle given 
 *  the SPH smoothing length, the local sound speed, the total entropic function A, 
 *  and the instantaneous rate of entropy injection due to shocks dA/dt.
 *  It uses the Newton-Raphson method to find the root of following equation:
 *  [f(M) - 1] * M = h / (c_sound * A) * dA/dt,
 *  where f(M) = A_2 / A_1 (ratio of post-shock to pre-shock entropic function)
 */

/* --- function MachNumber: for the thermal gas without cosmic rays --- */
void GetMachNumber(struct sph_particle_data *Particle)
{
  double M, dM, r;
  double f, df, rhs, fac_hsml, csnd, rho1;
  int i = 0;

#if ( CR_SHOCK != 2 )
  double DeltaDecayTime;
#endif

  int index = Particle - SphP;

  /* set physical pre-shock density: */
  rho1 = Particle->d.Density / (atime * atime * atime);

  /* sound velocity (in physical units): */
  csnd = sqrt(GAMMA * Particle->Pressure / Particle->d.Density) * pow(atime, -3. / 2. * GAMMA_MINUS1);

  fac_hsml = All.Shock_Length;
  rhs = fac_hsml * PPP[index].Hsml * atime / (csnd * Particle->Entropy) * Particle->e.DtEntropy * hubble_a;



  /* initial guess */
  if(rhs > 0)
    M = 2.;
  else
    M = 0.5;

  /* Newton-Raphson scheme */
  r = 1.0;
  while((i < 100) && (r > 1e-4))
    {
      AuxMach(M, &f, &df);
      dM = (f - rhs) / df;
      M -= dM;
      r = fabs(dM / M);
      i++;
    }
  if(i == 100)
    printf("Error: too many iterations in function MachNumber!\n");

  Mestimate = M;

  /* The following part is only for thermal gas, calibration for the Mach number 
   * of a CR+th gas is done in function MachNumberCR!
   */

#if ( CR_SHOCK != 2 )

  /* Semianalytic recalibration of Mach number acording to Shock tube simulations. */
  M = MachNumberCalibration(M);

  /* Introduce decay time of the shock meanwhile the Particle Mach number will not be 
   * updated! Mach numbers of all particles are initialized in init.c with M=1!
   */
  if(M > Particle->Shock_MachNumber)
    {
      Particle->Shock_MachNumber = M;
      DeltaDecayTime = fac_hsml * Particle->Hsml * atime / (Mestimate * csnd);

      /* convert (Delta t_physical) -> (Delta log a): */
      DeltaDecayTime *= hubble_a;
      DeltaDecayTime = myfmin(DeltaDecayTime, All.Shock_DeltaDecayTimeMax);

      if(All.ComovingIntegrationOn)
	{
	  Particle->Shock_DecayTime = All.Time * (1. + DeltaDecayTime);
	}
      else
	{
	  Particle->Shock_DecayTime = All.Time + DeltaDecayTime;
	}
    }
  else
    {
      if(All.Time > Particle->Shock_DecayTime)
	{
	  Particle->Shock_MachNumber = M;
#ifdef OUTPUT_PRESHOCK_CSND
	  Particle->PreShock_PhysicalSoundSpeed = csnd;
	  Particle->PreShock_PhysicalDensity = rho1;
#endif
	}
    }

#endif

  return;
}

/* --- end of function MachNumber --- */

/*********************************************************/
/************** end of 1D Mach number finder *************/
/*********************************************************/



#else

/*********************************************** CS MODEL ***********************************************/


/* This is the end of the "usual Mach finder", now rewite particle 
   structure to match requirment's of Cecilia's model */


/* --- function MachNumber: for the thermal gas without cosmic rays --- */
void GetMachNumber(struct sph_particle_data *Particle, struct particle_data *Particle_Hsml)
{
  double M, dM, r;
  double f, df, rhs, fac_hsml, csnd, rho1;
  int i = 0;

#if ( CR_SHOCK != 2 )
  double DeltaDecayTime;
#endif

  /* set physical pre-shock density: */
  rho1 = Particle->d.Density / (atime * atime * atime);

  /* sound velocity (in physical units): */
  csnd = sqrt(GAMMA * Particle->Pressure / Particle->d.Density) * pow(atime, -3. / 2. * GAMMA_MINUS1);

  fac_hsml = All.Shock_Length;
  rhs =
    fac_hsml * Particle_Hsml->Hsml * atime / (csnd * Particle->Entropy) * Particle->e.DtEntropy * hubble_a;

  /* initial guess */
  if(rhs > 0)
    M = 2.;
  else
    M = 0.5;

  /* Newton-Raphson scheme */
  r = 1.0;
  while((i < 100) && (r > 1e-4))
    {
      AuxMach(M, &f, &df);
      dM = (f - rhs) / df;
      M -= dM;
      r = fabs(dM / M);
      i++;
    }
  if(i == 100)
    printf("Error: too many iterations in function MachNumber!\n");

  Mestimate = M;

  /* The following part is only for thermal gas, calibration for the Mach number 
   * of a CR+th gas is done in function MachNumberCR!
   */

#if ( CR_SHOCK != 2 )

  /* Semianalytic recalibration of Mach number acording to Shock tube simulations. */
  M = MachNumberCalibration(M);

  /* Introduce decay time of the shock meanwhile the Particle Mach number will not be 
   * updated! Mach numbers of all particles are initialized in init.c with M=1!
   */
  if(M > Particle->Shock_MachNumber)
    {
      Particle->Shock_MachNumber = M;
      DeltaDecayTime = fac_hsml * Particle_Hsml->Hsml * atime / (Mestimate * csnd);

      /* convert (Delta t_physical) -> (Delta log a): */
      DeltaDecayTime *= hubble_a;
      DeltaDecayTime = myfmin(DeltaDecayTime, All.Shock_DeltaDecayTimeMax);

      if(All.ComovingIntegrationOn)
	{
	  Particle->Shock_DecayTime = All.Time * (1. + DeltaDecayTime);
	}
      else
	{
	  Particle->Shock_DecayTime = All.Time + DeltaDecayTime;
	}
    }
  else
    {
      if(All.Time > Particle->Shock_DecayTime)
	{
	  Particle->Shock_MachNumber = M;
#ifdef  OUTPUT_PRESHOCK_CSND
	  Particle->PreShock_PhysicalSoundSpeed = csnd;
	  Particle->PreShock_PhysicalDensity = rho1;
#endif
	}
    }

#endif

  return;
}

/* --- end of function MachNumber --- */


#endif
/* end of rewrite for Cecilia's model */

/*********************************************** CS MODEL ***********************************************/



#ifdef COSMIC_RAYS

/*********************************************************/
/**************    2D Mach number finder    **************/
/*********************************************************/

/* --- function print_state_f: for diagnostic outputs! --- */
void print_state_f(size_t iter, gsl_multiroot_fsolver * s)
{
  printf("iter = %3u, x = % .3f % .3f, "
	 "f(x) = % .3e % .3e\n",
	 iter,
	 gsl_vector_get(s->x, 0), gsl_vector_get(s->x, 1), gsl_vector_get(s->f, 0), gsl_vector_get(s->f, 1));
}

/* --- end of function print_state_f --- */

/* --- function print_state_fdf: for diagnostic outputs! --- */
void print_state_fdf(size_t iter, gsl_multiroot_fdfsolver * s)
{
  printf("iter = %3u, x = % .3f % .3f, "
	 "f(x) = % .3e % .3e\n",
	 iter,
	 gsl_vector_get(s->x, 0), gsl_vector_get(s->x, 1), gsl_vector_get(s->f, 0), gsl_vector_get(s->f, 1));
}

/* --- end of function print_state_fdf --- */

#if ( NUMCRPOP == 1)

/* --- Defining the system of 2 non-linear       --- */
/* --- equations which evince the roots x and z. --- */

/* --- function shock_conditions_fnd             --- */
/* --- only used for the numerical root finder   --- */

int shock_conditions_fnd(const gsl_vector * v, void *auxParticle, gsl_vector * f)
{
  double Pth1, rho1, hsml, Ath1, dAth1dt;
  double PCR1, eCR1, gCR;
  double P1, eps1, c1, g1, A1, dA1dt, a, fac_hsml;
  double xgCR, Pth2, P2, eps2, rho2g2;
  double f0, f1, y;
  double xa, ya, za, fa;
  double XCR, inj_rate;

  const double x = gsl_vector_get(v, 0);
  const double z = gsl_vector_get(v, 1);
  struct sph_particle_data *Particle;

  Particle = (struct sph_particle_data *) auxParticle;
  int index = Particle - SphP;


  /* Generally: comoving -> physical quantities! */
  if(All.ComovingIntegrationOn)
    {
      /* CR definitions: */
      rho1 = Particle->d.Density / mycube(All.Time);
      PCR1 = CR_Particle_Pressure(Particle, 0) * pow(All.Time, -3.0 * GAMMA);
      eCR1 = CR_Particle_SpecificEnergy(Particle, 0) * rho1;
      gCR = Particle->CR_Gamma0[0];

      /* thermal gas definitions: */
      P1 = Particle->Pressure * pow(All.Time, -3.0 * GAMMA);	/* = PCR1 + Pth1 */
      Pth1 = P1 - PCR1;
      hsml = PPP[index].Hsml * All.Time;
      Ath1 = Particle->Entropy;
      dAth1dt = Particle->e.DtEntropy * hubble_a;
    }
  else
    {
      /* CR definitions: */
      rho1 = Particle->d.Density;
      PCR1 = CR_Particle_Pressure(Particle, 0);
      eCR1 = CR_Particle_SpecificEnergy(Particle, 0) * rho1;
      gCR = Particle->CR_Gamma0[0];

      /* thermal gas definitions: */
      P1 = Particle->Pressure;	/* = PCR1 + Pth1 */
      Pth1 = P1 - PCR1;
      hsml = PPP[index].Hsml;
      Ath1 = Particle->Entropy;
      dAth1dt = Particle->e.DtEntropy;
    }

  /* defining upstream quantities: */
  eps1 = eCR1 + 1. / (GAMMA - 1.) * Pth1;
  g1 = (gCR * PCR1 + GAMMA * Pth1) / P1;
  c1 = sqrt(g1 * P1 / rho1);
  A1 = P1 * pow(rho1, -g1);
  dA1dt = dAth1dt * pow(rho1, GAMMA - g1);
  fac_hsml = All.Shock_Length;
  a = fac_hsml * hsml / (c1 * A1) * dA1dt;

  y = 4.0 * z + x;

  if(x >= 1.0 && y >= 1.0)
    {
      /* defining downstream quantities: */
      xgCR = pow(x, gCR);
      Pth2 = (x + 4. * z) * Pth1;
      P2 = PCR1 * xgCR + Pth2;
      eps2 = eCR1 * xgCR + 1. / (GAMMA - 1.) * Pth2;
      rho2g2 = pow(rho1 * x, -(gCR * PCR1 * xgCR + GAMMA * Pth1 * (x + 4. * z)) / P2);

      /* system of nonlinear equations: */
      f0 =
	x * (P2 - P1) * SQR(P2 * rho2g2 - P1 * pow(rho1, -g1)) -
	a * a * c1 * c1 * P1 * P1 * (x - 1.) * pow(rho1, 1. - 2. * g1);

      f1 = 2. * eps2 + P1 + P2 - x * (2. * eps1 + P1 + P2);
    }
  else
    {
      /* idea: attach function x-1 / y-1 to xa=1 / ya=1 in order to push the root finder 
         in the right direction by enforcing the derivative to be equal to 1! */
      if(x < 1.0 && y >= 1.0)
	{
	  xa = 1.0;
	  fa = x - 1.0;

	  /* defining downstream quantities: */
	  xgCR = pow(xa, gCR);
	  Pth2 = (xa + 4. * z) * Pth1;
	  P2 = PCR1 * xgCR + Pth2;
	  eps2 = eCR1 * xgCR + 1. / (GAMMA - 1.) * Pth2;
	  rho2g2 = pow(rho1 * xa, -(gCR * PCR1 * xgCR + GAMMA * Pth1 * (xa + 4. * z)) / P2);

	  /* system of nonlinear equations: */
	  f0 =
	    xa * (P2 - P1) * SQR(P2 * rho2g2 - P1 * pow(rho1, -g1)) -
	    a * a * c1 * c1 * P1 * P1 * (xa - 1.) * pow(rho1, 1. - 2. * g1);

	  f1 = 2. * eps2 + P1 + P2 - xa * (2. * eps1 + P1 + P2);
	}
      else if(y < 1.0 && x >= 1.0)
	{
	  ya = 1.0;
	  za = (ya - x) / 4.;
	  fa = y - 1.0;

	  /* defining downstream quantities: */
	  xgCR = pow(x, gCR);
	  Pth2 = (x + 4. * za) * Pth1;
	  P2 = PCR1 * xgCR + Pth2;
	  eps2 = eCR1 * xgCR + 1. / (GAMMA - 1.) * Pth2;
	  rho2g2 = pow(rho1 * x, -(gCR * PCR1 * xgCR + GAMMA * Pth1 * (x + 4. * za)) / P2);

	  /* system of nonlinear equations: */
	  f0 =
	    x * (P2 - P1) * SQR(P2 * rho2g2 - P1 * pow(rho1, -g1)) -
	    a * a * c1 * c1 * P1 * P1 * (x - 1.) * pow(rho1, 1. - 2. * g1);

	  f1 = 2. * eps2 + P1 + P2 - x * (2. * eps1 + P1 + P2);
	}
      else			/* we have y < 1.0 && x < 1.0 */
	{
	  xa = 1.0;
	  ya = 1.0;
	  za = 0.0;		/* za = (ya - xa) / 4. */
	  fa = x + y - 2.0;

	  /* defining downstream quantities: */
	  xgCR = pow(xa, gCR);
	  Pth2 = (xa + 4. * za) * Pth1;
	  P2 = PCR1 * xgCR + Pth2;
	  eps2 = eCR1 * xgCR + 1. / (GAMMA - 1.) * Pth2;
	  rho2g2 = pow(rho1 * xa, -(gCR * PCR1 * xgCR + GAMMA * Pth1 * (xa + 4. * za)) / P2);

	  /* system of nonlinear equations: */
	  f0 =
	    xa * (P2 - P1) * SQR(P2 * rho2g2 - P1 * pow(rho1, -g1)) -
	    a * a * c1 * c1 * P1 * P1 * (xa - 1.) * pow(rho1, 1. - 2. * g1);

	  f1 = 2. * eps2 + P1 + P2 - xa * (2. * eps1 + P1 + P2);
	}

      f0 += fa;			/* attach the auxiliary functions at the problematic points! */
      f1 += fa;
    }

  gsl_vector_set(f, 0, f0);
  gsl_vector_set(f, 1, f1);

  return GSL_SUCCESS;
}

/* --- end of function shock_conditions_fnd --- */

/* --- function shock_conditions_f          --- */

int shock_conditions_f(const gsl_vector * v, void *auxParticle, gsl_vector * f)
{
  double Pth1, rho1, hsml, Ath1, dAth1dt;
  double PCR1, eCR1, gCR;
  double P1, eps1, c1, g1, A1, dA1dt, a, fac_hsml;
  double xgCR, Pth2, P2, eps2, rho2g2;
  double f0, f1;
  double XCR, inj_rate;

  const double x = gsl_vector_get(v, 0);
  const double z = gsl_vector_get(v, 1);
  struct sph_particle_data *Particle;

  Particle = (struct sph_particle_data *) auxParticle;
  int index = Particle - SphP;

  /* Generally: comoving -> physical quantities! */
  if(All.ComovingIntegrationOn)
    {
      /* CR definitions: */
      rho1 = Particle->d.Density / mycube(All.Time);
      PCR1 = CR_Particle_Pressure(Particle, 0) * pow(All.Time, -3.0 * GAMMA);
      eCR1 = CR_Particle_SpecificEnergy(Particle, 0) * rho1;
      gCR = Particle->CR_Gamma0[0];

      /* thermal gas definitions: */
      P1 = Particle->Pressure * pow(All.Time, -3.0 * GAMMA);	/* = PCR1 + Pth1 */
      Pth1 = P1 - PCR1;
      hsml = PPP[index].Hsml * All.Time;
      Ath1 = Particle->Entropy;
      dAth1dt = Particle->e.DtEntropy * hubble_a;
    }
  else
    {
      /* CR definitions: */
      rho1 = Particle->d.Density;
      PCR1 = CR_Particle_Pressure(Particle, 0);
      eCR1 = CR_Particle_SpecificEnergy(Particle, 0) * rho1;
      gCR = Particle->CR_Gamma0[0];

      /* thermal gas definitions: */
      P1 = Particle->Pressure;	/* = PCR1 + Pth1 */
      Pth1 = P1 - PCR1;
      hsml = PPP[index].Hsml;
      Ath1 = Particle->Entropy;
      dAth1dt = Particle->e.DtEntropy;
    }

  /* defining upstream quantities: */
  eps1 = eCR1 + 1. / (GAMMA - 1.) * Pth1;
  g1 = (gCR * PCR1 + GAMMA * Pth1) / P1;
  c1 = sqrt(g1 * P1 / rho1);
  A1 = P1 * pow(rho1, -g1);
  dA1dt = dAth1dt * pow(rho1, GAMMA - g1);
  fac_hsml = All.Shock_Length;
  a = fac_hsml * hsml / (c1 * A1) * dA1dt;

  /* debugging: */
  if(x < 0.0)
    {
      XCR = PCR1 / Pth1;
      inj_rate = Particle->e.DtEntropy * hubble_a / Particle->Entropy;
      printf("  x < 0.0 in function %s()!\n", __func__);
      printf("  Comoving quantities:\n");
      printf("  #define hubble_a %e  #define atime %e\n", hubble_a, atime);
      printf("  XCR  = %e;  inj_rate = %e;\n", XCR, inj_rate);
      printf("  rho1 = %e;  hsml     = %e;  Ath1 = %e;  dAth1dt = %e;\n",
	     Particle->d.Density, PPP[index].Hsml, Particle->Entropy, Particle->e.DtEntropy);
      printf("  P1   = %e;  PCR1     = %e;  eCR1 = %e;  gCR     = %e;\n\n",
	     Particle->Pressure, PCR1 * pow(atime, 3.0 * GAMMA), eCR1 / rho1, Particle->CR_Gamma0[0]);
      fflush(stdout);
    }
  assert(x >= 0.0);

  /* defining downstream quantities: */
  xgCR = pow(x, gCR);
  Pth2 = (x + 4. * z) * Pth1;
  P2 = PCR1 * xgCR + Pth2;
  eps2 = eCR1 * xgCR + 1. / (GAMMA - 1.) * Pth2;
  rho2g2 = pow(rho1 * x, -(gCR * PCR1 * xgCR + GAMMA * Pth1 * (x + 4. * z)) / P2);

  /* system of nonlinear equations: */
  f0 =
    x * (P2 - P1) * SQR(P2 * rho2g2 - P1 * pow(rho1, -g1)) -
    a * a * c1 * c1 * P1 * P1 * (x - 1.) * pow(rho1, 1. - 2. * g1);

  f1 = 2. * eps2 + P1 + P2 - x * (2. * eps1 + P1 + P2);

  gsl_vector_set(f, 0, f0);
  gsl_vector_set(f, 1, f1);

  return GSL_SUCCESS;
}

/* --- end of function shock_conditions_f --- */

/* --- Defining the Jacobian of the system. --- */
/* --- function shock_conditions_df         --- */
int shock_conditions_df(const gsl_vector * v, void *auxParticle, gsl_matrix * J)
{
  double Pth1, rho1, hsml, Ath1, dAth1dt;
  double PCR1, eCR1, gCR, gth;
  double P1, eps1, c1, g1, A1, dA1dt, a, fac_hsml;
  double xgCR, Pth2, P2, eps2, rho1g1, rho2g2;
  double XCR, inj_rate;

  double df00, df01, df10, df11;
  const double x = gsl_vector_get(v, 0);
  const double z = gsl_vector_get(v, 1);
  struct sph_particle_data *Particle;

  Particle = (struct sph_particle_data *) auxParticle;
  int index = Particle - SphP;

  /* Generally: comoving -> physical quantities! */
  if(All.ComovingIntegrationOn)
    {
      /* CR definitions: */
      rho1 = Particle->d.Density / mycube(All.Time);
      PCR1 = CR_Particle_Pressure(Particle, 0) * pow(All.Time, -3.0 * GAMMA);
      eCR1 = CR_Particle_SpecificEnergy(Particle, 0) * rho1;
      gCR = Particle->CR_Gamma0[0];

      /* thermal gas definitions: */
      P1 = Particle->Pressure * pow(All.Time, -3.0 * GAMMA);	/* = PCR1 + Pth1 */
      Pth1 = P1 - PCR1;
      hsml = PPP[index].Hsml * All.Time;
      Ath1 = Particle->Entropy;
      dAth1dt = Particle->e.DtEntropy * hubble_a;
    }
  else
    {
      /* CR definitions: */
      rho1 = Particle->d.Density;
      PCR1 = CR_Particle_Pressure(Particle, 0);
      eCR1 = CR_Particle_SpecificEnergy(Particle, 0) * Particle->d.Density;;
      gCR = Particle->CR_Gamma0[0];

      /* thermal gas definitions: */
      P1 = Particle->Pressure;	/* = PCR1 + Pth1 */
      Pth1 = P1 - PCR1;
      hsml = PPP[index].Hsml;
      Ath1 = Particle->Entropy;
      dAth1dt = Particle->e.DtEntropy;
    }

  /* defining upstream quantities: */
  gth = GAMMA;
  eps1 = eCR1 + 1. / (GAMMA - 1.) * Pth1;
  g1 = (gCR * PCR1 + GAMMA * Pth1) / P1;
  c1 = sqrt(g1 * P1 / rho1);
  A1 = P1 * pow(rho1, -g1);
  dA1dt = dAth1dt * pow(rho1, GAMMA - g1);
  fac_hsml = All.Shock_Length;
  a = fac_hsml * hsml / (c1 * A1) * dA1dt;

  /* debugging: */
  if(x < 0.0)
    {
      XCR = PCR1 / Pth1;
      inj_rate = Particle->e.DtEntropy * hubble_a / Particle->Entropy;
      printf("  x < 0.0 in function %s()!\n", __func__);
      printf("  Comoving quantities:\n");
      printf("  #define hubble_a %e  #define atime %e\n", hubble_a, atime);
      printf("  XCR  = %e;  inj_rate = %e;\n", XCR, inj_rate);
      printf("  rho1 = %e;  hsml     = %e;  Ath1 = %e;  dAth1dt = %e;\n",
	     Particle->d.Density, PPP[index].Hsml, Particle->Entropy, Particle->e.DtEntropy);
      printf("  P1   = %e;  PCR1     = %e;  eCR1 = %e;  gCR     = %e;\n\n",
	     Particle->Pressure, PCR1 * pow(atime, 3.0 * GAMMA), eCR1 / rho1, Particle->CR_Gamma0[0]);
      fflush(stdout);
    }
  assert(x >= 0.0);

  /* defining downstream quantities: */
  xgCR = pow(x, gCR);
  Pth2 = (x + 4. * z) * Pth1;
  P2 = PCR1 * xgCR + Pth2;
  eps2 = eCR1 * xgCR + 1. / (GAMMA - 1.) * Pth2;
  rho1g1 = pow(rho1, -g1);
  rho2g2 = pow(rho1 * x, -(gCR * PCR1 * xgCR + GAMMA * Pth1 * (x + 4. * z)) / P2);

  /* entries of the Jacobian: */
  df00 =
    -(a * a * c1 * c1 * P1 * P1 * pow(rho1, 1. - 2. * g1)) +
    x * (Pth1 + gCR * PCR1 * pow(x, gCR - 1.)) *
    SQR(P1 * rho1g1 - P2 * rho2g2) +
    (P2 - P1) * SQR(P1 * rho1g1 - P2 * rho2g2) -
    (1. / P2) * 2. * rho1g1 * rho2g2 * rho2g2 * (P2 - P1) *
    (P2 / rho1g1 - P1 / rho2g2) * Pth1 *
    (((gth - 1.) * x + 4. * gth * z) * P2 + (gCR - gth) * PCR1 * xgCR *
     ((gCR - 1.) * x + 4. * gCR * z) * log(rho1 * x));


  df01 =
    4 * Pth1 * x * SQR(P1 * rho1g1 - P2 * rho2g2) +
    (1. / P2) * 8. * Pth1 * rho1g1 * x * rho2g2 * rho2g2 * (P2 - P1) *
    (P2 / rho1g1 - P1 / rho2g2) * (Pth1 * (x + 4. * z) + PCR1 * xgCR * (1 + (gCR - gth) * log(rho1 * x)));


  df10 =
    -2. * eps1 - P1 + pow(x, gCR - 1.) *
    (2. * eCR1 * gCR + PCR1 * (gCR - (1. + gCR) * x)) + Pth1 * (1 + 2. / (gth - 1.) - 2. * x - 4. * z);


  df11 = 4. * Pth1 * (1. + 2. / (gth - 1.) - x);


  gsl_matrix_set(J, 0, 0, df00);
  gsl_matrix_set(J, 0, 1, df01);
  gsl_matrix_set(J, 1, 0, df10);
  gsl_matrix_set(J, 1, 1, df11);

  return GSL_SUCCESS;
}

/* --- end of function shock_conditions_df --- */

#else /* multiple alpha, i.e.e NUMCRPOP > 1 */

/* --- Defining the system of 2 non-linear       --- */
/* --- equations which evince the roots x and z. --- */

/* --- function shock_conditions_fnd             --- */
/* --- only used for the numerical root finder   --- */

int shock_conditions_fnd(const gsl_vector * v, void *auxParticle, gsl_vector * f)
{
  double Pth1, rho1, hsml, Ath1, dAth1dt;
  double PCR1[NUMCRPOP], eCR1[NUMCRPOP], gCR[NUMCRPOP];
  double P1, eps1, c1, g1, A1, dA1dt, a, fac_hsml;
  double xgCR[NUMCRPOP], Pth2, P2, eps2, rho2g2;
  double f0, f1, y;
  double xa, ya, za, fa;
  double XCR, inj_rate;
  int CRpop;
  double PCR1sum, eCR1sum, gCRsum, PCR1_xgCR_sum, gCR_PCR1_xgCR_sum, eCR1_xgCR_sum;

  PCR1sum = 0.0;
  eCR1sum = 0.0;
  gCRsum = 0.0;
  PCR1_xgCR_sum = 0.0;
  gCR_PCR1_xgCR_sum = 0.0;
  eCR1_xgCR_sum = 0.0;

  const double x = gsl_vector_get(v, 0);
  const double z = gsl_vector_get(v, 1);
  struct sph_particle_data *Particle;


  Particle = (struct sph_particle_data *) auxParticle;

  /* Generally: comoving -> physical quantities! */
  if(All.ComovingIntegrationOn)
    {
      /* CR definitions: */
      rho1 = Particle->d.Density / mycube(All.Time);
      for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
	{
	  PCR1[CRpop] = CR_Particle_Pressure(Particle, CRpop) * pow(All.Time, -3.0 * GAMMA);
	  PCR1sum += PCR1[CRpop];
	  eCR1[CRpop] = CR_Particle_SpecificEnergy(Particle, CRpop) * rho1;
	  eCR1sum += eCR1[CRpop];
	  gCR[CRpop] = Particle->CR_Gamma0[CRpop];
	}
      /* thermal gas definitions: */
      P1 = Particle->Pressure * pow(All.Time, -3.0 * GAMMA);	/* = PCR1 + Pth1 */
      Pth1 = P1 - PCR1sum;
      hsml = Particle->Hsml * All.Time;
      Ath1 = Particle->Entropy;
      dAth1dt = Particle->e.DtEntropy * hubble_a;
    }
  else
    {
      /* CR definitions: */
      rho1 = Particle->d.Density;
      for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
	{
	  PCR1[CRpop] = CR_Particle_Pressure(Particle, CRpop);
	  PCR1sum += PCR1[CRpop];
	  eCR1[CRpop] = CR_Particle_SpecificEnergy(Particle, CRpop) * rho1;
	  eCR1sum += eCR1[CRpop];
	  gCR[CRpop] = Particle->CR_Gamma0[CRpop];
	}
      /* thermal gas definitions: */
      P1 = Particle->Pressure;	/* = PCR1 + Pth1 */
      Pth1 = P1 - PCR1sum;
      hsml = Particle->Hsml;
      Ath1 = Particle->Entropy;
      dAth1dt = Particle->e.DtEntropy;
    }

  /* defining upstream quantities: */
  eps1 = eCR1sum + 1. / (GAMMA - 1.) * Pth1;
  for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
    gCRsum += gCR[CRpop] * PCR1[CRpop];
  g1 = (gCRsum + GAMMA * Pth1) / P1;
  c1 = sqrt(g1 * P1 / rho1);
  A1 = P1 * pow(rho1, -g1);
  dA1dt = dAth1dt * pow(rho1, GAMMA - g1);
  fac_hsml = All.Shock_Length;
  a = fac_hsml * hsml / (c1 * A1) * dA1dt;

  y = 4.0 * z + x;

  if(x >= 1.0 && y >= 1.0)
    {
      /* defining downstream quantities: */
      for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
	xgCR[CRpop] = pow(x, gCR[CRpop]);
      Pth2 = (x + 4. * z) * Pth1;
      for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
	{
	  PCR1_xgCR_sum += PCR1[CRpop] * xgCR[CRpop];
	  gCR_PCR1_xgCR_sum += gCR[CRpop] * PCR1[CRpop] * xgCR[CRpop];
	  eCR1_xgCR_sum += eCR1[CRpop] * xgCR[CRpop];
	}
      P2 = PCR1_xgCR_sum + Pth2;
      eps2 = eCR1_xgCR_sum + 1. / (GAMMA - 1.) * Pth2;
      rho2g2 = pow(rho1 * x, -(gCR_PCR1_xgCR_sum + GAMMA * Pth1 * (x + 4. * z)) / P2);

      /* system of nonlinear equations: */
      f0 =
	x * (P2 - P1) * SQR(P2 * rho2g2 - P1 * pow(rho1, -g1)) -
	a * a * c1 * c1 * P1 * P1 * (x - 1.) * pow(rho1, 1. - 2. * g1);

      f1 = 2. * eps2 + P1 + P2 - x * (2. * eps1 + P1 + P2);
    }
  else
    {
      /* idea: attach function x-1 / y-1 to xa=1 / ya=1 in order to push the root finder 
         in the right direction by enforcing the derivative to be equal to 1! */
      if(x < 1.0 && y >= 1.0)
	{
	  xa = 1.0;
	  fa = x - 1.0;

	  /* defining downstream quantities: */
	  for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
	    xgCR[CRpop] = pow(xa, gCR[CRpop]);
	  Pth2 = (xa + 4. * z) * Pth1;
	  for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
	    {
	      PCR1_xgCR_sum += PCR1[CRpop] * xgCR[CRpop];
	      gCR_PCR1_xgCR_sum += gCR[CRpop] * PCR1[CRpop] * xgCR[CRpop];
	      eCR1_xgCR_sum += eCR1[CRpop] * xgCR[CRpop];
	    }
	  P2 = PCR1_xgCR_sum + Pth2;
	  eps2 = eCR1_xgCR_sum + 1. / (GAMMA - 1.) * Pth2;
	  rho2g2 = pow(rho1 * xa, -(gCR_PCR1_xgCR_sum + GAMMA * Pth1 * (xa + 4. * z)) / P2);

	  /* system of nonlinear equations: */
	  f0 =
	    xa * (P2 - P1) * SQR(P2 * rho2g2 - P1 * pow(rho1, -g1)) -
	    a * a * c1 * c1 * P1 * P1 * (xa - 1.) * pow(rho1, 1. - 2. * g1);

	  f1 = 2. * eps2 + P1 + P2 - xa * (2. * eps1 + P1 + P2);
	}
      else if(y < 1.0 && x >= 1.0)
	{
	  ya = 1.0;
	  za = (ya - x) / 4.;
	  fa = y - 1.0;

	  /* defining downstream quantities: */
	  for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
	    xgCR[CRpop] = pow(x, gCR[CRpop]);
	  Pth2 = (x + 4. * za) * Pth1;
	  for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
	    {
	      PCR1_xgCR_sum += PCR1[CRpop] * xgCR[CRpop];
	      gCR_PCR1_xgCR_sum += gCR[CRpop] * PCR1[CRpop] * xgCR[CRpop];
	      eCR1_xgCR_sum += eCR1[CRpop] * xgCR[CRpop];
	    }
	  P2 = PCR1_xgCR_sum + Pth2;
	  eps2 = eCR1_xgCR_sum + 1. / (GAMMA - 1.) * Pth2;
	  rho2g2 = pow(rho1 * x, -(gCR_PCR1_xgCR_sum + GAMMA * Pth1 * (x + 4. * za)) / P2);

	  /* system of nonlinear equations: */
	  f0 =
	    x * (P2 - P1) * SQR(P2 * rho2g2 - P1 * pow(rho1, -g1)) -
	    a * a * c1 * c1 * P1 * P1 * (x - 1.) * pow(rho1, 1. - 2. * g1);

	  f1 = 2. * eps2 + P1 + P2 - x * (2. * eps1 + P1 + P2);
	}
      else			/* we have y < 1.0 && x < 1.0 */
	{
	  xa = 1.0;
	  ya = 1.0;
	  za = 0.0;		/* za = (ya - xa) / 4. */
	  fa = x + y - 2.0;

	  /* defining downstream quantities: */
	  for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
	    xgCR[CRpop] = pow(xa, gCR[CRpop]);
	  Pth2 = (xa + 4. * za) * Pth1;
	  for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
	    {
	      PCR1_xgCR_sum += PCR1[CRpop] * xgCR[CRpop];
	      gCR_PCR1_xgCR_sum += gCR[CRpop] * PCR1[CRpop] * xgCR[CRpop];
	      eCR1_xgCR_sum += eCR1[CRpop] * xgCR[CRpop];
	    }
	  P2 = PCR1_xgCR_sum + Pth2;
	  eps2 = eCR1_xgCR_sum + 1. / (GAMMA - 1.) * Pth2;
	  rho2g2 = pow(rho1 * xa, -(gCR_PCR1_xgCR_sum + GAMMA * Pth1 * (xa + 4. * za)) / P2);

	  /* system of nonlinear equations: */
	  f0 =
	    xa * (P2 - P1) * SQR(P2 * rho2g2 - P1 * pow(rho1, -g1)) -
	    a * a * c1 * c1 * P1 * P1 * (xa - 1.) * pow(rho1, 1. - 2. * g1);

	  f1 = 2. * eps2 + P1 + P2 - xa * (2. * eps1 + P1 + P2);
	}

      f0 += fa;			/* attach the auxiliary functions at the problematic points! */
      f1 += fa;
    }

  gsl_vector_set(f, 0, f0);
  gsl_vector_set(f, 1, f1);

  return GSL_SUCCESS;
}

/* --- end of function shock_conditions_fnd --- */

/* --- function shock_conditions_f          --- */

int shock_conditions_f(const gsl_vector * v, void *auxParticle, gsl_vector * f)
{
  double Pth1, rho1, hsml, Ath1, dAth1dt;
  double PCR1[NUMCRPOP], eCR1[NUMCRPOP], gCR[NUMCRPOP];
  double P1, eps1, c1, g1, A1, dA1dt, a, fac_hsml;
  double xgCR[NUMCRPOP], Pth2, P2, eps2, rho2g2;
  double f0, f1;
  double XCR, inj_rate;
  int CRpop;
  double PCR1sum, eCR1sum, gCRsum, PCR1_xgCR_sum, gCR_PCR1_xgCR_sum, eCR1_xgCR_sum;

  PCR1sum = 0.0;
  eCR1sum = 0.0;
  gCRsum = 0.0;
  PCR1_xgCR_sum = 0.0;
  gCR_PCR1_xgCR_sum = 0.0;
  eCR1_xgCR_sum = 0.0;

  const double x = gsl_vector_get(v, 0);
  const double z = gsl_vector_get(v, 1);
  struct sph_particle_data *Particle;

  Particle = (struct sph_particle_data *) auxParticle;

  /* Generally: comoving -> physical quantities! */
  if(All.ComovingIntegrationOn)
    {
      /* CR definitions: */
      rho1 = Particle->d.Density / mycube(All.Time);
      for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
	{
	  PCR1[CRpop] = CR_Particle_Pressure(Particle, CRpop) * pow(All.Time, -3.0 * GAMMA);
	  PCR1sum += PCR1[CRpop];
	  eCR1[CRpop] = CR_Particle_SpecificEnergy(Particle, CRpop) * rho1;
	  eCR1sum += eCR1[CRpop];
	  gCR[CRpop] = Particle->CR_Gamma0[CRpop];
	}
      /* thermal gas definitions: */
      P1 = Particle->Pressure * pow(All.Time, -3.0 * GAMMA);	/* = PCR1 + Pth1 */
      Pth1 = P1 - PCR1sum;
      hsml = Particle->Hsml * All.Time;
      Ath1 = Particle->Entropy;
      dAth1dt = Particle->e.DtEntropy * hubble_a;
    }
  else
    {
      /* CR definitions: */
      rho1 = Particle->d.Density;
      for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
	{
	  PCR1[CRpop] = CR_Particle_Pressure(Particle, CRpop);
	  PCR1sum += PCR1[CRpop];
	  eCR1[CRpop] = CR_Particle_SpecificEnergy(Particle, CRpop) * rho1;
	  eCR1sum += eCR1[CRpop];
	  gCR[CRpop] = Particle->CR_Gamma0[CRpop];
	}
      /* thermal gas definitions: */
      P1 = Particle->Pressure;	/* = PCR1 + Pth1 */
      Pth1 = P1 - PCR1sum;
      hsml = Particle->Hsml;
      Ath1 = Particle->Entropy;
      dAth1dt = Particle->e.DtEntropy;
    }

  /* defining upstream quantities: */
  eps1 = eCR1sum + 1. / (GAMMA - 1.) * Pth1;
  for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
    gCRsum += gCR[CRpop] * PCR1[CRpop];
  g1 = (gCRsum + GAMMA * Pth1) / P1;
  c1 = sqrt(g1 * P1 / rho1);
  A1 = P1 * pow(rho1, -g1);
  dA1dt = dAth1dt * pow(rho1, GAMMA - g1);
  fac_hsml = All.Shock_Length;
  a = fac_hsml * hsml / (c1 * A1) * dA1dt;

  /* debugging: */
  if(x < 0.0)
    {
      XCR = PCR1sum / Pth1;
      inj_rate = Particle->e.DtEntropy * hubble_a / Particle->Entropy;
      printf("  x < 0.0 in function %s()!\n", __func__);
      printf("  Comoving quantities:\n");
      printf("  #define hubble_a %e  #define atime %e\n", hubble_a, atime);
      printf("  XCR  = %e;  inj_rate = %e;\n", XCR, inj_rate);
      printf("  rho1 = %e;  hsml     = %e;  Ath1 = %e;  dAth1dt = %e;\n",
	     Particle->d.Density, Particle->Hsml, Particle->Entropy, Particle->e.DtEntropy);
      printf("  P1   = %e;  PCR1sum     = %e;  eCR1sum = %e;\n\n",
	     Particle->Pressure, PCR1sum * pow(atime, 3.0 * GAMMA), eCR1sum / rho1);
      for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
	printf("  gCR[%i]     = %e;", CRpop, Particle->CR_Gamma0[CRpop]);
      printf(" \n\n ");
      fflush(stdout);
    }
  assert(x >= 0.0);

  /* defining downstream quantities: */
  for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
    xgCR[CRpop] = pow(x, gCR[CRpop]);
  Pth2 = (x + 4. * z) * Pth1;
  for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
    {
      PCR1_xgCR_sum += PCR1[CRpop] * xgCR[CRpop];
      gCR_PCR1_xgCR_sum += gCR[CRpop] * PCR1[CRpop] * xgCR[CRpop];
      eCR1_xgCR_sum += eCR1[CRpop] * xgCR[CRpop];
    }
  P2 = PCR1_xgCR_sum + Pth2;
  eps2 = eCR1_xgCR_sum + 1. / (GAMMA - 1.) * Pth2;
  rho2g2 = pow(rho1 * x, -(gCR_PCR1_xgCR_sum + GAMMA * Pth1 * (x + 4. * z)) / P2);

  /* system of nonlinear equations: */
  f0 =
    x * (P2 - P1) * SQR(P2 * rho2g2 - P1 * pow(rho1, -g1)) -
    a * a * c1 * c1 * P1 * P1 * (x - 1.) * pow(rho1, 1. - 2. * g1);

  f1 = 2. * eps2 + P1 + P2 - x * (2. * eps1 + P1 + P2);

  gsl_vector_set(f, 0, f0);
  gsl_vector_set(f, 1, f1);

  return GSL_SUCCESS;
}

/* --- end of function shock_conditions_f --- */

/* --- Defining the Jacobian of the system. --- */
/* --- function shock_conditions_df         --- */
int shock_conditions_df(const gsl_vector * v, void *auxParticle, gsl_matrix * J)
{
  double Pth1, rho1, hsml, dAth1dt;
  double PCR1[NUMCRPOP], eCR1[NUMCRPOP], gCR[NUMCRPOP], gth;
  double P1, eps1, c1, g1, A1, dA1dt, a, fac_hsml;
  double xgCR[NUMCRPOP], Pth2, P2, eps2, rho1g1, rho2g2;
  double XCR, inj_rate;

  double df00, df01, df10, df11;
  const double x = gsl_vector_get(v, 0);
  const double z = gsl_vector_get(v, 1);
  struct sph_particle_data *Particle;
  int CRpop;
  double PCR1sum, eCR1sum, gCRsum, PCR1_xgCR_sum, gCR2_PCR1_xgCR_sum;
  double gCR_PCR1_xgCR_sum, gCR_PCR1_xgCR_m1_sum;
  double A_sum, B_sum, C_sum, D_sum;

  PCR1sum = 0.0;
  eCR1sum = 0.0;
  gCRsum = 0.0;
  PCR1_xgCR_sum = 0.0;
  gCR_PCR1_xgCR_sum = 0.0;
  gCR_PCR1_xgCR_m1_sum = 0.0;
  gCR2_PCR1_xgCR_sum = 0.0;
  A_sum = 0.0;
  B_sum = 0.0;
  C_sum = 0.0;
  D_sum = 0.0;

  Particle = (struct sph_particle_data *) auxParticle;

  /* Generally: comoving -> physical quantities! */
  if(All.ComovingIntegrationOn)
    {
      /* CR definitions: */
      rho1 = Particle->d.Density / mycube(All.Time);
      for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
	{
	  PCR1[CRpop] = CR_Particle_Pressure(Particle, CRpop) * pow(All.Time, -3.0 * GAMMA);
	  PCR1sum += PCR1[CRpop];
	  eCR1[CRpop] = CR_Particle_SpecificEnergy(Particle, CRpop) * rho1;
	  eCR1sum += eCR1[CRpop];
	  gCR[CRpop] = Particle->CR_Gamma0[CRpop];
	}
      /* thermal gas definitions: */
      P1 = Particle->Pressure * pow(All.Time, -3.0 * GAMMA);	/* = PCR1 + Pth1 */
      Pth1 = P1 - PCR1sum;
      hsml = Particle->Hsml * All.Time;
      dAth1dt = Particle->e.DtEntropy * hubble_a;
    }
  else
    {
      /* CR definitions: */
      rho1 = Particle->d.Density;
      for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
	{
	  PCR1[CRpop] = CR_Particle_Pressure(Particle, CRpop);
	  PCR1sum += PCR1[CRpop];
	  eCR1[CRpop] = CR_Particle_SpecificEnergy(Particle, CRpop) * rho1;
	  eCR1sum += eCR1[CRpop];
	  gCR[CRpop] = Particle->CR_Gamma0[CRpop];
	}
      /* thermal gas definitions: */
      P1 = Particle->Pressure;	/* = PCR1 + Pth1 */
      Pth1 = P1 - PCR1sum;
      hsml = Particle->Hsml;
      dAth1dt = Particle->e.DtEntropy;
    }

  /* defining upstream quantities: */
  gth = GAMMA;
  eps1 = eCR1sum + 1. / (GAMMA - 1.) * Pth1;
  for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
    {
      gCRsum += gCR[CRpop] * PCR1[CRpop];
    }
  g1 = (gCRsum + GAMMA * Pth1) / P1;
  c1 = sqrt(g1 * P1 / rho1);
  A1 = P1 * pow(rho1, -g1);
  dA1dt = dAth1dt * pow(rho1, GAMMA - g1);
  fac_hsml = All.Shock_Length;
  a = fac_hsml * hsml / (c1 * A1) * dA1dt;

  /* debugging: */
  if(x < 0.0)
    {
      XCR = PCR1sum / Pth1;
      inj_rate = Particle->e.DtEntropy * hubble_a / Particle->Entropy;
      printf("  x < 0.0 in function %s()!\n", __func__);
      printf("  Comoving quantities:\n");
      printf("  #define hubble_a %e  #define atime %e\n", hubble_a, atime);
      printf("  XCR  = %e;  inj_rate = %e;\n", XCR, inj_rate);
      printf("  rho1 = %e;  hsml     = %e;  Ath1 = %e;  dAth1dt = %e;\n",
	     Particle->d.Density, Particle->Hsml, Particle->Entropy, Particle->e.DtEntropy);
      printf("  P1   = %e;  PCR1sum     = %e;  eCR1sum = %e;\n\n",
	     Particle->Pressure, PCR1sum * pow(atime, 3.0 * GAMMA), eCR1sum / rho1);
      for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
	printf("  gCR[%i]     = %e;", CRpop, Particle->CR_Gamma0[CRpop]);
      printf(" \n\n ");
      fflush(stdout);
    }
  assert(x >= 0.0);

  /* defining downstream quantities: */
  for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
    xgCR[CRpop] = pow(x, gCR[CRpop]);
  Pth2 = (x + 4. * z) * Pth1;
  for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
    {
      PCR1_xgCR_sum += PCR1[CRpop] * xgCR[CRpop];
      gCR_PCR1_xgCR_sum += gCR[CRpop] * PCR1[CRpop] * xgCR[CRpop];
      gCR2_PCR1_xgCR_sum += gCR[CRpop] * gCR[CRpop] * PCR1[CRpop] * xgCR[CRpop];
      gCR_PCR1_xgCR_m1_sum += gCR[CRpop] * PCR1[CRpop] * pow(x, gCR[CRpop] - 1.);
      A_sum += (gCR[CRpop] - gth) * PCR1[CRpop] * xgCR[CRpop] * ((gCR[CRpop] - 1.) * x + 4. * gCR[CRpop] * z);
      B_sum += PCR1[CRpop] * xgCR[CRpop] * (1. + (gCR[CRpop] - gth) * log(rho1 * x));
      C_sum +=
	pow(x,
	    gCR[CRpop] - 1.) * (2. * eCR1[CRpop] * gCR[CRpop] + PCR1[CRpop] * (gCR[CRpop] -
									       (1. + gCR[CRpop]) * x));
    }
  D_sum = PCR1_xgCR_sum * gCR2_PCR1_xgCR_sum - gCR_PCR1_xgCR_sum * gCR_PCR1_xgCR_sum;
  P2 = PCR1_xgCR_sum + Pth2;
  rho1g1 = pow(rho1, -g1);
  rho2g2 = pow(rho1 * x, -(gCR_PCR1_xgCR_sum + GAMMA * Pth1 * (x + 4. * z)) / P2);

  /* entries of the Jacobian: */
  /* The if is introduced to get rid of possible round of effects from D_sum which only occurs if NumCRpop > 1. */

  df00 =
    -(a * a * c1 * c1 * P1 * P1 * pow(rho1, 1. - 2. * g1)) +
    x * (Pth1 + gCR_PCR1_xgCR_m1_sum) *
    SQR(P1 * rho1g1 - P2 * rho2g2) +
    (P2 - P1) * SQR(P1 * rho1g1 - P2 * rho2g2) -
    (1. / P2) * 2. * rho1g1 * rho2g2 * rho2g2 * (P2 - P1) *
    (P2 / rho1g1 - P1 / rho2g2) * (((gth - 1.) * x + 4. * gth * z) * P2 * Pth1 +
				   (D_sum + Pth1 * A_sum) * log(rho1 * x));


  df01 =
    4 * Pth1 * x * SQR(P1 * rho1g1 - P2 * rho2g2) +
    (1. / P2) * 8. * Pth1 * rho1g1 * x * rho2g2 * rho2g2 * (P2 - P1) *
    (P2 / rho1g1 - P1 / rho2g2) * (Pth1 * (x + 4. * z) + B_sum);


  df10 = -2. * eps1 - P1 + C_sum + Pth1 * (1 + 2. / (gth - 1.) - 2. * x - 4. * z);


  df11 = 4. * Pth1 * (1. + 2. / (gth - 1.) - x);

  gsl_matrix_set(J, 0, 0, df00);
  gsl_matrix_set(J, 0, 1, df01);
  gsl_matrix_set(J, 1, 0, df10);
  gsl_matrix_set(J, 1, 1, df11);

  return GSL_SUCCESS;
}
#endif /* end of multiple alpha */

/* --- end of function shock_conditions_df --- */

/* --- function shock_conditions_fdf --- */
int shock_conditions_fdf(const gsl_vector * v, void *auxParticle, gsl_vector * f, gsl_matrix * J)
{
  struct sph_particle_data *Particle;

  Particle = (struct sph_particle_data *) auxParticle;

  shock_conditions_f(v, Particle, f);
  shock_conditions_df(v, Particle, J);

  return GSL_SUCCESS;
}

/* --- end of function shock_conditions_fdf --- */


/*  This function identifies shocks in simulations with a gas composed 
 *  of cosmic rays and thermal particles. It computes the Mach number, 
 *  the density jump, and the temperature jump at the shock given the SPH 
 *  smoothing length, the local effective sound speed, the total entropic  
 *  function A, and the instantaneous rate of entropy injection due to 
 *  shocks, dA/dt. It uses a 2D "globally converging" Newton-Raphson method.
 */

#if ( NUMCRPOP == 1)

/* --- function MachNumberCR: 2D Newton-Raphson method --- */

void GetMachNumberCR(struct sph_particle_data *Particle)
{

  static double rho1, uth1, P1, PCR1, Pth1, gCR, g1, c1, P2, PCR2, M;
  static double x0, y0, z0, t0, x, y, z, xiter, yiter, ziter;
  static double XCR, inj_rate;
  static double Mach, DensityJump, EnergyJump, z1, z2, M1, M2;
  static double MCRestimate, DeltaDecayTime, fac_hsml;

  const gsl_multiroot_fdfsolver_type *T;
  gsl_multiroot_fdfsolver *s;
  const gsl_multiroot_fsolver_type *Tnd;
  gsl_multiroot_fsolver *snd;

  double XCRSwitch = 30.0;
  double epsabs;
  double v_init[2];
  int status;
  size_t iter = 0;
  const size_t n = 2;
  int SwitchDiagnostic = 0;

  int index = Particle - SphP;

  /* definitions for Mach number: */
  if(All.ComovingIntegrationOn)
    {
      rho1 = Particle->d.Density / mycube(All.Time);
      P1 = Particle->Pressure * pow(All.Time, -3.0 * GAMMA);	/* = PCR1 + Pth1 */
      PCR1 = CR_Particle_Pressure(Particle, 0) * pow(All.Time, -3.0 * GAMMA);
    }
  else
    {
      rho1 = Particle->d.Density;
      P1 = Particle->Pressure;
      PCR1 = CR_Particle_Pressure(Particle, 0);
    }

  uth1 = Particle->Entropy / GAMMA_MINUS1 * pow(rho1, GAMMA_MINUS1);
  Pth1 = P1 - PCR1;
  gCR = Particle->CR_Gamma0[0];
  g1 = (gCR * PCR1 + GAMMA * Pth1) / P1;
  c1 = sqrt(g1 * P1 / rho1);

  /* tree algorithm to infer the start values for the multiroot solvers: */
  XCR = PCR1 / Pth1;		/* note that XCR denotes the pressure ratio! */

  if(SwitchDiagnostic == 1)
    {
      /* Comoving quantities: */
      XCR = PCR1 / Pth1;
      inj_rate = Particle->e.DtEntropy * hubble_a / Particle->Entropy;
      printf("  XCR  = %e;  inj_rate = %e;  hubble_a %e   atime     %e\n", XCR, inj_rate, hubble_a, atime);
      printf("  rho1 = %e;  hsml     = %e;  Ath1   = %e;  dAth1dt = %e;\n",
	     Particle->d.Density, PPP[index].Hsml, Particle->Entropy, Particle->e.DtEntropy);
      printf("  P1   = %e;  PCR1     = %e;  eCR1   = %e;  gCR     = %e;\n\n",
	     Particle->Pressure, PCR1 * pow(atime, 3.0 * GAMMA),
	     CR_Particle_SpecificEnergy(Particle, 0) * Particle->d.Density, Particle->CR_Gamma0[0]);
      fflush(stdout);
    }

  xiter = 1.0;
  yiter = 1.0;
  x0 = 2.;
  z0 = 2.;

  GetMachNumber(Particle);

  if(Mestimate < 1.4 || XCR < 0.01)
    {
      /* Weak shock (CRs weaken the Shock even more!)   */
      /* OR shock dominated by thermal particles:       */
      Mach = Mestimate;
      MCRestimate = Mestimate;
    }
  else
    {
      if((XCR >= 0.01) && (XCR < XCRSwitch))
	{
	  /* Use 1D MachFinder estimate for initial conditions (but: P = PCR + Pth): */
	  M = Mestimate;
	  x0 = ((GAMMA + 1.) * M * M) / (GAMMA_MINUS1 * M * M + 2.);
	  t0 = (2. * GAMMA * M * M - GAMMA_MINUS1) * (GAMMA_MINUS1 * M * M + 2.) /
	    ((GAMMA + 1.) * (GAMMA + 1.) * M * M);
	  y0 = x0 * t0;
	  z0 = (y0 - x0) / 4.;

	  /* define interation construct: */
	  gsl_multiroot_function_fdf f = { &shock_conditions_f,
	    &shock_conditions_df,
	    &shock_conditions_fdf,
	    n, Particle
	  };

	  epsabs = 1.;
	  v_init[0] = x0;
	  v_init[1] = z0;
	  gsl_vector *v = gsl_vector_alloc(n);

	  gsl_vector_set(v, 0, v_init[0]);
	  gsl_vector_set(v, 1, v_init[1]);

	  /* appropriate multiroot fdf solvers: */
	  /* ordered in reliability and performance for our problem. */
	  T = gsl_multiroot_fdfsolver_gnewton;
	  // T = gsl_multiroot_fdfsolver_hybridsj;
	  // T = gsl_multiroot_fdfsolver_hybridj;

	  s = gsl_multiroot_fdfsolver_alloc(T, n);
	  gsl_multiroot_fdfsolver_set(s, &f, v);

	  /* for diagnostic outputs: */
	  if(SwitchDiagnostic == 1)
	    print_state_fdf(iter, s);

	  do
	    {
	      iter++;

	      status = gsl_multiroot_fdfsolver_iterate(s);

	      /* for diagnostic outputs: */
	      if(SwitchDiagnostic == 1)
		print_state_fdf(iter, s);

	      if(status)
		break;

	      status = gsl_multiroot_test_residual(s->f, epsabs);


	      xiter = gsl_vector_get(gsl_multiroot_fdfsolver_root(s), 0);
	      ziter = gsl_vector_get(gsl_multiroot_fdfsolver_root(s), 1);
	      yiter = 4. * ziter + xiter;
	    }
	  while((status == GSL_CONTINUE) && (iter < 50) && (xiter > 0.) && (yiter > 0.));

	  x = gsl_vector_get(gsl_multiroot_fdfsolver_root(s), 0);
	  z = gsl_vector_get(gsl_multiroot_fdfsolver_root(s), 1);
	  y = 4. * z + x;

	  if((x < 1.0) || (y < 1.0))
	    {
	      x = 1.0;
	      y = 1.0;
	    }

	  /* for diagnostic outputs: */
	  if(SwitchDiagnostic == 1)
	    {
	      printf("status = %s\n", gsl_strerror(status));
	      printf("\n");
	      printf("x = %f, y = %f, z = %f\n\n", x, y, z);
	    }

	  gsl_multiroot_fdfsolver_free(s);
	  gsl_vector_free(v);

	  /*************************************************************
	   * interception routine using numerical derivatives:          
	   * (less efficient, but monotonic decline towards the roots
	   *  compared to the previous case using the analytic Jacobian, 
	   *  which might take unphysical detours!)
	   *************************************************************/
	}
      if((XCR >= XCRSwitch) || (xiter <= 0.) || (yiter <= 0.))
	{
	  /* CRs are the dominant component -> weak shock: */
	  x0 = 1.5;
	  z0 = 0.5;

	  if((xiter <= 0.) || (yiter <= 0.))
	    {
	      printf("Analytic Jacobian iteration failed because x < 0!\n");
	    }

	  if(SwitchDiagnostic == 1)
	    printf("Numerical Jacobian iteration:\n");

	  /* define interation construct: */
	  gsl_multiroot_function fnd = { &shock_conditions_fnd, n, Particle };

	  epsabs = 1.;
	  v_init[0] = x0;
	  v_init[1] = z0;
	  gsl_vector *v = gsl_vector_alloc(n);

	  gsl_vector_set(v, 0, v_init[0]);
	  gsl_vector_set(v, 1, v_init[1]);

	  /* appropriate multiroot f solver (nd = numerical derivative): */
	  Tnd = gsl_multiroot_fsolver_hybrids;
	  snd = gsl_multiroot_fsolver_alloc(Tnd, n);
	  gsl_multiroot_fsolver_set(snd, &fnd, v);

	  /* for diagnostic outputs: */
	  iter = 0;
	  if(SwitchDiagnostic == 1)
	    print_state_f(iter, snd);

	  do
	    {
	      iter++;

	      status = gsl_multiroot_fsolver_iterate(snd);

	      /* for diagnostic outputs: */
	      if(SwitchDiagnostic == 1)
		print_state_f(iter, snd);

	      if(status)
		break;

	      status = gsl_multiroot_test_residual(snd->f, epsabs);

	      xiter = gsl_vector_get(gsl_multiroot_fsolver_root(snd), 0);
	    }
	  while((status == GSL_CONTINUE) && (iter < 200));

	  x = gsl_vector_get(gsl_multiroot_fsolver_root(snd), 0);
	  z = gsl_vector_get(gsl_multiroot_fsolver_root(snd), 1);
	  y = 4. * z + x;

	  /* for diagnostic outputs: */
	  if(SwitchDiagnostic == 1)
	    {
	      printf("status = %s\n", gsl_strerror(status));
	      printf("\n");
	      printf("x = %f, y = %f, z = %f\n\n", x, y, z);
	    }

	  /* clipping the jump values in case the numerical 
	     root finder returned unphysical values: */

	  if((x < 1.0) || (y < 1.0))
	    {
	      x = 1.0;
	      y = 1.0;
	    }

	  gsl_multiroot_fsolver_free(snd);
	  gsl_vector_free(v);
	}

      /**************************************************************/

      assert(x > 0.0);
      /* infering the physical downstream quantities: */
      PCR2 = PCR1 * pow(x, gCR);
      P2 = PCR2 + y * Pth1;

      if((x - 1.0) < 1e-6)
	{
	  M = 1.0;
	}
      else
	{
	  M = sqrt((P2 - P1) * x / (rho1 * c1 * c1 * (x - 1.0)));
	}

      /* returning the following quantities: */
      MCRestimate = M;
      Mach = M;
      DensityJump = x;
      EnergyJump = y / x;	/* if (1.4 <= M <= 3) AND (XCR >= 0.01) */
    }

  /* Semianalytic recalibration of Mach number according to Shock tube simulations. */
  Mach = MachNumberCalibration(Mach);

  /* z2 = y/x = EnergyJump for the high Mach number regime: */
  z2 = (2. * GAMMA * Mach * Mach - GAMMA_MINUS1) * (GAMMA_MINUS1 * Mach * Mach + 2.) /
    ((GAMMA + 1.) * (GAMMA + 1.) * Mach * Mach) * (Particle->PreShock_XCR + 1.);

  if(Mach < 1.25 || XCR < 0.01)
    {
      /* Weak shock (CRs weaken the Shock even more!)                 */
      /* OR shock dominated by thermal particles -> 1D Mach finder:   */
      DensityJump = ((GAMMA + 1.) * Mach * Mach) / (GAMMA_MINUS1 * Mach * Mach + 2.);
      EnergyJump = (2. * GAMMA * Mach * Mach - GAMMA_MINUS1) * (GAMMA_MINUS1 * Mach * Mach + 2.) /
	((GAMMA + 1.) * (GAMMA + 1.) * Mach * Mach);
    }
  else if(Mach > 3. && Mach < 6.)
    {
      /* Linear interpolation between the high-M regime (M > 6),   */
      /* and our 2D Mach finder estimate for the energy jump, z1.  */
      z1 = EnergyJump;
      M1 = 3.;
      M2 = 6.;
      EnergyJump = (z1 * M2 - z2 * M1 + (z2 - z1) * Mach) / (M2 - M1);
    }
  else if(Mach >= 6.)
    {
      /* Energy jumps of strong shocks are dominated by thermal population! */
      /* -> use recalibrated Mach number!                                   */
      EnergyJump = z2;
    }

  /* Introduce decay time of the shock meanwhile the Particle Mach number will not be 
   * updated! Mach numbers of all particles are initialized in init.c with M=1!
   */
  fac_hsml = All.Shock_Length;

  if(Mach > Particle->Shock_MachNumber)
    {
      Particle->Shock_MachNumber = Mach;
      Particle->Shock_DensityJump = DensityJump;
      Particle->Shock_EnergyJump = EnergyJump;

      DeltaDecayTime = fac_hsml * PPP[index].Hsml * atime / (MCRestimate * c1);

      /* convert (Delta t_physical) -> (Delta log a): */
      DeltaDecayTime *= hubble_a;
      DeltaDecayTime = myfmin(DeltaDecayTime, All.Shock_DeltaDecayTimeMax);

      if(All.ComovingIntegrationOn)
	{
	  Particle->Shock_DecayTime = All.Time * (1. + DeltaDecayTime);
	}
      else
	{
	  Particle->Shock_DecayTime = All.Time + DeltaDecayTime;
	}
    }
  else if(All.Time > Particle->Shock_DecayTime)
    {
      Particle->Shock_MachNumber = Mach;
      Particle->Shock_DensityJump = DensityJump;
      Particle->Shock_EnergyJump = EnergyJump;

      Particle->PreShock_PhysicalDensity = rho1;
      Particle->PreShock_PhysicalEnergy = uth1;
      Particle->PreShock_XCR = XCR;
    }

  return;
}

/* --- end of function MachNumberCR --- */

#else /* multiple alpha, i.e.e NUMCRPOP > 1 */

/* --- function MachNumberCR: 2D Newton-Raphson method --- */

void GetMachNumberCR(struct sph_particle_data *Particle)
{

  static double rho1, uth1, P1, PCR1[NUMCRPOP], Pth1, gCR[NUMCRPOP], eCR1[NUMCRPOP];
  static double g1, c1, P2;
  static double PCR2[NUMCRPOP], M;
  static double x0, y0, z0, t0, x, y, z, xiter, yiter, ziter;
  static double XCR, inj_rate;
  static double Mach, DensityJump, EnergyJump;
  static double z1, z2, M1, M2;
  static double MCRestimate, DeltaDecayTime, fac_hsml;

  int CRpop;
  double PCR1sum, eCR1sum, gCRsum, PCR2sum;

  PCR1sum = 0.0;
  eCR1sum = 0.0;
  gCRsum = 0.0;
  PCR2sum = 0.0;

  const gsl_multiroot_fdfsolver_type *T;
  gsl_multiroot_fdfsolver *s;
  const gsl_multiroot_fsolver_type *Tnd;
  gsl_multiroot_fsolver *snd;

  double XCRSwitch = 30.0;
  double epsabs;
  double v_init[2];
  int status;
  size_t iter = 0;
  const size_t n = 2;
  int SwitchDiagnostic = 0;

  /* definitions for Mach number: */
  if(All.ComovingIntegrationOn)
    {
      /* CR definitions: */
      rho1 = Particle->d.Density / mycube(All.Time);
      P1 = Particle->Pressure * pow(All.Time, -3.0 * GAMMA);	/* = PCR1 + Pth1 */
      for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
	{
	  PCR1[CRpop] = CR_Particle_Pressure(Particle, CRpop) * pow(All.Time, -3.0 * GAMMA);
	  PCR1sum += PCR1[CRpop];
	  eCR1[CRpop] = CR_Particle_SpecificEnergy(Particle, CRpop) * rho1;
	  eCR1sum += eCR1[CRpop];
	}
    }
  else
    {
      rho1 = Particle->d.Density;
      P1 = Particle->Pressure;	/* = PCR1 + Pth1 */
      for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
	{
	  PCR1[CRpop] = CR_Particle_Pressure(Particle, CRpop);
	  PCR1sum += PCR1[CRpop];
	  eCR1[CRpop] = CR_Particle_SpecificEnergy(Particle, CRpop) * rho1;
	  eCR1sum += eCR1[CRpop];
	}
    }

  uth1 = Particle->Entropy / GAMMA_MINUS1 * pow(rho1, GAMMA_MINUS1);
  Pth1 = P1 - PCR1sum;
  for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
    {
      gCR[CRpop] = Particle->CR_Gamma0[CRpop];
      gCRsum += gCR[CRpop] * PCR1[CRpop];
    }
  g1 = (gCRsum + GAMMA * Pth1) / P1;
  c1 = sqrt(g1 * P1 / rho1);

  /* tree algorithm to infer the start values for the multiroot solvers: */
  XCR = PCR1sum / Pth1;		/* note that XCR denotes the pressure ratio! */

  if(SwitchDiagnostic == 1)
    {
      /* Comoving quantities: */
      XCR = PCR1sum / Pth1;
      inj_rate = Particle->e.DtEntropy * hubble_a / Particle->Entropy;
      printf("  XCR  = %e;  inj_rate = %e;  hubble_a %e   atime     %e\n", XCR, inj_rate, hubble_a, atime);
      printf("  rho1 = %e;  hsml     = %e;  Ath1   = %e;  dAth1dt = %e;\n",
	     Particle->d.Density, Particle->Hsml, Particle->Entropy, Particle->e.DtEntropy);
      printf("  P1   = %e;  PCR1sum     = %e;  eCR1sum = %e;\n\n",
	     Particle->Pressure, PCR1sum * pow(atime, 3.0 * GAMMA), eCR1sum / rho1);
      for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
	printf("  gCR[%i]     = %e;", CRpop, Particle->CR_Gamma0[CRpop]);
      printf(" \n\n ");
      fflush(stdout);
    }


  xiter = 1.0;
  yiter = 1.0;
  x0 = 2.;
  z0 = 2.;

  GetMachNumber(Particle);

  if(Mestimate < 1.4 || XCR < 0.01)
    {
      /* Weak shock (CRs weaken the Shock even more!)   */
      /* OR shock dominated by thermal particles:       */
      Mach = Mestimate;
      MCRestimate = Mestimate;
    }
  else
    {
      if((XCR >= 0.01) && (XCR < XCRSwitch))
	{

	  /* Use 1D MachFinder estimate for initial conditions (but: P = PCR + Pth): */
	  M = Mestimate;
	  x0 = ((GAMMA + 1.) * M * M) / (GAMMA_MINUS1 * M * M + 2.);
	  t0 = (2. * GAMMA * M * M - GAMMA_MINUS1) * (GAMMA_MINUS1 * M * M + 2.) /
	    ((GAMMA + 1.) * (GAMMA + 1.) * M * M);
	  y0 = x0 * t0;
	  z0 = (y0 - x0) / 4.;

	  /* define interation construct: */
	  gsl_multiroot_function_fdf f = { &shock_conditions_f,
	    &shock_conditions_df,
	    &shock_conditions_fdf,
	    n, Particle
	  };

	  epsabs = 1.;
	  v_init[0] = x0;
	  v_init[1] = z0;
	  gsl_vector *v = gsl_vector_alloc(n);

	  gsl_vector_set(v, 0, v_init[0]);
	  gsl_vector_set(v, 1, v_init[1]);

	  /* appropriate multiroot fdf solvers: */
	  /* ordered in reliability and performance for our problem. */
	  T = gsl_multiroot_fdfsolver_gnewton;
	  // T = gsl_multiroot_fdfsolver_hybridsj;
	  // T = gsl_multiroot_fdfsolver_hybridj;

	  s = gsl_multiroot_fdfsolver_alloc(T, n);
	  gsl_multiroot_fdfsolver_set(s, &f, v);


	  /* for diagnostic outputs: */
	  if(SwitchDiagnostic == 1)
	    print_state_fdf(iter, s);

	  do
	    {
	      iter++;

	      status = gsl_multiroot_fdfsolver_iterate(s);

	      /* for diagnostic outputs: */
	      if(SwitchDiagnostic == 1)
		print_state_fdf(iter, s);

	      if(status)
		break;

	      status = gsl_multiroot_test_residual(s->f, epsabs);

	      xiter = gsl_vector_get(gsl_multiroot_fdfsolver_root(s), 0);
	      ziter = gsl_vector_get(gsl_multiroot_fdfsolver_root(s), 1);
	      yiter = 4. * ziter + xiter;
	    }
	  while((status == GSL_CONTINUE) && (iter < 50) && (xiter > 0.) && (yiter > 0.));


	  x = gsl_vector_get(gsl_multiroot_fdfsolver_root(s), 0);
	  z = gsl_vector_get(gsl_multiroot_fdfsolver_root(s), 1);
	  y = 4. * z + x;

	  /* for diagnostic outputs: */
	  if(SwitchDiagnostic == 1)
	    {
	      printf("status = %s\n", gsl_strerror(status));
	      printf("\n");
	      printf("x = %f, y = %f, z = %f\n\n", x, y, z);
	    }

	  if((x < 1.0) || (y < 1.0))
	    {
	      x = 1.0;
	      y = 1.0;
	    }

	  gsl_multiroot_fdfsolver_free(s);
	  gsl_vector_free(v);



	  /*************************************************************
	   * interception routine using numerical derivatives:          
	   * (less efficient, but monotonic decline towards the roots
	   *  compared to the previous case using the analytic Jacobian, 
	   *  which might take unphysical detours!)
	   *************************************************************/
	}
      if((XCR >= XCRSwitch) || (xiter <= 0.) || (yiter <= 0.))
	{
	  /* CRs are the dominant component -> weak shock: */
	  x0 = 1.5;
	  z0 = 0.5;

	  if((xiter <= 0.) || (yiter <= 0.))
	    {
	      printf("Analytic Jacobian iteration failed because x or y < 0!\n");
	    }

	  if(SwitchDiagnostic == 1)
	    printf("Numerical Jacobian iteration:\n");

	  /* define interation construct: */
	  gsl_multiroot_function fnd = { &shock_conditions_fnd, n, Particle };

	  epsabs = 1.;
	  v_init[0] = x0;
	  v_init[1] = z0;
	  gsl_vector *v = gsl_vector_alloc(n);

	  gsl_vector_set(v, 0, v_init[0]);
	  gsl_vector_set(v, 1, v_init[1]);

	  /* appropriate multiroot f solver (nd = numerical derivative): */
	  Tnd = gsl_multiroot_fsolver_hybrids;
	  snd = gsl_multiroot_fsolver_alloc(Tnd, n);
	  gsl_multiroot_fsolver_set(snd, &fnd, v);

	  /* for diagnostic outputs: */
	  iter = 0;
	  if(SwitchDiagnostic == 1)
	    print_state_f(iter, snd);

	  do
	    {
	      iter++;

	      status = gsl_multiroot_fsolver_iterate(snd);

	      /* for diagnostic outputs: */
	      if(SwitchDiagnostic == 1)
		print_state_f(iter, snd);

	      if(status)
		break;

	      status = gsl_multiroot_test_residual(snd->f, epsabs);

	      xiter = gsl_vector_get(gsl_multiroot_fsolver_root(snd), 0);
	    }
	  while((status == GSL_CONTINUE) && (iter < 200));

	  x = gsl_vector_get(gsl_multiroot_fsolver_root(snd), 0);
	  z = gsl_vector_get(gsl_multiroot_fsolver_root(snd), 1);
	  y = 4. * z + x;

	  /* for diagnostic outputs: */
	  if(SwitchDiagnostic == 1)
	    {
	      printf("status = %s\n", gsl_strerror(status));
	      printf("\n");
	      printf("x = %f, y = %f, z = %f\n\n", x, y, z);
	    }

	  /* clipping the jump values in case the numerical 
	     root finder returned unphysical values: */
	  if((x < 1.0) || (y < 1.0))
	    {
	      x = 1.0;
	      y = 1.0;
	    }

	  gsl_multiroot_fsolver_free(snd);
	  gsl_vector_free(v);

	}


      /**************************************************************/

      assert(x > 0.0);
      PCR2sum = 0.0;
      /* infering the physical downstream quantities: */
      for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
	{
	  PCR2[CRpop] = PCR1[CRpop] * pow(x, gCR[CRpop]);
	  PCR2sum += PCR2[CRpop];
	}
      P2 = PCR2sum + y * Pth1;

      if((x - 1.0) < 1e-6)
	{
	  M = 1.0;
	}
      else
	{
	  M = sqrt((P2 - P1) * x / (rho1 * c1 * c1 * (x - 1.0)));
	}

      /* returning the following quantities: */
      MCRestimate = M;
      Mach = M;
      DensityJump = x;
      EnergyJump = y / x;	/* if (1.4 <= M <= 3) AND (XCR >= 0.01) */
    }

  /* Semianalytic recalibration of Mach number according to Shock tube simulations. */
  Mach = MachNumberCalibration(Mach);

  /* z2 = y/x = EnergyJump for the high Mach number regime: */
  z2 = (2. * GAMMA * Mach * Mach - GAMMA_MINUS1) * (GAMMA_MINUS1 * Mach * Mach + 2.) /
    ((GAMMA + 1.) * (GAMMA + 1.) * Mach * Mach) * (Particle->PreShock_XCR + 1.);

  if(Mach < 1.25 || XCR < 0.01)
    {
      /* Weak shock (CRs weaken the Shock even more!)                 */
      /* OR shock dominated by thermal particles -> 1D Mach finder:   */
      DensityJump = ((GAMMA + 1.) * Mach * Mach) / (GAMMA_MINUS1 * Mach * Mach + 2.);
      EnergyJump = (2. * GAMMA * Mach * Mach - GAMMA_MINUS1) * (GAMMA_MINUS1 * Mach * Mach + 2.) /
	((GAMMA + 1.) * (GAMMA + 1.) * Mach * Mach);
    }
  else if(Mach > 3. && Mach < 6.)
    {
      /* Linear interpolation between the high-M regime (M > 6),   */
      /* and our 2D Mach finder estimate for the energy jump, z1.  */
      z1 = EnergyJump;
      M1 = 3.;
      M2 = 6.;
      EnergyJump = (z1 * M2 - z2 * M1 + (z2 - z1) * Mach) / (M2 - M1);
    }
  else if(Mach >= 6.)
    {
      /* Energy jumps of strong shocks are dominated by thermal population! */
      /* -> use recalibrated Mach number!                                   */
      EnergyJump = z2;
    }

  /* Introduce decay time of the shock meanwhile the Particle Mach number will not be 
   * updated! Mach numbers of all particles are initialized in init.c with M=1!
   */
  fac_hsml = All.Shock_Length;

  if(Mach > Particle->Shock_MachNumber)
    {
      Particle->Shock_MachNumber = Mach;
      Particle->Shock_DensityJump = DensityJump;
      Particle->Shock_EnergyJump = EnergyJump;

      DeltaDecayTime = fac_hsml * Particle->Hsml * atime / (MCRestimate * c1);

      /* convert (Delta t_physical) -> (Delta log a): */
      DeltaDecayTime *= hubble_a;
      DeltaDecayTime = myfmin(DeltaDecayTime, All.Shock_DeltaDecayTimeMax);

      if(All.ComovingIntegrationOn)
	{
	  Particle->Shock_DecayTime = All.Time * (1. + DeltaDecayTime);
	}
      else
	{
	  Particle->Shock_DecayTime = All.Time + DeltaDecayTime;
	}
    }
  else if(All.Time > Particle->Shock_DecayTime)
    {
      Particle->Shock_MachNumber = Mach;
      Particle->Shock_DensityJump = DensityJump;
      Particle->Shock_EnergyJump = EnergyJump;

      Particle->PreShock_PhysicalDensity = rho1;
      Particle->PreShock_PhysicalEnergy = uth1;
      Particle->PreShock_XCR = XCR;
    }

  return;
}

/* --- end of function MachNumberCR --- */

#endif

/*********************************************************/
/************** end of 2D Mach number finder *************/
/*********************************************************/

#endif

#ifdef MACHSTATISTIC
/* --- function: GetShock_DtEnergy (actually change of dissipated specific energy)  --- */
/* --- output: d u_diss / (d ln a), u_diss in physical units!                       --- */
void GetShock_DtEnergy(struct sph_particle_data *Particle)
{
  double dudt;

  if(All.ComovingIntegrationOn)
    {
      dudt = Particle->e.DtEntropy / GAMMA_MINUS1 *
	pow(Particle->d.Density, GAMMA_MINUS1) * pow(All.Time, -3.0 * GAMMA_MINUS1);
    }
  else
    {
      dudt = Particle->e.DtEntropy * pow(Particle->d.Density, GAMMA_MINUS1) / GAMMA_MINUS1;
    }

  Particle->Shock_DtEnergy = dudt;

  return;
}

/* --- end of function GetShock_DtEnergy --- */
#endif

#endif



#ifdef REIONIZATION
void heating(void)
{
  int i;
  double u, temp, meanweight, a3inv;

  /* reionization: Tmin = 10^4 Kelvin @ z = 10: */

  if(Flag_FullStep)
    {
      if(All.not_yet_reionized)
	{
	  if(1 / All.Time - 1 < 10.0)
	    {
	      All.not_yet_reionized = 0;

	      meanweight = 4.0 / (8 - 5 * (1 - HYDROGEN_MASSFRAC)) * PROTONMASS;	/* fully reionized */

	      a3inv = 1 / (All.Time * All.Time * All.Time);

	      for(i = 0; i < N_gas; i++)
		{
		  u = SphP[i].Entropy / GAMMA_MINUS1 * pow(SphP[i].d.Density * a3inv, GAMMA_MINUS1);

		  temp =
		    meanweight / BOLTZMANN * GAMMA_MINUS1 * u * All.UnitEnergy_in_cgs / All.UnitMass_in_g;

		  if(temp < 1.0e4)
		    temp = 1.0e4;

		  u =
		    temp / (meanweight / BOLTZMANN * GAMMA_MINUS1 * All.UnitEnergy_in_cgs /
			    All.UnitMass_in_g);

		  SphP[i].Entropy = u * GAMMA_MINUS1 / pow(SphP[i].d.Density * a3inv, GAMMA_MINUS1);
		}
	    }
	}
    }
}
#endif
