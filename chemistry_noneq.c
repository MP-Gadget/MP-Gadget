#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef CHEMISTRY

#include "allvars.h"
#include "proto.h"
#include "tags.h"
#include "chemistry.h"

/*  chemical reactions                                  */
/*  mostly from Abel, Annions, Zhang & Norman (1997), with recent updated rates 
    
   ------- 1:	HI    + e   -> HII   + 2e    hydrogen collisional ionization
   ------- 2:	HII   + e   -> H     + p     hydrogen recombination
   ------- 3:	HeI   + e   -> HeII  + 2e    He collisional ionization
   ------- 4:	HeII  + e   -> HeI   + p     He recombination 
   ------- 5:	HeII  + e   -> HeIII + 2e    HeII collisional ionization
   ------- 6:	HeIII + e   -> HeII  + p     HeII recombination 
   ------- 7:	HI    + e   -> HM    + p     photo-attachment
   ------- 8:	HM    + HI  -> H2I*  + e     H2 formation path (H-)
   ------- 9:   HI    + HII -> H2II  + p     H+ channel
   ------- 10:	H2II  + HI  -> H2I*  + HII   H+ channel
   ------- 11:	H2I   + H   -> 3H
   ------- 12:	H2I   + HII -> H2II  + H
   ------- 13:    omitted
   ------- 14:	H2I   + e   -> 2HI   + e     collisional dissociation
   ------- 15:    omitted
   ------- 16:	HM    + e   -> HI    + 2e
   ------- 17:	HM    + HI  -> 2H    + e
   ------- 18:	HM    + HII -> 2HI
   ------- 19:	HM    + HII -> H2II  + e
   ------- 20:	H2II  + e   -> 2HI
   ------- 21:	H2II  + HM  -> HI    + H2I

   ------- 24:	HI    + p   -> HII   + e     photo-ionization H
   ------- 25:	HeII  + p   -> HeIII + e     HeII ionization
   ------- 26:	HeI   + p   -> HeII  + e     photo-ionization He
   ------- 27:	HM    + p   -> HI    + e     photo-detachment
   ------- 28:	H2II  + p   -> HI    + HII   low energy photo-dissociation
   ------- 29:	H2I   + p   -> H2II  + e     
   ------- 30:	H2II  + p   -> 2HII  + e
   ------- 31:	H2I   + p   -> 2HI           photo-dissociation      */



/* ----- Declare some physical constants (cgs). */
#define  eV_to_K      11606.0
#define  eV_to_erg    1.60184e-12
#define  eV_to_Hz     2.41838e14
#define  CLIGHT      2.9979e10
#define  PLANCK      6.6262e-27
#define  tiny         0.0

#define  alpha       1.0


/* ----- Some grid bounds. */
#define T_start    0.001
#define T_end      1.0e9
#define nu_start   0.74
#define nu_end     0.74e4



static struct base_info
{
  float len;
  float time;
  float vol;
  float numden;
  float J;
  float kunit;
} base;




/* -------------- VARIABLES ----------------- */

/* ----- Grid spacings. */
static double dlog_T;

#ifdef RADIATIVE_RATES
static double dlog_nu;
#endif


static double piHI, piHeI, piHeII;

/* ----- Energy arrays  */
static double edot_ceHI, edot_ceHeI, edot_ceHeII;
static double edot_ciHI, edot_ciHeI, edot_ciHeII;
static double edot_ciHeIS, edot_reHI, edot_reHeII1;
static double edot_reHeII2, edot_reHeIII, edot_comp;
static double edot_brem, edot_H2, edot_radHI;
static double edot_radHeI, edot_radHeII;


static double k24, k25, k26, k27, k28, k29, k30, k31;




int energy_solver(int solve_it_flag, int ithis, double current_a, double dt, double rho_gas,
		  double T_gas_in, double *T_gas_out, double *t_cool);
int species_solver(int solve_it, int ithis, double a, double dt, double rho_gas, double T_gas,
		   double *elec_dot);
int radiative_rates(int i, double a, double J_21);
int compute_abundances(int mode, int ithis, double a_start, double a_end);
int chem_tabulate(void);
int init_rad(double);
int heap_sort(unsigned long n, double ra[]);
int compute_gamma(int ipart);
double radiation_field(double a);


#ifdef CHEMISTRY

int compute_gamma(int ithis)
{
  /* variable gamma
     Equations (5)-(7) in Omukai-Nishi */
  int i = 1;
  double y_HI, y_HII, y_HM, y_HeI, y_HeII, y_HeIII, y_elec, y_H2;
  double n_H;			/* number density of hydrogen nuclei */
  double x, egyspec_to_temp, gas_temp, denominator, sum_y;
  double gamma_H2_minus1_inv;
  double n_den, mean_nden;
  double rho_gas, a3inv, egyspec;

  exit(3333);			/* not yet clarified so not used */

  a3inv = 1 / (All.Time * All.Time * All.Time);
  rho_gas = All.UnitDensity_in_cgs * SphP[ithis].d.Density * All.HubbleParam * All.HubbleParam * a3inv;
  n_den = rho_gas / (PROTONMASS * (4 * SphP[ithis].HeI + SphP[ithis].HII + 4 * SphP[ithis].HeIII
				   + SphP[ithis].HI + 4 * SphP[ithis].HeII
				   + SphP[ithis].HM + 2 * SphP[ithis].H2II + 2 * SphP[ithis].H2I));

  mean_nden = SphP[ithis].HI + SphP[ithis].HII + SphP[ithis].elec
    + SphP[ithis].HeI + SphP[ithis].HeII + SphP[ithis].HeIII
    + SphP[ithis].HM + SphP[ithis].H2II + SphP[ithis].H2I;

  /* unit conversion */
  egyspec_to_temp = All.UnitPressure_in_cgs / All.UnitDensity_in_cgs
    * (SphP[ithis].Gamma - 1.0) / BOLTZMANN * PROTONMASS / (mean_nden * base.numden);


  egyspec = SphP[ithis].Entropy / GAMMA_MINUS1 * pow(SphP[ithis].d.Density * a3inv, GAMMA_MINUS1);

  gas_temp = egyspec * egyspec_to_temp;


  n_H = SphP[ithis].HI + SphP[ithis].HII + SphP[ithis].HM + 2 * SphP[ithis].H2I;

  y_HI = SphP[ithis].HI / n_H;
  y_HII = SphP[ithis].HII / n_H;
  y_HeI = SphP[ithis].HeI / n_H;
  y_HeII = SphP[ithis].HeII / n_H;
  y_HeIII = SphP[ithis].HeIII / n_H;
  y_HM = SphP[ithis].HM / n_H;
  y_H2 = SphP[ithis].H2I / n_H;
  y_elec = SphP[ithis].elec / n_H;



  x = 6100.0 / gas_temp;

  gamma_H2_minus1_inv = 0.5 * (5.0 + 2.0 * x * x * exp(x) / (exp(x) - 1.0) / (exp(x) - 1.0));


  sum_y = y_HI + y_HII + y_HM + y_HeI + y_HeII + y_HeIII + y_elec + y_H2;

  denominator = 1.5 * (y_HI + y_HII + y_HM + y_HeI + y_HeII + y_HeIII + y_elec) + gamma_H2_minus1_inv * y_H2;

  SphP[ithis].Gamma = 1.0 + sum_y / denominator;

  if(SphP[ithis].Gamma > 1.8 || SphP[ithis].Gamma < 1.4)
    {
      printf("Are you sure ?");
      endrun(332);
    }

  return (i);

}





/* compute the new temperature, number density, fraction.
 * Arguments are passed in code units, density is proper density.
 */

int compute_abundances(int mode, int ithis, double a_start, double a_end)
{
  /* *******************************************************************
     Variables:
     ********************************************************************  */

  /*    time variable */
  double a_now;

  /*    temperature, density in g/cm^3, pressure in K/cm^3  */
  double T_gas, rho_gas, T_gas_in, T_gas_out;

  /*    time stepping, epsilon: fraction of t_cool over which we subcycle. */
  double dt = 0, dt_in = 0, t_cool = 0, elec_dot = 0;
  double dt_min = 0, time1 = 0, time2 = 0, time3 = 0;

  int i, ifunc, advance_flag, MAX_ITERATION;

  /* hydro and geometry: [T_gas]=K, [rho_gas]=gram/cm^3  */
  double i_T_gas, i_rho_gas;

  /* Radiation field: J21 is the max amplitude of the radiation field.
     The time (in)dependent radiation field is specified in 
     radiation_model.f */
  double J_21;
  double a3inv, hubble_a, da;
  double egyspec_to_temp, egyspec;
  double sum_nuclei, mean_nden, molecular_weight;

  /* *******************************************************************
     Initialization
     ******************************************************************* */


#ifdef CMB
  J_21 = 1;			/* no normalization necessary for CMB */
#else
  J_21 = 0;			/* radiation field */
#endif


#ifdef RADIATIVE_RATES
  ifunc = radiative_rates(ithis, a_start, J_21);
#else
  k24 = k25 = k26 = k27 = k28 = k29 = k30 = k31 = 0;
  piHI = piHeI = piHeII = 0;
#endif

  if(All.ComovingIntegrationOn)
    {
      a3inv = 1 / (a_start * a_start * a_start);
      rho_gas = All.UnitDensity_in_cgs * SphP[ithis].d.Density * All.HubbleParam * All.HubbleParam * a3inv;	/* physical gram / cm^3 */
    }
  else
    {
      a3inv = 1;
      rho_gas = All.UnitDensity_in_cgs * SphP[ithis].d.Density;
    }



  /* here mean molecular weight must be taken into account */
  base.numden = rho_gas / (PROTONMASS * (4 * SphP[ithis].HeI + SphP[ithis].HII + 4 * SphP[ithis].HeIII
					 + SphP[ithis].HI + 4 * SphP[ithis].HeII
					 + SphP[ithis].HM + 2 * SphP[ithis].H2II + 2 * SphP[ithis].H2I));


  sum_nuclei = 4 * SphP[ithis].HeI + SphP[ithis].HII + 4 * SphP[ithis].HeIII
    + SphP[ithis].HI + 4 * SphP[ithis].HeII + SphP[ithis].HM + 2 * SphP[ithis].H2II + 2 * SphP[ithis].H2I;


  mean_nden = SphP[ithis].HI + SphP[ithis].HII + SphP[ithis].elec
    + SphP[ithis].HeI + SphP[ithis].HeII + SphP[ithis].HeIII
    + SphP[ithis].HM + SphP[ithis].H2II + SphP[ithis].H2I;


  molecular_weight = sum_nuclei / mean_nden;


  /* unit conversion */
  egyspec_to_temp = All.UnitPressure_in_cgs / All.UnitDensity_in_cgs
    * (SphP[ithis].Gamma - 1.0) / BOLTZMANN * PROTONMASS * molecular_weight;

  egyspec = SphP[ithis].Entropy / GAMMA_MINUS1 * pow(SphP[ithis].d.Density * a3inv, GAMMA_MINUS1);

  T_gas = egyspec * egyspec_to_temp;


  i_T_gas = T_gas;		/* starting value */
  i_rho_gas = rho_gas;

  if(P[ithis].ID == 1)
    {
      printf("Initial temperature    = %g %g %g %g %g\n",
	     i_T_gas, SphP[ithis].Entropy, GAMMA_MINUS1, SphP[ithis].d.Density, a3inv);
      printf("Initial density in cgs = %g\n", i_rho_gas);
      printf("egyspec_to_temp = %g, mean_nden = %g, molecular_weight = %g\n", egyspec_to_temp, mean_nden,
	     molecular_weight);
    }


  advance_flag = 0;
  dt_in = 0;
  T_gas_in = T_gas;

  ifunc = energy_solver(advance_flag, ithis, a_start, dt_in, rho_gas, T_gas_in, &T_gas_out, &t_cool);


  if(P[ithis].ID == 1)
    printf("Initial cooling time in Myrs=%g\n", t_cool);


  if(mode == 0)
    {				/* just compute the timestep and then return it */

      T_gas_in = T_gas;
      ifunc = species_solver(advance_flag = 0, ithis, a_start, dt_in, rho_gas, T_gas_in, &elec_dot);

      SphP[ithis].t_cool = t_cool * 1.0e6;	/* yrs */
      SphP[ithis].t_elec = SphP[ithis].elec * base.numden / elec_dot * 1.0e6;	/* yrs */

      return (ithis);
    }



  /*    intialize current redshift */
  a_now = a_start;


#ifdef NONEQUILIBRIUM
  MAX_ITERATION = 2000000000;
#else
  MAX_ITERATION = 2000000000;
#endif


  /*    MAIN LOOP: ------------------------------------------------------------------ */
  for(i = 0; i < MAX_ITERATION; i++)
    {

      /*    get current cooling time  */
      dt_in = dt;

      ifunc = energy_solver(advance_flag = 0, ithis, a_now, dt_in, rho_gas, T_gas_in, &T_gas_out, &t_cool);


      T_gas_in = T_gas_out;
      dt_in = dt;
    /*------- Compute rate of change in electron density. */
      ifunc = species_solver(advance_flag = 0, ithis, a_now, dt_in, rho_gas, T_gas_in, &elec_dot);


    /*---------  set time step. */
      dt_min = MAX_REAL_NUMBER;


      time1 = All.Epsilon * fabs(t_cool);
      time2 = All.Epsilon * fabs(SphP[ithis].elec * base.numden / elec_dot);

      if(All.ComovingIntegrationOn)
	{
	  da = a_end - a_now;
	  hubble_a = All.Hubble * sqrt(All.Omega0 / (a_now * a_now * a_now)
				       + (1 - All.Omega0 - All.OmegaLambda) / (a_now * a_now) +
				       All.OmegaLambda);

	  time3 = da / (a_now * hubble_a);
	  /*
	     if(ThisTask==0) printf("time1 %f time2 %f time3 %f da %f\n",time1,time2,a_now,a_end); 
	   */
	  time3 *= All.UnitTime_in_s;
	  time3 /= All.HubbleParam;	/* in second */
	  time3 /= base.time;
	}
      else
	{
	  time3 = (a_end - a_now) * All.UnitTime_in_s / base.time;
	}

      /* take the minimum of the new timescales */
      if(time1 < dt_min)
	dt_min = time1;

      if(time2 < dt_min)
	dt_min = time2;

      if(time3 < dt_min)
	dt_min = time3;


      dt = dt_min;		/* actual time step to advance */

      /*    Update temperature    */
      dt_in = dt;

      ifunc = energy_solver(advance_flag = 1, ithis, a_now, dt_in, rho_gas, T_gas_in, &T_gas_out, &t_cool);


      /*    Update species */
      T_gas_in = T_gas_out;
      dt_in = dt;
      ifunc = species_solver(advance_flag = 1, ithis, a_now, dt_in, rho_gas, T_gas_in, &elec_dot);


      /* Note. When optical depths are used, radiative rates need to be re-evaluated here */



      if(All.ComovingIntegrationOn)
	{
	  hubble_a = All.Hubble * sqrt(All.Omega0 / (a_now * a_now * a_now)
				       + (1 - All.Omega0 - All.OmegaLambda) / (a_now * a_now) +
				       All.OmegaLambda);

	  /* get dt to be in the system units */
	  dt = dt * base.time;	/* in seconds */
	  dt /= All.UnitTime_in_s;	/* system units */
	  dt *= All.HubbleParam;	/* h correction */

	  a_now = a_now + a_now * hubble_a * dt;
	}
      else
	{

	  dt = dt * base.time;	/* in seconds */
	  dt /= All.UnitTime_in_s;	/* system units */

	  a_now = a_now + dt;
	}


      /* stop if time = a_end */
      if(a_now >= a_end)
	goto LOOP_OUT;

    }

LOOP_OUT:;


  sum_nuclei = 4 * SphP[ithis].HeI + SphP[ithis].HII + 4 * SphP[ithis].HeIII
    + SphP[ithis].HI + 4 * SphP[ithis].HeII + SphP[ithis].HM + 2 * SphP[ithis].H2II + 2 * SphP[ithis].H2I;

  mean_nden = SphP[ithis].HI + SphP[ithis].HII + SphP[ithis].elec
    + SphP[ithis].HeI + SphP[ithis].HeII + SphP[ithis].HeIII
    + SphP[ithis].HM + SphP[ithis].H2II + SphP[ithis].H2I;


  molecular_weight = sum_nuclei / mean_nden;
  egyspec_to_temp = All.UnitPressure_in_cgs / All.UnitDensity_in_cgs
    * (SphP[ithis].Gamma - 1.0) / BOLTZMANN * PROTONMASS * molecular_weight;

  egyspec = T_gas_out / egyspec_to_temp;


  SphP[ithis].Entropy = egyspec * GAMMA_MINUS1 / pow(SphP[ithis].d.Density * a3inv, GAMMA_MINUS1);
  SphP[ithis].Pressure = (SphP[ithis].Gamma - 1.0) * egyspec * SphP[ithis].d.Density;
  SphP[ithis].t_cool = t_cool * 1.0e6;	/* yrs */
  SphP[ithis].t_elec = SphP[ithis].elec * base.numden / elec_dot * 1.0e6;	/* yrs */

  return (i);

}


/* ----- Compute cooling time of gas and
   ----- solve for thermal energy.  */

int energy_solver(int solve_it_flag, int ithis, double current_a, double dt,
		  double rho_gas, double T_gas, double *T_gas_out, double *t_cool)
{
  /* ----- Collisional excitation coefs. */
  double ceHI, ceHeI, ceHeII;

  /* ----- Collisional ionization coefs. */
  double ciHI, ciHeI, ciHeIS, ciHeII;

  /* ----- Recombination coefs. */
  double reHII, reHeII1, reHeII2, reHeIII;

  /* ----- Compton cooling/heating and bremstrahlung. */
  double comp1, comp2, brem;


  int i_T, i_energy, N_energy_iter;
  double dt_half, dt_internal, t_count, mean_nden, energy = 0, edot = 0;
  double log_T, c_T, log_T0, log_T9, T_CMB;

  double TM, LT, LDL, T3, HDLR, HDLV, HDL, cool_rate, heat_rate;

  double dmin1, dmin2, dmax1, dmax2;

  log_T0 = log(T_start);
  log_T9 = log(T_end);



  /*     compute total numberdensity in units of mh
     hence base.numden * HI gives the number density of HI  */

  base.numden = rho_gas / (PROTONMASS * (4 * SphP[ithis].HeI + SphP[ithis].HII + 4 * SphP[ithis].HeIII
					 + SphP[ithis].HI + 4 * SphP[ithis].HeII
					 + SphP[ithis].HM + 2 * SphP[ithis].H2II + 2 * SphP[ithis].H2I));



  N_energy_iter = 1;

#ifdef NONEQUILIBRIUM
  if(solve_it_flag)
    N_energy_iter = 2000000000;
#else
  if(solve_it_flag)
    N_energy_iter = 2000000000;
#endif


  dt_half = dt / 2.0;

  mean_nden = SphP[ithis].HI + SphP[ithis].HII + SphP[ithis].elec
    + SphP[ithis].HeI + SphP[ithis].HeII + SphP[ithis].HeIII
    + SphP[ithis].HM + SphP[ithis].H2II + SphP[ithis].H2I;


  /* ------- Init time counter. */
  t_count = 0.0;


  for(i_energy = 1; i_energy <= N_energy_iter; i_energy++)
    {


      /* --------- Compute thermal energy from temperature and mass. */
      energy = (1.0 / (SphP[ithis].Gamma - 1.0)) * (mean_nden * base.numden) * (BOLTZMANN * (T_gas));


      /* --------- Compute temperture index.  */
      log_T = log(T_gas);
      if(log_T < log_T0)
	log_T = log_T0;
      if(log_T > log_T9)
	log_T = log_T9;

      i_T = (log_T - log_T0) / dlog_T;
      if(i_T < 1)
	i_T = 1;
      if(i_T > N_T - 2)
	i_T = N_T - 2;

      c_T = (T_gas - T[i_T - 1]) / (T[i_T] - T[i_T - 1]);

      /* --------- Interpolate temperature dependent coefficients. */
      ceHI = ceHIa[i_T] + c_T * (ceHIa[i_T + 1] - ceHIa[i_T]);
      ceHeI = ceHeIa[i_T] + c_T * (ceHeIa[i_T + 1] - ceHeIa[i_T]);
      ceHeII = ceHeIIa[i_T] + c_T * (ceHeIIa[i_T + 1] - ceHeIIa[i_T]);

      ciHI = ciHIa[i_T] + c_T * (ciHIa[i_T + 1] - ciHIa[i_T]);
      ciHeI = ciHeIa[i_T] + c_T * (ciHeIa[i_T + 1] - ciHeIa[i_T]);
      ciHeII = ciHeIIa[i_T] + c_T * (ciHeIIa[i_T + 1] - ciHeIIa[i_T]);
      ciHeIS = ciHeISa[i_T] + c_T * (ciHeISa[i_T + 1] - ciHeISa[i_T]);

      reHII = reHIIa[i_T] + c_T * (reHIIa[i_T + 1] - reHIIa[i_T]);
      reHeII1 = reHeII1a[i_T] + c_T * (reHeII1a[i_T + 1] - reHeII1a[i_T]);
      reHeII2 = reHeII2a[i_T] + c_T * (reHeII2a[i_T + 1] - reHeII2a[i_T]);
      reHeIII = reHeIIIa[i_T] + c_T * (reHeIIIa[i_T + 1] - reHeIIIa[i_T]);

      /* ========= Compton heating ========================================================= */


      if(All.ComovingIntegrationOn)
	{
	  comp1 = 5.65e-36 * pow(current_a, -4);
	  comp2 = T_CMB0 / current_a;
	}
      else
	{

	  endrun(432);		/* be careful when inprementing the Compton process for physical coordinates */
	}

      brem = brema[i_T] + c_T * (brema[i_T + 1] - brema[i_T]);


      /* ========= Cooling Functions ======================================================= */

      /* --------- Collisional excitations */
      edot_ceHI = -ceHI * SphP[ithis].HI * SphP[ithis].elec * base.numden * base.numden;
      edot_ceHeI =
	-ceHeI * SphP[ithis].HeII * SphP[ithis].elec * SphP[ithis].elec * base.numden * base.numden *
	base.numden;
      edot_ceHeII = -ceHeII * SphP[ithis].HeII * SphP[ithis].elec * base.numden * base.numden;


      /* --------- Collisional ionizations */
      edot_ciHI = -ciHI * SphP[ithis].HI * SphP[ithis].elec * base.numden * base.numden;
      edot_ciHeI = -ciHeI * SphP[ithis].HeI * SphP[ithis].elec * base.numden * base.numden;
      edot_ciHeII = -ciHeII * SphP[ithis].HeII * SphP[ithis].elec * base.numden * base.numden;
      edot_ciHeIS =
	-ciHeIS * SphP[ithis].HeII * SphP[ithis].elec * SphP[ithis].elec * base.numden * base.numden *
	base.numden;

      /* --------- Recombinations  */
      edot_reHI = -reHII * SphP[ithis].HII * SphP[ithis].elec * base.numden * base.numden;
      edot_reHeII1 = -reHeII1 * SphP[ithis].HeII * SphP[ithis].elec * base.numden * base.numden;
      edot_reHeII2 = -reHeII2 * SphP[ithis].HeII * SphP[ithis].elec * base.numden * base.numden;
      edot_reHeIII = -reHeIII * SphP[ithis].HeIII * SphP[ithis].elec * base.numden * base.numden;


      /* --------- Compton cooling or heating */
      edot_comp = -comp1 * (T_gas - comp2) * SphP[ithis].elec * base.numden;


      /* --------- Bremsstrahlung */
      edot_brem =
	-brem * (SphP[ithis].HII + SphP[ithis].HeII +
		 SphP[ithis].HeIII) * SphP[ithis].elec * base.numden * base.numden;


      /* --------- Cooling from H2, Galli-Palla 1998 */

      TM = DMAX(T_gas, 13.);	/* no cooling below 13 Kelvin ... */
      TM = DMIN(TM, 1.e5);	/* fixes numerics */
      LT = log10(TM);

      /*    low density limit from Galli and Palla  */
      LDL =
	pow(10, -103.0 + 97.59 * LT - 48.05 * LT * LT + 10.80 * LT * LT * LT - 0.9032 * LT * LT * LT * LT);



      /* high density limit from Hollenbach Mckee 1979    */
      T3 = TM / 1000.;
      HDLR =
	((9.5e-22 * pow(T3, 3.76)) / (1. + 0.12 * pow(T3, 2.1)) * exp(-pow(0.13 / T3, 3)) +
	 3.e-24 * exp(-0.51 / T3));

      HDLV = (6.7e-19 * exp(-5.86 / T3) + 1.6e-18 * exp(-11.7 / T3));
      HDL = (HDLR + HDLV) / (SphP[ithis].HI * base.numden);

      /*    cooling rate per molecule [erg/s]  */
      cool_rate = SphP[ithis].HI * base.numden * HDL / (1. + (HDL / LDL));


      /* heating by CMB */
      T_CMB = T_CMB0 / current_a;
      TM = DMAX(T_CMB, 13.);
      TM = DMIN(TM, 1.e5);
      LT = log10(TM);
      LDL =
	pow(10, -103.0 + 97.59 * LT - 48.05 * LT * LT + 10.80 * LT * LT * LT - 0.9032 * LT * LT * LT * LT);
      T3 = TM / 1000.;
      HDLR =
	((9.5e-22 * pow(T3, 3.76)) / (1. + 0.12 * pow(T3, 2.1)) * exp(-pow(0.13 / T3, 3)) +
	 3.e-24 * exp(-0.51 / T3));
      HDLV = (6.7e-19 * exp(-5.86 / T3) + 1.6e-18 * exp(-11.7 / T3));
      HDL = (HDLR + HDLV) / (SphP[ithis].HI * base.numden);

      heat_rate = SphP[ithis].HI * base.numden * HDL / (1. + (HDL / LDL));

      edot_H2 = -SphP[ithis].H2I * base.numden * (cool_rate - heat_rate);



      /* ========= Heating functions ============================================================== */
      /* --------- Photoinization heating */
      edot_radHI = piHI * SphP[ithis].HI * base.numden;
      edot_radHeI = piHeI * SphP[ithis].HeI * base.numden;
      edot_radHeII = piHeII * SphP[ithis].HeII * base.numden;


      edot = edot_ceHI + edot_ceHeI + edot_ceHeII
	+ edot_ciHI + edot_ciHeI + edot_ciHeII + edot_ciHeIS
	+ edot_reHI + edot_reHeII1 + edot_reHeII2 + edot_reHeIII
	+ edot_comp + edot_brem + edot_H2 + edot_radHI + edot_radHeI + edot_radHeII;


      *t_cool = (energy / edot) / base.time;	/* in units of million years */


      if(solve_it_flag)
	{

	  /* ----------- Compute internal time step.  
	     compare the cooling time, 
	     the time remaining,
	     the given half time */

	  dt_internal = fabs(All.Epsilon * (*t_cool));

	  if(fabs(dt - t_count) < dt_internal)
	    dt_internal = fabs(dt - t_count);
	  if(dt_half < dt_internal)
	    dt_internal = dt_half;


	  /* ----------- Advance gas temperature
	     NOTE this update is only accurate for constant density during
	     the temperature update!  */

	  T_gas = T_gas + (dt_internal / (*t_cool)) * T_gas;
	  if(T_gas < T[1])
	    T_gas = T[1];	/* temperature boundary */
	  if(T_gas > T[N_T - 2])
	    T_gas = T[N_T - 2];

	  t_count = t_count + dt_internal;


	  if(dt - t_count < (dt / N_energy_iter))
	    goto LOOP_OUT;

	}

    }

LOOP_OUT:;

  if(solve_it_flag && (i_energy >= N_energy_iter))
    {
      printf("ENERGY i_energy exceeds %d for particle %d on PE %d\n", N_energy_iter, ithis, ThisTask);
      printf("T_gas=%g, t_cool=%g, energy=%g, edot=%g \n", T_gas, *t_cool, energy, edot);
    }


  *T_gas_out = T_gas;

  return i_energy;
}






int species_solver(int solve_it, int ithis, double a_now, double dt, double rho_gas, double T_gas,
		   double *elec_dot)
{

  int i_rate = 0, N_rate_iter, i_T;


  double dt_half, dt_internal, t_count, Ccoef, Dcoef, denom, correction;
  double HIp, HIIp, HeIp, HeIIp, HeIIIp, elecp;
  double log_T, c_T, log_T0, log_T9;

  double k1, k2, k3, k4, k5, k6;
  double k7, k8, k9, k10, k11, k12;
  double k14, k16, k17, k18;
  double k19, k20, k21;

  log_T0 = log(T_start);
  log_T9 = log(T_end);

#ifdef NONEQUILIBRIUM
  N_rate_iter = 2000000000;
#else
  N_rate_iter = 2000000000;
#endif


  /*     compute total numberdensity in units of mh
     hence base.numden*HI gives the number density of HI   */
  base.numden = rho_gas / (PROTONMASS * (4 * SphP[ithis].HeI + SphP[ithis].HII + 4 * SphP[ithis].HeIII
					 + SphP[ithis].HI + 4 * SphP[ithis].HeII
					 + SphP[ithis].HM + 2 * SphP[ithis].H2II + 2 * SphP[ithis].H2I));


  /* ----- Initialize rate coefs. */
  k1 = k2 = k3 = k4 = k5 = k6 = k7 = k8 = k9 = k10 = k11 = k12 = k14 = k16 = k17 = k18 = k19 = k20 = k21 = 0;

  /* ------- Init time counter. */

  t_count = 0.0;

  /* ------- Compute temperature index. */
  log_T = log(T_gas);
  if(log_T < log_T0)
    log_T = log_T0;

  if(log_T > log_T9)
    log_T = log_T9;

  /*          i_T = MIN0(N_T-1,
     &               MAX0(1,ifIX((log_T-log_T0)/dlog_T)+1)) */

  i_T = (log_T - log_T0) / dlog_T;
  if(i_T < 1)
    i_T = 1;

  if(i_T > N_T - 2)
    i_T = N_T - 2;

  c_T = (T_gas - T[i_T]) / (T[i_T + 1] - T[i_T]);


  /* ------- Interpolate coefficients. */
  if(k1_flag)
    k1 = k1a[i_T] + c_T * (k1a[i_T + 1] - k1a[i_T]);
  if(k2_flag)
    k2 = k2a[i_T] + c_T * (k2a[i_T + 1] - k2a[i_T]);
  if(k3_flag)
    k3 = k3a[i_T] + c_T * (k3a[i_T + 1] - k3a[i_T]);
  if(k4_flag)
    k4 = k4a[i_T] + c_T * (k4a[i_T + 1] - k4a[i_T]);
  if(k5_flag)
    k5 = k5a[i_T] + c_T * (k5a[i_T + 1] - k5a[i_T]);
  if(k6_flag)
    k6 = k6a[i_T] + c_T * (k6a[i_T + 1] - k6a[i_T]);
  if(k7_flag)
    k7 = k7a[i_T] + c_T * (k7a[i_T + 1] - k7a[i_T]);
  if(k8_flag)
    k8 = k8a[i_T] + c_T * (k8a[i_T + 1] - k8a[i_T]);
  if(k9_flag)
    k9 = k9a[i_T] + c_T * (k9a[i_T + 1] - k9a[i_T]);
  if(k10_flag)
    k10 = k10a[i_T] + c_T * (k10a[i_T + 1] - k10a[i_T]);
  if(k11_flag)
    k11 = k11a[i_T] + c_T * (k11a[i_T + 1] - k11a[i_T]);
  if(k12_flag)
    k12 = k12a[i_T] + c_T * (k12a[i_T + 1] - k12a[i_T]);
  if(k14_flag)
    k14 = k14a[i_T] + c_T * (k14a[i_T + 1] - k14a[i_T]);
  if(k16_flag)
    k16 = k16a[i_T] + c_T * (k16a[i_T + 1] - k16a[i_T]);
  if(k17_flag)
    k17 = k17a[i_T] + c_T * (k17a[i_T + 1] - k17a[i_T]);
  if(k18_flag)
    k18 = k18a[i_T] + c_T * (k18a[i_T + 1] - k18a[i_T]);
  if(k19_flag)
    k19 = k19a[i_T] + c_T * (k19a[i_T + 1] - k19a[i_T]);
  if(k20_flag)
    k20 = k20a[i_T] + c_T * (k20a[i_T + 1] - k20a[i_T]);
  if(k21_flag)
    k21 = k21a[i_T] + c_T * (k21a[i_T + 1] - k21a[i_T]);


  if(!solve_it)
    {
      *elec_dot = k1 * SphP[ithis].HI * SphP[ithis].elec * base.numden * base.numden
	+ k3 * SphP[ithis].HeI * SphP[ithis].elec * base.numden * base.numden
	+ k5 * SphP[ithis].HeII * SphP[ithis].elec * base.numden * base.numden
	- k2 * SphP[ithis].HII * SphP[ithis].elec * base.numden * base.numden
	- k4 * SphP[ithis].HeII * SphP[ithis].elec * base.numden * base.numden
	- k6 * SphP[ithis].HeIII * SphP[ithis].elec * base.numden * base.numden
	/* check base.vol is properly updated */
	+ (k24 * SphP[ithis].HI * base.numden / base.vol
	   + k25 * SphP[ithis].HeII * base.numden / base.vol
	   + k26 * SphP[ithis].HeI * base.numden / base.vol);

      goto LEAVE2;
    }



  /* ------- Integrate rate equations. */
  for(i_rate = 1; i_rate <= N_rate_iter; i_rate++)
    {

      dt_half = dt / 2.;

      /* ------- Compute electron density rate of change. */
      *elec_dot = k1 * SphP[ithis].HI * SphP[ithis].elec * base.numden * base.numden
	+ k3 * SphP[ithis].HeI * SphP[ithis].elec * base.numden * base.numden
	+ k5 * SphP[ithis].HeII * SphP[ithis].elec * base.numden * base.numden
	- k2 * SphP[ithis].HII * SphP[ithis].elec * base.numden * base.numden
	- k4 * SphP[ithis].HeII * SphP[ithis].elec * base.numden * base.numden
	- k6 * SphP[ithis].HeIII * SphP[ithis].elec * base.numden * base.numden
	+ (k24 * SphP[ithis].HI * base.numden / base.vol
	   + k25 * SphP[ithis].HeII * base.numden / base.vol
	   + k26 * SphP[ithis].HeI * base.numden / base.vol);



      dt_internal = fabs(All.Epsilon * SphP[ithis].elec / (*elec_dot) * base.numden / base.vol);


      if(dt - t_count < dt_internal)
	dt_internal = dt - t_count;
      if(dt_half < dt_internal)
	dt_internal = dt_half;



      if(H2_flag)
	{

	  /* HM non-equilibrium treatment 

	     Ccoef = k7*SphP[ithis].HI*SphP[ithis].elec*base.kunit*base.numden*base.numden;

	     Dcoef = (k8+k17)*SphP[ithis].HI*base.kunit*base.numden
	     + (k18+k19)*SphP[ithis].HII*base.kunit*base.numden
	     + k16*SphP[ithis].elec*base.kunit*base.numden
	     + k27/base.time;


	     SphP[ithis].HM = ( Ccoef*dt_internal*base.time + SphP[ithis].HM*base.numden ) 
	     / ( 1. + Dcoef*dt_internal*base.time ) / base.numden;


	     Ccoef = k9 * SphP[ithis].HI * SphP[ithis].HII*base.kunit*base.numden*base.numden
	     + k12 * SphP[ithis].H2I * SphP[ithis].HII*base.kunit*base.numden*base.numden
	     + k19 * SphP[ithis].HM  * SphP[ithis].HII*base.kunit*base.numden*base.numden
	     + k29 /base.time * SphP[ithis].H2I * base.numden;

	     Dcoef = (k10*SphP[ithis].HI  + k20*SphP[ithis].elec + k21*SphP[ithis].HM) * base.kunit*base.numden
	     + (k28+k30)/base.time;

	     SphP[ithis].H2II =  ( Ccoef*dt_internal*base.time + SphP[ithis].H2II*base.numden ) 
	     / ( 1. + Dcoef*dt_internal*base.time ) / base.numden;

	     Ccoef =   ( k8  * SphP[ithis].HM   * SphP[ithis].HI
	     + k10 * SphP[ithis].H2II * SphP[ithis].HI
	     + k21 * SphP[ithis].H2II * SphP[ithis].HM )*base.kunit*base.numden*base.numden;

	     Dcoef = ( k11 * SphP[ithis].HI + k12 * SphP[ithis].HII + k14 * SphP[ithis].elec )*base.kunit*base.numden
	     + (k29+k31)/base.time;

	     SphP[ithis].H2I = ( Ccoef*dt_internal*base.time + SphP[ithis].H2I*base.numden ) 
	     / ( 1. + Dcoef*dt_internal*base.time ) / base.numden;
	   */



	  /* HM, H2II, equilibrium treatment */
	  denom = (k8 + k17) * SphP[ithis].HI + (k18 + k19) * SphP[ithis].HII + k16 * SphP[ithis].elec
	    + k27 / (base.time * base.kunit * base.numden);

	  SphP[ithis].HM = (k7 * SphP[ithis].HI * SphP[ithis].elec) / denom;

	  SphP[ithis].H2II = (k9 * SphP[ithis].HI * SphP[ithis].HII
			      + k12 * SphP[ithis].H2I * SphP[ithis].HII
			      + k19 * SphP[ithis].HM * SphP[ithis].HII
			      + k29 * SphP[ithis].H2I / (base.time * base.kunit * base.numden))
	    / (k10 * SphP[ithis].HI + k20 * SphP[ithis].elec + k21 * SphP[ithis].HM
	       + (k28 + k30) / (base.time * base.kunit * base.numden));

	  Ccoef = (k8 * SphP[ithis].HM * SphP[ithis].HI
		   + k10 * SphP[ithis].H2II * SphP[ithis].HI
		   + k21 * SphP[ithis].H2II * SphP[ithis].HM) * base.kunit * base.numden * base.numden;

	  Dcoef = (k11 * SphP[ithis].HI
		   + k12 * SphP[ithis].HII
		   + k14 * SphP[ithis].elec) * base.kunit * base.numden + (k29 + k31) / base.time;

	  SphP[ithis].H2I = (Ccoef * dt_internal * base.time + SphP[ithis].H2I * base.numden)
	    / (1. + Dcoef * dt_internal * base.time) / base.numden;

	}





      if(H_flag)
	{
	  /* ----------- HI */
	  Ccoef = k2 * SphP[ithis].HII * SphP[ithis].elec * base.kunit * base.numden * base.numden;
	  Dcoef = k1 * SphP[ithis].elec * base.kunit * base.numden + k24 / base.time;

	  HIp = (Ccoef * dt_internal * base.time + SphP[ithis].HI * base.numden)
	    / (1. + Dcoef * dt_internal * base.time) / base.numden;

	  /* ----------- HII */
	  Ccoef =
	    k1 * HIp * SphP[ithis].elec * base.kunit * base.numden * base.numden +
	    k24 / base.time * HIp * base.numden;

	  Dcoef = k2 * SphP[ithis].elec * base.kunit * base.numden;
	  HIIp = (Ccoef * dt_internal * base.time + SphP[ithis].HII * base.numden)
	    / (1. + Dcoef * dt_internal * base.time) / base.numden;

	}


      if(He_flag)
	{
	  /* ----------- HeI */
	  Ccoef = k4 * SphP[ithis].HeII * SphP[ithis].elec * base.kunit * base.numden * base.numden;
	  Dcoef = k3 * SphP[ithis].elec * base.kunit * base.numden + k26 / base.time;
	  HeIp = (Ccoef * dt_internal * base.time + SphP[ithis].HeI * base.numden)
	    / (1. + Dcoef * dt_internal * base.time) / base.numden;

	  /* ----------- HeII */
	  Ccoef = k3 * HeIp * SphP[ithis].elec * base.kunit * base.numden * base.numden
	    + k6 * SphP[ithis].HeIII * SphP[ithis].elec * base.kunit * base.numden * base.numden
	    + k26 / base.time * HeIp * base.numden;

	  Dcoef = (k4 * SphP[ithis].elec + k5 * SphP[ithis].elec) * base.kunit * base.numden
	    + k25 / base.time;

	  HeIIp = (Ccoef * dt_internal * base.time + SphP[ithis].HeII * base.numden)
	    / (1. + Dcoef * dt_internal * base.time) / base.numden;




	  /* ----------- HeIII */
	  Ccoef = k5 * HeIIp * SphP[ithis].elec * base.kunit * base.numden * base.numden
	    + k25 / base.time * HeIIp * base.numden;

	  Dcoef = k6 * SphP[ithis].elec * base.kunit * base.numden;

	  HeIIIp = (Ccoef * dt_internal * base.time + SphP[ithis].HeIII * base.numden)
	    / (1. + Dcoef * dt_internal * base.time) / base.numden;
	}

      /* --------- elec */
      Ccoef = k24 / base.time * HIp * base.numden
	+ k25 / base.time * HeIIp * base.numden + k26 / base.time * HeIp * base.numden;

      Dcoef = -(k1 * HIp - k2 * HIIp
		+ k3 * HeIp - k6 * HeIIIp + k5 * HeIIp - k4 * HeIIp) * base.kunit * base.numden;

      elecp = (Ccoef * dt_internal * base.time + SphP[ithis].elec * base.numden)
	/ (1. + Dcoef * dt_internal * base.time) / base.numden;


      SphP[ithis].HI = HIp;
      SphP[ithis].HII = HIIp;
      SphP[ithis].HeI = HeIp;
      SphP[ithis].HeII = HeIIp;
      SphP[ithis].HeIII = HeIIIp;
      SphP[ithis].elec = elecp;

      if(SphP[ithis].HeII < 0)
	SphP[ithis].HeII = 0;





      /* --------- Undertake mass conservation corrections. */
      correction =
	H_number_fraction / (SphP[ithis].HI + SphP[ithis].HII + SphP[ithis].HM + 2. * SphP[ithis].H2I +
			     2. * SphP[ithis].H2II);
      SphP[ithis].elec = 0;

      if(H_flag)
	{
	  SphP[ithis].HI *= correction;
	  SphP[ithis].HII *= correction;
	  SphP[ithis].elec += SphP[ithis].HII;
	}

      if(H2_flag)
	{
	  SphP[ithis].HM *= correction;
	  SphP[ithis].H2I *= correction;
	  SphP[ithis].H2II *= correction;
	  SphP[ithis].elec = SphP[ithis].elec - SphP[ithis].HM + SphP[ithis].H2II;
	}

      if(He_flag)
	{
	  correction = He_number_fraction / (SphP[ithis].HeI + SphP[ithis].HeII + SphP[ithis].HeIII);
	  SphP[ithis].HeI *= correction;
	  SphP[ithis].HeII *= correction;
	  SphP[ithis].HeIII *= correction;
	  SphP[ithis].elec = SphP[ithis].elec + SphP[ithis].HeII + 2. * SphP[ithis].HeIII;
	}

      t_count = t_count + dt_internal;

      if(fabs(dt - t_count) < 1.0e-3 * dt)
	{
	  goto LEAVE;
	}

    }


LEAVE:;

  if(i_rate >= N_rate_iter)
    {
      printf("RATE i_rate exceeds %d for particle %d on PE %d\n", N_rate_iter, ithis, ThisTask);
      printf("dt=%g, t_count%g \n", dt, t_count);
    }


LEAVE2:;

  return i_rate;
}












#ifdef RADIATIVE_RATES
int radiative_rates(int ithis, double a_now, double J_21)
{
  /* Compute column densities, radiative rate coefficients for a given 
     radiation field and geometry.

     J_21:  amplitude of radiation field  */


  /*----- Declare other values. */
  double tau_unit, sigma_unit, delta_nu, c0, c1;
  double tau;
  double T_eff;
  int j;

#ifdef H2_SHIELD
  double U_1000A, f_shield, f_shield_avg, x;
#endif

  /*----- Now compute column densities */

  /*----- Set units. */
  sigma_unit = 1.0;
  tau_unit = base.len * base.numden * sigma_unit;


  /*----- Now compute J_nu from column densities.  */

    /*------- Do optically thin case. */
  if(opt_thin_flag)
    for(j = 0; j < N_nu; j++)
      J_nu[j] = J0_nu[j];
  else
    {
      for(j = 0; j < N_nu; j++)
	{
	  tau = 0.0;

	  if(H_flag)
	    tau = tau + SphP[ithis].HI * sigma24[j];

	  if(He_flag)
	    tau = tau + SphP[ithis].HeI * sigma26[j] + SphP[ithis].HeII * sigma25[j];

	  if(H2_flag)
	    tau = tau + SphP[ithis].HM * sigma27[j]
	      + SphP[ithis].H2I * (sigma29[j] + sigma31[j]) + SphP[ithis].H2II * (sigma28[j] + sigma30[j]);


      /*------- Compute J_nu for this frequency and radius. */
	  if(tau > 1.0e-40)
	    J_nu[j] = J_21 * radiation_field(a_now) * J0_nu[j] * exp(-tau_unit * tau);
	  else
	    J_nu[j] = J_21 * radiation_field(a_now) * J0_nu[j];

	}
    }

  /*----- Compute radiatve coefficients. */
  k24 = k25 = k26 = k27 = k28 = k29 = k30 = k31 = piHI = piHeI = piHeII = tiny;


  /* if no radiation then leave */
  if(!rad_flag)
    goto LOOP_OUT;



  /*----- Compute k values integrating over frequency. */
  for(j = 1; j < N_nu; j++)
    {

      delta_nu = (nu[j] - nu[j - 1]) * eV_to_Hz;
      c0 = base.J * J_nu[j] / (nu[j] * eV_to_erg);
      c1 = base.J * J_nu[j - 1] / (nu[j - 1] * eV_to_erg);


      if(k24_flag)
	k24 = k24 + (0.5 * 4. * M_PI) * (c0 * sigma24[j] + c1 * sigma24[j - 1]) * delta_nu;

      if(k25_flag)
	k25 = k25 + (0.5 * 4. * M_PI) * (c0 * sigma25[j] + c1 * sigma25[j - 1]) * delta_nu;

      if(k26_flag)
	k26 = k26 + (0.5 * 4. * M_PI) * (c0 * sigma26[j] + c1 * sigma26[j - 1]) * delta_nu;

      if(k27_flag)
	k27 = k27 + (0.5 * 4. * M_PI) * (c0 * sigma27[j] + c1 * sigma27[j - 1]) * delta_nu;

      if(k28_flag)
	k28 = k28 + (0.5 * 4. * M_PI) * (c0 * sigma28[j] + c1 * sigma28[j - 1]) * delta_nu;

      if(k29_flag)
	k29 = k29 + (0.5 * 4. * M_PI) * (c0 * sigma29[j] + c1 * sigma29[j - 1]) * delta_nu;

      if(k30_flag)
	k30 = k30 + (0.5 * 4. * M_PI) * (c0 * sigma30[j] + c1 * sigma30[j - 1]) * delta_nu;

      if(k31_flag)
	k31 = k31 + (0.5 * 4. * M_PI) * (c0 * sigma31[j] + c1 * sigma31[j - 1]) * delta_nu;




      if(nu[j - 1] >= e24)
	{
	  c0 = base.J * J_nu[j] * (nu[j] - e24) / nu[j];
	  c1 = base.J * J_nu[j - 1] * (nu[j - 1] - e24) / nu[j - 1];
	  piHI = piHI + (0.5 * 4. * M_PI) * (c0 * sigma24[j] + c1 * sigma24[j - 1]) * delta_nu;
	}

      if(nu[j - 1] >= e26)
	{
	  c0 = base.J * J_nu[j] * (nu[j] - e26) / nu[j];
	  c1 = base.J * J_nu[j - 1] * (nu[j - 1] - e26) / nu[j - 1];
	  piHeI = piHeI + (0.5 * 4. * M_PI) * (c0 * sigma26[j] + c1 * sigma26[j - 1]) * delta_nu;
	}

      if(nu[j - 1] >= e25)
	{
	  c0 = base.J * J_nu[j] * (nu[j] - e25) / nu[j];
	  c1 = base.J * J_nu[j - 1] * (nu[j - 1] - e25) / nu[j - 1];
	  piHeII = piHeII + (0.5 * 4. * M_PI) * (c0 * sigma25[j] + c1 * sigma25[j - 1]) * delta_nu;
	}

    }



#ifdef CMB
  /* LTE dissociation rate for CMB run, fit by Galli-Palla (1998) */
  T_eff = T_CMB0 / a_now;

  k28 = 1.63e7 * exp(-32400.0 / T_eff);
#endif




  k24 = k24 * base.time;
  k25 = k25 * base.time;
  k26 = k26 * base.time;
  k27 = k27 * base.time;
  k28 = k28 * base.time;
  k29 = k29 * base.time;
  k30 = k30 * base.time;
  k31 = k31 * base.time;


  if(ThisTask == 0 && ithis == 1)
    printf("Radiative rates k24=%g k25=%g k26=%g k27=%g k28=%g k29=%g k30=%g k31=%g \n",
	   k24, k25, k26, k27, k28, k29, k30, k31);



#ifdef H2_SHIELD
  /*----- Compute rate using Bruce Draine's */
  /*----- fitting formula. */
  /*    ------- Approximate energy density by J_21 */
  /*    ------- which is at 912A. */
  U_1000A = (base.J * J_21 * radiation_field(a_now) * pow(1000. / 912., alpha)) / (1000. * 1.e-10);


  /*    --------- Compute average f_shield over all angles. */
  f_shield_avg = 0.0;
  x = (SphP[ithis].H2I * base.numden * base.len) / 1.e14;
  f_shield = 1.0;
  if(x >= 1.0)
    f_shield = pow(x, -0.75);

  k31 = 0.15 * 3.e-10 * (U_1000A / 4.e-14) * f_shield * base.time;
#endif

LOOP_OUT:;

  return j;
}

double radiation_field(double a)
{
  /*    ----- variables  */
  double res;

  res = 1.0;

  return (res);
}

#endif


int InitChem(void)
{
  int ifunc;


  /* ----- Set various coefs. */
  base.len = 1.0;
  base.time = 1.e6 * SEC_PER_YEAR;
  base.vol = base.len * base.len * base.len;
#ifdef CMB
  base.J = 1.0;
#else
  base.J = 1.E-21;
#endif
  base.kunit = base.vol / base.time;


#ifdef RADIATIVE_RATES
  /* ----- Initialize tables for reaction rates, 
     ----- set up powerlaw radiation field, and frequency array  */
  ifunc = init_rad(All.Time);
#endif

  /* ----- cross-sections, etc ... */
  ifunc = chem_tabulate();

#ifdef RADIATIVE_RATES
  ifunc = radiative_rates(1, All.Time, 1);
#endif

  return (ifunc);

}


/* probably this part should be parallelized in the near future */
int chem_tabulate(void)
{

  /* ----- Declare other values. */
  int i;
  double log_T, log_T_eV, T_eV;
  double log_T_eV2, log_T_eV3, log_T_eV4, log_T_eV5;
  double log_T_eV6, log_T_eV7, log_T_eV8, log_T_eV9;

  double k1b, k2b, k3b, k4b, k5b, k6b;

#ifdef RADIATIVE_RATES
  double dum;
#endif

  /* ----- Initialize tables for reaction rates,
     ----- cross-sections, etc ... */


  /* ----- Set endpoints of T grid in K.
     ----- Might want to contract this range. */

  /* ----- Compute temperature grid. */
  dlog_T = (log(T_end) - log(T_start)) / (N_T - 1);
  for(i = 0; i < N_T; i++)
    {
      log_T = log(T_start) + i * dlog_T;
      T[i] = exp(log_T);
    }

  /* ----- Initialize arrays. */
  for(i = 0; i < N_T; i++)
    {
      k1a[i] = tiny;
      k2a[i] = tiny;
      k3a[i] = tiny;
      k4a[i] = tiny;
      k5a[i] = tiny;
      k6a[i] = tiny;
      k7a[i] = tiny;
      k8a[i] = tiny;
      k9a[i] = tiny;
      k10a[i] = tiny;
      k11a[i] = tiny;
      k12a[i] = tiny;
      k13a[i] = tiny;
      k14a[i] = tiny;
      k15a[i] = tiny;
      k16a[i] = tiny;
      k17a[i] = tiny;
      k18a[i] = tiny;
      k19a[i] = tiny;
      k20a[i] = tiny;
      k21a[i] = tiny;


      ceHIa[i] = tiny;
      ceHeIa[i] = tiny;
      ceHeIIa[i] = tiny;
      ciHIa[i] = tiny;
      ciHeIa[i] = tiny;
      ciHeISa[i] = tiny;
      ciHeIIa[i] = tiny;
      reHIIa[i] = tiny;
      reHeII1a[i] = tiny;
      reHeII2a[i] = tiny;
      reHeIIIa[i] = tiny;
      brema[i] = tiny;

    }




  /* ----- Fill in tables over the range T_start to T_end. */
  for(i = 0; i < N_T; i++)
    {

      /* ------- Compute various values of T. */
      log_T = log(T[i]);
      T_eV = T[i] / eV_to_K;
      log_T_eV = log(T_eV);

      log_T_eV2 = log_T_eV * log_T_eV;
      log_T_eV3 = log_T_eV * log_T_eV2;
      log_T_eV4 = log_T_eV * log_T_eV3;
      log_T_eV5 = log_T_eV * log_T_eV4;
      log_T_eV6 = log_T_eV * log_T_eV5;
      log_T_eV7 = log_T_eV * log_T_eV6;
      log_T_eV8 = log_T_eV * log_T_eV7;
      log_T_eV9 = log_T_eV * log_T_eV8;


      if(T_eV > 0.8)
	{
	  k1a[i] = exp(-32.71396786375
		       + 13.53655609057 * log_T_eV
		       - 5.739328757388 * log_T_eV2
		       + 1.563154982022 * log_T_eV3
		       - 0.2877056004391 * log_T_eV4
		       + 0.03482559773736999 * log_T_eV5
		       - 0.00263197617559 * log_T_eV6
		       + 0.0001119543953861 * log_T_eV7 - 2.039149852002e-6 * log_T_eV8);

	  k3a[i] = exp(-44.09864886561001
		       + 23.91596563469 * log_T_eV
		       - 10.75323019821 * log_T_eV2
		       + 3.058038757198 * log_T_eV3
		       - 0.5685118909884001 * log_T_eV4
		       + 0.06795391233790001 * log_T_eV5
		       - 0.005009056101857001 * log_T_eV6
		       + 0.0002067236157507 * log_T_eV7 - 3.649161410833e-6 * log_T_eV8);

	  k4a[i] = 1.54e-9 * (1. + 0.3 / exp(8.099328789667 / T_eV))
	    / (exp(40.49664394833662 / T_eV) * pow(T_eV, 1.5)) + 3.92e-13 / pow(T_eV, 0.6353);

	  k5a[i] = exp(-68.71040990212001
		       + 43.93347632635 * log_T_eV
		       - 18.48066993568 * log_T_eV2
		       + 4.701626486759002 * log_T_eV3
		       - 0.7692466334492 * log_T_eV4
		       + 0.08113042097303 * log_T_eV5
		       - 0.005324020628287001 * log_T_eV6
		       + 0.0001975705312221 * log_T_eV7 - 3.165581065665e-6 * log_T_eV8);
	}
      else
	{
	  k1a[i] = tiny;
	  k3a[i] = tiny;
	  k4a[i] = 3.92e-13 / pow(T_eV, 0.6353);
	  k5a[i] = tiny;
	}

      if(T[i] > 5500.0)
	{
	  k2a[i] = exp(-28.61303380689232
		       - 0.7241125657826851 * log_T_eV
		       - 0.02026044731984691 * log_T_eV2
		       - 0.002380861877349834 * log_T_eV3
		       - 0.0003212605213188796 * log_T_eV4
		       - 0.00001421502914054107 * log_T_eV5
		       + 4.989108920299513e-6 * log_T_eV6
		       + 5.755614137575758e-7 * log_T_eV7
		       - 1.856767039775261e-8 * log_T_eV8 - 3.071135243196595e-9 * log_T_eV9);
	}
      else
	{
	  k2a[i] = k4a[i];
	}

      k6a[i] = 3.36e-10 / sqrt(T[i]) / pow(T[i] / 1.e3, 0.2) / (1 + pow(T[i] / 4.e6, 0.7));

      k7a[i] = 6.77e-15 * pow(T_eV, 0.8779);

      if(T_eV > 0.1)
	{
	  k8a[i] = exp(-20.06913897587003
		       + 0.2289800603272916 * log_T_eV
		       + 0.03599837721023835 * log_T_eV2
		       - 0.004555120027032095 * log_T_eV3
		       - 0.0003105115447124016 * log_T_eV4
		       + 0.0001073294010367247 * log_T_eV5
		       - 8.36671960467864e-6 * log_T_eV6 + 2.238306228891639e-7 * log_T_eV7);
	}
      else
	{
	  k8a[i] = 1.43e-9;
	}

      k9a[i] = 1.85e-23 * pow(T[i], 1.8);
      if(T[i] > 6.7e3)
	k9a[i] = 5.81e-16 * pow(T[i] / 56200, -0.6657 * log10(T[i] / 56200));

      k10a[i] = 6.0e-10;

      if(T_eV > 0.3)
	{
	  k11a[i] = 1.0670825e-10 * pow(T_eV, 2.012) / (exp(4.463 / T_eV) * pow(1 + 0.2472 * T_eV, 3.512));

	  k12a[i] = exp(-24.24914687731536
			+ 3.400824447095291 * log_T_eV
			- 3.898003964650152 * log_T_eV2
			+ 2.045587822403071 * log_T_eV3
			- 0.5416182856220388 * log_T_eV4
			+ 0.0841077503763412 * log_T_eV5
			- 0.007879026154483455 * log_T_eV6
			+ 0.0004138398421504563 * log_T_eV7 - 9.36345888928611e-6 * log_T_eV8);

	  k14a[i] = 4.38e-10 * exp(-102000.0 / T[i]) * pow(T[i], 0.35);
	}
      else
	{
	  k11a[i] = tiny;
	  k12a[i] = tiny;
	  k14a[i] = tiny;
	}


      k13a[i] = 0.0;
      k15a[i] = 0.0;

      if(T_eV > 0.04)
	{
	  k16a[i] = exp(-18.01849334273
			+ 2.360852208681 * log_T_eV
			- 0.2827443061704 * log_T_eV2
			+ 0.01623316639567 * log_T_eV3
			- 0.03365012031362999 * log_T_eV4
			+ 0.01178329782711 * log_T_eV5
			- 0.001656194699504 * log_T_eV6
			+ 0.0001068275202678 * log_T_eV7 - 2.631285809207e-6 * log_T_eV8);
	}
      else
	{
	  k16a[i] = tiny;
	}

      if(T_eV > 0.1)
	{
	  k17a[i] = exp(-20.37260896533324
			+ 1.139449335841631 * log_T_eV
			- 0.1421013521554148 * log_T_eV2
			+ 0.00846445538663 * log_T_eV3
			- 0.0014327641212992 * log_T_eV4
			+ 0.0002012250284791 * log_T_eV5
			+ 0.0000866396324309 * log_T_eV6
			- 0.00002585009680264 * log_T_eV7
			+ 2.4555011970392e-6 * log_T_eV8 - 8.06838246118e-8 * log_T_eV9);
	}
      else
	{
	  k17a[i] = 2.56e-9 * pow(T_eV, 1.78186);
	}

      k18a[i] = 6.5e-9 / sqrt(T_eV);

      k19a[i] = 1.0e-8 * pow(T[i], -0.4);

      if(T[i] > 1.0e4)
	k19a[i] = 4.0e-4 * pow(T[i], -1.4) * exp(-15100.0 / T[i]);

      k20a[i] = 5.56396e-8 / pow(T_eV, 0.6035);
      k21a[i] = 4.64e-8 / sqrt(T_eV);
    }


  /* ----- Convert to base.kunits. */
  for(i = 0; i < N_T; i++)
    {

      k1a[i] = k1a[i] / base.kunit;
      k2a[i] = k2a[i] / base.kunit;
      k3a[i] = k3a[i] / base.kunit;
      k4a[i] = k4a[i] / base.kunit;
      k5a[i] = k5a[i] / base.kunit;
      k6a[i] = k6a[i] / base.kunit;
      k7a[i] = k7a[i] / base.kunit;
      k8a[i] = k8a[i] / base.kunit;
      k9a[i] = k9a[i] / base.kunit;
      k10a[i] = k10a[i] / base.kunit;
      k11a[i] = k11a[i] / base.kunit;
      k12a[i] = k12a[i] / base.kunit;
      k13a[i] = k13a[i] / base.kunit;
      k14a[i] = k14a[i] / base.kunit;
      k15a[i] = k15a[i] / base.kunit;
      k16a[i] = k16a[i] / base.kunit;
      k17a[i] = k17a[i] / base.kunit;
      k18a[i] = k18a[i] / base.kunit;
      k19a[i] = k19a[i] / base.kunit;
      k20a[i] = k20a[i] / base.kunit;
      k21a[i] = k21a[i] / base.kunit;
    }

  k1b = k2b = k3b = k4b = k5b = k6b = 0;
  for(i = 0; i < N_T; i++)
    {
      k1b += k1a[i];
      k2b += k2a[i];
      k3b += k3a[i];
      k4b += k4a[i];
      k5b += k5a[i];
      k6b += k6a[i];
    }

  for(i = 0; i < N_T; i++)
    {

      /* ------- Collisional excitations (Black 1981; Cen 1992) */
      ceHIa[i] = 7.5e-19 * exp(-118348. / T[i]) / (1. + sqrt(T[i] / 1.e5));

      ceHeIa[i] = 9.1e-27 * exp(-13179. / T[i]) * pow(T[i], -0.1687) / (1. + sqrt(T[i] / 1.e5));

      ceHeIIa[i] = 5.54e-17 * exp(-473638. / T[i]) * pow(T[i], -0.397) / (1. + sqrt(T[i] / 1.e5));

      ciHeISa[i] = 5.01e-27 * pow(T[i], -0.1687) / (1. + sqrt(T[i] / 1.e5)) * exp(-55338. / T[i]);


      /* ------- Collisional ionizations (Tom's polynomial fits) */
      ciHIa[i] = 2.18e-11 * k1a[i] * base.kunit;
      ciHeIa[i] = 3.94e-11 * k3a[i] * base.kunit;
      ciHeIIa[i] = 8.72e-11 * k5a[i] * base.kunit;

      /* ------- Recombinations (Cen 1992) */
      reHIIa[i] = 8.70e-27 * sqrt(T[i]) * pow(T[i] / 1000.0, -0.2) / (1.0 + pow(T[i] / 1.e6, 0.7));

      reHeII1a[i] = 1.55e-26 * pow(T[i], 0.3647);

      /* dielectric recombination */
      reHeII2a[i] = 1.24e-13 * pow(T[i], -1.5) * exp(-470000 / T[i]) * (1. + 0.3 * exp(-94000. / T[i]));

      reHeIIIa[i] = 3.48e-26 * sqrt(T[i]) * pow(T[i] / 1000.0, -0.2) / (1.0 + pow(T[i] / 4.e6, 0.7));

      /* ------- Bremsstrahlung (Black 1981; Spitzer & Hart 1979) */

      brema[i] = 1.43e-27 * sqrt(T[i]) * (1.1 + 0.34 * exp(-pow(5.5 - log10(T[i]), 2.0) / 3.0));

    }


#ifdef RADIATIVE_RATES

  /* ----- Compute cross-sections. */
  for(i = 0; i < N_nu; i++)
    {

      /* photoionisation heating, HI */
      if(nu[i] > e24)
	{
	  dum = sqrt(nu[i] / e24 - 1.0);
	  sigma24[i] =
	    6.3e-18 * pow(e24 / nu[i], 4) * exp(4.0 - 4.0 * atan(dum) / dum) / (1 - exp(-2.0 * M_PI / dum));
	}
      else
	{
	  sigma24[i] = 0.0;
	}

      /* photoionisation heating, HeII */
      if(nu[i] > e25)
	{
	  dum = sqrt(nu[i] / e25 - 1.0);
	  sigma25[i] =
	    1.58e-18 * pow(e25 / nu[i], 4) * exp(4.0 - 4.0 * atan(dum) / dum) / (1 - exp(-2.0 * M_PI / dum));
	}
      else
	{
	  sigma25[i] = 0.0;
	}

      /* photoionisation heating HeI */
      if(nu[i] > e26)
	sigma26[i] = 7.42e-18 * (1.66 * pow(nu[i] / e26, -2.05) - 0.66 * pow(nu[i] / e26, -3.05));
      else
	sigma26[i] = 0.0;


      /* photodissociation heating, HM */
      if(nu[i] > e27)
	sigma27[i] = 2.11e-16 * pow(nu[i] - e27, 1.5) / pow(nu[i], 3);
      else
	sigma27[i] = 0.0;


      /* photodissociation heating, H2II */
      if(nu[i] > e28a && nu[i] <= e28b)
	sigma28[i] =
	  pow(10, -40.97 + 6.03 * nu[i] - 0.504 * nu[i] * nu[i] + 1.387e-2 * nu[i] * nu[i] * nu[i]);
      else if(nu[i] > e28b && nu[i] < e28c)
	sigma28[i] =
	  pow(10, -30.26 + 2.79 * nu[i] - 0.184 * nu[i] * nu[i] + 3.535e-3 * nu[i] * nu[i] * nu[i]);
      else
	sigma28[i] = 0.0;


      /* photodissociation heating, H2I */
      if(nu[i] > e29a && nu[i] <= e29b)
	sigma29[i] = 6.2e-18 * nu[i] - 9.4e-17;
      else if(nu[i] > e29b && nu[i] <= e29c)
	sigma29[i] = 1.4e-18 * nu[i] - 1.48e-17;
      else if(nu[i] > e29c)
	sigma29[i] = 2.5e-14 * pow(nu[i], -2.71);
      else
	sigma29[i] = 0.0;


      /* photodissociation heating, H2II */
      if(nu[i] >= e30a && nu[i] < e30b)
	sigma30[i] =
	  pow(10.0, -16.926 - 4.528e-2 * nu[i] + 2.238e-4 * nu[i] * nu[i] + 4.245e-7 * nu[i] * nu[i] * nu[i]);
      else
	sigma30[i] = 0.0;


      /*  photodissociation heating, H2I */
      if(nu[i] > e28b && nu[i] < e24)
	sigma31[i] = 3.71e-18;
      else
	sigma31[i] = 0.0;

    }
#endif


  return (i);
}







#ifdef RADIATIVE_RATES
int init_rad(double atime)
{

  /* ----- Initialize nu grid  and radiation field.  */

  /* ----- Declare other values.  */
  int i, j, ifunc;
  double log_nu, eps, nu_bound[N_nu_bound];
  char spec_name[200];
  FILE *fd;

#ifdef CMB
  double T_eff, hc;
#endif

  /* ----- Compute base nu grid (in eV) and intrinsic J0_nu.  */
  /*        base.J = 1.e-21  ! now in cloud9.f  */

  dlog_nu = (log(nu_end) - log(nu_start)) / (N_nu_base - 1);
  for(i = 0; i < N_nu_base; i++)
    {
      log_nu = log(nu_start) + i * dlog_nu;
      nu[i] = exp(log_nu);


    }

  /* ----- Set transition points.  */
  nu_bound[0] = e24;
  nu_bound[1] = e25;
  nu_bound[2] = e26;
  nu_bound[3] = e27;
  nu_bound[4] = e28a;
  nu_bound[5] = e28b;
  nu_bound[6] = e28c;
  nu_bound[7] = e29a;
  nu_bound[8] = e29b;
  nu_bound[9] = e29c;
  nu_bound[10] = e30a;
  nu_bound[11] = e30b;

  /* ----- Append boundary points to nu.  */
  eps = 1.e-5;
  for(i = 0; i < N_nu_bound; i++)
    {
      j = i * N_pts_bound + N_nu_base - 1;
      nu[j + 1] = nu_bound[i] * (1. - 4. * eps);
      nu[j + 2] = nu_bound[i] * (1. - 2. * eps);
      nu[j + 3] = nu_bound[i] * (1. - 1. * eps);
      nu[j + 4] = nu_bound[i];
      nu[j + 5] = nu_bound[i] * (1. + 1. * eps);
      nu[j + 6] = nu_bound[i] * (1. + 2. * eps);
      nu[j + 7] = nu_bound[i] * (1. + 4. * eps);
    }

  /* ----- Sort nu.  */
  ifunc = heap_sort(N_nu, nu - 1);


#ifdef CMB
  /* CMB thermal radiation */

  T_eff = T_CMB0 / atime;
  hc = CLIGHT * PLANCK;

  for(i = 0; i < N_nu; i++)
    {
      if(nu[i] == 0)
	J0_nu[i] = 0;
      else
	{
	  J0_nu[i] = 2.0 * pow(nu[i] * eV_to_erg, 3) / (hc * hc)
	    / (exp(nu[i] * eV_to_erg / (BOLTZMANN * T_eff)) - 1.0);

	  if(J0_nu[i] < 1.0e-60)
	    J0_nu[i] = 0.0;
	}
    }
#endif


  if(ThisTask == 0 && All.NumCurrentTiStep == 0)
    {
      sprintf(spec_name, "spectrum.J0.%4.4d", All.NumCurrentTiStep);
      fd = fopen(spec_name, "w");
      for(i = 0; i < N_nu; i++)
	fprintf(fd, "%g %g \n", nu[i], J0_nu[i]);
      fclose(fd);
    }



  return (i);
}


int heap_sort(unsigned long n, double ra[])
{
  int ifunc = 1;
  unsigned long i, ir, j, l;
  double rra;

  if(n < 2)
    return n;
  l = (n >> 1) + 1;
  ir = n;
  for(;;)
    {
      if(l > 1)
	{
	  rra = ra[--l];
	}
      else
	{
	  rra = ra[ir];
	  ra[ir] = ra[1];
	  if(--ir == 1)
	    {
	      ra[1] = rra;
	      break;
	    }
	}
      i = l;
      j = l + l;
      while(j <= ir)
	{
	  if(j < ir && ra[j] < ra[j + 1])
	    j++;
	  if(rra < ra[j])
	    {
	      ra[i] = ra[j];
	      i = j;
	      j <<= 1;
	    }
	  else
	    j = ir + 1;
	}
      ra[i] = rra;
    }

  return (ifunc);
}
#endif



#endif

#endif
