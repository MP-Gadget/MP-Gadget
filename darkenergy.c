/* Using Dark Energy instead of a Cosmological constant can be archived by
 * replacing Lambda by Lambda * a^(-3*(1+w)) in the Hubble function.
 * So easy to see that w = -1 gives back a standard Cosmological Constant !
 * Also w = -1/3 gives Lambda / a^2 which then cancel within the Hubble
 * function and is then equal to the dynamics of a universe with Lambda = 0 !
 *
 * For a time varying w once has to replace Lambda * a^(-3*(1+w)) by
 * Lambda * exp(Integral(a,1,3*(1+w)/a))
 *
 * Once can now also read in colums for the change of the gravitational
 * constant and the correction by this to the hubble function.
 *
 * Additional once can read also an "external" hubble function from a
 * column of the dark energy file.
 *
 * Note that the first column is 'z+1' !
 *
 * Dark Energy does not alter the powerspectrum of initial conditions.
 * To get the same cluster for various values or functions of w, once
 * has do assign a new redshift to the initial cond. to match the
 * linear growth factors, so g(z=0)/g(z_ini) == g_w(z=0)/g_w(z_ini^new)
 * Also the initial velocities field has to be scaled by 
 * (Hubble_w(z_ini^new)*Omega_w(z_ini^new)^0.6)/(Hubble(z_ini)*Omega(z_ini)^0.6)
 * where _w means the according functions including the terms for
 * Dark Energy.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "allvars.h"
#include "proto.h"

#ifdef DARKENERGY
#ifdef TIMEDEPDE

#define ANZ_W_A_IN 4000
#define ANZ_W_A 10000

static MyFloat atab[ANZ_W_A_IN];
static MyFloat wtab[ANZ_W_A_IN];
static MyFloat intwtab[ANZ_W_A + 1];
static MyFloat intwatab[ANZ_W_A + 1];

#ifdef TIMEDEPGRAV
static MyFloat dHtab[ANZ_W_A_IN];
static MyFloat dGtab[ANZ_W_A_IN];
static MyFloat intdHtab[ANZ_W_A + 1];
static MyFloat intdGtab[ANZ_W_A + 1];
#endif
#ifdef EXTERNALHUBBLE
static MyFloat Htab[ANZ_W_A_IN];
static MyFloat intHtab[ANZ_W_A + 1];
#endif

/* set-up table with Exp(-Integral_a^1 (q+w(a'))/a' da'
 * needed in hubble function for time dependent w
 */

void fwa_init(void)
{
  int count = 0, i;
  char buf[200], buf1[200], buf2[200];

#ifdef TIMEDEPGRAV
  char buf3[200], buf4[200];
#endif
#ifdef EXTERNALHUBBLE
  char buf5[200];
#endif
  FILE *fd;
  MyFloat a_first, w_first, a, w, sum;


  if(ThisTask == 0)
    {
      printf("initialize time dependent w ...\n");
      fflush(stdout);
    }

  if((fd = fopen(All.DarkEnergyFile, "r")))
    {
      if(ThisTask == 0)
	{
	  printf("\nreading w of a from file `%s'\n", All.DarkEnergyFile);
	  fflush(stdout);
	}
      atab[0] = -1.0;		/* we have to extrapolate wtab[0] later ! */
      count = 1;
      while(!feof(fd) && count < ANZ_W_A_IN)
	{
	  if(fgets(buf, 200, fd))
	    {
#ifdef TIMEDEPGRAV
#ifdef EXTERNALHUBBLE
	      if(sscanf(buf, "%s%s%s%s%s", buf1, buf2, buf3, buf4, buf5) < 5)
#else
	      if(sscanf(buf, "%s%s%s%s", buf1, buf2, buf3, buf4) < 4)
#endif
#else
	      if(sscanf(buf, "%s%s", buf1, buf2) < 2)
#endif
		{
		  if(ThisTask == 0)
		    {
		      printf("Wrong syntax in file `%s', line %d\n", All.DarkEnergyFile, count);
		      fflush(stdout);
		    }
		  endrun(0);
		}
	      a = 1. / atof(buf1);
	      if(a == 0.0 && count == 1)
		count--;	/* w(0) present in file, so fill the first element ! */
	      atab[count] = a;
	      wtab[count] = atof(buf2);
#ifdef TIMEDEPGRAV
	      dHtab[count] = atof(buf3);
	      dGtab[count] = atof(buf4);
#endif
#ifdef EXTERNALHUBBLE
	      Htab[count] = atof(buf5);
#endif
	      count++;
	    }
	}
      fclose(fd);
      if(count >= ANZ_W_A_IN - 1)
	{
	  if(ThisTask == 0)
	    {
	      printf("File `%s' contains to many datapoints, increase ANZ_W_A_IN !\n", All.DarkEnergyFile);
	      fflush(stdout);
	    }
	  endrun(0);
	}
      if(count <= 2)
	{
	  if(ThisTask == 0)
	    {
	      printf("File `%s' has to less Data Points (%d) !\n", All.DarkEnergyFile, count);
	      fflush(stdout);
	    }
	  endrun(0);
	}

      if(atab[0] < 0.)		/* We still have to extrapolate w to a = 0 (w[0]) */
	{
	  atab[0] = 0.;
	  wtab[0] = wtab[1] - (wtab[2] - wtab[1]) / (atab[2] - atab[1]) * (atab[1] - atab[0]);
#ifdef TIMEDEPGRAV
	  dHtab[0] = dHtab[1] - (dHtab[2] - dHtab[1]) / (atab[2] - atab[1]) * (atab[1] - atab[0]);
	  dGtab[0] = dGtab[1] - (dGtab[2] - dGtab[1]) / (atab[2] - atab[1]) * (atab[1] - atab[0]);
#endif
#ifdef EXTERNALHUBBLE
	  Htab[0] = Htab[1] - (Htab[2] - Htab[1]) / (atab[2] - atab[1]) * (atab[1] - atab[0]);
#endif
	}

/* Setp back if tables go bejond a=1 */
      while(atab[count - 1] > 1.0)
	count--;

/* Calculate w(1) if needed */
      if(atab[count - 1] < 1.)
	{
	  atab[count] = 1.0;
	  wtab[count] = wtab[count - 1] + (wtab[count - 1] - wtab[count - 2])
	    / (atab[count - 1] - atab[count - 2]) * (1. - atab[count - 1]);
#ifdef TIMEDEPGRAV
	  dHtab[count] = dHtab[count - 1] + (dHtab[count - 1] - dHtab[count - 2])
	    / (atab[count - 1] - atab[count - 2]) * (1. - atab[count - 1]);
	  dGtab[count] = dGtab[count - 1] + (dGtab[count - 1] - dGtab[count - 2])
	    / (atab[count - 1] - atab[count - 2]) * (1. - atab[count - 1]);
#endif
#ifdef EXTERNALHUBBLE
	  Htab[count] = Htab[count - 1] + (Htab[count - 1] - Htab[count - 2])
	    / (atab[count - 1] - atab[count - 2]) * (1. - atab[count - 1]);
#endif
/*            if(ThisTask ==0) 
              {
                printf("%d %f %f %f\n",count,atab[count-2],atab[count-1],atab[count]);
                printf("%d %f %f %f\n",count,wtab[count-2],wtab[count-1],wtab[count]);
              }*/
	  count++;
	}

/* Now calculated the integral (starting from last to first to save Time !
 * Explicit asume that a[0]=0. and a[count-1]=1. , which is enshured by   
 * the loading precedure !                                                  */

/* Set todays values in the tables */
      intwtab[ANZ_W_A] = All.OmegaLambda;
      intwatab[ANZ_W_A] = wtab[count - 1];
#ifdef TIMEDEPGRAV
      intdHtab[ANZ_W_A] = dHtab[count - 1];
      intdGtab[ANZ_W_A] = dGtab[count - 1];
#endif
#ifdef EXTERNALHUBBLE
      intHtab[ANZ_W_A] = Htab[count - 1];
#endif

/* Place count on last entry in table */
      count--;

      a_first = atab[count];	/* Startinv value should be 1.0 ! */
      w_first = wtab[count];	/* Starting value from table ! */
      sum = 0.0;		/* set int to 0.0 */
      for(i = ANZ_W_A - 1; i >= 1; i--)
	{
	  a = (MyFloat) i / (MyFloat) ANZ_W_A;
	  if(count > 1)		/* Still inside the table */
	    {
	      while(atab[count - 1] > a && count > 0)
		{
		  sum += 0.5 * ((1. + w_first) / a_first + (1. + wtab[count - 1]) / atab[count - 1])
		    * (a_first - atab[count - 1]);
		  count--;
		  a_first = atab[count];
		  w_first = wtab[count];
		}
	      w = w_first - (wtab[count] - wtab[count - 1]) / (atab[count] - atab[count - 1]) * (a_first - a);
	      sum += 0.5 * ((1. + w_first) / a_first + (1. + w) / a) * (a_first - a);
	      w_first = w;
	      a_first = a;
	    }
	  else
	    {
	      w = w_first - (wtab[count] - wtab[count - 1]) / (atab[count] - atab[count - 1]) * (a_first - a);
	      sum += 0.5 * ((1. + w_first) / a_first + (1. + w) / a) * (a_first - a);
	      w_first = w;
	      a_first = a;
	    }
	  intwtab[i] = All.OmegaLambda * exp(3. * sum);
	  intwatab[i] = wtab[count - 1] + (wtab[count] - wtab[count - 1]) /
	    (atab[count] - atab[count - 1]) * (a - atab[count - 1]);
#ifdef TIMEDEPGRAV
	  intdHtab[i] = dHtab[count - 1] + (dHtab[count] - dHtab[count - 1]) /
	    (atab[count] - atab[count - 1]) * (a - atab[count - 1]);
	  intdGtab[i] = dGtab[count - 1] + (dGtab[count] - dGtab[count - 1]) /
	    (atab[count] - atab[count - 1]) * (a - atab[count - 1]);
#endif
#ifdef EXTERNALHUBBLE
	  intHtab[i] = Htab[count - 1] + (Htab[count] - Htab[count - 1]) /
	    (atab[count] - atab[count - 1]) * (a - atab[count - 1]);
#endif
	}
      /* artificially define value for a=0 */
      intwtab[0] = intwtab[1];
      intwatab[0] = intwatab[1];
#ifdef TIMEDEPGRAV
      intdHtab[0] = intdHtab[1];
      intdGtab[0] = intdGtab[1];
#endif
#ifdef EXTERNALHUBBLE
      intHtab[0] = intHtab[1];
#endif
    }
  else
    {
      if(ThisTask == 0)
	{
	  printf("\nFile `%s' not found !\n", All.DarkEnergyFile);
	  fflush(stdout);
	}
      endrun(0);
    }

  if(ThisTask == 0)
    {
      printf("Integrating w(a) finisched.\n");
      fflush(stdout);
    }
}

/* This function the integral w(a) therm for the actual time
 * needed in the hubble function.
 */
double INLINE_FUNC fwa(double a)
{
  int ai;
  double fwa = 0.0;

  ai = a * ANZ_W_A;
  if(ai >= ANZ_W_A)
    {
      fwa = intwtab[ANZ_W_A];
    }
  else
    {
      fwa = intwtab[ai] + (intwtab[ai + 1] - intwtab[ai]) * (a * (double) ANZ_W_A - (double) ai);
    }
/*   if(ThisTask==0) printf("%f %f %f %f\n",a,intwtab[ai],fwa,intwtab[ai+1]);*/
  return (fwa);
}

#endif



double DarkEnergy_a(double a)	/* only needed for comoving integration */
{
#ifdef TIMEDEPDE
  return fwa(a);
#else
  return (All.OmegaLambda * pow(a, -3. * (1 + All.DarkEnergyParam)));
#endif
}


double DarkEnergy_t(double Time)	/* only needed for physical integration */
{
  return All.DarkEnergyParam;
}

#ifdef TIMEDEPDE

/* This function returns the interpolated equation of state parameter.
This is only used for information in the log files.
 */
double INLINE_FUNC get_wa(double a)
{
  int ai;
  double fw = 0.0;

  ai = a * ANZ_W_A;
  if(ai >= ANZ_W_A)
    {
      fw = intwatab[ANZ_W_A];
    }
  else
    {
      fw = intwatab[ai] + (intwatab[ai + 1] - intwatab[ai]) * (a * (double) ANZ_W_A - (double) ai);
    }
  return (fw);
}

#ifdef TIMEDEPGRAV

/* This function returns the interpolated correction for the Hubble function
 */
double INLINE_FUNC dHfak(double a)
{
  int ai;
  double fdH = 0.0;

  ai = a * ANZ_W_A;
  if(ai >= ANZ_W_A)
    {
      fdH = intdHtab[ANZ_W_A];
    }
  else
    {
      fdH = intdHtab[ai] + (intdHtab[ai + 1] - intdHtab[ai]) * (a * (double) ANZ_W_A - (double) ai);
    }
  return (fdH);
}

/* This function returns the interpolated correction for the Gravitational constant
 */
double INLINE_FUNC dGfak(double a)
{
  int ai;
  double fdG = 0.0;

  ai = a * ANZ_W_A;
  if(ai >= ANZ_W_A)
    {
      fdG = intdGtab[ANZ_W_A];
    }
  else
    {
      fdG = intdGtab[ai] + (intdGtab[ai + 1] - intdGtab[ai]) * (a * (double) ANZ_W_A - (double) ai);
    }
  return (fdG);
}

#endif

#ifdef EXTERNALHUBBLE

/* This function returns the interpolated correction for the Hubble function
 */
double INLINE_FUNC hubble_function_external(double a)
{
  int ai;
  double H = 0.0;

  ai = a * ANZ_W_A;
  if(ai >= ANZ_W_A)
    {
      H = intHtab[ANZ_W_A];
    }
  else
    {
      H = intHtab[ai] + (intHtab[ai + 1] - intHtab[ai]) * (a * (double) ANZ_W_A - (double) ai);
    }
  return (H);
}
#endif

#endif
#endif

double INLINE_FUNC hubble_function(double a)
{
  double hubble_a;

#ifdef EXTERNALHUBBLE
  hubble_a = hubble_function_external(a);
#else
  hubble_a = All.Omega0 / (a * a * a) + (1 - All.Omega0 - All.OmegaLambda) / (a * a)
#ifdef DARKENERGY
    + DarkEnergy_a(a);
#else
    + All.OmegaLambda;
#endif
  hubble_a = All.Hubble * sqrt(hubble_a);
#endif
#ifdef TIMEDEPGRAV
  hubble_a *= dHfak(a);
#endif
  return (hubble_a);
}
