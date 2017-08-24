#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <gsl/gsl_integration.h>
#include "genic/allvars.h"
#include "genic/proto.h"
#include "cosmology.h"
#include "mymalloc.h"
#include "endrun.h"

static double PowerSpec_EH(double k);
static double PowerSpec_Tabulated(double k);
static double sigma2_int(double k, void * params);
static double TopHatSigma2(double R);
static double tk_eh(double k);
static int compare_logk(const void *a, const void *b);


static double Norm;

static int NPowerTable;

static struct pow_table
{
  double logk, logD;
}
 *PowerTable;

double PowerSpec(double k)
{
  double power;

  switch (WhichSpectrum)
  {
    case 2:
      power = PowerSpec_Tabulated(k);
      break;

    default:
      power = PowerSpec_EH(k);
      break;
  }

  /*Normalise the power spectrum*/
  power *= Norm;

  return power;
}

void read_power_table(void)
{
    FILE *fd;
    char buf[500];
    double k, p;

    int InputInLog10 = 0;

    strcpy(buf, FileWithInputSpectrum);

    if(ThisTask == 0) {
        if(!(fd = fopen(buf, "r")))
        {
            endrun(1, "can't read input spectrum in file '%s' on task %d\n", buf, ThisTask);
        }

        NPowerTable = 0;
        do
        {
            if(fscanf(fd, " %lg %lg ", &k, &p) == 2) {
                if(k < 0 && ! InputInLog10) {
                    message(1, "some input k is negative, guessing the file is in log10 units\n");
                    InputInLog10 = 1;
                }
                NPowerTable++;
            } else
                break;
        }
        while(1);

        fclose(fd);

        message(1, "found %d pairs of values in input spectrum table\n", NPowerTable);
    }
    MPI_Bcast(&NPowerTable, 1, MPI_INT, 0, MPI_COMM_WORLD);

    PowerTable = mymalloc("Powertable", NPowerTable * sizeof(struct pow_table));

    if(ThisTask == 0) {
        int i = 0;
        sprintf(buf, FileWithInputSpectrum);

        if(!(fd = fopen(buf, "r")))
        {
            endrun(1, "can't read input spectrum in file '%s' on task %d\n", buf, ThisTask);
        }

        i = 0;
        do
        {
            if(fscanf(fd, " %lg %lg ", &k, &p) == 2)
            {
                if (!InputInLog10) {
                    k = log10(k);
                    p = log10(p);
                }

                k -= log10(InputSpectrum_UnitLength_in_cm / UnitLength_in_cm);	/* convert to h/Kpc */
                PowerTable[i].logk = k;

                p += 3 * log10(InputSpectrum_UnitLength_in_cm / UnitLength_in_cm);	/* convert to Kpc/h  */
                PowerTable[i].logD = p;
                i++;
            }
            else
                break;
        }
        while(1);

        fclose(fd);

    }

    MPI_Bcast(PowerTable, NPowerTable * sizeof(struct pow_table), MPI_BYTE, 0, MPI_COMM_WORLD);
    qsort(PowerTable, NPowerTable, sizeof(struct pow_table), compare_logk);
}

int compare_logk(const void *a, const void *b)
{
  if(((struct pow_table *) a)->logk < (((struct pow_table *) b)->logk))
    return -1;

  if(((struct pow_table *) a)->logk > (((struct pow_table *) b)->logk))
    return +1;

  return 0;
}

void initialize_powerspectrum(void)
{
    if(WhichSpectrum == 2)
        read_power_table();

    Norm = 1.0;
    if (Sigma8 > 0) {
        double R8 = 8 * (3.085678e24 / UnitLength_in_cm);	/* 8 Mpc/h */
        double res = TopHatSigma2(R8);
        Norm = Sigma8 * Sigma8 / res;
        message(0, "Normalization adjusted to  Sigma8=%g   (Normfac=%g). \n", Sigma8, Norm);
    }
    if(InputPowerRedshift >= 0) {
        double Dplus = GrowthFactor(InitTime, 1/(1+InputPowerRedshift));
        Norm /= sqrt(Dplus);
        message(0,"Growth factor to z=0: %g \n", Dplus);
    }

}

double PowerSpec_Tabulated(double k)
{
  double logk, logD, P, u, dlogk; // Delta2; //kold
  
  double mydlogk,dlogk_PowerTable;
  int mybinhigh,mybinlow;
  


//   kold = k;

  logk = log10(k);

  if(logk < PowerTable[0].logk || logk > PowerTable[NPowerTable - 1].logk)
    return 0;


  dlogk_PowerTable = PowerTable[1].logk-PowerTable[0].logk;
  mydlogk = logk - PowerTable[0].logk;
  mybinlow = (int)(mydlogk/dlogk_PowerTable);
  mybinhigh = mybinlow+1;

  dlogk = PowerTable[mybinhigh].logk - PowerTable[mybinlow].logk;

  if(dlogk == 0)
    endrun(1, "dlogk is 0");

  u = (logk - PowerTable[mybinlow].logk) / dlogk;

  logD = (1 - u) * PowerTable[mybinlow].logD + u * PowerTable[mybinhigh].logD;

  P = pow(10.0, logD);//*2*M_PI*M_PI;

  //  Delta2 = pow(10.0, logD);

  //  P = Norm * Delta2 / (4 * M_PI * kold * kold * kold);

  //  if(ThisTask == 0)
  //    printf("%lg %lg %d %d %d %d\n",k,P,binlow,binhigh,mybinlow,mybinhigh);

  return P;
}

double PowerSpec_EH(double k)	/* Eisenstein & Hu */
{
  return k * pow(tk_eh(k), 2)* pow(k, PrimordialIndex - 1.0);
}


double tk_eh(double k)		/* from Martin White */
{
  double q, theta, ommh2, a, s, gamma, L0, C0;
  double tmp;
  double omegam, ombh2, hubble;

  /* other input parameters */
  hubble = CP.HubbleParam;

  omegam = CP.Omega0;
  ombh2 = CP.OmegaBaryon * CP.HubbleParam * CP.HubbleParam;

  if(CP.OmegaBaryon == 0)
    ombh2 = 0.044 * CP.HubbleParam * CP.HubbleParam;

  k *= (3.085678e24 / UnitLength_in_cm);	/* convert to h/Mpc */

  theta = 2.728 / 2.7;
  ommh2 = omegam * hubble * hubble;
  s = 44.5 * log(9.83 / ommh2) / sqrt(1. + 10. * exp(0.75 * log(ombh2))) * hubble;
  a = 1. - 0.328 * log(431. * ommh2) * ombh2 / ommh2
    + 0.380 * log(22.3 * ommh2) * (ombh2 / ommh2) * (ombh2 / ommh2);
  gamma = a + (1. - a) / (1. + exp(4 * log(0.43 * k * s)));
  gamma *= omegam * hubble;
  q = k * theta * theta / gamma;
  L0 = log(2. * exp(1.) + 1.8 * q);
  C0 = 14.2 + 731. / (1. + 62.5 * q);
  tmp = L0 / (L0 + C0 * q * q);
  return (tmp);
}


double TopHatSigma2(double R)
{
  gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
  double result,abserr;
  gsl_function F;
  F.function = &sigma2_int;
  F.params = &R;

  /* note: 500/R is here chosen as integration boundary (infinity) */
  gsl_integration_qags (&F, 0, 500. / R, 0, 1e-4,1000,w,&result, &abserr);
//   printf("gsl_integration_qng in TopHatSigma2. Result %g, error: %g, intervals: %lu\n",result, abserr,w->size);
  gsl_integration_workspace_free (w);
  return result;
}


double sigma2_int(double k, void * params)
{
  double kr, kr3, kr2, w, x;

  double r_tophat = *(double *) params;
  kr = r_tophat * k;
  kr2 = kr * kr;
  kr3 = kr2 * kr;

  if(kr < 1e-8)
    return 0;

  w = 3 * (sin(kr) / kr3 - cos(kr) / kr2);
  x = 4 * M_PI / (2 * M_PI * 2 * M_PI * 2 * M_PI) * k * k * w * w * PowerSpec(k);

  return x;

}

static double A, B, alpha, beta, V, gf;

double fnl(double x)		/* Peacock & Dodds formula */
{
  return x * pow((1 + B * beta * x + pow(A * x, alpha * beta)) /
		 (1 + pow(pow(A * x, alpha) * gf * gf * gf / (V * sqrt(x)), beta)), 1 / beta);
}

void print_spec(void)
{
  double k, knl, po, dl, dnl, neff, kf, kstart, kend, po2, po1, DDD;
  char buf[1000];
  FILE *fd;

  if(ThisTask == 0)
    {
      sprintf(buf, "%s/inputspec_%s.txt", OutputDir, FileBase);
      
      fd = fopen(buf, "w");
      if (fd == NULL) {
          printf("Failed to create powerspec file at:%s\n", buf);
        return;
      }
      gf = GrowthFactor(0.001, 1.0) / (1.0 / 0.001);

      DDD = GrowthFactor(InitTime, 1.0);

      fprintf(fd, "%12g %12g %12g\n", 1/InitTime-1, DDD, Norm);	/* print actual starting redshift and 
							   linear growth factor for this cosmology */
      kstart = 2 * M_PI / (1000.0 * (3.085678e24 / UnitLength_in_cm));	/* 1000 Mpc/h */
      kend = 2 * M_PI / (0.001 * (3.085678e24 / UnitLength_in_cm));	/* 0.001 Mpc/h */

      printf("kstart=%lg kend=%lg\n",kstart,kend);

      for(k = kstart; k < kend; k *= 1.025)
	{
	  po = PowerSpec(k);
	  dl = 4.0 * M_PI * k * k * k * po;

	  kf = 0.5;

	  po2 = PowerSpec(1.001 * k * kf);
	  po1 = PowerSpec(k * kf);

	  if(po != 0 && po1 != 0 && po2 != 0)
	    {
	      neff = (log(po2) - log(po1)) / (log(1.001 * k * kf) - log(k * kf));

	      if(1 + neff / 3 > 0)
		{
		  A = 0.482 * pow(1 + neff / 3, -0.947);
		  B = 0.226 * pow(1 + neff / 3, -1.778);
		  alpha = 3.310 * pow(1 + neff / 3, -0.244);
		  beta = 0.862 * pow(1 + neff / 3, -0.287);
		  V = 11.55 * pow(1 + neff / 3, -0.423) * 1.2;

		  dnl = fnl(dl);
		  knl = k * pow(1 + dnl, 1.0 / 3);
		}
	      else
		{
		  dnl = 0;
		  knl = 0;
		}
	    }
	  else
	    {
	      dnl = 0;
	      knl = 0;
	    }

	  fprintf(fd, "%12g %12g    %12g %12g\n", k, dl, knl, dnl);
	  //	  printf("%12g %12g    %12g %12g\n", k, dl, knl, dnl);
	}
      fclose(fd);
    }
}
