#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <gsl/gsl_integration.h>
#include "allvars.h"
#include "proto.h"


static double R8;
static double r_tophat;

static double AA, BB, CC;
static double nu;
static double Norm;


static int NPowerTable;

static struct pow_table
{
  double logk, logD;
}
 *PowerTable;


double PowerSpec(double k)
{
  double power, alpha, Tf;

  switch (WhichSpectrum)
    {
    case 1:
      power = PowerSpec_EH(k);
      break;

    case 2:
      power = PowerSpec_Tabulated(k);
      break;

    default:
      power = PowerSpec_Efstathiou(k);
      break;
    }


  if(WDM_On == 1)
    {
      /* Eqn. (A9) in Bode, Ostriker & Turok (2001), assuming gX=1.5  */
      alpha =
	0.048 * pow((Omega - OmegaBaryon) / 0.4, 0.15) * pow(HubbleParam / 0.65,
							     1.3) * pow(1.0 / WDM_PartMass_in_kev, 1.15);
      Tf = pow(1 + pow(alpha * k * (3.085678e24 / UnitLength_in_cm), 2 * 1.2), -5.0 / 1.2);
      power *= Tf * Tf;
    }

#if defined(MULTICOMPONENTGLASSFILE) && defined(DIFFERENT_TRANSFER_FUNC)

  if(Type == 2)
    {
      power = PowerSpec_DM_2ndSpecies(k);
    }

#endif

  if(WhichSpectrum != 2) {
    /* because a tabulated power is already tilted */
    //printf("PrimordialIndex =%g is not used for Table Power spectrum\n", PrimordialIndex);
    power *= pow(k, PrimordialIndex - 1.0);
  }

  return power;
}


double PowerSpec_DM_2ndSpecies(double k)
{
  /* at the moment, we simply call the Eistenstein & Hu spectrum
   * for the second DM species, but this could be replaced with
   * something more physical, say for neutrinos
   */

  double power;

  power = Norm * k * pow(tk_eh(k), 2);

  return power;
}



void read_power_table(void)
{
    FILE *fd;
    char buf[500];
    double k, p;


    strcpy(buf, FileWithInputSpectrum);

    if(ThisTask == 0) {
        if(!(fd = fopen(buf, "r")))
        {
            printf("can't read input spectrum in file '%s' on task %d\n", buf, ThisTask);
            FatalError(17);
        }

        NPowerTable = 0;
        do
        {
            if(fscanf(fd, " %lg %lg ", &k, &p) == 2)
                NPowerTable++;
            else
                break;
        }
        while(1);

        fclose(fd);

        printf("found %d pairs of values in input spectrum table\n", NPowerTable);
        fflush(stdout);
    }
    MPI_Bcast(&NPowerTable, 1, MPI_INT, 0, MPI_COMM_WORLD);

    PowerTable = malloc(NPowerTable * sizeof(struct pow_table));

    if(ThisTask == 0) {
        int i = 0;
        sprintf(buf, FileWithInputSpectrum);

        if(!(fd = fopen(buf, "r")))
        {
            printf("can't read input spectrum in file '%s' on task %d\n", buf, ThisTask);
            FatalError(18);
        }

        i = 0;
        do
        {
            if(fscanf(fd, " %lg %lg ", &k, &p) == 2)
            {
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
  double res;

  InitTime = 1 / (1 + Redshift);

  AA = 6.4 / ShapeGamma * (3.085678e24 / UnitLength_in_cm);
  BB = 3.0 / ShapeGamma * (3.085678e24 / UnitLength_in_cm);
  CC = 1.7 / ShapeGamma * (3.085678e24 / UnitLength_in_cm);
  nu = 1.13;

  R8 = 8 * (3.085678e24 / UnitLength_in_cm);	/* 8 Mpc/h */

  if(WhichSpectrum == 2)
    read_power_table();

#ifdef DIFFERENT_TRANSFER_FUNC
  Type = 1;
#endif

  Norm = 1.0;
  res = TopHatSigma2(R8);

  if(ThisTask == 0 && WhichSpectrum == 2)
    printf("\nNormalization of spectrum in file:  Sigma8 = %g\n", sqrt(res));

  if(ThisTask == 0 && WhichSpectrum == 1)
    printf("\nNormalization of spectrum in file:  Sigma8 = %g\n", sqrt(res));

  Norm = Sigma8 * Sigma8 / res;

  if(ThisTask == 0 && WhichSpectrum == 2)
    printf("Normalization adjusted to  Sigma8=%g   (Normfac=%g)\n\n", Sigma8, Norm);

  if(ThisTask == 0 && WhichSpectrum == 1)
    printf("Normalization adjusted to  Sigma8=%g   (Normfac=%g)\n\n", Sigma8, Norm);

  Dplus = GrowthFactor(InitTime, 1.0);
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

  /*  
  binlow = 0;
  binhigh = NPowerTable - 1;

  while(binhigh - binlow > 1)
    {
      binmid = (binhigh + binlow) / 2;
      if(logk < PowerTable[binmid].logk)
	binhigh = binmid;
      else
	binlow = binmid;
    }

  dlogk = PowerTable[binhigh].logk - PowerTable[binlow].logk;

  if(dlogk == 0)
    FatalError(777);


  u = (logk - PowerTable[binlow].logk) / dlogk;

  logD = (1 - u) * PowerTable[binlow].logD + u * PowerTable[binhigh].logD;
  */



  dlogk_PowerTable = PowerTable[1].logk-PowerTable[0].logk;
  mydlogk = logk - PowerTable[0].logk;
  mybinlow = (int)(mydlogk/dlogk_PowerTable);
  mybinhigh = mybinlow+1;

  dlogk = PowerTable[mybinhigh].logk - PowerTable[mybinlow].logk;

  if(dlogk == 0)
    FatalError(777);

  u = (logk - PowerTable[mybinlow].logk) / dlogk;

  logD = (1 - u) * PowerTable[mybinlow].logD + u * PowerTable[mybinhigh].logD;

  P = Norm*pow(10.0, logD);//*2*M_PI*M_PI;

  //  Delta2 = pow(10.0, logD);

  //  P = Norm * Delta2 / (4 * M_PI * kold * kold * kold);

  //  if(ThisTask == 0)
  //    printf("%lg %lg %d %d %d %d\n",k,P,binlow,binhigh,mybinlow,mybinhigh);

  return P;
}

double PowerSpec_Efstathiou(double k)
{
  return Norm * k / pow(1 + pow(AA * k + pow(BB * k, 1.5) + CC * CC * k * k, nu), 2 / nu);
}



double PowerSpec_EH(double k)	/* Eisenstein & Hu */
{
  return Norm * k * pow(tk_eh(k), 2);
}




double tk_eh(double k)		/* from Martin White */
{
  double q, theta, ommh2, a, s, gamma, L0, C0;
  double tmp;
  double omegam, ombh2, hubble;

  /* other input parameters */
  hubble = HubbleParam;

  omegam = Omega;
  ombh2 = OmegaBaryon * HubbleParam * HubbleParam;

  if(OmegaBaryon == 0)
    ombh2 = 0.044 * HubbleParam * HubbleParam;

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
  r_tophat = R;

  gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
  double result,abserr;
  gsl_function F;
  F.function = &sigma2_int;
  F.params = NULL;

  /* note: 500/R is here chosen as integration boundary (infinity) */
  gsl_integration_qags (&F, 0, 500. / R, 0, 1e-4,1000,w,&result, &abserr);
//   printf("gsl_integration_qng in TopHatSigma2. Result %g, error: %g, intervals: %lu\n",result, abserr,w->size);
  gsl_integration_workspace_free (w);
  return result;
}


double sigma2_int(double k, void * params)
{
  double kr, kr3, kr2, w, x;

  kr = r_tophat * k;
  kr2 = kr * kr;
  kr3 = kr2 * kr;

  if(kr < 1e-8)
    return 0;

  w = 3 * (sin(kr) / kr3 - cos(kr) / kr2);
  x = 4 * PI * k * k * w * w * PowerSpec(k);

  return x;

}


double GrowthFactor(double astart, double aend)
{
  return growth(aend) / growth(astart);
}


double growth(double a)
{
  gsl_integration_workspace * w = gsl_integration_workspace_alloc (200);
  double hubble_a;
  double result,abserr;
  gsl_function F;
  F.function = &growth_int;

  hubble_a = sqrt(Omega / (a * a * a) + (1 - Omega - OmegaLambda) / (a * a) + OmegaLambda);

  gsl_integration_qag (&F, 0, a, 0, 1e-4,200,GSL_INTEG_GAUSS61, w,&result, &abserr);
//   printf("gsl_integration_qng in growth. Result %g, error: %g, intervals: %lu\n",result, abserr,w->size);
  gsl_integration_workspace_free (w);
  return hubble_a * result;
}


double growth_int(double a, void * params)
{
  return pow(a / (Omega + (1 - Omega - OmegaLambda) * a + OmegaLambda * a * a * a), 1.5);
}


double F_Omega(double a)
{
  double omega_a;

  omega_a = Omega / (Omega + a * (1 - Omega - OmegaLambda) + a * a * a * OmegaLambda);

  return pow(omega_a, 0.6);
}

/*  Here comes the stuff to compute the thermal WDM velocity distribution */


#define LENGTH_FERMI_DIRAC_TABLE 2000
#define MAX_FERMI_DIRAC          20.0

double fermi_dirac_vel[LENGTH_FERMI_DIRAC_TABLE];
double fermi_dirac_cumprob[LENGTH_FERMI_DIRAC_TABLE];

double WDM_V0 = 0;

double fermi_dirac_kernel(double x, void * params)
{
  return x * x / (exp(x) + 1);
}

void fermi_dirac_init(void)
{
  int i;

  /*These functions are so smooth that we don't need much space*/
  gsl_integration_workspace * w = gsl_integration_workspace_alloc (100);
  double abserr;
  gsl_function F;
  F.function = &fermi_dirac_kernel;
  F.params = NULL;

  for(i = 0; i < LENGTH_FERMI_DIRAC_TABLE; i++)
    {
      fermi_dirac_vel[i] = MAX_FERMI_DIRAC * i / (LENGTH_FERMI_DIRAC_TABLE - 1.0);
      gsl_integration_qag (&F, 0, fermi_dirac_vel[i], 0, 1e-6,100,GSL_INTEG_GAUSS61, w,&(fermi_dirac_cumprob[i]), &abserr);
    }

  gsl_integration_workspace_free (w);
  for(i = 0; i < LENGTH_FERMI_DIRAC_TABLE; i++)
    fermi_dirac_cumprob[i] /= fermi_dirac_cumprob[LENGTH_FERMI_DIRAC_TABLE - 1];

  WDM_V0 = 0.012 * (1 + Redshift) * pow((Omega - OmegaBaryon) / 0.3, 1.0 / 3) * pow(HubbleParam / 0.65,
										    2.0 / 3) * pow(1.0 /
												   WDM_PartMass_in_kev,
												   4.0 / 3);

  if(ThisTask == 0)
    printf("\nWarm dark matter rms velocity dispersion at starting redshift = %g km/sec\n\n",
	   3.59714 * WDM_V0);

  WDM_V0 *= 1.0e5 / UnitVelocity_in_cm_per_s;

  /* convert from peculiar velocity to gadget's cosmological velocity */
  WDM_V0 *= sqrt(1 + Redshift);
}



double get_fermi_dirac_vel(void)
{
  int i;
  double p, u;

  p = drand48();
  i = 0;

  while(i < LENGTH_FERMI_DIRAC_TABLE - 2)
    if(p > fermi_dirac_cumprob[i + 1])
      i++;
    else
      break;

  u = (p - fermi_dirac_cumprob[i]) / (fermi_dirac_cumprob[i + 1] - fermi_dirac_cumprob[i]);

  return fermi_dirac_vel[i] * (1 - u) + fermi_dirac_vel[i + 1] * u;
}



void add_WDM_thermal_speeds(float *vel)
{
  double v, phi, theta, vx, vy, vz;

  if(WDM_V0 == 0)
    fermi_dirac_init();

  v = WDM_V0 * get_fermi_dirac_vel();

  phi = 2 * M_PI * drand48();
  theta = acos(2 * drand48() - 1);

  vx = v * sin(theta) * cos(phi);
  vy = v * sin(theta) * sin(phi);
  vz = v * cos(theta);

  vel[0] += vx;
  vel[1] += vy;
  vel[2] += vz;
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

      DDD = GrowthFactor(1.0 / (Redshift + 1), 1.0);

      fprintf(fd, "%12g %12g %12g\n", Redshift, DDD, Norm);	/* print actual starting redshift and 
							   linear growth factor for this cosmology */
      kstart = 2 * PI / (1000.0 * (3.085678e24 / UnitLength_in_cm));	/* 1000 Mpc/h */
      kend = 2 * PI / (0.001 * (3.085678e24 / UnitLength_in_cm));	/* 0.001 Mpc/h */

      printf("kstart=%lg kend=%lg\n",kstart,kend);

      for(k = kstart; k < kend; k *= 1.025)
	{
	  po = PowerSpec(k);
	  dl = 4.0 * PI * k * k * k * po;

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
