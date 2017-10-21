#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp.h>
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

static double Norm;

#define MAXCOLS 13
struct table
{
    int Nentry;
    double * logk;
    double * logD[MAXCOLS];
    gsl_interp * mat_intp[MAXCOLS];
    gsl_interp_accel * mat_intp_acc[MAXCOLS];
};
static struct table power_table;
static struct table transfer_table;

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

void read_power_table(const char * inputfile, const int ncols, struct table * out_tab)
{
    FILE *fd = NULL;
    int j;
    int InputInLog10 = 0;

    if(ThisTask == 0) {
        if(!(fd = fopen(inputfile, "r")))
            endrun(1, "can't read input spectrum in file '%s' on task %d\n", inputfile, ThisTask);

        out_tab->Nentry = 0;
        do
        {
            char buffer[1024];
            char * retval = fgets(buffer, 1024, fd);
            /*Happens on end of file*/
            if(!retval)
                break;
            retval = strtok(buffer, " \t");
            if(!retval || retval[0] == '#')
                continue;
            out_tab->Nentry++;
        }
        while(1);

        message(1, "found %d pairs of values in input spectrum table\n", out_tab->Nentry);
        rewind(fd);
    }
    MPI_Bcast(&(out_tab->Nentry), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(out_tab->Nentry < 2)
        endrun(1, "Input spectrum too short\n");
    out_tab->logk = mymalloc("Powertable", (ncols+1)*out_tab->Nentry * sizeof(double));
    for(j=0; j<ncols; j++)
        out_tab->logD[j] = out_tab->logk + (j+1)*out_tab->Nentry;

    if(ThisTask == 0)
    {
        int i = 0;
        do
        {
            double k, p;
            char buffer[1024];
            char * retval = fgets(buffer, 1024, fd);
            /*Happens on end of file*/
            if(!retval)
                break;
            retval = strtok(buffer, " \t");
            if(!retval || retval[0] == '#')
                continue;
            k = atof(retval);
            if(!InputInLog10) {
                if(k < 0) {
                    message(1, "some input k is negative, guessing the file is in log10 units\n");
                    InputInLog10 = 1;
                }
                else
                    k = log10(k);
            }
            k -= log10(InputSpectrum_UnitLength_in_cm / UnitLength_in_cm);	/* convert to h/Kpc */
            out_tab->logk[i] = k;
            for(j=0; j<ncols;j++) {
                retval = strtok(NULL, " \t");
                if(!retval)
                    endrun(1,"Incomplete line in power spectrum: %s\n",buffer);
                p = atof(retval);
                out_tab->logk[i] = k;
                if (!InputInLog10)
                    p = log10(p);
                p += 3 * log10(InputSpectrum_UnitLength_in_cm / UnitLength_in_cm);	/* convert to Kpc/h  */
                out_tab->logD[j][i] = p;
            }
            i++;
        }
        while(1);

        fclose(fd);
    }

    MPI_Bcast(out_tab->logk, (ncols+1)*out_tab->Nentry, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for(j=0; j<ncols; j++) {
        out_tab->mat_intp[j] = gsl_interp_alloc(gsl_interp_cspline,out_tab->Nentry);
        out_tab->mat_intp_acc[j] = gsl_interp_accel_alloc();
        gsl_interp_init(out_tab->mat_intp[j],out_tab->logk, out_tab->logD[j],out_tab->Nentry);
    }
}

void initialize_powerspectrum(void)
{
    if(WhichSpectrum == 2) {
        read_power_table(FileWithInputSpectrum, 1, &power_table);
    }

    Norm = 1.0;
    if (Sigma8 > 0) {
        double R8 = 8 * (3.085678e24 / UnitLength_in_cm);	/* 8 Mpc/h */
        double res = TopHatSigma2(R8);
        Norm = Sigma8 * Sigma8 / res;
        message(0, "Normalization adjusted to  Sigma8=%g   (Normfac=%g). \n", Sigma8, Norm);
    }
    if(InputPowerRedshift >= 0) {
        double Dplus = GrowthFactor(InitTime, 1/(1+InputPowerRedshift));
        Norm *= (Dplus * Dplus);
        message(0,"Growth factor to z=0: %g \n", Dplus);
    }

}

double PowerSpec_Tabulated(double k)
{
  const double logk = log10(k);

  if(logk < power_table.logk[0] || logk > power_table.logk[power_table.Nentry - 1])
    return 0;

  const double logD = gsl_interp_eval(power_table.mat_intp[0], power_table.logk, power_table.logD[0], logk, power_table.mat_intp_acc[0]);

  double power = pow(10.0, logD);//*2*M_PI*M_PI;

  //  Delta2 = pow(10.0, logD);

  //  P = Norm * Delta2 / (4 * M_PI * kold * kold * kold);

  //  if(ThisTask == 0)
  //    printf("%lg %lg %d %d %d %d\n",k,P,binlow,binhigh,mybinlow,mybinhigh);

  return power;
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
/*   printf("gsl_integration_qng in TopHatSigma2. Result %g, error: %g, intervals: %lu\n",result, abserr,w->size); */
  gsl_integration_workspace_free (w);
  return result;
}


double sigma2_int(double k, void * params)
{
  double w, x;

  double r_tophat = *(double *) params;
  const double kr = r_tophat * k;
  const double kr2 = kr * kr;

  /*Series expansion; actually good until kr~1*/
  if(kr < 1e-3)
      w = 1./3. - kr2/30. +kr2*kr2/840.;
  else
      w = 3 * (sin(kr) / kr - cos(kr)) / kr2;
  x = 4 * M_PI / (2 * M_PI * 2 * M_PI * 2 * M_PI) * k * k * w * w * PowerSpec(k);

  return x;

}

void print_spec(void)
{
  if(ThisTask == 0)
    {
      double k, po, dl, kstart, kend, DDD;
      char buf[1000];
      FILE *fd;

      sprintf(buf, "%s/inputspec_%s.txt", OutputDir, FileBase);
      
      fd = fopen(buf, "w");
      if (fd == NULL) {
          printf("Failed to create powerspec file at:%s\n", buf);
        return;
      }
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
	    fprintf(fd, "%12g %12g\n", k, dl);
	  }
      fclose(fd);
    }
}
