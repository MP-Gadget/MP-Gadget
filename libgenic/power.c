#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp.h>

#include <libgadget/cosmology.h>
#include <libgadget/utils.h>

#include "power.h"

static double Delta_EH(double k);
static double Delta_Tabulated(double k, int Type);
static double sigma2_int(double k, void * params);
static double TopHatSigma2(double R);
static double tk_eh(double k);

static double Norm;
static int WhichSpectrum;
/*Only used for tk_eh, WhichSpectrum == 0*/
static double PrimordialIndex;
static double UnitLength_in_cm;
static Cosmology * CP;

#define MAXCOLS 5

/*Columns: 0 == baryon, 1 == CDM, 2 == neutrino, 3 == CDM + baryons, 4 == total*/
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
static struct table growth_table;

double DeltaSpec(double k, int Type)
{
  double power;

  if(WhichSpectrum == 2)
      power = Delta_Tabulated(k, Type);
  else
      power = Delta_EH(k);

  /*Normalise the power spectrum*/
  power *= Norm;

  return power;
}

double dlogGrowth(double kmag, int Type)
{
  const double logk = log10(kmag);

  if(logk < growth_table.logk[0] || logk > growth_table.logk[growth_table.Nentry - 1])
      return 1;

  /*Default to total*/
  if(Type < 0 || Type >= MAXCOLS)
      Type = MAXCOLS-1;

  double growth =  gsl_interp_eval(growth_table.mat_intp[Type], growth_table.logk, growth_table.logD[Type], logk, growth_table.mat_intp_acc[Type]);
  if(isinf(growth) || isnan(growth) || growth < 0)
      endrun(1,"Growth function is: %g for k = %g, Type = %d\n",growth, kmag, Type);
  return growth;
}

void parse_power(int i, double k, char * line, struct table *out_tab, int * InputInLog10, double scale)
{
    char * retval;
    if((*InputInLog10) == 0) {
        if(k < 0) {
            message(1, "some input k is negative, guessing the file is in log10 units\n");
            *InputInLog10 = 1;
        }
        else
            k = log10(k);
    }
    k -= log10(scale);	/* convert to h/Kpc */
    out_tab->logk[i] = k;
    retval = strtok(NULL, " \t");
    if(!retval)
        endrun(1,"Incomplete line in power spectrum: %s\n",line);
    double p = atof(retval);
    if ((*InputInLog10) == 0)
        p = log10(p);
    p += 3 * log10(scale);	/* convert to Kpc/h  */
    /*Store delta, square root of power*/
    out_tab->logD[0][i] = p/2;
}

void parse_transfer(int i, double k, char * line, struct table *out_tab, int * InputInLog10, double scale)
{
    int j;
    double transfers[7];
    k = log10(k);
    k -= log10(scale);	/* convert to h/Kpc */
    out_tab->logk[i] = k;
    /*CDM, Baryons, photons, massless nu, massive nu, tot, CDM+bar, other stuff*/
    for(j = 0; j< 7; j++) {
        char * retval = strtok(NULL, " \t");
        if(!retval)
            endrun(1,"Incomplete line in power spectrum: %s\n",line);
        transfers[j] = atof(retval);
    }
    /*Order of the transfer table matches the particle types:
     * 0 is baryons, 1 is CDM, 2 is CMB + baryons, 3 is massive neutrinos.
     * Everything is a ratio against tot.
     * In the input CAMB file 0 is CDM, 1 is baryons, 2 is photons,
     * 3 is massless neutrinos, 4 is massive neutrinos, 5 is total,
     * 6 is cdm + baryons.*/
    out_tab->logD[0][i] = transfers[1];
    out_tab->logD[1][i] = transfers[0];
    out_tab->logD[2][i] = transfers[4];
    out_tab->logD[3][i] = transfers[6];
    out_tab->logD[4][i] = transfers[5];
}

void read_power_table(int ThisTask, const char * inputfile, const int ncols, struct table * out_tab, double scale, void (*parse_line)(int i, double k, char * line, struct table *, int *InputInLog10, double scale))
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
            char buffer[1024];
            char * line = fgets(buffer, 1024, fd);
            /*Happens on end of file*/
            if(!line)
                break;
            char * retval = strtok(line, " \t");
            if(!retval || retval[0] == '#')
                continue;
            double k = atof(retval);
            parse_line(i, k, line, out_tab, &InputInLog10, scale);
            i++;
        }
        while(1);

        fclose(fd);
    }

    MPI_Bcast(out_tab->logk, (ncols+1)*out_tab->Nentry, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for(j=0; j<ncols; j++) {
        out_tab->mat_intp[j] = gsl_interp_alloc(gsl_interp_cspline,out_tab->Nentry);
        out_tab->mat_intp_acc[j] = gsl_interp_accel_alloc();
    }
}

int
initialise_transfer_table(int ThisTask, double InitTime, struct power_params * ppar)
{
    int i, t;
    if(strlen(ppar->FileWithTransferFunction) > 0)
        read_power_table(ThisTask, ppar->FileWithTransferFunction, MAXCOLS, &transfer_table, ppar->SpectrumLengthScale, parse_transfer);
    if(strlen(ppar->FileWithFutureTransferFunction) > 0) {
        double meangrowth[MAXCOLS] = {0};
        /*Read and build growth table*/
        read_power_table(ThisTask, ppar->FileWithFutureTransferFunction, MAXCOLS, &growth_table, ppar->SpectrumLengthScale, parse_transfer);
        if(growth_table.Nentry != transfer_table.Nentry)
            endrun(1,"Transfer tables differ in number of entries: %d %d\n", growth_table.Nentry, transfer_table.Nentry);
        /*Numerically differentiate the transfer functions*/
        double afut = 1/(1.+ppar->InputFutureRedshift);
        double dloga = log(afut) - log(InitTime);
        /*This is d log D[type] / d log a*/
        for(t = 0; t < MAXCOLS; t++) {
        int nmean=0;
            for(i=0; i< growth_table.Nentry; i++) {
                if(growth_table.logD[t][i] > 0)
                    growth_table.logD[t][i] = (log(growth_table.logD[t][i]) - log(transfer_table.logD[t][i]))/dloga;
                if(growth_table.logk[i] > power_table.logk[0]) {
                    meangrowth[t] += growth_table.logD[t][i];
                    nmean++;
                }
            }
            gsl_interp_init(growth_table.mat_intp[t],growth_table.logk, growth_table.logD[t],growth_table.Nentry);
        meangrowth[t]/= nmean;
        }
        message(0,"Scale-dependent growth calculated. Mean = %g %g %g %g\n",meangrowth[0], meangrowth[1], meangrowth[2], meangrowth[3]);
    }
    if(transfer_table.Nentry == 0 || growth_table.Nentry == 0) {
        endrun(1, "Could not read both transfer tables at: '%s' or '%s'\n",ppar->FileWithFutureTransferFunction, ppar->FileWithTransferFunction);
    }

    /*Transform T(k) to to T(k)/T_tot(k)*/
    for(t = 0; t < MAXCOLS-1; t++) {
        for(i=0; i< transfer_table.Nentry; i++)
            transfer_table.logD[t][i] = transfer_table.logD[t][i]/transfer_table.logD[MAXCOLS-1][i];
        gsl_interp_init(transfer_table.mat_intp[t],transfer_table.logk, transfer_table.logD[t],transfer_table.Nentry);
    }
    message(0, "Power spectrum rows: %d, Transfer: %d (%g -> %g) (second transfer: %d)\n", power_table.Nentry, transfer_table.Nentry, growth_table.Nentry, transfer_table.logD[0][0],transfer_table.logD[0][transfer_table.Nentry-1]);
    return transfer_table.Nentry;
}

int initialize_powerspectrum(int ThisTask, double InitTime, double UnitLength_in_cm_in, Cosmology * CPin, struct power_params * ppar)
{
    WhichSpectrum = ppar->WhichSpectrum;
    /*Used only for tk_eh*/
    PrimordialIndex = ppar->PrimordialIndex;
    UnitLength_in_cm = UnitLength_in_cm_in;
    CP = CPin;

    if(ppar->WhichSpectrum == 2) {
        read_power_table(ThisTask, ppar->FileWithInputSpectrum, 1, &power_table, ppar->SpectrumLengthScale, parse_power);
        /*Initialise the interpolator*/
        gsl_interp_init(power_table.mat_intp[0],power_table.logk, power_table.logD[0],power_table.Nentry);
        transfer_table.Nentry = 0;
        growth_table.Nentry = 0;
        if(ppar->DifferentTransferFunctions || ppar->ScaleDepVelocity) {
            initialise_transfer_table(ThisTask, InitTime, ppar);
        }
    }

    Norm = 1.0;
    if (ppar->Sigma8 > 0) {
        double R8 = 8 * (3.085678e24 / UnitLength_in_cm);	/* 8 Mpc/h */
        double res = TopHatSigma2(R8);
        Norm = ppar->Sigma8 / sqrt(res);
        message(0, "Normalization adjusted to  Sigma8=%g   (Normfac=%g). \n", ppar->Sigma8, Norm);
    }
    if(ppar->InputPowerRedshift >= 0) {
        double Dplus = GrowthFactor(InitTime, 1/(1+ppar->InputPowerRedshift));
        Norm *= Dplus;
        message(0,"Growth factor to z=%g: %g \n", ppar->InputPowerRedshift, Dplus);
    }
    return power_table.Nentry;
}

double Delta_Tabulated(double k, int Type)
{
  const double logk = log10(k);

  if(logk < power_table.logk[0] || logk > power_table.logk[power_table.Nentry - 1])
    return 0;

  const double logD = gsl_interp_eval(power_table.mat_intp[0], power_table.logk, power_table.logD[0], logk, power_table.mat_intp_acc[0]);
  double trans = 1;
  /*Transfer table stores (T_type(k) / T_tot(k))^2*/
  if(Type >= 0 && Type < MAXCOLS-1 && transfer_table.Nentry > 0) {
      trans = gsl_interp_eval(transfer_table.mat_intp[Type], transfer_table.logk, transfer_table.logD[Type], logk, transfer_table.mat_intp_acc[Type]);
  }

  double delta = pow(10.0, logD) * trans;

  if(!isfinite(delta))
      endrun(1,"Power spectrum is: %g for k = %g, Type = %d (tk = %g, logD = %g)\n",delta, k, Type, trans, logD);
  return delta;
}

double Delta_EH(double k)	/* Eisenstein & Hu */
{
  return sqrt(k * pow(tk_eh(k), 2)* pow(k, PrimordialIndex - 1.0));
}


double tk_eh(double k)		/* from Martin White */
{
  double q, theta, ommh2, a, s, gamma, L0, C0;
  double tmp;
  double omegam, ombh2, hubble;

  /* other input parameters */
  hubble = CP->HubbleParam;

  omegam = CP->Omega0;
  ombh2 = CP->OmegaBaryon * CP->HubbleParam * CP->HubbleParam;

  if(CP->OmegaBaryon == 0)
    ombh2 = 0.044 * CP->HubbleParam * CP->HubbleParam;

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
  x = 4 * M_PI / (2 * M_PI * 2 * M_PI * 2 * M_PI) * k * k * w * w * pow(DeltaSpec(k, -1),2);

  return x;

}
