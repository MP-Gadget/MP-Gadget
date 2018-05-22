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

#define MAXCOLS 8


struct table
{
    int Nentry;
    double * logk;
    double * logD[MAXCOLS];
    gsl_interp * mat_intp[MAXCOLS];
    gsl_interp_accel * mat_intp_acc[MAXCOLS];
};

static struct table power_table;
/*Columns: 0 == baryon, 1 == CDM, 2 == neutrino, 3 == baryon velocity, 4 == CDM velocity, 5 = neutrino velocity*/
static struct table transfer_table;
/*Symbolic constants for the rows of the transfer table*/
/*Number of types with defined transfers.*/
enum TransferCols
{
    DELTA_BAR = 0,
    DELTA_CDM = 1,
    DELTA_NU = 2,
    VEL_BAR = 3,
    VEL_CDM = 4,
    VEL_NU = 5,
    VEL_CB = 6,
    VEL_TOT = 7
};

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

  if(logk < transfer_table.logk[0] || logk > transfer_table.logk[transfer_table.Nentry - 1])
      return 1;

  /*Default to total growth: type 3 is cdm + baryons.*/
  if(Type < 0 || Type >= 3) {
      Type = MAXCOLS-4;
  }
  /*Use the velocity entries*/
  double growth =  gsl_interp_eval(transfer_table.mat_intp[VEL_BAR + Type], transfer_table.logk, transfer_table.logD[VEL_BAR + Type], logk, transfer_table.mat_intp_acc[VEL_BAR+Type]);

  if(!isfinite(growth))
      endrun(1,"Growth function is: %g for k = %g, Type = %d\n", growth, kmag, Type);
  return growth;
}

void parse_power(int i, double k, char * line, struct table *out_tab, int * InputInLog10, const int nnu, double scale)
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

void parse_transfer(int i, double k, char * line, struct table *out_tab, int * InputInLog10, const int nnu, double scale)
{
    int j;
    const int ncols = 15 + nnu * 2;
    double transfers[ncols];
    k = log10(k);
    k -= log10(scale);  /* convert to h/Kpc */
    out_tab->logk[i] = k;
    /* Note: the ncdm entries change depending on the number of neutrino species. The first row, k,
     * is read in read_power_table and passed as a parameter.
     * but only the first is used for particles.
     * 1:k (h/Mpc)              2:d_g                    3:d_b                    4:d_cdm                  5:d_ur
     * 6:d_ncdm[0]              7:d_ncdm[1]              8:d_ncdm[2]              9:d_tot                 10:phi
     * 11:psi                   12:h                     13:h_prime               14:eta                   15:eta_prime
     * 16:t_g                   17:t_b                   18:t_ur
     * 19:t_ncdm[0]             20:t_ncdm[1]             21:t_ncdm[2]             22:t_tot*/
    for(j = 0; j< ncols; j++) {
        char * retval = strtok(NULL, " \t");
        /*This happens if we do not have as many neutrino species as we expect, or we don't find h_prime.*/
        if(!retval)
            endrun(1,"Incomplete line in power spectrum: only %d columns, expecting %d. Did you remember to set extra metric transfer functions=y?\n",j, ncols);
        transfers[j] = atof(retval);
    }
    /*Order of the transfer table matches the particle types:
     * 0 is baryons, 1 is CDM, 2 is massive neutrinos (we use the most massive neutrino species).
     * 3 is growth function for baryons, 4 is growth function for CDM, 5 is growth function for massive neutrinos.
     * We use the formulae for velocities from fastpm in synchronous gauge:
     * https://github.com/rainwoodman/fastpm-python/blob/02ce2ff87897f713c7b9204630f4e0257d703784/fastpm/multi.py#L185
     * CDM = - h_prime / 2 / d_cdm
     * bar = - (h_prime / 2  + t_b) / d_b
     * nu = - (h_prime / 2 + t_ncdm) / d_ncdm
     * and there is a normalisation factor of (1+z)/ hubble applied later on.
     *
     * See http://adsabs.harvard.edu/abs/1995ApJ...455....7M
     * Eq. 42 (cdm), not using 49(ur), eq. 66 (baryons, and ncdm, truncation at the same order as baryon).
     * These are in CLASS units: they are converted to Gadget units in init_transfer_table().
     * */
    out_tab->logD[DELTA_BAR][i] = -1*transfers[1];
    out_tab->logD[DELTA_CDM][i] = -1*transfers[2];
    /*This is ur if neutrinos are massless*/
    out_tab->logD[DELTA_NU][i] = -1*transfers[3+nnu];
    /*h_prime is entry 8 + nnu. t_b is 12 + nnu, t_ncdm[2] is 13 + nnu * 2.*/
    out_tab->logD[VEL_BAR][i] = transfers[12+nnu];
    out_tab->logD[VEL_CDM][i] = transfers[8+nnu] * 0.5;
    out_tab->logD[VEL_NU][i] = transfers[13+nnu*2];
}

void read_power_table(int ThisTask, const char * inputfile, const int ncols, struct table * out_tab, double scale, const int nnu, void (*parse_line)(int i, double k, char * line, struct table *, int *InputInLog10, const int nnu, double scale))
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
            parse_line(i, k, line, out_tab, &InputInLog10, nnu, scale);
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
init_transfer_table(int ThisTask, double InitTime, const struct power_params * const ppar)
{
    int i, t;
    const int nnu = (CP->MNu[0] > 0) + (CP->MNu[1] > 0) + (CP->MNu[2] > 0);
    if(strlen(ppar->FileWithTransferFunction) > 0) {
        read_power_table(ThisTask, ppar->FileWithTransferFunction, MAXCOLS, &transfer_table, ppar->SpectrumLengthScale, nnu, parse_transfer);
    }
    if(transfer_table.Nentry == 0) {
        endrun(1, "Could not read transfer table at: '%s'\n",ppar->FileWithTransferFunction);
    }
    /*Normalise the transfer functions.*/

    /*Now normalise the velocity transfer functions: divide by a * hubble, where hubble is the hubble function in Mpc^-1, so H0/c*/
    const double fac = InitTime * hubble_function(InitTime)/CP->Hubble * 100 * CP->HubbleParam/(LIGHTCGS / 1e5);
    const double onu = get_omega_nu(&CP->ONu, 1);
    double meangrowth[5] = {0};
    /* At this point the transfer table contains: (3,4,5) t_b, 0.5 * h_prime, t_ncdm.
     * After, if t_cdm = 0.5 h_prime / (a H(a) / H0 /c) we need:
     * (t_b + t_cdm) / d_b, t_cdm/d_cdm, (t_ncdm + t_cdm) / d_ncdm*/
    for(i=0; i< transfer_table.Nentry; i++) {
        /* Now row 4 is t_cdm*/
        transfer_table.logD[VEL_CDM][i] /= fac;
        transfer_table.logD[VEL_BAR][i] += transfer_table.logD[VEL_CDM][i];
        transfer_table.logD[VEL_NU][i] += transfer_table.logD[VEL_CDM][i];

        /*CDM + baryon growth*/
        transfer_table.logD[VEL_CB][i] = CP->OmegaBaryon * transfer_table.logD[VEL_BAR][i] + CP->OmegaCDM * transfer_table.logD[VEL_CDM][i];
        /*total growth*/
        transfer_table.logD[VEL_TOT][i] = transfer_table.logD[VEL_CB][i];
        /*Total delta*/
        double T_tot = CP->OmegaBaryon * transfer_table.logD[DELTA_BAR][i] + CP->OmegaCDM * transfer_table.logD[DELTA_CDM][i];
        /*Divide cdm +  bar total velocity transfer by d_cdm + bar*/
        transfer_table.logD[VEL_CB][i] /= T_tot;
        if(nnu > 0) {
            /*Add neutrino growth to total growth*/
            transfer_table.logD[VEL_TOT][i] += onu *  transfer_table.logD[VEL_NU][i];
            T_tot += onu * transfer_table.logD[DELTA_NU][i];
        }
        /* Total growth normalized by total delta*/
        transfer_table.logD[VEL_TOT][i] /= T_tot;
        /*Normalize growth_i by delta_i, and transform delta_i to delta_i/delta_tot*/
        for(t = 3; t < 6; t++) {
            transfer_table.logD[t][i] /= transfer_table.logD[t-3][i];
            transfer_table.logD[t-3][i] /= (T_tot / CP->Omega0);
        }
    }

    /*Now compute mean growths*/
    for(t = 3; t < 8; t++) {
        int nmean=0;
        for(i=0; i< transfer_table.Nentry; i++)
            if(transfer_table.logk[i] > power_table.logk[0]) {
                meangrowth[t-3] += transfer_table.logD[t][i];
                nmean++;
            }
        meangrowth[t-3]/= nmean;
    }
    /*Initialise the interpolators*/
    for(t = 0; t < MAXCOLS; t++)
        gsl_interp_init(transfer_table.mat_intp[t],transfer_table.logk, transfer_table.logD[t],transfer_table.Nentry);

    message(0,"Scale-dependent growth calculated. Mean = %g %g %g %g %g\n",meangrowth[0], meangrowth[1], meangrowth[2], meangrowth[3], meangrowth[4]);
    message(0, "Power spectrum rows: %d, Transfer: %d (%g -> %g)\n", power_table.Nentry, transfer_table.Nentry, transfer_table.logD[DELTA_BAR][0],transfer_table.logD[DELTA_BAR][transfer_table.Nentry-1]);
    return transfer_table.Nentry;
}

int init_powerspectrum(int ThisTask, double InitTime, double UnitLength_in_cm_in, Cosmology * CPin, struct power_params * ppar)
{
    WhichSpectrum = ppar->WhichSpectrum;
    /*Used only for tk_eh*/
    PrimordialIndex = ppar->PrimordialIndex;
    UnitLength_in_cm = UnitLength_in_cm_in;
    CP = CPin;

    if(ppar->WhichSpectrum == 2) {
        read_power_table(ThisTask, ppar->FileWithInputSpectrum, 1, &power_table, ppar->SpectrumLengthScale, 0, parse_power);
        /*Initialise the interpolator*/
        gsl_interp_init(power_table.mat_intp[0],power_table.logk, power_table.logD[0],power_table.Nentry);
        transfer_table.Nentry = 0;
        if(ppar->DifferentTransferFunctions || ppar->ScaleDepVelocity) {
            init_transfer_table(ThisTask, InitTime, ppar);
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
  /*Transfer table stores (T_type(k) / T_tot(k))*/
  if(transfer_table.Nentry > 0) {
    if(Type >= 0 && Type < 3) {
        trans = gsl_interp_eval(transfer_table.mat_intp[Type], transfer_table.logk, transfer_table.logD[Type], logk, transfer_table.mat_intp_acc[Type]);
    }
    /*CDM + baryons*/
    else if (Type == 3){
        double db =  gsl_interp_eval(transfer_table.mat_intp[DELTA_BAR], transfer_table.logk, transfer_table.logD[DELTA_BAR], logk, transfer_table.mat_intp_acc[DELTA_BAR]);
        double dcdm =  gsl_interp_eval(transfer_table.mat_intp[DELTA_CDM], transfer_table.logk, transfer_table.logD[DELTA_CDM], logk, transfer_table.mat_intp_acc[DELTA_CDM]);
        trans = (CP->OmegaBaryon * db + CP->OmegaCDM * dcdm)/(CP->OmegaCDM + CP->OmegaBaryon);
    }
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
