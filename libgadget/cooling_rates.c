/* This version of cooling.c is reimplemented from a
 * python rate network, to cleanly allow self-shielding,
 * metal cooling and UVB fluctuations. It shares no code
 * with the original Gadget-3 cooling code.
 *
 * Copyright (c) Simeon Bird 2019 and Yu Feng 2015-2019.
 * Released under the terms of the GPLv2 or later.
 */

/* A rate network for neutral hydrogen following
 * Katz, Weinberg & Hernquist 1996, astro-ph/9509107, eq. 28-32.

    Most internal methods are CamelCapitalized and follow a convention that
    they are named like the process and then the ion they refer to.
    eg:
        CollisionalExciteHe0 is the neutral Helium collisional excitation rate.
        RecombHp is the recombination rate for ionized hydrogen.

    Externally useful methods (the API) are named like get_*.
    These are:
        get_temp() - gets the temperature from the density and internal energy.
        get_heatingcooling_rate() - gets the total (net) heating and cooling rate from density and internal energy.
        get_neutral_fraction() - gets the neutral fraction from the rate network given density and internal energy.
        get_global_UVBG() - Interpolates the TreeCool table to a desired redshift and returns a struct UVBG.
    Two useful helper functions:
        get_equilib_ne() - gets the equilibrium electron density.
        get_ne_by_nh() - gets the above, divided by the hydrogen density (Gadget reports this as ElectronAbundance).

    Global Variables:
        redshift - the redshift at which to evaluate the cooling. Affects the photoionization rate,
                    the Inverse Compton cooling and the self shielding threshold.
        photo_factor - Factor by which to multiply the UVB amplitude.
        f_bar - Baryon fraction. Omega_b / Omega_cdm.
        selfshield - Flag to enable self-shielding following Rahmati 2012
        cool - which cooling rate coefficient table to use.
                Supported are: KWH (original Gadget rates)
                                Nyx (rates used in Nyx (Lukic 2015) and in Enzo 2.
                                Sherwood (rates used in the Sherwood simulation,
                                          which fix a bug in the Cen 92 rates, but are otherwise the same as KWH.
                                          Sherwood also uses the more accurate recombination and collisional ionization rates).
        recomb - which recombination and collisional ionization rate table to use.
                    Supported are: C92 (Cen 1992, the Gadget default)
                                V96 (Verner & Ferland 1996, more accurate rates). Voronov 97 is used for collisional ionizations.
                                B06 (Badnell 2006 rates, current cloudy defaults. Very similar to V96).

        The default is to follow what is done in the Sherwood simulations, Bolton et al 2016 1605.03462
        These are essentially KWH but with collisional ionization and recombination rates and cooling changed to match Verner & Ferland 96.
        The Cen 92 high temp correction factor is also changed.
        treecool_file - File to read a UV background from. Same format as Gadget has always used.
*/

#include "cooling_rates.h"

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <gsl/gsl_interp.h>
#include "physconst.h"
#include "utils/endrun.h"
#include "utils/mymalloc.h"

/* 1 eV in ergs*/
#define eVinergs 1.60218e-12

static struct cooling_params CoolingParams;

static gsl_interp * GrayOpac;

/*Tables for the self-shielding correction. Note these are not well-measured for z > 5!*/
#define NGRAY 9
static gsl_interp_accel *GrayOpac_acc;
static double GrayOpac_ydata[NGRAY] = { 2.59e-18, 2.37e-18, 2.27e-18, 2.15e-18, 2.02e-18, 1.94e-18, 1.82e-18, 1.71e-18, 1.60e-18};
static const double GrayOpac_zz[NGRAY] = {0, 1, 2, 3, 4, 5, 6, 7, 8};

/*Convenience structure bundling together the gsl interpolation routines.*/
struct itp_type
{
    double * ydata;
    gsl_interp * intp;
    gsl_interp_accel * acc;
};
/*Interpolation objects for the redshift evolution of the UVB.*/
/*Number of entries in the table*/
static int NTreeCool;
/*Redshift bins*/
static double * Gamma_log1z;
/*These are the photo-ionization rates*/
static struct itp_type Gamma_HI, Gamma_HeI, Gamma_HeII;
/*These are the photo-heating rates*/
static struct itp_type Eps_HI, Eps_HeI, Eps_HeII;
/*Recombination and collisional rates*/
#define NRECOMBTAB 300
static double * temp_tab;
static struct itp_type rec_alphaHp, rec_alphaHep, rec_alphad, rec_alphaHepp;
static struct itp_type rec_GammaH0, rec_GammaHe0, rec_GammaHep;
static struct itp_type cool_collisH0, cool_collisHe0, cool_collisHeP;

static void
init_itp_type(double * xarr, struct itp_type * Gamma, int Nelem)
{
    Gamma->intp = gsl_interp_alloc(gsl_interp_cspline,Nelem);
    gsl_interp_init(Gamma->intp, xarr, Gamma->ydata, Nelem);
    Gamma->acc = gsl_interp_accel_alloc();
}

/* This function loads the treecool file into the (global function) data arrays.
 * Format of the treecool table:
    log_10(1+z), Gamma_HI, Gamma_HeI, Gamma_HeII,  Qdot_HI, Qdot_HeI, Qdot_HeII,
    where 'Gamma' is the photoionization rate and 'Qdot' is the photoheating rate.
    The Gamma's are in units of s^-1, and the Qdot's are in units of erg s^-1.
*/
static void
load_treecool(const char * TreeCoolFile)
{
    if(!CoolingParams.PhotoIonizationOn)
        return;
    FILE * fd = fopen(TreeCoolFile, "r");
    if(!fd)
        endrun(456, "Could not open photon background (TREECOOL) file at: '%s'\n", TreeCoolFile);

    /*Find size of file*/
    NTreeCool = 0;
    do
    {
        char buffer[1024];
        char * retval = fgets(buffer, 1024, fd);
        /*Happens on end of file*/
        if(!retval)
            break;
        retval = strtok(buffer, " \t");
        /*Discard comments*/
        if(!retval || retval[0] == '#')
            continue;
        NTreeCool++;
    }
    while(1);
    rewind(fd);

    MPI_Bcast(&(NTreeCool), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(NTreeCool<= 2)
        endrun(1, "Photon background contains: %d entries, not enough.\n", NTreeCool);

    /*Allocate memory for the photon background table.*/
    Gamma_log1z = mymalloc("TreeCoolTable", 7 * NTreeCool * sizeof(double));
    Gamma_HI.ydata = Gamma_log1z + NTreeCool;
    Gamma_HeI.ydata = Gamma_log1z + 2 * NTreeCool;
    Gamma_HeII.ydata = Gamma_log1z + 3 * NTreeCool;
    Eps_HI.ydata = Gamma_log1z + 4 * NTreeCool;
    Eps_HeI.ydata = Gamma_log1z + 5 * NTreeCool;
    Eps_HeII.ydata = Gamma_log1z + 6 * NTreeCool;

    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0)
    {
        int i;
        for(i = 0; i < NTreeCool; i++)
        {
            char buffer[1024];
            char * line = fgets(buffer, 1024, fd);
            /*Happens on end of file*/
            if(!line)
                break;
            char * retval = strtok(line, " \t");
            if(!retval || retval[0] == '#')
                continue;
            Gamma_log1z[i] = atof(retval);
            /*Get the rest*/
            Gamma_HI.ydata[i] = atof(strtok(NULL, " \t"));
            Gamma_HeI.ydata[i] = atof(strtok(NULL, " \t"));
            Gamma_HeII.ydata[i] = atof(strtok(NULL, " \t"));
            Eps_HI.ydata[i] = atof(strtok(NULL, " \t"));
            Eps_HeI.ydata[i] = atof(strtok(NULL, " \t"));
            Eps_HeII.ydata[i] = atof(strtok(NULL, " \t"));
        }

        fclose(fd);
    }

    /*Broadcast data to other processors*/
    MPI_Bcast(Gamma_log1z, 7 * NTreeCool, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    /*Initialize the UVB redshift interpolation: reticulate the splines*/
    init_itp_type(Gamma_log1z, &Gamma_HI, NTreeCool);
    init_itp_type(Gamma_log1z, &Gamma_HeI, NTreeCool);
    init_itp_type(Gamma_log1z, &Gamma_HeII, NTreeCool);
    init_itp_type(Gamma_log1z, &Eps_HI, NTreeCool);
    init_itp_type(Gamma_log1z, &Eps_HeI, NTreeCool);
    init_itp_type(Gamma_log1z, &Eps_HeII, NTreeCool);
}

/*Get interpolated value for one of the recombination interpolators. Takes natural log of temperature.*/
static double
get_interpolated_recomb(double logt, struct itp_type * rec_tab, double rec_func(double))
{
    /*Just call the function directly if we are out of interpolation range*/
    if (logt >= temp_tab[NRECOMBTAB- 1] || logt < temp_tab[0])
        return rec_func(exp(logt));
    return gsl_interp_eval(rec_tab->intp, temp_tab, rec_tab->ydata, logt, rec_tab->acc);
}

/*Get photo ionization rate for neutral Hydrogen*/
static double
get_photo_rate(double redshift, struct itp_type * Gamma_tab)
{
    if(!CoolingParams.PhotoIonizationOn)
        return 0;
    double log1z = log10(1+redshift);
    double photo_rate;
    if (log1z >= Gamma_log1z[NTreeCool - 1])
        photo_rate = Gamma_tab->ydata[NTreeCool - 1];
    else if (log1z < Gamma_log1z[0])
        photo_rate = 0;
    else {
        photo_rate = gsl_interp_eval(Gamma_tab->intp, Gamma_log1z, Gamma_tab->ydata, log1z, Gamma_tab->acc);
    }
    return photo_rate * CoolingParams.PhotoIonizeFactor;
}

/* This initializes a global UVBG by interpolating the redshift tables,
 * to which the UV fluctuations can be applied*/
struct UVBG get_global_UVBG(double redshift)
{
    struct UVBG GlobalUVBG = {0};

    if(!CoolingParams.PhotoIonizationOn)
        return GlobalUVBG;
    GlobalUVBG.gJH0 = get_photo_rate(redshift, &Gamma_HI);
    GlobalUVBG.gJHe0 = get_photo_rate(redshift, &Gamma_HeI);
    GlobalUVBG.gJHep = get_photo_rate(redshift, &Gamma_HeII);

    GlobalUVBG.epsH0 = get_photo_rate(redshift, &Eps_HI);
    GlobalUVBG.epsHe0 = get_photo_rate(redshift, &Eps_HI);
    GlobalUVBG.epsHep = get_photo_rate(redshift, &Eps_HI);

    return GlobalUVBG;
}

/*Calculate the critical self-shielding density. Rahmati 2012 eq. 13.
  gray_opac is a parameter of the UVB used.
  gray_opac is in cm^2 (2.49e-18 is HM01 at z=3)
  temp is particle temperature in K
  f_bar is the baryon fraction. 0.17 is roughly 0.045/0.265
  Returns density in atoms/cm^3"""
*/
static double
self_shield_dens(double redshift, double logt, const struct UVBG * uvbg)
{
    double T4 = exp(logt)/1e4;
    double G12 = uvbg->gJH0/1e-12;
    double greyopac;
    if (redshift <= GrayOpac_zz[0])
        greyopac = GrayOpac_ydata[0];
    else if (redshift >= GrayOpac_zz[NGRAY-1])
        greyopac = GrayOpac_ydata[NGRAY-1];
    else {
        greyopac = gsl_interp_eval(GrayOpac, GrayOpac_zz, GrayOpac_ydata,redshift, GrayOpac_acc);
    }
    return 6.73e-3 * pow(greyopac / 2.49e-18, -2./3)*pow(T4, 0.17)*pow(G12, 2./3)*pow(CoolingParams.fBar/0.17,-1./3);
}

/*Correction to the photoionisation rate as a function of density from Rahmati 2012, eq. 14.
  Calculates Gamma_{Phot} / Gamma_{UVB}.
  Inputs: hydrogen density, temperature
      n_H
  The coefficients are their best-fit from appendix A."""
*/
static double
self_shield_corr(double nh, double logt, double redshift, const struct UVBG * uvbg)
{
    if(!CoolingParams.SelfShieldingOn)
        return 1;
    double nSSh = 1.003*self_shield_dens(redshift, logt, uvbg);
    return 0.98*pow(1+pow(nh/nSSh,1.64),-2.28)+0.02*pow(1+nh/nSSh, -0.84);
}

/*Here come the recombination rates*/

/* Recombination rates and collisional ionization rates, as a function of temperature. There are three options implemented.
 *
 * Cen92:
 * used in standard Gadget and Illustris.
 * Taken from KWH 06, astro-ph/9509107, Table 2, based on Cen 1992.
 *
 * Verner96:
 * the fit from Verner & Ferland 1996 (astro-ph/9509083).
 * Collisional rates are the fit from Voronov 1997 (http://www.sciencedirect.com/science/article/pii/S0092640X97907324).
 * In a very photoionised medium this changes the neutral hydrogen abundance by approximately 10% compared to Cen 1992.
 * These rates are those used by Nyx and Sherwood.
 *
 * Badnell06:
 * Recombination rates are the fit from Badnell's website: http://amdpp.phys.strath.ac.uk/tamoc/RR/#partial.
 * These are used in CLOUDY, and for our purposes basically identical to Verner 96. Included for completeness.
 */

/*Formula used as a fitting function in Verner & Ferland 1996 (astro-ph/9509083).*/
static double
_Verner96Fit(double temp, double aa, double bb, double temp0, double temp1)
{
        double sqrttt0 = sqrt(temp/temp0);
        double sqrttt1 = sqrt(temp/temp1);
        return aa / ( sqrttt0 * pow(1 + sqrttt0, 1-bb)*pow(1+sqrttt1, 1+bb) );
}

/*Recombination rate for H+, ionized hydrogen, in cm^3/s. Temp in K.*/
double
recomb_alphaHp(double temp)
{
    switch(CoolingParams.recomb)
    {
        case Cen92:
            return 8.4e-11 / sqrt(temp) / pow(temp/1000, 0.2) / (1+ pow(temp/1e6, 0.7));
        case Verner96:
            /*The V&F 96 fitting formula is accurate to < 1% in the worst case.
            See line 1 of V&F96 table 1.*/
            return _Verner96Fit(temp, 7.982e-11, 0.748, 3.148, 7.036e+05);
        case Badnell06:
            /*Slightly different fit coefficients*/
            return _Verner96Fit(temp, 8.318e-11, 0.7472, 2.965, 7.001e5);
        default:
            endrun(3, "Recombination rate not supported\n");
    }
}

/*Helper function to implement Verner & Ferland 96 recombination rate.*/
static double
_Verner96alphaHep(double temp)
{
    /*Accurate to ~2% for T < 10^6 and 5% for T< 10^10.
     * VF96 give two rates. The first is more accurate for T < 10^6, the second is valid up to T = 10^10.
     * We use the most accurate allowed. See lines 2 and 3 of Table 1 of VF96.*/
    double lowTfit = _Verner96Fit(temp, 3.294e-11, 0.6910, 1.554e+01, 3.676e+07);
    double highTfit = _Verner96Fit(temp, 9.356e-10, 0.7892, 4.266e-02, 4.677e+06);
    /*Note that at 10^6K the two fits differ by ~10%. This may lead one to disbelieve the quoted accuracies!
    * We thus switch over at a slightly lower temperature. The two fits cross at T ~ 3e5K.*/
    double swtmp = 7e5;
    double deltat = 1e5;
    double upper = swtmp + deltat;
    double lower = swtmp - deltat;
    //In order to avoid a sharp feature at 10^6 K, we linearly interpolate between the two fits around 10^6 K.
    double interpfit = (lowTfit * (upper - temp) + highTfit * (temp - lower))/(2*deltat);
    return (temp < lower)*lowTfit + (temp > upper)*highTfit + (upper > temp)*(temp > lower)*interpfit;
}

/*Recombination rate for He+, ionized helium, in cm^3/s. Temp in K.*/
static double
recomb_alphaHep(double temp)
{
    switch(CoolingParams.recomb)
    {
        case Cen92:
            return 1.5e-10 / pow(temp,0.6353);
        case Verner96:
            return _Verner96alphaHep(temp);
        case Badnell06:
            return _Verner96Fit(temp, 1.818E-10, 0.7492, 10.17, 2.786e6);
        default:
            endrun(3, "Recombination rate not supported\n");
    }
}

/* Recombination rate for dielectronic recombination, in cm^3/s. Temp in K.
 * The HeII->HeI dielectronic recombination rate is the cooling rate divided by 3Ry (for the HeII LyA transition).
 */
static double
recomb_alphad(double temp)
{
    switch(CoolingParams.recomb)
    {
        case Cen92:
            return 1.9e-3 / pow(temp,1.5) * exp(-4.7e5/temp)*(1+0.3*exp(-9.4e4/temp));
        case Verner96:
        case Badnell06:
             /* Private communication from Avery Meiksin:
                The rate in Black (1981) is wrong by a factor of 0.65. He took the rate fit from Aldrovandi & Pequignot (1973),
                who based it on Burgess (1965), but was unaware of the correction factor in Burgess & Tworkowski (1976, ApJ 205:L105-L107, fig 1).
                The correct dielectronic cooling rate should have the coefficient 0.813e-13 instead of 1.24e-13 as in Black's table 3.
             */
            return 1.23e-3 / pow(temp,1.5) * exp(-4.72e5/temp)*(1+0.3*exp(-9.4e4/temp));
        default:
            endrun(3, "Recombination rate not supported\n");
    }
}

/* Recombination rate for doubly ionized recombination, in cm^3/s. Temp in K.*/
static double
recomb_alphaHepp(double temp)
{
    switch(CoolingParams.recomb)
    {
        case Cen92:
            return 4 * recomb_alphaHp(temp);
        case Verner96:
            return _Verner96Fit(temp, 1.891e-10, 0.7524, 9.370, 2.774e6);
        case Badnell06:
            return _Verner96Fit(temp, 5.235E-11, 0.6988 + 0.0829*exp(-1.682e5/temp), 7.301, 4.475e6);
        default:
            endrun(3, "Recombination rate not supported\n");
    }
}

/*Fitting function for collisional rates. Eq. 1 of Voronov 1997. Accurate to 10%, but data is only accurate to 50%.*/
static double
_Voronov96Fit(double temp, double dE, double PP, double AA, double XX, double KK)
{
    double UU = dE / (BOLEVK * temp);
    return AA * (1 + PP * sqrt(UU))/(XX+UU) * pow(UU, KK) * exp(-UU);
}

/* Collisional ionization rate for H0 in cm^3/s. Temp in K.*/
static double
recomb_GammaeH0(double temp)
{
    switch(CoolingParams.recomb)
    {
        case Cen92:
            return 5.85e-11 * sqrt(temp) * exp(-157809.1/temp) / (1 + sqrt(temp/1e5));
        case Verner96:
        case Badnell06:
            //Voronov 97 Table 1
            return _Voronov96Fit(temp, 13.6, 0, 0.291e-07, 0.232, 0.39);
        default:
            endrun(3, "Collisional rate not supported\n");
    }
}

/* Collisional ionization rate for He0 in cm^3/s. Temp in K.*/
static double
recomb_GammaeHe0(double temp)
{
    switch(CoolingParams.recomb)
    {
        case Cen92:
            return 2.38e-11 * sqrt(temp) * exp(-285335.4/temp) / (1+ sqrt(temp/1e5));
        case Verner96:
        case Badnell06:
            //Voronov 97 Table 1
            return _Voronov96Fit(temp, 24.6, 0, 0.175e-07, 0.180, 0.35);
        default:
            endrun(3, "Collisional rate not supported\n");
    }
}

/* Collisional ionization rate for He+ in cm^3/s. Temp in K.*/
static double
recomb_GammaeHep(double temp)
{
    switch(CoolingParams.recomb)
    {
        case Cen92:
            return 5.68e-12 * sqrt(temp) * exp(-631515.0/temp) / (1+ sqrt(temp/1e5));
        case Verner96:
        case Badnell06:
            //Voronov 97 Table 1
            return _Voronov96Fit(temp, 54.4, 1, 0.205e-08, 0.265, 0.25);
        default:
            endrun(3, "Collisional rate not supported\n");
    }
}

/*The neutral hydrogen number density. Eq. 33 of KWH.*/
static double
nH0_internal(double nh, double logt, double ne, double redshift, const struct UVBG * uvbg)
{
    double alphaHp = get_interpolated_recomb(logt, &rec_alphaHp, &recomb_alphaHp);
    double GammaeH0 = get_interpolated_recomb(logt, &rec_GammaH0, &recomb_GammaeH0);
    double photorate = uvbg->gJH0/(1e-30 + ne) * self_shield_corr(nh, logt, redshift, uvbg);
    return nh * alphaHp/ (alphaHp + GammaeH0 + photorate);
}

/*The ionised hydrogen number density. Eq. 34 of KWH.*/
static double
nHp_internal(double nh, double logt, double ne, double redshift, const struct UVBG * uvbg)
{
    return nh - nH0_internal(nh, logt, ne, redshift, uvbg);
}

/*The ionised helium number density, divided by the helium number fraction. Eq. 35 of KWH.*/
static double
nHep_internal(double nh, double logt, double ne, double redshift, const struct UVBG * uvbg)
{
    double alphaHep = get_interpolated_recomb(logt, &rec_alphaHep, &recomb_alphaHep) + get_interpolated_recomb(logt, &rec_alphad, &recomb_alphad);
    double alphaHepp = get_interpolated_recomb(logt, &rec_alphaHepp, &recomb_alphaHepp);
    double photofac = self_shield_corr(nh, logt, redshift, uvbg);
    double GammaHe0 = get_interpolated_recomb(logt, &rec_GammaH0, &recomb_GammaeH0) + uvbg->gJHe0/(1e-30 + ne)*photofac;
    double GammaHep = get_interpolated_recomb(logt, &rec_GammaHep, &recomb_GammaeHep) + uvbg->gJHep/(1e-30 + ne)*photofac;
    return nh / (1 + alphaHep / GammaHe0 + GammaHep/alphaHepp);
}

/*The neutral helium number density, divided by the helium number fraction. Eq. 36 of KWH.*/
static double
nHe0_internal(double nh, double logt, double ne, double redshift, const struct UVBG * uvbg)
{
    double alphaHep = get_interpolated_recomb(logt, &rec_alphaHep, &recomb_alphaHep) + get_interpolated_recomb(logt, &rec_alphad, &recomb_alphad);
    double photofac = self_shield_corr(nh, logt, redshift, uvbg);
    double GammaHe0 = get_interpolated_recomb(logt, &rec_GammaHe0, &recomb_GammaeHe0) + uvbg->gJHep/(1e-30 + ne)*photofac;
    return nHep_internal(nh, logt, ne, redshift, uvbg) * alphaHep / GammaHe0;
}

/* The doubly ionised helium number density, divided by the helium number fraction. Eq. 37 of KWH.*/
static double
nHepp_internal(double nh, double logt, double ne, double redshift, const struct UVBG * uvbg)
{
    double photofac = self_shield_corr(nh, logt, redshift, uvbg);
    double GammaHep = get_interpolated_recomb(logt, &rec_GammaHep, &recomb_GammaeHep) + uvbg->gJHep/(1e-30 + ne)*photofac;
    double alphaHepp = get_interpolated_recomb(logt, &rec_alphaHepp, &recomb_alphaHepp);
    return nHep_internal(nh, logt, ne, redshift, uvbg) * GammaHep / alphaHepp;
}

/*The electron number density. Eq. 38 of KWH.*/
static double
ne_internal(double nh, double logt, double ne, double helium, double redshift, const struct UVBG * uvbg)
{
    double yy = helium / 4 / (1 - helium);
    return nHp_internal(nh, logt, ne, redshift, uvbg) + yy * nHep_internal(nh, logt, ne, redshift, uvbg) + 2 * yy * nHepp_internal(nh, logt, ne, redshift, uvbg);
}

/*Compute temperature (in K) from internal energy and electron density.
    Uses: internal energy
            electron abundance per H atom (ne/nH)
            hydrogen mass fraction (0.76)
    Internal energy is in erg/g.
    Factor to convert U (erg/g) to T (K) : U = N k T / (γ - 1)
    T = U (γ-1) μ m_P / k_B
    where k_B is the Boltzmann constant
    γ is 5/3, the perfect gas constant
    m_P is the proton mass

    μ = 1 / (mean no. molecules per unit atomic weight)
        = 1 / (X + Y /4 + E)
        where E = Ne * X, and Y = (1-X).
        Can neglect metals as they are heavy.
        Leading contribution is from electrons, which is already included
        [+ Z / (12->16)] from metal species
        [+ Z/16*4 ] for OIV from electrons.*/
static double
get_temp_internal(double nebynh, double ienergy, double helium)
{
    /*convert U (erg/g) to T (K) : U = N k T / (γ - 1)
    T = U (γ-1) μ m_P / k_B
    where k_B is the Boltzmann constant
    γ is 5/3, the perfect gas constant
    m_P is the proton mass
    μ is 1 / (mean no. molecules per unit atomic weight) calculated in loop.
    Internal energy units are 10^-10 erg/g*/
    double hy_mass = 1 - helium;
    double muienergy = 4 / (hy_mass * (3 + 4*nebynh) + 1)*ienergy;
    /*So for T in K, Boltzmann in erg/K, internal energy has units of erg/g*/
    double temp = GAMMA_MINUS1 * PROTONMASS / BOLTZMANN * muienergy;
    return temp;
}

/*Maximum number of iterations to perform*/
#define MAXITER 1000
/*Tolerance to converge the rate network towards*/
#define ITERCONV 1e-8

/* This finds a fixed point of the function where ``func(x0) == x0``.
    Uses Steffensen's Method with Aitken's ``Del^2`` convergence
    acceleration (Burden, Faires, "Numerical Analysis", 5th edition, pg. 80).
    This routine is ported from scipy.optimize.fixed_point.
*/
static double
scipy_optimize_fixed_point(double ne_init, double nh, double ienergy, double helium, double redshift, const struct UVBG * uvbg)
{
    int i;
    double ne0 = ne_init;
    /*A better initialization*/
    if(ne0 > nh * (1 + helium))
        ne0 = nh * (1 + helium);
    for(i = 0; i < MAXITER; i++)
    {
        double ne1 = ne_internal(nh, log(get_temp_internal(ne0/nh, ienergy, helium)), ne0, helium, redshift, uvbg);
        double ne2 = ne_internal(nh, log(get_temp_internal(ne1/nh, ienergy, helium)), ne1, helium, redshift, uvbg);

        double d = ne2 - 2.0 * ne1 + ne0;
        double pp = ne2;
        /*This is del^2*/
        if (d != 0.)
            pp = ne0 - (ne1 - ne0)*(ne1 - ne0) / d;
        double relerr = pp;
        if(ne0 != 0)
            relerr = fabs(pp/ne0 - 1);
        ne0 = pp;
        if (relerr < ITERCONV)
            break;
    }
    if (!isfinite(ne0) || fabs(ne_internal(nh, log(get_temp_internal(ne0/nh, ienergy, helium)), ne0, helium, redshift, uvbg) - ne0) > ITERCONV)
            endrun(1, "Ionization rate network failed to converge for %g %g %g %g: last ne = %g\n", nh, ienergy, helium, redshift, ne0);
    return ne0;
}

/*Solve the system of equations for photo-ionization equilibrium,
  starting with ne = nH and continuing until convergence.
  density is gas density in protons/cm^3
  Internal energy is in ergs/g.
  helium is a mass fraction.
*/
double
get_equilib_ne(double density, double ienergy, double helium, double redshift, const struct UVBG * uvbg, double ne_init)
{
    /*Get hydrogen number density*/
    double nh = density * (1-helium);
    return scipy_optimize_fixed_point(ne_init, nh, ienergy, helium, redshift, uvbg);
}

/*Same as above, but get electrons per proton.*/
double
get_ne_by_nh(double density, double ienergy, double helium, double redshift, const struct UVBG * uvbg, double ne_init)
{
    return get_equilib_ne(density, ienergy, helium, redshift, uvbg, ne_init)/(density*(1-helium));
}

/*Here come the cooling rates. These are in erg s^-1 cm^-3 (cgs).
All rates are divided by the abundance of the ions involved in the interaction.
So we are computing the cooling rate divided by n_e n_X. Temperatures in K.

Cen92 rates:
None of these rates are original to KWH92, but are taken from Cen 1992,
and originally from older references. The hydrogen rates in particular are probably inaccurate.
Cen 1992 modified (arbitrarily) the excitation and ionisation rates for high temperatures.
There is no collisional excitation rate for He0 - not sure why.
References:
    Black 1981, from Lotz 1967, Seaton 1959, Burgess & Seaton 1960.
    Recombination rates are from Spitzer 1978.
    Free-free: Spitzer 1978.
Collisional excitation and ionisation cooling rates are merged.

These rates are used in the Sherwood simulation, Bolton et al 2016 1605.03462, but with updated collision and recombination rates,
and an improvement to the Cen 92 large temperature correction factor.

Nyx/enzo2 rates:
The cooling rates used in the Nyx paper Lukic 2014, 1406.6361, in erg s^-1 cm^-3 (cgs)
and Enzo v.2. Originally these come from Avery Meiksin, and questions should be directed to him.

Major differences from KWH are the use of the Scholz & Walter 1991
hydrogen collisional cooling rates, a less aggressive high temperature correction for helium, and
Shapiro & Kang 1987 for free free.

References:
    Scholz & Walters 1991 (0.45% accuracy)
    Black 1981 (recombination and helium)
    Shapiro & Kang 1987

They use the recombination rates from Verner & Ferland 96, but do not change the cooling rates to match.
This is because at the temperatures where this matters, T > 10^5, all the H should be ionized anyway.
Ditto the ionization rates from Voronov 1997: they should also use these rates for collisional ionisation,
although this is hard to do in practice because Scholz & Walters don't break their rates into ionization and excitation.

Sherwood rates:
Rates
Small numerical differences from Nyx, but they do not used Scholz & Walter, instead Cen 92 for collisional excitation,
and Verner & Ferland 96 for recombination.
*/

/*Commonly used Cen 1992 correction factor for large temperatures.*/
static double
_t5(double temp)
{
    double t0;
    if(CoolingParams.cooling == KWH92)
        t0 = 1e5;
    /* The original Cen 92 correction is implemented in order to achieve the
     * correct asymptotic behaviour at high temperatures. However, the rates he is correcting
     * should be good up to 10^7 K, so he is being too aggressive.*/
    else
        t0 = 5e7;
    return sqrt(1+temp/t0);
}

/*Collisional excitation cooling rate for n_H0 and n_e. Gadget-3 calls this BetaH0.*/
static double
cool_CollisionalExciteH0(double temp)
{
    return 7.5e-19 * exp(-118348.0/temp) / _t5(temp);
}

/*Collisional excitation cooling rate for n_He+ and n_e. Gadget-3 calls this BetaHep.*/
static double
cool_CollisionalExciteHeP(double temp)
{
    return 5.54e-17 * pow(temp, -0.397)*exp(-473638./temp)/_t5(temp);
}

/*This is listed in Cen 92 but neglected in KWH 97, presumably because it is very small.*/
static double
cool_CollisionalExciteHe0(double temp)
{
    return 9.1e-27 * pow(temp, -0.1687) * exp(-473638/temp) / _t5(temp);
}

/* Collisional ionisation cooling rate for n_H0 and n_e. Gadget calls this GammaeH0.*/
static double
cool_CollisionalIonizeH0(double temp)
{
    /*H ionization potential*/
    return 13.5984 * eVinergs * recomb_GammaeH0(temp);
}

/*Collisional ionisation cooling rate for n_He0 and n_e. Gadget calls this GammaeHe0.*/
static double
cool_CollisionalIonizeHe0(double temp)
{
    return 24.5874 * eVinergs * recomb_GammaeHe0(temp);
}

/*Collisional ionisation cooling rate for n_He+ and n_e. Gadget calls this GammaeHep.*/
static double
cool_CollisionalIonizeHeP(double temp)
{
    return 54.417760 * eVinergs * recomb_GammaeHep(temp);
}

/*Total collisional cooling for H0*/
static double
cool_CollisionalH0(double temp)
{
    if(CoolingParams.cooling == Enzo2Nyx) {
        /*Formula from Eq. 23, Table 4 of Scholz & Walters, claimed good to 0.45 %.
        Note though that they have two datasets which differ by a factor of two.
        Differs from Cen 92 by a factor of two.*/
        //Technically only good for T > 2000.
        double y = log(temp);
        //Constant is 0.75/k_B in Rydberg
        double Ryd = 2.1798741e-11;
        double tot = -0.75/BOLTZMANN *Ryd/temp;
        double coeffslowT[6] = {213.7913, 113.9492, 25.06062, 2.762755, 0.1515352, 3.290382e-3};
        double coeffshighT[6] = {271.25446, 98.019455, 14.00728, 0.9780842, 3.356289e-2, 4.553323e-4};
        int j;
        for(j=0; j < 6; j++)
            tot += ((temp < 1e5)*coeffslowT[j]+(temp >=1e5)*coeffshighT[j])*pow(-y, j);
        return 1e-20 * exp(tot);
    }
    else /*Everyone else splits collisional from excitation, because collisional is mostly exact*/
        return cool_CollisionalExciteH0(temp) + cool_CollisionalIonizeH0(temp);
}

/*Total collisional cooling for He0*/
static double
cool_CollisionalHe0(double temp)
{
    return cool_CollisionalExciteHe0(temp) + cool_CollisionalIonizeHe0(temp);
}

/*Total collisional cooling for HeP*/
static double
cool_CollisionalHeP(double temp)
{
    return cool_CollisionalExciteHeP(temp) + cool_CollisionalIonizeHeP(temp);
}

/*Recombination cooling rate for H+ and e. Gadget calls this AlphaHp.*/
static double
cool_RecombHp(double logt)
{
    double temp = exp(logt);
    if(CoolingParams.cooling == Enzo2Nyx) {
        /*Recombination cooling rate from Black 1981: these increase rapidly for T > 5e5K. */
        return 2.851e-27 * sqrt(temp) * (5.914 - 0.5 * logt + 0.01184 * pow(temp, 1./3));
    }
    return 0.75 * BOLTZMANN * temp * get_interpolated_recomb(logt, &rec_alphaHp, &recomb_alphaHp);
}

/*Recombination cooling rate for He+ and e. Gadget calls this Alphad */
static double
cool_RecombDielect(double logt)
{
    /*What is this magic number?*/
    return 6.526e-11* get_interpolated_recomb(logt, &rec_alphad, &recomb_alphad);
}

/*Recombination cooling rate for He+ and e. Gadget calls this AlphaHep.*/
static double
cool_RecombHeP(double logt)
{
    return 0.75 * BOLTZMANN * exp(logt) * get_interpolated_recomb(logt, &rec_alphaHep, &recomb_alphaHep) + cool_RecombDielect(logt);
}

/*Recombination cooling rate for He+ and e. Gadget calls this AlphaHepp.*/
static double
cool_RecombHePP(double logt)
{
    double temp = exp(logt);
    if(CoolingParams.cooling == Enzo2Nyx) {
        /*Recombination cooling rate from Black 1981: these increase rapidly for T > 5e5K. */
        return 1.140e-26 * sqrt(temp) * (6.607 - 0.5 * logt + 7.459e-3 * pow(temp, 1./3));
    }
    return 0.75 * BOLTZMANN * temp * get_interpolated_recomb(logt, &rec_alphaHepp, &recomb_alphaHepp);
}

/*Free-free cooling rate for electrons scattering on ions without being captured.
Factors here are n_e and total ionized species:
    (FreeFree(zz=1)*(n_H+ + n_He+) + FreeFree(zz=2)*n_He++)*/
static double
cool_FreeFree(double temp, int zz)
{
    /*Formula for the Gaunt factor. KWH takes this from Spitzer 1978.*/
    double gff = 1.1+0.34*exp(-pow(5.5 - log10(temp),2) /3.);
    /*Formula for the Gaunt factor from Shapiro & Kang 1987. ZZ is 1 for H+ and He+ and 2 for He++.
      This is almost identical to the KWH rate but not continuous.*/
    if(CoolingParams.cooling == Enzo2Nyx) {
        double lt = 2 * log10(temp/zz);
        if(pow(temp/zz,2) <= 3.2e5)
            gff = (0.79464 + 0.1243*lt);
        else
            gff = ( 2.13164 - 0.1240 * lt);
    }
    return 1.426e-27*sqrt(temp)* pow(zz,2) * gff;
}

/* Cooling rate for inverse Compton from the microwave background.
 * Multiply this only by n_e. Note the CMB temperature is hardcoded in KWH92 to 2.7.*/
static double
cool_InverseCompton(double temp, double redshift)
{
        double tcmb_red = CoolingParams.CMBTemperature * (1+redshift);
        return 4 * THOMPSON * RAD_CONST / (ELECTRONMASS * LIGHTCGS ) * pow(tcmb_red, 4) * BOLTZMANN * (temp - tcmb_red);
}

/* This function modifies the photoheating rates by
 * a density dependent factor.
 * This is a hack to attempt to account for helium reionisation,
 * especially for the Lyman alpha forest.
 * It is not a good model for helium reionisation, and needs to be replaced!
 * Takes hydrogen number density in cgs units.
 */
static double
cool_he_reion_factor(double nHcgs, double helium, double redshift)
{
  if(!CoolingParams.HeliumHeatOn)
      return 1.;
  const double rho = PROTONMASS * nHcgs / (1 - helium);
  double overden = rho/(CoolingParams.rho_crit_baryon * pow(1+redshift,3.0));
  if (overden >= CoolingParams.HeliumHeatThresh)
      overden = CoolingParams.HeliumHeatThresh;
  return CoolingParams.HeliumHeatAmp*pow(overden, CoolingParams.HeliumHeatExp);
}

/*Initialize the cooling rate module. This builds a lot of interpolation tables.
 * Defaults: TCMB 2.7255, recomb = Verner96, cooling = Sherwood.*/
void
init_cooling_rates(const char * TreeCoolFile, struct cooling_params coolpar)
{
    CoolingParams = coolpar;

    /*Initialize the interpolation for the self-shielding module as a function of redshift.*/
    GrayOpac = gsl_interp_alloc(gsl_interp_cspline,NGRAY);
    gsl_interp_init(GrayOpac,GrayOpac_zz,GrayOpac_ydata, NGRAY);
    GrayOpac_acc = gsl_interp_accel_alloc();

    if(strlen(TreeCoolFile) == 0) {
        CoolingParams.PhotoIonizationOn = 0;
        message(0, "No TreeCool file is provided. Cooling is broken. OK for DM only runs. \n");
    }
    else {
        message(0, "Using uniform UVB from file %s\n", TreeCoolFile);
        /* Load the TREECOOL into Gamma_HI->ydata, and initialise the interpolators*/
        load_treecool(TreeCoolFile);
    }

    /*Initialize the recombination tables*/
    temp_tab = mymalloc("Recombination_tables", NRECOMBTAB * sizeof(double) * 11);
    rec_GammaH0.ydata = temp_tab + NRECOMBTAB;
    rec_GammaHe0.ydata = temp_tab + 2 * NRECOMBTAB;
    rec_GammaHep.ydata = temp_tab + 3 * NRECOMBTAB;
    rec_alphaHp.ydata = temp_tab + 4 * NRECOMBTAB;
    rec_alphaHep.ydata = temp_tab + 5 * NRECOMBTAB;
    rec_alphad.ydata = temp_tab + 6 * NRECOMBTAB;
    rec_alphaHepp.ydata = temp_tab + 7 * NRECOMBTAB;
    cool_collisH0.ydata = temp_tab + 8 * NRECOMBTAB;
    cool_collisHe0.ydata = temp_tab + 9 * NRECOMBTAB;
    cool_collisHeP.ydata = temp_tab + 10 * NRECOMBTAB;

    int i;
    double Tmin = log(10), Tmax = log(1e10);
    for(i = 0 ; i < NRECOMBTAB; i++)
    {
        temp_tab[i] = Tmin + (Tmax - Tmin) * i / NRECOMBTAB;
        double tt = exp(temp_tab[i]);
        rec_GammaH0.ydata[i] = recomb_GammaeH0(tt);
        rec_GammaHe0.ydata[i] = recomb_GammaeHe0(tt);
        rec_GammaHep.ydata[i] = recomb_GammaeHe0(tt);
        rec_alphaHp.ydata[i] = recomb_alphaHp(tt);
        rec_alphaHep.ydata[i] = recomb_alphaHep(tt);
        rec_alphad.ydata[i] = recomb_alphad(tt);
        rec_alphaHepp.ydata[i] = recomb_alphaHepp(tt);
        cool_collisH0.ydata[i] = cool_CollisionalH0(tt);
        cool_collisHe0.ydata[i] = cool_CollisionalHe0(tt);
        cool_collisHeP.ydata[i] = cool_CollisionalHeP(tt);

    }
    init_itp_type(temp_tab, &rec_GammaH0, NRECOMBTAB);
    init_itp_type(temp_tab, &rec_GammaHe0, NRECOMBTAB);
    init_itp_type(temp_tab, &rec_GammaHep, NRECOMBTAB);
    init_itp_type(temp_tab, &rec_alphaHp, NRECOMBTAB);
    init_itp_type(temp_tab, &rec_alphaHep, NRECOMBTAB);
    init_itp_type(temp_tab, &rec_alphad, NRECOMBTAB);
    init_itp_type(temp_tab, &rec_alphaHepp, NRECOMBTAB);
    init_itp_type(temp_tab, &cool_collisH0, NRECOMBTAB);
    init_itp_type(temp_tab, &cool_collisHe0, NRECOMBTAB);
    init_itp_type(temp_tab, &cool_collisHeP, NRECOMBTAB);
}

/*Get the total (photo) heating and cooling rate for a given temperature (internal energy) and density.
  density is total gas density in protons/cm^3
  Internal energy is in ergs/g.
  helium is a mass fraction, 1 - HYDROGEN_MASSFRAC = 0.24 for primordial gas.
  Returns heating - cooling.
  Sets ne_equilib to the new equilibrium electron density.
 */
double
get_heatingcooling_rate(double density, double ienergy, double helium, double redshift, const struct UVBG * uvbg, double *ne_equilib, double *temp_ext)
{
    double ne = get_equilib_ne(density, ienergy, helium, redshift, uvbg, *ne_equilib);
    if(!isfinite(ne))
        endrun(4," Bad electron density: %g\n", ne);
    double nh = density * (1 - helium);
    *temp_ext = get_temp_internal(ne/nh, ienergy, helium);
    double logt = log(*temp_ext);
    double nH0 = nH0_internal(nh, logt, ne, redshift, uvbg);
    double nHe0 = nHe0_internal(nh, logt, ne, redshift, uvbg);
    double nHp = nHp_internal(nh, logt, ne, redshift, uvbg);
    double nHep = nHep_internal(nh, logt, ne, redshift, uvbg);
    double nHepp = nHepp_internal(nh, logt, ne, redshift, uvbg);
    /*Collisional ionization and excitation rate*/
    double LambdaCollis = ne * (get_interpolated_recomb(logt, &cool_collisH0, cool_CollisionalH0) * nH0 +
            get_interpolated_recomb(logt, &cool_collisHe0, cool_CollisionalHe0) * nHe0 +
            get_interpolated_recomb(logt, &cool_collisHeP, cool_CollisionalHeP) * nHep);
    double LambdaRecomb = ne * (cool_RecombHp(logt) * nHp + cool_RecombHeP(logt) * nHep + cool_RecombHePP(logt) * nHepp);
    /*Free-free cooling rate*/
    if(!isfinite(LambdaRecomb) || !isfinite(LambdaCollis))
        endrun(4," Bad electron density: %g\n", ne);

    double LambdaFF = ne * (cool_FreeFree(*temp_ext, 1) * (nHp + nHep) + cool_FreeFree(*temp_ext, 2) * nHepp);
    double LambdaCmptn = ne * cool_InverseCompton(*temp_ext, redshift);
    /*Total cooling rate*/
    double Lambda = LambdaCollis + LambdaRecomb + LambdaFF + LambdaCmptn;

    double Heat = nH0 * uvbg->epsH0 + nHe0 * uvbg->epsHe0 + nHep * uvbg->epsHep;

    Heat *= cool_he_reion_factor(density, helium, redshift);
    /*Set external equilibrium electron density*/
    *ne_equilib = ne;

    return Heat - Lambda;
}

/*Get the equilibrium temperature at given internal energy.
    density is total gas density in protons/cm^3
    Internal energy is in ergs/g.
    helium is a mass fraction*/
double
get_temp(double density, double ienergy, double helium, double redshift, const struct UVBG * uvbg, double * ne_init)
{
    double ne = get_equilib_ne(density, ienergy, helium, redshift, uvbg, *ne_init);
    double nh = density * (1 - helium);
    *ne_init = ne;
    return get_temp_internal(ne/nh, ienergy, helium);
}

/*Get the neutral hydrogen fraction at a given temperature and density.
density is gas density in protons/cm^3
Internal energy is in ergs/g.
helium is a mass fraction.*/
double
get_neutral_fraction(double density, double ienergy, double helium, double redshift, const struct UVBG * uvbg, double * ne_init)
{
    double ne = get_equilib_ne(density, ienergy, helium, redshift, uvbg, *ne_init);
    double nh = density * (1-helium);
    double temp = get_temp_internal(ne/nh, ienergy, helium);
    *ne_init = ne;
    return nH0_internal(nh, log(temp), ne, redshift, uvbg) / nh;
}
