#ifndef _COOLING_H_
#define _COOLING_H_

/* Ultra-violet background structure.
 * Can be changed on a particle-by-particle basis*/
struct UVBG {
    double gJH0;
    double gJHep;
    double gJHe0;
    double epsH0;
    double epsHep;
    double epsHe0;
};

/* Definitions for the cooling rates code*/
enum RecombType {
    Cen92 = 0, // Recombination from Cen 92
    Verner96 = 1, //Verner 96 recombination rates. Basically accurate, used by Sherwood and Nyx by default.
    Badnell = 2, //Even more up to date rates from Badnell 2006, cloudy's current default.
};

enum CoolingType {
    KWH92 = 0, //Cooling from KWH 92
    Enzo2Nyx = 1, //Updated cooling rates from Avery Meiksin, used in Nyx and Enzo 2.
    Sherwood =2 ,  //Same as KWH92, but with an improved large temperature correction factor.
};


struct cooling_params
{
    /*Default: Verner96*/
    enum RecombType recomb;
    /*Default: Sherwood*/
    enum CoolingType cooling;

    /*Enable a self-shielding cooling and ionization correction from Rahmati & Schaye 2013. Default: on.*/
    int SelfShieldingOn;
    /*Disable using photo-ionization and heating table, for testing.*/
    int PhotoIonizationOn;
    /*Global baryon fraction, Omega_b/Omega_cdm, used for the self-shielding formula.*/
    double fBar;

    /*Normalization factor to apply to the UVB: Default: 1.*/
    double PhotoIonizeFactor;

    /*CMB temperature in K*/
    double CMBTemperature;

    /*Parameters for the 'extra heating' Helium photoionization model.*/
    int HeliumHeatOn;
    double HeliumHeatThresh;
    double HeliumHeatAmp;
    double HeliumHeatExp;
    double rho_crit_baryon;
};

/*Initialise the cooling module and subsidiaries.*/
void InitCool(void);

/* Get the cooling time for a particle from the internal energy and density, specifying a UVB appropriately.
 * Sets ne_guess to the equilibrium electron density.*/
double GetCoolingTime(double redshift, double u_old, double rho, struct UVBG * uvbg,  double *ne_guess, double Z);

/*Get the new internal energy per unit mass. ne_guess is set to the new internal equilibrium electron density*/
double DoCooling(double redshift, double u_old, double rho, double dt, struct UVBG * uvbg, double *ne_guess, double Z);

/*Sets the global variable corresponding to the uniform part of the UV background.*/
void set_global_uvbg(double redshift);

/*Interpolates the ultra-violet background tables to the desired redshift and returns a cooling rate table*/
struct UVBG get_global_UVBG(double redshift);

/* Change the ultra-violet background table according to a pre-computed table of UV fluctuations.
 * This zeros the UVBG if this particular particle has not reionized yet*/
struct UVBG get_particle_UVBG(double redshift, double * Pos);

/*Get the equilibrium temperature at given internal energy.
    density is total gas density in protons/cm^3
    Internal energy is in J/kg == 10^-10 ergs/g.
    helium is a mass fraction*/
double get_temp(double density, double ienergy, double helium, double redshift, const struct UVBG * uvbg);

/*Get the neutral hydrogen fraction at a given temperature and density.
density is gas density in protons/cm^3
Internal energy is in J/kg == 10^-10 ergs/g.
helium is a mass fraction.*/
double get_neutral_fraction(double density, double ienergy, double helium, double redshift, const struct UVBG * uvbg);

#endif
