#ifndef _COOLING_H_
#define _COOLING_H_

/* Ultra-violet background structure.
 * Can be changed on a particle-by-particle basis*/
struct UVBG {
    double J_UV;
    double gJH0;
    double gJHep;
    double gJHe0;
    double epsH0;
    double epsHep;
    double epsHe0;
};

void GetParticleUVBG(int i, struct UVBG * uvbg);
void GetGlobalUVBG(struct UVBG * uvbg);
double AbundanceRatios(double u, double rho, struct UVBG * uvbg, double *ne_guess, double *nH0_pointer, double *nHeII_pointer);
double GetCoolingTime(double u_old, double rho, struct UVBG * uvbg,  double *ne_guess, double Z);
double DoCooling(double u_old, double rho, double dt, struct UVBG * uvbg, double *ne_guess, double Z);
double ConvertInternalEnergy2Temperature(double u, double ne);

void   InitCool(void);
void   IonizeParams(void);
void   MakeCoolingTable(void);
void   SetZeroIonization(void);

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

/*Initialize the cooling rate module. This builds a lot of interpolation tables.
 * Defaults: TCMB 2.7255, recomb = Verner96, cooling = Sherwood.*/
void init_cooling_rates(const char * TreeCoolFile, struct cooling_params coolpar);

/*Interpolates the ultra-violet background tables to the desired redshift and returns a cooling rate table*/
struct UVBG get_global_UVBG(double redshift);

/*Reads and initialises the tables for a spatially varying redshift of reionization*/
void init_uvf_table(const char * UVFluctuationFile, double UVRedshiftThreshold);

/* Change the ultra-violet background table according to a pre-computed table of UV fluctuations.
 * This zeros the UVBG if this particular particle has not reionized yet*/
struct UVBG get_particle_UVBG(double redshift, double * Pos, const struct UVBG * GlobalUVBG);

/*Solve the system of equations for photo-ionization equilibrium,
  starting with ne = nH and continuing until convergence.
  density is gas density in protons/cm^3
  Internal energy is in J/kg == 10^-10 ergs/g.
  helium is a mass fraction.
*/
double get_equilib_ne(double density, double ienergy, double helium, double redshift, const struct UVBG * uvbg);

/*Same as above, but get electrons per proton.*/
double get_ne_by_nh(double density, double ienergy, double helium, double redshift, const struct UVBG * uvbg);

/*Get the total (photo) heating and cooling rate for a given temperature (internal energy) and density.
  density is total gas density in protons/cm^3
  Internal energy is in J/kg == 10^-10 ergs/g.
  helium is a mass fraction, 1 - HYDROGEN_MASSFRAC = 0.24 for primordial gas.
  Returns heating - cooling.
 */
double get_heatingcooling_rate(double density, double ienergy, double helium, double redshift, const struct UVBG * uvbg);

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
