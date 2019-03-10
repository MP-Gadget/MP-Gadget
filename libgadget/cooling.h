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
    double self_shield_dens;
};

/*Global unit system for the cooling module*/
struct cooling_units
{
    /*Flag to enable or disable all cooling.*/
    int CoolingOn;
    /*Factor to convert internal density to cgs. By default should be UnitDensity_in_cgs * h^2 */
    double density_in_phys_cgs; //All.UnitDensity_in_cgs * All.CP.HubbleParam * All.CP.HubbleParam
    /*Factor to convert internal internal energy to cgs. By default should be UnitPressure_in_cgs / UnitDensity_in_cgs */
    double uu_in_cgs; //All.UnitPressure_in_cgs / All.UnitDensity_in_cgs
    /*Factor to convert time to s. By default should be UnitTime_in_s / h */
    double tt_in_s; //All.UnitTime_in_s / All.CP.HubbleParam
};

/*Initialise the unit system for the cooling module.*/
void init_cool_units(struct cooling_units cu);

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
double get_temp(double density, double ienergy, double helium, const struct UVBG * uvbg, double * ne_init);

/*Get the neutral hydrogen fraction at a given temperature and density.
density is gas density in protons/cm^3
Internal energy is in J/kg == 10^-10 ergs/g.
helium is a mass fraction.*/
double get_neutral_fraction(double density, double ienergy, double helium, const struct UVBG * uvbg, double * ne_init);

#endif
