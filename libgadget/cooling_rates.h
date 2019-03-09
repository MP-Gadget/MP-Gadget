/*Support header for functions used internally in the cooling module*/

#ifndef _COOLING_RATES_H_
#define _COOLING_RATES_H_

#include "cooling.h"

/*Initialize the cooling rate module. This builds a lot of interpolation tables.
 * Defaults: TCMB 2.7255, recomb = Verner96, cooling = Sherwood.*/
void init_cooling_rates(const char * TreeCoolFile, struct cooling_params coolpar);

/*Reads and initialises the tables for a spatially varying redshift of reionization*/
void init_uvf_table(const char * UVFluctuationFile, double UVRedshiftThreshold);

/* Read a big array from filename/dataset into an array, allocating memory in buffer.
 * which is returned. Nread argument is set equal to number of elements read.*/
double * read_big_array(const char * filename, char * dataset, int * Nread);

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
double get_heatingcooling_rate(double density, double ienergy, double helium, double redshift, const struct UVBG * uvbg, double * ne_equilib);

#endif
