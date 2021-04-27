#ifndef __SFR_H
#define __SFR_H

#include "forcetree.h"
#include "utils/paramset.h"
#include "timestep.h"
#include "partmanager.h"
#include "slotsmanager.h"

#define  METAL_YIELD       0.02	/*!< effective metal yield for star formation */

/*
 * additional sfr criteria
 */
enum StarformationCriterion {
    SFR_CRITERION_DENSITY = 1,
    SFR_CRITERION_MOLECULAR_H2 = 3, /* 2 + 1 */
    SFR_CRITERION_SELFGRAVITY = 5,  /* 4 + 1 */
    /* below are additional flags in SELFGRAVITY */
    SFR_CRITERION_CONVERGENT_FLOW = 13, /* 8 + 4 + 1 */
    SFR_CRITERION_CONTINUOUS_CUTOFF= 21, /* 16 + 4 + 1 */
};

/*Set the parameters of the star formation module*/
void set_sfr_params(ParameterSet * ps);

void init_cooling_and_star_formation(int CoolingOn);
/*Do the cooling and the star formation. The tree is required for the winds only.*/
void cooling_and_starformation(ActiveParticles * act, ForceTree * tree, MyFloat * GradRho, FILE * FdSfr);

/*Get the neutral fraction of a particle correctly, even when on the star-forming equation of state.
 * This calls the cooling routines for the current internal energy when off the equation of state, but
 * when on the equation of state calls them separately for the cold and hot gas.*/
double get_neutral_fraction_sfreff(double redshift, struct particle_data * partdata, struct sph_particle_data * sphdata);

/*Get the helium ionic fraction of a particle correctly, even when on the star-forming equation of state.
 * This calls the cooling routines for the current internal energy when off the equation of state, but
 * when on the equation of state calls them separately for the cold and hot gas.*/
double get_helium_neutral_fraction_sfreff(int ion, double redshift, struct particle_data * partdata, struct sph_particle_data * sphdata);

/* Return whether we are using a star formation model that needs grad rho computed for the gas particles*/
int sfr_need_to_compute_sph_grad_rho(void);

/* Get the number of generations of stars that may form*/
int get_generations(void);

/* Returns 1 if particle is on effective EOS, 0 otherwise*/
int sfreff_on_eeqos(const struct sph_particle_data * sph, const double a3inv);

/* Returns 1 if particle is on effective EOS, 0 otherwise. Does not check if particle is in wind.*/
int sfreff_eqos_density_no_delay(const struct sph_particle_data * sph, const double a3inv);

#endif
