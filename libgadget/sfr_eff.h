#ifndef __SFR_H
#define __SFR_H

#include "forcetree.h"
#include "utils/paramset.h"
#include "utils/system.h"
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

void init_cooling_and_star_formation(int CoolingOn, int StarformationOn, Cosmology * CP, const double avg_baryon_mass, const double BoxSize, const struct UnitSystem units);
/*Do the cooling and the star formation. The tree is required for the winds only.*/
void cooling_and_starformation(ActiveParticles * act, double Time, double dloga, ForceTree * tree, struct grav_accel_store GravAccel, DomainDecomp * ddecomp, Cosmology *CP, MyFloat * GradRho, RandTable * rnd, FILE * FdSfr);

/*Get the neutral fraction of a particle correctly, even when on the star-forming equation of state.
 * This calls the cooling routines for the current internal energy when off the equation of state, but
 * when on the equation of state calls them separately for the cold and hot gas.*/
double get_neutral_fraction_sfreff(double redshift, double hubble, struct particle_data * partdata, struct sph_particle_data * sphdata);

/*Get the helium ionic fraction of a particle correctly, even when on the star-forming equation of state.
 * This calls the cooling routines for the current internal energy when off the equation of state, but
 * when on the equation of state calls them separately for the cold and hot gas.*/
double get_helium_neutral_fraction_sfreff(int ion, double redshift, double hubble, struct particle_data * partdata, struct sph_particle_data * sphdata);

/* Return whether we are using a star formation model that needs grad rho computed for the gas particles*/
int sfr_need_to_compute_sph_grad_rho(void);

/* Get the number of generations of stars that may form*/
int get_generations(void);

/* Returns 1 if particle is on effective EOS, 0 otherwise*/
int sfreff_on_eeqos(const struct sph_particle_data * sph, const double a3inv);

/* Get the Minimum temperature in internal energy*/
double get_MinEgySpec(void);

/* Returns the density threshold for star formation in comoving units*/
double sfr_density_threshold(const double atime);

#endif
