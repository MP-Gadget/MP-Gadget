#ifndef HYDRA_H
#define HYDRA_H

#include "forcetree.h"
#include "types.h"
#include "timestep.h"
#include "utils/paramset.h"

/* Function to get the center of mass density and HSML correction factor for an SPH particle with index i.
 * Encodes the main difference between pressure-entropy SPH and regular SPH.*/
MyFloat SPH_EOMDensity(int i);

/*Function to compute hydro accelerations and adiabatic entropy change*/
void hydro_force(const ActiveParticles * act, int WindOn, const double HydroCostFactor, const double hubble, const double atime, ForceTree * tree);

void set_hydro_params(ParameterSet * ps);

/* Gets whether we are using Density Independent Sph*/
int DensityIndependentSphOn(void);

#endif
