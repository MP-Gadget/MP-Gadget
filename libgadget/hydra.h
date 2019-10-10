#ifndef HYDRA_H
#define HYDRA_H

#include "forcetree.h"
#include "types.h"
#include "timestep.h"

/* Function to get the center of mass density and HSML correction factor for an SPH particle with index i.
 * Encodes the main difference between pressure-entropy SPH and regular SPH.*/
MyFloat SPH_EOMDensity(int i);

/*Function to compute hydro accelerations and adiabatic entropy change*/
void hydro_force(const ActiveParticles * act, ForceTree * tree);

#endif
