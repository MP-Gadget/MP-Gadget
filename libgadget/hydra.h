#ifndef HYDRA_H
#define HYDRA_H

#include "forcetree.h"
#include "types.h"

/*Function to get the pressure from the entropy and the density*/
double PressurePred(int i);

/* Functions to get the center of mass density and HSML correction factor for an SPH particle with index i.
 * These encode the main difference between pressure-entropy SPH and regular SPH.*/
MyFloat SPH_EOMDensity(int i);
MyFloat SPH_DhsmlDensityFactor(int i);

/*Function to compute hydro accelerations and adiabatic entropy change*/
void hydro_force(ForceTree * tree);

#endif
