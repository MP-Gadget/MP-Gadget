#ifndef HYDRA_H
#define HYDRA_H

#include "forcetree.h"
#include "timestep.h"
#include "density.h"
#include "utils/paramset.h"

/*Function to compute hydro accelerations and adiabatic entropy change*/
void hydro_force(const ActiveParticles * act, const double atime, struct sph_pred_data * SPH_predicted, const DriftKickTimes times,  Cosmology * CP, const ForceTree * const tree);

void set_hydro_params(ParameterSet * ps);

/* Gets whether we are using Density Independent Sph*/
int DensityIndependentSphOn(void);

#endif
