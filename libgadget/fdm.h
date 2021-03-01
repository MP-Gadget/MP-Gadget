#ifndef FDM_H
#define FDM_H

#include "forcetree.h"
#include "timestep.h"
#include "utils/paramset.h"
#include "slotsmanager.h"

void set_fdm_params(ParameterSet * ps);

void dm_density(const ActiveParticles * act, const ForceTree * const tree);
void quantum_pressure(const ActiveParticles * act, const ForceTree * const tree);

#endif
