#ifndef METAL_RETURN_H
#define METAL_RETURN_H

#include "forcetree.h"
#include "timestep.h"
#include "utils/paramset.h"

/*Function to compute metal return from star particles, adding metals to the gas.*/
void metal_return(const ActiveParticles * act, const ForceTree * const tree, Cosmology * CP, const double atime, double * StarVolumeSPH);

void set_metal_return_params(ParameterSet * ps);

#endif
