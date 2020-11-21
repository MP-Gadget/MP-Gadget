#ifndef METAL_RETURN_H
#define METAL_RETURN_H

#include "forcetree.h"
#include "timestep.h"
#include "utils/paramset.h"

/*Function to compute metal return from star particles, adding metals to the gas.*/
void metal_return(const ActiveParticles * act, const double atime, const ForceTree * const tree);

void set_metal_return_params(ParameterSet * ps);

#endif
