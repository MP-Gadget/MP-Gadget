#ifndef __DRIFT_H
#define __DRIFT_H
#include "cosmology.h"
#include "types.h"

/* Updates all particles to the current drift time*/
void drift_all_particles(inttime_t ti1, const double BoxSize, Cosmology * CP, const double random_shift[3]);

#endif
