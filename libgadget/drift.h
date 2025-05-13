#ifndef __DRIFT_H
#define __DRIFT_H
#include "cosmology.h"
#include "types.h"
#include "partmanager.h"
#include "slotsmanager.h"

/* Updates all particles to the current drift time*/
void drift_all_particles(inttime_t ti0, inttime_t ti1, Cosmology * CP, const double random_shift[3]);

void real_drift_particle(struct particle_data * pp, struct slots_manager_type * sman, const double ddrift, const double BoxSize, const double random_shift[3], const int NonPeriodic);

struct DriftData
{
    inttime_t ti0;
    inttime_t ti1;
    Cosmology * CP;
};

#endif
