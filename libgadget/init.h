#ifndef INIT_H
#define INIT_H

#include "domain.h"
#include "utils/paramset.h"
#include "petaio.h"

/* Loads and validates a particle table and initialise properties of the particle distribution.*/
inttime_t init(int RestartSnapNum, const char * OutputDir, struct header_data * header, Cosmology * CP);

/* Finds smoothing lengths and the energy weighted density*/
void setup_smoothinglengths(int RestartSnapNum, DomainDecomp * ddecomp, Cosmology * CP, int BlackHoleOn, double MinEgySpec, double uu_in_cgs, const inttime_t Ti_Current, const double atime, const int64_t NTotGasInit);

/* When we restart, validate the SPH properties of the particles.
 * This also allows us to increase MinEgySpec on a restart if we choose.*/
void check_density_entropy(Cosmology * CP, const double MinEgySpec, const double atime);

void set_init_params(ParameterSet * ps);

/* Setup a list of sync points until the end of the simulation.*/
void init_timeline(Cosmology * CP, int RestartSnapNum, double TimeMax, const struct header_data * header, const int SnapshotWithFOF);

#endif
