#ifndef INIT_H
#define INIT_H

#include "domain.h"
#include "utils/paramset.h"
#include "timebinmgr.h"
#include "petaio.h"

/* Loads and validates a particle table and initialise properties of the particle distribution.*/
inttime_t init(int RestartSnapNum, const char * OutputDir, struct header_data * header, const double PartAllocFactor, double TimeMax, Cosmology * CP, const int SnapshotWithFOF);

/* Finds smoothing lengths and the energy weighted density*/
void setup_smoothinglengths(int RestartSnapNum, DomainDecomp * ddecomp, Cosmology * CP, int BlackHoleOn, double MinEgySpec, double uu_in_cgs, const inttime_t Ti_Current, const double atime, const int64_t NTotGasInit);

void set_init_params(ParameterSet * ps);

void init_timeline(int RestartSnapNum, double TimeMax, const struct header_data * header, const int SnapshotWithFOF);

#endif
