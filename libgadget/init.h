#ifndef INIT_H
#define INIT_H

#include "domain.h"
#include "utils/paramset.h"
#include "timebinmgr.h"

/* Loads and validates a particle table and initialise properties of the particle distribution.*/
inttime_t init(int RestartSnapNum, double TimeIC, double TimeInit, double TimeMax, Cosmology * CP, int SnapshotWithFOF, int MassiveNuLinRespOn, double G, double * MassTable, double * MeanSeparation);

/* Finds smoothing lengths and the energy weighted density*/
void setup_smoothinglengths(int RestartSnapNum, DomainDecomp * ddecomp, Cosmology * CP, int BlackHoleOn, double MinEgySpec, double uu_in_cgs, const inttime_t Ti_Current, const double atime, const double MeanGasSeparation);

void set_init_params(ParameterSet * ps);

void get_mean_separation(double * MeanSeparation, double BoxSize, int64_t * NTotalInit);

#endif
