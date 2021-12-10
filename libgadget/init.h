#ifndef INIT_H
#define INIT_H

#include "domain.h"
#include "utils/paramset.h"
#include "timebinmgr.h"

/* Loads a snapshot, finds smoothing lengths and does the initial domain decomposition*/
inttime_t init(int snapnum, DomainDecomp * ddecomp);

void set_init_params(ParameterSet * ps);

void set_all_global_params(ParameterSet * ps);
#endif
