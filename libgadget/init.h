#ifndef INIT_H
#define INIT_H

#include "domain.h"
#include "utils/paramset.h"

/* Loads a snapshot, finds smoothing lengths and does the initial domain decomposition*/
void init(int snapnum, DomainDecomp * ddecomp);

void set_init_params(ParameterSet * ps);

#endif
