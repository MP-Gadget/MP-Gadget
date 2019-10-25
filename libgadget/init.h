#ifndef INIT_H
#define INIT_H

#include "domain.h"

/* Loads a snapshot, finds smoothing lengths and does the initial domain decomposition*/
void init(int snapnum, DomainDecomp * ddecomp);

#endif
