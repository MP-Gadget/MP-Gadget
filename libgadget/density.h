#ifndef DENSITY_H
#define DENSITY_H

#include "forcetree.h"

/* This routine computes the particle densities. If update_hsml is true
 * it runs multiple times, changing the smoothing length until
 * there are enough neighbours. If update_hsml is false (when initializing the EgyWtDensity)
 * it just computes densities.
 * If DoEgyDensity is true it also computes the entropy-weighted density for
 * pressure-entropy SPH. */
void density(int update_hsml, int DoEgyDensity, ForceTree * tree);

#endif
