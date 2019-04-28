#ifndef __BLACKHOLE_H
#define __BLACKHOLE_H
#include "utils/paramset.h"

/*Set the parameters of the star formation module*/
void set_blackhole_params(ParameterSet * ps);

/* Does the black hole feedback and accretion.
 * Sets TimeNextSeedingCheck to the scale factor of the next BH seeding check*/
void blackhole(ForceTree * tree, double * TimeNextSeedingCheck);
void blackhole_make_one(int index);

#endif
