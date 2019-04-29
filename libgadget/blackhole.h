#ifndef __BLACKHOLE_H
#define __BLACKHOLE_H
#include "utils/paramset.h"

enum BlackHoleFeedbackMethod {
     BH_FEEDBACK_TOPHAT   = 0x2,
     BH_FEEDBACK_SPLINE   = 0x4,
     BH_FEEDBACK_MASS     = 0x8,
     BH_FEEDBACK_VOLUME   = 0x10,
     BH_FEEDBACK_OPTTHIN  = 0x20,
};

/*Set the parameters of the star formation module*/
void set_blackhole_params(ParameterSet * ps);

/* Does the black hole feedback and accretion.
 * Sets TimeNextSeedingCheck to the scale factor of the next BH seeding check*/
void blackhole(ForceTree * tree, double * TimeNextSeedingCheck);
void blackhole_make_one(int index);

#endif
