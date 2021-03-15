#ifndef __BLACKHOLE_H
#define __BLACKHOLE_H
#include "utils/paramset.h"
#include "timestep.h"
#include "forcetree.h"

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
 * TimeNextSeedingCheck is the time of the BH next seeding check.
 * It will be compared to the current time and updated after seeding takes place.
 * tree is a valid ForceTree.
 */
void blackhole(const ActiveParticles * act, ForceTree * tree, FILE * FdBlackHoles, FILE * FdBlackholeDetails);

/* Make a black hole from the particle at index. */
void blackhole_make_one(int index);

/* Decide whether black hole repositioning is enabled. */
int BHGetRepositionEnabled(void);
#endif
