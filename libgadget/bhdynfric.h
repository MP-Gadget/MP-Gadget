#ifndef BHDYNFRIC_H
#define BHDYNFRIC_H

#include "utils/paramset.h"
#include "forcetree.h"
#include "density.h"

/* Parameters used in the dynamic friction treewalk. */
struct BHDynFricPriv {
    MyFloat * BH_SurroundingDensity;
    int * BH_SurroundingParticles;
    MyFloat (*BH_SurroundingVel)[3];
    MyFloat * BH_SurroundingRmsVel;

    /* Time factors*/
    double atime;
    Cosmology * CP;
    struct kick_factor_data * kf;
};

/* Do the dynamic friction treewalk if BH_DynFrictionMethod > 0.
 * Tree needs stars and gas.*/
void blackhole_dynfric(int * ActiveBlackHoles, int64_t NumActiveBlackHoles, ForceTree * tree, struct BHDynFricPriv * priv);
void set_blackhole_dynfric_params(ParameterSet * ps);
void blackhole_dynpriv_free(struct BHDynFricPriv * dynpriv);
/* Get the particle types used in dynfric*/
int blackhole_dynfric_treemask(void);

#endif
