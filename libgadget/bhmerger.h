#ifndef BH_MERGER
#define BH_MERGER

#include "domain.h"

typedef struct {
    /* These are temporaries used when accreting particles to store masses
     * for the BH Details file, and avoid roundoff.*/
    MyFloat * BH_accreted_Mass;
    MyFloat * BH_accreted_BHMass;
    MyFloat (*BH_accreted_momentum)[3];
} AccretedVariables;

/* Allocate and free the accreted variables*/
void
bh_alloc_accreted(AccretedVariables * accrete);
void
bh_free_accreted(AccretedVariables * accrete);

/* Merge two nearby black holes. Inactive BHs are merged into active ones. No inactive BHs increase their mass.
 * Arguments: ActiveBlackHoles - candidates for accreting another black hole.
              NumActiveBlackHoles - number of active black holes.
              ddecomp - domain decomposition for building a BH tree.
              kf - velocity prediction data.
              atime - current time.
              SeedBHDynMass - initial BH dynamic mass. Useful for keeping track of the mergers.
              MergeGravBound - whether to check for the BH being gravitationally bound. Note this should be 0 if BH repositioning is on.
    Returns: the number of BH mergers.*/
int64_t
blackhole_mergers(int * ActiveBlackHoles, const int64_t NumActiveBlackHoles, DomainDecomp * ddecomp, struct kick_factor_data *kf, const double atime, const double SeedBHDynMass, const int MergeGravBound, AccretedVariables * BHaccrete);


#endif
