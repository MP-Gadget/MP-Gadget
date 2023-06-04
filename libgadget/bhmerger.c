#include <mpi.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "treewalk.h"
#include "slotsmanager.h"
#include "gravity.h"
#include "walltime.h"
#include "density.h"
#include "bhmerger.h"

/*! \file blackhole.c
 *  \brief routines for gas accretion onto black holes, and black hole mergers
 */
struct BHMergerPriv {
    /* Time factors*/
    double atime;
    /* Counters*/
    int64_t * N_BH_swallowed;
    struct kick_factor_data * kf;
    double SeedBHDynMass; /* The initial dynamic mass of BH particle */
    int MergeGravBound;
    /* Temporary array to store the IDs of the swallowing black hole for mergers.
     * We store ID + 1 so that SwallowID == 0 can correspond to the unswallowed case. */
    MyIDType * BH_SwallowID;
    /* Used in the real merger treewalk to store masses
     * for the BH Details file, and avoid roundoff.*/
    AccretedVariables * accreted;
    /* Count of number of mergers 'expected' (actual mergers may be less than this
     * if one of the merger targets is merging).*/
    int mergercount;
};

#define BH_GET_PRIV(tw) ((struct BHMergerPriv *) (tw->priv))

typedef struct {
    TreeWalkQueryBase base;
    MyFloat Hsml;
    MyFloat Vel[3];
    MyFloat Accel[3];
    MyIDType ID;
} TreeWalkQueryBHMerger;

typedef struct {
    TreeWalkResultBase base;
    int encounter;
    int alignment; //Ensure 64-bit alignment
} TreeWalkResultBHMerger;

/* check if two BHs are gravitationally bounded, input dv, da, dx in code unit */
/* same as Bellovary2011, Tremmel2017 */
static int
check_grav_bound(double dx[3], double dv[3], double da[3], const double atime)
{
    int j;
    double KE = 0;
    double PE = 0;

    for(j = 0; j < 3; j++){
        KE += 0.5 * pow(dv[j], 2);
        PE += da[j] * dx[j];
    }

    KE /= (atime * atime); /* convert to proper velocity */
    PE /= atime; /* convert to proper unit */

    /* The gravitationally bound condition is PE + KE < 0.
     * Still merge if it is marginally bound so that we merge
     * particles at zero distance and velocity from each other.*/
    return (PE + KE <= 0);
}

static void
blackhole_merger_ngbiter(TreeWalkQueryBHMerger * I,
        TreeWalkResultBHMerger * O,
        TreeWalkNgbIterBase * iter,
        LocalTreeWalk * lv)
{

    if(iter->other == -1) {
        O->encounter = 0;
        iter->mask = BHMASK;
        /* Search within Hsml*/
        iter->Hsml = I->Hsml;
        /* Symmetric hxml because black holes may merge symmetrically.*/
        iter->symmetric = NGB_TREEFIND_SYMMETRIC;
        return;
    }

    int other = iter->other;
    double r = iter->r;

    /* Accretion / merger doesn't do self interaction */
    if(P[other].ID == I->ID)
        return;

    if(P[other].Type != 5)
        return;

    /* we have a black hole merger. Now we use 2 times GravitationalSoftening as merging criteria,
     * previously we used the SPH smoothing length. Both need to be satisfied.*/
    if(r >= (2*FORCE_SOFTENING(0,1)/2.8))
        return;

    O->encounter = 1; // mark the event when two BHs encounter each other

    int flag = 0; // the flag for BH merge

    /* apply Grav Bound check only when Reposition is disabled, otherwise BHs would be repositioned to the same location but not merge */
    if(BH_GET_PRIV(lv->tw)->MergeGravBound == 1){
        double dx[3];
        double dv[3];
        double da[3];
        int d;
        MyFloat VelPred[3];
        DM_VelPred(other, VelPred, BH_GET_PRIV(lv->tw)->kf);
        for(d = 0; d < 3; d++){
            dx[d] = NEAREST(I->base.Pos[d] - P[other].Pos[d], PartManager->BoxSize);
            dv[d] = I->Vel[d] - VelPred[d];
            /* we include long range PM force, short range force from the last long timestep and DF */
            da[d] = (I->Accel[d] - P[other].FullTreeGravAccel[d] - P[other].GravPM[d] - BHP(other).DFAccel[d]);
        }
        flag = check_grav_bound(dx,dv,da, BH_GET_PRIV(lv->tw)->atime);
        /*if(flag == 0)
            message(0, "dx %g %g %g dv %g %g %g da %g %g %g\n",dx[0], dx[1], dx[2], dv[0], dv[1], dv[2], da[0], da[1], da[2]);*/
    }
    else {
        flag = 1;
    }

    /* Mark the BH via SwallowID.*/
    if(flag == 1)
    {
        MyIDType readid, newswallowid;

        int PI = P[other].PI;
        MyIDType * swal = BH_GET_PRIV(lv->tw)->BH_SwallowID + PI;
        #pragma omp atomic read
        readid = *swal;

        /* Here we mark the black hole as "ready to be swallowed" using the SwallowID.
            * The actual swallowing is done in the feedback treewalk by setting Swallowed = 1
            * and merging the masses.*/
        do {
            /* Already marked, prefer to be swallowed by a bigger ID */
            if(readid != 0 && readid < I->ID ) {
                /* Already marked, prefer to be swallowed by a bigger ID */
                newswallowid = I->ID + 1;
            } else if(readid == 0 && (P[other].ID < I->ID || !is_timebin_active(P[other].TimeBinHydro, P[other].Ti_drift))) {
                /* Unmarked, the BH with bigger ID swallows This avoids two BHs trying to swallow each other.
                    * (in which case neither would merge). If only one is active, only one can swallow.*/
                newswallowid = I->ID + 1;
            }
            else
                break;
        /* Swap in the new id only if the old one hasn't changed:
            * in principle an extension, but supported on at least clang >= 9, gcc >= 5 and icc >= 18.*/
        } while(!__atomic_compare_exchange_n(swal, &readid, newswallowid, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));

        #pragma omp atomic update
        BH_GET_PRIV(lv->tw)->mergercount++;
    }
}

static void
blackhole_merger_reduce(int place, TreeWalkResultBHMerger * remote, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    /* Set encounter to true if it is true on any remote*/
    if (mode == 0 || BHP(place).encounter < remote->encounter) {
        BHP(place).encounter = remote->encounter;
    }
}

static void
blackhole_merger_copy(int place, TreeWalkQueryBHMerger * I, TreeWalk * tw)
{
    int k;
    for(k = 0; k < 3; k++)
    {
        I->Vel[k] = P[place].Vel[k];
        I->Accel[k] = P[place].FullTreeGravAccel[k] + P[place].GravPM[k] + BHP(place).DFAccel[k];
    }
    I->Hsml = P[place].Hsml;
    I->ID = P[place].ID;
}

typedef struct {
    TreeWalkQueryBase base;
    MyFloat Hsml;
    MyFloat Mtrack;
    MyIDType ID;
} TreeWalkQueryBHRealMerger;

typedef struct {
    TreeWalkResultBase base;
    MyFloat Mass; /* the accreted Mdyn */
    MyFloat AccretedMomentum[3];
    MyFloat BH_Mass;
    int BH_CountProgs;
    int alignment; /* Ensure alignment*/
} TreeWalkResultBHRealMerger;

/**
 * perform blackhole swallow / merger;
 */
static void
blackhole_real_merger_ngbiter(TreeWalkQueryBHRealMerger * I,
        TreeWalkResultBHRealMerger * O,
        TreeWalkNgbIterBase * iter,
        LocalTreeWalk * lv)
{
    if(iter->other == -1) {
        iter->mask = BHMASK;
        iter->Hsml = I->Hsml;
        iter->symmetric = NGB_TREEFIND_SYMMETRIC;
        return;
    }

    int other = iter->other;
    /* Exclude self interaction */
    if(P[other].ID == I->ID) return;

    /* we have a black hole merger! */
    int PI = P[other].PI;
    if(P[other].Type == 5 && BH_GET_PRIV(lv->tw)->BH_SwallowID[PI] != 0)
    {
        if(BH_GET_PRIV(lv->tw)->BH_SwallowID[PI] != I->ID + 1) return;

        /* Swallow the particle*/
        /* A note on Swallowed vs SwallowID: black hole particles which have been completely swallowed
         * (ie, their mass has been added to another particle) have Swallowed = 1.
         * These particles are ignored in future tree walks. SwallowID is set to the swallowing particle.
         */
        BHP(other).SwallowID = BH_GET_PRIV(lv->tw)->BH_SwallowID[PI] - 1;
        BHP(other).SwallowTime = BH_GET_PRIV(lv->tw)->atime;
        P[other].Swallowed = 1;
        /* Set encounter to zero when we merge*/
        BHP(other).encounter = 0;
        O->BH_CountProgs += BHP(other).CountProgs;
        O->BH_Mass += (BHP(other).Mass);

        /* Active mass tracer for the other BH*/
        double othermass = P[other].Mass;
        if (BH_GET_PRIV(lv->tw)->SeedBHDynMass>0 && I->Mtrack>0){
            /* If the other BH has not yet reached the dynamical mass, we should accrete the Mtrack*/
            if(BHP(other).Mtrack < BH_GET_PRIV(lv->tw)->SeedBHDynMass)
                othermass = BHP(other).Mtrack;
        }
        /* Add the accreted mass tracer to the total accreted Mass.
         * We will decide in postprocess if it goes into Mtrack or Mass*/
        O->Mass += othermass;

        MyFloat VelPred[3];
        DM_VelPred(other, VelPred, BH_GET_PRIV(lv->tw)->kf);
        /* Conserve momentum during accretion*/
        int d;
        for(d = 0; d < 3; d++)
            O->AccretedMomentum[d] += (othermass * VelPred[d]);

        if(BHP(other).SwallowTime < BH_GET_PRIV(lv->tw)->atime)
            endrun(2, "Encountered BH %i swallowed at earlier time %g\n", other, BHP(other).SwallowTime);

        int tid = omp_get_thread_num();
        BH_GET_PRIV(lv->tw)->N_BH_swallowed[tid]++;
    }
}

static int
blackhole_real_merger_haswork(int n, TreeWalk * tw)
{
    /*Black hole not being swallowed*/
    return (P[n].Type == 5) && (!P[n].Swallowed) && (BH_GET_PRIV(tw)->BH_SwallowID[P[n].PI] == 0);
}

static void
blackhole_real_merger_copy(int i, TreeWalkQueryBHRealMerger * I, TreeWalk * tw)
{
    I->Hsml = P[i].Hsml;
    I->ID = P[i].ID;
    I->Mtrack = BHP(i).Mtrack;
}


static void
blackhole_real_merger_reduce(int place, TreeWalkResultBHRealMerger * remote, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    int k;
    int PI = P[place].PI;

    TREEWALK_REDUCE(BH_GET_PRIV(tw)->accreted->BH_accreted_Mass[PI], remote->Mass);
    TREEWALK_REDUCE(BH_GET_PRIV(tw)->accreted->BH_accreted_BHMass[PI], remote->BH_Mass);
    for(k = 0; k < 3; k++) {
        TREEWALK_REDUCE(BH_GET_PRIV(tw)->accreted->BH_accreted_momentum[PI][k], remote->AccretedMomentum[k]);
    }

    TREEWALK_REDUCE(BHP(place).CountProgs, remote->BH_CountProgs);
}

static void
bh_update_from_accreted(int n, AccretedVariables * accreted, const double SeedBHDynMass)
{
    const int PI = P[n].PI;
    if(accreted->BH_accreted_BHMass[PI] > 0){
       BHP(n).Mass += accreted->BH_accreted_BHMass[PI];
    }
    if(accreted->BH_accreted_Mass[PI] > 0)
    {
        /* velocity feedback due to accretion; momentum conservation.
         * This does nothing with repositioning on.*/
        const MyFloat accmass = accreted->BH_accreted_Mass[PI];
        int k;
        /* Need to add the momentum from Mtrack as well*/
        for(k = 0; k < 3; k++)
            P[n].Vel[k] = (P[n].Vel[k] * P[n].Mass + accreted->BH_accreted_momentum[PI][k]) / (P[n].Mass + accmass);
        /* Add the mass to Mtrack if there is room*/
        if(SeedBHDynMass > 0 && BHP(n).Mtrack + accmass < SeedBHDynMass) {
            /* Still seed mass regime*/
            BHP(n).Mtrack += accmass;
        } else if(BHP(n).Mtrack < SeedBHDynMass) {
            /* Transitioning to regular BH */
            P[n].Mass = BHP(n).Mtrack + accmass;
            BHP(n).Mtrack = SeedBHDynMass;
        }
        else {
            /* Already regular BH, add accretion to regular mass*/
            P[n].Mass += accmass;
        }
    }
}

static void
blackhole_real_merger_postprocess(int n, TreeWalk * tw)
{
    bh_update_from_accreted(n, BH_GET_PRIV(tw)->accreted, BH_GET_PRIV(tw)->SeedBHDynMass);
}

int64_t
blackhole_mergers(int * ActiveBlackHoles, const int64_t NumActiveBlackHoles, DomainDecomp * ddecomp, struct kick_factor_data *kf, const double atime, const double SeedBHDynMass, const int MergeGravBound, AccretedVariables * accrete)
{
    /* Small tree containing only BHs */
    ForceTree tree[1] = {0};

    struct BHMergerPriv priv[1] = {0};
    priv->accreted = accrete;
    bh_alloc_accreted(priv->accreted);

    force_tree_rebuild_mask(tree, ddecomp, BHMASK, NULL);
    /* Symmetric tree walk needs hmax values*/
    force_tree_calc_moments(tree, ddecomp);
    walltime_measure("/BH/BuildMerger");

    priv->atime = atime;
    priv->kf = kf;
    priv->SeedBHDynMass = SeedBHDynMass;
    priv->MergeGravBound = MergeGravBound;
    /* Let's determine which BHs may be swallowed and calculate total feedback weights */
    priv->BH_SwallowID = (MyIDType *) mymalloc("BH_SwallowID", SlotsManager->info[5].size * sizeof(MyIDType));
    memset(priv->BH_SwallowID, 0, SlotsManager->info[5].size * sizeof(MyIDType));
    priv->mergercount = 0;
    /*************************************************************************/
    TreeWalk tw_merger[1] = {{0}};

    /* This treewalk marks all black holes which can be swallowed with a SwllowID of a potential swallower.
     * The treewalk is symmetric. A swallower needs to be active, the black holes must be within each other's
     * smoothing radius and optionally gravitationally bound. In case a black hole can be swallowed by multiple others,
     * we prefer the one with the largest ID. */
    tw_merger->ev_label = "BH_MERGER";
    tw_merger->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw_merger->ngbiter_type_elsize = sizeof(TreeWalkNgbIterBase);
    tw_merger->ngbiter = (TreeWalkNgbIterFunction) blackhole_merger_ngbiter;
    tw_merger->haswork = NULL;
    tw_merger->postprocess = NULL;
    tw_merger->preprocess = NULL;
    tw_merger->fill = (TreeWalkFillQueryFunction) blackhole_merger_copy;
    tw_merger->reduce = (TreeWalkReduceResultFunction) blackhole_merger_reduce;
    tw_merger->query_type_elsize = sizeof(TreeWalkQueryBHMerger);
    tw_merger->result_type_elsize = sizeof(TreeWalkResultBHMerger);
    tw_merger->tree = tree;
    tw_merger->priv = priv;

    /* Merge BHs here. Only BHs which are not themselves
     * being swallowed are eligible to swallow. If there are multiple BHs within a search radius,
     * the BH with the largest ID will be selected by the SwallowID search and will swallow all the
     * surrounding BHs.
     * Example 1:
     * We have BHs A,B,C, where A and B are close and B and C are close. B.ID < C.ID and A.ID < B.ID.
     * B will be swallowed by C. A will not be swallowed by B as B is itself being swallowed (A will
     * have a non-zero encounter).
     * Example 2:
     * We have BHs A,B,C, where A and C are both close to B. B.ID > C.ID and A.ID < B.ID.
     * In this case B will swallow C and A.
     * Example 3:
     * We have BHs A,B,C, where A and B are close and B and C are close. B.ID < C.ID and A.ID > B.ID.
     * In this case B will be swallowed by whichever of A and C has the larger ID.
    */
    treewalk_run(tw_merger, ActiveBlackHoles, NumActiveBlackHoles);

    TreeWalk tw_real_merger[1] = {{0}};
    tw_real_merger->ev_label = "BH_REAL_MERGER";
    tw_real_merger->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw_real_merger->ngbiter_type_elsize = sizeof(TreeWalkNgbIterBase);
    tw_real_merger->ngbiter = (TreeWalkNgbIterFunction) blackhole_real_merger_ngbiter;
    tw_real_merger->haswork = blackhole_real_merger_haswork;
    tw_real_merger->fill = (TreeWalkFillQueryFunction) blackhole_real_merger_copy;
    tw_real_merger->postprocess = blackhole_real_merger_postprocess;
    tw_real_merger->reduce = (TreeWalkReduceResultFunction) blackhole_real_merger_reduce;
    tw_real_merger->query_type_elsize = sizeof(TreeWalkQueryBHRealMerger);
    tw_real_merger->result_type_elsize = sizeof(TreeWalkResultBHRealMerger);
    tw_real_merger->tree = tree;
    tw_real_merger->priv = priv;
    tw_real_merger->repeatdisallowed = 1;

    /* If no BHs are expecting a merger, we don't need to do anything here.*/
    if(!MPIU_Any(priv->mergercount > 0, MPI_COMM_WORLD)) {
        myfree(priv->BH_SwallowID);
        force_tree_free(tree);
        return 0;
    }
    /* Ionization counters*/
    priv[0].N_BH_swallowed = ta_malloc("n_BH_swallowed", int64_t, omp_get_max_threads());
    memset(priv[0].N_BH_swallowed, 0, sizeof(int64_t) * omp_get_max_threads());

    treewalk_run(tw_real_merger, ActiveBlackHoles, NumActiveBlackHoles);

    myfree(priv->BH_SwallowID);
    int i;
    int64_t Ntot_BH_swallowed;
    int64_t N_BH_swallowed = 0;
    for(i = 0; i < omp_get_max_threads(); i++) {
        N_BH_swallowed += priv[0].N_BH_swallowed[i];
    }
    ta_free(priv[0].N_BH_swallowed);

    force_tree_free(tree);
    MPI_Reduce(&N_BH_swallowed, &Ntot_BH_swallowed, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
    return Ntot_BH_swallowed;
}


void
bh_alloc_accreted(AccretedVariables * accrete)
{
    accrete->BH_accreted_Mass = (MyFloat *) mymalloc("BH_accretedmass", SlotsManager->info[5].size * sizeof(MyFloat));
    accrete->BH_accreted_BHMass = (MyFloat *) mymalloc("BH_accreted_BHMass", SlotsManager->info[5].size * sizeof(MyFloat));
    accrete->BH_accreted_momentum = (MyFloat (*) [3]) mymalloc("BH_accretemom", 3* SlotsManager->info[5].size * sizeof(accrete->BH_accreted_momentum[0]));
}

void
bh_free_accreted(AccretedVariables * accrete)
{
    myfree(accrete->BH_accreted_momentum);
    myfree(accrete->BH_accreted_BHMass);
    myfree(accrete->BH_accreted_Mass);
}
