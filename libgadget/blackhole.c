#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <omp.h>

#include "physconst.h"
#include "cooling.h"
#include "gravity.h"
#include "densitykernel.h"
#include "treewalk.h"
#include "slotsmanager.h"
#include "blackhole.h"
#include "timestep.h"
#include "hydra.h"
#include "density.h"
#include "sfr_eff.h"
#include "winds.h"
#include "walltime.h"
#include "bhinfo.h"
#include "bhdynfric.h"

/*! \file blackhole.c
 *  \brief routines for gas accretion onto black holes, and black hole mergers
 */

struct BlackholeParams
{
    double BlackHoleAccretionFactor;	/*!< Fraction of BH bondi accretion rate */
    double BlackHoleFeedbackFactor;	/*!< Fraction of the black luminosity feed into thermal feedback */
    enum BlackHoleFeedbackMethod BlackHoleFeedbackMethod;	/*!< method of the feedback*/
    double BlackHoleEddingtonFactor;	/*! Factor above Eddington */
    int BlackHoleRepositionEnabled; /* If true, enable repositioning the BH to the potential minimum*/

    int BlackHoleKineticOn; /*If 1, perform AGN kinetic feedback when the Eddington accretion rate is low */
    double BHKE_EddingtonThrFactor; /*Threshold of the Eddington rate for the kinetic feedback*/
    double BHKE_EddingtonMFactor; /* Factor for mbh-dependent Eddington threshold */
    double BHKE_EddingtonMPivot; /* Pivot MBH for mbh-dependent Eddington threshold */
    double BHKE_EddingtonMIndex; /* Powlaw index for mbh-dependent Eddington threshold */
    double BHKE_EffRhoFactor; /* Minimum kinetic feedback efficiency factor scales with BH density*/
    double BHKE_EffCap; /* Cap of the kinetic feedback efficiency factor */
    double BHKE_InjEnergyThr; /*Minimum injection of KineticFeedbackEnergy, controls the burstiness of kinetic feedback*/
    double BHKE_SfrCritOverDensity; /*for KE efficiency calculation, borrow from sfr.params */
    /**********************************************************************/
    int MergeGravBound; /*if 1, apply gravitational bound criteria for BH mergers */
    int BH_DRAG; /*Hydro drag force*/

    double SeedBHDynMass; /* The initial dynamic mass of BH particle */

    double SeedBlackHoleMass;	/*!< (minimum) Seed black hole mass */
    double MaxSeedBlackHoleMass; /* Maximum black hole seed mass*/
    double SeedBlackHoleMassIndex; /* Power law index for BH seed mass*/
    /************************************************************************/
} blackhole_params;

int
BHGetRepositionEnabled(void)
{
    return blackhole_params.BlackHoleRepositionEnabled;
}

typedef struct {
    TreeWalkQueryBase base;
    MyFloat Density;
    MyFloat Hsml;
    MyFloat Mass;
    MyFloat BH_Mass;
    MyFloat Vel[3];
    MyFloat Accel[3];
    MyIDType ID;
    MyFloat Mtrack;
} TreeWalkQueryBHAccretion;

typedef struct {
    TreeWalkResultBase base;
    MyFloat BH_MinPotPos[3];
    MyFloat BH_MinPotVel[3];
    MyFloat BH_MinPot;
    int BH_minTimeBin;
    int encounter;
    MyFloat FeedbackWeightSum;

    MyFloat SmoothedEntropy;
    MyFloat GasVel[3];
    /* used for AGN kinetic feedback */
    MyFloat V2sumDM;
    MyFloat V1sumDM[3];
    MyFloat NumDM;
    MyFloat MgasEnc;
} TreeWalkResultBHAccretion;

typedef struct {
    TreeWalkNgbIterBase base;
    DensityKernel accretion_kernel;
    DensityKernel feedback_kernel;
} TreeWalkNgbIterBHAccretion;


/*****************************************************************************/

/*Set the parameters of the BH module*/
void set_blackhole_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        blackhole_params.BlackHoleAccretionFactor = param_get_double(ps, "BlackHoleAccretionFactor");
        blackhole_params.BlackHoleEddingtonFactor = param_get_double(ps, "BlackHoleEddingtonFactor");
        blackhole_params.BlackHoleFeedbackFactor = param_get_double(ps, "BlackHoleFeedbackFactor");

        blackhole_params.BlackHoleFeedbackMethod = (enum BlackHoleFeedbackMethod) param_get_enum(ps, "BlackHoleFeedbackMethod");
        blackhole_params.BlackHoleRepositionEnabled = param_get_int(ps, "BlackHoleRepositionEnabled");

        blackhole_params.BlackHoleKineticOn = param_get_int(ps,"BlackHoleKineticOn");
        blackhole_params.BHKE_EddingtonThrFactor = param_get_double(ps, "BHKE_EddingtonThrFactor");
        blackhole_params.BHKE_EddingtonMFactor = param_get_double(ps, "BHKE_EddingtonMFactor");
        blackhole_params.BHKE_EddingtonMPivot = param_get_double(ps, "BHKE_EddingtonMPivot");
        blackhole_params.BHKE_EddingtonMIndex = param_get_double(ps, "BHKE_EddingtonMIndex");
        blackhole_params.BHKE_EffRhoFactor = param_get_double(ps, "BHKE_EffRhoFactor");
        blackhole_params.BHKE_EffCap = param_get_double(ps, "BHKE_EffCap");
        blackhole_params.BHKE_InjEnergyThr = param_get_double(ps, "BHKE_InjEnergyThr");
        blackhole_params.BHKE_SfrCritOverDensity = param_get_double(ps, "CritOverDensity");
        /***********************************************************************************/
        blackhole_params.BH_DRAG = param_get_int(ps, "BH_DRAG");
        blackhole_params.MergeGravBound = param_get_int(ps, "MergeGravBound");
        blackhole_params.SeedBHDynMass = param_get_double(ps,"SeedBHDynMass");


        blackhole_params.SeedBlackHoleMass = param_get_double(ps, "SeedBlackHoleMass");
        blackhole_params.MaxSeedBlackHoleMass = param_get_double(ps,"MaxSeedBlackHoleMass");
        blackhole_params.SeedBlackHoleMassIndex = param_get_double(ps,"SeedBlackHoleMassIndex");
        /***********************************************************************************/
    }
    MPI_Bcast(&blackhole_params, sizeof(struct BlackholeParams), MPI_BYTE, 0, MPI_COMM_WORLD);

    set_blackhole_dynfric_params(ps);
}

/* accretion routines */
static void
blackhole_accretion_postprocess(int n, TreeWalk * tw);

static void
blackhole_accretion_reduce(int place, TreeWalkResultBHAccretion * remote, enum TreeWalkReduceMode mode, TreeWalk * tw);

static void
blackhole_accretion_copy(int place, TreeWalkQueryBHAccretion * I, TreeWalk * tw);

/* Initializes the minimum potentials*/
static void
blackhole_accretion_preprocess(int n, TreeWalk * tw);

static void
blackhole_accretion_ngbiter(TreeWalkQueryBHAccretion * I,
        TreeWalkResultBHAccretion * O,
        TreeWalkNgbIterBHAccretion * iter,
        LocalTreeWalk * lv);

/* Do the black hole feedback tree walk. Tree needs to have gas and BH.*/
static void
blackhole_feedback(int * ActiveBlackHoles, int64_t NumActiveBlackHoles, ForceTree * tree, struct BHPriv * priv);

/*************************************************************************************/

#define BHPOTVALUEINIT 1.0e29

static double blackhole_soundspeed(double entropy, double rho, const double atime) {
    /* rho is comoving !*/
    if(rho <= 0)
        return 0;
    double cs = sqrt(GAMMA * entropy * pow(rho, GAMMA_MINUS1));

    cs *= pow(atime, -1.5 * GAMMA_MINUS1);

    return cs;
}

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

/*******************************************************************/
static int
blackhole_haswork(int n, TreeWalk * tw){
    /*Black hole not being swallowed*/
    return (P[n].Type == 5) && (!P[n].Swallowed);
}

/* Build a list of active black holes, done once and reused for all the later treewalks.*/
int
blackholes_active(const ActiveParticles * act, int ** ActiveBlackHoles, int64_t * NumActiveBlackHoles)
{
    TreeWalk tw_bh[1] = {{0}};
    tw_bh->haswork = blackhole_haswork;

    /* Build the queue once, since it is really 'all black holes' and similar for all treewalks.*/
    treewalk_build_queue(tw_bh, act->ActiveParticle, act->NumActiveParticle, 0);
    /* If this queue is empty, nothing to do.*/
    int64_t totbh;
    MPI_Allreduce(&tw_bh->WorkSetSize, &totbh, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

    /* Now we have a BH queue and we can re-use it. Create a new variable so that
     * treewalk_run does not mess with the pointer. */
    /* Move the working set high so we can keep the tree we build after making the list and still free this active set.*/
    *NumActiveBlackHoles = tw_bh->WorkSetSize;
    if(totbh > 0) {
        *ActiveBlackHoles = mymalloc2("activeBH", tw_bh->WorkSetSize * sizeof(int));
        memcpy(*ActiveBlackHoles, tw_bh->WorkSet, tw_bh->WorkSetSize * sizeof(int));
    }
    myfree(tw_bh->WorkSet);
    return totbh;
}

void
blackhole(const ActiveParticles * act, double atime, Cosmology * CP, ForceTree * tree, DomainDecomp * ddecomp, DriftKickTimes * times, const struct UnitSystem units, FILE * FdBlackHoles, FILE * FdBlackholeDetails)
{
    /* Do nothing if no black holes*/
    int64_t totbh;
    MPI_Allreduce(&SlotsManager->info[5].size, &totbh, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    if(totbh == 0)
        return;

    walltime_measure("/Misc");
    /* Build the queue once, since it is really 'all black holes' and similar for all treewalks.*/
    int * ActiveBlackHoles = NULL;
    int64_t NumActiveBlackHoles = 0;
    totbh = blackholes_active(act, &ActiveBlackHoles, &NumActiveBlackHoles);
    if(totbh == 0) {
        return;
    }

    /* Types used in treewalks:
     * dynamical friction uses: stars, DM if BH_DynFrictionMethod > 1 gas if BH_DynFrictionMethod  == 3.
     * accretion uses: all types but ONLY for repositioning potential minimum. Otherwise gas + black holes. gas + stars + BH is probably fine.
     * gas + BH probably not enough if gas is sparse in halo.
     * feedback uses: gas + black holes.
     * The DM in dynamic friction and accretion doesn't really do anything, so could perhaps be removed from the treebuild later.
     * However, we would still need a tree with gas + DM in the wind code.
     */
    if(!tree->tree_allocated_flag)
    {
        message(0, "Building tree in blackhole\n");
        int treemask = blackhole_dynfric_treemask();
        treemask += GASMASK | BHMASK;
        force_tree_rebuild_mask(tree, ddecomp, treemask, NULL);
        walltime_measure("/BH/Build");
    }

    struct kick_factor_data kf;
    init_kick_factor_data(&kf, times, CP);

    /*************************************************************************/
    /*  Dynamical Friction Treewalk */
    /*************************************************************************/
    struct BHDynFricPriv dynpriv[1] = {0};
    dynpriv->atime = atime;
    dynpriv->CP = CP;
    dynpriv->kf = &kf;
    blackhole_dynfric(ActiveBlackHoles, NumActiveBlackHoles, tree, dynpriv);
    walltime_measure("/BH/DynFric");

    struct BHPriv priv[1] = {0};
    priv->units = units;

    /*************************************************************************/
    priv->atime = atime;
    priv->a3inv = 1./(atime * atime * atime);
    priv->hubble = hubble_function(CP, atime);
    priv->CP = CP;
    priv->kf = &kf;

    /* Let's determine which particles may be swallowed and calculate total feedback weights */
    priv->SPH_SwallowID = (MyIDType *) mymalloc("SPH_SwallowID", SlotsManager->info[0].size * sizeof(MyIDType));
    memset(priv->SPH_SwallowID, 0, SlotsManager->info[0].size * sizeof(MyIDType));

    /* Computed in accretion, used in feedback*/
    priv->BH_FeedbackWeightSum = (MyFloat *) mymalloc("BH_FeedbackWeightSum", SlotsManager->info[5].size * sizeof(MyFloat));

    /* These are initialized in preprocess and used to reposition the BH in postprocess*/
    priv->MinPot = (MyFloat *) mymalloc("BH_MinPot", SlotsManager->info[5].size * sizeof(MyFloat));

    /* Local to this treewalk*/
    priv->BH_Entropy = (MyFloat *) mymalloc("BH_Entropy", SlotsManager->info[5].size * sizeof(MyFloat));
    priv->BH_SurroundingGasVel = (MyFloat (*) [3]) mymalloc("BH_SurroundVel", 3* SlotsManager->info[5].size * sizeof(priv->BH_SurroundingGasVel[0]));

    /* For AGN kinetic feedback */
    priv->NumDM = mymalloc("NumDM", SlotsManager->info[5].size * sizeof(MyFloat));
    priv->MgasEnc = mymalloc("MgasEnc", SlotsManager->info[5].size * sizeof(MyFloat));
    /* mark the state of AGN kinetic feedback */
    priv->KEflag = mymalloc("KEflag", SlotsManager->info[5].size * sizeof(int));

    walltime_measure("/BH/Init");

    /*************************************************************************/
    TreeWalk tw_accretion[1] = {{0}};

    tw_accretion->ev_label = "BH_ACCRETION";
    tw_accretion->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw_accretion->ngbiter_type_elsize = sizeof(TreeWalkNgbIterBHAccretion);
    tw_accretion->ngbiter = (TreeWalkNgbIterFunction) blackhole_accretion_ngbiter;
    tw_accretion->haswork = NULL;
    tw_accretion->postprocess = (TreeWalkProcessFunction) blackhole_accretion_postprocess;
    tw_accretion->preprocess = (TreeWalkProcessFunction) blackhole_accretion_preprocess;
    tw_accretion->fill = (TreeWalkFillQueryFunction) blackhole_accretion_copy;
    tw_accretion->reduce = (TreeWalkReduceResultFunction) blackhole_accretion_reduce;
    tw_accretion->query_type_elsize = sizeof(TreeWalkQueryBHAccretion);
    tw_accretion->result_type_elsize = sizeof(TreeWalkResultBHAccretion);
    tw_accretion->tree = tree;
    tw_accretion->priv = priv;

    treewalk_run(tw_accretion, ActiveBlackHoles, NumActiveBlackHoles);

    /*************************************************************************/

    walltime_measure("/BH/Accretion");

    priv->BH_accreted_Mass = (MyFloat *) mymalloc("BH_accretedmass", SlotsManager->info[5].size * sizeof(MyFloat));
    priv->BH_accreted_BHMass = (MyFloat *) mymalloc("BH_accreted_BHMass", SlotsManager->info[5].size * sizeof(MyFloat));
    priv->BH_accreted_Mtrack = (MyFloat *) mymalloc("BH_accreted_Mtrack", SlotsManager->info[5].size * sizeof(MyFloat));
    priv->BH_accreted_momentum = (MyFloat (*) [3]) mymalloc("BH_accretemom", 3* SlotsManager->info[5].size * sizeof(priv->BH_accreted_momentum[0]));

    /* Now do the swallowing of particles and dump feedback energy */
    blackhole_feedback(ActiveBlackHoles, NumActiveBlackHoles, tree, priv);

    walltime_measure("/BH/Feedback");

    if(FdBlackholeDetails){
        collect_BH_info(ActiveBlackHoles, NumActiveBlackHoles, priv, dynpriv, FdBlackholeDetails);
    }

    myfree(priv->BH_accreted_momentum);
    myfree(priv->BH_accreted_Mtrack);
    myfree(priv->BH_accreted_BHMass);
    myfree(priv->BH_accreted_Mass);

    /*****************************************************************/
    myfree(priv->KEflag);
    myfree(priv->MgasEnc);
    myfree(priv->NumDM);

    myfree(priv->BH_SurroundingGasVel);
    myfree(priv->BH_Entropy);
    myfree(priv->MinPot);

    myfree(priv->BH_FeedbackWeightSum);
    myfree(priv->SPH_SwallowID);

    blackhole_dynpriv_free(dynpriv);

    myfree(ActiveBlackHoles);

    write_blackhole_txt(FdBlackHoles, units, atime);
    walltime_measure("/BH/Info");
}

static void
blackhole_accretion_postprocess(int i, TreeWalk * tw)
{
    int k;
    int PI = P[i].PI;
    double mdot = 0;    /* if no accretion model is enabled, we have mdot=0 */
    /* Note: we take here a radiative efficiency of 0.1 for Eddington accretion */
    const double meddington = (4 * M_PI * GRAVITY * LIGHTCGS * PROTONMASS / (0.1 * LIGHTCGS * LIGHTCGS * THOMPSON)) * BHP(i).Mass
            * BH_GET_PRIV(tw)->units.UnitTime_in_s / BH_GET_PRIV(tw)->CP->HubbleParam;

    if(BHP(i).Density > 0)
    {
        BH_GET_PRIV(tw)->BH_Entropy[PI] /= BHP(i).Density;
        for(k = 0; k < 3; k++)
            BH_GET_PRIV(tw)->BH_SurroundingGasVel[PI][k] /= BHP(i).Density;

        double bhvel = 0;
        for(k = 0; k < 3; k++)
            bhvel += pow(P[i].Vel[k] - BH_GET_PRIV(tw)->BH_SurroundingGasVel[PI][k], 2);

        bhvel = sqrt(bhvel);
        bhvel /= BH_GET_PRIV(tw)->atime;
        double rho = BHP(i).Density;
        double rho_proper = rho * BH_GET_PRIV(tw)->a3inv;

        double soundspeed = blackhole_soundspeed(BH_GET_PRIV(tw)->BH_Entropy[PI], rho, BH_GET_PRIV(tw)->atime);

        double norm = pow((pow(soundspeed, 2) + pow(bhvel, 2)), 1.5);

        if(norm > 0)
            mdot = 4. * M_PI * blackhole_params.BlackHoleAccretionFactor * BH_GET_PRIV(tw)->CP->GravInternal * BH_GET_PRIV(tw)->CP->GravInternal *
                BHP(i).Mass * BHP(i).Mass * rho_proper / norm;
    }

    if(blackhole_params.BlackHoleEddingtonFactor > 0.0 &&
        mdot > blackhole_params.BlackHoleEddingtonFactor * meddington) {
        mdot = blackhole_params.BlackHoleEddingtonFactor * meddington;
    }
    BHP(i).Mdot = mdot;

    double dtime = get_dloga_for_bin(P[i].TimeBinHydro, P[i].Ti_drift) / BH_GET_PRIV(tw)->hubble;

    BHP(i).Mass += BHP(i).Mdot * dtime;

    /*************************************************************************/

    if(blackhole_params.BH_DRAG > 0){
        /* a_BH = (v_gas - v_BH) Mdot/M_BH                                   */
        /* motivated by BH gaining momentum from the accreted gas            */
        /*c.f.section 3.2,in http://www.tapir.caltech.edu/~phopkins/public/notes_blackholes.pdf */
        double fac = 0;
        if (blackhole_params.BH_DRAG == 1) fac = BHP(i).Mdot/P[i].Mass;
        if (blackhole_params.BH_DRAG == 2) fac = blackhole_params.BlackHoleEddingtonFactor * meddington/BHP(i).Mass;
        fac *= BH_GET_PRIV(tw)->atime; /* dv = acc * kick_fac = acc * a^{-1}dt, therefore acc = a*dv/dt  */
        for(k = 0; k < 3; k++) {
            BHP(i).DragAccel[k] = -(P[i].Vel[k] - BH_GET_PRIV(tw)->BH_SurroundingGasVel[PI][k])*fac;
        }
    }
    else{
        for(k = 0; k < 3; k++){
            BHP(i).DragAccel[k] = 0;
        }
    }
    /*************************************************************************/

    if(blackhole_params.BlackHoleKineticOn == 1){
        /* accumulate kenetic feedback energy by dE = epsilon x mdot x c^2 */
        /* epsilon = Min(rho_BH/(BHKE_EffRhoFactor*rho_sfr),BHKE_EffCap)   */
        /* KE is released when exceeding injection energy threshold        */
        BH_GET_PRIV(tw)->KEflag[PI] = 0;
        double Edd_ratio = BHP(i).Mdot/meddington;
        double lam_thresh = blackhole_params.BHKE_EddingtonThrFactor;
        double x = blackhole_params.BHKE_EddingtonMFactor * pow(BHP(i).Mass/blackhole_params.BHKE_EddingtonMPivot, blackhole_params.BHKE_EddingtonMIndex);
        if (lam_thresh > x)
            lam_thresh = x;
        if (Edd_ratio < lam_thresh){
            /* mark this timestep is accumulating KE feedback energy */
            BH_GET_PRIV(tw)->KEflag[PI] = 1;
            const double rho_crit_baryon = BH_GET_PRIV(tw)->CP->OmegaBaryon * 3 * pow(BH_GET_PRIV(tw)->CP->Hubble, 2) / (8 * M_PI * BH_GET_PRIV(tw)->CP->GravInternal);
            const double rho_sfr = blackhole_params.BHKE_SfrCritOverDensity * rho_crit_baryon;
            double epsilon = (BHP(i).Density/rho_sfr)/blackhole_params.BHKE_EffRhoFactor;
            if (epsilon > blackhole_params.BHKE_EffCap){
                epsilon = blackhole_params.BHKE_EffCap;
            }

            BHP(i).KineticFdbkEnergy += epsilon * (BHP(i).Mdot * dtime * pow(LIGHTCGS / BH_GET_PRIV(tw)->units.UnitVelocity_in_cm_per_s, 2));
        }
        /* decide whether to release KineticFdbkEnergy*/
        double KE_thresh = 0.5 * BHP(i).VDisp * BHP(i).VDisp * BH_GET_PRIV(tw)->MgasEnc[PI];
        KE_thresh *= blackhole_params.BHKE_InjEnergyThr;

        if (BHP(i).VDisp > 0 && BHP(i).KineticFdbkEnergy > KE_thresh){
            /* mark KineticFdbkEnergy is ready to be released in the feedback treewalk */
            BH_GET_PRIV(tw)->KEflag[PI] = 2;
        }
    }
}

static void
blackhole_accretion_preprocess(int n, TreeWalk * tw)
{
    int j;
    /* Note that the potential is only updated when it is from all particles.
     * In particular this means that it is not updated for hierarchical gravity
     * when the number of active particles is less than the total number of particles
     * (because then the tree does not contain all forces). */
    BH_GET_PRIV(tw)->MinPot[P[n].PI] = P[n].Potential;

    for(j = 0; j < 3; j++) {
        BHP(n).MinPotPos[j] = P[n].Pos[j];
    }

}

static void
blackhole_accretion_ngbiter(TreeWalkQueryBHAccretion * I,
        TreeWalkResultBHAccretion * O,
        TreeWalkNgbIterBHAccretion * iter,
        LocalTreeWalk * lv)
{

    if(iter->base.other == -1) {
        O->BH_minTimeBin = TIMEBINS;
        O->encounter = 0;

        O->BH_MinPot = BHPOTVALUEINIT;

        int d;
        for(d = 0; d < 3; d++) {
            O->BH_MinPotPos[d] = I->base.Pos[d];
        }
        iter->base.mask = GASMASK + STARMASK + BHMASK;
        iter->base.Hsml = I->Hsml;
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;

        density_kernel_init(&iter->accretion_kernel, I->Hsml, GetDensityKernelType());
        density_kernel_init(&iter->feedback_kernel, I->Hsml, GetDensityKernelType());
        return;
    }

    int other = iter->base.other;
    double r = iter->base.r;
    double r2 = iter->base.r2;

    if(P[other].Mass < 0) return;

    /* For accretion stability, set the BH timestep to the smallest gas timestep.*/
    if(P[other].Type == 0) {
        if (O->BH_minTimeBin > P[other].TimeBinHydro)
            O->BH_minTimeBin = P[other].TimeBinHydro;
    }

     /* BH does not accrete wind */
    if(winds_is_particle_decoupled(other)) return;

    /* Find the black hole potential minimum. */
    if(r2 < iter->accretion_kernel.HH)
    {
        if(P[other].Potential < O->BH_MinPot)
        {
            int d;
            O->BH_MinPot = P[other].Potential;
            for(d = 0; d < 3; d++) {
                O->BH_MinPotPos[d] = P[other].Pos[d];
                O->BH_MinPotVel[d] = P[other].Vel[d];
            }
        }
    }

    /* Accretion / merger doesn't do self interaction */
    if(P[other].ID == I->ID) return;

    /* we have a black hole merger. Now we use 2 times GravitationalSoftening as merging criteria, previously we used the SPH smoothing length. */
    if(P[other].Type == 5 && r < (2*FORCE_SOFTENING(0,1)/2.8))
    {
        O->encounter = 1; // mark the event when two BHs encounter each other

        int flag = 0; // the flag for BH merge

        if(blackhole_params.BlackHoleRepositionEnabled == 1) // directly merge if reposition is enabled
            flag = 1;
        if(blackhole_params.MergeGravBound == 0)
            flag = 1;
        /* apply Grav Bound check only when Reposition is disabled, otherwise BHs would be repositioned to the same location but not merge */
        if(blackhole_params.MergeGravBound == 1 && blackhole_params.BlackHoleRepositionEnabled == 0){

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

        /* Mark the BH via SwallowID.*/
        if(flag == 1)
        {
            MyIDType readid, newswallowid;

            #pragma omp atomic read
            readid = (BHP(other).SwallowID);

            /* Here we mark the black hole as "ready to be swallowed" using the SwallowID.
             * The actual swallowing is done in the feedback treewalk by setting Swallowed = 1
             * and merging the masses.*/
            do {
                /* Generate the new ID from the old*/
                if(readid != (MyIDType) -1 && readid < I->ID ) {
                    /* Already marked, prefer to be swallowed by a bigger ID */
                    newswallowid = I->ID;
                } else if(readid == (MyIDType) -1 && P[other].ID < I->ID) {
                    /* Unmarked, the BH with bigger ID swallows */
                    newswallowid = I->ID;
                }
                else
                    break;
            /* Swap in the new id only if the old one hasn't changed:
             * in principle an extension, but supported on at least clang >= 9, gcc >= 5 and icc >= 18.*/
            } while(!__atomic_compare_exchange_n(&(BHP(other).SwallowID), &readid, newswallowid, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
        }
    }


    if(P[other].Type == 0) {
        if(r2 < iter->accretion_kernel.HH) {
            double u = r * iter->accretion_kernel.Hinv;
            double wk = density_kernel_wk(&iter->accretion_kernel, u);
            float mass_j = P[other].Mass;

            O->SmoothedEntropy += (mass_j * wk * SPHP(other).Entropy);
            MyFloat VelPred[3];
            SPH_VelPred(other, VelPred, BH_GET_PRIV(lv->tw)->kf);
            O->GasVel[0] += (mass_j * wk * VelPred[0]);
            O->GasVel[1] += (mass_j * wk * VelPred[1]);
            O->GasVel[2] += (mass_j * wk * VelPred[2]);
            /* here we have a gas particle; check for swallowing */

            /* compute accretion probability */
            double p = 0;

            MyFloat BHPartMass = I->Mass;
            /* If SeedBHDynMass is larger than gas paricle mass, we use Mtrack to do the gas accretion
             * when BHP.Mass < SeedBHDynMass. Mtrack is initialized as gas particle mass and is capped
             * at SeedBHDynMass. Mtrack traces the BHP.Mass by stochastically swallowing gas and
             * therefore ensures mass conservation.*/
            if(blackhole_params.SeedBHDynMass > 0 && I->Mtrack < blackhole_params.SeedBHDynMass)
                BHPartMass = I->Mtrack;

            /* This is an averaged Mdot, because Mdot increases BH_Mass but not Mass.
             * So if the total accretion is significantly above the dynamical mass,
             * a particle is swallowed. */
            if((I->BH_Mass - BHPartMass) > 0 && I->Density > 0)
                p = (I->BH_Mass - BHPartMass) * wk / I->Density;

            /* compute random number, uniform in [0,1] */
            const double w = get_random_number(P[other].ID);
            if(w < p)
            {
                MyIDType * SPH_SwallowID = BH_GET_PRIV(lv->tw)->SPH_SwallowID;
                MyIDType readid, newswallowid;
                #pragma omp atomic read
                readid = SPH_SwallowID[P[other].PI];
                do {
                    /* Already marked, prefer to be swallowed by a bigger ID.
                     * Not marked, the SwallowID is 0 */
                    if(readid < I->ID + 1) {
                        newswallowid = I->ID + 1;
                    }
                    else
                        break;
                    /* Swap in the new id only if the old one hasn't changed*/
                } while(!__atomic_compare_exchange_n(&SPH_SwallowID[P[other].PI], &readid, newswallowid, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
            }
        }

        if(r2 < iter->feedback_kernel.HH) {
            /* update the feedback weighting */
            double mass_j;
            if(HAS(blackhole_params.BlackHoleFeedbackMethod, BH_FEEDBACK_OPTTHIN)) {
                double redshift = 1./BH_GET_PRIV(lv->tw)->atime - 1;
                double nh0 = get_neutral_fraction_sfreff(redshift, BH_GET_PRIV(lv->tw)->hubble, &P[other], &SPHP(other));
                if(r2 > 0)
                    O->FeedbackWeightSum += (P[other].Mass * nh0) / r2;
            } else {
                if(HAS(blackhole_params.BlackHoleFeedbackMethod, BH_FEEDBACK_MASS)) {
                    mass_j = P[other].Mass;
                } else {
                    mass_j = P[other].Hsml * P[other].Hsml * P[other].Hsml;
                }
                if(HAS(blackhole_params.BlackHoleFeedbackMethod, BH_FEEDBACK_SPLINE)) {
                    double u = r * iter->feedback_kernel.Hinv;
                    O->FeedbackWeightSum += (mass_j *
                          density_kernel_wk(&iter->feedback_kernel, u)
                           );
                } else {
                    O->FeedbackWeightSum += (mass_j);
                }
            }
        }
    }

    /* collect info for sigmaDM and Menc for kinetic feedback */
    if(blackhole_params.BlackHoleKineticOn == 1 &&
        P[other].Type == 0 &&
        r2 < iter->feedback_kernel.HH ){
            O->MgasEnc += P[other].Mass;
        }
}

static void
blackhole_accretion_reduce(int place, TreeWalkResultBHAccretion * remote, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    int k;
    MyFloat * MinPot = BH_GET_PRIV(tw)->MinPot;

    int PI = P[place].PI;
    if(MinPot[PI] > remote->BH_MinPot)
    {
        BHP(place).JumpToMinPot = blackhole_params.BlackHoleRepositionEnabled;
        MinPot[PI] = remote->BH_MinPot;
        for(k = 0; k < 3; k++) {
            /* Movement occurs in drift.c */
            BHP(place).MinPotPos[k] = remote->BH_MinPotPos[k];
            BHP(place).MinPotVel[k] = remote->BH_MinPotVel[k];
        }
    }

    /* Set encounter to true if it is true on any remote*/
    if (mode == 0 || BHP(place).encounter < remote->encounter) {
        BHP(place).encounter = remote->encounter;
    }
    if (mode == 0 || BHP(place).minTimeBin > remote->BH_minTimeBin) {
        BHP(place).minTimeBin = remote->BH_minTimeBin;
    }

    TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_FeedbackWeightSum[PI], remote->FeedbackWeightSum);
    TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_Entropy[PI], remote->SmoothedEntropy);
    for (k = 0; k < 3; k++){
        TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_SurroundingGasVel[PI][k], remote->GasVel[k]);
    }
    TREEWALK_REDUCE(BH_GET_PRIV(tw)->NumDM[PI], remote->NumDM);
    TREEWALK_REDUCE(BH_GET_PRIV(tw)->MgasEnc[PI], remote->MgasEnc);
}

static void
blackhole_accretion_copy(int place, TreeWalkQueryBHAccretion * I, TreeWalk * tw)
{
    int k;
    for(k = 0; k < 3; k++)
    {
        I->Vel[k] = P[place].Vel[k];
        I->Accel[k] = P[place].FullTreeGravAccel[k] + P[place].GravPM[k] + BHP(place).DFAccel[k];
    }
    I->Hsml = P[place].Hsml;
    I->Mass = P[place].Mass;
    I->BH_Mass = BHP(place).Mass;
    I->Density = BHP(place).Density;
    I->ID = P[place].ID;
    I->Mtrack = BHP(place).Mtrack;
}

typedef struct {
    TreeWalkQueryBase base;
    MyFloat Hsml;
    MyFloat Mtrack;
    MyFloat BH_Mass;
    MyIDType ID;
    MyFloat Density;
    MyFloat FeedbackEnergy;
    MyFloat FeedbackWeightSum;
    MyFloat KEFeedbackEnergy;
    int FdbkChannel; /* 0 thermal, 1 kinetic */
} TreeWalkQueryBHFeedback;

typedef struct {
    TreeWalkResultBase base;
    MyFloat Mass; /* the accreted Mdyn */
    MyFloat AccretedMomentum[3];
    MyFloat BH_Mass;
    int BH_CountProgs;
    MyFloat acMtrack; /* the accreted Mtrack */
    int alignment; /* Ensure alignment*/
} TreeWalkResultBHFeedback;

typedef struct {
    TreeWalkNgbIterBase base;
    DensityKernel feedback_kernel;
} TreeWalkNgbIterBHFeedback;

/* Adds the injected black hole energy to an internal energy and caps it at a maximum temperature*/
static double
add_injected_BH_energy(double unew, double injected_BH_energy, double mass, double uu_in_cgs)
{
    unew += injected_BH_energy / mass;
    const double u_to_temp_fac = (4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC))) * PROTONMASS / BOLTZMANN * GAMMA_MINUS1 * uu_in_cgs;
    /* Cap temperature*/
    if(unew > 5.0e8 / u_to_temp_fac)
        unew = 5.0e8 / u_to_temp_fac;

    return unew;
}

static int
get_random_dir(int i, double dir[3])
{
    double theta = acos(2 * get_random_number(P[i].ID + 3) - 1);
    double phi = 2 * M_PI * get_random_number(P[i].ID + 4);

    dir[0] = sin(theta) * cos(phi);
    dir[1] = sin(theta) * sin(phi);
    dir[2] = cos(theta);
    return 0;
}

/**
 * perform blackhole swallow / merger;
 */
static void
blackhole_feedback_ngbiter(TreeWalkQueryBHFeedback * I,
        TreeWalkResultBHFeedback * O,
        TreeWalkNgbIterBHFeedback * iter,
        LocalTreeWalk * lv)
{

    if(iter->base.other == -1) {

        iter->base.mask = GASMASK + BHMASK;
        iter->base.Hsml = I->Hsml;
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        density_kernel_init(&iter->feedback_kernel, I->Hsml, GetDensityKernelType());
        return;
    }

    int other = iter->base.other;
    double r2 = iter->base.r2;
    double r = iter->base.r;
    /* Exclude self interaction */

    if(P[other].ID == I->ID) return;

     /* BH does not accrete wind */
    if(winds_is_particle_decoupled(other))
        return;


     /* we have a black hole merger! */
    if(P[other].Type == 5 && BHP(other).SwallowID != (MyIDType) -1)
    {
        if(BHP(other).SwallowID != I->ID) return;

        /* Swallow the particle*/
        /* A note on Swallowed vs SwallowID: black hole particles which have been completely swallowed
         * (ie, their mass has been added to another particle) have Swallowed = 1.
         * These particles are ignored in future tree walks. We set Swallowed here so that this process is atomic:
         * the total mass in the tree is always conserved.
         *
         * We also set SwallowID != -1 in the accretion treewalk. This marks the black hole as ready to be swallowed
         * by something. It is actually swallowed only by the nearby black hole with the largest ID. In rare cases
         * it may happen that the swallower is itself swallowed before swallowing the marked black hole. However,
         * in practice the new swallower should also take the marked black hole next timestep.
         */

        BHP(other).SwallowTime = BH_GET_PRIV(lv->tw)->atime;
        P[other].Swallowed = 1;
        /* Set encounter to zero when we merge*/
        BHP(other).encounter = 0;
        O->BH_CountProgs += BHP(other).CountProgs;
        O->BH_Mass += (BHP(other).Mass);

        if (blackhole_params.SeedBHDynMass>0 && I->Mtrack>0){
        /* Make sure that the final dynamic mass (I->Mass + O->Mass) = MAX(SeedDynMass, total_gas_accreted),
           I->Mtrack only need to be updated when I->Mtrack < SeedBHDynMass, */
            if(I->Mtrack < blackhole_params.SeedBHDynMass && BHP(other).Mtrack < blackhole_params.SeedBHDynMass){
            /* I->Mass = SeedBHDynMass, total_gas_accreted = I->Mtrack + BHP(other).Mtrack */
                O->acMtrack += BHP(other).Mtrack;
                double delta_m = I->Mtrack + BHP(other).Mtrack - blackhole_params.SeedBHDynMass;
                O->Mass += DMAX(0,delta_m);
            }
            if(I->Mtrack >= blackhole_params.SeedBHDynMass && BHP(other).Mtrack < blackhole_params.SeedBHDynMass){
            /* I->Mass = gas_accreted, total_gas_accreted = I->Mass + BHP(other).Mtrack */
                O->Mass += BHP(other).Mtrack;
            }
            if(I->Mtrack < blackhole_params.SeedBHDynMass && BHP(other).Mtrack >= blackhole_params.SeedBHDynMass){
            /* I->Mass = SeedBHDynMass, P[other].Mass = gas_accreted,
               total_gas_accreted = I->track + P[other].Mass */
                O->acMtrack += BHP(other).Mtrack;
                O->Mass += (P[other].Mass + I->Mtrack - blackhole_params.SeedBHDynMass);
            }
            if(I->Mtrack >= blackhole_params.SeedBHDynMass && BHP(other).Mtrack >= blackhole_params.SeedBHDynMass){
            /* trivial case, total_gas_accreted = I->Mass + P[other].Mass */
                O->Mass += P[other].Mass;
            }
        }
        else{
            O->Mass += P[other].Mass;
        }
        MyFloat VelPred[3];
        DM_VelPred(other, VelPred, BH_GET_PRIV(lv->tw)->kf);
        /* Conserve momentum during accretion*/
        int d;
        for(d = 0; d < 3; d++)
            O->AccretedMomentum[d] += (P[other].Mass * VelPred[d]);

        if(BHP(other).SwallowTime < BH_GET_PRIV(lv->tw)->atime)
            endrun(2, "Encountered BH %i swallowed at earlier time %g\n", other, BHP(other).SwallowTime);

        int tid = omp_get_thread_num();
        BH_GET_PRIV(lv->tw)->N_BH_swallowed[tid]++;

    }

    MyIDType * SPH_SwallowID = BH_GET_PRIV(lv->tw)->SPH_SwallowID;

    /* perform thermal or kinetic feedback into non-swallowed particles. */
    if(P[other].Type == 0 && SPH_SwallowID[P[other].PI] == 0 &&
        (r2 < iter->feedback_kernel.HH && P[other].Mass > 0) &&
            (I->FeedbackWeightSum > 0))
    {
        double u = r * iter->feedback_kernel.Hinv;
        double wk = 1.0;
        double mass_j;

        if(HAS(blackhole_params.BlackHoleFeedbackMethod, BH_FEEDBACK_MASS)) {
            mass_j = P[other].Mass;
        } else {
            mass_j = P[other].Hsml * P[other].Hsml * P[other].Hsml;
        }
        if(HAS(blackhole_params.BlackHoleFeedbackMethod, BH_FEEDBACK_SPLINE))
            wk = density_kernel_wk(&iter->feedback_kernel, u);

        /* thermal feedback */
        if(I->FeedbackEnergy > 0 && I->FdbkChannel == 0){
            const double injected_BH = I->FeedbackEnergy * mass_j * wk / I->FeedbackWeightSum;
            /* Set a flag for star-forming particles:
                * we want these to cool to the EEQOS via
                * tcool rather than trelax.*/
            if(sfreff_on_eeqos(&SPHP(other), BH_GET_PRIV(lv->tw)->a3inv)) {
                /* We cannot atomically set a bitfield.
                 * This flag is never read in this thread loop, and we are careful not to
                 * do this with a swallowed particle (as this can race with IsGarbage being set).
                 * So lack of atomicity is (I think) not a problem.*/
                //#pragma omp atomic write
                P[other].BHHeated = 1;
            }
            const double enttou = pow(SPHP(other).Density * BH_GET_PRIV(lv->tw)->a3inv, GAMMA_MINUS1) / GAMMA_MINUS1;
            const double uu_in_cgs = BH_GET_PRIV(lv->tw)->units.UnitEnergy_in_cgs / BH_GET_PRIV(lv->tw)->units.UnitMass_in_g;

            double entold, entnew;
            double * entptr = &(SPHP(other).Entropy);
            #pragma omp atomic read
            entold = *entptr;
            do {
                entnew = add_injected_BH_energy(entold * enttou, injected_BH, P[other].Mass, uu_in_cgs) / enttou;
                /* Swap in the new gas entropy only if the old one hasn't changed.*/
            } while(!__atomic_compare_exchange(entptr, &entold, &entnew, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
        }

        /* kinetic feedback */
        if(I->KEFeedbackEnergy > 0 && I->FdbkChannel == 1 && I->Density > 0){
            /* Kick the gas particle*/
            double dvel = sqrt(2 * I->KEFeedbackEnergy * wk / I->Density);
            double dir[3];
            get_random_dir(other, dir);
            int j;
            for(j = 0; j < 3; j++){
                #pragma omp atomic update
                P[other].Vel[j] += (dvel*dir[j]);
            }
        }
    }

    /* Swallowing a gas */
    /* This will only be true on one thread so we do not need a lock here*/
    /* Note that it will rarely happen that gas is swallowed by a BH which is itself swallowed.
     * In that case we do not swallow this particle: all swallowing changes before this are temporary*/
    if(P[other].Type == 0 && SPH_SwallowID[P[other].PI] == I->ID+1)
    {
        /* We do not know how to notify the tree of mass changes. so
         * enforce a mass conservation. */
        if(blackhole_params.SeedBHDynMass > 0 && I->Mtrack < blackhole_params.SeedBHDynMass) {
            /* we just add gas mass to Mtrack instead of dynMass */
            O->acMtrack += P[other].Mass;
        } else
            O->Mass += P[other].Mass;
        P[other].Mass = 0;
        MyFloat VelPred[3];
        SPH_VelPred(other, VelPred, BH_GET_PRIV(lv->tw)->kf);
        /* Conserve momentum during accretion*/
        int d;
        for(d = 0; d < 3; d++)
            O->AccretedMomentum[d] += (P[other].Mass * VelPred[d]);

        slots_mark_garbage(other, PartManager, SlotsManager);

        int tid = omp_get_thread_num();
        BH_GET_PRIV(lv->tw)->N_sph_swallowed[tid]++;
    }
}
static int
blackhole_feedback_haswork(int n, TreeWalk * tw)
{
    /*Black hole not being swallowed*/
    return (P[n].Type == 5) && (!P[n].Swallowed) && (BHP(n).SwallowID == (MyIDType) -1);
}

static void
blackhole_feedback_copy(int i, TreeWalkQueryBHFeedback * I, TreeWalk * tw)
{
    I->Hsml = P[i].Hsml;
    I->BH_Mass = BHP(i).Mass;
    I->ID = P[i].ID;
    I->Mtrack = BHP(i).Mtrack;
    I->Density = BHP(i).Density;
    int PI = P[i].PI;

    I->FeedbackWeightSum = BH_GET_PRIV(tw)->BH_FeedbackWeightSum[PI];
    I->FdbkChannel = 0; /* thermal feedback mode */

    double dtime = get_dloga_for_bin(P[i].TimeBinHydro, P[i].Ti_drift) / BH_GET_PRIV(tw)->hubble;

    I->FeedbackEnergy = blackhole_params.BlackHoleFeedbackFactor * 0.1 * BHP(i).Mdot * dtime *
                pow(LIGHTCGS / BH_GET_PRIV(tw)->units.UnitVelocity_in_cm_per_s, 2);
    I->KEFeedbackEnergy = 0;
    if (blackhole_params.BlackHoleKineticOn == 1 && BH_GET_PRIV(tw)->KEflag[PI] > 0){
        I->FdbkChannel = 1; /* kinetic feedback mode, (no thermal feedback for this timestep) */
        /* KEflag = 1: KEFeedbackEnergy is accumulating; KEflag = 2: KEFeedbackEnergy is released. */
        if (BH_GET_PRIV(tw)->KEflag[PI] == 2){
            I->KEFeedbackEnergy = BHP(i).KineticFdbkEnergy;
        }
    }
}

static void
blackhole_feedback_reduce(int place, TreeWalkResultBHFeedback * remote, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    int k;
    int PI = P[place].PI;

    TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_accreted_Mass[PI], remote->Mass);
    TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_accreted_BHMass[PI], remote->BH_Mass);
    TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_accreted_Mtrack[PI], remote->acMtrack);
    for(k = 0; k < 3; k++) {
        TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_accreted_momentum[PI][k], remote->AccretedMomentum[k]);
    }

    TREEWALK_REDUCE(BHP(place).CountProgs, remote->BH_CountProgs);
}

static void
blackhole_feedback_postprocess(int n, TreeWalk * tw)
{
    const int PI = P[n].PI;
    if(BH_GET_PRIV(tw)->BH_accreted_BHMass[PI] > 0){
       BHP(n).Mass += BH_GET_PRIV(tw)->BH_accreted_BHMass[PI];
    }
    if(BH_GET_PRIV(tw)->BH_accreted_Mass[PI] > 0)
    {
        /* velocity feedback due to accretion; momentum conservation.
         * This does nothing with repositioning on.*/
        const MyFloat accmass = BH_GET_PRIV(tw)->BH_accreted_Mass[PI];
        int k;
        /* Need to add the momentum from Mtrack as well*/
        for(k = 0; k < 3; k++)
            P[n].Vel[k] = (P[n].Vel[k] * P[n].Mass + BH_GET_PRIV(tw)->BH_accreted_momentum[PI][k]) /
                    (P[n].Mass + accmass + BH_GET_PRIV(tw)->BH_accreted_Mtrack[PI]);
        P[n].Mass += accmass;
    }

    if(blackhole_params.SeedBHDynMass>0){
        if(BH_GET_PRIV(tw)->BH_accreted_Mtrack[PI] > 0){
            BHP(n).Mtrack += BH_GET_PRIV(tw)->BH_accreted_Mtrack[PI];
        }
        if(BHP(n).Mtrack > blackhole_params.SeedBHDynMass){
            BHP(n).Mtrack = blackhole_params.SeedBHDynMass; /*cap Mtrack at SeedBHDynMass*/
        }
    }
    /* Reset KineticFdbkEnerg to 0 after released */
    if(BH_GET_PRIV(tw)->KEflag[PI] == 2){
        BHP(n).KineticFdbkEnergy = 0;
    }
}

/* Do the black hole feedback tree walk. Tree needs to have gas and BH.*/
static void
blackhole_feedback(int * ActiveBlackHoles, int64_t NumActiveBlackHoles, ForceTree * tree, struct BHPriv * priv)
{
    if(!(tree->mask & GASMASK) || !(tree->mask & BHMASK))
        endrun(5, "Error: BH tree types GAS: %d BH %d\n", tree->mask & GASMASK, tree->mask & BHMASK);

    TreeWalk tw_feedback[1] = {{0}};
    tw_feedback->ev_label = "BH_FEEDBACK";
    tw_feedback->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw_feedback->ngbiter_type_elsize = sizeof(TreeWalkNgbIterBHFeedback);
    tw_feedback->ngbiter = (TreeWalkNgbIterFunction) blackhole_feedback_ngbiter;
    tw_feedback->haswork = blackhole_feedback_haswork;
    tw_feedback->fill = (TreeWalkFillQueryFunction) blackhole_feedback_copy;
    tw_feedback->postprocess = (TreeWalkProcessFunction) blackhole_feedback_postprocess;
    tw_feedback->reduce = (TreeWalkReduceResultFunction) blackhole_feedback_reduce;
    tw_feedback->query_type_elsize = sizeof(TreeWalkQueryBHFeedback);
    tw_feedback->result_type_elsize = sizeof(TreeWalkResultBHFeedback);
    tw_feedback->tree = tree;
    tw_feedback->priv = priv;
    tw_feedback->repeatdisallowed = 1;

    /* Ionization counters*/
    priv[0].N_sph_swallowed = ta_malloc("n_sph_swallowed", int64_t, omp_get_max_threads());
    priv[0].N_BH_swallowed = ta_malloc("n_BH_swallowed", int64_t, omp_get_max_threads());
    memset(priv[0].N_sph_swallowed, 0, sizeof(int64_t) * omp_get_max_threads());
    memset(priv[0].N_BH_swallowed, 0, sizeof(int64_t) * omp_get_max_threads());

    treewalk_run(tw_feedback, ActiveBlackHoles, NumActiveBlackHoles);

    int i;
    int64_t Ntot_gas_swallowed, Ntot_BH_swallowed;
    int64_t N_sph_swallowed = 0, N_BH_swallowed = 0;
    for(i = 0; i < omp_get_max_threads(); i++) {
        N_sph_swallowed += priv[0].N_sph_swallowed[i];
        N_BH_swallowed += priv[0].N_BH_swallowed[i];
    }
    ta_free(priv[0].N_BH_swallowed);
    ta_free(priv[0].N_sph_swallowed);

    MPI_Reduce(&N_sph_swallowed, &Ntot_gas_swallowed, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&N_BH_swallowed, &Ntot_BH_swallowed, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);

    MPIU_Barrier(MPI_COMM_WORLD);
    message(0, "Accretion done: %d gas particles swallowed, %d BH particles swallowed\n",
                Ntot_gas_swallowed, Ntot_BH_swallowed);
}

/* Sample from a power law to get the initial BH mass*/
static double
bh_powerlaw_seed_mass(MyIDType ID)
{
    /* compute random number, uniform in [0,1] */
    const double w = get_random_number(ID+23);
    /* Normalisation for this power law index*/
    double norm = pow(blackhole_params.MaxSeedBlackHoleMass, 1+blackhole_params.SeedBlackHoleMassIndex)
                - pow(blackhole_params.SeedBlackHoleMass, 1+blackhole_params.SeedBlackHoleMassIndex);
    /* Sample from the CDF:
     * w  = [M^(1+x) - M_0^(1+x)]/[M_1^(1+x) - M_0^(1+x)]
     * w [M_1^(1+x) - M_0^(1+x)] + M_0^(1+x) = M^(1+x)
     * M = pow((w [M_1^(1+x) - M_0^(1+x)] + M_0^(1+x)), 1/(1+x))*/
    double mass = pow(w * norm + pow(blackhole_params.SeedBlackHoleMass, 1+blackhole_params.SeedBlackHoleMassIndex),
                      1./(1+blackhole_params.SeedBlackHoleMassIndex));
    return mass;
}

void blackhole_make_one(int index, const double atime) {
    if(P[index].Type != 0)
        endrun(7772, "Only Gas turns into blackholes, what's wrong?");

    int child = index;

    /* Make the new particle a black hole: use all the P[i].Mass
     * so we don't have lots of low mass tracers.
     * If the BH seed mass is small this may lead to a mismatch
     * between the gas and BH mass. */
    child = slots_convert(child, 5, -1, PartManager, SlotsManager);

    /* The accretion mass should always be the seed black hole mass,
     * irrespective of the gravitational mass of the particle.*/
    if(blackhole_params.MaxSeedBlackHoleMass > 0)
        BHP(child).Mass = bh_powerlaw_seed_mass(P[child].ID);
    else
        BHP(child).Mass = blackhole_params.SeedBlackHoleMass;

    BHP(child).Mseed = BHP(child).Mass;
    BHP(child).Mdot = 0;
    BHP(child).FormationTime = atime;
    BHP(child).SwallowID = (MyIDType) -1;
    BHP(child).Density = 0;

    /* It is important to initialize MinPotPos to the current position of
     * a BH to avoid drifting to unknown locations (0,0,0) immediately
     * after the BH is created. */
    int j;
    for(j = 0; j < 3; j++) {
        BHP(child).MinPotPos[j] = P[child].Pos[j];
        BHP(child).DFAccel[j] = 0;
        BHP(child).DragAccel[j] = 0;
    }
    BHP(child).JumpToMinPot = 0;
    BHP(child).CountProgs = 1;

    if (blackhole_params.SeedBHDynMass>0){
        BHP(child).Mtrack = P[child].Mass;
        P[child].Mass = blackhole_params.SeedBHDynMass;
    }
    else{
        BHP(child).Mtrack = -1; /* This column is not used then. */
    }
    /* Initialize KineticFdbkEnergy, keep zero if BlackHoleKineticOn is not turned on */
    BHP(child).KineticFdbkEnergy = 0;
    BHP(child).VDisp = 0;
}
