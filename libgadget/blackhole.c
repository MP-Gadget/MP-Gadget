#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "utils.h"
#include "cooling.h"
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
/*! \file blackhole.c
 *  \brief routines for gas accretion onto black holes, and black hole mergers
 */

struct BlackholeParams
{
    double BlackHoleAccretionFactor;	/*!< Fraction of BH bondi accretion rate */
    double BlackHoleFeedbackFactor;	/*!< Fraction of the black luminosity feed into thermal feedback */
    enum BlackHoleFeedbackMethod BlackHoleFeedbackMethod;	/*!< method of the feedback*/
    double BlackHoleFeedbackRadius;	/*!< Radius the thermal feedback is fed comoving*/
    double BlackHoleFeedbackRadiusMaxPhys;	/*!< Radius the thermal cap */
    double BlackHoleEddingtonFactor;	/*! Factor above Eddington */
    int BlackHoleRepositionEnabled; /* If true, enable repositioning the BH to the potential minimum*/

    /**********************************************************************/
    int MergeGravBound; /*if 1, apply gravitational bound criteria for BH mergers */

    int BH_DynFrictionMethod;/*0 for off; 1 for Star Only; 2 for DM+Star; 3 for DM+Star+Gas */
    int BH_DFBoostFactor; /*Optional boost factor for DF */
    double BH_DFbmax; /* the maximum impact range, in physical unit of kpc. */
    int BH_DRAG; /*Hydro drag force*/

    double SeedBHDynMass; /* The initial dynamic mass of BH particle */

    double SeedBlackHoleMass;	/*!< (minimum) Seed black hole mass */
    double MaxSeedBlackHoleMass; /* Maximum black hole seed mass*/
    double SeedBlackHoleMassIndex; /* Power law index for BH seed mass*/
    /************************************************************************/
} blackhole_params;

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
} TreeWalkResultBHAccretion;

typedef struct {
    TreeWalkNgbIterBase base;
    DensityKernel accretion_kernel;
    DensityKernel feedback_kernel;
} TreeWalkNgbIterBHAccretion;


/*****************************************************************************/
typedef struct {
    TreeWalkQueryBase base;
    MyFloat Hsml;
} TreeWalkQueryBHDynfric;

typedef struct {
    TreeWalkResultBase base;

    MyFloat SurroundingVel[3];
    MyFloat SurroundingDensity;
    MyFloat SurroundingParticles;
    MyFloat SurroundingRmsVel;

} TreeWalkResultBHDynfric;

typedef struct {
    TreeWalkNgbIterBase base;
    DensityKernel dynfric_kernel;
} TreeWalkNgbIterBHDynfric;

/*****************************************************************************/


typedef struct {
    TreeWalkQueryBase base;
    MyFloat Hsml;
    MyFloat Mtrack;
    MyFloat BH_Mass;
    MyIDType ID;
    MyFloat FeedbackEnergy;
    MyFloat FeedbackWeightSum;
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

struct BHPriv {
    /* Temporary array to store the IDs of the swallowing black hole for gas.
     * We store ID + 1 so that SwallowID == 0 can correspond to the unswallowed case. */
    MyIDType * SPH_SwallowID;
    /* These are temporaries used in the accretion treewalk*/
    MyFloat * MinPot;
    MyFloat * BH_Entropy;
    MyFloat (*BH_SurroundingGasVel)[3];

    /*************************************************************************/
    /* used in the dynamic friction treewalk*/
    MyFloat * BH_SurroundingDensity;
    MyFloat * BH_SurroundingParticles;
    MyFloat (*BH_SurroundingVel)[3];
    MyFloat * BH_SurroundingRmsVel;

    /*************************************************************************/

    MyFloat (*BH_accreted_momentum)[3];

    /* These are temporaries used in the feedback treewalk.*/
    MyFloat * BH_accreted_Mass;
    MyFloat * BH_accreted_BHMass;
    MyFloat * BH_accreted_Mtrack;

    /* This is a temporary computed in the accretion treewalk and used
     * in the feedback treewalk*/
    MyFloat * BH_FeedbackWeightSum;

    double a3inv;
    /* Counters*/
    int64_t * N_sph_swallowed;
    int64_t * N_BH_swallowed;
};
#define BH_GET_PRIV(tw) ((struct BHPriv *) (tw->priv))

struct BHinfo{

    MyIDType ID;
    MyFloat Mass;
    MyFloat Mdot;
    MyFloat Density;
    int minTimeBin;
    int encounter;

    double  MinPotPos[3];
    MyFloat MinPot;
    MyFloat BH_Entropy;
    MyFloat BH_SurroundingGasVel[3];
    MyFloat BH_accreted_momentum[3];

    MyFloat BH_accreted_Mass;
    MyFloat BH_accreted_BHMass;
    MyFloat BH_FeedbackWeightSum;

    MyIDType SPH_SwallowID;
    MyIDType SwallowID;

    int CountProgs;
    int Swallowed;

    /****************************************/
    double Pos[3];
    MyFloat BH_SurroundingDensity;
    MyFloat BH_SurroundingParticles;
    MyFloat BH_SurroundingVel[3];
    MyFloat BH_SurroundingRmsVel;

    double BH_DFAccel[3];
    double BH_DragAccel[3];
    double BH_GravAccel[3];
    double Velocity[3];
    double Mtrack;
    double Mdyn;

    MyDouble a;
};

/*Set the parameters of the BH module*/
void set_blackhole_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        blackhole_params.BlackHoleAccretionFactor = param_get_double(ps, "BlackHoleAccretionFactor");
        blackhole_params.BlackHoleEddingtonFactor = param_get_double(ps, "BlackHoleEddingtonFactor");

        blackhole_params.BlackHoleFeedbackFactor = param_get_double(ps, "BlackHoleFeedbackFactor");
        blackhole_params.BlackHoleFeedbackRadius = param_get_double(ps, "BlackHoleFeedbackRadius");

        blackhole_params.BlackHoleFeedbackRadiusMaxPhys = param_get_double(ps, "BlackHoleFeedbackRadiusMaxPhys");

        blackhole_params.BlackHoleFeedbackMethod = param_get_enum(ps, "BlackHoleFeedbackMethod");
        blackhole_params.BlackHoleRepositionEnabled = param_get_int(ps, "BlackHoleRepositionEnabled");
        /***********************************************************************************/
        blackhole_params.BH_DynFrictionMethod = param_get_int(ps, "BH_DynFrictionMethod");
        blackhole_params.BH_DFBoostFactor = param_get_int(ps, "BH_DFBoostFactor");
        blackhole_params.BH_DFbmax = param_get_double(ps, "BH_DFbmax");
        blackhole_params.BH_DRAG = param_get_int(ps, "BH_DRAG");
        blackhole_params.MergeGravBound = param_get_int(ps, "MergeGravBound");
        blackhole_params.SeedBHDynMass = param_get_double(ps,"SeedBHDynMass");

        blackhole_params.SeedBlackHoleMass = param_get_double(ps, "SeedBlackHoleMass");
        blackhole_params.MaxSeedBlackHoleMass = param_get_double(ps,"MaxSeedBlackHoleMass");
        blackhole_params.SeedBlackHoleMassIndex = param_get_double(ps,"SeedBlackHoleMassIndex");
        /***********************************************************************************/
    }
    MPI_Bcast(&blackhole_params, sizeof(struct BlackholeParams), MPI_BYTE, 0, MPI_COMM_WORLD);
}

/* accretion routines */
static void
blackhole_accretion_postprocess(int n, TreeWalk * tw);
/* feedback routines. currently also performs the drifting(move it to gravtree / force tree?) */
static int
blackhole_accretion_haswork(int n, TreeWalk * tw);

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



/*************************************************************************************/
/* DF routines */
static void
blackhole_dynfric_postprocess(int n, TreeWalk * tw);

static int
blackhole_dynfric_haswork(int n, TreeWalk * tw);

static void
blackhole_dynfric_reduce(int place, TreeWalkResultBHDynfric * remote, enum TreeWalkReduceMode mode, TreeWalk * tw);

static void
blackhole_dynfric_copy(int place, TreeWalkQueryBHDynfric * I, TreeWalk * tw);

static void
blackhole_dynfric_ngbiter(TreeWalkQueryBHDynfric * I,
        TreeWalkResultBHDynfric * O,
        TreeWalkNgbIterBHDynfric * iter,
        LocalTreeWalk * lv);

/*************************************************************************************/



/* feedback routines */

static void
blackhole_feedback_postprocess(int n, TreeWalk * tw);

static int
blackhole_feedback_haswork(int n, TreeWalk * tw);

static void
blackhole_feedback_reduce(int place, TreeWalkResultBHFeedback * remote, enum TreeWalkReduceMode mode, TreeWalk * tw);

static void
blackhole_feedback_copy(int place, TreeWalkQueryBHFeedback * I, TreeWalk * tw);

static void
blackhole_feedback_ngbiter(TreeWalkQueryBHFeedback * I,
        TreeWalkResultBHFeedback * O,
        TreeWalkNgbIterBHFeedback * iter,
        LocalTreeWalk * lv);

static double
decide_hsearch(double h);

#define BHPOTVALUEINIT 1.0e29

static double blackhole_soundspeed(double entropy, double rho) {
    /* rho is comoving !*/
    double cs = sqrt(GAMMA * entropy * pow(rho, GAMMA_MINUS1));

    cs *= pow(All.Time, -1.5 * GAMMA_MINUS1);

    return cs;
}

/* Adds the injected black hole energy to an internal energy and caps it at a maximum temperature*/
static double
add_injected_BH_energy(double unew, double injected_BH_energy, double mass)
{
    unew += injected_BH_energy / mass;
    const double u_to_temp_fac = (4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC))) * PROTONMASS / BOLTZMANN * GAMMA_MINUS1
    * All.UnitEnergy_in_cgs / All.UnitMass_in_g;

    /* Cap temperature*/
    if(unew > 5.0e8 / u_to_temp_fac)
        unew = 5.0e8 / u_to_temp_fac;

    return unew;
}

/* check if two BHs are gravitationally bounded, input dv, da, dx in code unit */
/* same as Bellovary2011, Tremmel2017 */
static int
check_grav_bound(double dx[3], double dv[3], double da[3])
{
    int j;
    double KE = 0;
    double PE = 0;

    for(j = 0; j < 3; j++){
        KE += 0.5 * pow(dv[j], 2);
        PE += da[j] * dx[j];
    }

    KE /= (All.cf.a*All.cf.a); /* convert to proper velocity */
    PE /= All.cf.a; /* convert to proper unit */

    /* The gravitationally bound condition is PE + KE < 0.
     * Still merge if it is marginally bound so that we merge
     * particles at zero distance and velocity from each other.*/
    return (PE + KE <= 0);
}



static void
collect_BH_info(int * ActiveParticle,int NumActiveParticle, struct BHPriv *priv, FILE * FdBlackholeDetails)
{
    int i;
    int c=0;

    for(i = 0; i < NumActiveParticle; i++)
    {
        int p_i = ActiveParticle ? ActiveParticle[i] : i;

        if(P[p_i].Type != 5 || P[p_i].IsGarbage || P[p_i].Mass <= 0)
          continue;

        int PI = P[p_i].PI;

        struct BHinfo info = {0};
        info.ID = P[p_i].ID;
        info.Mass = BHP(p_i).Mass;
        info.Mdot = BHP(p_i).Mdot;
        info.Density = BHP(p_i).Density;
        info.minTimeBin = BHP(p_i).minTimeBin;
        info.encounter = BHP(p_i).encounter;

        if(priv->MinPot) {
            info.MinPot = priv->MinPot[PI];
        }
        info.BH_Entropy = priv->BH_Entropy[PI];
        int k;
        for(k=0; k < 3; k++) {
            info.MinPotPos[k] = BHP(p_i).MinPotPos[k] - PartManager->CurrentParticleOffset[k];
            info.BH_SurroundingGasVel[k] = priv->BH_SurroundingGasVel[PI][k];
            info.BH_accreted_momentum[k] = priv->BH_accreted_momentum[PI][k];
            info.BH_DragAccel[k] = BHP(p_i).DragAccel[k];
            info.BH_GravAccel[k] = P[p_i].GravAccel[k];
            info.Pos[k] = P[p_i].Pos[k] - PartManager->CurrentParticleOffset[k];
            info.Velocity[k] = P[p_i].Vel[k];
            info.BH_DFAccel[k] = BHP(p_i).DFAccel[k];
        }

        /****************************************************************************/
        /* Output some DF info for debugging */
        info.BH_SurroundingDensity = priv->BH_SurroundingDensity[PI];
        info.BH_SurroundingRmsVel = priv->BH_SurroundingRmsVel[PI];
        info.BH_SurroundingParticles = priv->BH_SurroundingParticles[PI];
        info.BH_SurroundingVel[0] = priv->BH_SurroundingVel[PI][0];
        info.BH_SurroundingVel[1] = priv->BH_SurroundingVel[PI][1];
        info.BH_SurroundingVel[2] = priv->BH_SurroundingVel[PI][2];

        /****************************************************************************/
        info.BH_accreted_BHMass = priv->BH_accreted_BHMass[PI];
        info.BH_accreted_Mass = priv->BH_accreted_Mass[PI];
        info.BH_FeedbackWeightSum = priv->BH_FeedbackWeightSum[PI];

        info.SPH_SwallowID = priv->SPH_SwallowID[PI];
        info.SwallowID =  BHP(p_i).SwallowID;
        info.CountProgs = BHP(p_i).CountProgs;
        info.Swallowed =  P[p_i].Swallowed;
        /************************************************************************************************/
        /* When SeedBHDynMass is larger than gas particle mass, we have three mass tracer of blackhole. */
        /* BHP(p_i).Mass : intrinsic mass of BH, accreted every (active) time step.                     */
        /* P[p_i].Mass :  Dynamic mass of BH, used for gravitational interaction.                       */
        /*                Starts to accrete gas particle when BHP(p_i).Mass > SeedBHDynMass             */
        /* BHP(p_i).Mtrack: Initialized as gas particle mass, and is capped at SeedBHDynMass,           */
        /*                 it traces BHP(p_i).Mass by swallowing gas when BHP(p_i).Mass < SeedBHDynMass */
        /************************************************************************************************/
        info.Mtrack = BHP(p_i).Mtrack;
        info.Mdyn = P[p_i].Mass;

        info.a = All.Time;

        int size = sizeof(info);

        fwrite(&size, sizeof(size), 1, FdBlackholeDetails);
        fwrite(&info,sizeof(info),1,FdBlackholeDetails);
        fwrite(&size, sizeof(size), 1, FdBlackholeDetails);
        c++;
    }

    fflush(FdBlackholeDetails);
    int64_t totalN;

    sumup_large_ints(1, &c, &totalN);
    message(0, "Written details of %ld blackholes.\n", totalN);
}


void
blackhole(const ActiveParticles * act, ForceTree * tree, FILE * FdBlackHoles, FILE * FdBlackholeDetails)
{
    if(!All.BlackHoleOn)
        return;
    /* Do nothing if no black holes*/
    int64_t totbh;
    sumup_large_ints(1, &SlotsManager->info[5].size, &totbh);
    if(totbh == 0)
        return;
    int i;

    walltime_measure("/Misc");
    struct BHPriv priv[1] = {0};

    /*************************************************************************/
    TreeWalk tw_dynfric[1] = {{0}};
    tw_dynfric->ev_label = "BH_DYNFRIC";
    tw_dynfric->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw_dynfric->ngbiter_type_elsize = sizeof(TreeWalkNgbIterBHDynfric);
    tw_dynfric->ngbiter = (TreeWalkNgbIterFunction) blackhole_dynfric_ngbiter;
    tw_dynfric->haswork = blackhole_dynfric_haswork;
    tw_dynfric->postprocess = (TreeWalkProcessFunction) blackhole_dynfric_postprocess;
    tw_dynfric->fill = (TreeWalkFillQueryFunction) blackhole_dynfric_copy;
    tw_dynfric->reduce = (TreeWalkReduceResultFunction) blackhole_dynfric_reduce;
    tw_dynfric->query_type_elsize = sizeof(TreeWalkQueryBHDynfric);
    tw_dynfric->result_type_elsize = sizeof(TreeWalkResultBHDynfric);
    tw_dynfric->tree = tree;
    tw_dynfric->priv = priv;

    /*************************************************************************/
    TreeWalk tw_accretion[1] = {{0}};

    tw_accretion->ev_label = "BH_ACCRETION";
    tw_accretion->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw_accretion->ngbiter_type_elsize = sizeof(TreeWalkNgbIterBHAccretion);
    tw_accretion->ngbiter = (TreeWalkNgbIterFunction) blackhole_accretion_ngbiter;
    tw_accretion->haswork = blackhole_accretion_haswork;
    tw_accretion->postprocess = (TreeWalkProcessFunction) blackhole_accretion_postprocess;
    tw_accretion->preprocess = (TreeWalkProcessFunction) blackhole_accretion_preprocess;
    tw_accretion->fill = (TreeWalkFillQueryFunction) blackhole_accretion_copy;
    tw_accretion->reduce = (TreeWalkReduceResultFunction) blackhole_accretion_reduce;
    tw_accretion->query_type_elsize = sizeof(TreeWalkQueryBHAccretion);
    tw_accretion->result_type_elsize = sizeof(TreeWalkResultBHAccretion);
    tw_accretion->tree = tree;
    tw_accretion->priv = priv;

    /*************************************************************************/

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


    priv->a3inv = 1./(All.Time * All.Time * All.Time);

    /*************************************************************************/
    /*  Dynamical Friction Treewalk */

    MPIU_Barrier(MPI_COMM_WORLD);
    message(0, "Beginning dynamical friction calculation. \n");
    /* Environment variables for DF */
    priv->BH_SurroundingRmsVel = mymalloc("BH_SurroundingRmsVel", SlotsManager->info[5].size * sizeof(priv->BH_SurroundingRmsVel));
    priv->BH_SurroundingVel = (MyFloat (*) [3]) mymalloc("BH_SurroundingVel", 3* SlotsManager->info[5].size * sizeof(priv->BH_SurroundingVel[0]));
    priv->BH_SurroundingParticles = mymalloc("BH_SurroundingParticles", SlotsManager->info[5].size * sizeof(priv->BH_SurroundingParticles));
    priv->BH_SurroundingDensity = mymalloc("BH_SurroundingDensity", SlotsManager->info[5].size * sizeof(priv->BH_SurroundingDensity));
    /* guard treewalk */
    if (blackhole_params.BH_DynFrictionMethod > 0)
        treewalk_run(tw_dynfric, act->ActiveParticle, act->NumActiveParticle);

    /*************************************************************************/

    walltime_measure("/BH/DynFric");
    MPIU_Barrier(MPI_COMM_WORLD);
    message(0, "Beginning black-hole accretion\n");

    /* Let's determine which particles may be swallowed and calculate total feedback weights */
    priv->SPH_SwallowID = mymalloc("SPH_SwallowID", SlotsManager->info[0].size * sizeof(MyIDType));
    memset(priv->SPH_SwallowID, 0, SlotsManager->info[0].size * sizeof(MyIDType));

    /* Computed in accretion, used in feedback*/
    priv->BH_FeedbackWeightSum = mymalloc("BH_FeedbackWeightSum", SlotsManager->info[5].size * sizeof(MyFloat));

    /* These are initialized in preprocess and used to reposition the BH in postprocess*/
    priv->MinPot = mymalloc("BH_MinPot", SlotsManager->info[5].size * sizeof(MyFloat));

    /* Local to this treewalk*/
    priv->BH_Entropy = mymalloc("BH_Entropy", SlotsManager->info[5].size * sizeof(MyFloat));
    priv->BH_SurroundingGasVel = (MyFloat (*) [3]) mymalloc("BH_SurroundVel", 3* SlotsManager->info[5].size * sizeof(priv->BH_SurroundingGasVel[0]));

    /* This allocates memory*/
    treewalk_run(tw_accretion, act->ActiveParticle, act->NumActiveParticle);

    /*************************************************************************/

    walltime_measure("/BH/Accretion");
    MPIU_Barrier(MPI_COMM_WORLD);
    message(0, "Start swallowing of gas particles and black holes\n");

    /* Now do the swallowing of particles and dump feedback energy */

    /* Ionization counters*/
    priv[0].N_sph_swallowed = ta_malloc("n_sph_swallowed", int64_t, omp_get_max_threads());
    priv[0].N_BH_swallowed = ta_malloc("n_BH_swallowed", int64_t, omp_get_max_threads());
    memset(priv[0].N_sph_swallowed, 0, sizeof(int64_t) * omp_get_max_threads());
    memset(priv[0].N_BH_swallowed, 0, sizeof(int64_t) * omp_get_max_threads());

    priv->BH_accreted_Mass = mymalloc("BH_accretedmass", SlotsManager->info[5].size * sizeof(MyFloat));
    priv->BH_accreted_BHMass = mymalloc("BH_accreted_BHMass", SlotsManager->info[5].size * sizeof(MyFloat));
    priv->BH_accreted_Mtrack = mymalloc("BH_accreted_Mtrack", SlotsManager->info[5].size * sizeof(MyFloat));
    priv->BH_accreted_momentum = (MyFloat (*) [3]) mymalloc("BH_accretemom", 3* SlotsManager->info[5].size * sizeof(priv->BH_accreted_momentum[0]));

    treewalk_run(tw_feedback, act->ActiveParticle, act->NumActiveParticle);

    /*************************************************************************/
    walltime_measure("/BH/Feedback");

    if(FdBlackholeDetails){
        collect_BH_info(act->ActiveParticle, act->NumActiveParticle, priv, FdBlackholeDetails);
    }

    myfree(priv->BH_accreted_momentum);
    myfree(priv->BH_accreted_Mtrack);
    myfree(priv->BH_accreted_BHMass);
    myfree(priv->BH_accreted_Mass);

    /*****************************************************************/
    myfree(priv->BH_SurroundingGasVel);
    myfree(priv->BH_Entropy);
    myfree(priv->MinPot);

    myfree(priv->BH_FeedbackWeightSum);
    myfree(priv->SPH_SwallowID);

    /*****************************************************************/
    myfree(priv->BH_SurroundingDensity);
    myfree(priv->BH_SurroundingParticles);
    myfree(priv->BH_SurroundingVel);
    myfree(priv->BH_SurroundingRmsVel);
    /*****************************************************************/

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

    int total_bh;
    double total_mdoteddington;
    double total_mass_holes, total_mdot;

    double Local_BH_mass = 0;
    double Local_BH_Mdot = 0;
    double Local_BH_Medd = 0;
    int Local_BH_num = 0;
    /* Compute total mass of black holes
     * present by summing contents of black hole array*/
    #pragma omp parallel for reduction(+ : Local_BH_num) reduction(+: Local_BH_mass) reduction(+: Local_BH_Mdot) reduction(+: Local_BH_Medd)
    for(i = 0; i < SlotsManager->info[5].size; i ++)
    {
        if(BhP[i].SwallowID != (MyIDType) -1)
            continue;
        Local_BH_num++;
        Local_BH_mass += BhP[i].Mass;
        Local_BH_Mdot += BhP[i].Mdot;
        Local_BH_Medd += BhP[i].Mdot/BhP[i].Mass;
    }

    MPI_Reduce(&Local_BH_mass, &total_mass_holes, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Local_BH_Mdot, &total_mdot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Local_BH_Medd, &total_mdoteddington, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Local_BH_num, &total_bh, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(FdBlackHoles)
    {
        /* convert to solar masses per yr */
        double mdot_in_msun_per_year =
            total_mdot * (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);

        total_mdoteddington *= 1.0 / ((4 * M_PI * GRAVITY * LIGHTCGS * PROTONMASS /
                    (0.1 * LIGHTCGS * LIGHTCGS * THOMPSON)) * All.UnitTime_in_s);

        fprintf(FdBlackHoles, "%g %d %g %g %g %g\n",
                All.Time, total_bh, total_mass_holes, total_mdot, mdot_in_msun_per_year, total_mdoteddington);
        fflush(FdBlackHoles);
    }
    walltime_measure("/BH/Info");
}


/*************************************************************************************/
/* DF routines */
static void
blackhole_dynfric_postprocess(int n, TreeWalk * tw){

    int PI = P[n].PI;
    int j;

    /***********************************************************************************/
    /* This is Gizmo's implementation of dynamic friction                              */
    /* c.f. section 3.1 in http://www.tapir.caltech.edu/~phopkins/public/notes_blackholes.pdf */
    /* Compute dynamic friction accel when DF turned on                                */
    /* averaged value for colomb logarithm and integral over the distribution function */
    /* acc_friction = -4*pi*G^2 * Mbh * log(lambda) * rho * f_of_x * bhvel / |bhvel^3| */
    /*       f_of_x = [erf(x) - 2*x*exp(-x^2)/sqrt(pi)]                                */
    /*       lambda = b_max * v^2 / G / (M+m)                                          */
    /*        b_max = Size of system (e.g. Rvir)                                       */
    /*            v = Relative velocity of BH with respect to the environment          */
    /*            M = Mass of BH                                                       */
    /*            m = individual mass elements composing the large system (e.g. m<<M)  */
    /*            x = v/sqrt(2)/sigma                                                  */
    /*        sigma = width of the max. distr. of the host system                      */
    /*                (e.g. sigma = v_disp / 3                                         */

    if(BH_GET_PRIV(tw)->BH_SurroundingDensity[PI] > 0){
        double bhvel;
        double lambda, x, f_of_x;
        const double a_erf = 8 * (M_PI - 3) / (3 * M_PI * (4. - M_PI));

        /* normalize velocity/dispersion */
        BH_GET_PRIV(tw)->BH_SurroundingRmsVel[PI] /= BH_GET_PRIV(tw)->BH_SurroundingDensity[PI];
        BH_GET_PRIV(tw)->BH_SurroundingRmsVel[PI] = sqrt(BH_GET_PRIV(tw)->BH_SurroundingRmsVel[PI]);
        for(j = 0; j < 3; j++)
            BH_GET_PRIV(tw)->BH_SurroundingVel[PI][j] /= BH_GET_PRIV(tw)->BH_SurroundingDensity[PI];

        /* Calculate Coulumb Logarithm */
        bhvel = 0;
        for(j = 0; j < 3; j++)
        {
            bhvel += pow(P[n].Vel[j] - BH_GET_PRIV(tw)->BH_SurroundingVel[PI][j], 2);
        }
        bhvel = sqrt(bhvel);

        /* There is no parameter in physical unit, so I kept everything in code unit */

        x = bhvel / sqrt(2) / (BH_GET_PRIV(tw)->BH_SurroundingRmsVel[PI] / 3);
        /* First term is aproximation of the error function */
        f_of_x = x / fabs(x) * sqrt(1 - exp(-x * x * (4 / M_PI + a_erf * x * x)
            / (1 + a_erf * x * x))) - 2 * x / sqrt(M_PI) * exp(-x * x);
        /* Floor at zero */
        if (f_of_x < 0)
            f_of_x = 0;

        lambda = 1. + blackhole_params.BH_DFbmax * pow((bhvel/All.cf.a),2) / All.G / P[n].Mass;

        for(j = 0; j < 3; j++)
        {
            BHP(n).DFAccel[j] = - 4. * M_PI * All.G * All.G * P[n].Mass * BH_GET_PRIV(tw)->BH_SurroundingDensity[PI] *
            log(lambda) * f_of_x * (P[n].Vel[j] - BH_GET_PRIV(tw)->BH_SurroundingVel[PI][j]) / pow(bhvel, 3);
            BHP(n).DFAccel[j] *= All.cf.a;  // convert to code unit of acceleration
            BHP(n).DFAccel[j] *= blackhole_params.BH_DFBoostFactor; // Add a boost factor
        }
#ifdef DEBUG
        message(2,"x=%e, log(lambda)=%e, fof_x=%e, Mbh=%e, ratio=%e \n",
           x,log(lambda),f_of_x,P[n].Mass,BHP(n).DFAccel[0]/P[n].GravAccel[0]);
#endif
    }
    else
    {
        message(2, "Density is zero in DF kernel, kernel may be too small.\n");
        for(j = 0; j < 3; j++)
        {
            BHP(n).DFAccel[j] = 0;
        }
    }
}
    /*******************************************************************/

static int
blackhole_dynfric_haswork(int n, TreeWalk * tw){
    /*Black hole not being swallowed*/
    return (P[n].Type == 5) && (!P[n].Swallowed);
}

static void
blackhole_dynfric_reduce(int place, TreeWalkResultBHDynfric * remote, enum TreeWalkReduceMode mode, TreeWalk * tw){
    int PI = P[place].PI;

    TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_SurroundingDensity[PI], remote->SurroundingDensity);
    TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_SurroundingParticles[PI], remote->SurroundingParticles);
    TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_SurroundingVel[PI][0], remote->SurroundingVel[0]);
    TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_SurroundingVel[PI][1], remote->SurroundingVel[1]);
    TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_SurroundingVel[PI][2], remote->SurroundingVel[2]);
    TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_SurroundingRmsVel[PI], remote->SurroundingRmsVel);

}

static void
blackhole_dynfric_copy(int place, TreeWalkQueryBHDynfric * I, TreeWalk * tw){
    /* SPH kernel width should be the only thing needed */
    I->Hsml = P[place].Hsml;
}


static void
blackhole_dynfric_ngbiter(TreeWalkQueryBHDynfric * I,
        TreeWalkResultBHDynfric * O,
        TreeWalkNgbIterBHDynfric * iter,
        LocalTreeWalk * lv){

   if(iter->base.other == -1) {
        iter->base.mask = 1 + 2 + 4 + 8 + 16 + 32;
        iter->base.Hsml = I->Hsml;
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        density_kernel_init(&iter->dynfric_kernel, I->Hsml, GetDensityKernelType());
        return;
    }

    int other = iter->base.other;
    double r = iter->base.r;
    double r2 = iter->base.r2;

    /* Collect Star/+DM/+Gas density/velocity for DF computation */
    if(P[other].Type == 4 || (P[other].Type == 1 && blackhole_params.BH_DynFrictionMethod > 1) ||
        (P[other].Type == 0 && blackhole_params.BH_DynFrictionMethod == 3) ){
        if(r2 < iter->dynfric_kernel.HH) {
            double u = r * iter->dynfric_kernel.Hinv;
            double wk = density_kernel_wk(&iter->dynfric_kernel, u);
            float mass_j = P[other].Mass;

            O->SurroundingParticles += 1;
            O->SurroundingDensity += (mass_j * wk);
            for (int k = 0; k < 3; k++){
                O->SurroundingVel[k] += (mass_j * wk * P[other].Vel[k]);
                O->SurroundingRmsVel += (mass_j * wk * pow(P[other].Vel[k], 2));
            }
        }
    }
}

/*************************************************************************************/


static void
blackhole_accretion_postprocess(int i, TreeWalk * tw)
{
    int k;
    int PI = P[i].PI;
    if(BHP(i).Density > 0)
    {
        BH_GET_PRIV(tw)->BH_Entropy[PI] /= BHP(i).Density;
        for(k = 0; k < 3; k++)
            BH_GET_PRIV(tw)->BH_SurroundingGasVel[PI][k] /= BHP(i).Density;
    }

    double mdot = 0;		/* if no accretion model is enabled, we have mdot=0 */

    double rho = BHP(i).Density;
    double bhvel = 0;
    for(k = 0; k < 3; k++)
        bhvel += pow(P[i].Vel[k] - BH_GET_PRIV(tw)->BH_SurroundingGasVel[PI][k], 2);

    bhvel = sqrt(bhvel);
    bhvel /= All.cf.a;
    double rho_proper = rho * BH_GET_PRIV(tw)->a3inv;

    double soundspeed = blackhole_soundspeed(BH_GET_PRIV(tw)->BH_Entropy[PI], rho);

    /* Note: we take here a radiative efficiency of 0.1 for Eddington accretion */
    double meddington = (4 * M_PI * GRAVITY * LIGHTCGS * PROTONMASS / (0.1 * LIGHTCGS * LIGHTCGS * THOMPSON)) * BHP(i).Mass
        * All.UnitTime_in_s / All.CP.HubbleParam;

    double norm = pow((pow(soundspeed, 2) + pow(bhvel, 2)), 1.5);

    if(norm > 0)
        mdot = 4. * M_PI * blackhole_params.BlackHoleAccretionFactor * All.G * All.G *
            BHP(i).Mass * BHP(i).Mass * rho_proper / norm;

    if(blackhole_params.BlackHoleEddingtonFactor > 0.0 &&
        mdot > blackhole_params.BlackHoleEddingtonFactor * meddington) {
        mdot = blackhole_params.BlackHoleEddingtonFactor * meddington;
    }
    BHP(i).Mdot = mdot;

    double dtime = get_dloga_for_bin(P[i].TimeBin, P[i].Ti_drift) / All.cf.hubble;

    BHP(i).Mass += BHP(i).Mdot * dtime;

    /*************************************************************************/

    if(blackhole_params.BH_DRAG > 0){
        /* a_BH = (v_gas - v_BH) Mdot/M_BH                                   */
        /* motivated by BH gaining momentum from the accreted gas            */
        /*c.f.section 3.2,in http://www.tapir.caltech.edu/~phopkins/public/notes_blackholes.pdf */
        double fac = 0;
        if (blackhole_params.BH_DRAG == 1) fac = BHP(i).Mdot/P[i].Mass;
        if (blackhole_params.BH_DRAG == 2) fac = blackhole_params.BlackHoleEddingtonFactor * meddington/BHP(i).Mass;
        fac *= All.cf.a; /* dv = acc * kick_fac = acc * a^{-1}dt, therefore acc = a*dv/dt  */
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
}

static void
blackhole_accretion_preprocess(int n, TreeWalk * tw)
{
    int j;
    BH_GET_PRIV(tw)->MinPot[P[n].PI] = P[n].Potential;

    for(j = 0; j < 3; j++) {
        BHP(n).MinPotPos[j] = P[n].Pos[j];
    }

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
        double hsearch;
        hsearch = decide_hsearch(I->Hsml);

        iter->base.mask = 1 + 2 + 4 + 8 + 16 + 32;
        iter->base.Hsml = hsearch;
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;

        density_kernel_init(&iter->accretion_kernel, I->Hsml, GetDensityKernelType());
        density_kernel_init(&iter->feedback_kernel, hsearch, GetDensityKernelType());
        return;
    }

    int other = iter->base.other;
    double r = iter->base.r;
    double r2 = iter->base.r2;

    if(P[other].Mass < 0) return;

    if(P[other].Type != 5) {
        if (O->BH_minTimeBin > P[other].TimeBin)
            O->BH_minTimeBin = P[other].TimeBin;
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
        if(blackhole_params.MergeGravBound == 1){

            double dx[3];
            double dv[3];
            double da[3];
            int d;

            for(d = 0; d < 3; d++){
                dx[d] = NEAREST(I->base.Pos[d] - P[other].Pos[d], All.BoxSize);
                dv[d] = I->Vel[d] - P[other].Vel[d];
                /* we include long range PM force, short range force and DF */
                da[d] = (I->Accel[d] - P[other].GravAccel[d] - P[other].GravPM[d] - BHP(other).DFAccel[d]);
            }
            flag = check_grav_bound(dx,dv,da);
            /*if(flag == 0)
                message(0, "dx %g %g %g dv %g %g %g da %g %g %g\n",dx[0], dx[1], dx[2], dv[0], dv[1], dv[2], da[0], da[1], da[2]);*/
        }

        /* do the merge */
        if(flag == 1)
        {
            O->encounter = 0;
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
            O->GasVel[0] += (mass_j * wk * P[other].Vel[0]);
            O->GasVel[1] += (mass_j * wk * P[other].Vel[1]);
            O->GasVel[2] += (mass_j * wk * P[other].Vel[2]);

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
                double redshift = 1./All.Time - 1;
                double nh0 = get_neutral_fraction_sfreff(redshift, &P[other], &SPHP(other));
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
        double hsearch;
        hsearch = decide_hsearch(I->Hsml);

        iter->base.mask = 1 + 32;
        iter->base.Hsml = hsearch;
        /* Swallow is symmetric, but feedback dumping is asymetric;
         * we apply a cut in r to break the symmetry. */
        iter->base.symmetric = NGB_TREEFIND_SYMMETRIC;
        density_kernel_init(&iter->feedback_kernel, hsearch, DENSITY_KERNEL_CUBIC_SPLINE);
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

        BHP(other).SwallowTime = All.Time;
        P[other].Swallowed = 1;
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

        /* Conserve momentum during accretion*/
        int d;
        for(d = 0; d < 3; d++)
            O->AccretedMomentum[d] += (P[other].Mass * P[other].Vel[d]);

        if(BHP(other).SwallowTime < All.Time)
            endrun(2, "Encountered BH %i swallowed at earlier time %g\n", other, BHP(other).SwallowTime);

        int tid = omp_get_thread_num();
        BH_GET_PRIV(lv->tw)->N_BH_swallowed[tid]++;

    }

    /* Dump feedback energy */
    if(P[other].Type == 0 &&
        (r2 < iter->feedback_kernel.HH && P[other].Mass > 0) &&
            (I->FeedbackWeightSum > 0 && I->FeedbackEnergy > 0))
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

        const double injected_BH = I->FeedbackEnergy * mass_j * wk / I->FeedbackWeightSum;
        /* Set a flag for star-forming particles:
            * we want these to cool to the EEQOS via
            * tcool rather than trelax.*/
        if(sfreff_on_eeqos(&SPHP(other), BH_GET_PRIV(lv->tw)->a3inv)) {
            /* We cannot atomically set a bitfield.
             * This flag is never read in this thread loop, nor
             * are other flags set here. So lack of atomicity is (I think) not a problem*/
            //#pragma omp atomic write
            P[other].BHHeated = 1;
        }
        const double enttou = pow(SPH_EOMDensity(&SPHP(other)) * BH_GET_PRIV(lv->tw)->a3inv, GAMMA_MINUS1) / GAMMA_MINUS1;
        double entold, entnew;
        #pragma omp atomic read
        entold = SPHP(other).Entropy;
        do {
            entnew = add_injected_BH_energy(entold * enttou, injected_BH, P[other].Mass) / enttou;
            /* Swap in the new gas entropy only if the old one hasn't changed.*/
        } while(!__atomic_compare_exchange(&(SPHP(other).Entropy), &entold, &entnew, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
    }

    MyIDType * SPH_SwallowID = BH_GET_PRIV(lv->tw)->SPH_SwallowID;

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
        /* Conserve momentum during accretion*/
        int d;
        for(d = 0; d < 3; d++)
            O->AccretedMomentum[d] += (P[other].Mass * P[other].Vel[d]);

        slots_mark_garbage(other, PartManager, SlotsManager);

        int tid = omp_get_thread_num();
        BH_GET_PRIV(lv->tw)->N_sph_swallowed[tid]++;
    }
}

static int
blackhole_accretion_haswork(int n, TreeWalk * tw)
{
    /* We need black holes not already swallowed (on a previous timestep).*/
    return (P[n].Type == 5) && (P[n].Mass > 0) && (!P[n].Swallowed);
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

    BHP(place).encounter = remote->encounter;

    if (mode == 0 || BHP(place).minTimeBin > remote->BH_minTimeBin) {
        BHP(place).minTimeBin = remote->BH_minTimeBin;
    }

    TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_FeedbackWeightSum[PI], remote->FeedbackWeightSum);
    TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_Entropy[PI], remote->SmoothedEntropy);
    TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_SurroundingGasVel[PI][0], remote->GasVel[0]);
    TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_SurroundingGasVel[PI][1], remote->GasVel[1]);
    TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_SurroundingGasVel[PI][2], remote->GasVel[2]);
}

static void
blackhole_accretion_copy(int place, TreeWalkQueryBHAccretion * I, TreeWalk * tw)
{
    int k;
    for(k = 0; k < 3; k++)
    {
        I->Vel[k] = P[place].Vel[k];
        I->Accel[k] = P[place].GravAccel[k] + P[place].GravPM[k] + BHP(place).DFAccel[k];
    }
    I->Hsml = P[place].Hsml;
    I->Mass = P[place].Mass;
    I->BH_Mass = BHP(place).Mass;
    I->Density = BHP(place).Density;
    I->ID = P[place].ID;
    I->Mtrack = BHP(place).Mtrack;
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
    int PI = P[i].PI;

    I->FeedbackWeightSum = BH_GET_PRIV(tw)->BH_FeedbackWeightSum[PI];

    double dtime = get_dloga_for_bin(P[i].TimeBin, P[i].Ti_drift) / All.cf.hubble;

    I->FeedbackEnergy = blackhole_params.BlackHoleFeedbackFactor * 0.1 * BHP(i).Mdot * dtime *
                pow(LIGHTCGS / All.UnitVelocity_in_cm_per_s, 2);
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

void blackhole_make_one(int index) {
    if(!All.BlackHoleOn)
        return;
    if(P[index].Type != 0)
        endrun(7772, "Only Gas turns into blackholes, what's wrong?");

    int child = index;

    /* Make the new particle a black hole: use all the P[i].Mass
     * so we don't have lots of low mass tracers.
     * If the BH seed mass is small this may lead to a mismatch
     * between the gas and BH mass. */
    child = slots_convert(child, 5, -1, PartManager, SlotsManager);

    BHP(child).base.ID = P[child].ID;
    /* The accretion mass should always be the seed black hole mass,
     * irrespective of the gravitational mass of the particle.*/
    if(blackhole_params.MaxSeedBlackHoleMass > 0)
        BHP(child).Mass = bh_powerlaw_seed_mass(P[child].ID);
    else
        BHP(child).Mass = blackhole_params.SeedBlackHoleMass;

    BHP(child).Mseed = BHP(child).Mass;
    BHP(child).Mdot = 0;
    BHP(child).FormationTime = All.Time;
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
}

static double
decide_hsearch(double h)
{
    if(blackhole_params.BlackHoleFeedbackRadius > 0) {
        /* BlackHoleFeedbackRadius is in comoving.
         * The Phys radius is capped by BlackHoleFeedbackRadiusMaxPhys
         * just like how it was done for grav smoothing.
         * */
        double rds;
        rds = blackhole_params.BlackHoleFeedbackRadiusMaxPhys / All.cf.a;

        if(rds > blackhole_params.BlackHoleFeedbackRadius) {
            rds = blackhole_params.BlackHoleFeedbackRadius;
        }
        return rds;
    } else {
        return h;
    }
}
