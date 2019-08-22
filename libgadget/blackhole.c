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
#include "sfr_eff.h"
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
    double SeedBlackHoleMass;	/*!< Seed black hole mass */
    double BlackHoleEddingtonFactor;	/*! Factor above Eddington */
} blackhole_params;

typedef struct {
    TreeWalkQueryBase base;
    MyFloat Density;
    MyFloat Hsml;
    MyFloat Mass;
    MyFloat BH_Mass;
    MyFloat Vel[3];
    MyFloat Csnd;
    MyIDType ID;
} TreeWalkQueryBHAccretion;

typedef struct {
    TreeWalkResultBase base;
    MyFloat BH_MinPotPos[3];
    MyFloat BH_MinPot;

    short int BH_minTimeBin;
    MyFloat FeedbackWeightSum;

    MyFloat Rho;
    MyFloat SmoothedEntropy;
    MyFloat GasVel[3];
} TreeWalkResultBHAccretion;

typedef struct {
    TreeWalkNgbIterBase base;
    DensityKernel accretion_kernel;
    DensityKernel feedback_kernel;
} TreeWalkNgbIterBHAccretion;

typedef struct {
    TreeWalkQueryBase base;
    MyFloat Hsml;
    MyFloat BH_Mass;
    MyIDType ID;
    MyFloat FeedbackEnergy;
    MyFloat FeedbackWeightSum;
} TreeWalkQueryBHFeedback;

typedef struct {
    TreeWalkResultBase base;
    MyFloat Mass;
    MyFloat BH_Mass;
    MyFloat AccretedMomentum[3];
    int BH_CountProgs;
} TreeWalkResultBHFeedback;

typedef struct {
    TreeWalkNgbIterBase base;
    DensityKernel feedback_kernel;
} TreeWalkNgbIterBHFeedback;

struct BHPriv {
    /*Temporary array to store the IDs of the swallowing black hole for gas*/
    MyIDType * SPH_SwallowID;
    /* These are temporaries used in the accretion treewalk*/
    MyFloat * MinPot;
    MyFloat * BH_Entropy;
    MyFloat (*BH_SurroundingGasVel)[3];

    /* These are temporaries used in the feedback treewalk.*/
    MyFloat * BH_accreted_Mass;
    MyFloat * BH_accreted_BHMass;

    /* This is a temporary computed in the accretion treewalk and used
     * in the feedback treewalk*/
    MyFloat * BH_FeedbackWeightSum;

    /* Particle SpinLocks*/
    struct SpinLocks * spin;
    /* Counters*/
    int N_sph_swallowed, N_BH_swallowed;
};
#define BH_GET_PRIV(tw) ((struct BHPriv *) (tw->priv))

/*Set the parameters of the BH module*/
void set_blackhole_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        blackhole_params.BlackHoleAccretionFactor = param_get_double(ps, "BlackHoleAccretionFactor");
        blackhole_params.BlackHoleEddingtonFactor = param_get_double(ps, "BlackHoleEddingtonFactor");
        blackhole_params.SeedBlackHoleMass = param_get_double(ps, "SeedBlackHoleMass");

        blackhole_params.BlackHoleFeedbackFactor = param_get_double(ps, "BlackHoleFeedbackFactor");
        blackhole_params.BlackHoleFeedbackRadius = param_get_double(ps, "BlackHoleFeedbackRadius");

        blackhole_params.BlackHoleFeedbackRadiusMaxPhys = param_get_double(ps, "BlackHoleFeedbackRadiusMaxPhys");

        blackhole_params.BlackHoleFeedbackMethod = param_get_enum(ps, "BlackHoleFeedbackMethod");
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

void
blackhole(ForceTree * tree)
{
    if(!All.BlackHoleOn)
        return;
    /* Do nothing if no black holes*/
    int64_t totbh;
    sumup_large_ints(1, &SlotsManager->info[5].size, &totbh);
    if(totbh == 0)
        return;
    int i;
    int Ntot_gas_swallowed, Ntot_BH_swallowed;

    walltime_measure("/Misc");
    TreeWalk tw_accretion[1] = {{0}};
    struct BHPriv priv[1];

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

    MPIU_Barrier(MPI_COMM_WORLD);
    message(0, "Beginning black-hole accretion\n");

    priv->N_sph_swallowed = priv->N_BH_swallowed = 0;

    /* Let's determine which particles may be swallowed and calculate total feedback weights */
    priv->SPH_SwallowID = mymalloc("SPH_SwallowID", SlotsManager->info[0].size * sizeof(MyIDType));
    if(ActiveParticle) {
        #pragma omp parallel for
        for(i = 0; i < NumActiveParticle; i ++) {
            int p_i = ActiveParticle[i];
            if(P[p_i].Type == 0)
                priv->SPH_SwallowID[P[p_i].PI] = -1;
        }
    }
    else {
        #pragma omp parallel for
        for(i = 0; i < SlotsManager->info[0].size; i ++) {
            priv->SPH_SwallowID[i] = -1;
        }
    }

    /* This allocates memory*/
    priv[0].spin = init_spinlocks(PartManager->NumPart);

    /* Computed in accretion, used in feedback*/
    priv->BH_FeedbackWeightSum = mymalloc("BH_FeedbackWeightSum", SlotsManager->info[5].size * sizeof(MyFloat));

    /* These are initialized in preprocess and used to reposition the BH in postprocess*/
    priv->MinPot = mymalloc("BH_MinPot", SlotsManager->info[5].size * sizeof(MyFloat));

    /* Local to this treewalk*/
    priv->BH_Entropy = mymalloc("BH_Entropy", SlotsManager->info[5].size * sizeof(MyFloat));
    priv->BH_SurroundingGasVel = (MyFloat (*) [3]) mymalloc("BH_SurroundVel", 3* SlotsManager->info[5].size * sizeof(priv->BH_SurroundingGasVel[0]));
    treewalk_run(tw_accretion, ActiveParticle, NumActiveParticle);
    myfree(priv->BH_SurroundingGasVel);
    myfree(priv->BH_Entropy);
    myfree(priv->MinPot);

    MPIU_Barrier(MPI_COMM_WORLD);
    message(0, "Start swallowing of gas particles and black holes\n");

    /* Allocate array for storing the feedback energy.*/
    SphP_scratch->Injected_BH_Energy = mymalloc2("Injected_BH_Energy", SlotsManager->info[0].size * sizeof(MyFloat));
    memset(SphP_scratch->Injected_BH_Energy, 0, SlotsManager->info[0].size * sizeof(MyFloat));
    /* Now do the swallowing of particles and dump feedback energy */

    /* Local to this treewalk*/
    priv->BH_accreted_Mass = mymalloc("BH_accretedmass", SlotsManager->info[5].size * sizeof(MyFloat));
    priv->BH_accreted_BHMass = mymalloc("BH_accreted_BHMass", SlotsManager->info[5].size * sizeof(MyFloat));
    treewalk_run(tw_feedback, ActiveParticle, NumActiveParticle);
    myfree(priv->BH_accreted_BHMass);
    myfree(priv->BH_accreted_Mass);

    myfree(priv->BH_FeedbackWeightSum );

    free_spinlocks(priv[0].spin);
    myfree(priv->SPH_SwallowID);

    MPI_Reduce(&priv->N_sph_swallowed, &Ntot_gas_swallowed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&priv->N_BH_swallowed, &Ntot_BH_swallowed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPIU_Barrier(MPI_COMM_WORLD);
    message(0, "Accretion done: %d gas particles swallowed, %d BH particles swallowed\n",
                Ntot_gas_swallowed, Ntot_BH_swallowed);

    double total_mdoteddington;
    double total_mass_holes, total_mdot;

    double Local_BH_mass = 0;
    double Local_BH_Mdot = 0;
    double Local_BH_Medd = 0;
    /* Compute total mass of black holes
     * present by summing contents of black hole array*/
    for(i = 0; i < SlotsManager->info[5].size; i ++)
    {
        Local_BH_mass += BhP[i].Mass;
        Local_BH_Mdot += BhP[i].Mdot;
        Local_BH_Medd += BhP[i].Mdot/BhP[i].Mass;
    }

    MPI_Reduce(&Local_BH_mass, &total_mass_holes, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Local_BH_Mdot, &total_mdot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Local_BH_Medd, &total_mdoteddington, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(ThisTask == 0)
    {
        /* convert to solar masses per yr */
        double mdot_in_msun_per_year =
            total_mdot * (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);

        total_mdoteddington *= 1.0 / ((4 * M_PI * GRAVITY * LIGHTCGS * PROTONMASS /
                    (0.1 * LIGHTCGS * LIGHTCGS * THOMPSON)) * All.UnitTime_in_s);

        fprintf(FdBlackHoles, "%g %d %g %g %g %g\n",
                All.Time, SlotsManager->info[5].size, total_mass_holes, total_mdot, mdot_in_msun_per_year, total_mdoteddington);
        fflush(FdBlackHoles);
    }
    walltime_measure("/BH");
}

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
    double rho_proper = rho * All.cf.a3inv;

    double soundspeed = blackhole_soundspeed(BH_GET_PRIV(tw)->BH_Entropy[PI], rho);

    /* Note: we take here a radiative efficiency of 0.1 for Eddington accretion */
    double meddington = (4 * M_PI * GRAVITY * LIGHTCGS * PROTONMASS / (0.1 * LIGHTCGS * LIGHTCGS * THOMPSON)) * BHP(i).Mass
        * All.UnitTime_in_s;

    double norm = pow((pow(soundspeed, 2) + pow(bhvel, 2)), 1.5);

    if(norm > 0)
        mdot = 4. * M_PI * blackhole_params.BlackHoleAccretionFactor * All.G * All.G *
            BHP(i).Mass * BHP(i).Mass * rho_proper / norm;
    else
        mdot = 0;

    if(blackhole_params.BlackHoleEddingtonFactor > 0.0 &&
        mdot > blackhole_params.BlackHoleEddingtonFactor * meddington) {
        mdot = blackhole_params.BlackHoleEddingtonFactor * meddington;
    }
    BHP(i).Mdot = mdot;

    double dtime = get_dloga_for_bin(P[i].TimeBin) / All.cf.hubble;

    BHP(i).Mass += BHP(i).Mdot * dtime;
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
    int PI = P[n].PI;
    if(BH_GET_PRIV(tw)->BH_accreted_Mass[PI] > 0)
    {
        P[n].Mass += BH_GET_PRIV(tw)->BH_accreted_Mass[PI];
        BHP(n).Mass += BH_GET_PRIV(tw)->BH_accreted_BHMass[PI];
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

        density_kernel_init(&iter->accretion_kernel, I->Hsml);
        density_kernel_init(&iter->feedback_kernel, hsearch);
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
    if(P[other].Type == 0 && SPHP(other).DelayTime > 0) return;

    /* Find the black hole potential minimum. */
    if(r2 < iter->accretion_kernel.HH)
    {
        if(P[other].Potential < O->BH_MinPot)
        {
            int d;
            O->BH_MinPot = P[other].Potential;
            for(d = 0; d < 3; d++) {
                O->BH_MinPotPos[d] = P[other].Pos[d];
            }
        }
    }

    /* Accretion / merger doesn't do self interaction */
    if(P[other].ID == I->ID) return;

    struct SpinLocks * spin = BH_GET_PRIV(lv->tw)->spin;

    if(P[other].Type == 5 && r2 < iter->accretion_kernel.HH)	/* we have a black hole merger */
    {
        /* We do not depend on the BH relative velocity.
         * Because the BHs are not dissipative, their relative velocities
         * can be large, causing clumps of BHs to build up
         * at the same position without merging. */

        lock_spinlock(other, spin);
        if(P[other].Swallowed) {
            /* Already marked, prefer to be swallowed by a bigger ID */
            if(BHP(other).SwallowID < I->ID) {
                BHP(other).SwallowID = I->ID;
            }
        } else {
            /* Unmarked, the BH with bigger ID swallows */
            if(P[other].ID < I->ID) {
                P[other].Swallowed = 1;
                BHP(other).SwallowID = I->ID;
            }
        }
        unlock_spinlock(other, spin);
    }

    if(P[other].Type == 0) {
        if(r2 < iter->accretion_kernel.HH) {
            double u = r * iter->accretion_kernel.Hinv;
            double wk = density_kernel_wk(&iter->accretion_kernel, u);
            float mass_j = P[other].Mass;

            /* FIXME: volume correction doesn't work on BH yet. */
            O->Rho += (mass_j * wk);

            O->SmoothedEntropy += (mass_j * wk * SphP_scratch->EntVarPred[P[other].PI]);
            O->GasVel[0] += (mass_j * wk * SphP_scratch->VelPred[3 * P[other].PI]);
            O->GasVel[1] += (mass_j * wk * SphP_scratch->VelPred[3 * P[other].PI+1]);
            O->GasVel[2] += (mass_j * wk * SphP_scratch->VelPred[3 * P[other].PI+2]);

            /* here we have a gas particle; check for swallowing */

            lock_spinlock(other, spin);
            /* compute accretion probability */
            double p, w;

            if((I->BH_Mass - I->Mass) > 0 && I->Density > 0)
                p = (I->BH_Mass - I->Mass) * wk / I->Density;
            else
                p = 0;

            /* compute random number, uniform in [0,1] */
            w = get_random_number(P[other].ID);
            if(w < p)
            {
                MyIDType * SPH_SwallowID = BH_GET_PRIV(lv->tw)->SPH_SwallowID;

                if(P[other].Swallowed) {
                    /* Already marked, prefer to be swallowed by a bigger ID */
                    if(SPH_SwallowID[P[other].PI] < I->ID) {
                        SPH_SwallowID[P[other].PI] = I->ID;
                    }
                } else {
                    /* Unmarked mark it */
                    P[other].Swallowed = 1;
                    SPH_SwallowID[P[other].PI] = I->ID;
                }
            }
            unlock_spinlock(other, spin);
        }

        if(r2 < iter->feedback_kernel.HH) {
            /* update the feedback weighting */
            double mass_j;
            if(HAS(blackhole_params.BlackHoleFeedbackMethod, BH_FEEDBACK_OPTTHIN)) {
                double redshift = 1./All.Time - 1;
                double nh0 = get_neutral_fraction_sfreff(other, redshift);
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

        density_kernel_init(&iter->feedback_kernel, hsearch);
        return;
    }

    int other = iter->base.other;
    double r2 = iter->base.r2;
    double r = iter->base.r;
    /* Exclude self interaction */

    if(P[other].ID == I->ID) return;

     /* BH does not accrete wind */
    if(P[other].Type == 0 && SPHP(other).DelayTime > 0) return;

    struct SpinLocks * spin = BH_GET_PRIV(lv->tw)->spin;

    if(P[other].Swallowed && P[other].Type == 5)	/* we have a black hole merger */
    {
        if(BHP(other).SwallowID != I->ID) return;

        lock_spinlock(other, spin);

        int d;
        for(d = 0; d < 3; d++)
            O->AccretedMomentum[d] += (P[other].Mass * P[other].Vel[d]);

        O->BH_CountProgs += BHP(other).CountProgs;

        /* We do not know how to notify the tree of mass changes. so
         * blindly enforce a mass conservation for now. */
        O->Mass += (P[other].Mass);
        O->BH_Mass += (BHP(other).Mass);
        P[other].Mass = 0;
        BHP(other).Mass = 0;

        slots_mark_garbage(other);
        BHP(other).Mdot = 0;
        unlock_spinlock(other, spin);

        #pragma omp atomic
        BH_GET_PRIV(lv->tw)->N_BH_swallowed++;

    }

    /* Dump feedback energy */
    if(!P[other].Swallowed && P[other].Type == 0) {
        if(r2 < iter->feedback_kernel.HH && P[other].Mass > 0) {
            if(I->FeedbackWeightSum > 0 && I->FeedbackEnergy > 0)
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

                lock_spinlock(other, spin);
                SphP_scratch->Injected_BH_Energy[P[other].PI] += (I->FeedbackEnergy * mass_j * wk / I->FeedbackWeightSum);
                unlock_spinlock(other, spin);
            }
        }
    }

    /* Swallowing a gas */
    if(P[other].Swallowed && P[other].Type == 0)
    {
        MyIDType * SPH_SwallowID = BH_GET_PRIV(lv->tw)->SPH_SwallowID;

        if(SPH_SwallowID[P[other].PI] != I->ID) return;

        lock_spinlock(other, spin);

        int d;
        for(d = 0; d < 3; d++)
            O->AccretedMomentum[d] += (P[other].Mass * P[other].Vel[d]);

        /* We do not know how to notify the tree of mass changes. so
         * blindly enforce a mass conservation for now. */
        O->Mass += (P[other].Mass);
        P[other].Mass = 0;

        slots_mark_garbage(other);
        unlock_spinlock(other, spin);

        #pragma omp atomic
        BH_GET_PRIV(lv->tw)->N_sph_swallowed++;
    }
}

static int
blackhole_accretion_haswork(int n, TreeWalk * tw)
{
    return (P[n].Type == 5) && (P[n].Mass > 0);
}

static void
blackhole_accretion_reduce(int place, TreeWalkResultBHAccretion * remote, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    int k;
    MyFloat * MinPot = BH_GET_PRIV(tw)->MinPot;
    int PI = P[place].PI;
    if(MinPot[PI] > remote->BH_MinPot)
    {
        BHP(place).JumpToMinPot = 1;
        MinPot[PI] = remote->BH_MinPot;
        for(k = 0; k < 3; k++) {
            /* Movement occurs in drift.c */
            BHP(place).MinPotPos[k] = remote->BH_MinPotPos[k];
        }
    }
    if (mode == 0 || BHP(place).minTimeBin > remote->BH_minTimeBin) {
        BHP(place).minTimeBin = remote->BH_minTimeBin;
    }

    TREEWALK_REDUCE(BHP(place).Density, remote->Rho);

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
    }

    int PI = P[place].PI;
    I->Hsml = P[place].Hsml;
    I->Mass = P[place].Mass;
    I->BH_Mass = BHP(place).Mass;
    I->Density = BHP(place).Density;
    I->Csnd = blackhole_soundspeed(BH_GET_PRIV(tw)->BH_Entropy[PI], BHP(place).Density);
    I->ID = P[place].ID;
}

static int
blackhole_feedback_haswork(int n, TreeWalk * tw)
{
    /*Black hole not being swallowed*/
    return (P[n].Type == 5) && (!P[n].Swallowed);
}

static void
blackhole_feedback_copy(int i, TreeWalkQueryBHFeedback * I, TreeWalk * tw)
{
    I->Hsml = P[i].Hsml;
    I->BH_Mass = BHP(i).Mass;
    I->ID = P[i].ID;
    int PI = P[i].PI;
    I->FeedbackWeightSum = BH_GET_PRIV(tw)->BH_FeedbackWeightSum[PI];

    double dtime = get_dloga_for_bin(P[i].TimeBin) / All.cf.hubble;

    I->FeedbackEnergy = blackhole_params.BlackHoleFeedbackFactor * 0.1 * BHP(i).Mdot * dtime *
                pow(LIGHTCGS / All.UnitVelocity_in_cm_per_s, 2);
}

static void
blackhole_feedback_reduce(int place, TreeWalkResultBHFeedback * remote, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    int PI = P[place].PI;
    TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_accreted_Mass[PI], remote->Mass);
    TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_accreted_BHMass[PI], remote->BH_Mass);
    TREEWALK_REDUCE(BHP(place).CountProgs, remote->BH_CountProgs);
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
    child = slots_convert(child, 5, -1);

    BHP(child).base.ID = P[child].ID;
    /* The accretion mass should always be the seed black hole mass,
     * irrespective of the gravitational mass of the particle.*/
    BHP(child).Mass = blackhole_params.SeedBlackHoleMass;
    BHP(child).Mdot = 0;
    BHP(child).FormationTime = All.Time;

    /* It is important to initialize MinPotPos to the current position of
     * a BH to avoid drifting to unknown locations (0,0,0) immediately
     * after the BH is created. */
    int j;
    for(j = 0; j < 3; j++) {
        BHP(child).MinPotPos[j] = P[child].Pos[j];
    }
    BHP(child).JumpToMinPot = 0;

    BHP(child).CountProgs = 1;
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

