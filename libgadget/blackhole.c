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
    double SeedBlackHoleMass;	/*!< Seed black hole mass */
    double BlackHoleEddingtonFactor;	/*! Factor above Eddington */
    int BlackHoleRepositionEnabled; /* If true, enable repositioning the BH to the potential minimum*/
} blackhole_params;

typedef struct {
    TreeWalkQueryBase base;
    MyFloat Density;
    MyFloat Hsml;
    MyFloat Mass;
    MyFloat BH_Mass;
    MyFloat Vel[3];
    MyIDType ID;
    /* The index of the original black hole,
     * for doing mergers*/
    int index;
} TreeWalkQueryBHAccretion;

typedef struct {
    TreeWalkResultBase base;
    MyFloat BH_MinPotPos[3];
    MyFloat BH_MinPotVel[3];
    MyFloat BH_MinPot;

    int BH_minTimeBin;
    MyFloat FeedbackWeightSum;

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
    MyFloat AccretedMomentum[3];
    MyFloat BH_Mass;
    int BH_CountProgs;
} TreeWalkResultBHFeedback;

typedef struct {
    TreeWalkNgbIterBase base;
    DensityKernel feedback_kernel;
} TreeWalkNgbIterBHFeedback;

/* struct to store details of particles
 * to be swallowed on this timestep.
 * The list of pairs is made and, to
 * avoid particles being swallowed twice,
 * is evaluated afterwards. */
typedef struct swallow_pair
{
    int Swallower;
    MyIDType SwallowerID;
    int Swallowed;
} swallow_pair;

void gadget_setup_thread_arrays_pair(swallow_pair * dest, swallow_pair * srcs[], size_t sizes[], size_t total_size, int narrays)
{
    int i;
    srcs[0] = dest;
    for(i=0; i < narrays; i++) {
        srcs[i] = dest + i * total_size;
        sizes[i] = 0;
    }
}

size_t gadget_compact_thread_arrays_pair(swallow_pair * dest, swallow_pair * srcs[], size_t sizes[], int narrays)
{
    int i;
    size_t asize = 0;
    for(i = 0; i < narrays; i++)
    {
        memmove(dest + asize, srcs[i], sizeof(swallow_pair) * sizes[i]);
        asize += sizes[i];
    }
    return asize;
}


struct BHPriv {
    /* Temporary array to store a list containing the swallowed particles and the swallowing black hole.*/
    swallow_pair * SwallowedList;
    size_t * swnqthr;
    swallow_pair ** swthrqueue;

    /* These are temporaries used in the accretion treewalk*/
    MyFloat * MinPot;
    MyFloat * BH_Entropy;
    MyFloat (*BH_SurroundingGasVel)[3];

    /* Temporary used in the feedback treewalk.*/
    MyFloat * Injected_BH_Energy;

    /* This is a temporary computed in the accretion treewalk and used
     * in the feedback treewalk*/
    MyFloat * BH_FeedbackWeightSum;

    /* Just a convenient storage place for the per-timestep data saved in collect_BH_info.*/
    MyFloat (*BH_accreted_momentum)[3];
    MyFloat * BH_accreted_Mass;
    MyFloat * BH_accreted_BHMass;


};
#define BH_GET_PRIV(tw) ((struct BHPriv *) (tw->priv))

struct BHinfo{

    MyIDType ID;
    MyFloat Mass;
    MyFloat Mdot;
    MyFloat Density;
    int minTimeBin;

    double  MinPotPos[3];
    MyFloat MinPot;
    MyFloat BH_Entropy;
    MyFloat BH_SurroundingGasVel[3];
    MyFloat BH_accreted_momentum[3];

    MyFloat BH_accreted_Mass;
    MyFloat BH_accreted_BHMass;
    MyFloat BH_FeedbackWeightSum;

    MyIDType SwallowID;
    /* Empty*/
    MyIDType SPH_SwallowID;

    int CountProgs;
    int Swallowed;

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
        blackhole_params.SeedBlackHoleMass = param_get_double(ps, "SeedBlackHoleMass");

        blackhole_params.BlackHoleFeedbackFactor = param_get_double(ps, "BlackHoleFeedbackFactor");
        blackhole_params.BlackHoleFeedbackRadius = param_get_double(ps, "BlackHoleFeedbackRadius");

        blackhole_params.BlackHoleFeedbackRadiusMaxPhys = param_get_double(ps, "BlackHoleFeedbackRadiusMaxPhys");

        blackhole_params.BlackHoleFeedbackMethod = param_get_enum(ps, "BlackHoleFeedbackMethod");
        blackhole_params.BlackHoleRepositionEnabled = param_get_int(ps, "BlackHoleRepositionEnabled");
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

static int
blackhole_feedback_haswork(int n, TreeWalk * tw);

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

    double temp = u_to_temp_fac * unew;

    if(temp > 5.0e8)
        unew = 5.0e8 / u_to_temp_fac;

    return unew;
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
        info.MinPotPos[0] = BHP(p_i).MinPotPos[0] - PartManager->CurrentParticleOffset[0];
        info.MinPotPos[1] = BHP(p_i).MinPotPos[1] - PartManager->CurrentParticleOffset[1];
        info.MinPotPos[2] = BHP(p_i).MinPotPos[2] - PartManager->CurrentParticleOffset[2];

        if(priv->MinPot) {
            info.MinPot = priv->MinPot[PI];
        }
        info.BH_Entropy = priv->BH_Entropy[PI];
        int k;
        for(k=0; k < 3; k++) {
            info.BH_SurroundingGasVel[k] = priv->BH_SurroundingGasVel[PI][k];
            info.BH_accreted_momentum[k] = priv->BH_accreted_momentum[PI][k];
        }

        info.BH_accreted_BHMass = priv->BH_accreted_BHMass[PI];
        info.BH_accreted_Mass = priv->BH_accreted_Mass[PI];
        info.BH_FeedbackWeightSum = priv->BH_FeedbackWeightSum[PI];

        info.SwallowID =  BHP(p_i).SwallowID;
        info.CountProgs = BHP(p_i).CountProgs;
        info.Swallowed =  P[p_i].Swallowed;

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


/* Swallows a single particle, marking it garbage if gas,
 * and as swallowed if a BH. This does no threading so
 * should not be called in parallel.*/
int
blackhole_swallow_particle(swallow_pair pair, MyFloat * AccretedMomentum, MyFloat * AccretedMass, MyFloat * AccretedBHMass)
{
    const int swallower = pair.Swallower;
    const int other = pair.Swallowed;

    /* Confirm no indexing error*/
    if(P[swallower].ID != pair.SwallowerID)
        endrun(3, "ID mismatch swallowing %d: expected %ld got %ld\n",
               other, pair.SwallowerID, P[swallower].ID);

    /* If we screwed up and the swallowing black hole was already swallowed.*/
    if(P[swallower].Swallowed)
        endrun(2, "BH %i already swallowed at time %g\n", swallower, BHP(swallower).SwallowTime);

    P[swallower].Mass += P[other].Mass;
    /* Conserve momentum during accretion*/
    int d;
    for(d = 0; d < 3; d++)
        AccretedMomentum[d] += (P[other].Mass * P[other].Vel[d]);
    *AccretedMass += P[other].Mass;

    /* we have a black hole merger! */
    if(P[other].Type == 5)
    {
        if(P[other].Swallowed)
            endrun(2, "BH %i already swallowed at time %g\n", other, BHP(other).SwallowTime);

        /* Leave the swallowed BH mass around
         * so we can work out mass at merger. */
        BHP(swallower).Mass += BHP(other).Mass;
        *AccretedBHMass += BHP(other).Mass;

        /* Swallow the particle*/
        /* Black hole particles which have been completely swallowed
         * (ie, their mass has been added to another particle) have Swallowed = 1.
         * These particles are ignored in future tree walks.
         * Swallowing is done so that the black hole with the largest ID is the swallower.*/
        BHP(other).SwallowID = P[swallower].ID;
        BHP(other).SwallowTime = All.Time;
        P[other].Swallowed = 1;
        BHP(swallower).CountProgs += BHP(other).CountProgs;
    }
    /* Swallowing something else (usually a gas). */
    /* Note that it will rarely happen that gas is swallowed by a BH which is itself swallowed.*/
    else
    {
        /* Enforce mass conservation even though particle is now garbage. */
        BHP(swallower).Mass += P[other].Mass;
        P[other].Mass = 0;
        slots_mark_garbage(other, PartManager, SlotsManager);
    }
    return P[other].Type;
}

static int
order_by_swallowed(const void * c1, const void * c2)
{
    const swallow_pair * p1 = (const swallow_pair *) c1;
    const swallow_pair * p2 = (const swallow_pair *) c2;
    if(p1->Swallowed < p2->Swallowed) return -1;
    if(p1->Swallowed > p2->Swallowed) return 1;
    return 0;
}

/* Removes duplicated swallowed particles from the SwallowList.
 * Returns new number of entries in deduplicated list.*/
int
deduplicate_swallowed_list(swallow_pair * SwallowedList, int nswallowed)
{
    int ndedupswallowed = 0;
    int i = 0;
        /* Sort the SwallowedList by swallowed particle.*/
    qsort_openmp(SwallowedList,nswallowed, sizeof(swallow_pair), order_by_swallowed);
    /* Ensure that each particle is only swallowed once by making
     * sure that particles are only present once in this list.*/
    /* Compute total momentum accretion over swallowers*/
    while(i<nswallowed-1)
    {
        int nxtacctd;
        /* Find any duplicate particles, swallowed by two different black holes*/
        const int gas1 = SwallowedList[i].Swallowed;
        for(nxtacctd = i+1; nxtacctd<nswallowed; nxtacctd++)
            if(gas1 != SwallowedList[nxtacctd].Swallowed)
                break;
        /* Now nxtacctd points to the first list entry with a different swallowed particle*/
        int k;
        MyIDType MaxID = SwallowedList[i].SwallowerID;
        int MaxIDentry = i;
        /* Find the largest swallower ID of these duplicates*/
        for(k = i; k < nxtacctd; k++) {
            if(SwallowedList[k].SwallowerID > MaxID) {
                MaxIDentry = k;
                MaxID = SwallowedList[i].SwallowerID;
            }
        }
        /* Copy the maximum ID entry over the first free entry in the deduplicated list.*/
        memmove(&SwallowedList[ndedupswallowed++], &SwallowedList[MaxIDentry],sizeof(swallow_pair));
        /* Increment the counter to the next block*/
        i=nxtacctd;
    }
    return ndedupswallowed;
}

/* Swallow the particles marked for swallowing and make them garbage.
 * Note this separates black hole mergers from other accretion, but does not
 * enforce that only gas is accreted. priv is used only to store the accreted momenta.*/
void
blackhole_swallow_particles(swallow_pair * SwallowedList, int nswallowed, struct BHPriv *priv)
{
    int i;
    /* Get a swallow list with duplicate entries removed*/
    nswallowed = deduplicate_swallowed_list(SwallowedList, nswallowed);
    /* Some counters*/
    int64_t N_BH_swallowed = 0;
    int64_t N_sph_swallowed = 0;
    MyFloat *AccretedMass = mymalloc("BH_accretedmass", nswallowed * sizeof(MyFloat));
    MyFloat *AccretedBHMass = mymalloc("BH_accretedbhmass", nswallowed * sizeof(MyFloat));
    MyFloat *AccretedMom = mymalloc("BH_accretemom", 3* nswallowed * sizeof(MyFloat));

    /* Do the swallowing*/
    for(i=0; i<nswallowed; i++)
    {
        /* Check that the swallowing particle is not itself swallowed
         * by someone else. If it is, skip this entry.*/
        const int swallower = SwallowedList[i].Swallower;
        int j;
        int skip = 0;
        for(j = 0; j< nswallowed; j++)
        {
            if(swallower == SwallowedList[j].Swallowed) {
                skip = 1;
                break;
            }
            if(swallower > SwallowedList[j].Swallowed)
                break;
        }
        if(skip)
            continue;

        /* Swallow the particle*/
        int type = blackhole_swallow_particle(SwallowedList[i], &AccretedMom[3*i], &AccretedMass[i], &AccretedBHMass[i]);
        if(type == 5)
            N_BH_swallowed++;
        else
            N_sph_swallowed++;
    }

    /* Compute total momentum accretion over swallowers*/
    for(i=nswallowed-1; i>=0; i--)
    {
        /* Find another pair with the same swallower before this one and sum its momenta*/
        int j;
        for(j=0; j<i; j++)
        {
            if(SwallowedList[i].Swallower == SwallowedList[j].Swallower)
            {
                AccretedMass[i] += AccretedMass[j];
                AccretedBHMass[i] += AccretedBHMass[j];
                AccretedMom[i] += AccretedMom[j];
                AccretedMass[j] = 0;
                AccretedBHMass[j] = 0;
                AccretedMom[j] = 0;
                break;
            }
        }
    }

    /* Enforce velocity conservation for the Swallowers.*/
    #pragma omp parallel for
    for(i = 0; i<nswallowed; i++)
    {
        int n = SwallowedList[i].Swallower;
        if(AccretedMass[i] > 0) {
            priv->BH_accreted_Mass[P[i].PI]+= AccretedMass[i];
            priv->BH_accreted_BHMass[P[i].PI]+= AccretedBHMass[i];

            /* velocity feedback due to accretion; momentum conservation.
             * This does nothing important with repositioning on.*/
            int k;
            for(k = 0; k < 3; k++) {
                P[n].Vel[k] = (P[n].Vel[k] * P[n].Mass + AccretedMom[3*i+k]) /
                        (P[n].Mass + AccretedMass[i]);
                priv->BH_accreted_momentum[P[i].PI][k] += AccretedMom[3*i+k];
            }
        }
    }
    myfree(AccretedMom);
    myfree(AccretedBHMass);
    myfree(AccretedMass);

    int64_t Ntot_BH_swallowed = 0;
    int64_t Ntot_gas_swallowed = 0;

    MPI_Reduce(&N_sph_swallowed, &Ntot_gas_swallowed, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&N_BH_swallowed, &Ntot_BH_swallowed, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);

    message(0, "Accretion done: %d gas particles swallowed, %d BH particles swallowed\n",
                Ntot_gas_swallowed, Ntot_BH_swallowed);
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
    TreeWalk tw_accretion[1] = {{0}};
    struct BHPriv priv[1] = {0};

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
    tw_feedback->postprocess = NULL;
    tw_feedback->reduce = NULL;
    tw_feedback->query_type_elsize = sizeof(TreeWalkQueryBHFeedback);
    tw_feedback->result_type_elsize = sizeof(TreeWalkResultBHFeedback);
    tw_feedback->tree = tree;
    tw_feedback->priv = priv;

    MPIU_Barrier(MPI_COMM_WORLD);
    message(0, "Beginning black-hole accretion\n");

    /* Computed in accretion, used in feedback*/
    priv->BH_FeedbackWeightSum = mymalloc("BH_FeedbackWeightSum", SlotsManager->info[5].size * sizeof(MyFloat));

    /* These are initialized in preprocess and used to reposition the BH in postprocess*/
    priv->MinPot = mymalloc("BH_MinPot", SlotsManager->info[5].size * sizeof(MyFloat));

    /* Local to this treewalk*/
    priv->BH_Entropy = mymalloc("BH_Entropy", SlotsManager->info[5].size * sizeof(MyFloat));
    priv->BH_SurroundingGasVel = (MyFloat (*) [3]) mymalloc("BH_SurroundVel", 3* SlotsManager->info[5].size * sizeof(priv->BH_SurroundingGasVel[0]));

    const int nthread = omp_get_max_threads();
    priv->swnqthr = ta_malloc("nqthr", size_t, nthread);
    priv->swthrqueue = ta_malloc("thrqueue", swallow_pair *, nthread);

    /* Let's determine which particles may be swallowed and calculate total feedback weights */
    priv->SwallowedList= mymalloc("SPH_SwallowList", SlotsManager->info[0].size * sizeof(swallow_pair) * nthread);
    /* Can't use the default gadget_setup_thread_arrays function because queue is not int type*/
    gadget_setup_thread_arrays_pair(priv->SwallowedList, priv->swthrqueue, priv->swnqthr, SlotsManager->info[0].size, nthread);

    /* Compute the feedback weightings and mark particles for accretion.*/
    treewalk_run(tw_accretion, act->ActiveParticle, act->NumActiveParticle);

    /* Create the list of swallowed gas particles*/
    int nswallowed = gadget_compact_thread_arrays_pair(priv->SwallowedList, priv->swthrqueue, priv->swnqthr, nthread);
    ta_free(priv->swthrqueue);
    ta_free(priv->swnqthr);

    MPIU_Barrier(MPI_COMM_WORLD);
    message(0, "Start swallowing of gas particles and black holes\n");

    /* Now do the swallowing of particles and dump feedback energy */

    /* Allocate array for storing the feedback energy.*/
    priv->Injected_BH_Energy = mymalloc2("Injected_BH_Energy", SlotsManager->info[0].size * sizeof(MyFloat));
    memset(priv->Injected_BH_Energy, 0, SlotsManager->info[0].size * sizeof(MyFloat));

    treewalk_run(tw_feedback, act->ActiveParticle, act->NumActiveParticle);

    const double a3inv = 1./(All.Time * All.Time * All.Time);
    /* This function changes the entropy of the particle due to the BH heating. */
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)
    {
        if(P[i].Type == 0 && priv->Injected_BH_Energy[P[i].PI] > 0)
        {
            /* Set a flag for star-forming particles:
             * we want these to cool to the EEQOS via
             * tcool rather than trelax.*/
            if(sfreff_on_eeqos(&SPHP(i), a3inv))
                P[i].BHHeated = 1;
            const double enttou = pow(SPH_EOMDensity(i) * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1;
            double uold = SPHP(i).Entropy * enttou;
            uold = add_injected_BH_energy(uold, priv->Injected_BH_Energy[P[i].PI], P[i].Mass);
            SPHP(i).Entropy = uold / enttou;
        }
    }

    blackhole_swallow_particles(priv->SwallowedList, nswallowed, priv);

    if(FdBlackholeDetails){
        collect_BH_info(act->ActiveParticle, act->NumActiveParticle, priv, FdBlackholeDetails);
    }

    myfree(priv->Injected_BH_Energy);
    myfree(priv->SwallowedList);
    myfree(priv->BH_SurroundingGasVel);
    myfree(priv->BH_Entropy);
    myfree(priv->MinPot);
    myfree(priv->BH_FeedbackWeightSum);

    int total_bh;
    double total_mdoteddington;
    double total_mass_holes, total_mdot;

    double Local_BH_mass = 0;
    double Local_BH_Mdot = 0;
    double Local_BH_Medd = 0;
    int Local_BH_num = 0;
    /* Compute total mass of black holes
     * present by summing contents of black hole array*/
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

    /* Only process feedback or mergers with gas or BH*/
    if(!(P[other].Type == 5 || P[other].Type == 0))
        return;

    /* Possible accretion*/
    if(r2 < iter->accretion_kernel.HH)
    {
        double p = 1;
        /* Compute BH thermo quantities*/
        if(P[other].Type == 0) {
            double u = r * iter->accretion_kernel.Hinv;
            double wk = density_kernel_wk(&iter->accretion_kernel, u);
            float mass_j = P[other].Mass;

            O->SmoothedEntropy += (mass_j * wk * SPHP(other).Entropy);
            O->GasVel[0] += (mass_j * wk * P[other].Vel[0]);
            O->GasVel[1] += (mass_j * wk * P[other].Vel[1]);
            O->GasVel[2] += (mass_j * wk * P[other].Vel[2]);
            /* compute accretion probability */
            p = 0;

            /* This is an averaged Mdot, because Mdot increases BH_Mass but not Mass.
             * So if the total accretion is significantly above the dynamical mass,
             * a particle is swallowed. Note that if a large number of black holes
             * are swallowed with a BH mass less than the dynamical mass,
             * this will not increase and gas accretion will be suppressed.
             * To avoid this, ensure that the BH seed mass is not much less
             * than the dynamical mass. */
            if((I->BH_Mass - I->Mass) > 0 && I->Density > 0)
                p = (I->BH_Mass - I->Mass) * wk / I->Density;
        }
        /* Black holes always merge*/
        else if(P[other].Type == 5)
            p = 1;

        /* compute random number, uniform in [0,1] */
        const double w = get_random_number(P[other].ID);
        /* we have a merger */
        if(w < p) {

            /* We do not depend on the BH relative velocity.
            * Because the BHs are not dissipative, their relative velocities
            * can be large, causing clumps of BHs to build up
            * at the same position without merging. */
            int tid = omp_get_thread_num();
            int swtid = BH_GET_PRIV(lv->tw)->swnqthr[tid];
            swallow_pair *pair = BH_GET_PRIV(lv->tw)->swthrqueue[swtid++];
            pair->Swallowed = other;
            pair->Swallower = I->index;
            pair->SwallowerID = I->ID;
        }
    }

    /* Compute feedback kernel*/
    if(P[other].Type == 0 && r2 < iter->feedback_kernel.HH) {
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

     /* BH does not feedback on wind */
    if(winds_is_particle_decoupled(other))
        return;

    /* Dump feedback energy */
    if(P[other].Type == 0) {
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

                double * iBHPI = &BH_GET_PRIV(lv->tw)->Injected_BH_Energy[P[other].PI];
                const double injected_BH = I->FeedbackEnergy * mass_j * wk / I->FeedbackWeightSum;
                #pragma omp atomic update
                (*iBHPI) += injected_BH;
            }
        }
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
    }
    I->Hsml = P[place].Hsml;
    I->Mass = P[place].Mass;
    I->BH_Mass = BHP(place).Mass;
    I->Density = BHP(place).Density;
    I->ID = P[place].ID;
    I->index = place;
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
    int PI = P[i].PI;
    I->FeedbackWeightSum = BH_GET_PRIV(tw)->BH_FeedbackWeightSum[PI];

    double dtime = get_dloga_for_bin(P[i].TimeBin) / All.cf.hubble;

    I->FeedbackEnergy = blackhole_params.BlackHoleFeedbackFactor * 0.1 * BHP(i).Mdot * dtime *
                pow(LIGHTCGS / All.UnitVelocity_in_cm_per_s, 2);
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
    BHP(child).Mass = blackhole_params.SeedBlackHoleMass;
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
