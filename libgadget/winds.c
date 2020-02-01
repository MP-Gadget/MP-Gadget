/*Prototypes and structures for the wind model*/

#include <math.h>
#include <string.h>
#include <omp.h>
#include "winds.h"
#include "physconst.h"
#include "treewalk.h"
#include "slotsmanager.h"
#include "timebinmgr.h"
#include "walltime.h"
#include "allvars.h"

/*Parameters of the wind model*/
static struct WindParams
{
    enum WindModel WindModel;  /*!< Which wind model is in use? */
    double WindFreeTravelLength;
    double WindFreeTravelDensFac;
    /*Density threshold at which to recouple wind particles.*/
    double WindFreeTravelDensThresh;
    /* used in VS08 and SH03*/
    double WindEfficiency;
    double WindSpeed;
    double WindEnergyFraction;
    /* used in OFJT10*/
    double WindSigma0;
    double WindSpeedFactor;
} wind_params;


typedef struct {
    TreeWalkQueryBase base;
    double Sfr;
    double Dt;
    double Mass;
    double Hsml;
    double TotalWeight;
    double DMRadius;
    double Vdisp;
} TreeWalkQueryWind;

typedef struct {
    TreeWalkResultBase base;
    double TotalWeight;
    double V1sum[3];
    double V2sum;
    int Ngb;
} TreeWalkResultWind;

typedef struct {
    TreeWalkNgbIterBase base;
} TreeWalkNgbIterWind;

/*Set the parameters of the wind module.
 ofjt10 is Okamoto, Frenk, Jenkins and Theuns 2010 https://arxiv.org/abs/0909.0265
 VS08 is Dalla Vecchia & Schaye 2008 https://arxiv.org/abs/0801.2770
 SH03 is Springel & Hernquist 2003 https://arxiv.org/abs/astro-ph/0206395*/
void set_winds_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        /*Wind model parameters*/
        wind_params.WindModel = param_get_enum(ps, "WindModel");
        /* The following two are for VS08 and SH03*/
        wind_params.WindEfficiency = param_get_double(ps, "WindEfficiency");
        wind_params.WindEnergyFraction = param_get_double(ps, "WindEnergyFraction");

        /* The following two are for OFJT10*/
        wind_params.WindSigma0 = param_get_double(ps, "WindSigma0");
        wind_params.WindSpeedFactor = param_get_double(ps, "WindSpeedFactor");

        wind_params.WindFreeTravelLength = param_get_double(ps, "WindFreeTravelLength");
        wind_params.WindFreeTravelDensFac = param_get_double(ps, "WindFreeTravelDensFac");
    }
    MPI_Bcast(&wind_params, sizeof(struct WindParams), MPI_BYTE, 0, MPI_COMM_WORLD);
}

void
init_winds(double FactorSN, double EgySpecSN, double PhysDensThresh)
{
    wind_params.WindSpeed = sqrt(2 * wind_params.WindEnergyFraction * FactorSN * EgySpecSN / (1 - FactorSN));

    wind_params.WindFreeTravelDensThresh = wind_params.WindFreeTravelDensFac * PhysDensThresh;
    if(HAS(wind_params.WindModel, WIND_FIXED_EFFICIENCY)) {
        wind_params.WindSpeed /= sqrt(wind_params.WindEfficiency);
        message(0, "Windspeed: %g\n", wind_params.WindSpeed);
    } else {
        message(0, "Reference Windspeed: %g\n", wind_params.WindSigma0 * wind_params.WindSpeedFactor);
    }

}

int
winds_is_particle_decoupled(int i)
{
    if(HAS(wind_params.WindModel, WIND_DECOUPLE_SPH)
        && P[i].Type == 0 && SPHP(i).DelayTime > 0)
            return 1;
    return 0;
}

void
winds_decoupled_hydro(int i, double atime)
{
    int k;
    for(k = 0; k < 3; k++)
        SPHP(i).HydroAccel[k] = 0;

    SPHP(i).DtEntropy = 0;

    double windspeed = wind_params.WindSpeed * atime;
    const double fac_mu = pow(atime, 3 * (GAMMA - 1) / 2) / atime;
    windspeed *= fac_mu;
    double hsml_c = cbrt(wind_params.WindFreeTravelDensThresh /SPHP(i).Density) * atime;
    SPHP(i).MaxSignalVel = hsml_c * DMAX((2 * windspeed), SPHP(i).MaxSignalVel);
}

static void
sfr_wind_reduce_weight(int place, TreeWalkResultWind * remote, enum TreeWalkReduceMode mode, TreeWalk * tw);

static void
sfr_wind_copy(int place, TreeWalkQueryWind * input, TreeWalk * tw);

static void
sfr_wind_weight_postprocess(const int i, TreeWalk * tw);

static void
sfr_wind_weight_ngbiter(TreeWalkQueryWind * I,
        TreeWalkResultWind * O,
        TreeWalkNgbIterWind * iter,
        LocalTreeWalk * lv);

static void
sfr_wind_feedback_ngbiter(TreeWalkQueryWind * I,
        TreeWalkResultWind * O,
        TreeWalkNgbIterWind * iter,
        LocalTreeWalk * lv);

struct winddata {
    double DMRadius;
    double Left;
    double Right;
    double TotalWeight;
    union {
        double Vdisp;
        double V2sum;
    };
    double V1sum[3];
    int Ngb;
};

struct WindPriv {
    size_t * NPLeft;
    int** NPRedo;
    struct winddata * Winddata;
    struct SpinLocks * spin;
};

#define WIND_GET_PRIV(tw) ((struct WindPriv *) (tw->priv))
#define WINDP(i, tw) WIND_GET_PRIV(tw)->Winddata[P[i].PI]


/*Do a treewalk for the wind model. This only changes newly created star particles.*/
void
winds_and_feedback(int * NewStars, int NumNewStars, ForceTree * tree)
{
    /*The subgrid model does nothing here*/
    if(HAS(wind_params.WindModel, WIND_SUBGRID))
        return;

    if(!MPIU_Any(NumNewStars > 0, MPI_COMM_WORLD))
        return;

    TreeWalk tw[1] = {{0}};

    int NumThreads = omp_get_max_threads();
    tw->ev_label = "SFR_WIND";
    tw->fill = (TreeWalkFillQueryFunction) sfr_wind_copy;
    tw->reduce = (TreeWalkReduceResultFunction) sfr_wind_reduce_weight;
    tw->query_type_elsize = sizeof(TreeWalkQueryWind);
    tw->result_type_elsize = sizeof(TreeWalkResultWind);
    tw->tree = tree;

    /* sum the total weight of surrounding gas */
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterWind);
    tw->ngbiter = (TreeWalkNgbIterFunction) sfr_wind_weight_ngbiter;

    tw->haswork = NULL;
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw->postprocess = (TreeWalkProcessFunction) sfr_wind_weight_postprocess;
    struct WindPriv priv[1];
    tw->priv = priv;

    int64_t totalleft = 0;
    sumup_large_ints(1, &NumNewStars, &totalleft);
    priv->NPLeft = ta_malloc("NPLeft", size_t, NumThreads);
    priv->NPRedo = ta_malloc("NPRedo", int *, NumThreads);
    priv->Winddata = (struct winddata * ) mymalloc("WindExtraData", SlotsManager->info[4].size * sizeof(struct winddata));

    int i;
    /*Initialise the WINDP array*/
    #pragma omp parallel for
    for (i = 0; i < NumNewStars; i++) {
        int n = NewStars[i];
        priv->Winddata[n].DMRadius = 2 * P[n].Hsml;
        priv->Winddata[n].Left = 0;
        priv->Winddata[n].Right = tree->BoxSize;
    }

    int alloc_high = 0;
    int * ReDoQueue = NewStars;
    int size = NumNewStars;
    int iter=0;

    /* we will repeat the whole thing for those particles where we didn't find enough neighbours */
    do {
        int * CurQueue = ReDoQueue;
        int tsize = NumNewStars / NumThreads + 2;
        /* The ReDoQueue swaps between high and low allocations so we can have two allocated alternately*/
        if(!alloc_high) {
            ReDoQueue = (int *) mymalloc2("redoqueue", tsize * sizeof(int) * NumThreads);
            alloc_high = 1;
        }
        else {
            ReDoQueue = (int *) mymalloc("redoqueue", tsize * sizeof(int) * NumThreads);
            alloc_high = 0;
        }
        gadget_setup_thread_arrays(ReDoQueue, WIND_GET_PRIV(tw)->NPRedo, WIND_GET_PRIV(tw)->NPLeft, tsize, NumThreads);

        memset(WIND_GET_PRIV(tw)->NPLeft, 0, sizeof(int)*NumThreads);
        treewalk_run(tw, CurQueue, size);

        tw->haswork = NULL;
        /* Now done with the current queue*/
        if(iter > 0)
            myfree(CurQueue);

        /* Set up the next queue*/
        size = gadget_compact_thread_arrays(ReDoQueue, WIND_GET_PRIV(tw)->NPRedo, WIND_GET_PRIV(tw)->NPLeft, NumThreads);

        sumup_large_ints(1, &size, &totalleft);
        if(totalleft == 0){
            myfree(ReDoQueue);
            break;
        }

        /*Shrink memory*/
        ReDoQueue = myrealloc(ReDoQueue, sizeof(int) * size);

        iter++;
        message(0, "Star DM iteration Total left = %ld\n", totalleft);
    } while(1);

    ta_free(priv->NPRedo);
    ta_free(priv->NPLeft);

    /* Then run feedback */
    tw->haswork = NULL;
    tw->ngbiter = (TreeWalkNgbIterFunction) sfr_wind_feedback_ngbiter;
    tw->postprocess = NULL;
    tw->reduce = NULL;

    priv[0].spin = init_spinlocks(PartManager->NumPart);
    treewalk_run(tw, NewStars, NumNewStars);
    free_spinlocks(priv[0].spin);
    myfree(priv->Winddata);
    walltime_measure("/Cooling/Wind");
}

/*Evolve a wind particle, reducing its DelayTime*/
void
winds_evolve(int i, double a3inv, double hubble)
{
    /*Remove a wind particle from the delay mode if the (physical) density has dropped sufficiently.*/
    if(SPHP(i).DelayTime > 0 && SPHP(i).Density * a3inv < wind_params.WindFreeTravelDensThresh) {
        SPHP(i).DelayTime = 0;
    }
    /*Reduce the time until the particle can form stars again by the current timestep*/
    if(SPHP(i).DelayTime > 0) {
        const double dloga = get_dloga_for_bin(P[i].TimeBin);
        /*  the proper time duration of the step */
        const double dtime = dloga / hubble;
        SPHP(i).DelayTime = DMAX(SPHP(i).DelayTime - dtime, 0);
    }
}

static void
sfr_wind_weight_postprocess(const int i, TreeWalk * tw)
{
    int done = 0;
    if(P[i].Type != 4)
        endrun(23, "Wind called on something not a star particle: (i=%d, t=%d, id = %ld)\n", i, P[i].Type, P[i].ID);
    int diff = WINDP(i, tw).Ngb - 40;
    if(diff < -2) {
        /* too few */
        WINDP(i, tw).Left = WINDP(i, tw).DMRadius;
    } else if(diff > 2) {
        /* too many */
        WINDP(i, tw).Right = WINDP(i, tw).DMRadius;
    } else {
        done = 1;
    }
    if(WINDP(i, tw).Right < tw->tree->BoxSize * 0.99 && WINDP(i, tw).Left > 0) {
        WINDP(i, tw).DMRadius = pow(0.5 * pow(WINDP(i, tw).Left, 3) + pow(WINDP(i, tw).Right, 3), 1./3);
    } else {
        WINDP(i, tw).DMRadius *= 1.3;
    }

    if(done) {
        double vdisp = WINDP(i, tw).V2sum / WINDP(i, tw).Ngb;
        int d;
        for(d = 0; d < 3; d ++) {
            vdisp -= pow(WINDP(i, tw).V1sum[d] / WINDP(i, tw).Ngb,2);
        }
        WINDP(i, tw).Vdisp = sqrt(vdisp / 3);
    } else {
        /* More work needed: add this particle to the redo queue*/
        int tid = omp_get_thread_num();
        WIND_GET_PRIV(tw)->NPLeft[tid] ++;
        WIND_GET_PRIV(tw)->NPRedo[tid][WIND_GET_PRIV(tw)->NPLeft[tid]] = i;
    }
}

static void
sfr_wind_reduce_weight(int place, TreeWalkResultWind * O, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    TREEWALK_REDUCE(WINDP(place, tw).TotalWeight, O->TotalWeight);
    int k;
    for(k = 0; k < 3; k ++) {
        TREEWALK_REDUCE(WINDP(place, tw).V1sum[k], O->V1sum[k]);
    }
    TREEWALK_REDUCE(WINDP(place, tw).V2sum, O->V2sum);
    TREEWALK_REDUCE(WINDP(place, tw).Ngb, O->Ngb);
    /*
    message(1, "Reduce ID=%ld, NGB=%d TotalWeight=%g V2sum=%g V1sum=%g %g %g\n",
            P[place].ID, O->Ngb, O->TotalWeight, O->V2sum,
            O->V1sum[0], O->V1sum[1], O->V1sum[2]);
            */
}

static void
sfr_wind_copy(int place, TreeWalkQueryWind * input, TreeWalk * tw)
{
    double dtime = get_dloga_for_bin(P[place].TimeBin) / All.cf.hubble;
    input->Dt = dtime;
    input->Mass = P[place].Mass;
    input->Hsml = P[place].Hsml;
    input->TotalWeight = WINDP(place, tw).TotalWeight;

    input->DMRadius = WINDP(place, tw).DMRadius;
    input->Vdisp = WINDP(place, tw).Vdisp;
}

static void
sfr_wind_weight_ngbiter(TreeWalkQueryWind * I,
        TreeWalkResultWind * O,
        TreeWalkNgbIterWind * iter,
        LocalTreeWalk * lv)
{
    /* this evaluator walks the tree and sums the total mass of surrounding gas
     * particles as described in VS08. */
    /* it also calculates the DM dispersion of the nearest 40 DM particles */
    if(iter->base.other == -1) {
        double hsearch = DMAX(I->Hsml, I->DMRadius);
        iter->base.Hsml = hsearch;
        iter->base.mask = 1 + 2; /* gas and dm */
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        return;
    }

    int other = iter->base.other;
    double r = iter->base.r;
    double * dist = iter->base.dist;

    if(P[other].Type == 0) {
        if(r > I->Hsml) return;
        /* NOTE: think twice if we want a symmetric tree walk when wk is used. */
        //double wk = density_kernel_wk(&kernel, r);
        double wk = 1.0;
        O->TotalWeight += wk * P[other].Mass;
    }

    if(P[other].Type == 1) {
        if(r > I->DMRadius) return;
        O->Ngb ++;
        int d;
        for(d = 0; d < 3; d ++) {
            /* Add hubble flow; FIXME: this shall be a function, and the direction looks wrong too. */
            double vel = P[other].Vel[d] + All.cf.hubble * All.cf.a * All.cf.a * dist[d];
            O->V1sum[d] += vel;
            O->V2sum += vel * vel;
        }
    }

    /*
    message(1, "ThisTask = %d %ld ngb=%d NGB=%d TotalWeight=%g V2sum=%g V1sum=%g %g %g\n",
    ThisTask, I->ID, numngb, O->Ngb, O->TotalWeight, O->V2sum,
    O->V1sum[0], O->V1sum[1], O->V1sum[2]);
    */
}


static int
get_wind_dir(int i, double dir[3]) {
    /* v and vmean are in internal units (km/s *a ), not km/s !*/
    /* returns 0 if particle i is converted to wind. */
    // message(1, "%ld Making ID=%ld (%g %g %g) to wind with v= %g\n", ID, P[i].ID, P[i].Pos[0], P[i].Pos[1], P[i].Pos[2], v);
    /* ok, make the particle go into the wind */
    double theta = acos(2 * get_random_number(P[i].ID + 3) - 1);
    double phi = 2 * M_PI * get_random_number(P[i].ID + 4);

    dir[0] = sin(theta) * cos(phi);
    dir[1] = sin(theta) * sin(phi);
    dir[2] = cos(theta);
    return 0;
}

static void
sfr_wind_feedback_ngbiter(TreeWalkQueryWind * I,
        TreeWalkResultWind * O,
        TreeWalkNgbIterWind * iter,
        LocalTreeWalk * lv)
{

    /* this evaluator walks the tree and blows wind. */

    if(iter->base.other == -1) {
        iter->base.mask = 1;
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        iter->base.Hsml = I->Hsml;
        return;
    }
    int other = iter->base.other;
    double r = iter->base.r;

    /* this is radius cut is redundant because the tree walk is asymmetric
     * we may want to use fancier weighting that requires symmetric in the future. */
    if(r > I->Hsml) return;

    double windeff=0;
    double v=0;
    if(HAS(wind_params.WindModel, WIND_FIXED_EFFICIENCY)) {
        windeff = wind_params.WindEfficiency;
        v = wind_params.WindSpeed * All.cf.a;
    } else if(HAS(wind_params.WindModel, WIND_USE_HALO)) {
        windeff = 1.0 / (I->Vdisp / All.cf.a / wind_params.WindSigma0);
        windeff *= windeff;
        v = wind_params.WindSpeedFactor * I->Vdisp;
    } else {
        endrun(1, "WindModel = 0x%X is strange. This shall not happen.\n", wind_params.WindModel);
    }

    double p = windeff * I->Mass / I->TotalWeight;
    double random = get_random_number(I->base.ID + P[other].ID);

    if (random < p) {
        double dir[3];
        get_wind_dir(other, dir);
        /* in this case the particle is already locked by the tree walker */
        /* we may want to add another lock to avoid this. */
        if(P[other].ID != I->base.ID)
            lock_spinlock(other, WIND_GET_PRIV(lv->tw)->spin);
        int j;
        for(j = 0; j < 3; j++)
        {
            P[other].Vel[j] += v * dir[j];
        }
        /* If the particle is already a wind, just use the largest DelayTime, and still add wind energy.
         * If we ignore wind particles, as was done before, we end up giving each particle the velocity dispersion
         * associated with the first particle that hits it, which is very timing dependent in threaded environments. */
        SPHP(other).DelayTime = DMAX(wind_params.WindFreeTravelLength / (v / All.cf.a), SPHP(other).DelayTime);
        if(P[other].ID != I->base.ID)
            unlock_spinlock(other, WIND_GET_PRIV(lv->tw)->spin);
    }
}

int
winds_make_after_sf(int i, double sm, double atime)
{
    if(!HAS(wind_params.WindModel, WIND_SUBGRID))
        return 0;
    /* Here comes the Springel Hernquist 03 wind model */
    /* Notice that this is the mass of the gas particle after forking a star, 1/GENERATIONS
        * what it was before.*/
    double pw = wind_params.WindEfficiency * sm / P[i].Mass;
    double prob = 1 - exp(-pw);
    if(get_random_number(P[i].ID + 2) < prob) {
        double dir[3];
        get_wind_dir(i, dir);
        int j;
        for(j = 0; j < 3; j++)
        {
            P[i].Vel[j] += wind_params.WindSpeed * atime * dir[j];
        }

        SPHP(i).DelayTime = wind_params.WindFreeTravelLength / (wind_params.WindSpeed * atime / All.cf.a);
    }

    return 0;
}
