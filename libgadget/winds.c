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
#include "density.h"
#include "hydra.h"
#include "sfr_eff.h"

/*Parameters of the wind model*/
static struct WindParams
{
    enum WindModel WindModel;  /*!< Which wind model is in use? */
    double WindFreeTravelLength;
    double WindFreeTravelDensFac;
    /*Density threshold at which to recouple wind particles.*/
    double WindFreeTravelDensThresh;
    /* Maximum time in internal time units to allow the wind to be free-streaming.*/
    double MaxWindFreeTravelTime;
    /* used in VS08 and SH03*/
    double WindEfficiency;
    double WindSpeed;
    double WindEnergyFraction;
    /* used in OFJT10*/
    double WindSigma0;
    double WindSpeedFactor;
    /* Minimum wind velocity for kicked particles, in internal velocity units*/
    double MinWindVelocity;
    /* Fraction of wind energy in thermal energy*/
    double WindThermalFactor;
} wind_params;

#define NWINDHSML 5 /* Number of densities to evaluate for wind weight ngbiter*/
#define NUMDMNGB 40 /*Number of DM ngb to evaluate vel dispersion */
#define MAXDMDEVIATION 2

typedef struct {
    TreeWalkQueryBase base;
    MyIDType ID;
    double Mass;
    double Hsml;
    double TotalWeight;
    double Vdisp;
} TreeWalkQueryWind;

typedef struct {
    TreeWalkResultBase base;
    double TotalWeight;
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
        wind_params.WindModel = (enum WindModel) param_get_enum(ps, "WindModel");
        /* The following two are for VS08 and SH03*/
        wind_params.WindEfficiency = param_get_double(ps, "WindEfficiency");
        wind_params.WindEnergyFraction = param_get_double(ps, "WindEnergyFraction");

        /* The following two are for OFJT10*/
        wind_params.WindSigma0 = param_get_double(ps, "WindSigma0");
        wind_params.WindSpeedFactor = param_get_double(ps, "WindSpeedFactor");

        wind_params.WindThermalFactor = param_get_double(ps, "WindThermalFactor");
        wind_params.MinWindVelocity = param_get_double(ps, "MinWindVelocity");
        wind_params.MaxWindFreeTravelTime = param_get_double(ps, "MaxWindFreeTravelTime");
        wind_params.WindFreeTravelLength = param_get_double(ps, "WindFreeTravelLength");
        wind_params.WindFreeTravelDensFac = param_get_double(ps, "WindFreeTravelDensFac");
    }
    MPI_Bcast(&wind_params, sizeof(struct WindParams), MPI_BYTE, 0, MPI_COMM_WORLD);
}

void
init_winds(double FactorSN, double EgySpecSN, double PhysDensThresh, double UnitTime_in_s)
{
    wind_params.WindSpeed = sqrt(2 * wind_params.WindEnergyFraction * FactorSN * EgySpecSN / (1 - FactorSN));
    /* Convert wind free travel time from Myr to internal units*/
    wind_params.MaxWindFreeTravelTime = wind_params.MaxWindFreeTravelTime * SEC_PER_MEGAYEAR / UnitTime_in_s;
    wind_params.WindFreeTravelDensThresh = wind_params.WindFreeTravelDensFac * PhysDensThresh;
    if(HAS(wind_params.WindModel, WIND_FIXED_EFFICIENCY)) {
        wind_params.WindSpeed /= sqrt(wind_params.WindEfficiency);
        message(0, "Windspeed: %g MaxDelay %g\n", wind_params.WindSpeed, wind_params.MaxWindFreeTravelTime);
    } else if(HAS(wind_params.WindModel, WIND_USE_HALO)) {
        message(0, "Reference Windspeed: %g, MaxDelay %g\n", wind_params.WindSigma0 * wind_params.WindSpeedFactor, wind_params.MaxWindFreeTravelTime);
    } else {
        /* Check for undefined wind models*/
        endrun(1, "WindModel = 0x%X is strange. This shall not happen.\n", wind_params.WindModel);
    }
}

int
winds_are_subgrid(void)
{
    if(HAS(wind_params.WindModel, WIND_SUBGRID))
        return 1;
    else
        return 0;
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
wind_do_kick(int other, double vel, double therm, double atime);

static int
get_wind_dir(int i, double dir[3]);

static void
get_wind_params(double * vel, double * windeff, double * utherm, const double vdisp, const double time);

static void
sfr_wind_reduce_weight(int place, TreeWalkResultWind * remote, enum TreeWalkReduceMode mode, TreeWalk * tw);

static void
sfr_wind_copy(int place, TreeWalkQueryWind * input, TreeWalk * tw);

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

/* Returns 1 if the winds ever decouple, 0 otherwise*/
int winds_ever_decouple(void)
{
    if(wind_params.MaxWindFreeTravelTime > 0)
        return 1;
    else
        return 0;
}

/* Structure to store a potential kick
 * to a gas particle from a newly formed star.
 * We add a queue of these and resolve them
 * after the treewalk. Note this star may
 * be on another processor.*/
struct StarKick
{
    /* Index of the kicked particle*/
    int part_index;
    /* Distance to the star. The closest star does the kick.*/
    double StarDistance;
    /* Star ID, for resolving ties.*/
    MyIDType StarID;
    /* Kick velocity if this kick is the one used*/
    double StarKickVelocity;
    /* Thermal energy included in the kick*/
    double StarTherm;
};

struct WindPriv {
    double * TotalWeight;
    struct StarKick * kicks;
    int64_t nkicks;
    double Time;
    int64_t maxkicks;
    int * nvisited;
    /* Flags that the tree was allocated here
     * and we need to free it to preserve memory order*/
    int tree_alloc_in_wind;
};

/* Comparison function to sort the StarKicks by particle id, distance and star ID.
 * The closest star is used. */
int cmp_by_part_id(const void * a, const void * b)
{
    const struct StarKick * stara = (const struct StarKick * ) a;
    const struct StarKick * starb = (const struct StarKick *) b;
    if(stara->part_index > starb->part_index)
        return 1;
    if(stara->part_index < starb->part_index)
        return -1;
    if(stara->StarDistance > starb->StarDistance)
        return 1;
    if(stara->StarDistance < starb->StarDistance)
        return -1;
    if(stara->StarID > starb->StarID)
        return 1;
    if(stara->StarID < starb->StarID)
        return -1;
    return 0;
}

#define WIND_GET_PRIV(tw) ((struct WindPriv *) (tw->priv))
#define WINDP(i, wind) wind[P[i].PI]

/* Find the 1D DM velocity dispersion of the winds by running a density loop.*/
static void
winds_find_weights(TreeWalk * tw, struct WindPriv * priv, int * NewStars, int NumNewStars, ForceTree * tree, DomainDecomp * ddecomp)
{
    /* Flags that we need to free the tree to preserve memory order*/
    priv->tree_alloc_in_wind = 0;
    if(!tree->tree_allocated_flag) {
        message(0, "Building tree in wind\n");
        priv->tree_alloc_in_wind = 1;
        force_tree_rebuild_mask(tree, ddecomp, GASMASK, NULL);
        walltime_measure("/Cooling/Build");
    }
    /* Types used: gas + DM*/
    tw->ev_label = "WIND_WEIGHT";
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
    tw->postprocess = NULL;

    tw->priv = priv;

    int64_t totalleft = 0;
    sumup_large_ints(1, &NumNewStars, &totalleft);
    /* Subgrid winds come from gas, regular wind from stars: size array accordingly.*/
    int64_t winddata_sz = SlotsManager->info[4].size;
    if(HAS(wind_params.WindModel, WIND_SUBGRID))
        winddata_sz = SlotsManager->info[0].size;
    priv->TotalWeight= (double * ) mymalloc("WindWeight", winddata_sz * sizeof(double));

    /* Note that this will be an over-count because each loop will add more.*/
    priv->nvisited = ta_malloc("nvisited", int, omp_get_max_threads());
    memset(WIND_GET_PRIV(tw)->nvisited, 0, omp_get_max_threads()* sizeof(int));

    /* Find densities*/
    treewalk_run(tw, NewStars, NumNewStars);
}

/* This function spawns winds for the subgrid model, which comes from the star-forming gas.
 * Does a little more calculation than is really necessary, due to shared code, but that shouldn't matter. */
void
winds_subgrid(int * MaybeWind, int NumMaybeWind, const double Time, MyFloat * StellarMasses)
{
    /*The non-subgrid model does nothing here*/
    if(!HAS(wind_params.WindModel, WIND_SUBGRID))
        return;

    if(!MPIU_Any(NumMaybeWind > 0, MPI_COMM_WORLD))
        return;

    int n;
    #pragma omp parallel for
    for(n = 0; n < NumMaybeWind; n++)
    {
        int i = MaybeWind ? MaybeWind[n] : n;
        /* Notice that StellarMasses is indexed like PI, not i!*/
        MyFloat sm = StellarMasses[P[i].PI];
        winds_make_after_sf(i, sm, SPHP(i).VDisp, Time);
    }
    walltime_measure("/Cooling/Wind");
}

/*Do a treewalk for the wind model. This only changes newly created star particles.*/
void
winds_and_feedback(int * NewStars, int NumNewStars, const double Time, ForceTree * tree, DomainDecomp * ddecomp)
{
    /*The subgrid model does nothing here*/
    if(HAS(wind_params.WindModel, WIND_SUBGRID))
        return;

    if(!MPIU_Any(NumNewStars > 0, MPI_COMM_WORLD))
        return;

    TreeWalk tw[1] = {{0}};
    struct WindPriv priv[1];
    priv->Time = Time;
    int i;
    winds_find_weights(tw, priv, NewStars, NumNewStars, tree, ddecomp);

    for (i = 1; i < omp_get_max_threads(); i++)
        priv->nvisited[0] += priv->nvisited[i];
    priv->maxkicks = priv->nvisited[0]+2;

    /* Some particles may be kicked by multiple stars on the same timestep.
    * To ensure this happens only once and does not depend on the order in
    * which the loops are executed, particles are kicked by the nearest new star.
    * This struct stores all such possible kicks, and we sort it out after the treewalk.*/
    priv->kicks = (struct StarKick * ) mymalloc("StarKicks", priv->maxkicks * sizeof(struct StarKick));
    priv->nkicks = 0;
    ta_free(priv->nvisited);

    /* Then run feedback: types used: gas. */
    tw->haswork = NULL;
    tw->ngbiter = (TreeWalkNgbIterFunction) sfr_wind_feedback_ngbiter;
    tw->postprocess = NULL;
    tw->reduce = NULL;
    tw->ev_label = "WIND_KICK";

    treewalk_run(tw, NewStars, NumNewStars);

    /* Sort the possible kicks*/
    qsort_openmp(priv->kicks, priv->nkicks, sizeof(struct StarKick), cmp_by_part_id);
    /* Not parallel as the number of kicked particles should be pretty small*/
    int64_t last_part = -1;
    int64_t nkicked = 0;
    for(i = 0; i < priv->nkicks; i++) {
        /* Only do the kick for the first particle, which is the closest*/
        if(priv->kicks[i].part_index == last_part)
            continue;
        int other = priv->kicks[i].part_index;
        last_part = other;
        nkicked++;
        wind_do_kick(other, priv->kicks[i].StarKickVelocity, priv->kicks[i].StarTherm, Time);
        if(priv->kicks[i].StarKickVelocity <= 0 || !isfinite(priv->kicks[i].StarKickVelocity) || !isfinite(SPHP(other).DelayTime))
        {
            endrun(5, "Odd v: other = %d, DT = %g v = %g i = %d, nkicks %d maxkicks %d dist %g id %ld\n",
                   other, SPHP(other).DelayTime, priv->kicks[i].StarKickVelocity, i, priv->nkicks, priv->maxkicks,
                   priv->kicks[i].StarDistance, priv->kicks[i].StarID);
        }
    }
    /* Get total number of potential new stars to allocate memory.*/
    int64_t tot_newstars, tot_kicks, tot_applied;
    sumup_large_ints(1, &NumNewStars, &tot_newstars);
    MPI_Allreduce(&priv->nkicks, &tot_kicks, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&nkicked, &tot_applied, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    message(0, "Made %ld gas wind, discarded %ld kicks from %d stars. Vel %g\n", tot_applied, tot_kicks - tot_applied, tot_newstars, priv->kicks[0].StarKickVelocity);

    myfree(priv->kicks);
    myfree(priv->TotalWeight);
    if(priv->tree_alloc_in_wind)
        force_tree_free(tree);
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
        /* Enforce the maximum in case of restarts*/
        if(SPHP(i).DelayTime > wind_params.MaxWindFreeTravelTime)
            SPHP(i).DelayTime = wind_params.MaxWindFreeTravelTime;
        const double dloga = get_dloga_for_bin(P[i].TimeBinHydro, P[i].Ti_drift);
        /*  the proper time duration of the step */
        const double dtime = dloga / hubble;
        SPHP(i).DelayTime = DMAX(SPHP(i).DelayTime - dtime, 0);
    }
}

static void
sfr_wind_reduce_weight(int place, TreeWalkResultWind * O, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    int pi = P[place].PI;
    TREEWALK_REDUCE(WIND_GET_PRIV(tw)->TotalWeight[pi], O->TotalWeight);
}

static void
sfr_wind_copy(int place, TreeWalkQueryWind * input, TreeWalk * tw)
{
    input->ID = P[place].ID;
    input->Mass = P[place].Mass;
    input->Hsml = P[place].Hsml;
    if(P[place].Type != 4)
        endrun(5, "Particle %d has type %d not a star, id %ld mass %g\n", place, P[place].Type, P[place].ID, P[place].Mass);
    input->Vdisp = STARP(place).VDisp;

    int pi = P[place].PI;
    input->TotalWeight = WIND_GET_PRIV(tw)->TotalWeight[pi];
}

static void
sfr_wind_weight_ngbiter(TreeWalkQueryWind * I,
        TreeWalkResultWind * O,
        TreeWalkNgbIterWind * iter,
        LocalTreeWalk * lv)
{
    /* this evaluator walks the tree and sums the total mass of surrounding gas
     * particles as described in VS08. */
    /* it also calculates the velocity dispersion of the nearest 40 DM or gas particles */
    if(iter->base.other == -1) {
        iter->base.Hsml = I->Hsml;
        iter->base.mask = GASMASK; /* gas and dm */
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        return;
    }

    int other = iter->base.other;
    double r = iter->base.r;

    if(r > I->Hsml) return;
    /* skip earlier wind particles, which receive
        * no feedback energy */
    if(SPHP(other).DelayTime > 0) return;

    /* NOTE: think twice if we want a symmetric tree walk when wk is used. */
    //double wk = density_kernel_wk(&kernel, r);
    double wk = 1.0;
    O->TotalWeight += wk * P[other].Mass;
    /* Sum up all particles visited on this processor*/
    WIND_GET_PRIV(lv->tw)->nvisited[omp_get_thread_num()]++;

    /*
    message(1, "ThisTask = %d %ld ngb=%d NGB=%d TotalWeight=%g V2sum=%g V1sum=%g %g %g\n",
    ThisTask, I->ID, numngb, O->Ngb, O->TotalWeight, O->V2sum,
    O->V1sum[0], O->V1sum[1], O->V1sum[2]);
    */
}

/* Do the actual kick of the gas particle*/
static void
wind_do_kick(int other, double vel, double therm, double atime)
{
    /* Kick the gas particle*/
    double dir[3];
    get_wind_dir(other, dir);
    int j;
    if(vel > 0 && atime > 0) {
        for(j = 0; j < 3; j++)
        {
            P[other].Vel[j] += vel * dir[j];
        }
        /* StarTherm is internal energy per unit mass. Need to convert to entropy*/
        const double enttou = pow(SPHP(other).Density / pow(atime, 3), GAMMA_MINUS1) / GAMMA_MINUS1;
        SPHP(other).Entropy += therm/enttou;
        if(winds_ever_decouple()) {
            double delay = wind_params.WindFreeTravelLength / (vel / atime);
            if(delay > wind_params.MaxWindFreeTravelTime)
                delay = wind_params.MaxWindFreeTravelTime;
            SPHP(other).DelayTime = delay;
        }
    }
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

/* Get the parameters of the wind kick*/
static void
get_wind_params(double * vel, double * windeff, double * utherm, const double vdisp, const double time)
{
    /* Physical velocity*/
    double vphys = vdisp / time;
    *utherm = wind_params.WindThermalFactor * 1.5 * vphys * vphys;
    if(HAS(wind_params.WindModel, WIND_FIXED_EFFICIENCY)) {
        *windeff = wind_params.WindEfficiency;
        *vel = wind_params.WindSpeed * time;
    } else if(HAS(wind_params.WindModel, WIND_USE_HALO)) {
        *windeff = pow(wind_params.WindSigma0, 2) / (vphys * vphys + 2 * (*utherm));
        *vel = wind_params.WindSpeedFactor * vdisp;
    } else {
        endrun(1, "WindModel = 0x%X is strange. This shall not happen.\n", wind_params.WindModel);
    }
    /* Minimum wind velocity. This ensures particles do not remain in the wind forever*/
    if(*vel < wind_params.MinWindVelocity * time)
        *vel = wind_params.MinWindVelocity * time;
}

static void
sfr_wind_feedback_ngbiter(TreeWalkQueryWind * I,
        TreeWalkResultWind * O,
        TreeWalkNgbIterWind * iter,
        LocalTreeWalk * lv)
{

    /* this evaluator walks the tree and blows wind. */

    if(iter->base.other == -1) {
        iter->base.mask = GASMASK;
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        iter->base.Hsml = I->Hsml;
        return;
    }
    int other = iter->base.other;
    double r = iter->base.r;

    /* this is radius cut is redundant because the tree walk is asymmetric
     * we may want to use fancier weighting that requires symmetric in the future. */
    if(r > I->Hsml) return;

    /* skip earlier wind particles */
    if(SPHP(other).DelayTime > 0) return;

    /* No eligible gas particles not in wind*/
    if(I->TotalWeight == 0 || I->Vdisp <= 0) return;

    /* Paranoia*/
    if(P[other].Type != 0 || P[other].IsGarbage || P[other].Swallowed)
        return;

    /* Get the velocity, thermal energy and efficiency of the kick*/
    double utherm = 0, v=0, windeff = 0;
    get_wind_params(&v, &windeff, &utherm, I->Vdisp, WIND_GET_PRIV(lv->tw)->Time);

    double p = windeff * I->Mass / I->TotalWeight;
    double random = get_random_number(I->ID + P[other].ID);

    if (random < p && v > 0) {
        /* Store a potential kick. This might not be the kick actually used,
         * because another star particle may be closer, but we can resolve
         * that after the treewalk*/
        int64_t * nkicks = &WIND_GET_PRIV(lv->tw)->nkicks;
        /* Use a single global kick list.*/
        int64_t ikick = atomic_fetch_and_add_64(nkicks, 1);
        if(ikick >= WIND_GET_PRIV(lv->tw)->maxkicks)
            endrun(5, "Not enough room in kick queue: %ld > %ld for particle %d starid %ld distance %g\n",
                   ikick, WIND_GET_PRIV(lv->tw)->maxkicks, other, I->ID, r);
        struct StarKick * kick = &WIND_GET_PRIV(lv->tw)->kicks[ikick];
        kick->StarDistance = r;
        kick->StarID = I->ID;
        kick->StarKickVelocity = v;
        kick->StarTherm = utherm;
        kick->part_index = other;
    }
}

int
winds_make_after_sf(int i, double sm, double vdisp, double atime)
{
    if(!HAS(wind_params.WindModel, WIND_SUBGRID))
        return 0;

    /* Get the velocity, thermal energy and efficiency of the kick*/
    double utherm = 0, vel=0, windeff = 0;
    get_wind_params(&vel, &windeff, &utherm, vdisp, atime);

    /* Here comes the Springel Hernquist 03 wind model */
    /* Notice that this is the mass of the gas particle after forking a star, Mass - Mass/GENERATIONS.*/
    double pw = windeff * sm / P[i].Mass;
    double prob = 1 - exp(-pw);
    if(get_random_number(P[i].ID + 2) < prob) {
        wind_do_kick(i, vel, utherm, atime);
    }
    return 0;
}


/* Code to compute velocity dispersions*/

typedef struct {
    TreeWalkQueryBase base;
    double Mass;
    double DMRadius[NWINDHSML];
    double Vel[3];
} TreeWalkQueryWindVDisp;

typedef struct {
    TreeWalkResultBase base;
    double V1sum[NWINDHSML][3];
    double V2sum[NWINDHSML];
    double Ngb[NWINDHSML];
    int alignment; /* Ensure alignment*/
    int maxcmpte;
} TreeWalkResultWindVDisp;

struct WindVDispPriv {
    double Time;
    double hubble;
    double FgravkickB;
    double gravkicks[TIMEBINS+1];
    double hydrokicks[TIMEBINS+1];
    double ddrift;
    /* Lower and upper bounds on smoothing length*/
    MyFloat *Left, *Right, *DMRadius;
    /* Maximum index where NumNgb is valid. */
    int * maxcmpte;
    MyFloat (* V2sum)[NWINDHSML];
    MyFloat (* V1sum)[NWINDHSML][3];
    MyFloat (* Ngb) [NWINDHSML];
};
#define WINDV_GET_PRIV(tw) ((struct WindVDispPriv *) (tw->priv))

static inline double
vdispeffdmradius(int place, int i, TreeWalk * tw)
{
    double left = WINDV_GET_PRIV(tw)->Left[place];
    double right = WINDV_GET_PRIV(tw)->Right[place];
    /*The asymmetry is because it is free to compute extra densities for h < Hsml, but not for h > Hsml*/
    if (right > 0.99*tw->tree->BoxSize){
        right = WINDV_GET_PRIV(tw)->DMRadius[place];
    }
    if(left == 0)
        left = 0.1 * WINDV_GET_PRIV(tw)->DMRadius[place];
    /*Evenly split in volume*/
    double rvol = pow(right, 3);
    double lvol = pow(left, 3);
    return pow((1.0*i+1)/(1.0*NWINDHSML+1) * (rvol - lvol) + lvol, 1./3);
}

static void
wind_vdisp_copy(int place, TreeWalkQueryWindVDisp * input, TreeWalk * tw)
{
    input->Mass = P[place].Mass;
    int i;
    for(i=0; i<3; i++)
        input->Vel[i] = P[place].Vel[i];
    for(i = 0; i<NWINDHSML; i++){
        input->DMRadius[i]=vdispeffdmradius(place,i,tw);
    }
}

static void
wind_vdisp_ngbiter(TreeWalkQueryWindVDisp * I,
        TreeWalkResultWindVDisp * O,
        TreeWalkNgbIterWind * iter,
        LocalTreeWalk * lv)
{
    /* this evaluator walks the tree and sums the total mass of surrounding gas
     * particles as described in VS08. */
    /* it also calculates the velocity dispersion of the nearest 40 DM or gas particles */
    if(iter->base.other == -1) {
        iter->base.Hsml = I->DMRadius[NWINDHSML-1];
        iter->base.mask = DMMASK; /* gas and dm */
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        O->maxcmpte = NWINDHSML;
        return;
    }

    int other = iter->base.other;
    double r = iter->base.r;
    double * dist = iter->base.dist;

    int i;
    const double atime = WINDV_GET_PRIV(lv->tw)->Time;
    for (i = 0; i < O->maxcmpte; i++) {
        if(r < I->DMRadius[i]) {
            O->Ngb[i] += 1;
            int d;
            MyFloat VelPred[3];
            DM_VelPred(other, VelPred, WINDV_GET_PRIV(lv->tw)->FgravkickB, WINDV_GET_PRIV(lv->tw)->gravkicks);
            for(d = 0; d < 3; d ++) {
                /* Add hubble flow to relative velocity. Use predicted velocity to current time.
                 * The I particle is active so always at current time.*/
                double vel = VelPred[d] - I->Vel[d] + WINDV_GET_PRIV(lv->tw)->hubble * atime * atime * dist[d];
                O->V1sum[i][d] += vel;
                O->V2sum[i] += vel * vel;
            }
        }
    }

    for(i = 0; i<NWINDHSML; i++){
        if(O->Ngb[i] > NUMDMNGB){
            O->maxcmpte = i+1;
            iter->base.Hsml = I->DMRadius[i];
            break;
        }
    }
    /*
    message(1, "ThisTask = %d %ld ngb=%d NGB=%d TotalWeight=%g V2sum=%g V1sum=%g %g %g\n",
    ThisTask, I->ID, numngb, O->Ngb, O->TotalWeight, O->V2sum,
    O->V1sum[0], O->V1sum[1], O->V1sum[2]);
    */
}

static void
wind_vdisp_reduce(int place, TreeWalkResultWindVDisp * O, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    int pi = P[place].PI;
    int i;
    if(mode == 0 || WINDV_GET_PRIV(tw)->maxcmpte[pi] > O->maxcmpte)
        WINDV_GET_PRIV(tw)->maxcmpte[pi] = O->maxcmpte;
    int k;
    for (i = 0; i < O->maxcmpte; i++){
        TREEWALK_REDUCE(WINDV_GET_PRIV(tw)->Ngb[pi][i], O->Ngb[i]);
        TREEWALK_REDUCE(WINDV_GET_PRIV(tw)->V2sum[pi][i], O->V2sum[i]);
        for(k = 0; k < 3; k ++) {
            TREEWALK_REDUCE(WINDV_GET_PRIV(tw)->V1sum[pi][i][k], O->V1sum[i][k]);
        }
    }
//     message(1, "Reduce ID=%ld, NGB_first=%d NGB_last=%d maxcmpte = %d, left = %g, right = %g\n",
//             P[place].ID, O->Ngb[0],O->Ngb[O->maxcmpte-1],WINDP(place, Windd).maxcmpte,WINDP(place, Windd).Left,WINDP(place, Windd).Right);
}

static void
wind_vdisp_postprocess(const int i, TreeWalk * tw)
{
    const int pi = P[i].PI;
    const int maxcmpt = WINDV_GET_PRIV(tw)->maxcmpte[pi];
    int j;
    double evaldmradius[NWINDHSML];
    for(j = 0; j < maxcmpt; j++){
        evaldmradius[j] = vdispeffdmradius(i,j,tw);
    }
    int close = 0;
    WINDV_GET_PRIV(tw)->DMRadius[pi] = ngb_narrow_down(&WINDV_GET_PRIV(tw)->Right[pi], &WINDV_GET_PRIV(tw)->Left[pi], evaldmradius, WINDV_GET_PRIV(tw)->Ngb[pi], maxcmpt, NUMDMNGB, &close, tw->tree->BoxSize);
    double numngb = WINDV_GET_PRIV(tw)->Ngb[pi][close];

    int tid = omp_get_thread_num();
    /*  If we have 40 neighbours, or if DMRadius is narrow, set vdisp. Otherwise add to redo queue */
    if((numngb < (NUMDMNGB - MAXDMDEVIATION) || numngb > (NUMDMNGB + MAXDMDEVIATION)) &&
        (WINDV_GET_PRIV(tw)->Right[pi] - WINDV_GET_PRIV(tw)->Left[pi] > 1e-2)) {
        /* More work needed: add this particle to the redo queue*/
        tw->NPRedo[tid][tw->NPLeft[tid]] = i;
        tw->NPLeft[tid] ++;
    }
    else{
        double vdisp = WINDV_GET_PRIV(tw)->V2sum[pi][close] / numngb;
        int d;
        for(d = 0; d < 3; d++){
            vdisp -= pow(WINDV_GET_PRIV(tw)->V1sum[pi][close][d] / numngb,2);
        }
        if(vdisp > 0) {
            if(P[i].Type == 0)
                SPHP(i).VDisp = sqrt(vdisp / 3);
        }
    }

    if(tw->maxnumngb[tid] < numngb)
        tw->maxnumngb[tid] = numngb;
    if(tw->minnumngb[tid] > numngb)
        tw->minnumngb[tid] = numngb;
}

static int
winds_veldisp_haswork(int n, TreeWalk * tw)
{
    /* Don't want a density for swallowed particles*/
    if(P[n].Swallowed || P[n].IsGarbage)
        return 0;
    /* Only want gas*/
    if(P[n].Type != 0)
        return 0;
    /* We only want VDisp for gas particles that may be star-forming over the next PM timestep.
     * Use DtHsml.*/
    double densfac = (P[n].Hsml + P[n].DtHsml * WINDV_GET_PRIV(tw)->ddrift)/P[n].Hsml;
    if(densfac > 1)
        densfac = 1;
    if(SPHP(n).Density/(densfac * densfac * densfac) < 0.1 * sfr_density_threshold(WINDV_GET_PRIV(tw)->Time))
        return 0;
    /* Update veldisp only on a gravitationally active timestep,
     * since this is the frequency at which the gravitational acceleration,
     * which is the only thing DM contributes to, could change. This is probably
     * overly conservative, because most of the acceleration will be from other stars. */
//     if(!is_timebin_active(P[n].TimeBinGravity, P[n].Ti_drift))
//         return 0;
    return 1;
}

/* Find the 1D DM velocity dispersion of all gas particles by running a density loop.
 * Stores it in VDisp in the slots structure.*/
void
winds_find_vel_disp(const ActiveParticles * act, const double Time, const double hubble, Cosmology * CP, DriftKickTimes * times, DomainDecomp * ddecomp)
{
    TreeWalk tw[1] = {0};
    struct WindVDispPriv priv[1] = {0};
    ForceTree tree[1] = {0};
    /* Types used: gas*/
    tw->ev_label = "WIND_VDISP";
    tw->fill = (TreeWalkFillQueryFunction) wind_vdisp_copy;
    tw->reduce = (TreeWalkReduceResultFunction) wind_vdisp_reduce;
    tw->query_type_elsize = sizeof(TreeWalkQueryWindVDisp);
    tw->result_type_elsize = sizeof(TreeWalkResultWindVDisp);
    tw->tree = tree;

    /* sum the total weight of surrounding gas */
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterWind);
    tw->ngbiter = (TreeWalkNgbIterFunction) wind_vdisp_ngbiter;

    tw->haswork = winds_veldisp_haswork;
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_nolist_ngbiter;
    tw->postprocess = (TreeWalkProcessFunction) wind_vdisp_postprocess;

    priv[0].Time = Time;
    priv[0].hubble = hubble;
    priv[0].ddrift = get_exact_drift_factor(CP, times->Ti_Current, times->PM_length);
    tw->priv = priv;

    /* Build the queue to check that we have something to do before we rebuild the tree.*/
    treewalk_build_queue(tw, act->ActiveParticle, act->NumActiveParticle, 0);

    int * ActiveVDisp = tw->WorkSet;
    int64_t NumVDisp = tw->WorkSetSize;
    int64_t totvdisp;
    /* If this queue is empty, nothing to do.*/
    MPI_Allreduce(&NumVDisp, &totvdisp, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    if(totvdisp == 0) {
        myfree(ActiveVDisp);
        return;
    }

    force_tree_rebuild_mask(tree, ddecomp, DMMASK, NULL);
    tw->haswork = NULL;

    priv->FgravkickB = get_exact_gravkick_factor(CP, times->PM_kick, times->Ti_Current);
    memset(priv->gravkicks, 0, sizeof(priv->gravkicks[0])*(TIMEBINS+1));
    memset(priv->hydrokicks, 0, sizeof(priv->hydrokicks[0])*(TIMEBINS+1));
    /* Compute the factors to move a current kick times velocity to the drift time velocity.
     * We need to do the computation for all timebins up to the maximum because even inactive
     * particles may have interactions. */
    int i;
    #pragma omp parallel for
    for(i = times->mintimebin; i <= TIMEBINS; i++)
    {
        priv->gravkicks[i] = get_exact_gravkick_factor(CP, times->Ti_kick[i], times->Ti_Current);
        priv->hydrokicks[i] = get_exact_hydrokick_factor(CP, times->Ti_kick[i], times->Ti_Current);
    }

    priv->Left = (MyFloat *) mymalloc("VDISP->Left", SlotsManager->info[0].size * sizeof(MyFloat));
    priv->Right = (MyFloat *) mymalloc("VDISP->Right", SlotsManager->info[0].size * sizeof(MyFloat));
    priv->DMRadius = (MyFloat *) mymalloc("VDISP->DMRadius", SlotsManager->info[0].size * sizeof(MyFloat));
    priv->Ngb = (MyFloat (*) [NWINDHSML]) mymalloc("VDISP->NumNgb", SlotsManager->info[0].size * sizeof(priv->Ngb[0]));
    priv->V1sum = (MyFloat (*) [NWINDHSML][3]) mymalloc("VDISP->V1Sum", SlotsManager->info[0].size * sizeof(priv->V1sum[0]));
    priv->V2sum = (MyFloat (*) [NWINDHSML]) mymalloc("VDISP->V2Sum", SlotsManager->info[0].size * sizeof(priv->V2sum[0]));
    priv->maxcmpte = (int *) mymalloc("maxcmpte", SlotsManager->info[0].size * sizeof(int));
    report_memory_usage("WIND_VDISP");

    /*Initialise the WINDP array*/
    #pragma omp parallel for
    for (i = 0; i < act->NumActiveParticle; i++) {
        const int n = act->ActiveParticle ? act->ActiveParticle[i] : i;
        if(P[n].Type == 0) {
            const int pi = P[n].PI;
            priv->DMRadius[pi] = P[n].Hsml;
            priv->Left[pi] = 0;
            priv->Right[pi] = tree->BoxSize;
            priv->maxcmpte[pi] = NUMDMNGB;
        }
    }
    /* Find densities*/
    treewalk_do_hsml_loop(tw, ActiveVDisp, NumVDisp, 1);
    /* Free memory*/
    myfree(priv->maxcmpte);
    myfree(priv->V2sum);
    myfree(priv->V1sum);
    myfree(priv->Ngb);
    myfree(priv->DMRadius);
    myfree(priv->Right);
    myfree(priv->Left);
    force_tree_free(tree);
    myfree(ActiveVDisp);
    walltime_measure("/Cooling/VDisp");

}
