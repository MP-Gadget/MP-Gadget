/*Prototypes and structures for the wind model*/
#ifdef SFR

#include <math.h>
#include <string.h>
#include "winds.h"
#include "treewalk.h"
#include "drift.h"
#include "slotsmanager.h"
#include "timebinmgr.h"

typedef struct {
    TreeWalkQueryBase base;
    int NodeList[NODELISTLENGTH];
    double Sfr;
    double Dt;
    double Mass;
    double Hsml;
    double TotalWeight;
    double DMRadius;
    double Vdisp;
    double Vmean[3];
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

static struct winddata {
    double DMRadius;
    double Left;
    double Right;
    double TotalWeight;
    union {
        double Vdisp;
        double V2sum;
    };
    union {
        double Vmean[3];
        double V1sum[3];
    };
    int Ngb;
} * Wind;

static int
sfr_wind_weight_haswork(int target, TreeWalk * tw);

static int
sfr_wind_feedback_haswork(int target, TreeWalk * tw);

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

static int* NPLeft;

static void
sfr_wind_feedback_preprocess(const int n, TreeWalk * tw)
{
    Wind[n].DMRadius = 2 * P[n].Hsml;
    Wind[n].Left = 0;
    Wind[n].Right = -1;
    P[n].DensityIterationDone = 0;
}

static void
sfr_wind_feedback_postprocess(const int i, TreeWalk * tw)
{
    P[i].IsNewParticle = 0;
}

/*Do a treewalk for the wind model. This only changes newly created star particles.*/
void
winds_and_feedback(int * NewStars, int NumNewStars)
{
    if(!All.WindOn)
        return;
    /*The subgrid model does nothing here*/
    if(HAS(All.WindModel, WIND_SUBGRID))
        return;
    Wind = (struct winddata * ) mymalloc("WindExtraData", PartManager->NumPart * sizeof(struct winddata));
    TreeWalk tw[1] = {{0}};

    tw->ev_label = "SFR_WIND";
    tw->fill = (TreeWalkFillQueryFunction) sfr_wind_copy;
    tw->reduce = (TreeWalkReduceResultFunction) sfr_wind_reduce_weight;
    tw->UseNodeList = 1;
    tw->query_type_elsize = sizeof(TreeWalkQueryWind);
    tw->result_type_elsize = sizeof(TreeWalkResultWind);

    /* sum the total weight of surrounding gas */
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterWind);
    tw->ngbiter = (TreeWalkNgbIterFunction) sfr_wind_weight_ngbiter;

    /* First set DensityIterationDone for weighting */
    /* Watchout: the process function name is preprocess, but not called in the feedback tree walk
        * because we need to compute the normalization before the feedback . */
    tw->visit = NULL;
    tw->haswork = (TreeWalkHasWorkFunction) sfr_wind_feedback_haswork;
    tw->postprocess = (TreeWalkProcessFunction) sfr_wind_feedback_preprocess;
    treewalk_run(tw, NewStars, NumNewStars);

    tw->haswork = sfr_wind_weight_haswork;
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw->postprocess = (TreeWalkProcessFunction) sfr_wind_weight_postprocess;

    int64_t totalleft = 0;
    sumup_large_ints(1, &NumNewStars, &totalleft);
    NPLeft = ta_malloc("NPLeft", int, All.NumThreads);
    while(totalleft > 0) {
        int i;
        memset(NPLeft, 0, sizeof(int)*All.NumThreads);

        treewalk_run(tw, NewStars, NumNewStars);
        int Nleft = 0;

        for(i = 0; i< All.NumThreads; i++)
            Nleft += NPLeft[i];

        sumup_large_ints(1, &Nleft, &totalleft);
        message(0, "Star DM iteration Total left = %ld\n", totalleft);
    }
    ta_free(NPLeft);

    /* Then run feedback */
    tw->haswork = (TreeWalkHasWorkFunction) sfr_wind_feedback_haswork;
    tw->ngbiter = (TreeWalkNgbIterFunction) sfr_wind_feedback_ngbiter;
    tw->postprocess = (TreeWalkProcessFunction) sfr_wind_feedback_postprocess;
    tw->reduce = NULL;

    treewalk_run(tw, NewStars, NumNewStars);
    myfree(Wind);
    walltime_measure("/Cooling/Wind");
}

/*Evolve a wind particle, reducing its DelayTime*/
void
wind_evolve(int i)
{
    /*Remove a wind particle from the delay mode if the (physical) density has dropped sufficiently.*/
    if(SPHP(i).DelayTime > 0 && SPHP(i).Density * All.cf.a3inv < All.WindFreeTravelDensFac * All.PhysDensThresh) {
            SPHP(i).DelayTime = 0;
    }
    /*Reduce the time until the particle can form stars again by the current timestep*/
    if(SPHP(i).DelayTime > 0) {
        const double dloga = get_dloga_for_bin(P[i].TimeBin);
        /*  the proper time duration of the step */
        const double dtime = dloga / All.cf.hubble;
        SPHP(i).DelayTime = DMAX(SPHP(i).DelayTime - dtime, 0);
    }
}

static void
sfr_wind_weight_postprocess(const int i, TreeWalk * tw)
{
    int diff = Wind[i].Ngb - 40;
    if(diff < -2) {
        /* too few */
        Wind[i].Left = Wind[i].DMRadius;
    } else if(diff > 2) {
        /* too many */
        Wind[i].Right = Wind[i].DMRadius;
    } else {
        P[i].DensityIterationDone = 1;
    }
    if(Wind[i].Right >= 0) {
        /* if Ngb hasn't converged to 40, see if DMRadius converged*/
        if(Wind[i].Right - Wind[i].Left < 1e-2) {
            P[i].DensityIterationDone = 1;
        } else {
            Wind[i].DMRadius = 0.5 * (Wind[i].Left + Wind[i].Right);
        }
    } else {
        Wind[i].DMRadius *= 1.3;
    }

    if(P[i].DensityIterationDone) {
        double vdisp = Wind[i].V2sum / Wind[i].Ngb;
        int d;
        for(d = 0; d < 3; d ++) {
            Wind[i].Vmean[d] = Wind[i].V1sum[d] / Wind[i].Ngb;
            vdisp -= Wind[i].Vmean[d] * Wind[i].Vmean[d];
        }
        Wind[i].Vdisp = sqrt(vdisp / 3);
    } else {
        int tid = omp_get_thread_num();
        NPLeft[tid] ++;
    }
}

static int
sfr_wind_weight_haswork(int target, TreeWalk * tw)
{
    if(P[target].Type == 4) {
        if(P[target].IsNewParticle && !P[target].DensityIterationDone) {
             return 1;
        }
    }
    return 0;
}

static int
sfr_wind_feedback_haswork(int target, TreeWalk * tw)
{
    if(P[target].Type == 4) {
        if(P[target].IsNewParticle) {
             return 1;
        }
    }
    return 0;
}

static void
sfr_wind_reduce_weight(int place, TreeWalkResultWind * O, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    TREEWALK_REDUCE(Wind[place].TotalWeight, O->TotalWeight);
    int k;
    for(k = 0; k < 3; k ++) {
        TREEWALK_REDUCE(Wind[place].V1sum[k], O->V1sum[k]);
    }
    TREEWALK_REDUCE(Wind[place].V2sum, O->V2sum);
    TREEWALK_REDUCE(Wind[place].Ngb, O->Ngb);
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
    input->TotalWeight = Wind[place].TotalWeight;

    input->DMRadius = Wind[place].DMRadius;
    input->Vdisp = Wind[place].Vdisp;

    int k;
    for (k = 0; k < 3; k ++)
        input->Vmean[k] = Wind[place].Vmean[k];
}

static void
sfr_wind_weight_ngbiter(TreeWalkQueryWind * I,
        TreeWalkResultWind * O,
        TreeWalkNgbIterWind * iter,
        LocalTreeWalk * lv)
{
    /* this evaluator walks the tree and sums the total mass of surrounding gas
     * particles as described in VS08. */
    /* it also calculates the DM dispersion of the nearest 40 DM paritlces */
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
        /* Ignore wind particles */
        if(SPHP(other).DelayTime > 0) return;
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

    /* skip wind particles */
    if(SPHP(other).DelayTime > 0) return;

    /* this is radius cut is redundant because the tree walk is asymmetric
     * we may want to use fancier weighting that requires symmetric in the future. */
    if(r > I->Hsml) return;

    double windeff=0;
    double v=0;
    if(HAS(All.WindModel, WIND_FIXED_EFFICIENCY)) {
        windeff = All.WindEfficiency;
        v = All.WindSpeed * All.cf.a;
    } else if(HAS(All.WindModel, WIND_USE_HALO)) {
        windeff = 1.0 / (I->Vdisp / All.cf.a / All.WindSigma0);
        windeff *= windeff;
        v = All.WindSpeedFactor * I->Vdisp;
    } else {
        endrun(1, "WindModel = 0x%X is strange. This shall not happen.\n", All.WindModel);
    }

    //double wk = density_kernel_wk(&kernel, r);

    /* in this case the particle is already locked by the tree walker */
    /* we may want to add another lock to avoid this. */
    if(P[other].ID != I->base.ID)
        lock_particle(other);

    double wk = 1.0;
    double p = windeff * wk * I->Mass / I->TotalWeight;
    double random = get_random_number(I->base.ID + P[other].ID);
    if (random < p) {
        make_particle_wind(I->base.ID, other, v, I->Vmean);
    }

    if(P[other].ID != I->base.ID)
        unlock_particle(other);

}

int
make_particle_wind(MyIDType ID, int i, double v, double vmean[3]) {
    /* v and vmean are in internal units (km/s *a ), not km/s !*/
    /* returns 0 if particle i is converted to wind. */
    // message(1, "%ld Making ID=%ld (%g %g %g) to wind with v= %g\n", ID, P[i].ID, P[i].Pos[0], P[i].Pos[1], P[i].Pos[2], v);
    int j;
    /* ok, make the particle go into the wind */
    double dir[3];
    if(HAS(All.WindModel, WIND_ISOTROPIC)) {
        double theta = acos(2 * get_random_number(P[i].ID + 3) - 1);
        double phi = 2 * M_PI * get_random_number(P[i].ID + 4);

        dir[0] = sin(theta) * cos(phi);
        dir[1] = sin(theta) * sin(phi);
        dir[2] = cos(theta);
    } else {
        double vel[3];
        for(j = 0; j < 3; j++) {
            vel[j] = P[i].Vel[j] - vmean[j];
        }
        /*FIXME: Shouldn't this be total accel, not short-range accel?*/
        dir[0] = P[i].GravAccel[1] * vel[2] - P[i].GravAccel[2] * vel[1];
        dir[1] = P[i].GravAccel[2] * vel[0] - P[i].GravAccel[0] * vel[2];
        dir[2] = P[i].GravAccel[0] * vel[1] - P[i].GravAccel[1] * vel[0];
    }

    double norm = 0;
    for(j = 0; j < 3; j++)
        norm += dir[j] * dir[j];

    /*FIXME: Should this be inside the !WIND_ISOTROPIC case?*/
    norm = sqrt(norm);
    if(get_random_number(P[i].ID + 5) < 0.5)
        norm = -norm;

    if(norm != 0)
    {
        for(j = 0; j < 3; j++)
            dir[j] /= norm;

        for(j = 0; j < 3; j++)
        {
            P[i].Vel[j] += v * dir[j];
        }
        SPHP(i).DelayTime = All.WindFreeTravelLength / (v / All.cf.a);
    }
    return 0;
}
#endif
