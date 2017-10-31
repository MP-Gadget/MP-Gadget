/***
 * Multi-Phase star formaiton
 *
 * The algorithm here is based on Springel Hernequist 2003, and Okamoto 2010.
 *
 * The source code originally came from sfr_eff.c in Gadget-3. This version has
 * been heavily rewritten to add support for new wind models, new star formation
 * criterions, and more importantly, use use the new tree walker routines.
 *
 * I (Yu Feng) feel it is appropriate to release most of this file with a free license,
 * because the implementation here has diverged from the original code by too far.
 *
 * The largest remaining concern are a few functions there were obtained from Gadget-P. 
 * They are for * self-gravity starformation condition and H2.
 * Eventhough they have been heavily rewritten, the core math is the same.
 * the license is still murky. Do not use them unless Phil Hopkins has agreed.
 *
 * */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "allvars.h"
#include "sfr_eff.h"
#include "drift.h"
#include "treewalk.h"
#include "cooling.h"
#include "slotsmanager.h"
#include "mymalloc.h"
#include "endrun.h"
#include "timestep.h"
#include "system.h"

/*Cooling only: no star formation*/
static void cooling_direct(int i);

#ifdef SFR
static double u_to_temp_fac; /* assuming very hot !*/

/* these guys really shall be local to cooling_and_starformation, but
 * I am too lazy to pass them around to subroutines.
 */
static int stars_converted;
static int stars_spawned;
static double sum_sm;
static double sum_mass_stars;

static void cooling_relaxed(int i, double egyeff, double dtime, double trelax);

static int get_sfr_condition(int i);
static int make_particle_star(int i);
static void starformation(int i);
static double get_sfr_factor_due_to_selfgravity(int i);
static double get_sfr_factor_due_to_h2(int i);
static double get_starformation_rate_full(int i, double dtime, MyFloat * ne_new, double * trelax, double * egyeff);
static double find_star_mass(int i);

/*
 * This routine does cooling and star formation for
 * the effective multi-phase model.
 */
static int
sfr_cooling_haswork(int target, TreeWalk * tw)
{
    return P[target].Type == 0 && P[target].Mass > 0;
}

/*Prototypes and structures for the wind model*/
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
make_particle_wind(MyIDType ID, int i, double v, double vmean[3]);

static int
sfr_wind_weight_haswork(int target, TreeWalk * tw);

static int
sfr_wind_feedback_haswork(int target, TreeWalk * tw);

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

static int NPLeft;

static void
sfr_wind_feedback_preprocess(int n, TreeWalk * tw)
{
    Wind[n].DMRadius = 2 * P[n].Hsml;
    Wind[n].Left = 0;
    Wind[n].Right = -1;
    P[n].DensityIterationDone = 0;
}

static void
sfr_wind_weight_postprocess(int i)
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
#pragma omp atomic
        NPLeft ++;
    }
}

static void
sfr_wind_feedback_postprocess(int i)
{
    P[i].IsNewParticle = 0;
}

/*End of wind model functions*/

static void
sfr_cool_postprocess(int i, TreeWalk * tw)
{
        int flag;
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

        /* check whether conditions for star formation are fulfilled.
         *
         * f=1  normal cooling
         * f=0  star formation
         */
        flag = get_sfr_condition(i);

        /* normal implicit isochoric cooling */
        if(flag == 1 || All.QuickLymanAlphaProbability > 0) {
            cooling_direct(i);
        }
        if(flag == 0) {
            /* active star formation */
            starformation(i);
        }
}

void cooling_and_starformation(void)
    /* cooling routine when star formation is enabled */
{
    u_to_temp_fac = (4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC))) * PROTONMASS / BOLTZMANN * GAMMA_MINUS1
        * All.UnitEnergy_in_cgs / All.UnitMass_in_g;

    walltime_measure("/Misc");

    stars_spawned = stars_converted = 0;
    sum_sm = sum_mass_stars = 0;

    TreeWalk tw[1] = {0};

    tw->visit = NULL; /* no tree walk */
    tw->ev_label = "SFR_COOL";
    tw->haswork = sfr_cooling_haswork;
    tw->postprocess = (TreeWalkProcessFunction) sfr_cool_postprocess;

    treewalk_run(tw, ActiveParticle, NumActiveParticle);

    int tot_spawned, tot_converted;
    MPI_Allreduce(&stars_spawned, &tot_spawned, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&stars_converted, &tot_converted, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if(tot_spawned > 0 || tot_converted > 0)
    {
        message(0, "SFR: spawned %d stars, converted %d gas particles into stars\n",
                    tot_spawned, tot_converted);

        /* Note: N_sph is only reduced once domain_garbage_collection is called */

        /* Note: New tree construction can be avoided because of  `force_add_star_to_tree()' */
    }

    double totsfrrate, localsfr=0;
    int i;
    /* FIXME: this is inaccurate if some particles are made into garbage. may want to propagate garbage flag to the slots.*/
    #pragma omp parallel for reduction(+: localsfr)
    for(i = 0; i < SlotsManager->info[0].size; i++)
        localsfr += SphP[i].Sfr;

    MPI_Allreduce(&localsfr, &totsfrrate, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double total_sum_mass_stars, total_sm;

    MPI_Reduce(&sum_sm, &total_sm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_mass_stars, &total_sum_mass_stars, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(ThisTask == 0)
    {
        double rate;
        double rate_in_msunperyear;
        if(All.TimeStep > 0)
            rate = total_sm / (All.TimeStep / (All.Time * All.cf.hubble));
        else
            rate = 0;

        /* convert to solar masses per yr */

        rate_in_msunperyear = rate * (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);

        fprintf(FdSfr, "%g %g %g %g %g\n", All.Time, total_sm, totsfrrate, rate_in_msunperyear,
                total_sum_mass_stars);
        fflush(FdSfr);
    }
    walltime_measure("/Cooling/StarFormation");

    /* now lets make winds. this has to be after NumPart is updated */
    if(All.WindOn && !HAS(All.WindModel, WIND_SUBGRID)){
        Wind = (struct winddata * ) mymalloc("WindExtraData", NumPart * sizeof(struct winddata));
        TreeWalk tw[1] = {0};

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
        treewalk_run(tw, ActiveParticle, NumActiveParticle);

        tw->haswork = sfr_wind_weight_haswork;
        tw->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
        tw->postprocess = (TreeWalkProcessFunction) sfr_wind_weight_postprocess;

        int done = 0;
        while(!done) {
            NPLeft = 0;
            treewalk_run(tw, ActiveParticle, NumActiveParticle);

            int64_t totalleft = 0;
            sumup_large_ints(1, &NPLeft, &totalleft);
            message(0, "Star DM iteration Total left = %ld\n", totalleft);
            done = totalleft == 0;
        }

        /* Then run feedback */
        tw->haswork = (TreeWalkHasWorkFunction) sfr_wind_feedback_haswork;
        tw->ngbiter = (TreeWalkNgbIterFunction) sfr_wind_feedback_ngbiter;
        tw->postprocess = (TreeWalkProcessFunction) sfr_wind_feedback_postprocess;
        tw->reduce = NULL;

        treewalk_run(tw, ActiveParticle, NumActiveParticle);
        myfree(Wind);
    }
    walltime_measure("/Cooling/Wind");
}

#else //No SFR

static void
cool_postprocess(int i, TreeWalk * tw)
{
    cooling_direct(i);
}

/* cooling routine when star formation is disabled */
void cooling_only(void)
{
    if(!All.CoolingOn) return;
    walltime_measure("/Misc");

    TreeWalk tw[1] = {0};

    /* Only used to list all active particles for the parallel loop */
    /* no tree walking and no need to export / copy particles. */

    tw->visit = NULL; /* no tree walk */
    tw->ev_label = "SFR_COOL";
    tw->haswork = sfr_cooling_haswork;
    tw->postprocess = (TreeWalkProcessFunction) cool_postprocess;

    treewalk_run(tw, ActiveParticle, NumActiveParticle);

    walltime_measure("/Cooling/StarFormation");
}

#endif

static void
cooling_direct(int i) {

    /*  the actual time-step */
    double dloga = get_dloga_for_bin(P[i].TimeBin);
    double dtime = dloga / All.cf.hubble;

#ifdef SFR
    SPHP(i).Sfr = 0;
#endif

    double ne = SPHP(i).Ne;	/* electron abundance (gives ionization state and mean molecular weight) */

    double unew = DMAX(All.MinEgySpec,
            (SPHP(i).Entropy + SPHP(i).DtEntropy * dloga) /
            GAMMA_MINUS1 * pow(SPHP(i).EOMDensity * All.cf.a3inv, GAMMA_MINUS1));

#ifdef BLACK_HOLES
    if(SPHP(i).Injected_BH_Energy)
    {
        if(P[i].Mass == 0) {
            endrun(-1, "Encoutered zero mass particle during sfr;"
                      " We haven't implemented tracer particles and this shall not happen\n");
            /* This shall not happend */
            SPHP(i).Injected_BH_Energy = 0;
        }

        unew += SPHP(i).Injected_BH_Energy / P[i].Mass;
        double temp = u_to_temp_fac * unew;


        if(temp > 5.0e9)
            unew = 5.0e9 / u_to_temp_fac;

        SPHP(i).Injected_BH_Energy = 0;
    }
#endif

    struct UVBG uvbg;
    GetParticleUVBG(i, &uvbg);
    unew = DoCooling(unew, SPHP(i).Density * All.cf.a3inv, dtime, &uvbg, &ne, SPHP(i).Metallicity);

    SPHP(i).Ne = ne;

    if(P[i].TimeBin)	/* upon start-up, we need to protect against dt==0 */
    {
        /* note: the adiabatic rate has been already added in ! */

        if(dloga > 0)
        {

            SPHP(i).DtEntropy = (unew * GAMMA_MINUS1 /
                    pow(SPHP(i).EOMDensity * All.cf.a3inv,
                        GAMMA_MINUS1) - SPHP(i).Entropy) / dloga;

            if(SPHP(i).DtEntropy < -0.5 * SPHP(i).Entropy / dloga)
                SPHP(i).DtEntropy = -0.5 * SPHP(i).Entropy / dloga;
        }
    }
}

#if defined(SFR)


/* returns 0 if the particle is actively forming stars */
static int get_sfr_condition(int i) {
    int flag = 1;
/* no sfr !*/
    if(!All.StarformationOn) {
        return flag;
    }
    if(SPHP(i).Density * All.cf.a3inv >= All.PhysDensThresh)
        flag = 0;

    if(SPHP(i).Density < All.OverDensThresh)
        flag = 1;

    /* massless particles never form stars! */
    if(P[i].Mass == 0) {
        endrun(-1, "Encoutered zero mass particle during sfr ;"
                  " We haven't implemented tracer particles and this shall not happen\n");
    }

    if(SPHP(i).DelayTime > 0)
        flag = 1;		/* only normal cooling for particles in the wind */

    if(All.QuickLymanAlphaProbability > 0) {
        double dloga = get_dloga_for_bin(P[i].TimeBin);
        double unew = DMAX(All.MinEgySpec,
                (SPHP(i).Entropy + SPHP(i).DtEntropy * dloga) /
                GAMMA_MINUS1 * pow(SPHP(i).EOMDensity * All.cf.a3inv, GAMMA_MINUS1));

        double temp = u_to_temp_fac * unew;

        if(SPHP(i).Density > All.OverDensThresh && temp < 1.0e5)
            flag = 0;
        else
            flag = 1;
    }

    return flag;
}

/*These functions are for the wind models*/
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

static int make_particle_wind(MyIDType ID, int i, double v, double vmean[3]) {
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
        dir[0] = P[i].GravAccel[1] * vel[2] - P[i].GravAccel[2] * vel[1];
        dir[1] = P[i].GravAccel[2] * vel[0] - P[i].GravAccel[0] * vel[2];
        dir[2] = P[i].GravAccel[0] * vel[1] - P[i].GravAccel[1] * vel[0];
    }

    double norm = 0;
    for(j = 0; j < 3; j++)
        norm += dir[j] * dir[j];

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
/*End wind model functions*/

static int make_particle_star(int i) {
    double mass_of_star = find_star_mass(i);
    if(P[i].Type != 0)
        endrun(7772, "Only gas forms stars, what's wrong?");

    /* if we get all mass or a fraction */
    int child = domain_fork_particle(i, 4);

    /* ok, make a star */
    if(P[i].Mass < 1.1 * mass_of_star || All.QuickLymanAlphaProbability > 0)
    {
        /* here the gas particle is eliminated because remaining mass is all converted. */
        stars_converted++;

        P[child].Mass = P[i].Mass;
        P[i].Mass -= P[child].Mass;
        P[i].IsGarbage = 1;
    }
    else
    {
        /* FIXME: sorry this is not thread safe */
        stars_spawned++;

        P[child].Mass = mass_of_star;
        P[i].Mass -= P[child].Mass;
    }

    /*Set properties*/
    sum_mass_stars += P[child].Mass;
    STARP(child).FormationTime = All.Time;
    STARP(child).BirthDensity = SPHP(i).Density;
    /*Copy metallicity*/
    STARP(child).Metallicity = SPHP(i).Metallicity;
    P[child].IsNewParticle = 1;
    return 0;
}

static void cooling_relaxed(int i, double egyeff, double dtime, double trelax) {
    const double densityfac = pow(SPHP(i).EOMDensity * All.cf.a3inv, GAMMA_MINUS1) / GAMMA_MINUS1;
    double egycurrent = SPHP(i).Entropy *  densityfac;

#ifdef BLACK_HOLES
    if(SPHP(i).Injected_BH_Energy > 0)
    {
        struct UVBG uvbg;
        GetParticleUVBG(i, &uvbg);
        egycurrent += SPHP(i).Injected_BH_Energy / P[i].Mass;

        double temp = u_to_temp_fac * egycurrent;

        if(temp > 5.0e9)
            egycurrent = 5.0e9 / u_to_temp_fac;

        if(egycurrent > egyeff)
        {
            double ne = SPHP(i).Ne;
            double tcool = GetCoolingTime(egycurrent, SPHP(i).Density * All.cf.a3inv, &uvbg, &ne, SPHP(i).Metallicity);

            if(tcool < trelax && tcool > 0)
                trelax = tcool;
        }

        SPHP(i).Injected_BH_Energy = 0;
    }
#endif

    SPHP(i).Entropy =  (egyeff + (egycurrent - egyeff) * exp(-dtime / trelax)) /densityfac;

    SPHP(i).DtEntropy = 0;

}

static void starformation(int i) {

    double mass_of_star = find_star_mass(i);

    /*  the proper time-step */
    double dloga = get_dloga_for_bin(P[i].TimeBin);
    double dtime = dloga / All.cf.hubble;

    double egyeff, trelax;
    double rateOfSF = get_starformation_rate_full(i, dtime, &SPHP(i).Ne, &trelax, &egyeff);

    /* amount of stars expect to form */

    double sm = rateOfSF * dtime;

    double p = sm / P[i].Mass;

    sum_sm += P[i].Mass * (1 - exp(-p));

    /* convert to Solar per Year but is this damn variable otherwise used
     * at all? */
    SPHP(i).Sfr = rateOfSF *
        (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);

    double w = get_random_number(P[i].ID);
    SPHP(i).Metallicity += w * METAL_YIELD * (1 - exp(-p));

    if(dloga > 0 && P[i].TimeBin)
    {
      	/* upon start-up, we need to protect against dloga ==0 */
        cooling_relaxed(i, egyeff, dtime, trelax);
    }

    double prob = P[i].Mass / mass_of_star * (1 - exp(-p));

    if(All.QuickLymanAlphaProbability > 0.0) {
        prob = All.QuickLymanAlphaProbability;
    }
    if(get_random_number(P[i].ID + 1) < prob) {
#pragma omp critical (_sfr_)
        make_particle_star(i);
    }

    if(P[i].Type == 0)	{
        /* to protect using a particle that has been turned into a star */
        SPHP(i).Metallicity += (1 - w) * METAL_YIELD * (1 - exp(-p));
        if(All.WindOn && HAS(All.WindModel, WIND_SUBGRID)) {
            /* Here comes the Springel Hernquist 03 wind model */
            double pw = All.WindEfficiency * sm / P[i].Mass;
            double prob = 1 - exp(-pw);
            double zero[3] = {0, 0, 0};
            if(get_random_number(P[i].ID + 2) < prob)
                make_particle_wind(P[i].ID, i, All.WindSpeed * All.cf.a, zero);
        }
    }


}

double get_starformation_rate(int i) {
    /* returns SFR in internal units */
    return get_starformation_rate_full(i, 0, NULL, NULL, NULL);
}

static double get_starformation_rate_full(int i, double dtime, MyFloat * ne_new, double * trelax, double * egyeff) {
    double rateOfSF;
    int flag;
    double tsfr;
    double factorEVP, egyhot, ne, tcool, y, x, cloudmass;
    struct UVBG uvbg;

    flag = get_sfr_condition(i);

    if(flag == 1) {
        /* this shall not happen but let's put in some safe
         * numbers in case the code run wary!
         *
         * the only case trelax and egyeff are
         * required is in starformation(i)
         * */
        if (trelax) {
            *trelax = All.MaxSfrTimescale;
        }
        if (egyeff) {
            *egyeff = All.EgySpecCold;
        }
        return 0;
    }

    tsfr = sqrt(All.PhysDensThresh / (SPHP(i).Density * All.cf.a3inv)) * All.MaxSfrTimescale;
    /*
     * gadget-p doesn't have this cap.
     * without the cap sm can be bigger than cloudmass.
    */
    if(tsfr < dtime)
        tsfr = dtime;

    GetParticleUVBG(i, &uvbg);

    factorEVP = pow(SPHP(i).Density * All.cf.a3inv / All.PhysDensThresh, -0.8) * All.FactorEVP;

    egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;

    ne = SPHP(i).Ne;

    tcool = GetCoolingTime(egyhot, SPHP(i).Density * All.cf.a3inv, &uvbg, &ne, SPHP(i).Metallicity);
    y = tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);

    x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));

    cloudmass = x * P[i].Mass;

    rateOfSF = (1 - All.FactorSN) * cloudmass / tsfr;

    if (ne_new ) {
        *ne_new = ne;
    }

    if (trelax) {
        *trelax = tsfr * (1 - x) / x / (All.FactorSN * (1 + factorEVP));
    }
    if (egyeff) {
        *egyeff = egyhot * (1 - x) + All.EgySpecCold * x;
    }

    if (HAS(All.StarformationCriterion, SFR_CRITERION_MOLECULAR_H2)) {
        rateOfSF *= get_sfr_factor_due_to_h2(i);
    }
    if (HAS(All.StarformationCriterion, SFR_CRITERION_SELFGRAVITY)) {
        rateOfSF *= get_sfr_factor_due_to_selfgravity(i);
    }
    return rateOfSF;
}

void init_clouds(void)
{
    if(!All.StarformationOn) return;

    double A0, dens, tcool, ne, coolrate, egyhot, x, u4, meanweight;
    double tsfr, y, peff, fac, neff, egyeff, factorEVP, sigma, thresholdStarburst;

    if(All.PhysDensThresh == 0)
    {
        A0 = All.FactorEVP;

        egyhot = All.EgySpecSN / A0;

        meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));	/* note: assuming FULL ionization */

        u4 = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * 1.0e4;
        u4 *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;


        dens = 1.0e6 * 3 * All.CP.Hubble * All.CP.Hubble / (8 * M_PI * All.G);

        /* to be guaranteed to get z=0 rate */
        set_global_time(1.0);

        ne = 1.0;

        SetZeroIonization();
        struct UVBG uvbg;
        GetGlobalUVBG(&uvbg);
        /*XXX: We set the threshold without metal cooling;
         * It probably make sense to set the parameters with
         * a metalicity dependence.
         * */
        tcool = GetCoolingTime(egyhot, dens, &uvbg, &ne, 0.0);

        coolrate = egyhot / tcool / dens;

        x = (egyhot - u4) / (egyhot - All.EgySpecCold);

        All.PhysDensThresh =
            x / pow(1 - x,
                    2) * (All.FactorSN * All.EgySpecSN - (1 -
                            All.FactorSN) * All.EgySpecCold) /
                        (All.MaxSfrTimescale * coolrate);

        message(0, "A0= %g  \n", A0);
        message(0, "Computed: PhysDensThresh= %g  (int units)         %g h^2 cm^-3\n", All.PhysDensThresh,
                All.PhysDensThresh / (PROTONMASS / HYDROGEN_MASSFRAC / All.UnitDensity_in_cgs));
        message(0, "EXPECTED FRACTION OF COLD GAS AT THRESHOLD = %g\n", x);
        message(0, "tcool=%g dens=%g egyhot=%g\n", tcool, dens, egyhot);

        dens = All.PhysDensThresh * 10;

        do
        {
            tsfr = sqrt(All.PhysDensThresh / (dens)) * All.MaxSfrTimescale;
            factorEVP = pow(dens / All.PhysDensThresh, -0.8) * All.FactorEVP;
            egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;

            ne = 0.5;
            tcool = GetCoolingTime(egyhot, dens, &uvbg, &ne, 0.0);

            y = tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);
            x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));
            egyeff = egyhot * (1 - x) + All.EgySpecCold * x;

            peff = GAMMA_MINUS1 * dens * egyeff;

            fac = 1 / (log(dens * 1.025) - log(dens));
            dens *= 1.025;

            neff = -log(peff) * fac;

            tsfr = sqrt(All.PhysDensThresh / (dens)) * All.MaxSfrTimescale;
            factorEVP = pow(dens / All.PhysDensThresh, -0.8) * All.FactorEVP;
            egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;

            ne = 0.5;
            tcool = GetCoolingTime(egyhot, dens, &uvbg, &ne, 0.0);

            y = tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);
            x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));
            egyeff = egyhot * (1 - x) + All.EgySpecCold * x;

            peff = GAMMA_MINUS1 * dens * egyeff;

            neff += log(peff) * fac;
        }
        while(neff > 4.0 / 3);

        thresholdStarburst = dens;

        message(0, "Run-away sets in for dens=%g\n", thresholdStarburst);
        message(0, "Dynamic range for quiescent star formation= %g\n", thresholdStarburst / All.PhysDensThresh);

        sigma = 10.0 / All.CP.Hubble * 1.0e-10 / pow(1.0e-3, 2);

        message(0, "Isotherm sheet central density: %g   z0=%g\n",
                M_PI * All.G * sigma * sigma / (2 * GAMMA_MINUS1) / u4,
                GAMMA_MINUS1 * u4 / (2 * M_PI * All.G * sigma));

    }
}

/********************
 *
 * The follow functions are from Desika and Gadget-P.
 * We really are mostly concerned about H2 here.
 *
 * You may need a license to run with these modess.
 
 * */
#if defined SPH_GRAD_RHO
static double ev_NH_from_GradRho(MyFloat gradrho[3], double hsml, double rho, double include_h)
{
    /* column density from GradRho, copied from gadget-p; what is it
     * calculating? */
    double gradrho_mag;
    if(rho<=0) {
        gradrho_mag = 0;
    } else {
        gradrho_mag = sqrt(gradrho[0]*gradrho[0]+gradrho[1]*gradrho[1]+gradrho[2]*gradrho[2]);
        if(gradrho_mag > 0) {gradrho_mag = rho*rho/gradrho_mag;} else {gradrho_mag=0;}
        if(include_h > 0) gradrho_mag += include_h*rho*hsml;
    }
    return gradrho_mag; // *(Z/Zsolar) add metallicity dependence
}
#endif

static double get_sfr_factor_due_to_h2(int i) {
    /*  Krumholz & Gnedin fitting function for f_H2 as a function of local
     *  properties, from gadget-p; we return the enhancement on SFR in this
     *  function */

#if ! defined SPH_GRAD_RHO
    /* if SPH_GRAD_RHO is not enabled, disable H2 molecular gas
     * this really shall not happen because begrun will check against the
     * condition. Ditto if not metal tracking.
     * */
    return 1.0;
#else
    double tau_fmol;
    double zoverzsun = SPHP(i).Metallicity/METAL_YIELD;
    tau_fmol = ev_NH_from_GradRho(SPHP(i).GradRho,P[i].Hsml,SPHP(i).Density,1) * All.cf.a2inv;
    tau_fmol *= (0.1 + zoverzsun);
    if(tau_fmol>0) {
        tau_fmol *= 434.78*All.UnitDensity_in_cgs*All.CP.HubbleParam*All.UnitLength_in_cm;
        double y = 0.756*(1+3.1*pow(zoverzsun,0.365));
        y = log(1+0.6*y+0.01*y*y)/(0.6*tau_fmol);
        y = 1-0.75*y/(1+0.25*y);
        if(y<0) y=0;
        if(y>1) y=1;
        return y;

    } // if(tau_fmol>0)
    return 1.0;
#endif
}

static double get_sfr_factor_due_to_selfgravity(int i) {
    double divv = SPHP(i).DivVel * All.cf.a2inv;

    divv += 3.0*All.cf.hubble_a2; // hubble-flow correction

    if(HAS(All.StarformationCriterion, SFR_CRITERION_CONVERGENT_FLOW)) {
        if( divv>=0 ) return 0; // restrict to convergent flows (optional) //
    }

    double dv2abs = (divv*divv
            + (SPHP(i).CurlVel*All.cf.a2inv)
            * (SPHP(i).CurlVel*All.cf.a2inv)
           ); // all in physical units
    double alpha_vir = 0.2387 * dv2abs/(All.G * SPHP(i).Density*All.cf.a3inv);

    double y = 1.0;

    if((alpha_vir < 1.0)
    || (SPHP(i).Density * All.cf.a3inv > 100. * All.PhysDensThresh)
    )  {
        y = 66.7;
    } else {
        y = 0.1;
    }
    // PFH: note the latter flag is an arbitrary choice currently set
    // -by hand- to prevent runaway densities from this prescription! //

    if (HAS(All.StarformationCriterion, SFR_CRITERION_CONTINUOUS_CUTOFF)) {
        // continuous cutoff w alpha_vir instead of sharp (optional) //
        y *= 1.0/(1.0 + alpha_vir);
    }
    return y;
}

static double
find_star_mass(int i)
{
    double mass_of_star =  All.MassTable[0] / GENERATIONS;
    if(mass_of_star > P[i].Mass) {
        /* if some mass has been stolen by BH, e.g */
        mass_of_star = P[i].Mass;
    }
    /* if we are the last particle */
    if(fabs(mass_of_star - P[i].Mass) / mass_of_star < 0.5) {
        mass_of_star = P[i].Mass;
    }
    return mass_of_star;
}

#endif

