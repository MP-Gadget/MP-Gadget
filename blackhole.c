#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "cooling.h"
#include "densitykernel.h"
#include "treewalk.h"
#include "slotsmanager.h"
#include "mymalloc.h"
#include "endrun.h"
#include "drift.h"
#include "system.h"
#include "fof.h"
#include "blackhole.h"
#include "forcetree.h"
#include "sfr_eff.h"
#include "timestep.h"
/*! \file blackhole.c
 *  \brief routines for gas accretion onto black holes, and black hole mergers
 */

#ifdef BLACK_HOLES

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
    MyFloat BH_MinPotVel[3];
    MyFloat BH_MinPot;

    int BH_TimeBinLimit;
    MyFloat FeedbackWeightSum;

    MyFloat Rho;
    MyFloat SmoothedEntropy;
    MyFloat SmoothedPressure;
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

static void
blackhole_accretion_ngbiter(TreeWalkQueryBHAccretion * I,
        TreeWalkResultBHAccretion * O,
        TreeWalkNgbIterBHAccretion * iter,
        LocalTreeWalk * lv);

/* feedback routines */

static void
blackhole_feedback_preprocess(int n, TreeWalk * tw);

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

static int N_sph_swallowed, N_BH_swallowed;

static double blackhole_soundspeed(double entropy, double pressure, double rho) {
    /* rho is comoving !*/
    double cs;
    if (All.BlackHoleSoundSpeedFromPressure) {
        cs = sqrt(GAMMA * pressure / rho);
    } else {
        cs = sqrt(GAMMA * entropy *
                pow(rho, GAMMA_MINUS1));
    }

    cs *= pow(All.Time, -1.5 * GAMMA_MINUS1);

    return cs;
}

void blackhole(void)
{
    if(!All.BlackHoleOn) return;
    int i;
    int Ntot_gas_swallowed, Ntot_BH_swallowed;

    walltime_measure("/Misc");
    TreeWalk tw_accretion[1] = {0};

    tw_accretion->ev_label = "BH_ACCRETION";
    tw_accretion->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw_accretion->ngbiter_type_elsize = sizeof(TreeWalkNgbIterBHAccretion);
    tw_accretion->ngbiter = (TreeWalkNgbIterFunction) blackhole_accretion_ngbiter;
    tw_accretion->haswork = blackhole_accretion_haswork;
    tw_accretion->postprocess = (TreeWalkProcessFunction) blackhole_accretion_postprocess;
    tw_accretion->fill = (TreeWalkFillQueryFunction) blackhole_accretion_copy;
    tw_accretion->reduce = (TreeWalkReduceResultFunction) blackhole_accretion_reduce;
    tw_accretion->UseNodeList = 1;
    tw_accretion->query_type_elsize = sizeof(TreeWalkQueryBHAccretion);
    tw_accretion->result_type_elsize = sizeof(TreeWalkResultBHAccretion);

    TreeWalk tw_feedback[1] = {0};
    tw_feedback->ev_label = "BH_FEEDBACK";
    tw_feedback->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw_feedback->ngbiter_type_elsize = sizeof(TreeWalkNgbIterBHFeedback);
    tw_feedback->ngbiter = (TreeWalkNgbIterFunction) blackhole_feedback_ngbiter;
    tw_feedback->haswork = blackhole_feedback_haswork;
    tw_feedback->fill = (TreeWalkFillQueryFunction) blackhole_feedback_copy;
    tw_feedback->postprocess = (TreeWalkProcessFunction) blackhole_feedback_postprocess;
    tw_feedback->preprocess = (TreeWalkProcessFunction) blackhole_feedback_preprocess;
    tw_feedback->reduce = (TreeWalkReduceResultFunction) blackhole_feedback_reduce;
    tw_feedback->UseNodeList = 1;
    tw_feedback->query_type_elsize = sizeof(TreeWalkQueryBHFeedback);
    tw_feedback->result_type_elsize = sizeof(TreeWalkResultBHFeedback);

    message(0, "Beginning black-hole accretion\n");

    N_sph_swallowed = N_BH_swallowed = 0;

    /* Let's determine which particles may be swalled and calculate total feedback weights */

    treewalk_run(tw_accretion, ActiveParticle, NumActiveParticle);

    message(0, "Start swallowing of gas particles and black holes\n");

    /* Now do the swallowing of particles and dump feedback energy */
    treewalk_run(tw_feedback, ActiveParticle, NumActiveParticle);

    MPI_Reduce(&N_sph_swallowed, &Ntot_gas_swallowed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&N_BH_swallowed, &Ntot_BH_swallowed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    message(0, "Accretion done: %d gas particles swallowed, %d BH particles swallowed\n",
                Ntot_gas_swallowed, Ntot_BH_swallowed);

    double total_mass_real, total_mdoteddington;
    double total_mass_holes, total_mdot;

    double Local_BH_mass = 0;
    double Local_BH_dynamicalmass = 0;
    double Local_BH_Mdot = 0;
    double Local_BH_Medd = 0;
    /* Compute total mass of black holes
     * present by summing contents of black hole array*/
    for(i = 0; i < SlotsManager->info[5].size; i ++)
    {
        Local_BH_mass += BhP[i].Mass;
        Local_BH_Mdot += BhP[i].Mdot;
        Local_BH_Medd += BhP[i].Mdot/BhP[i].Mass;
        /* FIXME : this is broken. ReverseLink is always -1 get rid of this metric?*/
        // Local_BH_dynamicalmass += P[BhP[i].ReverseLink].Mass;
    }

    MPI_Reduce(&Local_BH_mass, &total_mass_holes, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Local_BH_dynamicalmass, &total_mass_real, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Local_BH_Mdot, &total_mdot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Local_BH_Medd, &total_mdoteddington, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(ThisTask == 0)
    {
        /* convert to solar masses per yr */
        double mdot_in_msun_per_year =
            total_mdot * (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);

        total_mdoteddington *= 1.0 / ((4 * M_PI * GRAVITY * C * PROTONMASS /
                    (0.1 * C * C * THOMPSON)) * All.UnitTime_in_s);

        fprintf(FdBlackHoles, "%g %d %g %g %g %g %g\n",
                All.Time, SlotsManager->info[5].size, total_mass_holes, total_mdot, mdot_in_msun_per_year,
                total_mass_real, total_mdoteddington);
        fflush(FdBlackHoles);
    }

    /* this will find new black hole seed halos */
    if(All.Time >= All.TimeNextSeedingCheck)
    {
        /* Seeding */
        fof_fof();
        fof_seed();
        fof_finish();
        All.TimeNextSeedingCheck *= All.TimeBetweenSeedingSearch;
    }
    walltime_measure("/BH");
}

static void
blackhole_accretion_postprocess(int i, TreeWalk * tw)
{
    if(BHP(i).Density > 0)
    {
        BHP(i).Entropy /= BHP(i).Density;
        BHP(i).Pressure /= BHP(i).Density;

        BHP(i).SurroundingGasVel[0] /= BHP(i).Density;
        BHP(i).SurroundingGasVel[1] /= BHP(i).Density;
        BHP(i).SurroundingGasVel[2] /= BHP(i).Density;
    }
    double mdot = 0;		/* if no accretion model is enabled, we have mdot=0 */

    double rho = BHP(i).Density;
    double bhvel = sqrt(pow(P[i].Vel[0] - BHP(i).SurroundingGasVel[0], 2) +
            pow(P[i].Vel[1] - BHP(i).SurroundingGasVel[1], 2) +
            pow(P[i].Vel[2] - BHP(i).SurroundingGasVel[2], 2));

    bhvel /= All.cf.a;
    double rho_proper = rho * All.cf.a3inv;

    double soundspeed = blackhole_soundspeed(BHP(i).Entropy, BHP(i).Pressure, rho);

    /* Note: we take here a radiative efficiency of 0.1 for Eddington accretion */
    double meddington = (4 * M_PI * GRAVITY * C * PROTONMASS / (0.1 * C * C * THOMPSON)) * BHP(i).Mass
        * All.UnitTime_in_s;

    double norm = pow((pow(soundspeed, 2) + pow(bhvel, 2)), 1.5);

    if(norm > 0)
        mdot = 4. * M_PI * All.BlackHoleAccretionFactor * All.G * All.G *
            BHP(i).Mass * BHP(i).Mass * rho_proper / norm;
    else
        mdot = 0;

    if(All.BlackHoleEddingtonFactor > 0.0 &&
        mdot > All.BlackHoleEddingtonFactor * meddington) {
        mdot = All.BlackHoleEddingtonFactor * meddington;
    }
    BHP(i).Mdot = mdot;

    double dtime = get_dloga_for_bin(P[i].TimeBin) / All.cf.hubble;

    BHP(i).Mass += BHP(i).Mdot * dtime;
}

static void
blackhole_feedback_preprocess(int n, TreeWalk * tw)
{
    int j;
    for(j = 0; j < 3; j++) {
        BHP(n).MinPotPos[j] = P[n].Pos[j];
        BHP(n).MinPotVel[j] = P[n].Vel[j];
    }
    BHP(n).MinPot = P[n].Potential;
}

static void
blackhole_feedback_postprocess(int n, TreeWalk * tw)
{
    if(BHP(n).accreted_Mass > 0)
    {
        P[n].Mass += BHP(n).accreted_Mass;
        BHP(n).Mass += BHP(n).accreted_BHMass;
        BHP(n).accreted_Mass = 0;
    }
}

static void
blackhole_accretion_ngbiter(TreeWalkQueryBHAccretion * I,
        TreeWalkResultBHAccretion * O,
        TreeWalkNgbIterBHAccretion * iter,
        LocalTreeWalk * lv)
{

    if(iter->base.other == -1) {
        O->BH_TimeBinLimit = -1;
        O->BH_MinPot = BHPOTVALUEINIT;
        int d;
        for(d = 0; d < 3; d++) {
            O->BH_MinPotPos[d] = I->base.Pos[d];
            O->BH_MinPotVel[d] = I->Vel[d];
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
        if (O->BH_TimeBinLimit <= 0 || O->BH_TimeBinLimit >= P[other].TimeBin)
            O->BH_TimeBinLimit = P[other].TimeBin;
    }

#ifdef SFR
     /* BH does not accrete wind */
    if(P[other].Type == 0 && SPHP(other).DelayTime > 0) return;
#endif

    /* Drifting the blackhole towards minimum. This shall be refactored to some sink.c etc */
    if(r2 < iter->accretion_kernel.HH) // && r < All.FOFHaloComovingLinkingLength)
    {
        if(P[other].Potential < O->BH_MinPot)
        {
            if(P[other].Type == 0 || P[other].Type == 1 || P[other].Type == 4 || P[other].Type == 5)
            {
                /* FIXME: compute peculier velocities between two objects; this shall be a function */
                int d;
                double vrel[3];
                for(d = 0; d < 3; d++)
                    vrel[d] = (P[other].Vel[d] - I->Vel[d]);

                double vpec = sqrt(dotproduct(vrel, vrel)) / All.cf.a;

                if(vpec <= 0.25 * I->Csnd)
                {
                    O->BH_MinPot = P[other].Potential;
                    for(d = 0; d < 3; d++) {
                        O->BH_MinPotPos[d] = P[other].Pos[d];
                        O->BH_MinPotVel[d] = P[other].Vel[d];
                    }
                }
            }
        }
    }

    /* Accretion / merger doesn't do self iteraction */
    if(P[other].ID == I->ID) return;

    if(P[other].Type == 5 && r2 < iter->accretion_kernel.HH)	/* we have a black hole merger */
    {
        /* compute relative velocity of BHs */

        lock_particle(other);
        int d;
        double vrel[3];
        for(d = 0; d < 3; d++)
            vrel[d] = (P[other].Vel[d] - I->Vel[d]);

        double vpec = sqrt(dotproduct(vrel, vrel)) / All.cf.a;

        if(vpec <= 0.5 * I->Csnd)
        {
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
        }
        unlock_particle(other);
    }

    if(P[other].Type == 0) {
        if(r2 < iter->accretion_kernel.HH) {
            double u = r * iter->accretion_kernel.Hinv;
            double wk = density_kernel_wk(&iter->accretion_kernel, u);
            double mass_j = P[other].Mass;
            double VelPred[3];
            sph_VelPred(other, VelPred);

            /* FIXME: volume correction doesn't work on BH yet. */
            O->Rho += (mass_j * wk);

            O->SmoothedPressure += (mass_j * wk * PressurePred(other));
            O->SmoothedEntropy += (mass_j * wk * SPHP(other).Entropy);
            O->GasVel[0] += (mass_j * wk * VelPred[0]);
            O->GasVel[1] += (mass_j * wk * VelPred[1]);
            O->GasVel[2] += (mass_j * wk * VelPred[2]);

            /* here we have a gas particle; check for swallowing */

            lock_particle(other);
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
                if(P[other].Swallowed) {
                    /* Already marked, prefer to be swallowed by a bigger ID */
                    if(SPHP(other).SwallowID < I->ID) {
                        SPHP(other).SwallowID = I->ID;
                    }
                } else {
                    /* Unmarked mark it */
                    P[other].Swallowed = 1;
                    SPHP(other).SwallowID = I->ID;
                }
            }
            unlock_particle(other);
        }

        if(r2 < iter->feedback_kernel.HH) {
            /* update the feedback weighting */
            double mass_j;
            if(HAS(All.BlackHoleFeedbackMethod, BH_FEEDBACK_OPTTHIN)) {
                double nh0 = 1.0;
                double nHeII = 0;
                double ne = SPHP(other).Ne;
                struct UVBG uvbg;
                GetParticleUVBG(other, &uvbg);
                AbundanceRatios(DMAX(All.MinEgySpec,
                            SPHP(other).Entropy / GAMMA_MINUS1
                            * pow(SPHP(other).EOMDensity * All.cf.a3inv,
                                GAMMA_MINUS1)),
                        SPHP(other).Density * All.cf.a3inv, &uvbg, &ne, &nh0, &nHeII);
                if(r2 > 0)
                    O->FeedbackWeightSum += (P[other].Mass * nh0) / r2;
            } else {
                if(HAS(All.BlackHoleFeedbackMethod, BH_FEEDBACK_MASS)) {
                    mass_j = P[other].Mass;
                } else {
                    mass_j = P[other].Hsml * P[other].Hsml * P[other].Hsml;
                }
                if(HAS(All.BlackHoleFeedbackMethod, BH_FEEDBACK_SPLINE)) {
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

#ifdef SFR
     /* BH does not accrete wind */
    if(P[other].Type == 0 && SPHP(other).DelayTime > 0) return;
#endif

    if(P[other].Swallowed && P[other].Type == 5)	/* we have a black hole merger */
    {
        if(BHP(other).SwallowID != I->ID) return;

        lock_particle(other);

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

        P[other].IsGarbage = 1;
        BHP(other).Mdot = 0;

#pragma omp atomic
        N_BH_swallowed++;

        unlock_particle(other);
    }

    /* Dump feedback energy */
    if(P[other].Type == 0) {
        if(r2 < iter->feedback_kernel.HH && P[other].Mass > 0) {
            double u = r * iter->feedback_kernel.Hinv;
            double wk;
            double mass_j;

            lock_particle(other);

            if(HAS(All.BlackHoleFeedbackMethod, BH_FEEDBACK_MASS)) {
                mass_j = P[other].Mass;
            } else {
                mass_j = P[other].Hsml * P[other].Hsml * P[other].Hsml;
            }
            if(HAS(All.BlackHoleFeedbackMethod, BH_FEEDBACK_SPLINE))
                wk = density_kernel_wk(&iter->feedback_kernel, u);
            else
            wk = 1.0;

            if(I->FeedbackWeightSum > 0)
            {
                SPHP(other).Injected_BH_Energy += (I->FeedbackEnergy * mass_j * wk / I->FeedbackWeightSum);
            }

            unlock_particle(other);
        }
    }

    /* Swallowing a gas */
    if(P[other].Swallowed && P[other].Type == 0)
    {
        if(SPHP(other).SwallowID != I->ID) return;

        lock_particle(other);

        int d;
        for(d = 0; d < 3; d++)
            O->AccretedMomentum[d] += (P[other].Mass * P[other].Vel[d]);

        /* We do not know how to notify the tree of mass changes. so 
         * blindly enforce a mass conservation for now. */
        O->Mass += (P[other].Mass);
        P[other].Mass = 0;

        P[other].IsGarbage = 1;
#pragma omp atomic
        N_sph_swallowed++;
        unlock_particle(other);
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
    if(mode == 0 || BHP(place).MinPot > remote->BH_MinPot)
    {
        BHP(place).MinPot = remote->BH_MinPot;
        for(k = 0; k < 3; k++) {
            /* Movement occurs in predict.c */
            BHP(place).MinPotPos[k] = remote->BH_MinPotPos[k];
            BHP(place).MinPotVel[k] = remote->BH_MinPotVel[k];
        }
    }
    if (mode == 0 ||
            BHP(place).TimeBinLimit < 0 ||
            BHP(place).TimeBinLimit > remote->BH_TimeBinLimit) {
        BHP(place).TimeBinLimit = remote->BH_TimeBinLimit;
    }

    TREEWALK_REDUCE(BHP(place).Density, remote->Rho);
    TREEWALK_REDUCE(BHP(place).FeedbackWeightSum, remote->FeedbackWeightSum);
    TREEWALK_REDUCE(BHP(place).Entropy, remote->SmoothedEntropy);
    TREEWALK_REDUCE(BHP(place).Pressure, remote->SmoothedPressure);

    TREEWALK_REDUCE(BHP(place).SurroundingGasVel[0], remote->GasVel[0]);
    TREEWALK_REDUCE(BHP(place).SurroundingGasVel[1], remote->GasVel[1]);
    TREEWALK_REDUCE(BHP(place).SurroundingGasVel[2], remote->GasVel[2]);
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
    I->Csnd = blackhole_soundspeed(
                BHP(place).Entropy,
                BHP(place).Pressure,
                BHP(place).Density);
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
    I->FeedbackWeightSum = BHP(i).FeedbackWeightSum;

    double dtime = get_dloga_for_bin(P[i].TimeBin) / All.cf.hubble;

    I->FeedbackEnergy = All.BlackHoleFeedbackFactor * 0.1 * BHP(i).Mdot * dtime *
                pow(C / All.UnitVelocity_in_cm_per_s, 2);
}

static void
blackhole_feedback_reduce(int place, TreeWalkResultBHFeedback * remote, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    int k;

    TREEWALK_REDUCE(BHP(place).accreted_Mass, remote->Mass);
    TREEWALK_REDUCE(BHP(place).accreted_BHMass, remote->BH_Mass);
    for(k = 0; k < 3; k++) {
        TREEWALK_REDUCE(BHP(place).accreted_momentum[k], remote->AccretedMomentum[k]);
    }
    TREEWALK_REDUCE(BHP(place).CountProgs, remote->BH_CountProgs);
}

void blackhole_make_one(int index) {
    if(P[index].Type != 0) 
        endrun(7772, "Only Gas turns into blackholes, what's wrong?");

    int child = slots_fork(index, 5);

    BHP(child).FormationTime = All.Time;
    /*Ensure that mass is conserved*/
    double BHmass = All.SeedBlackHoleMass;
    if(P[index].Mass <= All.SeedBlackHoleMass) {
        P[index].IsGarbage = 1;
        BHmass = P[index].Mass;
    }
    P[child].Mass = BHmass;
    P[index].Mass -= BHmass;
    BHP(child).base.ID = P[child].ID;
    BHP(child).Mass = BHmass;
    BHP(child).Mdot = 0;

    /* It is important to initialize MinPotPos to the current position of 
     * a BH to avoid drifting to unknown locations (0,0,0) immediately 
     * after the BH is created. */
    int j;
    for(j = 0; j < 3; j++) {
        BHP(child).MinPotPos[j] = P[child].Pos[j];
        BHP(child).MinPotVel[j] = P[child].Vel[j];
    }

    BHP(child).MinPot = P[child].Potential;
    BHP(child).CountProgs = 1;
}

static double
decide_hsearch(double h)
{
    if(All.BlackHoleFeedbackRadius > 0) {
        /* BlackHoleFeedbackRadius is in comoving.
         * The Phys radius is capped by BlackHoleFeedbackRadiusMaxPhys
         * just like how it was done for grav smoothing.
         * */
        double rds;
        rds = All.BlackHoleFeedbackRadiusMaxPhys / All.cf.a;

        if(rds > All.BlackHoleFeedbackRadius) {
            rds = All.BlackHoleFeedbackRadius;
        }
        return rds;
    } else {
        return h;
    }
}


#endif
