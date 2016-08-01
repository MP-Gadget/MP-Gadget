#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "densitykernel.h"
#include "proto.h"
#include "forcetree.h"
#include "treewalk.h"
#include "domain.h"
#include "mymalloc.h"
#include "endrun.h"
/*! \file blackhole.c
 *  \brief routines for gas accretion onto black holes, and black hole mergers
 */

#ifdef BLACK_HOLES

typedef struct {
    TreeWalkQueryBase base;
    MyFloat Density;
    MyFloat FeedbackWeightSum;
    MyFloat Mdot;
    MyFloat Dt;
    MyFloat Hsml;
    MyFloat Mass;
    MyFloat BH_Mass;
    MyFloat Vel[3];
    MyFloat Csnd;
    MyIDType ID;
} TreeWalkQueryBHFeedback;

typedef struct {
    TreeWalkResultBase base;
    MyFloat BH_MinPotPos[3];
    MyFloat BH_MinPotVel[3];
    MyFloat BH_MinPot;

    short int BH_TimeBinLimit;
} TreeWalkResultBHFeedback;

typedef struct {
    TreeWalkNgbIterBase base;
    DensityKernel kernel;
    DensityKernel bh_feedback_kernel;
} TreeWalkNgbIterBHFeedback;

typedef struct {
    TreeWalkQueryBase base;
    MyFloat Hsml;
    MyFloat BH_Mass;
    MyIDType ID;
} TreeWalkQuerySwallow;

typedef struct {
    TreeWalkResultBase base;
    MyDouble Mass;
    MyDouble BH_Mass;
    MyDouble AccretedMomentum[3];
    int BH_CountProgs;
} TreeWalkResultSwallow;

typedef struct {
    TreeWalkNgbIterBase base;
} TreeWalkNgbIterSwallow;

/* accretion routines */
static void blackhole_accretion_visit(int n);
static void blackhole_postprocess(int n);

/* feedback routines. currently also performs the drifting(move it to gravtree / force tree?) */
static int
blackhole_feedback_isactive(int n);

static void
blackhole_feedback_reduce(int place, TreeWalkResultBHFeedback * remote, enum TreeWalkReduceMode mode);

static void
blackhole_feedback_copy(int place, TreeWalkQueryBHFeedback * I);

static void
blackhole_feedback_ngbiter(TreeWalkQueryBHFeedback * I,
        TreeWalkResultBHFeedback * O,
        TreeWalkNgbIterBHFeedback * iter,
        LocalTreeWalk * lv);

/* swallow routines */
static int
blackhole_swallow_isactive(int n);

static void
blackhole_swallow_reduce(int place, TreeWalkResultSwallow * remote, enum TreeWalkReduceMode mode);

static void
blackhole_swallow_copy(int place, TreeWalkQuerySwallow * I);

static void
blackhole_swallow_ngbiter(TreeWalkQuerySwallow * I,
        TreeWalkResultSwallow * O,
        TreeWalkNgbIterSwallow * iter,
        LocalTreeWalk * lv);

#define BHPOTVALUEINIT 1.0e30

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

void blackhole_accretion(void)
{
    int i, n, bin;
    int Ntot_gas_swallowed, Ntot_BH_swallowed;

    walltime_measure("/Misc");
    TreeWalk fbev[1];

    fbev->ev_label = "BH_FEEDBACK";
    fbev->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    fbev->ngbiter_type_elsize = sizeof(TreeWalkNgbIterBHFeedback);
    fbev->ngbiter = (TreeWalkNgbIterFunction) blackhole_feedback_ngbiter;
    fbev->isactive = blackhole_feedback_isactive;
    fbev->fill = (TreeWalkFillQueryFunction) blackhole_feedback_copy;
    fbev->reduce = (TreeWalkReduceResultFunction) blackhole_feedback_reduce;
    fbev->UseNodeList = 1;
    fbev->query_type_elsize = sizeof(TreeWalkQueryBHFeedback);
    fbev->result_type_elsize = sizeof(TreeWalkResultBHFeedback);

    TreeWalk swev[1];
    swev->ev_label = "BH_SWALLOW";
    swev->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    swev->ngbiter_type_elsize = sizeof(TreeWalkNgbIterSwallow);
    swev->ngbiter = (TreeWalkNgbIterFunction) blackhole_swallow_ngbiter;
    swev->isactive = blackhole_swallow_isactive;
    swev->fill = (TreeWalkFillQueryFunction) blackhole_swallow_copy;
    swev->reduce = (TreeWalkReduceResultFunction) blackhole_swallow_reduce;
    swev->UseNodeList = 1;
    swev->query_type_elsize = sizeof(TreeWalkQuerySwallow);
    swev->result_type_elsize = sizeof(TreeWalkResultSwallow);

    message(0, "Beginning black-hole accretion\n");


    /* Let's first compute the Mdot values */
    int Nactive;
    int * queue = treewalk_get_queue(fbev, &Nactive);

    for(i = 0; i < Nactive; i ++) {
        int n = queue[i];

        Local_BH_mass -= BHP(n).Mass;
        Local_BH_dynamicalmass -= P[n].Mass;
        Local_BH_Mdot -= BHP(n).Mdot;
        if(BHP(n).Mass > 0) {
            Local_BH_Medd -= BHP(n).Mdot / BHP(n).Mass;
        }

        blackhole_accretion_visit(n);

        int j;
        for(j = 0; j < 3; j++) {
            BHP(n).MinPotPos[j] = P[n].Pos[j];
            BHP(n).MinPotVel[j] = P[n].Vel[j];
        }
        BHP(n).MinPot = P[n].Potential;
    }

    /* Now let's invoke the functions that stochasticall swallow gas
     * and deal with black hole mergers.
     */

    message(0, "Start swallowing of gas particles and black holes\n");


    N_sph_swallowed = N_BH_swallowed = 0;

    /* Let's first spread the feedback energy,
     * and determine which particles may be swalled by whom */

    treewalk_run(fbev);

    /* Now do the swallowing of particles */
    treewalk_run(swev);

    MPI_Reduce(&N_sph_swallowed, &Ntot_gas_swallowed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&N_BH_swallowed, &Ntot_BH_swallowed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    message(0, "Accretion done: %d gas particles swallowed, %d BH particles swallowed\n",
                Ntot_gas_swallowed, Ntot_BH_swallowed);


    for(i = 0; i < Nactive; i++) {
        int n = queue[i];
        blackhole_postprocess(n);

        Local_BH_mass += BHP(n).Mass;
        Local_BH_dynamicalmass += P[n].Mass;
        Local_BH_Mdot += BHP(n).Mdot;
        if(BHP(n).Mass > 0) {
            Local_BH_Medd += BHP(n).Mdot / BHP(n).Mass;
        }
    }

    myfree(queue);

    double total_mass_real, total_mdoteddington;
    double total_mass_holes, total_mdot;

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

        fprintf(FdBlackHoles, "%g %td %g %g %g %g %g\n",
                All.Time, All.TotN_bh, total_mass_holes, total_mdot, mdot_in_msun_per_year,
                total_mass_real, total_mdoteddington);
        fflush(FdBlackHoles);
    }

    walltime_measure("/BH");
}

static void blackhole_accretion_visit(int n) {
    double mdot = 0;		/* if no accretion model is enabled, we have mdot=0 */

    double rho = BHP(n).Density;
    double bhvel = sqrt(pow(P[n].Vel[0] - BHP(n).SurroundingGasVel[0], 2) +
            pow(P[n].Vel[1] - BHP(n).SurroundingGasVel[1], 2) +
            pow(P[n].Vel[2] - BHP(n).SurroundingGasVel[2], 2));

    bhvel /= All.cf.a;
    double rho_proper = rho * All.cf.a3inv;

    double soundspeed = blackhole_soundspeed(BHP(n).Entropy, BHP(n).Pressure, rho);

    /* Note: we take here a radiative efficiency of 0.1 for Eddington accretion */
    double meddington = (4 * M_PI * GRAVITY * C * PROTONMASS / (0.1 * C * C * THOMPSON)) * BHP(n).Mass
        * All.UnitTime_in_s;

    double norm = pow((pow(soundspeed, 2) + pow(bhvel, 2)), 1.5);

    if(norm > 0)
        mdot = 4. * M_PI * All.BlackHoleAccretionFactor * All.G * All.G *
            BHP(n).Mass * BHP(n).Mass * rho_proper / norm;
    else
        mdot = 0;

    if(All.BlackHoleEddingtonFactor > 0.0 && 
        mdot > All.BlackHoleEddingtonFactor * meddington) {
        mdot = All.BlackHoleEddingtonFactor * meddington;
    }
    BHP(n).Mdot = mdot;

    double dt = (P[n].TimeBin ? (1 << P[n].TimeBin) : 0) * All.Timebase_interval / All.cf.hubble;

    BHP(n).Mass += BHP(n).Mdot * dt;

}

static void blackhole_postprocess(int n) {
    if(BHP(n).accreted_Mass > 0)
    {
        P[n].Mass += BHP(n).accreted_Mass;
        BHP(n).Mass += BHP(n).accreted_BHMass;
        BHP(n).accreted_Mass = 0;
    }
}

static void
blackhole_feedback_ngbiter(TreeWalkQueryBHFeedback * I,
        TreeWalkResultBHFeedback * O,
        TreeWalkNgbIterBHFeedback * iter,
        LocalTreeWalk * lv)
{

    if(iter->base.other == -1) {
        O->BH_TimeBinLimit = -1;
        O->BH_MinPot = BHPOTVALUEINIT;
        double hsearch;
        hsearch = density_decide_hsearch(5, I->Hsml);

        iter->base.mask = 1 + 2 + 4 + 8 + 16 + 32;
        iter->base.Hsml = hsearch;
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;

        density_kernel_init(&iter->kernel, I->Hsml);
        density_kernel_init(&iter->bh_feedback_kernel, hsearch);
        return;
    }

    int other = iter->base.other;
    double r = iter->base.r;
    double r2 = iter->base.r2;
    double * dist = iter->base.dist;

    if(P[other].Mass < 0) return;

    if(P[other].Type != 5) {
        if (O->BH_TimeBinLimit <= 0 || O->BH_TimeBinLimit >= P[other].TimeBin)
            O->BH_TimeBinLimit = P[other].TimeBin;
    }

    /* Drifting the blackhole towards minimum. This shall be refactored to some sink.c etc */
    if(r2 < iter->kernel.HH && r2 < All.FOFHaloComovingLinkingLength)
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

    if(P[other].Type == 5 && r2 < iter->kernel.HH)	/* we have a black hole merger */
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
            if(P[other].SwallowID < I->ID && P[other].ID < I->ID)
                P[other].SwallowID = I->ID;
        }
        unlock_particle(other);
    }

    if(P[other].Type == 0) {
#ifdef WINDS
        /* BH does not accrete wind */
        if(SPHP(other).DelayTime > 0) return;
#endif
        if(r2 < iter->kernel.HH) {
            /* here we have a gas particle */

            lock_particle(other);

            double r = sqrt(r2);
            double u = r * iter->kernel.Hinv;
            double wk = density_kernel_wk(&iter->kernel, u);
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
                if(P[other].SwallowID < I->ID)
                    P[other].SwallowID = I->ID;
            }
            unlock_particle(other);
        }

        if(r2 < iter->bh_feedback_kernel.HH && P[other].Mass > 0) {
            double r = sqrt(r2);
            double u = r * iter->bh_feedback_kernel.Hinv;
            double wk;
            double mass_j;

            lock_particle(other);

            if(HAS(All.BlackHoleFeedbackMethod, BH_FEEDBACK_MASS)) {
                mass_j = P[other].Mass;
            } else {
                mass_j = P[other].Hsml * P[other].Hsml * P[other].Hsml;
            }
            if(HAS(All.BlackHoleFeedbackMethod, BH_FEEDBACK_SPLINE))
                wk = density_kernel_wk(&iter->bh_feedback_kernel, u);
            else
            wk = 1.0;
            double energy = All.BlackHoleFeedbackFactor * 0.1 * I->Mdot * I->Dt *
                pow(C / All.UnitVelocity_in_cm_per_s, 2);

            if(I->FeedbackWeightSum > 0)
            {
                SPHP(other).Injected_BH_Energy += (energy * mass_j * wk / I->FeedbackWeightSum);
            }

            unlock_particle(other);
        }
    }
}


/**
 * perform blackhole swallow / merger;
 */
static void
blackhole_swallow_ngbiter(TreeWalkQuerySwallow * I,
        TreeWalkResultSwallow * O,
        TreeWalkNgbIterSwallow * iter,
        LocalTreeWalk * lv)
{

    if(iter->base.other == -1) {
        iter->base.mask = 1 + 2 + 4 + 8 + 16 + 32;
        iter->base.Hsml = I->Hsml;
        iter->base.symmetric = NGB_TREEFIND_SYMMETRIC;
        return;
    }

    int other = iter->base.other;

    /* Exclude self interaction */

    if(P[other].SwallowID != I->ID) return;
    if(P[other].ID == I->ID) return;
    if(P[other].Type == 5)	/* we have a black hole merger */
    {
        lock_particle(other);
        O->Mass += (P[other].Mass);
        O->BH_Mass += (BHP(other).Mass);

        int d;
        for(d = 0; d < 3; d++)
            O->AccretedMomentum[d] += (P[other].Mass * P[other].Vel[d]);

        O->BH_CountProgs += BHP(other).CountProgs;

#pragma omp atomic
        Local_BH_mass -= BHP(other).Mass;
#pragma omp atomic
        Local_BH_dynamicalmass -= P[other].Mass;
#pragma omp atomic
        Local_BH_Mdot -= BHP(other).Mdot;
        if(BHP(other).Mass > 0) {
#pragma omp atomic
            Local_BH_Medd -= BHP(other).Mdot / BHP(other).Mass;
        }

        P[other].Mass = 0;
        BHP(other).Mass = 0;
        BHP(other).Mdot = 0;

#pragma omp atomic
        N_BH_swallowed++;

        unlock_particle(other);
    }

    /* Swallow a gas */
    if(P[other].Type == 0)
    {
        lock_particle(other);

        O->Mass += (P[other].Mass);

        int d;
        for(d = 0; d < 3; d++)
            O->AccretedMomentum[d] += (P[other].Mass * P[other].Vel[d]);

        P[other].Mass = 0;
#pragma omp atomic
        N_sph_swallowed++;
        unlock_particle(other);
    }
}

static int blackhole_feedback_isactive(int n) {
    return (P[n].Type == 5) && (P[n].Mass > 0);
}

static void blackhole_feedback_reduce(int place, TreeWalkResultBHFeedback * remote, enum TreeWalkReduceMode mode) {
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
}

static void blackhole_feedback_copy(int place, TreeWalkQueryBHFeedback * I) {
    int k;
    for(k = 0; k < 3; k++)
    {
        I->Vel[k] = P[place].Vel[k];
    }

    I->Hsml = P[place].Hsml;
    I->Mass = P[place].Mass;
    I->BH_Mass = BHP(place).Mass;
    I->Density = BHP(place).Density;
    I->FeedbackWeightSum = BHP(place).FeedbackWeightSum;
    I->Mdot = BHP(place).Mdot;
    I->Csnd = blackhole_soundspeed(
                BHP(place).Entropy,
                BHP(place).Pressure,
                BHP(place).Density);
    I->Dt =
        (P[place].TimeBin ? (1 << P[place].TimeBin) : 0) * All.Timebase_interval / All.cf.hubble;
    I->ID = P[place].ID;
}
static int blackhole_swallow_isactive(int n) {
    return (P[n].Type == 5) && (P[n].SwallowID == 0);
}
static void blackhole_swallow_copy(int place, TreeWalkQuerySwallow * I) {
    I->Hsml = P[place].Hsml;
    I->BH_Mass = BHP(place).Mass;
    I->ID = P[place].ID;
}

static void blackhole_swallow_reduce(int place, TreeWalkResultSwallow * remote, enum TreeWalkReduceMode mode) {
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

    int child = domain_fork_particle(index);

    P[child].PI = atomic_fetch_and_add(&N_bh, 1);
    P[child].Type = 5;	/* make it a black hole particle */
#ifdef WINDS
    P[child].StellarAge = All.Time;
#endif
    P[child].Mass = All.SeedBlackHoleMass;
    P[index].Mass -= All.SeedBlackHoleMass;
    BHP(child).ID = P[child].ID;
    BHP(child).Mass = All.SeedBlackHoleMass;
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


#endif
