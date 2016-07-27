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

static void blackhole_accretion_evaluate(int n);
static void blackhole_postprocess(int n);

static int blackhole_feedback_isactive(int n);
static void blackhole_feedback_reduce(int place, TreeWalkResultBHFeedback * remote, enum TreeWalkReduceMode mode);
static void blackhole_feedback_copy(int place, TreeWalkQueryBHFeedback * I);

static int blackhole_feedback_evaluate(int target,
        TreeWalkQueryBHFeedback * I,
        TreeWalkResultBHFeedback * O,
        LocalTreeWalk * lv);

static int blackhole_swallow_isactive(int n);
static void blackhole_swallow_reduce(int place, TreeWalkResultSwallow * remote, enum TreeWalkReduceMode mode);
static void blackhole_swallow_copy(int place, TreeWalkQuerySwallow * I);

static int blackhole_swallow_evaluate(int target,
        TreeWalkQuerySwallow * I,
        TreeWalkResultSwallow * O,
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
    TreeWalk fbev = {0};

    fbev.ev_label = "BH_FEEDBACK";
    fbev.ev_evaluate = (ev_ev_func) blackhole_feedback_evaluate;
    fbev.ev_isactive = blackhole_feedback_isactive;
    fbev.ev_copy = (ev_copy_func) blackhole_feedback_copy;
    fbev.ev_reduce = (ev_reduce_func) blackhole_feedback_reduce;
    fbev.UseNodeList = 1;
    fbev.query_type_elsize = sizeof(TreeWalkQueryBHFeedback);
    fbev.result_type_elsize = sizeof(TreeWalkResultBHFeedback);

    TreeWalk swev = {0};
    swev.ev_label = "BH_SWALLOW";
    swev.ev_evaluate = (ev_ev_func) blackhole_swallow_evaluate;
    swev.ev_isactive = blackhole_swallow_isactive;
    swev.ev_copy = (ev_copy_func) blackhole_swallow_copy;
    swev.ev_reduce = (ev_reduce_func) blackhole_swallow_reduce;
    swev.UseNodeList = 1;
    swev.query_type_elsize = sizeof(TreeWalkQuerySwallow);
    swev.result_type_elsize = sizeof(TreeWalkResultSwallow);

    message(0, "Beginning black-hole accretion\n");


    /* Let's first compute the Mdot values */
    int Nactive;
    int * queue = treewalk_get_queue(&fbev, &Nactive);

    for(i = 0; i < Nactive; i ++) {
        int n = queue[i];

        Local_BH_mass -= BHP(n).Mass;
        Local_BH_dynamicalmass -= P[n].Mass;
        Local_BH_Mdot -= BHP(n).Mdot;
        if(BHP(n).Mass > 0) {
            Local_BH_Medd -= BHP(n).Mdot / BHP(n).Mass;
        }

        blackhole_accretion_evaluate(n);

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

    treewalk_run(&fbev);

    /* Now do the swallowing of particles */
    treewalk_run(&swev);

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

static void blackhole_accretion_evaluate(int n) {
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

static int blackhole_feedback_evaluate(int target,
        TreeWalkQueryBHFeedback * I,
        TreeWalkResultBHFeedback * O,
        LocalTreeWalk * lv)
{

    int startnode, numngb, k, n, listindex = 0;
    double hsearch;

    int ptypemask = 0;
    ptypemask = 1 + 2 + 4 + 8 + 16 + 32;

    O->BH_TimeBinLimit = -1;
    O->BH_MinPot = BHPOTVALUEINIT;

    startnode = I->base.NodeList[0];
    listindex ++;
    startnode = Nodes[startnode].u.d.nextnode;	/* open it */

    DensityKernel kernel;
    DensityKernel bh_feedback_kernel;
    hsearch = density_decide_hsearch(5, I->Hsml);

    density_kernel_init(&kernel, I->Hsml);
    density_kernel_init(&bh_feedback_kernel, hsearch);

    /* initialize variables before SPH loop is started */

    /* Now start the actual SPH computation for this particle */

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb = ngb_treefind_threads(I->base.Pos, hsearch, target, &startnode, lv,
                    NGB_TREEFIND_ASYMMETRIC, ptypemask);

            if(numngb < 0)
                return numngb;

            for(n = 0;
                n < numngb;
                (unlock_particle_if_not(lv->ngblist[n], I->ID), n++)
                )
            {
                lock_particle_if_not(lv->ngblist[n], I->ID);
                int j = lv->ngblist[n];

                if(P[j].Mass < 0) continue;

                if(P[j].Type != 5) {
                    if (O->BH_TimeBinLimit <= 0 || O->BH_TimeBinLimit >= P[j].TimeBin)
                        O->BH_TimeBinLimit = P[j].TimeBin;
                }
                double dx = I->base.Pos[0] - P[j].Pos[0];
                double dy = I->base.Pos[1] - P[j].Pos[1];
                double dz = I->base.Pos[2] - P[j].Pos[2];

                dx = NEAREST(dx);
                dy = NEAREST(dy);
                dz = NEAREST(dz);

                double r2 = dx * dx + dy * dy + dz * dz;

                /* if this option is switched on, we may also encounter dark matter particles or stars */
                if(r2 < kernel.HH && r2 < All.FOFHaloComovingLinkingLength)
                {
                    if(P[j].Potential < O->BH_MinPot)
                    {
                        if(P[j].Type == 0 || P[j].Type == 1 || P[j].Type == 4 || P[j].Type == 5)
                        {
                            /* compute relative velocities */

                            double vrel = 0;
                            for(k = 0, vrel = 0; k < 3; k++)
                                vrel += (P[j].Vel[k] - I->Vel[k]) * (P[j].Vel[k] - I->Vel[k]);

                            vrel = sqrt(vrel) / All.cf.a;

                            if(vrel <= 0.25 * I->Csnd)
                            {
                                O->BH_MinPot = P[j].Potential;
                                for(k = 0; k < 3; k++) {
                                    O->BH_MinPotPos[k] = P[j].Pos[k];
                                    O->BH_MinPotVel[k] = P[j].Vel[k];
                                }
                            }
                        }
                    }
                }
                if(P[j].Type == 5 && r2 < kernel.HH)	/* we have a black hole merger */
                {
                    if(I->ID != P[j].ID)
                    {
                        /* compute relative velocity of BHs */

                        double vrel = 0;
                        for(k = 0, vrel = 0; k < 3; k++)
                            vrel += (P[j].Vel[k] - I->Vel[k]) * (P[j].Vel[k] - I->Vel[k]);

                        vrel = sqrt(vrel) / All.cf.a;

                        if(vrel <= 0.5 * I->Csnd)
                        {
                            if(P[j].SwallowID < I->ID && P[j].ID < I->ID)
                                P[j].SwallowID = I->ID;
                        }
                    }
                }
                if(P[j].Type == 0) {
#ifdef WINDS
                    /* BH does not accrete wind */
                    if(SPHP(j).DelayTime > 0) continue;
#endif
                    if(r2 < kernel.HH) {
                        /* here we have a gas particle */

                        double r = sqrt(r2);
                        double u = r * kernel.Hinv;
                        double wk = density_kernel_wk(&kernel, u);
                        /* compute accretion probability */
                        double p, w;

                        if((I->BH_Mass - I->Mass) > 0 && I->Density > 0)
                            p = (I->BH_Mass - I->Mass) * wk / I->Density;
                        else
                            p = 0;

                        /* compute random number, uniform in [0,1] */
                        w = get_random_number(P[j].ID);
                        if(w < p)
                        {
                            if(P[j].SwallowID < I->ID)
                                P[j].SwallowID = I->ID;
                        }
                    }


                    if(r2 < bh_feedback_kernel.HH && P[j].Mass > 0) {
                        double r = sqrt(r2);
                        double u = r * bh_feedback_kernel.Hinv;
                        double wk;
                        double mass_j;
                        if(HAS(All.BlackHoleFeedbackMethod, BH_FEEDBACK_MASS)) {
                            mass_j = P[j].Mass;
                        } else {
                            mass_j = P[j].Hsml * P[j].Hsml * P[j].Hsml;
                        }
                        if(HAS(All.BlackHoleFeedbackMethod, BH_FEEDBACK_SPLINE))
                            wk = density_kernel_wk(&bh_feedback_kernel, u);
                        else
                        wk = 1.0;
                        double energy = All.BlackHoleFeedbackFactor * 0.1 * I->Mdot * I->Dt *
                            pow(C / All.UnitVelocity_in_cm_per_s, 2);

                        if(I->FeedbackWeightSum > 0)
                        {
                            SPHP(j).Injected_BH_Energy += (energy * mass_j * wk / I->FeedbackWeightSum);
                        }

                    }

                }
            }
        }

        if(listindex < NODELISTLENGTH)
        {
            startnode = I->base.NodeList[listindex];
            if(startnode >= 0) {
                startnode = Nodes[startnode].u.d.nextnode;	/* open it */
                listindex ++;
            }
        }
    }

    return 0;
}


/**
 * perform blackhole swallow / merger;
 */
int blackhole_swallow_evaluate(int target,
        TreeWalkQuerySwallow * I,
        TreeWalkResultSwallow * O,
        LocalTreeWalk * lv)
{
    int startnode, numngb, k, n, listindex = 0;

    int ptypemask = 0;
    ptypemask = 1 + 2 + 4 + 8 + 16 + 32;

    startnode = I->base.NodeList[0];
    listindex ++;
    startnode = Nodes[startnode].u.d.nextnode;	/* open it */

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb = ngb_treefind_threads(I->base.Pos, I->Hsml, target, &startnode,
                    lv, NGB_TREEFIND_SYMMETRIC, ptypemask);

            if(numngb < 0)
                return numngb;

            for(n = 0; n < numngb;
                 (unlock_particle_if_not(lv->ngblist[n], I->ID), n++)
                 )
            {
                lock_particle_if_not(lv->ngblist[n], I->ID);
                int j = lv->ngblist[n];
                if(P[j].SwallowID != I->ID) continue;

                if(P[j].Type == 5)	/* we have a black hole merger */
                {
                    O->Mass += (P[j].Mass);
                    O->BH_Mass += (BHP(j).Mass);

                    for(k = 0; k < 3; k++)
                        O->AccretedMomentum[k] += (P[j].Mass * P[j].Vel[k]);

                    O->BH_CountProgs += BHP(j).CountProgs;

#pragma omp atomic
                    Local_BH_mass -= BHP(j).Mass;
#pragma omp atomic
                    Local_BH_dynamicalmass -= P[j].Mass;
#pragma omp atomic
                    Local_BH_Mdot -= BHP(j).Mdot;
                    if(BHP(j).Mass > 0) {
#pragma omp atomic
                        Local_BH_Medd -= BHP(j).Mdot / BHP(j).Mass;
                    }

                    P[j].Mass = 0;
                    BHP(j).Mass = 0;
                    BHP(j).Mdot = 0;

#pragma omp atomic
                    N_BH_swallowed++;
                }

                if(P[j].Type == 0)
                {
                    O->Mass += (P[j].Mass);

                    for(k = 0; k < 3; k++)
                        O->AccretedMomentum[k] += (P[j].Mass * P[j].Vel[k]);

                    P[j].Mass = 0;
#pragma omp atomic
                    N_sph_swallowed++;
                }
            }
        }
        if(listindex < NODELISTLENGTH)
        {
            startnode = I->base.NodeList[listindex];
            if(startnode >= 0) {
                startnode = Nodes[startnode].u.d.nextnode;	/* open it */
                listindex++;
            }
        }
    }

    return 0;
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
