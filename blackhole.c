#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "densitykernel.h"
#include "proto.h"
#include "evaluator.h"
#include "domain.h"

/*! \file blackhole.c
 *  \brief routines for gas accretion onto black holes, and black hole mergers
 */

#ifdef BLACK_HOLES

struct blackhole_event {
    enum {
        BHEVENT_ACCRETION = 0,
        BHEVENT_SCATTER = 1,
        BHEVENT_SWALLOW = 2, 
    } type;
    double time;
    MyIDType ID;
    union {
        struct blackhole_accretion_event {
            /* for ACCRETION*/
            double pos[3];
            float bhmass;
            float bhmdot;
            float rho_proper;
            float soundspeed;
            float bhvel;
            float gasvel[3];
            float hsml;
        } a;
        struct blackhole_swallow_event { /* for SCATTER and SWALLOW */
            MyIDType ID_swallow;
            float bhmass_before;
            float bhmass_swallow;
            float vrel;   /* unused for SWALLOW */
            float soundspeed; /* unused for SWALLOW */
        } s;
    };
};

struct feedbackdata_in
{
    int NodeList[NODELISTLENGTH];
    MyDouble Pos[3];
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
};

struct feedbackdata_out
{
#ifdef BH_REPOSITION_ON_POTMIN
    MyFloat BH_MinPotPos[3];
    MyFloat BH_MinPotVel[3];
    MyFloat BH_MinPot;
#endif
    short int BH_TimeBinLimit;
};

struct swallowdata_in
{
    int NodeList[NODELISTLENGTH];
    MyDouble Pos[3];
    MyFloat Hsml;
    MyFloat BH_Mass;
    MyIDType ID;
};

struct swallowdata_out
{
    MyDouble Mass;
    MyDouble BH_Mass;
    MyDouble AccretedMomentum[3];
#ifdef BH_COUNTPROGS
    int BH_CountProgs;
#endif
};

static void * blackhole_alloc_ngblist();
static void blackhole_accretion_evaluate(int n);
static void blackhole_postprocess(int n);

static int blackhole_feedback_isactive(int n);
static void blackhole_feedback_reduce(int place, struct feedbackdata_out * remote, int mode);
static void blackhole_feedback_copy(int place, struct feedbackdata_in * I);

static int blackhole_feedback_evaluate(int target, int mode, 
        struct feedbackdata_in * I, 
        struct feedbackdata_out * O, 
        LocalEvaluator * lv, int * ngblist);

static int blackhole_swallow_isactive(int n);
static void blackhole_swallow_reduce(int place, struct swallowdata_out * remote, int mode);
static void blackhole_swallow_copy(int place, struct swallowdata_in * I);

static int blackhole_swallow_evaluate(int target, int mode, 
        struct swallowdata_in * I, 
        struct swallowdata_out * O, 
        LocalEvaluator * lv, int * ngblist);

#define BHPOTVALUEINIT 1.0e30

static int N_sph_swallowed, N_BH_swallowed;

static double blackhole_soundspeed(double entropy_or_pressure, double rho) {
    /* rho is comoving !*/
    double cs;
#ifdef BH_CSND_FROM_PRESSURE
    cs = sqrt(GAMMA * entropy_or_pressure / rho);

#else
    cs = sqrt(GAMMA * entropy_or_pressure * 
            pow(rho, GAMMA_MINUS1));
#endif
    cs *= pow(All.Time, -1.5 * GAMMA_MINUS1);

    return cs;
}

void blackhole_accretion(void)
{
    int i, j, k, n, bin;
    int ndone_flag, ndone;
    int ngrp, sendTask, recvTask, place, nexport, nimport, dummy;
    int Ntot_gas_swallowed, Ntot_BH_swallowed;

    walltime_measure("/Misc");
    Evaluator fbev = {0};

    fbev.ev_label = "BH_FEEDBACK";
    fbev.ev_evaluate = (ev_ev_func) blackhole_feedback_evaluate;
    fbev.ev_isactive = blackhole_feedback_isactive;
    fbev.ev_alloc = blackhole_alloc_ngblist;
    fbev.ev_copy = (ev_copy_func) blackhole_feedback_copy;
    fbev.ev_reduce = (ev_reduce_func) blackhole_feedback_reduce;
    fbev.UseNodeList = 1;
    fbev.ev_datain_elsize = sizeof(struct feedbackdata_in);
    fbev.ev_dataout_elsize = sizeof(struct feedbackdata_out);

    Evaluator swev = {0};
    swev.ev_label = "BH_SWALLOW";
    swev.ev_evaluate = (ev_ev_func) blackhole_swallow_evaluate;
    swev.ev_isactive = blackhole_swallow_isactive;
    swev.ev_alloc = blackhole_alloc_ngblist;
    swev.ev_copy = (ev_copy_func) blackhole_swallow_copy;
    swev.ev_reduce = (ev_reduce_func) blackhole_swallow_reduce;
    swev.UseNodeList = 1;
    swev.ev_datain_elsize = sizeof(struct swallowdata_in);
    swev.ev_dataout_elsize = sizeof(struct swallowdata_out);

    if(ThisTask == 0)
    {
        printf("Beginning black-hole accretion\n");
        fflush(stdout);
    }


    /* Let's first compute the Mdot values */
    int Nactive;
    int * queue = ev_get_queue(&fbev, &Nactive);

#ifdef BH_ACCRETION
    for(i = 0; i < Nactive; i ++) {
        int n = queue[i];
        blackhole_accretion_evaluate(n);
    }
#endif

    /* Now let's invoke the functions that stochasticall swallow gas
     * and deal with black hole mergers.
     */

    if(ThisTask == 0)
    {
        printf("Start swallowing of gas particles and black holes\n");
        fflush(stdout);
    }


    N_sph_swallowed = N_BH_swallowed = 0;


    /* allocate buffers to arrange communication */

    Ngblist = (int *) mymalloc("Ngblist", All.NumThreads * NumPart * sizeof(int));

    /* Let's first spread the feedback energy, 
     * and determine which particles may be swalled by whom */

    ev_run(&fbev);

    /* Now do the swallowing of particles */
#if defined(BH_SWALLOWGAS) || defined(BH_MERGER)
    ev_run(&swev);
#endif
    myfree(Ngblist);

    MPI_Reduce(&N_sph_swallowed, &Ntot_gas_swallowed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&N_BH_swallowed, &Ntot_BH_swallowed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(ThisTask == 0)
    {
        printf("Accretion done: %d gas particles swallowed, %d BH particles swallowed\n",
                Ntot_gas_swallowed, Ntot_BH_swallowed);
        fflush(stdout);
    }



    for(n = 0; n < TIMEBINS; n++)
    {
        if(TimeBinActive[n])
        {
            TimeBin_BH_mass[n] = 0;
            TimeBin_BH_dynamicalmass[n] = 0;
            TimeBin_BH_Mdot[n] = 0;
            TimeBin_BH_Medd[n] = 0;
        }
    }

    for(i = 0; i < Nactive; i++) {
        int n = queue[i];
        blackhole_postprocess(n);
    }

    myfree(queue);

    double total_mass_real, total_mdoteddington;
    double total_mass_holes, total_mdot;
    double mdot = 0;
    double mass_holes = 0;
    double mass_real = 0;
    double medd = 0;
    for(bin = 0; bin < TIMEBINS; bin++)
        if(TimeBinCount[bin])
        {
            mass_holes += TimeBin_BH_mass[bin];
            mass_real += TimeBin_BH_dynamicalmass[bin];
            mdot += TimeBin_BH_Mdot[bin];
            medd += TimeBin_BH_Medd[bin];
        }

    MPI_Reduce(&mass_holes, &total_mass_holes, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mass_real, &total_mass_real, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mdot, &total_mdot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&medd, &total_mdoteddington, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

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


    fflush(FdBlackHolesDetails);

    walltime_measure("/BH");
}

static void blackhole_accretion_evaluate(int n) {
    double mdot = 0;		/* if no accretion model is enabled, we have mdot=0 */

    double rho = BHP(n).Density;
#ifdef BH_USE_GASVEL_IN_BONDI
    double bhvel = sqrt(pow(P[n].Vel[0] - BHP(n).SurroundingGasVel[0], 2) +
            pow(P[n].Vel[1] - BHP(n).SurroundingGasVel[1], 2) +
            pow(P[n].Vel[2] - BHP(n).SurroundingGasVel[2], 2));
#else
    double bhvel = 0;
#endif

    bhvel /= All.cf.a;
    double rho_proper = rho * All.cf.a3inv;

    double soundspeed = blackhole_soundspeed(BHP(n).EntOrPressure, rho);

    /* Note: we take here a radiative efficiency of 0.1 for Eddington accretion */
    double meddington = (4 * M_PI * GRAVITY * C * PROTONMASS / (0.1 * C * C * THOMPSON)) * BHP(n).Mass
        * All.UnitTime_in_s;

    double norm = pow((pow(soundspeed, 2) + pow(bhvel, 2)), 1.5);

    if(norm > 0)
        mdot = 4. * M_PI * All.BlackHoleAccretionFactor * All.G * All.G *
            BHP(n).Mass * BHP(n).Mass * rho_proper / norm;
    else
        mdot = 0;

#ifdef BH_ENFORCE_EDDINGTON_LIMIT
    if(mdot > All.BlackHoleEddingtonFactor * meddington)
        mdot = All.BlackHoleEddingtonFactor * meddington;
#endif
    BHP(n).Mdot = mdot;

    if(BHP(n).Mass > 0)
    {
        struct blackhole_event event;
        event.type = BHEVENT_ACCRETION;
        event.time = All.Time;
        event.ID = P[n].ID;
        event.a.bhmass = BHP(n).Mass;
        event.a.bhmdot = mdot;
        event.a.rho_proper = rho_proper;
        event.a.soundspeed = soundspeed;
        event.a.bhvel = bhvel;
        event.a.pos[0] = P[n].Pos[0];
        event.a.pos[1] = P[n].Pos[1];
        event.a.pos[2] = P[n].Pos[2];
        event.a.hsml = P[n].Hsml;
        fwrite(&event, sizeof(event), 1, FdBlackHolesDetails);
    }
    double dt = (P[n].TimeBin ? (1 << P[n].TimeBin) : 0) * All.Timebase_interval / All.cf.hubble;

    BHP(n).Mass += BHP(n).Mdot * dt;

}

static void blackhole_postprocess(int n) {
    int k;
#ifdef BH_ACCRETION
    if(BHP(n).accreted_Mass > 0)
    {
#ifndef BH_REPOSITION_ON_POTMIN
        for(k = 0; k < 3; k++)
            P[n].Vel[k] =
                (P[n].Vel[k] * P[n].Mass + BHP(n).accreted_momentum[k]) /
                (P[n].Mass + BHP(n).accreted_Mass);
#endif
        P[n].Mass += BHP(n).accreted_Mass;
        BHP(n).Mass += BHP(n).accreted_BHMass;
        BHP(n).accreted_Mass = 0;
    }
    int bin = P[n].TimeBin;
#pragma omp atomic
    TimeBin_BH_mass[bin] += BHP(n).Mass;
#pragma omp atomic
    TimeBin_BH_dynamicalmass[bin] += P[n].Mass;
#pragma omp atomic
    TimeBin_BH_Mdot[bin] += BHP(n).Mdot;
    if(BHP(n).Mass > 0) {
#pragma omp atomic
        TimeBin_BH_Medd[bin] += BHP(n).Mdot / BHP(n).Mass;
    }
#endif
}

static int blackhole_feedback_evaluate(int target, int mode, 
        struct feedbackdata_in * I, 
        struct feedbackdata_out * O, 
        LocalEvaluator * lv, int * ngblist)
{

    int startnode, numngb, k, n, listindex = 0;
    double hsearch;

    int ptypemask = 0;
#ifndef BH_REPOSITION_ON_POTMIN
    ptypemask = 1 + (1 << 5);
#else
    ptypemask = 1 + 2 + 4 + 8 + 16 + 32;
#endif

    O->BH_TimeBinLimit = -1;
#ifdef BH_REPOSITION_ON_POTMIN
    O->BH_MinPot = BHPOTVALUEINIT;
#endif

    startnode = I->NodeList[0];
    listindex ++;
    startnode = Nodes[startnode].u.d.nextnode;	/* open it */

    density_kernel_t kernel;
    density_kernel_t bh_feedback_kernel;
    hsearch = density_decide_hsearch(5, I->Hsml);

    density_kernel_init(&kernel, I->Hsml);
    density_kernel_init(&bh_feedback_kernel, hsearch);

    /* initialize variables before SPH loop is started */

    /* Now start the actual SPH computation for this particle */

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb = ngb_treefind_threads(I->Pos, hsearch, target, &startnode, mode, lv,
                    ngblist, NGB_TREEFIND_ASYMMETRIC, ptypemask);

            if(numngb < 0)
                return numngb;

            for(n = 0; 
                n < numngb; 
                (unlock_particle_if_not(ngblist[n], I->ID), n++)
                )
            {
                lock_particle_if_not(ngblist[n], I->ID);
                int j = ngblist[n];

                if(P[j].Mass < 0) continue;

                if(P[j].Type != 5) {
                    if (O->BH_TimeBinLimit <= 0 || O->BH_TimeBinLimit >= P[j].TimeBin) 
                        O->BH_TimeBinLimit = P[j].TimeBin;
                }
                double dx = I->Pos[0] - P[j].Pos[0];
                double dy = I->Pos[1] - P[j].Pos[1];
                double dz = I->Pos[2] - P[j].Pos[2];

                dx = NEAREST(dx);
                dy = NEAREST(dy);
                dz = NEAREST(dz);

                double r2 = dx * dx + dy * dy + dz * dz;

#ifdef BH_REPOSITION_ON_POTMIN
                /* if this option is switched on, we may also encounter dark matter particles or stars */
                if(r2 < kernel.HH)
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
#endif
#ifdef BH_MERGER
                if(P[j].Type == 5 && r2 < kernel.HH)	/* we have a black hole merger */
                {
                    if(I->ID != P[j].ID)
                    {
                        /* compute relative velocity of BHs */

                        double vrel = 0;
                        for(k = 0, vrel = 0; k < 3; k++)
                            vrel += (P[j].Vel[k] - I->Vel[k]) * (P[j].Vel[k] - I->Vel[k]);

                        vrel = sqrt(vrel) / All.cf.a;

                        if(vrel > 0.5 * I->Csnd)
                        {
                            struct blackhole_event event;
                            event.type = BHEVENT_SCATTER;
                            event.time = All.Time;
                            event.ID = I->ID;
                            event.s.ID_swallow = P[j].ID;
                            event.s.bhmass_before = I->BH_Mass;
                            event.s.bhmass_swallow = BHP(j).Mass;
                            event.s.vrel = vrel;
                            event.s.soundspeed = I->Csnd;
                            fwrite(&event, sizeof(event), 1, FdBlackHolesDetails);
                        }
                        else
                        {
                            if(P[j].SwallowID < I->ID && P[j].ID < I->ID)
                                P[j].SwallowID = I->ID;
                        }
                    }
                }
#endif
                if(P[j].Type == 0) {
#ifdef WINDS
                    /* BH does not accrete wind */
                    if(SPHP(j).DelayTime > 0) continue;
#endif
#ifdef BH_SWALLOWGAS
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
#endif

#ifdef BH_THERMALFEEDBACK
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
#ifndef UNIFIED_FEEDBACK
                        double energy = All.BlackHoleFeedbackFactor * 0.1 * I->Mdot * I->Dt *
                            pow(C / All.UnitVelocity_in_cm_per_s, 2);

                        if(I->FeedbackWeightSum > 0)
                        {
                            SPHP(j).Injected_BH_Energy += (energy * mass_j * wk / I->FeedbackWeightSum);
                        }

#else
                        double meddington = (4 * M_PI * GRAVITY * C *
                                PROTONMASS / (0.1 * C * C * THOMPSON)) * I->BH_Mass *
                            All.UnitTime_in_s;

                        if(I->Mdot > All.RadioThreshold * meddington)
                        {
                            double energy =
                                All.BlackHoleFeedbackFactor * 0.1 * I->Mdot * I->Dt * pow(C /
                                        All.UnitVelocity_in_cm_per_s,
                                        2);
                            if(I->FeedbackWeightSum> 0) {
                                SPHP(j).Injected_BH_Energy += (energy * mass_j * wk / I->FeedbackWeightSum);
                            }
                        }
#endif
                    }
#endif
                }
            }
        }

        if(listindex < NODELISTLENGTH)
        {
            startnode = I->NodeList[listindex];
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
 * ran only if BH_MERGER or BH_SWALLOWGAS
 * is defined
 */
int blackhole_swallow_evaluate(int target, int mode, 
        struct swallowdata_in * I, 
        struct swallowdata_out * O, 
        LocalEvaluator * lv, int * ngblist)
{
    int startnode, numngb, k, n, listindex = 0;

    int ptypemask = 0;
#ifndef BH_REPOSITION_ON_POTMIN
    ptypemask = 1 + (1 << 5);
#else
    ptypemask = 1 + 2 + 4 + 8 + 16 + 32;
#endif

    startnode = I->NodeList[0];
    listindex ++;
    startnode = Nodes[startnode].u.d.nextnode;	/* open it */

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb = ngb_treefind_threads(I->Pos, I->Hsml, target, &startnode, 
                    mode, lv, ngblist, NGB_TREEFIND_SYMMETRIC, ptypemask);

            if(numngb < 0)
                return numngb;

            for(n = 0; n < numngb; 
                 (unlock_particle_if_not(ngblist[n], I->ID), n++)
                 )
            {
                lock_particle_if_not(ngblist[n], I->ID);
                int j = ngblist[n];
                if(P[j].SwallowID != I->ID) continue;
#ifdef BH_MERGER
                if(P[j].Type == 5)	/* we have a black hole merger */
                {
                    struct blackhole_event event;
                    event.type = BHEVENT_SWALLOW;
                    event.time = All.Time;
                    event.ID = I->ID;
                    event.s.ID_swallow = P[j].ID;
                    event.s.bhmass_before = I->BH_Mass;
                    event.s.bhmass_swallow = BHP(j).Mass;
                    fwrite(&event, sizeof(event), 1, FdBlackHolesDetails);

                    O->Mass += (P[j].Mass);
                    O->BH_Mass += (BHP(j).Mass);
                    for(k = 0; k < 3; k++)
                        O->AccretedMomentum[k] += (P[j].Mass * P[j].Vel[k]);

#ifdef BH_COUNTPROGS
                    O->BH_CountProgs += BHP(j).CountProgs;
#endif

                    int bin = P[j].TimeBin;
#pragma omp atomic
                    TimeBin_BH_mass[bin] -= BHP(j).Mass;
#pragma omp atomic
                    TimeBin_BH_dynamicalmass[bin] -= P[j].Mass;
#pragma omp atomic
                    TimeBin_BH_Mdot[bin] -= BHP(j).Mdot;
                    if(BHP(j).Mass > 0) {
#pragma omp atomic
                        TimeBin_BH_Medd[bin] -= BHP(j).Mdot / BHP(j).Mass;
                    }

                    P[j].Mass = 0;
                    BHP(j).Mass = 0;
                    BHP(j).Mdot = 0;

#pragma omp atomic
                    N_BH_swallowed++;
                }
#endif

#ifdef BH_SWALLOWGAS
                if(P[j].Type == 0)
                {
                    O->Mass += (P[j].Mass);

                    for(k = 0; k < 3; k++)
                        O->AccretedMomentum[k] += (P[j].Mass * P[j].Vel[k]);

                    P[j].Mass = 0;
#pragma omp atomic
                    N_sph_swallowed++;
                }
#endif 
            }
        }
        if(listindex < NODELISTLENGTH)
        {
            startnode = I->NodeList[listindex];
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
static void * blackhole_alloc_ngblist() {
    int threadid = omp_get_thread_num();
    return Ngblist + threadid * NumPart;
}
static void blackhole_feedback_reduce(int place, struct feedbackdata_out * remote, int mode) {
    int k;
#ifdef BH_REPOSITION_ON_POTMIN
    if(mode == 0 || BHP(place).MinPot > remote->BH_MinPot)
    {
        BHP(place).MinPot = remote->BH_MinPot;
        for(k = 0; k < 3; k++) {
            BHP(place).MinPotPos[k] = remote->BH_MinPotPos[k];
            BHP(place).MinPotVel[k] = remote->BH_MinPotVel[k];
        }
    }
#endif
    if (mode == 0 || 
            BHP(place).TimeBinLimit < 0 || 
            BHP(place).TimeBinLimit > remote->BH_TimeBinLimit) {
        BHP(place).TimeBinLimit = remote->BH_TimeBinLimit;
    }
}

static void blackhole_feedback_copy(int place, struct feedbackdata_in * I) {
    int k;
    for(k = 0; k < 3; k++)
    {
        I->Pos[k] = P[place].Pos[k];
        I->Vel[k] = P[place].Vel[k];
    }

    I->Hsml = P[place].Hsml;
    I->Mass = P[place].Mass;
    I->BH_Mass = BHP(place).Mass;
    I->Density = BHP(place).Density;
    I->FeedbackWeightSum = BHP(place).FeedbackWeightSum;
    I->Mdot = BHP(place).Mdot;
    I->Csnd =
        blackhole_soundspeed(
                BHP(place).EntOrPressure,
                BHP(place).Density);
    I->Dt =
        (P[place].TimeBin ? (1 << P[place].TimeBin) : 0) * All.Timebase_interval / All.cf.hubble;
    I->ID = P[place].ID;
}
static int blackhole_swallow_isactive(int n) {
    return (P[n].Type == 5) && (P[n].SwallowID == 0);
}
static void blackhole_swallow_copy(int place, struct swallowdata_in * I) {
    int k;
    for(k = 0; k < 3; k++)
    {
        I->Pos[k] = P[place].Pos[k];
    }
    I->Hsml = P[place].Hsml;
    I->BH_Mass = BHP(place).Mass;
    I->ID = P[place].ID;
}

static void blackhole_swallow_reduce(int place, struct swallowdata_out * remote, int mode) {
    int k;

#define EV_REDUCE(A, B) (A) = (mode==0)?(B):((A) + (B))
    EV_REDUCE(BHP(place).accreted_Mass, remote->Mass);
    EV_REDUCE(BHP(place).accreted_BHMass, remote->BH_Mass);
    for(k = 0; k < 3; k++) {
        EV_REDUCE(BHP(place).accreted_momentum[k], remote->AccretedMomentum[k]);
    }
#ifdef BH_COUNTPROGS
    EV_REDUCE(BHP(place).CountProgs, remote->BH_CountProgs);
#endif
}

void blackhole_make_one(int index) {
    if(P[index].Type != 0) endrun(7772);

    int child = domain_fork_particle(index);

    P[child].PI = atomic_fetch_and_add(&N_bh, 1);
    P[child].Type = 5;	/* make it a black hole particle */
#ifdef STELLARAGE
    P[child].StellarAge = All.Time;
#endif
    P[child].Mass = All.SeedBlackHoleMass;
    P[index].Mass -= All.SeedBlackHoleMass;
    BHP(child).ID = P[child].ID;
    BHP(child).Mass = All.SeedBlackHoleMass;
    BHP(child).Mdot = 0;
    BHP(child).MinPot = BHPOTVALUEINIT;

#ifdef BH_COUNTPROGS
    BHP(child).CountProgs = 1;
#endif
}
#endif
