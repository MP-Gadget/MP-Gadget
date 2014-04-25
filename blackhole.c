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
#ifdef REPOSITION_ON_POTMIN
    MyFloat BH_MinPotPos[3];
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
    MyLongDouble Mass;
    MyLongDouble BH_Mass;
    MyLongDouble AccretedMomentum[3];
#ifdef BH_BUBBLES
    MyLongDouble BH_Mass_bubbles;
#ifdef UNIFIED_FEEDBACK
    MyLongDouble BH_Mass_radio;
#endif
#endif
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
    if(All.ComovingIntegrationOn) {
        cs *= pow(All.Time, -1.5 * GAMMA_MINUS1);
    }
    return cs;
}

void blackhole_accretion(void)
{
    int i, j, k, n, bin;
    int ndone_flag, ndone;
    int ngrp, sendTask, recvTask, place, nexport, nimport, dummy;
    int Ntot_gas_swallowed, Ntot_BH_swallowed;

    Evaluator fbev = {0};

    fbev.ev_evaluate = (ev_evaluate_func) blackhole_feedback_evaluate;
    fbev.ev_isactive = blackhole_feedback_isactive;
    fbev.ev_alloc = blackhole_alloc_ngblist;
    fbev.ev_copy = (ev_copy_func) blackhole_feedback_copy;
    fbev.ev_reduce = (ev_reduce_func) blackhole_feedback_reduce;
    fbev.UseNodeList = 1;
    fbev.ev_datain_elsize = sizeof(struct feedbackdata_in);
    fbev.ev_dataout_elsize = sizeof(struct feedbackdata_out);

    Evaluator swev = {0};
    swev.ev_evaluate = (ev_evaluate_func) blackhole_swallow_evaluate;
    swev.ev_isactive = blackhole_swallow_isactive;
    swev.ev_alloc = blackhole_alloc_ngblist;
    swev.ev_copy = (ev_copy_func) blackhole_swallow_copy;
    swev.ev_reduce = (ev_reduce_func) blackhole_swallow_reduce;
    swev.UseNodeList = 1;
    swev.ev_datain_elsize = sizeof(struct swallowdata_in);
    swev.ev_dataout_elsize = sizeof(struct swallowdata_out);


#ifdef BH_BUBBLES
    MyFloat bh_center[3];
    double *bh_dmass, *tot_bh_dmass;
    float *bh_posx, *bh_posy, *bh_posz;
    float *tot_bh_posx, *tot_bh_posy, *tot_bh_posz;
    int l, num_activebh = 0, total_num_activebh = 0;
    int *common_num_activebh, *disp;
    MyIDType *bh_id, *tot_bh_id;
#endif
    MPI_Status status;

    if(ThisTask == 0)
    {
        printf("Beginning black-hole accretion\n");
        fflush(stdout);
    }

    CPU_Step[CPU_MISC] += measure_time();

    /* Let's first compute the Mdot values */
    int Nactive;
    int * queue = evaluate_get_queue(&fbev, &Nactive);

    for(i = 0; i < Nactive; i ++) {
        int n = queue[i];
        blackhole_accretion_evaluate(n);
    }


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

    /** Let's first spread the feedback energy, and determine which particles may be swalled by whom */

    evaluate_begin(&fbev);

    do
    {

        /* do local particles and prepare export list */
        evaluate_primary(&fbev);

        evaluate_get_remote(&fbev, TAG_BH_A);

        /* now do the particles that were sent to us */

        evaluate_secondary(&fbev);

        /* get the result */
        evaluate_reduce_result(&fbev, TAG_BH_B);
    }
    while(evaluate_ndone(&fbev) < NTask);

    evaluate_finish(&fbev);


    /* Now do the swallowing of particles */
    evaluate_begin(&swev);

    do
    {
        /* do local particles and prepare export list */
        evaluate_primary(&swev);

        evaluate_get_remote(&swev, TAG_BH_A);

        /* now do the particles that were sent to us */

        evaluate_secondary(&swev);

        /* get the result */
        evaluate_reduce_result(&swev, TAG_BH_B);

    }
    while(evaluate_ndone(&swev) < NTask);

    evaluate_finish(&swev);

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

#ifdef BH_BUBBLES
    Ngblist = (int *) mymalloc("Ngblist", NumPart * sizeof(int));

    MPI_Allreduce(&num_activebh, &total_num_activebh, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if(ThisTask == 0)
    {
        printf("The total number of active BHs is: %d\n", total_num_activebh);
        fflush(stdout);
    }

    if(total_num_activebh > 0)
    {
        bh_dmass = mymalloc("bh_dmass", num_activebh * sizeof(double));
        tot_bh_dmass = mymalloc("tot_bh_dmass", total_num_activebh * sizeof(double));
        bh_posx = mymalloc("bh_posx", num_activebh * sizeof(float));
        bh_posy = mymalloc("bh_posy", num_activebh * sizeof(float));
        bh_posz = mymalloc("bh_posz", num_activebh * sizeof(float));
        tot_bh_posx = mymalloc("tot_bh_posx", total_num_activebh * sizeof(float));
        tot_bh_posy = mymalloc("tot_bh_posy", total_num_activebh * sizeof(float));
        tot_bh_posz = mymalloc("tot_bh_posz", total_num_activebh * sizeof(float));
        //      bh_id = mymalloc("bh_id", num_activebh * sizeof(unsigned int));
        //      tot_bh_id = mymalloc("tot_bh_id", total_num_activebh * sizeof(unsigned int));
        bh_id = mymalloc("bh_id", num_activebh * sizeof(MyIDType));
        tot_bh_id = mymalloc("tot_bh_id", total_num_activebh * sizeof(MyIDType));

        for(n = 0; n < num_activebh; n++)
        {
            bh_dmass[n] = 0.0;
            bh_posx[n] = 0.0;
            bh_posy[n] = 0.0;
            bh_posz[n] = 0.0;
            bh_id[n] = 0;
        }

        for(n = 0; n < total_num_activebh; n++)
        {
            tot_bh_dmass[n] = 0.0;
            tot_bh_posx[n] = 0.0;
            tot_bh_posy[n] = 0.0;
            tot_bh_posz[n] = 0.0;
            tot_bh_id[n] = 0;
        }

        for(n = FirstActiveParticle, l = 0; n >= 0; n = NextActiveParticle[n])
            if(P[n].Type == 5)
            {
                if(BHP(n).Mass_bubbles > 0
                        && BHP(n).Mass_bubbles > All.BlackHoleRadioTriggeringFactor * BHP(n).Mass_ini)
                {
#ifndef UNIFIED_FEEDBACK
                    bh_dmass[l] = BHP(n).Mass_bubbles - BHP(n).Mass_ini;
#else
                    bh_dmass[l] = BHP(n).Mass_radio - BHP(n).Mass_ini;
                    BHP(n).Mass_radio = BHP(n).Mass;
#endif
                    BHP(n).Mass_ini = BHP(n).Mass;
                    BHP(n).Mass_bubbles = BHP(n).Mass;

                    bh_posx[l] = P[n].Pos[0];
                    bh_posy[l] = P[n].Pos[1];
                    bh_posz[l] = P[n].Pos[2];
                    bh_id[l] = P[n].ID;

                    l++;
                }
            }
        common_num_activebh = mymalloc("common_num_activebh", NTask * sizeof(int));
        disp = mymalloc("disp", NTask * sizeof(int));

        MPI_Allgather(&num_activebh, 1, MPI_INT, common_num_activebh, 1, MPI_INT, MPI_COMM_WORLD);

        for(k = 1, disp[0] = 0; k < NTask; k++)
            disp[k] = disp[k - 1] + common_num_activebh[k - 1];


        MPI_Allgatherv(bh_dmass, num_activebh, MPI_DOUBLE, tot_bh_dmass, common_num_activebh, disp, MPI_DOUBLE,
                MPI_COMM_WORLD);
        MPI_Allgatherv(bh_posx, num_activebh, MPI_FLOAT, tot_bh_posx, common_num_activebh, disp, MPI_FLOAT,
                MPI_COMM_WORLD);
        MPI_Allgatherv(bh_posy, num_activebh, MPI_FLOAT, tot_bh_posy, common_num_activebh, disp, MPI_FLOAT,
                MPI_COMM_WORLD);
        MPI_Allgatherv(bh_posz, num_activebh, MPI_FLOAT, tot_bh_posz, common_num_activebh, disp, MPI_FLOAT,
                MPI_COMM_WORLD);

        MPI_Allgatherv(bh_id, num_activebh, MPI_UNSIGNED_LONG_LONG, tot_bh_id, common_num_activebh, disp, MPI_UNSIGNED_LONG_LONG,
                MPI_COMM_WORLD);

        for(l = 0; l < total_num_activebh; l++)
        {
            bh_center[0] = tot_bh_posx[l];
            bh_center[1] = tot_bh_posy[l];
            bh_center[2] = tot_bh_posz[l];

            if(tot_bh_dmass[l] > 0)
                bh_bubble(tot_bh_dmass[l], bh_center, tot_bh_id[l]);

        }

        myfree(disp);
        myfree(common_num_activebh);
        myfree(tot_bh_id);
        myfree(bh_id);
        myfree(tot_bh_posz);
        myfree(tot_bh_posy);
        myfree(tot_bh_posx);
        myfree(bh_posz);
        myfree(bh_posy);
        myfree(bh_posx);
        myfree(tot_bh_dmass);
        myfree(bh_dmass);
    }
    myfree(Ngblist);
#endif

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

    CPU_Step[CPU_BLACKHOLES] += measure_time();
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

#ifdef BONDI
    double norm = pow((pow(soundspeed, 2) + pow(bhvel, 2)), 1.5);

    if(norm > 0)
        mdot = 4. * M_PI * All.BlackHoleAccretionFactor * All.G * All.G *
            BHP(n).Mass * BHP(n).Mass * rho_proper / norm;
    else
        mdot = 0;
#endif


#ifdef ENFORCE_EDDINGTON_LIMIT
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

#ifdef BH_DRAG
    /* add a drag force for the black-holes,
       accounting for the accretion */
    double fac;

    if(BHP(n).Mass > 0)
    {
        fac = BHP(n).Mdot * dt / BHP(n).Mass;
        /*
           fac = meddington * dt / BHP(n).Mass;
           */
        if(fac > 1)
            fac = 1;

        if(dt > 0)
            for(k = 0; k < 3; k++)
                P[n].g.GravAccel[k] +=
                    -All.cf.a * All.cf.a * fac / dt * (P[n].Vel[k] - BHP(n).SurroundingGasVel[k]) / All.cf.a;
    }
#endif

    BHP(n).Mass += BHP(n).Mdot * dt;

#ifdef BH_BUBBLES
    BHP(n).Mass_bubbles += BHP(n).Mdot * dt;
#ifdef UNIFIED_FEEDBACK
    if(BHP(n).Mdot < All.RadioThreshold * meddington)
        BHP(n).Mass_radio += BHP(n).Mdot * dt;
#endif
#endif
}

static void blackhole_postprocess(int n) {
    int k;
#ifdef REPOSITION_ON_POTMIN
    if(BHP(n).MinPot < 0.5 * BHPOTVALUEINIT)
        for(k = 0; k < 3; k++)
            P[n].Pos[k] = BHP(n).MinPotPos[k];
#endif
    if(BHP(n).accreted_Mass > 0)
    {
        for(k = 0; k < 3; k++)
            P[n].Vel[k] =
                (P[n].Vel[k] * P[n].Mass + BHP(n).accreted_momentum[k]) /
                (P[n].Mass + BHP(n).accreted_Mass);

        P[n].Mass += BHP(n).accreted_Mass;
        BHP(n).Mass += BHP(n).accreted_BHMass;
#ifdef BH_BUBBLES
        BHP(n).Mass_bubbles += BHP(n).accreted_BHMass_bubbles;
#ifdef UNIFIED_FEEDBACK
        BHP(n).Mass_radio += BHP(n).accreted_BHMass_radio;
#endif
#endif
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
#ifdef BH_BUBBLES
    if(BHP(n).Mass_bubbles > 0
            && BHP(n).Mass_bubbles > All.BlackHoleRadioTriggeringFactor * BHP(n).Mass_ini) {
#pragma omp atomic
        num_activebh++;
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
#ifndef REPOSITION_ON_POTMIN
    ptypemask = 1 + (1 << 5);
#else
    ptypemask = 1 + 2 + 4 + 8 + 16 + 32;
#endif

    O->BH_TimeBinLimit = -1;
#ifdef REPOSITION_ON_POTMIN
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

            for(n = 0; n < numngb; n++)
            {
                int j = ngblist[n];

                if(P[j].Mass > 0)
                {
                    if(I->Mass > 0)
                    {
                        double dx = I->Pos[0] - P[j].Pos[0];
                        double dy = I->Pos[1] - P[j].Pos[1];
                        double dz = I->Pos[2] - P[j].Pos[2];
#ifdef PERIODIC			/*  now find the closest image in the given box size  */
                        if(dx > boxHalf_X)
                            dx -= boxSize_X;
                        if(dx < -boxHalf_X)
                            dx += boxSize_X;
                        if(dy > boxHalf_Y)
                            dy -= boxSize_Y;
                        if(dy < -boxHalf_Y)
                            dy += boxSize_Y;
                        if(dz > boxHalf_Z)
                            dz -= boxSize_Z;
                        if(dz < -boxHalf_Z)
                            dz += boxSize_Z;
#endif
                        double r2 = dx * dx + dy * dy + dz * dz;

#ifdef REPOSITION_ON_POTMIN
                        /* if this option is switched on, we may also encounter dark matter particles or stars */
                        if(r2 < kernel.HH)
                        {
                            if(P[j].p.Potential < O->BH_MinPot)
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
                                        O->BH_MinPot = P[j].p.Potential;
                                        for(k = 0; k < 3; k++)
                                            O->BH_MinPotPos[k] = P[j].Pos[k];
                                    }
                                }
                            }
                        }
#endif
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
                        if(P[j].Type == 0 && r2 < kernel.HH)
                        {
                            /* here we have a gas particle */

                            double r = sqrt(r2);
                            double u = r * kernel.Hinv;
                            double wk = density_kernel_wk(&kernel, u);
#ifdef SWALLOWGAS
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
#endif

                        }
                        if(P[j].Type == 0 && r2 < bh_feedback_kernel.HH) {
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
                            if(P[j].Mass > 0)
                            {
                                if (O->BH_TimeBinLimit <= 0 || O->BH_TimeBinLimit >= P[j].TimeBin) 
                                    O->BH_TimeBinLimit = P[j].TimeBin;
#ifdef BH_THERMALFEEDBACK
#ifndef UNIFIED_FEEDBACK
                                double energy = All.BlackHoleFeedbackFactor * 0.1 * I->Mdot * I->Dt *
                                    pow(C / All.UnitVelocity_in_cm_per_s, 2);

                                if(I->FeedbackWeightSum > 0)
                                {
                                    SPHP(j).i.dInjected_BH_Energy += FLT(energy * mass_j * wk / I->FeedbackWeightSum);
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
                                    if(I->FeedbackWeightSum> 0)
                                        SPHP(j).i.dInjected_BH_Energy += FLT(energy * mass_j * wk / I->FeedbackWeightSum);
                                }
#endif
#endif
                            }

                        }
                    }
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


int blackhole_swallow_evaluate(int target, int mode, 
        struct swallowdata_in * I, 
        struct swallowdata_out * O, 
        LocalEvaluator * lv, int * ngblist)
{
    int startnode, numngb, k, n, listindex = 0;

    int ptypemask = 0;
#ifndef REPOSITION_ON_POTMIN
    ptypemask = 1 + (1 << 5);
#else
    ptypemask = 1 + 2 + 4 + 8 + 16 + 32;
#endif

    startnode = I->NodeList[0];
    listindex ++;
    startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    
    O->Mass = 0;
    O->BH_Mass = 0;
    O->AccretedMomentum[0] = O->AccretedMomentum[1] = O->AccretedMomentum[2] = 0;

#ifdef BH_COUNTPROGS
    O->BH_CountProgs = 0;
#endif
#ifdef BH_BUBBLES
    O->BH_Mass_bubbles = 0;
    O->BH_Mass_radio = 0;
#endif

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb = ngb_treefind_threads(I->Pos, I->Hsml, target, &startnode, 
                    mode, lv, ngblist, NGB_TREEFIND_SYMMETRIC, ptypemask);

            if(numngb < 0)
                return numngb;

            for(n = 0; n < numngb; n++)
            {
                int j = ngblist[n];

                if(P[j].SwallowID == I->ID)
                {
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

                        O->Mass += FLT(P[j].Mass);
                        O->BH_Mass += FLT(BHP(j).Mass);
#ifdef BH_BUBBLES
                        O->BH_Mass_bubbles += FLT(BHP(j).Mass_bubbles - BHP(j).Mass_ini);
#ifdef UNIFIED_FEEDBACK
                        O->BH_Mass_radio += FLT(BHP(j).Mass_radio - BHP(j).Mass_ini);
#endif
#endif
                        for(k = 0; k < 3; k++)
                            O->AccretedMomentum[k] += FLT(P[j].Mass * P[j].Vel[k]);

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

#ifdef BH_BUBBLES
                        BHP(j).Mass_bubbles = 0;
                        BHP(j).Mass_ini = 0;
#ifdef UNIFIED_FEEDBACK
                        BHP(j).Mass_radio = 0;
#endif
#endif
#pragma omp atomic
                        N_BH_swallowed++;
                    }
                }

                if(P[j].Type == 0)
                {
                    if(P[j].SwallowID == I->ID)
                    {
                        O->Mass += FLT(P[j].Mass);

                        for(k = 0; k < 3; k++)
                            O->AccretedMomentum[k] += FLT(P[j].Mass * P[j].Vel[k]);

                        P[j].Mass = 0;
                        int bin = P[j].TimeBin;
#pragma omp atomic
                        N_sph_swallowed++;
                    }
                }
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
    return P[n].Type == 5;
}
static void * blackhole_alloc_ngblist() {
    int threadid = omp_get_thread_num();
    return Ngblist + threadid * NumPart;
}
static void blackhole_feedback_reduce(int place, struct feedbackdata_out * remote, int mode) {
    int k;
#ifdef REPOSITION_ON_POTMIN
    if(mode == 0 || BHP(place).MinPot > remote->BH_MinPot)
    {
        BHP(place).MinPot = remote->BH_MinPot;
        for(k = 0; k < 3; k++)
            BHP(place).MinPotPos[k] = remote->BH_MinPotPos[k];
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

#define REDUCE(A, B) (A) = (mode==0)?(B):((A) + (B))
    REDUCE(BHP(place).accreted_Mass, remote->Mass);
    REDUCE(BHP(place).accreted_BHMass, remote->BH_Mass);
#ifdef BH_BUBBLES
    REDUCE(BHP(place).accreted_BHMass_bubbles, remote->BH_Mass_bubbles);
#ifdef UNIFIED_FEEDBACK
    REDUCE(BHP(place).accreted_BHMass_radio, remote->BH_Mass_radio);
#endif
#endif
    for(k = 0; k < 3; k++) {
        REDUCE(BHP(place).accreted_momentum[k], remote->AccretedMomentum[k]);
    }
#ifdef BH_COUNTPROGS
    REDUCE(BHP(place).CountProgs, remote->BH_CountProgs);
#endif
}

#ifdef BH_BUBBLES
void bh_bubble(double bh_dmass, MyFloat center[3], MyIDType BH_id)
{
    double phi, theta;
    double dx, dy, dz, rr, r2, dE;
    double E_bubble, totE_bubble;
    double BubbleDistance = 0.0, BubbleRadius = 0.0, BubbleEnergy = 0.0;
    double ICMDensity;
    double Mass_bubble, totMass_bubble;
    double u_to_temp_fac;
    MyDouble pos[3];
    int numngb, tot_numngb, startnode, numngb_inbox;
    int n, i, j, dummy;

#ifdef CR_BUBBLES
    double tinj = 0.0, instant_reheat = 0.0;
    double sum_instant_reheat = 0.0, tot_instant_reheat = 0.0;
#endif

    u_to_temp_fac = (4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC))) * PROTONMASS /
        BOLTZMANN * GAMMA_MINUS1 * All.UnitEnergy_in_cgs / All.UnitMass_in_g;

    if(All.ComovingIntegrationOn)
    {

        BubbleDistance = All.BubbleDistance;
        BubbleRadius = All.BubbleRadius;

        /*switch to comoving if it is assumed that Rbub should be constant with redshift */

        /* BubbleDistance = All.BubbleDistance / All.Time;
           BubbleRadius = All.BubbleRadius / All.Time; */
    }
    else
    {
        BubbleDistance = All.BubbleDistance;
        BubbleRadius = All.BubbleRadius;
    }

    BubbleEnergy = All.RadioFeedbackFactor * 0.1 * bh_dmass * All.UnitMass_in_g / All.HubbleParam * pow(C, 2);	/*in cgs units */

    phi = 2 * M_PI * get_random_number(BH_id);
    theta = acos(2 * get_random_number(BH_id + 1) - 1);
    rr = pow(get_random_number(BH_id + 2), 1. / 3.) * BubbleDistance;

    pos[0] = sin(theta) * cos(phi);
    pos[1] = sin(theta) * sin(phi);
    pos[2] = cos(theta);

    for(i = 0; i < 3; i++)
        pos[i] *= rr;

    for(i = 0; i < 3; i++)
        pos[i] += center[i];


    /* First, let's see how many particles are in the bubble of the default radius */

    numngb = 0;
    E_bubble = 0.;
    Mass_bubble = 0.;

    startnode = All.MaxPart;
    do
    {
        numngb_inbox = ngb_treefind_variable(pos, BubbleRadius, -1, &startnode, 0, &dummy, &dummy);

        for(n = 0; n < numngb_inbox; n++)
        {
            j = Ngblist[n];
            dx = pos[0] - P[j].Pos[0];
            dy = pos[1] - P[j].Pos[1];
            dz = pos[2] - P[j].Pos[2];

#ifdef PERIODIC			/*  now find the closest image in the given box size  */
            if(dx > boxHalf_X)
                dx -= boxSize_X;
            if(dx < -boxHalf_X)
                dx += boxSize_X;
            if(dy > boxHalf_Y)
                dy -= boxSize_Y;
            if(dy < -boxHalf_Y)
                dy += boxSize_Y;
            if(dz > boxHalf_Z)
                dz -= boxSize_Z;
            if(dz < -boxHalf_Z)
                dz += boxSize_Z;
#endif
            r2 = dx * dx + dy * dy + dz * dz;

            if(r2 < BubbleRadius * BubbleRadius)
            {
                if(P[j].Type == 0)
                {
                    numngb++;

                    if(All.ComovingIntegrationOn)
                        E_bubble +=
                            SPHP(j).Entropy * P[j].Mass * pow(SPHP(j).EOMDensity / pow(All.Time, 3),
                                    GAMMA_MINUS1) / GAMMA_MINUS1;
                    else
                        E_bubble +=
                            SPHP(j).Entropy * P[j].Mass * pow(SPHP(j).EOMDensity, GAMMA_MINUS1) / GAMMA_MINUS1;

                    Mass_bubble += P[j].Mass;
                }
            }
        }
    }
    while(startnode >= 0);


    MPI_Allreduce(&numngb, &tot_numngb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&E_bubble, &totE_bubble, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&Mass_bubble, &totMass_bubble, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    totE_bubble *= All.UnitEnergy_in_cgs;

    if(totMass_bubble > 0)
    {
        if(ThisTask == 0)
        {
            printf("found %d particles in bubble with energy %g and total mass %g \n",
                    tot_numngb, totE_bubble, totMass_bubble);
            fflush(stdout);
        }


        /*calculate comoving density of ICM inside the bubble */

        ICMDensity = totMass_bubble / (4.0 * M_PI / 3.0 * pow(BubbleRadius, 3));

        if(All.ComovingIntegrationOn)
            ICMDensity = ICMDensity / (pow(All.Time, 3));	/*now physical */

        /*Rbub=R0*[(Ejet/Ejet,0)/(rho_ICM/rho_ICM,0)]^(1./5.) - physical */

        rr = rr / BubbleDistance;

        BubbleRadius =
            All.BubbleRadius * pow((BubbleEnergy * All.DefaultICMDensity / (All.BubbleEnergy * ICMDensity)),
                    1. / 5.);

        BubbleDistance =
            All.BubbleDistance * pow((BubbleEnergy * All.DefaultICMDensity / (All.BubbleEnergy * ICMDensity)),
                    1. / 5.);

        if(All.ComovingIntegrationOn)
        {
            /*switch to comoving if it is assumed that Rbub should be constant with redshift */
            /* BubbleRadius = BubbleRadius / All.Time;
               BubbleDistance = BubbleDistance / All.Time; */
        }

        /*recalculate pos */
        rr = rr * BubbleDistance;

        pos[0] = sin(theta) * cos(phi);
        pos[1] = sin(theta) * sin(phi);
        pos[2] = cos(theta);

        for(i = 0; i < 3; i++)
            pos[i] *= rr;

        for(i = 0; i < 3; i++)
            pos[i] += center[i];

        /* now find particles in Bubble again,
           and recalculate number, mass and energy */

        numngb = 0;
        E_bubble = 0.;
        Mass_bubble = 0.;
        tot_numngb = 0;
        totE_bubble = 0.;
        totMass_bubble = 0.;

        startnode = All.MaxPart;

        do
        {
            numngb_inbox = ngb_treefind_variable(pos, BubbleRadius, -1, &startnode, 0, &dummy, &dummy);

            for(n = 0; n < numngb_inbox; n++)
            {
                j = Ngblist[n];
                dx = pos[0] - P[j].Pos[0];
                dy = pos[1] - P[j].Pos[1];
                dz = pos[2] - P[j].Pos[2];

#ifdef PERIODIC			/*  now find the closest image in the given box size  */
                if(dx > boxHalf_X)
                    dx -= boxSize_X;
                if(dx < -boxHalf_X)
                    dx += boxSize_X;
                if(dy > boxHalf_Y)
                    dy -= boxSize_Y;
                if(dy < -boxHalf_Y)
                    dy += boxSize_Y;
                if(dz > boxHalf_Z)
                    dz -= boxSize_Z;
                if(dz < -boxHalf_Z)
                    dz += boxSize_Z;
#endif
                r2 = dx * dx + dy * dy + dz * dz;

                if(r2 < BubbleRadius * BubbleRadius)
                {
                    if(P[j].Type == 0 && P[j].Mass > 0)
                    {
                        numngb++;

                        if(All.ComovingIntegrationOn)
                            E_bubble +=
                                SPHP(j).Entropy * P[j].Mass * pow(SPHP(j).EOMDensity / pow(All.Time, 3),
                                        GAMMA_MINUS1) / GAMMA_MINUS1;
                        else
                            E_bubble +=
                                SPHP(j).Entropy * P[j].Mass * pow(SPHP(j).EOMDensity, GAMMA_MINUS1) / GAMMA_MINUS1;

                        Mass_bubble += P[j].Mass;
                    }
                }
            }
        }
        while(startnode >= 0);


        MPI_Allreduce(&numngb, &tot_numngb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&E_bubble, &totE_bubble, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&Mass_bubble, &totMass_bubble, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        totE_bubble *= All.UnitEnergy_in_cgs;

        if(totMass_bubble > 0)
        {
            if(ThisTask == 0)
            {
                printf("found %d particles in bubble of rescaled radius with energy %g and total mass %g \n",
                        tot_numngb, totE_bubble, totMass_bubble);
                printf("energy shall be increased by: (Eini+Einj)/Eini = %g \n",
                        (BubbleEnergy + totE_bubble) / totE_bubble);
                fflush(stdout);
            }
        }

        /* now find particles in Bubble again, and inject energy */

#ifdef CR_BUBBLES
        sum_instant_reheat = 0.0;
        tot_instant_reheat = 0.0;
#endif

        startnode = All.MaxPart;

        do
        {
            numngb_inbox = ngb_treefind_variable(pos, BubbleRadius, -1, &startnode, 0, &dummy, &dummy);

            for(n = 0; n < numngb_inbox; n++)
            {
                j = Ngblist[n];
                dx = pos[0] - P[j].Pos[0];
                dy = pos[1] - P[j].Pos[1];
                dz = pos[2] - P[j].Pos[2];

#ifdef PERIODIC			/*  now find the closest image in the given box size  */
                if(dx > boxHalf_X)
                    dx -= boxSize_X;
                if(dx < -boxHalf_X)
                    dx += boxSize_X;
                if(dy > boxHalf_Y)
                    dy -= boxSize_Y;
                if(dy < -boxHalf_Y)
                    dy += boxSize_Y;
                if(dz > boxHalf_Z)
                    dz -= boxSize_Z;
                if(dz < -boxHalf_Z)
                    dz += boxSize_Z;
#endif
                r2 = dx * dx + dy * dy + dz * dz;

                if(r2 < BubbleRadius * BubbleRadius)
                {
                    if(P[j].Type == 0 && P[j].Mass > 0)
                    {
                        /* energy we want to inject in this particle */

                        if(All.StarformationOn)
                            dE = ((BubbleEnergy / All.UnitEnergy_in_cgs) / totMass_bubble) * P[j].Mass;
                        else
                            dE = (BubbleEnergy / All.UnitEnergy_in_cgs) / tot_numngb;

                        if(u_to_temp_fac * dE / P[j].Mass > 5.0e9)
                            dE = 5.0e9 * P[j].Mass / u_to_temp_fac;

#ifndef CR_BUBBLES
                        if(All.ComovingIntegrationOn)
                            SPHP(j).Entropy +=
                                GAMMA_MINUS1 * dE / P[j].Mass / pow(SPHP(j).EOMDensity / pow(All.Time, 3),
                                        GAMMA_MINUS1);
                        else
                            SPHP(j).Entropy +=
                                GAMMA_MINUS1 * dE / P[j].Mass / pow(SPHP(j).EOMDensity, GAMMA_MINUS1);
#else

                        tinj = 10.0 * All.HubbleParam * All.cf.hubble / All.UnitTime_in_Megayears;

                        instant_reheat =
                            CR_Particle_SupernovaFeedback(&SPHP(j), dE / P[j].Mass * All.CR_AGNEff, tinj);

                        if(instant_reheat > 0)
                        {
                            if(All.ComovingIntegrationOn)
                                SPHP(j).Entropy +=
                                    instant_reheat * GAMMA_MINUS1 / pow(SPHP(j).EOMDensity / pow(All.Time, 3),
                                            GAMMA_MINUS1);
                            else
                                SPHP(j).Entropy +=
                                    instant_reheat * GAMMA_MINUS1 / pow(SPHP(j).EOMDensity, GAMMA_MINUS1);
                        }

                        if(All.CR_AGNEff < 1)
                        {
                            if(All.ComovingIntegrationOn)
                                SPHP(j).Entropy +=
                                    (1 -
                                     All.CR_AGNEff) * dE * GAMMA_MINUS1 / P[j].Mass / pow(SPHP(j).EOMDensity /
                                         pow(All.Time, 3),
                                         GAMMA_MINUS1);
                            else
                                SPHP(j).Entropy +=
                                    (1 - All.CR_AGNEff) * dE * GAMMA_MINUS1 / P[j].Mass / pow(SPHP(j).EOMDensity,
                                            GAMMA_MINUS1);
                        }


                        sum_instant_reheat += instant_reheat * P[j].Mass;
#endif

                    }
                }
            }
        }
        while(startnode >= 0);

#ifdef CR_BUBBLES
        MPI_Allreduce(&sum_instant_reheat, &tot_instant_reheat, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if(ThisTask == 0)
        {
            printf("Total BubbleEnergy %g Thermalized Energy %g \n", BubbleEnergy,
                    tot_instant_reheat * All.UnitEnergy_in_cgs);
            fflush(stdout);

        }
#endif
    }
    else
    {
        if(ThisTask == 0)
        {
            printf("No particles in bubble found! \n");
            fflush(stdout);
        }

    }

}
#endif /* end of BH_BUBBLE */

void blackhole_make_one(int index) {
    if(P[index].Type != 0) endrun(7772);

    P[index].PI = N_bh;
    N_bh ++;
    P[index].Type = 5;	/* make it a black hole particle */
#ifdef STELLARAGE
    P[index].StellarAge = All.Time;
#endif
    BHP(index).ID = P[index].ID;
    BHP(index).Mass = All.SeedBlackHoleMass;
    BHP(index).Mdot = 0;

#ifdef BH_COUNTPROGS
    BHP(index).CountProgs = 1;
#endif

#ifdef BH_BUBBLES
    BHP(index).Mass_bubbles = All.SeedBlackHoleMass;
    BHP(index).Mass_ini = All.SeedBlackHoleMass;
#ifdef UNIFIED_FEEDBACK
    BHP(index).Mass_radio = All.SeedBlackHoleMass;
#endif
#endif

#ifdef SFR
    Stars_converted++;
#endif
    TimeBinCountSph[P[index].TimeBin]--;
}

void blackhole_make_extra() {
    int i;
    int converted = 0;
    int ntot = 0;
    for(i = 0; i < NumPart; i++) {
        if(P[i].Type != 0) continue;
        blackhole_make_one(i);
        converted ++;
        break;
    }
    MPI_Allreduce(&converted, &ntot, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    All.TotN_sph -= ntot;
    All.TotN_bh += ntot;
}

#endif
