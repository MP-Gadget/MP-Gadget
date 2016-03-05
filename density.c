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

extern int NextParticle;

/*! Structure for communication during the density computation. Holds data that is sent to other processors.
*/
struct densdata_in
{
    int NodeList[NODELISTLENGTH];
    MyIDType ID;
    MyDouble Pos[3];
    MyFloat Vel[3];
    MyFloat Hsml;
#ifdef VOLUME_CORRECTION
    MyFloat DensityOld;
#endif
#ifdef WINDS
    MyFloat DelayTime;
#endif
    int Type;
};

struct densdata_out
{
    MyIDType ID;
#ifdef DENSITY_INDEPENDENT_SPH
    MyFloat EgyRho;
    MyFloat DhsmlEgyDensity;
#endif
    MyDouble Rho;
    MyDouble DhsmlDensity;
    MyDouble Ngb;
    MyDouble Div, Rot[3];

#ifdef BLACK_HOLES
    MyDouble SmoothedEntOrPressure;
    MyDouble FeedbackWeightSum;
    MyDouble GasVel[3];
#endif

#ifdef HYDRO_COST_FACTOR
    int Ninteractions;
#endif

#ifdef VOLUME_CORRECTION
    MyFloat DensityStd;
#endif

#ifdef SPH_GRAD_RHO
    MyFloat GradRho[3];
#endif
};

static int density_isactive(int n);
static int density_evaluate(int target, int mode, struct densdata_in * I, struct densdata_out * O, LocalEvaluator * lv);
static void density_post_process(int i);
static void density_check_neighbours(int i, MyFloat * Left, MyFloat * Right);


static void density_reduce(int place, struct densdata_out * remote, int mode);
static void density_copy(int place, struct densdata_in * I);

/*! \file density.c
 *  \brief SPH density computation and smoothing length determination
 *
 *  This file contains the "first SPH loop", where the SPH densities and some
 *  auxiliary quantities are computed.  There is also functionality that
 *  corrects the smoothing length if needed.
 */


/*! This function computes the local density for each active SPH particle, the
 * number of neighbours in the current smoothing radius, and the divergence
 * and rotation of the velocity field.  The pressure is updated as well.  If a
 * particle with its smoothing region is fully inside the local domain, it is
 * not exported to the other processors. The function also detects particles
 * that have a number of neighbours outside the allowed tolerance range. For
 * these particles, the smoothing length is adjusted accordingly, and the
 * density() computation is called again.  Note that the smoothing length is
 * not allowed to fall below the lower bound set by MinGasHsml (this may mean
 * that one has to deal with substantially more than normal number of
 * neighbours.)
 */

void density(void)
{
    MyFloat *Left, *Right;

    Evaluator ev = {0};

    ev.ev_label = "DENSITY";
    ev.ev_evaluate = (ev_ev_func) density_evaluate;
    ev.ev_isactive = density_isactive;
    ev.ev_copy = (ev_copy_func) density_copy;
    ev.ev_reduce = (ev_reduce_func) density_reduce;
    ev.UseNodeList = 1;
    ev.ev_datain_elsize = sizeof(struct densdata_in);
    ev.ev_dataout_elsize = sizeof(struct densdata_out);

    int i, npleft, iter = 0;

    int64_t ntot = 0;

    double timeall = 0;
    double timecomp, timecomp3 = 0, timecomm, timewait;

    double tstart, tend;

    walltime_measure("/Misc");

    Left = (MyFloat *) mymalloc("Left", NumPart * sizeof(MyFloat));
    Right = (MyFloat *) mymalloc("Right", NumPart * sizeof(MyFloat));

    int Nactive;
    int * queue;

    /* this has to be done before get_queue so that
     * all particles are return for the first loop over all active particles.
     * */
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        P[i].DensityIterationDone = 0;
    }

    /* the queue has every particle. Later on after some iterations are done
     * Nactive will decrease -- the queue would be shorter.*/
    queue = ev_get_queue(&ev, &Nactive);
#pragma omp parallel for if(Nactive > 32)
    for(i = 0; i < Nactive; i ++) {
        int p = queue[i];
        Left[p] = 0;
        Right[p] = 0;
#ifdef BLACK_HOLES
        P[p].SwallowID = 0;
#endif
    }
    myfree(queue);

    /* allocate buffers to arrange communication */

    walltime_measure("/SPH/Density/Init");

    /* we will repeat the whole thing for those particles where we didn't find enough neighbours */
    do
    {

        ev_run(&ev);

        /* do final operations on results */
        tstart = second();

        queue = ev_get_queue(&ev, &Nactive);

        npleft = 0;
#pragma omp parallel for if(Nactive > 32)
        for(i = 0; i < Nactive; i++) {
            int p = queue[i];
            density_post_process(p);
            /* will notify by setting DensityIterationDone */
            density_check_neighbours(p, Left, Right);
            if(iter >= MAXITER - 10)
            {
                printf
                    ("i=%d task=%d ID=%lu Hsml=%g Left=%g Right=%g Ngbs=%g Right-Left=%g\n   pos=(%g|%g|%g)\n",
                     queue[p], ThisTask, P[p].ID, P[p].Hsml, Left[p], Right[p],
                     (float) P[p].n.NumNgb, Right[p] - Left[p], P[p].Pos[0], P[p].Pos[1], P[p].Pos[2]);
                fflush(stdout);
            }

            if(!P[p].DensityIterationDone) {
#pragma omp atomic
                npleft ++;
            }
        }

        myfree(queue);
        tend = second();
        timecomp3 += timediff(tstart, tend);

        sumup_large_ints(1, &npleft, &ntot);

        if(ntot > 0)
        {
            iter++;

            /*
            if(ntot < 1 ) {
                for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
                {
                    if(density_isactive(i) && !P[i].DensityIterationDone) {
                        printf
                            ("i=%d task=%d ID=%llu type=%d, Hsml=%g Left=%g Right=%g Ngbs=%g Right-Left=%g\n   pos=(%g|%g|%g)\n",
                             i, ThisTask, P[i].ID, P[i].Type, P[i].Hsml, Left[i], Right[i],
                             (float) P[i].n.NumNgb, Right[i] - Left[i], P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
                        fflush(stdout);
                    }
                }

            }
            */

            if(iter > 0 && ThisTask == 0)
            {
                printf("ngb iteration %d: need to repeat for %d%09d particles.\n", iter,
                        (int) (ntot / 1000000000), (int) (ntot % 1000000000));
                fflush(stdout);
            }

            if(iter > MAXITER)
            {
                printf("failed to converge in neighbour iteration in density()\n");
                fflush(stdout);
                endrun(1155);
            }
        }
    }
    while(ntot > 0);

    myfree(Right);
    myfree(Left);


    /* collect some timing information */

    timeall = walltime_measure(WALLTIME_IGNORE);

    timecomp = timecomp3 + ev.timecomp1 + ev.timecomp2;
    timewait = ev.timewait1 + ev.timewait2;
    timecomm = ev.timecommsumm1 + ev.timecommsumm2;

    walltime_add("/SPH/Density/Compute", timecomp);
    walltime_add("/SPH/Density/Wait", timewait);
    walltime_add("/SPH/Density/Comm", timecomm);
    walltime_add("/SPH/Density/Misc", timeall - (timecomp + timewait + timecomm));
}

double density_decide_hsearch(int targettype, double h) {
#ifdef BLACK_HOLES
    if(targettype == 5 && All.BlackHoleFeedbackRadius > 0) {
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
#else
    return h;
#endif

}

static void density_copy(int place, struct densdata_in * I) {
    I->ID = P[place].ID;
    I->Pos[0] = P[place].Pos[0];
    I->Pos[1] = P[place].Pos[1];
    I->Pos[2] = P[place].Pos[2];
    I->Hsml = P[place].Hsml;

    I->Type = P[place].Type;

#if defined(BLACK_HOLES)
    if(P[place].Type != 0)
    {
        I->Vel[0] = 0;
        I->Vel[1] = 0;
        I->Vel[2] = 0;
    }
    else
#endif
    {
        I->Vel[0] = SPHP(place).VelPred[0];
        I->Vel[1] = SPHP(place).VelPred[1];
        I->Vel[2] = SPHP(place).VelPred[2];
    }
#ifdef VOLUME_CORRECTION
    I->DensityOld = SPHP(place).DensityOld;
#endif

#ifdef WINDS
    I->DelayTime = SPHP(place).DelayTime;
#endif

}

static void density_reduce(int place, struct densdata_out * remote, int mode) {
    EV_REDUCE(P[place].n.dNumNgb, remote->Ngb);

    if(remote->ID != P[place].ID) {
        BREAKPOINT;
    }
#ifdef HYDRO_COST_FACTOR
    /* these will be added */
    P[place].GravCost += HYDRO_COST_FACTOR * All.cf.a * remote->Ninteractions;
#endif

    if(P[place].Type == 0)
    {
        EV_REDUCE(SPHP(place).Density, remote->Rho);
        EV_REDUCE(SPHP(place).DhsmlDensityFactor, remote->DhsmlDensity);
#ifdef DENSITY_INDEPENDENT_SPH
        EV_REDUCE(SPHP(place).EgyWtDensity, remote->EgyRho);
        EV_REDUCE(SPHP(place).DhsmlEgyDensityFactor, remote->DhsmlEgyDensity);
#endif

        EV_REDUCE(SPHP(place).DivVel, remote->Div);
        EV_REDUCE(SPHP(place).Rot[0], remote->Rot[0]);
        EV_REDUCE(SPHP(place).Rot[1], remote->Rot[1]);
        EV_REDUCE(SPHP(place).Rot[2], remote->Rot[2]);

#ifdef VOLUME_CORRECTION
        EV_REDUCE(SPHP(place).DensityStd, remote->DensityStd);
#endif

#ifdef SPH_GRAD_RHO
        EV_REDUCE(SPHP(place).GradRho[0], remote->GradRho[0]);
        EV_REDUCE(SPHP(place).GradRho[1], remote->GradRho[1]);
        EV_REDUCE(SPHP(place).GradRho[2], remote->GradRho[2]);
#endif

    }

#ifdef BLACK_HOLES
    if(P[place].Type == 5)
    {
        EV_REDUCE(BHP(place).Density, remote->Rho);
        EV_REDUCE(BHP(place).FeedbackWeightSum, remote->FeedbackWeightSum);
        EV_REDUCE(BHP(place).EntOrPressure, remote->SmoothedEntOrPressure);
#ifdef BH_USE_GASVEL_IN_BONDI
        EV_REDUCE(BHP(place).SurroundingGasVel[0], remote->GasVel[0]);
        EV_REDUCE(BHP(place).SurroundingGasVel[1], remote->GasVel[1]);
        EV_REDUCE(BHP(place).SurroundingGasVel[2], remote->GasVel[2]);
#endif
    }
#endif
}
/*! This function represents the core of the SPH density computation. The
 *  target particle may either be local, or reside in the communication
 *  buffer.
 */
static int density_evaluate(int target, int mode,
        struct densdata_in * I,
        struct densdata_out * O,
        LocalEvaluator * lv)
{
    int n;

    int startnode, numngb, numngb_inbox, listindex = 0;

    double h;
    double hsearch;

    int ninteractions = 0;
    int nnodesinlist = 0;

    density_kernel_t kernel;
#ifdef BLACK_HOLES
    density_kernel_t bh_feedback_kernel;
#endif

    startnode = I->NodeList[0];
    listindex ++;
    startnode = Nodes[startnode].u.d.nextnode;	/* open it */

    h = I->Hsml;
    hsearch = density_decide_hsearch(I->Type, h);

    density_kernel_init(&kernel, h);
    double kernel_volume = density_kernel_volume(&kernel);

#ifdef BLACK_HOLES
    density_kernel_init(&bh_feedback_kernel, hsearch);
#endif

    numngb = 0;

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb_inbox =
                ngb_treefind_threads(I->Pos, hsearch, target, &startnode,
                        mode, lv, NGB_TREEFIND_ASYMMETRIC, 1); /* gas only 1<<0 */

            if(numngb_inbox < 0)
                return numngb_inbox;

            for(n = 0; n < numngb_inbox; n++)
            {
                ninteractions++;
                int j = lv->ngblist[n];
#ifdef WINDS
                if(HAS(All.WindModel, WINDS_DECOUPLE_SPH)) {
                    if(SPHP(j).DelayTime > 0)	/* partner is a wind particle */
                        if(!(I->DelayTime > 0))	/* if I'm not wind, then ignore the wind particle */
                            continue;
                }
#endif
#ifdef BLACK_HOLES
                if(P[j].Mass == 0)
                    continue;
#ifdef WINDS
                    /* blackhole doesn't accrete from wind, regardlies coupled or
                     * not */
                if(I->Type == 5 && SPHP(j).DelayTime > 0)	/* partner is a wind particle */
                            continue;
#endif
#endif
                double dx = I->Pos[0] - P[j].Pos[0];
                double dy = I->Pos[1] - P[j].Pos[1];
                double dz = I->Pos[2] - P[j].Pos[2];

                dx = NEAREST(dx);
                dy = NEAREST(dy);
                dz = NEAREST(dz);

                double r2 = dx * dx + dy * dy + dz * dz;

                double r = sqrt(r2);

                if(r2 < kernel.HH)
                {

                    double u = r * kernel.Hinv;
                    double wk = density_kernel_wk(&kernel, u);
                    double dwk = density_kernel_dwk(&kernel, u);

                    double mass_j = P[j].Mass;

                    numngb++;
#ifdef VOLUME_CORRECTION
                    O->Rho += (mass_j * wk * pow(I->DensityOld / SPHP(j).DensityOld, VOLUME_CORRECTION));
                    O->DensityStd += (mass_j * wk);
#else
                    O->Rho += (mass_j * wk);
#endif
                    O->Ngb += wk * kernel_volume;

                    /* Hinv is here becuase O->DhsmlDensity is drho / dH.
                     * nothing to worry here */
                    O->DhsmlDensity += mass_j * density_kernel_dW(&kernel, u, wk, dwk);

#ifdef DENSITY_INDEPENDENT_SPH
                    O->EgyRho += mass_j * SPHP(j).EntVarPred * wk;
                    O->DhsmlEgyDensity += mass_j * SPHP(j).EntVarPred * density_kernel_dW(&kernel, u, wk, dwk);
#endif


#ifdef BLACK_HOLES
#ifdef BH_CSND_FROM_PRESSURE
                    O->SmoothedEntOrPressure += (mass_j * wk * SPHP(j).Pressure);
#else
                    O->SmoothedEntOrPressure += (mass_j * wk * SPHP(j).Entropy);
#endif
                    O->GasVel[0] += (mass_j * wk * SPHP(j).VelPred[0]);
                    O->GasVel[1] += (mass_j * wk * SPHP(j).VelPred[1]);
                    O->GasVel[2] += (mass_j * wk * SPHP(j).VelPred[2]);
#endif

#ifdef SPH_GRAD_RHO
                    if(r > 0)
                    {
                        O->GradRho[0] += mass_j * dwk * dx / r;
                        O->GradRho[1] += mass_j * dwk * dy / r;
                        O->GradRho[2] += mass_j * dwk * dz / r;
                    }
#endif


                    if(r > 0)
                    {
                        double fac = mass_j * dwk / r;

                        double dvx = I->Vel[0] - SPHP(j).VelPred[0];
                        double dvy = I->Vel[1] - SPHP(j).VelPred[1];
                        double dvz = I->Vel[2] - SPHP(j).VelPred[2];

                        O->Div += (-fac * (dx * dvx + dy * dvy + dz * dvz));

                        O->Rot[0] += (fac * (dz * dvy - dy * dvz));
                        O->Rot[1] += (fac * (dx * dvz - dz * dvx));
                        O->Rot[2] += (fac * (dy * dvx - dx * dvy));
                    }
                }
#ifdef BLACK_HOLES
                if(I->Type == 5 && r2 < bh_feedback_kernel.HH)
                {
#ifdef WINDS
                    /* blackhole doesn't accrete from wind, regardlies coupled or
                     * not */
                    if(SPHP(j).DelayTime > 0)	/* partner is a wind particle */
                            continue;
#endif
                    double mass_j;
                    if(HAS(All.BlackHoleFeedbackMethod, BH_FEEDBACK_OPTTHIN)) {
#ifdef COOLING
                        double nh0 = 0;
                        double nHeII = 0;
                        double ne = SPHP(j).Ne;
                        struct UVBG uvbg;
                        GetParticleUVBG(j, &uvbg);
#pragma omp critical (_abundance_)
                        AbundanceRatios(DMAX(All.MinEgySpec,
                                    SPHP(j).Entropy / GAMMA_MINUS1
                                    * pow(SPHP(j).EOMDensity * All.cf.a3inv,
                                        GAMMA_MINUS1)),
                                SPHP(j).Density * All.cf.a3inv, &uvbg, &ne, &nh0, &nHeII);
#else
                        double nh0 = 1.0;
#endif
                        if(r2 > 0)
                            O->FeedbackWeightSum += (P[j].Mass * nh0) / r2;
                    } else {
                        if(HAS(All.BlackHoleFeedbackMethod, BH_FEEDBACK_MASS)) {
                            mass_j = P[j].Mass;
                        } else {
                            mass_j = P[j].Hsml * P[j].Hsml * P[j].Hsml;
                        }
                        if(HAS(All.BlackHoleFeedbackMethod, BH_FEEDBACK_SPLINE)) {
                            double u = r * bh_feedback_kernel.Hinv;
                            O->FeedbackWeightSum += (mass_j *
                                  density_kernel_wk(&bh_feedback_kernel, u)
                                   );
                        } else {
                            O->FeedbackWeightSum += (mass_j);
                        }
                    }
                }
#endif
            }
        }
        /* now check next node in the node list */
        if(listindex < NODELISTLENGTH)
        {
            startnode = I->NodeList[listindex];
            if(startnode >= 0) {
                startnode = Nodes[startnode].u.d.nextnode;	/* open it */
                listindex++;
                nnodesinlist ++;
            }
        }
    }

    /* Now collect the result at the right place */

    O->ID = I->ID;
    /* some performance measures not currently used */
#ifdef HYDRO_COST_FACTOR
    O->Ninteractions = ninteractions;
#endif
    lv->Ninteractions += ninteractions;
    lv->Nnodesinlist += nnodesinlist;

    return 0;
}

static int density_isactive(int n)
{
    if(P[n].DensityIterationDone) return 0;

    if(P[n].TimeBin < 0) {
        printf("TimeBin negative!\n use DensityIterationDone flag");
        endrun(99999);
    }
#ifdef BLACK_HOLES
    if(P[n].Type == 5)
        return 1;
#endif

    if(P[n].Type == 0)
        return 1;

    return 0;
}

static void density_post_process(int i) {
    int dt_step;
    if(P[i].Type == 0)
    {
        if(SPHP(i).Density > 0)
        {
#ifdef VOLUME_CORRECTION
            SPHP(i).DensityOld = SPHP(i).DensityStd;
#endif
            SPHP(i).DhsmlDensityFactor *= P[i].Hsml / (NUMDIMS * SPHP(i).Density);
            if(SPHP(i).DhsmlDensityFactor > -0.9)	/* note: this would be -1 if only a single particle at zero lag is found */
                SPHP(i).DhsmlDensityFactor = 1 / (1 + SPHP(i).DhsmlDensityFactor);
            else
                SPHP(i).DhsmlDensityFactor = 1;

#ifdef DENSITY_INDEPENDENT_SPH
            if((SPHP(i).EntVarPred>0)&&(SPHP(i).EgyWtDensity>0))
            {
                SPHP(i).DhsmlEgyDensityFactor *= P[i].Hsml/ (NUMDIMS * SPHP(i).EgyWtDensity);
                SPHP(i).DhsmlEgyDensityFactor *= -SPHP(i).DhsmlDensityFactor;
                SPHP(i).EgyWtDensity /= SPHP(i).EntVarPred;
            } else {
                SPHP(i).DhsmlEgyDensityFactor=0;
                SPHP(i).EntVarPred=0;
                SPHP(i).EgyWtDensity=0;
            }
#endif

            SPHP(i).CurlVel = sqrt(SPHP(i).Rot[0] * SPHP(i).Rot[0] +
                    SPHP(i).Rot[1] * SPHP(i).Rot[1] +
                    SPHP(i).Rot[2] * SPHP(i).Rot[2]) / SPHP(i).Density;

            SPHP(i).DivVel /= SPHP(i).Density;

        }

#ifndef WAKEUP
        dt_step = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0);
#else
        dt_step = P[i].dt_step;
#endif

        int dt_entr = (All.Ti_Current - (P[i].Ti_begstep + dt_step / 2)) * All.Timebase_interval;
#ifndef EOS_DEGENERATE
    #ifndef SOFTEREQS
        #ifndef TRADITIONAL_SPH_FORMULATION
            #ifdef DENSITY_INDEPENDENT_SPH
        SPHP(i).Pressure = pow(SPHP(i).EntVarPred*SPHP(i).EgyWtDensity,GAMMA);
            #else
        SPHP(i).Pressure =
            (SPHP(i).Entropy + SPHP(i).e.DtEntropy * dt_entr) * pow(SPHP(i).Density, GAMMA);
            #endif // DENSITY_INDEPENDENT_SPH

        #else
        SPHP(i).Pressure =
            GAMMA_MINUS1 * (SPHP(i).Entropy + SPHP(i).e.DtEntropy * dt_entr) * SPHP(i).Density;
        #endif // TRADITIONAL_SPH_FORMULATION

    #else
        #ifdef TRADITIONAL_SPH_FORMULATION
            #error tranditional sph incompatible with softereqs
        #endif
        #ifdef DENSITY_INDEPENDENT_SPH
            #error pressure entropy incompatible with softereqs
        /* use an intermediate EQS, between isothermal and the full multiphase model */
        if(SPHP(i).Density * All.cf.a3inv >= All.PhysDensThresh)
            SPHP(i).Pressure = All.FactorForSofterEQS *
                (SPHP(i).Entropy + SPHP(i).e.DtEntropy * dt_entr) * pow(SPHP(i).Density, GAMMA) +
                (1 -
                 All.FactorForSofterEQS) * All.cf.fac_egy * GAMMA_MINUS1 * SPHP(i).Density * All.InitGasU;
        else
            SPHP(i).Pressure =
                (SPHP(i).Entropy + SPHP(i).e.DtEntropy * dt_entr) * pow(SPHP(i).Density, GAMMA);
        #else
        /* use an intermediate EQS, between isothermal and the full multiphase model */
        if(SPHP(i).Density * All.cf.a3inv >= All.PhysDensThresh)
            SPHP(i).Pressure = All.FactorForSofterEQS *
                (SPHP(i).Entropy + SPHP(i).e.DtEntropy * dt_entr) * pow(SPHP(i).Density, GAMMA) +
                (1 -
                 All.FactorForSofterEQS) * All.cf.fac_egy * GAMMA_MINUS1 * SPHP(i).Density * All.InitGasU;
        else
            SPHP(i).Pressure =
                (SPHP(i).Entropy + SPHP(i).e.DtEntropy * dt_entr) * pow(SPHP(i).Density, GAMMA);
        #endif // DENSITY_INDEPENDENT_SPH
    #endif // SOFTEREQS
#else
        /* call tabulated eos with physical units */
        eos_calc_egiven_v(SPHP(i).Density * All.UnitDensity_in_cgs, SPHP(i).xnuc,
                SPHP(i).dxnuc, dt_entr * All.UnitTime_in_s, SPHP(i).Entropy,
                SPHP(i).e.DtEntropy, &SPHP(i).temp, &SPHP(i).Pressure, &SPHP(i).dpdr);
        SPHP(i).Pressure /= All.UnitPressure_in_cgs;
#endif

    }

#ifdef BLACK_HOLES
    if(P[i].Type == 5)
    {
        if(BHP(i).Density > 0)
        {
            BHP(i).EntOrPressure /= BHP(i).Density;
#ifdef BH_USE_GASVEL_IN_BONDI
            BHP(i).SurroundingGasVel[0] /= BHP(i).Density;
            BHP(i).SurroundingGasVel[1] /= BHP(i).Density;
            BHP(i).SurroundingGasVel[2] /= BHP(i).Density;
#endif
        }
    }
#endif
}

void density_check_neighbours (int i, MyFloat * Left, MyFloat * Right) {
    /* now check whether we had enough neighbours */

    double desnumngb = All.DesNumNgb;

#ifdef BLACK_HOLES
    if(P[i].Type == 5)
        desnumngb = All.DesNumNgb * All.BlackHoleNgbFactor;
#endif

    if(P[i].n.NumNgb < (desnumngb - All.MaxNumNgbDeviation) ||
            (P[i].n.NumNgb > (desnumngb + All.MaxNumNgbDeviation)
             && P[i].Hsml > (1.01 * All.MinGasHsml)))
    {
        /* need to redo this particle */
        if(P[i].DensityIterationDone) {
            /* should have been 0*/
            endrun(999993);
        }

        if(Left[i] > 0 && Right[i] > 0)
            if((Right[i] - Left[i]) < 1.0e-3 * Left[i])
            {
                /* this one should be ok */
                P[i].DensityIterationDone = 1;
                return;
            }

        if(P[i].n.NumNgb < (desnumngb - All.MaxNumNgbDeviation))
            Left[i] = DMAX(P[i].Hsml, Left[i]);
        else
        {
            if(Right[i] != 0)
            {
                if(P[i].Hsml < Right[i])
                    Right[i] = P[i].Hsml;
            }
            else
                Right[i] = P[i].Hsml;
        }

        if(Right[i] > 0 && Left[i] > 0)
            P[i].Hsml = pow(0.5 * (pow(Left[i], 3) + pow(Right[i], 3)), 1.0 / 3);
        else
        {
            if(Right[i] == 0 && Left[i] == 0)
                endrun(8188);	/* can't occur */

            if(Right[i] == 0 && Left[i] > 0)
            {
                if(P[i].Type == 0 && fabs(P[i].n.NumNgb - desnumngb) < 0.5 * desnumngb)
                {
                    double fac = 1 - (P[i].n.NumNgb -
                            desnumngb) / (NUMDIMS * P[i].n.NumNgb) *
                        SPHP(i).DhsmlDensityFactor;

                    if(fac < 1.26)
                        P[i].Hsml *= fac;
                    else
                        P[i].Hsml *= 1.26;
                }
                else
                    P[i].Hsml *= 1.26;
            }

            if(Right[i] > 0 && Left[i] == 0)
            {
                if(P[i].Type == 0 && fabs(P[i].n.NumNgb - desnumngb) < 0.5 * desnumngb)
                {
                    double fac = 1 - (P[i].n.NumNgb -
                            desnumngb) / (NUMDIMS * P[i].n.NumNgb) *
                        SPHP(i).DhsmlDensityFactor;

                    if(fac > 1 / 1.26)
                        P[i].Hsml *= fac;
                    else
                        P[i].Hsml /= 1.26;
                }
                else
                    P[i].Hsml /= 1.26;
            }
        }

        if(P[i].Hsml < All.MinGasHsml)
            P[i].Hsml = All.MinGasHsml;

#ifdef BLACK_HOLES
        if(P[i].Type == 5)
            if(Left[i] > All.BlackHoleMaxAccretionRadius)
            {
                /* this will stop the search for a new BH smoothing length in the next iteration */
                P[i].Hsml = Left[i] = Right[i] = All.BlackHoleMaxAccretionRadius;
            }
#endif

    }
    else {
        P[i].DensityIterationDone = 1;
    }

}

