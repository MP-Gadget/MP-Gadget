#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "densitykernel.h"
#include "proto.h"
#ifdef COSMIC_RAYS
#include "cosmic_rays.h"
#endif
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
#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS) || defined(TRACEDIVB)
    MyFloat BPred[3];
#endif
#ifdef VECT_POTENTIAL
    MyFloat APred[3];
    MyFloat rrho;
#endif
#ifdef EULERPOTENTIALS
    MyFloat EulerA, EulerB;
#endif
#if defined(MAGNETICSEED)
    MyFloat MagSeed;
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
#ifndef NAVIERSTOKES
    MyDouble Div, Rot[3];
#else
    MyFloat DV[3][3];
#endif

#ifdef JD_VTURB
    MyFloat Vturb;
    MyFloat Vbulk[3];
    int TrueNGB;
#endif

#ifdef MAGNETIC
#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS)
    MyFloat RotB[3];
#endif
#ifdef TRACEDIVB
    MyFloat divB;
#endif
#ifdef VECT_PRO_CLEAN
    MyFloat BPredVec[3];
#endif
#endif

#ifdef RADTRANSFER_FLUXLIMITER
    MyFloat Grad_ngamma[3][N_BINS];
#endif

#ifdef BLACK_HOLES
    MyDouble SmoothedEntOrPressure;
    MyDouble FeedbackWeightSum;
    MyDouble GasVel[3];
#endif
#ifdef CONDUCTION_SATURATION
    MyFloat GradEntr[3];
#endif

#ifdef HYDRO_COST_FACTOR
    int Ninteractions;
#endif

#ifdef VOLUME_CORRECTION
    MyFloat DensityStd;
#endif

#ifdef EULERPOTENTIALS
    MyFloat dEulerA[3], dEulerB[3];
#endif
#ifdef VECT_POTENTIAL
    MyFloat BPred[3];
    MyFloat dA[6];
#endif
#ifdef SPH_GRAD_RHO
    MyFloat GradRho[3];
#endif
};

static int density_isactive(int n);
static int density_evaluate(int target, int mode, struct densdata_in * I, struct densdata_out * O, LocalEvaluator * lv, int * ngblist);
static void * density_alloc_ngblist();
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
double a3inv, afac;

void density(void)
{
    MyFloat *Left, *Right;

    Evaluator ev = {0};
   
    ev.ev_evaluate = (ev_evaluate_func) density_evaluate;
    ev.ev_isactive = density_isactive;
    ev.ev_alloc = density_alloc_ngblist;
    ev.ev_copy = (ev_copy_func) density_copy;
    ev.ev_reduce = (ev_reduce_func) density_reduce;
    ev.UseNodeList = 1;
    ev.ev_datain_elsize = sizeof(struct densdata_in);
    ev.ev_dataout_elsize = sizeof(struct densdata_out);

    int i, j, k, npleft, iter = 0;

    int64_t ntot = 0;

    double timeall = 0;
    double timecomp, timecomp3 = 0, timecomm, timewait;

    double dt_entr, tstart, tend;

    int64_t n_exported = 0;

#ifdef COSMIC_RAYS
    int CRpop;
#endif

    if(All.ComovingIntegrationOn)
    {
        a3inv = 1 / (All.Time * All.Time * All.Time);
        afac = pow(All.Time, 3 * GAMMA_MINUS1);
    }
    else
        a3inv = afac = 1;

#if defined(EULERPOTENTIALS) || defined(VECT_PRO_CLEAN) || defined(TRACEDIVB) || defined(VECT_POTENTIAL)
    double efak;

    if(All.ComovingIntegrationOn)
        efak = 1. / All.Time / All.HubbleParam;
    else
        efak = 1;
#endif

#if defined(MAGNETICSEED)
    int count_seed = 0, count_seed_tot=0;
    double mu0 = 1;
#ifndef MU0_UNITY
    mu0 *= (4 * M_PI);
    mu0 /= All.UnitTime_in_s * All.UnitTime_in_s *
        All.UnitLength_in_cm / All.UnitMass_in_g;
    if(All.ComovingIntegrationOn)
        mu0 /= (All.HubbleParam * All.HubbleParam);
#endif
#endif
    walltime_measure("/Misc");

    Ngblist = (int *) mymalloc("Ngblist", All.NumThreads * NumPart * sizeof(int));

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
    queue = evaluate_get_queue(&ev, &Nactive);
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

        evaluate_begin(&ev);

        do
        {

            evaluate_primary(&ev); /* do local particles and prepare export list */

            n_exported += ev.Nexport;

            /* exchange particle data */

            evaluate_get_remote(&ev, TAG_DENS_A);

            report_memory_usage(&HighMark_sphdensity, "SPH_DENSITY");

            fflush(stdout);
            /* now do the particles that were sent to us */

            evaluate_secondary(&ev);

            /* import the result to local particles */
            evaluate_reduce_result(&ev, TAG_DENS_B);
        }
        while(evaluate_ndone(&ev) < NTask);

        evaluate_finish(&ev);

        /* do final operations on results */
        tstart = second();

        queue = evaluate_get_queue(&ev, &Nactive);
        
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
                    ("i=%d task=%d ID=%llu Hsml=%g Left=%g Right=%g Ngbs=%g Right-Left=%g\n   pos=(%g|%g|%g)\n",
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
    myfree(Ngblist);


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
        if (All.ComovingIntegrationOn) {
            rds = All.BlackHoleFeedbackRadiusMaxPhys / All.Time;
        } else {
            rds = All.BlackHoleFeedbackRadiusMaxPhys;
        }

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

#ifdef EULERPOTENTIALS
    I->EulerA = SPHP(place).EulerA;
    I->EulerB = SPHP(place).EulerB;
#endif
#ifdef VECT_POTENTIAL
    I->APred[0] = SPHP(place).APred[0];
    I->APred[1] = SPHP(place).APred[1];
    I->APred[2] = SPHP(place).APred[2];
    I->rrho = SPHP(place).d.Density;
#endif
#if defined(MAGNETICSEED)
    I->MagSeed = SPHP(place).MagSeed;
#endif


#ifdef WINDS
    I->DelayTime = SPHP(place).DelayTime;
#endif

#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS) || defined(TRACEDIVB)
    I->BPred[0] = SPHP(place).BPred[0];
    I->BPred[1] = SPHP(place).BPred[1];
    I->BPred[2] = SPHP(place).BPred[2];
#endif
}

static void density_reduce(int place, struct densdata_out * remote, int mode) {
    EV_REDUCE(P[place].n.dNumNgb, remote->Ngb);

    if(remote->ID != P[place].ID) {
        BREAKPOINT; 
    }
#ifdef HYDRO_COST_FACTOR
    /* these will be added */
    if(All.ComovingIntegrationOn)
        P[place].GravCost += HYDRO_COST_FACTOR * All.Time * remote->Ninteractions;
    else
        P[place].GravCost += HYDRO_COST_FACTOR * remote->Ninteractions;
#endif

    if(P[place].Type == 0)
    {
        EV_REDUCE(SPHP(place).d.dDensity, remote->Rho);
        EV_REDUCE(SPHP(place).h.dDhsmlDensityFactor, remote->DhsmlDensity);
#ifdef DENSITY_INDEPENDENT_SPH
        EV_REDUCE(SPHP(place).EgyWtDensity, remote->EgyRho);
        EV_REDUCE(SPHP(place).DhsmlEgyDensityFactor, remote->DhsmlEgyDensity);
#endif

#ifndef NAVIERSTOKES
        EV_REDUCE(SPHP(place).v.dDivVel, remote->Div);
        EV_REDUCE(SPHP(place).r.dRot[0], remote->Rot[0]);
        EV_REDUCE(SPHP(place).r.dRot[1], remote->Rot[1]);
        EV_REDUCE(SPHP(place).r.dRot[2], remote->Rot[2]);
#else
        for(k = 0; k < 3; k++)
        {
            EV_REDUCE(SPHP(place).u.DV[k][0], remote->DV[k][0]);
            EV_REDUCE(SPHP(place).u.DV[k][1], remote->DV[k][1]);
            EV_REDUCE(SPHP(place).u.DV[k][2], remote->DV[k][2]);
        }
#endif

#ifdef VOLUME_CORRECTION
        EV_REDUCE(SPHP(place).DensityStd, remote->DensityStd);
#endif

#ifdef CONDUCTION_SATURATION
        EV_REDUCE(SPHP(place).GradEntr[0], remote->GradEntr[0]);
        EV_REDUCE(SPHP(place).GradEntr[1], remote->GradEntr[1]);
        EV_REDUCE(SPHP(place).GradEntr[2], remote->GradEntr[2]);
#endif

#ifdef SPH_GRAD_RHO
        EV_REDUCE(SPHP(place).GradRho[0], remote->GradRho[0]);
        EV_REDUCE(SPHP(place).GradRho[1], remote->GradRho[1]);
        EV_REDUCE(SPHP(place).GradRho[2], remote->GradRho[2]);
#endif

#ifdef RADTRANSFER_FLUXLIMITER
        for(k = 0; k< N_BINS; k++)
        {
            EV_REDUCE(SPHP(place).Grad_ngamma[0][k], remote->Grad_ngamma[0][k]);
            EV_REDUCE(SPHP(place).Grad_ngamma[1][k], remote->Grad_ngamma[1][k]);
            EV_REDUCE(SPHP(place).Grad_ngamma[2][k], remote->Grad_ngamma[2][k]);
        }
#endif


#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS)
        EV_REDUCE(SPHP(place).RotB[0], remote->RotB[0]);
        EV_REDUCE(SPHP(place).RotB[1], remote->RotB[1]);
        EV_REDUCE(SPHP(place).RotB[2], remote->RotB[2]);
#endif

#ifdef TRACEDIVB
        EV_REDUCE(SPHP(place).divB, remote->divB);
#endif

#ifdef JD_VTURB
        EV_REDUCE(SPHP(place).Vturb, remote->Vturb);
        EV_REDUCE(SPHP(place).Vbulk[0], remote->Vbulk[0]);
        EV_REDUCE(SPHP(place).Vbulk[1], remote->Vbulk[1]);
        EV_REDUCE(SPHP(place).Vbulk[2], remote->Vbulk[2]);
        EV_REDUCE(SPHP(place).TrueNGB, remote->TrueNGB);
#endif

#ifdef VECT_PRO_CLEAN
        EV_REDUCE(SPHP(place).BPredVec[0], remote->BPredVec[0]);
        EV_REDUCE(SPHP(place).BPredVec[1], remote->BPredVec[1]);
        EV_REDUCE(SPHP(place).BPredVec[2], remote->BPredVec[2]);
#endif
#ifdef EULERPOTENTIALS
        EV_REDUCE(SPHP(place).dEulerA[0], remote->dEulerA[0]);
        EV_REDUCE(SPHP(place).dEulerA[1], remote->dEulerA[1]);
        EV_REDUCE(SPHP(place).dEulerA[2], remote->dEulerA[2]);
        EV_REDUCE(SPHP(place).dEulerB[0], remote->dEulerB[0]);
        EV_REDUCE(SPHP(place).dEulerB[1], remote->dEulerB[1]);
        EV_REDUCE(SPHP(place).dEulerB[2], remote->dEulerB[2]);
#endif
#ifdef VECT_POTENTIAL
        EV_REDUCE(SPHP(place).dA[5], remote->dA[5]);
        EV_REDUCE(SPHP(place).dA[4], remote->dA[4]);
        EV_REDUCE(SPHP(place).dA[3], remote->dA[3]);
        EV_REDUCE(SPHP(place).dA[2], remote->dA[2]);
        EV_REDUCE(SPHP(place).dA[1], remote->dA[1]);
        EV_REDUCE(SPHP(place).dA[0], remote->dA[0]);
#endif
    }

#if (defined(RADTRANSFER) && defined(EDDINGTON_TENSOR_STARS)) || defined(SNIA_HEATING)
    if(P[place].Type == 4)
        EV_REDUCE(P[place].DensAroundStar, remote->Rho);
#endif

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
        LocalEvaluator * lv, int * ngblist)
{
    int n;

    int startnode, numngb, numngb_inbox, listindex = 0;

    double h;
    double hsearch;

    int ninteractions = 0;
    int nnodesinlist = 0;

    int k;

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

#if defined(MAGNETICSEED)
    double mu0_1 = 1;
#ifndef MU0_UNITY
    mu0_1 *= (4 * M_PI);
    mu0_1 /= All.UnitTime_in_s * All.UnitTime_in_s *
        All.UnitLength_in_cm / All.UnitMass_in_g;
    if(All.ComovingIntegrationOn)
        mu0_1 /= (All.HubbleParam * All.HubbleParam);
#endif
#endif

    numngb = 0;

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb_inbox =
                ngb_treefind_threads(I->Pos, hsearch, target, &startnode, 
                        mode, lv, ngblist, NGB_TREEFIND_ASYMMETRIC, 1); /* gas only 1<<0 */

            if(numngb_inbox < 0)
                return numngb_inbox;

            for(n = 0; n < numngb_inbox; n++)
            {
                ninteractions++;
                int j = ngblist[n];
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

#ifdef PERIODIC			/*  now find the closest image in the given box size  */
                dx = NEAREST_X(dx);
                dy = NEAREST_Y(dy);
                dz = NEAREST_Z(dz);
#endif
                double r2 = dx * dx + dy * dy + dz * dz;

                double r = sqrt(r2);

                if(r2 < kernel.HH)
                {

                    double u = r * kernel.Hinv;
                    double wk = density_kernel_wk(&kernel, u);
                    double dwk = density_kernel_dwk(&kernel, u);

                    double mass_j = P[j].Mass;

                    numngb++;
#ifdef JD_VTURB
                    O->Vturb += (SPHP(j).VelPred[0] - I->Vel[0]) * (SPHP(j).VelPred[0] - I->Vel[0]) +
                        (SPHP(j).VelPred[1] - I->Vel[1]) * (SPHP(j).VelPred[1] - I->Vel[1]) +
                        (SPHP(j).VelPred[2] - I->Vel[2]) * (SPHP(j).VelPred[2] - I->Vel[2]);
                    O->Vbulk[0] += SPHP(j).VelPred[0];
                    O->Vbulk[1] += SPHP(j).VelPred[1];
                    O->Vbulk[2] += SPHP(j).VelPred[2];
                    O->TrueNGB++;
#endif

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

#ifdef CONDUCTION_SATURATION
                    if(r > 0)
                    {
                        O->GradEntr[0] += mass_j * dwk * dx / r * SPHP(j).Entropy;
                        O->GradEntr[1] += mass_j * dwk * dy / r * SPHP(j).Entropy;
                        O->GradEntr[2] += mass_j * dwk * dz / r * SPHP(j).Entropy;
                    }
#endif

#ifdef SPH_GRAD_RHO
                    if(r > 0)
                    {
                        O->GradRho[0] += mass_j * dwk * dx / r;
                        O->GradRho[1] += mass_j * dwk * dy / r;
                        O->GradRho[2] += mass_j * dwk * dz / r;
                    }
#endif


#ifdef RADTRANSFER_FLUXLIMITER
                    if(r > 0)
                        for(k = 0; k < N_BINS; k++)
                        {
                            O->Grad_ngamma[0][k] += mass_j * dwk * dx / r * SPHP(j).n_gamma[k];
                            O->Grad_ngamma[1][k] += mass_j * dwk * dy / r * SPHP(j).n_gamma[k];
                            O->Grad_ngamma[2][k] += mass_j * dwk * dz / r * SPHP(j).n_gamma[k];
                        }
#endif


                    if(r > 0)
                    {
                        double fac = mass_j * dwk / r;

                        double dvx = I->Vel[0] - SPHP(j).VelPred[0];
                        double dvy = I->Vel[1] - SPHP(j).VelPred[1];
                        double dvz = I->Vel[2] - SPHP(j).VelPred[2];

#ifndef NAVIERSTOKES
                        O->Div += (-fac * (dx * dvx + dy * dvy + dz * dvz));

                        O->Rot[0] += (fac * (dz * dvy - dy * dvz));
                        O->Rot[1] += (fac * (dx * dvz - dz * dvx));
                        O->Rot[2] += (fac * (dy * dvx - dx * dvy));
#else
                        O->DV[0][0] -= fac * dx * dvx;
                        O->DV[0][1] -= fac * dx * dvy;
                        O->DV[0][2] -= fac * dx * dvz;
                        O->DV[1][0] -= fac * dy * dvx;
                        O->DV[1][1] -= fac * dy * dvy;
                        O->DV[1][2] -= fac * dy * dvz;
                        O->DV[2][0] -= fac * dz * dvx;
                        O->DV[2][1] -= fac * dz * dvy;
                        O->DV[2][2] -= fac * dz * dvz;
#endif
#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS) || defined(TRACEDIVB)
                        double dbx = I->BPred[0] - SPHP(j).BPred[0];
                        double dby = I->BPred[1] - SPHP(j).BPred[1];
                        double dbz = I->BPred[2] - SPHP(j).BPred[2];
#endif
#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS)
                        O->RotB[0] += (fac * (dz * dby - dy * dbz));
                        O->RotB[1] += (fac * (dx * dbz - dz * dbx));
                        O->RotB[2] += (fac * (dy * dbx - dx * dby));
#endif
#ifdef VECT_POTENTIAL
                        O->dA[0] += fac * (I->APred[0] - SPHP(j).APred[0]) * dy;	//dAx/dy
                        O->dA[1] += fac * (I->APred[0] - SPHP(j).APred[0]) * dz;	//dAx/dz
                        O->dA[2] += fac * (I->APred[1] - SPHP(j).APred[1]) * dx;	//dAy/dx
                        O->dA[3] += fac * (I->APred[1] - SPHP(j).APred[1]) * dz;	//dAy/dz
                        O->dA[4] += fac * (I->APred[2] - SPHP(j).APred[2]) * dx;	//dAz/dx
                        O->dA[5] += fac * (I->APred[2] - SPHP(j).APred[2]) * dy;	//dAz/dy
#endif
#ifdef TRACEDIVB
                        O->divB += (-fac * (dbx * dx + dby * dy + dbz * dz));
#endif
#ifdef MAGNETICSEED
                        double spin_0=sqrt(I->MagSeed*mu0_1*2.);//energy to B field
                        spin_0=3./2.*spin_0/(sqrt(I->Vel[0]*I->Vel[0]+I->Vel[1]*I->Vel[1]+I->Vel[2]*I->Vel[2]));//*SPHP(j).d.Density;

                        if(I->MagSeed)
                        {
#error This needs to be prortected by a lock
                            SPHP(j).BPred[0] += 1./(4.* M_PI * (pow(r,3.))) *
                                (3. *(dx*I->Vel[0] + dy*I->Vel[1] + dz*I->Vel[2]) * spin_0 / r  * dx / r - spin_0 * I->Vel[0]);
                            SPHP(j).BPred[1] += 1./(4.* M_PI * (pow(r,3.))) *
                                (3. *(dx*I->Vel[0] + dy*I->Vel[1] + dz*I->Vel[2]) * spin_0 / r  * dy / r - spin_0 * I->Vel[1]);
                            SPHP(j).BPred[2] += 1./(4.* M_PI * (pow(r,3.))) *
                                (3. *(dx*I->Vel[0] + dy*I->Vel[1] + dz*I->Vel[2]) * spin_0 / r  * dz / r - spin_0 * I->Vel[2]);
                        };
#endif
#ifdef VECT_PRO_CLEAN
                        O->BPredVec[0] +=
                            (fac * r2 * (SPHP(j).RotB[1] * dz - SPHP(j).RotB[2] * dy) / SPHP(j).d.Density);
                        O->BPredVec[1] +=
                            (fac * r2 * (SPHP(j).RotB[2] * dx - SPHP(j).RotB[0] * dz) / SPHP(j).d.Density);
                        O->BPredVec[2] +=
                            (fac * r2 * (SPHP(j).RotB[0] * dy - SPHP(j).RotB[1] * dx) / SPHP(j).d.Density);
#endif
#ifdef EULERPOTENTIALS
                        dea = I->EulerA - SPHP(j).EulerA;
                        deb = I->EulerB - SPHP(j).EulerB;
#ifdef EULER_VORTEX
                        deb = NEAREST_Z(deb);
#endif
                        O->dEulerA[0] -= fac * dx * dea;
                        O->dEulerA[1] -= fac * dy * dea;
                        O->dEulerA[2] -= fac * dz * dea;
                        O->dEulerB[0] -= fac * dx * deb;
                        O->dEulerB[1] -= fac * dy * deb;
                        O->dEulerB[2] -= fac * dz * deb;
#endif
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
                                    * pow(SPHP(j).EOMDensity * a3inv,
                                        GAMMA_MINUS1)),
                                SPHP(j).d.Density * a3inv, &uvbg, &ne, &nh0, &nHeII);
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

static void * density_alloc_ngblist() {
    int threadid = omp_get_thread_num();
    return Ngblist + threadid * NumPart;
}

static int density_isactive(int n)
{
    if(P[n].DensityIterationDone) return 0;

    if(P[n].TimeBin < 0) {
        printf("TimeBin negative!\n use DensityIterationDone flag");
        endrun(99999);
    }
#if (defined(RADTRANSFER) && defined(EDDINGTON_TENSOR_STARS))|| defined(SNIA_HEATING)
    if(P[n].Type == 4)
        return 1;
#endif

#ifdef BLACK_HOLES
    if(P[n].Type == 5)
        return 1;
#endif

    if(P[n].Type == 0)
        return 1;

    return 0;
}

static void density_post_process(int i) {
    int dt_step, dt_entr;
    if(P[i].Type == 0)
    {
        if(SPHP(i).d.Density > 0)
        {
#ifdef VOLUME_CORRECTION
            SPHP(i).DensityOld = SPHP(i).DensityStd;
#endif
            SPHP(i).h.DhsmlDensityFactor *= P[i].Hsml / (NUMDIMS * SPHP(i).d.Density);
            if(SPHP(i).h.DhsmlDensityFactor > -0.9)	/* note: this would be -1 if only a single particle at zero lag is found */
                SPHP(i).h.DhsmlDensityFactor = 1 / (1 + SPHP(i).h.DhsmlDensityFactor);
            else
                SPHP(i).h.DhsmlDensityFactor = 1;

#ifdef DENSITY_INDEPENDENT_SPH
            if((SPHP(i).EntVarPred>0)&&(SPHP(i).EgyWtDensity>0))
            {
                SPHP(i).DhsmlEgyDensityFactor *= P[i].Hsml/ (NUMDIMS * SPHP(i).EgyWtDensity);
                SPHP(i).DhsmlEgyDensityFactor *= -SPHP(i).h.DhsmlDensityFactor;
                SPHP(i).EgyWtDensity /= SPHP(i).EntVarPred;
            } else {
                SPHP(i).DhsmlEgyDensityFactor=0; 
                SPHP(i).EntVarPred=0; 
                SPHP(i).EgyWtDensity=0;
            }
#endif

#ifndef NAVIERSTOKES
            SPHP(i).r.CurlVel = sqrt(SPHP(i).r.Rot[0] * SPHP(i).r.Rot[0] +
                    SPHP(i).r.Rot[1] * SPHP(i).r.Rot[1] +
                    SPHP(i).r.Rot[2] * SPHP(i).r.Rot[2]) / SPHP(i).d.Density;

            SPHP(i).v.DivVel /= SPHP(i).d.Density;
#else
            for(k = 0; k < 3; k++)
            {
                O->DV[k][0] = SPHP(i).u.DV[k][0] / SPHP(i).d.Density;
                O->DV[k][1] = SPHP(i).u.DV[k][1] / SPHP(i).d.Density;
                O->DV[k][2] = SPHP(i).u.DV[k][2] / SPHP(i).d.Density;
            }
            SPHP(i).u.s.DivVel = O->DV[0][0] + O->DV[1][1] + O->DV[2][2];

            SPHP(i).u.s.StressDiag[0] = 2 * O->DV[0][0] - 2.0 / 3 * SPHP(i).u.s.DivVel;
            SPHP(i).u.s.StressDiag[1] = 2 * O->DV[1][1] - 2.0 / 3 * SPHP(i).u.s.DivVel;
            SPHP(i).u.s.StressDiag[2] = 2 * O->DV[2][2] - 2.0 / 3 * SPHP(i).u.s.DivVel;

            SPHP(i).u.s.StressOffDiag[0] = O->DV[0][1] + O->DV[1][0];	/* xy */
            SPHP(i).u.s.StressOffDiag[1] = O->DV[0][2] + O->DV[2][0];	/* xz */
            SPHP(i).u.s.StressOffDiag[2] = O->DV[1][2] + O->DV[2][1];	/* yz */

#ifdef NAVIERSTOKES_BULK
            SPHP(i).u.s.StressBulk = All.NavierStokes_BulkViscosity * SPHP(i).u.s.DivVel;
#endif
            double rotx = O->DV[1][2] - O->DV[2][1];
            double roty = O->DV[2][0] - O->DV[0][2];
            double rotz = O->DV[0][1] - O->DV[1][0];
            SPHP(i).u.s.CurlVel = sqrt(rotx * rotx + roty * roty + rotz * rotz);
#endif


#ifdef CONDUCTION_SATURATION
            SPHP(i).GradEntr[0] /= SPHP(i).d.Density;
            SPHP(i).GradEntr[1] /= SPHP(i).d.Density;
            SPHP(i).GradEntr[2] /= SPHP(i).d.Density;
#endif


#ifdef RADTRANSFER_FLUXLIMITER
            for(k = 0; k< N_BINS; k++)
            {
                SPHP(i).Grad_ngamma[0][k] /= SPHP(i).d.Density;
                SPHP(i).Grad_ngamma[1][k] /= SPHP(i).d.Density;
                SPHP(i).Grad_ngamma[2][k] /= SPHP(i).d.Density;
            }
#endif


#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS)
            SPHP(i).RotB[0] /= SPHP(i).d.Density;
            SPHP(i).RotB[1] /= SPHP(i).d.Density;
            SPHP(i).RotB[2] /= SPHP(i).d.Density;
#endif

#ifdef JD_VTURB
            SPHP(i).Vturb = sqrt(SPHP(i).Vturb / SPHP(i).TrueNGB);
            SPHP(i).Vbulk[0] /= SPHP(i).TrueNGB;
            SPHP(i).Vbulk[1] /= SPHP(i).TrueNGB;
            SPHP(i).Vbulk[2] /= SPHP(i).TrueNGB;
#endif

#ifdef TRACEDIVB
            SPHP(i).divB /= SPHP(i).d.Density;
#endif

#ifdef VECT_PRO_CLEAN
            SPHP(i).BPred[0] += efak * SPHP(i).BPredVec[0];
            SPHP(i).BPred[1] += efak * SPHP(i).BPredVec[1];
            SPHP(i).BPred[2] += efak * SPHP(i).BPredVec[2];
#endif
#ifdef EULERPOTENTIALS
            SPHP(i).dEulerA[0] *= efak / SPHP(i).d.Density;
            SPHP(i).dEulerA[1] *= efak / SPHP(i).d.Density;
            SPHP(i).dEulerA[2] *= efak / SPHP(i).d.Density;
            SPHP(i).dEulerB[0] *= efak / SPHP(i).d.Density;
            SPHP(i).dEulerB[1] *= efak / SPHP(i).d.Density;
            SPHP(i).dEulerB[2] *= efak / SPHP(i).d.Density;

            SPHP(i).BPred[0] =
                SPHP(i).dEulerA[1] * SPHP(i).dEulerB[2] - SPHP(i).dEulerA[2] * SPHP(i).dEulerB[1];
            SPHP(i).BPred[1] =
                SPHP(i).dEulerA[2] * SPHP(i).dEulerB[0] - SPHP(i).dEulerA[0] * SPHP(i).dEulerB[2];
            SPHP(i).BPred[2] =
                SPHP(i).dEulerA[0] * SPHP(i).dEulerB[1] - SPHP(i).dEulerA[1] * SPHP(i).dEulerB[0];
#endif
#ifdef	VECT_POTENTIAL
            SPHP(i).BPred[0] = (SPHP(i).dA[5] - SPHP(i).dA[3]) / SPHP(i).d.Density * efak;
            SPHP(i).BPred[1] = (SPHP(i).dA[1] - SPHP(i).dA[4]) / SPHP(i).d.Density * efak;
            SPHP(i).BPred[2] = (SPHP(i).dA[2] - SPHP(i).dA[0]) / SPHP(i).d.Density * efak;

#endif
#ifdef MAGNETICSEED
            if(SPHP(i).MagSeed!=0. )
            {
                SPHP(i).MagSeed=sqrt(2.0*mu0*SPHP(i).MagSeed)/ //// *SPHP(i).d.Density /
                    sqrt(
                            SPHP(i).VelPred[2]*SPHP(i).VelPred[2]+
                            SPHP(i).VelPred[1]*SPHP(i).VelPred[1]+
                            SPHP(i).VelPred[0]*SPHP(i).VelPred[0]);
                SPHP(i).BPred[0]+= SPHP(i).VelPred[0]*SPHP(i).MagSeed;
                SPHP(i).BPred[1]+= SPHP(i).VelPred[1]*SPHP(i).MagSeed;
                SPHP(i).BPred[2]+= SPHP(i).VelPred[2]*SPHP(i).MagSeed;

                if(ThisTask == 0 && count_seed == 1) printf("MAG  SEED %i and %e\n",count_seed, SPHP(i).MagSeed);
                if(ThisTask == 0 && count_seed == 1) printf("ONLY SEED %6e %6e %6e\n",SPHP(i).BPred[2],SPHP(i).BPred[1],SPHP(i).BPred[0]);
                fflush(stdout);
                SPHP(i).MagSeed=0.;
                count_seed++;
            }
#endif
        }

#ifndef WAKEUP
        dt_step = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0);
#else
        dt_step = P[i].dt_step;
#endif
        dt_entr = (All.Ti_Current - (P[i].Ti_begstep + dt_step / 2)) * All.Timebase_interval;

#ifndef EOS_DEGENERATE
#ifndef MHM
#ifndef SOFTEREQS
#ifndef TRADITIONAL_SPH_FORMULATION
#ifdef DENSITY_INDEPENDENT_SPH
        SPHP(i).Pressure = pow(SPHP(i).EntVarPred*SPHP(i).EgyWtDensity,GAMMA);
#else
        SPHP(i).Pressure =
            (SPHP(i).Entropy + SPHP(i).e.DtEntropy * dt_entr) * pow(SPHP(i).d.Density, GAMMA);
#endif // DENSITY_INDEPENDENT_SPH

#else
        SPHP(i).Pressure =
            GAMMA_MINUS1 * (SPHP(i).Entropy + SPHP(i).e.DtEntropy * dt_entr) * SPHP(i).d.Density;
#endif // TRADITIONAL_SPH_FORMULATION

#else
#ifdef TRADITIONAL_SPH_FORMULATION
#error tranditional sph incompatible with softereqs
#endif
#ifdef DENSITY_INDEPENDENT_SPH
#error pressure entropy incompatible with softereqs
        /* use an intermediate EQS, between isothermal and the full multiphase model */
        if(SPHP(i).d.Density * a3inv >= All.PhysDensThresh)
            SPHP(i).Pressure = All.FactorForSofterEQS *
                (SPHP(i).Entropy + SPHP(i).e.DtEntropy * dt_entr) * pow(SPHP(i).d.Density, GAMMA) +
                (1 -
                 All.FactorForSofterEQS) * afac * GAMMA_MINUS1 * SPHP(i).d.Density * All.InitGasU;
        else
            SPHP(i).Pressure =
                (SPHP(i).Entropy + SPHP(i).e.DtEntropy * dt_entr) * pow(SPHP(i).d.Density, GAMMA);
#else
        /* use an intermediate EQS, between isothermal and the full multiphase model */
        if(SPHP(i).d.Density * a3inv >= All.PhysDensThresh)
            SPHP(i).Pressure = All.FactorForSofterEQS *
                (SPHP(i).Entropy + SPHP(i).e.DtEntropy * dt_entr) * pow(SPHP(i).d.Density, GAMMA) +
                (1 -
                 All.FactorForSofterEQS) * afac * GAMMA_MINUS1 * SPHP(i).d.Density * All.InitGasU;
        else
            SPHP(i).Pressure =
                (SPHP(i).Entropy + SPHP(i).e.DtEntropy * dt_entr) * pow(SPHP(i).d.Density, GAMMA);
#endif // DENSITY_INDEPENDENT_SPH
#endif // SOFTEREQS
#else
        /* Here we use an isothermal equation of state */
        SPHP(i).Pressure = afac * GAMMA_MINUS1 * SPHP(i).d.Density * All.InitGasU;
        SPHP(i).Entropy = SPHP(i).Pressure / pow(SPHP(i).d.Density, GAMMA);
#endif
#else
        /* call tabulated eos with physical units */
        eos_calc_egiven_v(SPHP(i).d.Density * All.UnitDensity_in_cgs, SPHP(i).xnuc,
                SPHP(i).dxnuc, dt_entr * All.UnitTime_in_s, SPHP(i).Entropy,
                SPHP(i).e.DtEntropy, &SPHP(i).temp, &SPHP(i).Pressure, &SPHP(i).dpdr);
        SPHP(i).Pressure /= All.UnitPressure_in_cgs;
#endif

#ifdef COSMIC_RAYS
        for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
        {
            CR_Particle_Update(SphP + i, CRpop);
#ifndef CR_NOPRESSURE
            SPHP(i).Pressure += CR_Comoving_Pressure(SphP + i, CRpop);
#endif
        }
#endif

#ifdef BP_REAL_CRs
        bp_cr_update(SPHP(i));
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
                        SPHP(i).h.DhsmlDensityFactor;

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
                        SPHP(i).h.DhsmlDensityFactor;

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

#ifdef NAVIERSTOKES
double get_shear_viscosity(int i)
{
    return All.NavierStokes_ShearViscosity;
}
#endif

