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

static int density_isactive(int n);
static int density_evaluate(int target, int mode, Exporter * exporter, int * ngblist);
static void * density_alloc_ngblist();

/*! Structure for communication during the density computation. Holds data that is sent to other processors.
*/
static struct densdata_in
{
    int NodeList[NODELISTLENGTH];
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
} *DensDataGet;


static struct densdata_out
{
#ifdef DENSITY_INDEPENDENT_SPH
    MyFloat EgyRho;
    MyFloat DhsmlEgyDensity;
#endif
    MyLongDouble Rho;
    MyLongDouble DhsmlDensity;
    MyLongDouble Ngb;
#ifndef NAVIERSTOKES
    MyLongDouble Div, Rot[3];
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
    MyLongDouble SmoothedEntOrPressure;
    MyLongDouble FeedbackWeightSum;
    MyLongDouble GasVel[3];
    short int BH_TimeBinLimit;
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
    MyFloat bpred[3];
    MyFloat da[6];
#endif

}
*DensDataResult;

static void density_reduce(int place, struct densdata_out * remote, int mode);
static void density_copy(int place, struct densdata_in * input);

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

    int i, j, k, ndone, npleft, dt_step, iter = 0;

    int64_t ntot;

    double fac;

    double timeall = 0;
    double timecomp, timecomp3 = 0, timecomm, timewait;

    double dt_entr, tstart, tend, t0, t1;

    double desnumngb;

    int64_t n_exported = 0;

#ifdef COSMIC_RAYS
    int CRpop;
#endif

#ifdef NAVIERSTOKES
    double dvel[3][3];

    double rotx, roty, rotz;
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
    CPU_Step[CPU_DENSMISC] += measure_time();

    Ngblist = (int *) mymalloc("Ngblist", All.NumThreads * NumPart * sizeof(int));

    Left = (MyFloat *) mymalloc("Left", NumPart * sizeof(MyFloat));
    Right = (MyFloat *) mymalloc("Right", NumPart * sizeof(MyFloat));

    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        if(!density_isactive(i)) continue;
        Left[i] = Right[i] = 0;

#ifdef BLACK_HOLES
        P[i].SwallowID = 0;
#endif
#if defined(BLACK_HOLES) && defined(FLTROUNDOFFREDUCTION)
        if(P[i].Type == 0)
            SPHP(i).i.dInjected_BH_Energy = SPHP(i).i.Injected_BH_Energy;
#endif
    }

    /* allocate buffers to arrange communication */

    t0 = second();

    desnumngb = All.DesNumNgb;

    /* we will repeat the whole thing for those particles where we didn't find enough neighbours */
    do
    {

        evaluate_begin(&ev);

        do
        {

            evaluate_primary(&ev); /* do local particles and prepare export list */

            n_exported += ev.Nexport;

            /* exchange particle data */

            DensDataGet = (struct densdata_in * ) evaluate_get_remote(&ev, TAG_DENS_A);

            DensDataResult =
                (struct densdata_out *) mymalloc("DensDataResult", ev.Nimport * sizeof(struct densdata_out));

            report_memory_usage(&HighMark_sphdensity, "SPH_DENSITY");

            /* now do the particles that were sent to us */

            evaluate_secondary(&ev);

            /* import the result to local particles */
            evaluate_reduce_result(&ev, DensDataResult, TAG_DENS_B);

            myfree(DensDataResult);
            myfree(DensDataGet);
        }
        while(evaluate_ndone(&ev) < NTask);

        evaluate_finish(&ev);

#ifdef FLTROUNDOFFREDUCTION
        for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
            if(density_isactive(i))
            {
                P[i].n.NumNgb = FLT(P[i].n.dNumNgb);

                if(P[i].Type == 0)
                {
                    SPHP(i).d.Density = FLT(SPHP(i).d.dDensity);
                    SPHP(i).h.DhsmlDensityFactor = FLT(SPHP(i).h.dDhsmlDensityFactor);
                    SPHP(i).v.DivVel = FLT(SPHP(i).v.dDivVel);
                    for(j = 0; j < 3; j++)
                        SPHP(i).r.Rot[j] = FLT(SPHP(i).r.dRot[j]);
                }
            }
#endif


        /* do final operations on results */
        tstart = second();
        for(i = FirstActiveParticle, npleft = 0; i >= 0; i = NextActiveParticle[i])
        {
            if(density_isactive(i))
            {
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
                                dvel[k][0] = SPHP(i).u.DV[k][0] / SPHP(i).d.Density;
                                dvel[k][1] = SPHP(i).u.DV[k][1] / SPHP(i).d.Density;
                                dvel[k][2] = SPHP(i).u.DV[k][2] / SPHP(i).d.Density;
                            }
                            SPHP(i).u.s.DivVel = dvel[0][0] + dvel[1][1] + dvel[2][2];

                            SPHP(i).u.s.StressDiag[0] = 2 * dvel[0][0] - 2.0 / 3 * SPHP(i).u.s.DivVel;
                            SPHP(i).u.s.StressDiag[1] = 2 * dvel[1][1] - 2.0 / 3 * SPHP(i).u.s.DivVel;
                            SPHP(i).u.s.StressDiag[2] = 2 * dvel[2][2] - 2.0 / 3 * SPHP(i).u.s.DivVel;

                            SPHP(i).u.s.StressOffDiag[0] = dvel[0][1] + dvel[1][0];	/* xy */
                            SPHP(i).u.s.StressOffDiag[1] = dvel[0][2] + dvel[2][0];	/* xz */
                            SPHP(i).u.s.StressOffDiag[2] = dvel[1][2] + dvel[2][1];	/* yz */

#ifdef NAVIERSTOKES_BULK
                            SPHP(i).u.s.StressBulk = All.NavierStokes_BulkViscosity * SPHP(i).u.s.DivVel;
#endif
                            rotx = dvel[1][2] - dvel[2][1];
                            roty = dvel[2][0] - dvel[0][2];
                            rotz = dvel[0][1] - dvel[1][0];
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
#endif

#else
                        SPHP(i).Pressure =
                            GAMMA_MINUS1 * (SPHP(i).Entropy + SPHP(i).e.DtEntropy * dt_entr) * SPHP(i).d.Density;
#endif

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
#endif
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


                /* now check whether we had enough neighbours */

                desnumngb = All.DesNumNgb;

#ifdef BLACK_HOLES
                if(P[i].Type == 5)
                    desnumngb = All.DesNumNgb * All.BlackHoleNgbFactor;
#endif

                if(P[i].n.NumNgb < (desnumngb - All.MaxNumNgbDeviation) ||
                        (P[i].n.NumNgb > (desnumngb + All.MaxNumNgbDeviation)
                         && P[i].Hsml > (1.01 * All.MinGasHsml)))
                {
                    /* need to redo this particle */
                    npleft++;


                    if(Left[i] > 0 && Right[i] > 0)
                        if((Right[i] - Left[i]) < 1.0e-3 * Left[i])
                        {
                            /* this one should be ok */
                            npleft--;
                            P[i].DensityIterationDone = 1;
                            continue;
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

                    if(iter >= MAXITER - 10)
                    {
                        printf
                            ("i=%d task=%d ID=%llu Hsml=%g Left=%g Right=%g Ngbs=%g Right-Left=%g\n   pos=(%g|%g|%g)\n",
                             i, ThisTask, P[i].ID, P[i].Hsml, Left[i], Right[i],
                             (float) P[i].n.NumNgb, Right[i] - Left[i], P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
                        fflush(stdout);
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
                                fac = 1 - (P[i].n.NumNgb -
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
                                fac = 1 - (P[i].n.NumNgb -
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
                else
                    P[i].DensityIterationDone = 1;

            }
        }
        tend = second();
        timecomp3 += timediff(tstart, tend);

        sumup_large_ints(1, &npleft, &ntot);

        if(ntot > 0)
        {
            iter++;

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


    /* mark as active again */
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        P[i].DensityIterationDone = 0;
    }

    /* collect some timing information */

    t1 = WallclockTime = second();
    timeall += timediff(t0, t1);

    timecomp = timecomp3 + ev.timecomp1 + ev.timecomp2;
    timewait = ev.timewait1 + ev.timewait2;
    timecomm = ev.timecommsumm1 + ev.timecommsumm2;

    CPU_Step[CPU_DENSCOMPUTE] += timecomp;
    CPU_Step[CPU_DENSWAIT] += timewait;
    CPU_Step[CPU_DENSCOMM] += timecomm;
    CPU_Step[CPU_DENSMISC] += timeall - (timecomp + timewait + timecomm);
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

static void density_copy(int place, struct densdata_in * input) {
    input->Pos[0] = P[place].Pos[0];
    input->Pos[1] = P[place].Pos[1];
    input->Pos[2] = P[place].Pos[2];
    input->Hsml = P[place].Hsml;

    input->Type = P[place].Type;

#if defined(BLACK_HOLES)
    if(P[place].Type != 0)
    {
        input->Vel[0] = 0;
        input->Vel[1] = 0;
        input->Vel[2] = 0;
    }
    else
#endif
    {
        input->Vel[0] = SPHP(place).VelPred[0];
        input->Vel[1] = SPHP(place).VelPred[1];
        input->Vel[2] = SPHP(place).VelPred[2];
    }
#ifdef VOLUME_CORRECTION
    input->DensityOld = SPHP(place).DensityOld;
#endif

#ifdef EULERPOTENTIALS
    input->EulerA = SPHP(place).EulerA;
    input->EulerB = SPHP(place).EulerB;
#endif
#ifdef VECT_POTENTIAL
    input->APred[0] = SPHP(place).APred[0];
    input->APred[1] = SPHP(place).APred[1];
    input->APred[2] = SPHP(place).APred[2];
    input->rrho = SPHP(place).d.Density;
#endif
#if defined(MAGNETICSEED)
    input->MagSeed = SPHP(place).MagSeed;
#endif


#ifdef WINDS
    input->DelayTime = SPHP(place).DelayTime;
#endif

#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS) || defined(TRACEDIVB)
    input->BPred[0] = SPHP(place).BPred[0];
    input->BPred[1] = SPHP(place).BPred[1];
    input->BPred[2] = SPHP(place).BPred[2];
#endif
}

static void density_reduce(int place, struct densdata_out * remote, int mode) {

#define REDUCE(A, B) (A) = (mode==0)?0:((A) + (B))
    REDUCE(P[place].n.dNumNgb, remote->Ngb);

#ifdef HYDRO_COST_FACTOR
    /* these will be added */
    if(All.ComovingIntegrationOn)
        P[place].GravCost += HYDRO_COST_FACTOR * All.Time * remote->Ninteractions;
    else
        P[place].GravCost += HYDRO_COST_FACTOR * remote->Ninteractions;
#endif

    if(P[place].Type == 0)
    {
        REDUCE(SPHP(place).d.dDensity, remote->Rho);
        REDUCE(SPHP(place).h.dDhsmlDensityFactor, remote->DhsmlDensity);
#ifdef DENSITY_INDEPENDENT_SPH
        REDUCE(SPHP(place).EgyWtDensity, remote->EgyRho);
        REDUCE(SPHP(place).DhsmlEgyDensityFactor, remote->DhsmlEgyDensity);
#endif

#ifndef NAVIERSTOKES
        REDUCE(SPHP(place).v.dDivVel, remote->Div);
        REDUCE(SPHP(place).r.dRot[0], remote->Rot[0]);
        REDUCE(SPHP(place).r.dRot[1], remote->Rot[1]);
        REDUCE(SPHP(place).r.dRot[2], remote->Rot[2]);
#else
        for(k = 0; k < 3; k++)
        {
            REDUCE(SPHP(place).u.DV[k][0], remote->DV[k][0]);
            REDUCE(SPHP(place).u.DV[k][1], remote->DV[k][1]);
            REDUCE(SPHP(place).u.DV[k][2], remote->DV[k][2]);
        }
#endif

#ifdef VOLUME_CORRECTION
        REDUCE(SPHP(place).DensityStd, remote->DensityStd);
#endif

#ifdef CONDUCTION_SATURATION
        REDUCE(SPHP(place).GradEntr[0], remote->GradEntr[0]);
        REDUCE(SPHP(place).GradEntr[1], remote->GradEntr[1]);
        REDUCE(SPHP(place).GradEntr[2], remote->GradEntr[2]);
#endif

#ifdef RADTRANSFER_FLUXLIMITER
        for(k = 0; k< N_BINS; k++)
        {
            REDUCE(SPHP(place).Grad_ngamma[0][k], remote->Grad_ngamma[0][k]);
            REDUCE(SPHP(place).Grad_ngamma[1][k], remote->Grad_ngamma[1][k]);
            REDUCE(SPHP(place).Grad_ngamma[2][k], remote->Grad_ngamma[2][k]);
        }
#endif


#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS)
        REDUCE(SPHP(place).RotB[0], remote->RotB[0]);
        REDUCE(SPHP(place).RotB[1], remote->RotB[1]);
        REDUCE(SPHP(place).RotB[2], remote->RotB[2]);
#endif

#ifdef TRACEDIVB
        REDUCE(SPHP(place).divB, remote->divB);
#endif

#ifdef JD_VTURB
        REDUCE(SPHP(place).Vturb, remote->Vturb);
        REDUCE(SPHP(place).Vbulk[0], remote->Vbulk[0]);
        REDUCE(SPHP(place).Vbulk[1], remote->Vbulk[1]);
        REDUCE(SPHP(place).Vbulk[2], remote->Vbulk[2]);
        REDUCE(SPHP(place).TrueNGB, remote->TrueNGB);
#endif

#ifdef VECT_PRO_CLEAN
        REDUCE(SPHP(place).BPredVec[0], remote->BPredVec[0]);
        REDUCE(SPHP(place).BPredVec[1], remote->BPredVec[1]);
        REDUCE(SPHP(place).BPredVec[2], remote->BPredVec[2]);
#endif
#ifdef EULERPOTENTIALS
        REDUCE(SPHP(place).dEulerA[0], remote->dEulerA[0]);
        REDUCE(SPHP(place).dEulerA[1], remote->dEulerA[1]);
        REDUCE(SPHP(place).dEulerA[2], remote->dEulerA[2]);
        REDUCE(SPHP(place).dEulerB[0], remote->dEulerB[0]);
        REDUCE(SPHP(place).dEulerB[1], remote->dEulerB[1]);
        REDUCE(SPHP(place).dEulerB[2], remote->dEulerB[2]);
#endif
#ifdef VECT_POTENTIAL
        REDUCE(SPHP(place).dA[5], remote->da[5]);
        REDUCE(SPHP(place).dA[4], remote->da[4]);
        REDUCE(SPHP(place).dA[3], remote->da[3]);
        REDUCE(SPHP(place).dA[2], remote->da[2]);
        REDUCE(SPHP(place).dA[1], remote->da[1]);
        REDUCE(SPHP(place).dA[0], remote->da[0]);
#endif
    }

#if (defined(RADTRANSFER) && defined(EDDINGTON_TENSOR_STARS)) || defined(SNIA_HEATING)
    if(P[place].Type == 4)
        REDUCE(P[place].DensAroundStar, remote->Rho);
#endif

#ifdef BLACK_HOLES
    if(P[place].Type == 5)
    {
        if (mode == 0 || 
                BHP(place).TimeBinLimit < 0 || 
                BHP(place).TimeBinLimit > remote->BH_TimeBinLimit) {
            BHP(place).TimeBinLimit = remote->BH_TimeBinLimit;
        }
        REDUCE(BHP(place).Density, remote->Rho);
        REDUCE(BHP(place).FeedbackWeightSum, remote->FeedbackWeightSum);
        REDUCE(BHP(place).EntOrPressure, remote->SmoothedEntOrPressure);
#ifdef BH_USE_GASVEL_IN_BONDI
        REDUCE(BHP(place).SurroundingGasVel[0], remote->GasVel[0]);
        REDUCE(BHP(place).SurroundingGasVel[1], remote->GasVel[1]);
        REDUCE(BHP(place).SurroundingGasVel[2], remote->GasVel[2]);
#endif
        /*
        printf("%d BHP(%d), TimeBinLimit=%d, TimeBin=%d\n",
                ThisTask, place, BHP(place).TimeBinLimit, P[place].TimeBin);
                */
    }
#endif

}
/*! This function represents the core of the SPH density computation. The
 *  target particle may either be local, or reside in the communication
 *  buffer.
 */
static int density_evaluate(int target, int mode, Exporter * exporter, int * ngblist) 
{
    int j, n;

    int startnode, numngb, numngb_inbox, listindex = 0;

    double h;
    double hsearch;
    density_kernel_t kernel;
    MyLongDouble rho;
    short int timebin_min = -1;

#ifdef BLACK_HOLES
    MyLongDouble fb_weight_sum;  /*smoothing density used in feedback */
    density_kernel_t bh_feedback_kernel;
#endif
    int type;
    double wk, dwk;

    double dx, dy, dz, r, r2, mass_j;

    double dvx, dvy, dvz;

    MyLongDouble weighted_numngb;

    MyLongDouble dhsmlrho;

#ifdef HYDRO_COST_FACTOR
    int ninteractions = 0;
#endif

#ifdef BLACK_HOLES
    MyLongDouble gasvel[3];
#endif
#ifndef NAVIERSTOKES
    MyLongDouble divv, rotv[3];
#else
    int k;

    double dvel[3][3];
#endif

    MyDouble *pos;

    MyFloat *vel;
    static MyFloat veldummy[3] = { 0, 0, 0 };

#ifdef EULERPOTENTIALS
    MyDouble deulera[3], deulerb[3], eulera, eulerb, dea, deb;

    deulera[0] = deulera[1] = deulera[2] = deulerb[0] = deulerb[1] = deulerb[2] = 0;
#endif
#if defined(MAGNETICSEED)
    double spin=0,spin_0=0;
    double mu0_1 = 1;
#ifndef MU0_UNITY
    mu0_1 *= (4 * M_PI);
    mu0_1 /= All.UnitTime_in_s * All.UnitTime_in_s *
        All.UnitLength_in_cm / All.UnitMass_in_g;
    if(All.ComovingIntegrationOn)
        mu0_1 /= (All.HubbleParam * All.HubbleParam);
#endif
#endif

#ifdef VECT_POTENTIAL
    MyDouble dA[6], aflt[3];

    MyDouble bpred[3], rrho;

    dA[0] = dA[1] = dA[2] = 0.0;
    dA[3] = dA[4] = dA[5] = 0.0;
    double hinv_j, dwk_j, wk_j, hinv3_j, hinv4_j;
#endif

#ifdef DENSITY_INDEPENDENT_SPH
    double egyrho = 0, dhsmlegyrho = 0;
#endif


#if defined(BLACK_HOLES)
    MyLongDouble smoothent_or_pres;

    smoothent_or_pres = 0;
#endif

#ifdef WINDS
    double delaytime;
#endif

#ifdef VOLUME_CORRECTION
    double densityold, densitystd = 0;
#endif

#ifdef JD_VTURB
    MyFloat vturb = 0;
    MyFloat vbulk[3]={0};
    int trueNGB = 0;
#endif

#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS) || defined(TRACEDIVB)
    MyFloat bflt[3];

    MyDouble dbx, dby, dbz;
#endif

#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS)
    MyDouble rotb[3];
#endif

#ifdef TRACEDIVB
    double divB = 0;
#endif
#ifdef VECT_PRO_CLEAN
    double BVec[3];

    BVec[0] = BVec[1] = BVec[2] = 0;
#endif

#ifdef CONDUCTION_SATURATION
    double gradentr[3];

    gradentr[0] = gradentr[1] = gradentr[2] = 0;
#endif

#ifdef RADTRANSFER_FLUXLIMITER
    double grad_ngamma[3][N_BINS];
    int k;

    for(k = 0; k < N_BINS; k++)
        grad_ngamma[0][k] = grad_ngamma[1][k] = grad_ngamma[2][k] = 0;
#endif

#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS)
    rotb[0] = rotb[1] = rotb[2] = 0;
#endif

#ifndef NAVIERSTOKES
    divv = rotv[0] = rotv[1] = rotv[2] = 0;
#else
    for(k = 0; k < 3; k++)
        dvel[k][0] = dvel[k][1] = dvel[k][2] = 0;
#endif
#ifdef BLACK_HOLES
    gasvel[0] = gasvel[1] = gasvel[2] = 0;
    fb_weight_sum = 0;
#endif
    rho = weighted_numngb = dhsmlrho = 0;

    if(mode == 0)
    {
        pos = P[target].Pos;
        h = P[target].Hsml;
        type = P[target].Type;
        hsearch = density_decide_hsearch(P[target].Type, h);
#ifdef VOLUME_CORRECTION
        densityold = SPHP(target).DensityOld;
#endif
        if(P[target].Type == 0)
        {
            vel = SPHP(target).VelPred;
#ifdef WINDS
            delaytime = SPHP(target).DelayTime;
#endif
#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS) || defined(TRACEDIVB)
            bflt[0] = SPHP(target).BPred[0];
            bflt[1] = SPHP(target).BPred[1];
            bflt[2] = SPHP(target).BPred[2];
#endif
#ifdef VECT_POTENTIAL
            aflt[0] = SPHP(target).APred[0];
            aflt[1] = SPHP(target).APred[1];
            aflt[2] = SPHP(target).APred[2];
            rrho = SPHP(target).d.Density;
#endif
#ifdef EULERPOTENTIALS
            eulera = SPHP(target).EulerA;
            eulerb = SPHP(target).EulerB;
#endif
        }
        else
            vel = veldummy;
#if defined(MAGNETICSEED)
        spin = SPHP(target).MagSeed;
#endif
    }
    else
    {
        pos = DensDataGet[target].Pos;
        vel = DensDataGet[target].Vel;
        h = DensDataGet[target].Hsml;
        type = DensDataGet[target].Type;
        hsearch = density_decide_hsearch(DensDataGet[target].Type, h);

#ifdef WINDS
        delaytime = DensDataGet[target].DelayTime;
#endif
#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS) || defined(TRACEDIVB)
        bflt[0] = DensDataGet[target].BPred[0];
        bflt[1] = DensDataGet[target].BPred[1];
        bflt[2] = DensDataGet[target].BPred[2];
#endif
#ifdef VECT_POTENTIAL
        aflt[0] = DensDataGet[target].APred[0];
        aflt[1] = DensDataGet[target].APred[1];
        aflt[2] = DensDataGet[target].APred[2];
        rrho = DensDataGet[target].rrho;
#endif
#ifdef VOLUME_CORRECTION
        densityold = DensDataGet[target].DensityOld;
#endif
#ifdef EULERPOTENTIALS
        eulera = DensDataGet[target].EulerA;
        eulerb = DensDataGet[target].EulerB;
#endif
#if defined(MAGNETICSEED)
        spin =   DensDataGet[target].MagSeed;
#endif
    }


    density_kernel_init(&kernel, h);
    double kernel_volume = density_kernel_volume(&kernel);
#ifdef BLACK_HOLES
    density_kernel_init(&bh_feedback_kernel, hsearch);
#endif
    if(mode == 0)
    {
        startnode = All.MaxPart;	/* root node */
    }
    else
    {
        startnode = DensDataGet[target].NodeList[0];
        startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }

    numngb = 0;

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb_inbox =
                ngb_treefind_variable_threads(pos, hsearch, target, &startnode, 
                        mode, exporter, ngblist);

            if(numngb_inbox < 0)
                return -1;

            for(n = 0; n < numngb_inbox; n++)
            {
#ifdef HYDRO_COST_FACTOR
                ninteractions++;
#endif
                j = ngblist[n];
#ifdef WINDS
                    if(SPHP(j).DelayTime > 0)	/* partner is a wind particle */
                        if(!(delaytime > 0))	/* if I'm not wind, then ignore the wind particle */
                            continue;
#endif
#ifdef BLACK_HOLES
                if(P[j].Mass == 0)
                    continue;
#endif
                dx = pos[0] - P[j].Pos[0];
                dy = pos[1] - P[j].Pos[1];
                dz = pos[2] - P[j].Pos[2];

#ifdef PERIODIC			/*  now find the closest image in the given box size  */
                dx = NEAREST_X(dx);
                dy = NEAREST_Y(dy);
                dz = NEAREST_Z(dz);
#endif
                r2 = dx * dx + dy * dy + dz * dz;

                r = sqrt(r2);

                if(r2 < kernel.HH)
                {

                    double u = r * kernel.Hinv;
                    wk = density_kernel_wk(&kernel, u);
                    dwk = density_kernel_dwk(&kernel, u);

                    mass_j = P[j].Mass;

                        numngb++;
#ifdef JD_VTURB
                        vturb += (SPHP(j).VelPred[0] - vel[0]) * (SPHP(j).VelPred[0] - vel[0]) +
                            (SPHP(j).VelPred[1] - vel[1]) * (SPHP(j).VelPred[1] - vel[1]) +
                            (SPHP(j).VelPred[2] - vel[2]) * (SPHP(j).VelPred[2] - vel[2]);
                        vbulk[0] += SPHP(j).VelPred[0];
                        vbulk[1] += SPHP(j).VelPred[1];
                        vbulk[2] += SPHP(j).VelPred[2];
                        trueNGB++;
#endif

#ifdef VOLUME_CORRECTION
                        rho += FLT(mass_j * wk * pow(densityold / SPHP(j).DensityOld, VOLUME_CORRECTION));
                        densitystd += FLT(mass_j * wk);
#else
                        rho += FLT(mass_j * wk);
#endif
                        weighted_numngb += wk * kernel_volume;

                        /* Hinv is here becuase dhsmlrho is drho / dH.
                         * nothing to worry here */
                        dhsmlrho += mass_j * density_kernel_dW(&kernel, u, wk, dwk);

                        if (timebin_min <= 0 || timebin_min >= P[j].TimeBin) 
                            timebin_min = P[j].TimeBin;

#ifdef DENSITY_INDEPENDENT_SPH
                        egyrho += mass_j * SPHP(j).EntVarPred * wk;
                        dhsmlegyrho += mass_j * SPHP(j).EntVarPred * density_kernel_dW(&kernel, u, wk, dwk);
#endif


#ifdef BLACK_HOLES
#ifdef BH_CSND_FROM_PRESSURE
                        smoothent_or_pres += FLT(mass_j * wk * SPHP(j).Pressure);
#else
                        smoothent_or_pres += FLT(mass_j * wk * SPHP(j).Entropy);
#endif
                        gasvel[0] += FLT(mass_j * wk * SPHP(j).VelPred[0]);
                        gasvel[1] += FLT(mass_j * wk * SPHP(j).VelPred[1]);
                        gasvel[2] += FLT(mass_j * wk * SPHP(j).VelPred[2]);
#endif

#ifdef CONDUCTION_SATURATION
                        if(r > 0)
                        {
                            gradentr[0] += mass_j * dwk * dx / r * SPHP(j).Entropy;
                            gradentr[1] += mass_j * dwk * dy / r * SPHP(j).Entropy;
                            gradentr[2] += mass_j * dwk * dz / r * SPHP(j).Entropy;
                        }
#endif


#ifdef RADTRANSFER_FLUXLIMITER
                        if(r > 0)
                            for(k = 0; k < N_BINS; k++)
                            {
                                grad_ngamma[0][k] += mass_j * dwk * dx / r * SPHP(j).n_gamma[k];
                                grad_ngamma[1][k] += mass_j * dwk * dy / r * SPHP(j).n_gamma[k];
                                grad_ngamma[2][k] += mass_j * dwk * dz / r * SPHP(j).n_gamma[k];
                            }
#endif


                        if(r > 0)
                        {
                            double fac = mass_j * dwk / r;

                            dvx = vel[0] - SPHP(j).VelPred[0];
                            dvy = vel[1] - SPHP(j).VelPred[1];
                            dvz = vel[2] - SPHP(j).VelPred[2];

#ifndef NAVIERSTOKES
                            divv += FLT(-fac * (dx * dvx + dy * dvy + dz * dvz));

                            rotv[0] += FLT(fac * (dz * dvy - dy * dvz));
                            rotv[1] += FLT(fac * (dx * dvz - dz * dvx));
                            rotv[2] += FLT(fac * (dy * dvx - dx * dvy));
#else
                            dvel[0][0] -= fac * dx * dvx;
                            dvel[0][1] -= fac * dx * dvy;
                            dvel[0][2] -= fac * dx * dvz;
                            dvel[1][0] -= fac * dy * dvx;
                            dvel[1][1] -= fac * dy * dvy;
                            dvel[1][2] -= fac * dy * dvz;
                            dvel[2][0] -= fac * dz * dvx;
                            dvel[2][1] -= fac * dz * dvy;
                            dvel[2][2] -= fac * dz * dvz;
#endif
#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS) || defined(TRACEDIVB)
                            dbx = bflt[0] - SPHP(j).BPred[0];
                            dby = bflt[1] - SPHP(j).BPred[1];
                            dbz = bflt[2] - SPHP(j).BPred[2];
#endif
#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS)
                            rotb[0] += FLT(fac * (dz * dby - dy * dbz));
                            rotb[1] += FLT(fac * (dx * dbz - dz * dbx));
                            rotb[2] += FLT(fac * (dy * dbx - dx * dby));
#endif
#ifdef VECT_POTENTIAL
                            dA[0] += fac * (aflt[0] - SPHP(j).APred[0]) * dy;	//dAx/dy
                            dA[1] += fac * (aflt[0] - SPHP(j).APred[0]) * dz;	//dAx/dz
                            dA[2] += fac * (aflt[1] - SPHP(j).APred[1]) * dx;	//dAy/dx
                            dA[3] += fac * (aflt[1] - SPHP(j).APred[1]) * dz;	//dAy/dz
                            dA[4] += fac * (aflt[2] - SPHP(j).APred[2]) * dx;	//dAz/dx
                            dA[5] += fac * (aflt[2] - SPHP(j).APred[2]) * dy;	//dAz/dy
#endif
#ifdef TRACEDIVB
                            divB += FLT(-fac * (dbx * dx + dby * dy + dbz * dz));
#endif
#ifdef MAGNETICSEED
                            spin_0=sqrt(spin*mu0_1*2.);//energy to B field
                            spin_0=3./2.*spin_0/(sqrt(vel[0]*vel[0]+vel[1]*vel[1]+vel[2]*vel[2]));//*SPHP(j).d.Density;

                            if(spin)
                            {
                                SPHP(j).BPred[0] += 1./(4.* M_PI * (pow(r,3.))) *
                                    (3. *(dx*vel[0] + dy*vel[1] + dz*vel[2]) * spin_0 / r  * dx / r - spin_0 * vel[0]);
                                SPHP(j).BPred[1] += 1./(4.* M_PI * (pow(r,3.))) *
                                    (3. *(dx*vel[0] + dy*vel[1] + dz*vel[2]) * spin_0 / r  * dy / r - spin_0 * vel[1]);
                                SPHP(j).BPred[2] += 1./(4.* M_PI * (pow(r,3.))) *
                                    (3. *(dx*vel[0] + dy*vel[1] + dz*vel[2]) * spin_0 / r  * dz / r - spin_0 * vel[2]);
                            };
#endif
#ifdef VECT_PRO_CLEAN
                            BVec[0] +=
                                FLT(fac * r2 * (SPHP(j).RotB[1] * dz - SPHP(j).RotB[2] * dy) / SPHP(j).d.Density);
                            BVec[1] +=
                                FLT(fac * r2 * (SPHP(j).RotB[2] * dx - SPHP(j).RotB[0] * dz) / SPHP(j).d.Density);
                            BVec[2] +=
                                FLT(fac * r2 * (SPHP(j).RotB[0] * dy - SPHP(j).RotB[1] * dx) / SPHP(j).d.Density);
#endif
#ifdef EULERPOTENTIALS
                            dea = eulera - SPHP(j).EulerA;
                            deb = eulerb - SPHP(j).EulerB;
#ifdef EULER_VORTEX
                            deb = NEAREST_Z(deb);
#endif
                            deulera[0] -= fac * dx * dea;
                            deulera[1] -= fac * dy * dea;
                            deulera[2] -= fac * dz * dea;
                            deulerb[0] -= fac * dx * deb;
                            deulerb[1] -= fac * dy * deb;
                            deulerb[2] -= fac * dz * deb;
#endif
                        }
                }
#ifdef BLACK_HOLES
                if(type == 5 && r2 < bh_feedback_kernel.HH)
                {
                    double mass_j;
                    if(All.BlackHoleFeedbackMethod & BH_FEEDBACK_OPTTHIN) {
#ifdef COOLING
                        double nh0 = 0;
                        double nHeII = 0;
                        double ne = SPHP(j).Ne;
                        AbundanceRatios(DMAX(All.MinEgySpec,
                                    SPHP(j).Entropy / GAMMA_MINUS1 
                                    * pow(SPHP(j).EOMDensity * a3inv,
                                        GAMMA_MINUS1)),
                                SPHP(j).d.Density * a3inv, &ne, &nh0, &nHeII);
#else
                        double nh0 = 0;
#endif
                        if(r2 > 0)
                            fb_weight_sum += FLT(P[j].Mass * nh0) / r2;
                    } else {
                        if(All.BlackHoleFeedbackMethod & BH_FEEDBACK_MASS) {
                            mass_j = P[j].Mass;
                        } else {
                            mass_j = P[j].Hsml * P[j].Hsml * P[j].Hsml;
                        }
                        if(All.BlackHoleFeedbackMethod & BH_FEEDBACK_SPLINE) {
                            double u = r * bh_feedback_kernel.Hinv;
                            fb_weight_sum += FLT(mass_j * 
                                  density_kernel_wk(&bh_feedback_kernel, u)
                                   );
                        } else {
                            fb_weight_sum += FLT(mass_j);
                        }
                    }
                }
#endif
            }
        }
        if(mode == 1)
        {
            listindex++;
            if(listindex < NODELISTLENGTH)
            {
                startnode = DensDataGet[target].NodeList[listindex];
                if(startnode >= 0)
                    startnode = Nodes[startnode].u.d.nextnode;	/* open it */
            }
        }
    }
    if(mode == 0)
    {
        P[target].n.dNumNgb = weighted_numngb;
#ifdef HYDRO_COST_FACTOR
        if(All.ComovingIntegrationOn)
            P[target].GravCost += HYDRO_COST_FACTOR * All.Time * ninteractions;
        else
            P[target].GravCost += HYDRO_COST_FACTOR * ninteractions;
#endif
        if(P[target].Type == 0)
        {
            SPHP(target).d.dDensity = rho;

#ifdef DENSITY_INDEPENDENT_SPH
            SPHP(target).EgyWtDensity = egyrho;
            SPHP(target).DhsmlEgyDensityFactor = dhsmlegyrho;
#endif

#ifdef VOLUME_CORRECTION
            SPHP(target).DensityStd = densitystd;
#endif
            SPHP(target).h.dDhsmlDensityFactor = dhsmlrho;
#ifndef NAVIERSTOKES
            SPHP(target).v.dDivVel = divv;
            SPHP(target).r.dRot[0] = rotv[0];
            SPHP(target).r.dRot[1] = rotv[1];
            SPHP(target).r.dRot[2] = rotv[2];
#else
            for(k = 0; k < 3; k++)
            {
                SPHP(target).u.DV[k][0] = dvel[k][0];
                SPHP(target).u.DV[k][1] = dvel[k][1];
                SPHP(target).u.DV[k][2] = dvel[k][2];
            }
#endif

#ifdef CONDUCTION_SATURATION
            SPHP(target).GradEntr[0] = gradentr[0];
            SPHP(target).GradEntr[1] = gradentr[1];
            SPHP(target).GradEntr[2] = gradentr[2];
#endif

#ifdef RADTRANSFER_FLUXLIMITER
            for(k = 0; k < N_BINS; k++)
            {
                SPHP(target).Grad_ngamma[0][k] = grad_ngamma[0][k];
                SPHP(target).Grad_ngamma[1][k] = grad_ngamma[1][k];
                SPHP(target).Grad_ngamma[2][k] = grad_ngamma[2][k];
            }
#endif

#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS)
            SPHP(target).RotB[0] = rotb[0];
            SPHP(target).RotB[1] = rotb[1];
            SPHP(target).RotB[2] = rotb[2];
#endif
#ifdef VECT_POTENTIAL
            SPHP(target).dA[0] = dA[0];
            SPHP(target).dA[1] = dA[1];
            SPHP(target).dA[2] = dA[2];
            SPHP(target).dA[3] = dA[3];
            SPHP(target).dA[4] = dA[4];
            SPHP(target).dA[5] = dA[5];

#endif

#ifdef JD_VTURB
            SPHP(target).Vturb = vturb;
            SPHP(target).Vbulk[0] = vbulk[0];
            SPHP(target).Vbulk[1] = vbulk[1];
            SPHP(target).Vbulk[2] = vbulk[2];
            SPHP(target).TrueNGB = trueNGB;
#endif

#ifdef TRACEDIVB
            SPHP(target).divB = divB;
#endif
#ifdef EULERPOTENTIALS
            SPHP(target).dEulerA[0] = deulera[0];
            SPHP(target).dEulerA[1] = deulera[1];
            SPHP(target).dEulerA[2] = deulera[2];
            SPHP(target).dEulerB[0] = deulerb[0];
            SPHP(target).dEulerB[1] = deulerb[1];
            SPHP(target).dEulerB[2] = deulerb[2];
#endif
#ifdef VECT_PRO_CLEAN
            SPHP(target).BPredVec[0] = BVec[0];
            SPHP(target).BPredVec[1] = BVec[1];
            SPHP(target).BPredVec[2] = BVec[2];
#endif
        }
#ifdef BLACK_HOLES
        if(P[target].Type == 5)  {
            BHP(target).Density = rho;
            BHP(target).TimeBinLimit = timebin_min;
            BHP(target).FeedbackWeightSum = fb_weight_sum;
            BHP(target).EntOrPressure = smoothent_or_pres;
#ifdef BH_USE_GASVEL_IN_BONDI
            BHP(target).SurroundingGasVel[0] = gasvel[0];
            BHP(target).SurroundingGasVel[1] = gasvel[1];
            BHP(target).SurroundingGasVel[2] = gasvel[2];
#endif
        }
#endif
#if (defined(RADTRANSFER) && defined(EDDINGTON_TENSOR_STARS)) || defined(SNIA_HEATING)
        if(P[target].Type == 4)
            P[target].DensAroundStar = rho;
#endif
    }
    else
    {
#ifdef HYDRO_COST_FACTOR
        DensDataResult[target].Ninteractions = ninteractions;
#endif
        DensDataResult[target].Rho = rho;

#ifdef DENSITY_INDEPENDENT_SPH
        DensDataResult[target].EgyRho = egyrho;
        DensDataResult[target].DhsmlEgyDensity = dhsmlegyrho;
#endif

#ifdef VOLUME_CORRECTION
        DensDataResult[target].DensityStd = densitystd;
#endif
        DensDataResult[target].Ngb = weighted_numngb;
        DensDataResult[target].DhsmlDensity = dhsmlrho;
#ifndef NAVIERSTOKES
        DensDataResult[target].Div = divv;
        DensDataResult[target].Rot[0] = rotv[0];
        DensDataResult[target].Rot[1] = rotv[1];
        DensDataResult[target].Rot[2] = rotv[2];
#else
        for(k = 0; k < 3; k++)
        {
            DensDataResult[target].DV[k][0] = dvel[k][0];
            DensDataResult[target].DV[k][1] = dvel[k][1];
            DensDataResult[target].DV[k][2] = dvel[k][2];
        }
#endif

#if defined(BLACK_HOLES)
        DensDataResult[target].SmoothedEntOrPressure = smoothent_or_pres;
        DensDataResult[target].FeedbackWeightSum = fb_weight_sum;
        DensDataResult[target].BH_TimeBinLimit = timebin_min;
#endif
#ifdef CONDUCTION_SATURATION
        DensDataResult[target].GradEntr[0] = gradentr[0];
        DensDataResult[target].GradEntr[1] = gradentr[1];
        DensDataResult[target].GradEntr[2] = gradentr[2];
#endif

#ifdef RADTRANSFER_FLUXLIMITER
        for(k = 0; k < N_BINS; k++)
        {
            DensDataResult[target].Grad_ngamma[0][k] = grad_ngamma[0][k];
            DensDataResult[target].Grad_ngamma[1][k] = grad_ngamma[1][k];
            DensDataResult[target].Grad_ngamma[2][k] = grad_ngamma[2][k];
        }
#endif

#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS)
        DensDataResult[target].RotB[0] = rotb[0];
        DensDataResult[target].RotB[1] = rotb[1];
        DensDataResult[target].RotB[2] = rotb[2];
#endif

#ifdef JD_VTURB
        DensDataResult[target].Vturb = vturb;
        DensDataResult[target].Vbulk[0] = vbulk[0];
        DensDataResult[target].Vbulk[1] = vbulk[1];
        DensDataResult[target].Vbulk[2] = vbulk[2];
        DensDataResult[target].TrueNGB = trueNGB;
#endif

#ifdef TRACEDIVB
        DensDataResult[target].divB = divB;
#endif
#ifdef BLACK_HOLES
        DensDataResult[target].GasVel[0] = gasvel[0];
        DensDataResult[target].GasVel[1] = gasvel[1];
        DensDataResult[target].GasVel[2] = gasvel[2];
#endif
#ifdef VECT_POTENTIAL
        DensDataResult[target].da[0] = dA[0];
        DensDataResult[target].da[1] = dA[1];
        DensDataResult[target].da[2] = dA[2];
        DensDataResult[target].da[3] = dA[3];
        DensDataResult[target].da[4] = dA[4];
        DensDataResult[target].da[5] = dA[5];

#endif

#ifdef EULERPOTENTIALS
        DensDataResult[target].dEulerA[0] = deulera[0];
        DensDataResult[target].dEulerA[1] = deulera[1];
        DensDataResult[target].dEulerA[2] = deulera[2];
        DensDataResult[target].dEulerB[0] = deulerb[0];
        DensDataResult[target].dEulerB[1] = deulerb[1];
        DensDataResult[target].dEulerB[2] = deulerb[2];
#endif
#ifdef VECT_PRO_CLEAN
        DensDataResult[target].BPredVec[0] = BVec[0];
        DensDataResult[target].BPredVec[1] = BVec[1];
        DensDataResult[target].BPredVec[2] = BVec[2];
#endif
    }

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


#ifdef NAVIERSTOKES
double get_shear_viscosity(int i)
{
    return All.NavierStokes_ShearViscosity;
}
#endif

