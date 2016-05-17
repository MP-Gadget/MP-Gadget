#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>


#include "allvars.h"
#include "evaluator.h"
#include "proto.h"
#include "densitykernel.h"
#include "forcetree.h"
#include "mymalloc.h"

#ifndef DEBUG
#define NDEBUG
#endif

/*! \file hydra.c
 *  \brief Computation of SPH forces and rate of entropy generation
 *
 *  This file contains the "second SPH loop", where the SPH forces are
 *  computed, and where the rate of change of entropy due to the shock heating
 *  (via artificial viscosity) is computed.
 */
struct hydrodata_in
{
    int NodeList[NODELISTLENGTH];

#ifdef DENSITY_INDEPENDENT_SPH
    MyFloat EgyRho;
    MyFloat EntVarPred;
#endif

    MyDouble Pos[3];
    MyFloat Vel[3];
    MyFloat Hsml;
    MyFloat Mass;
    MyFloat Density;
    MyFloat Pressure;
    MyFloat F1;
    MyFloat DhsmlDensityFactor;
    int Timestep;

#ifdef PARTICLE_DEBUG
    MyIDType ID;			/*!< particle identifier */
#endif

};

struct hydrodata_out
{
    MyDouble Acc[3];
    MyDouble DtEntropy;
    MyFloat MaxSignalVel;

#ifdef HYDRO_COST_FACTOR
    int Ninteractions;
#endif
};


static int hydro_evaluate(int target, int mode,
        struct hydrodata_in * I,
        struct hydrodata_out * O,
        LocalEvaluator * lv);
static int hydro_isactive(int n);
static void hydro_post_process(int i);


static void hydro_copy(int place, struct hydrodata_in * input);
static void hydro_reduce(int place, struct hydrodata_out * result, int mode);

static double fac_mu, fac_vsic_fix;

/*! This function is the driver routine for the calculation of hydrodynamical
 *  force and rate of change of entropy due to shock heating for all active
 *  particles .
 */
void hydro_force(void)
{
    Evaluator ev = {0};

    ev.ev_label = "HYDRO";
    ev.ev_evaluate = (ev_ev_func) hydro_evaluate;
    ev.ev_isactive = hydro_isactive;
    ev.ev_copy = (ev_copy_func) hydro_copy;
    ev.ev_reduce = (ev_reduce_func) hydro_reduce;
    ev.UseNodeList = 0;
    ev.ev_datain_elsize = sizeof(struct hydrodata_in);
    ev.ev_dataout_elsize = sizeof(struct hydrodata_out);

    int i;
    double timeall = 0, timenetwork = 0;
    double timecomp, timecomm, timewait;

    walltime_measure("/Misc");

    fac_mu = pow(All.cf.a, 3 * (GAMMA - 1) / 2) / All.cf.a;
    fac_vsic_fix = All.cf.hubble * pow(All.cf.a, 3 * GAMMA_MINUS1);

    /* allocate buffers to arrange communication */

    walltime_measure("/SPH/Hydro/Init");

    ev_run(&ev);


    /* do final operations on results */

    int Nactive;
    int * queue = ev_get_queue(&ev, &Nactive);
#pragma omp parallel for if(Nactive > 64)
    for(i = 0; i < Nactive; i++)
        hydro_post_process(queue[i]);

    myfree(queue);

    /* collect some timing information */

    timeall += walltime_measure(WALLTIME_IGNORE);

    timecomp = ev.timecomp1 + ev.timecomp2;
    timewait = ev.timewait1 + ev.timewait2;
    timecomm = ev.timecommsumm1 + ev.timecommsumm2;

    walltime_add("/SPH/Hydro/Compute", timecomp);
    walltime_add("/SPH/Hydro/Wait", timewait);
    walltime_add("/SPH/Hydro/Comm", timecomm);
    walltime_add("/SPH/Hydro/Misc", timeall - (timecomp + timewait + timecomm + timenetwork));
}

static void hydro_copy(int place, struct hydrodata_in * input) {
    int k;
    double soundspeed_i;
    for(k = 0; k < 3; k++)
    {
        input->Pos[k] = P[place].Pos[k];
        input->Vel[k] = SPHP(place).VelPred[k];
    }
    input->Hsml = P[place].Hsml;
    input->Mass = P[place].Mass;
    input->Density = SPHP(place).Density;
#ifdef DENSITY_INDEPENDENT_SPH
    input->EgyRho = SPHP(place).EgyWtDensity;
    input->EntVarPred = SPHP(place).EntVarPred;
    input->DhsmlDensityFactor = SPHP(place).DhsmlEgyDensityFactor;
#else
    input->DhsmlDensityFactor = SPHP(place).DhsmlDensityFactor;
#endif

    input->Pressure = SPHP(place).Pressure;
    input->Timestep = (P[place].TimeBin ? (1 << P[place].TimeBin) : 0);
    /* calculation of F1 */
#ifndef ALTVISCOSITY
    soundspeed_i = sqrt(GAMMA * SPHP(place).Pressure / SPHP(place).EOMDensity);
    input->F1 = fabs(SPHP(place).DivVel) /
        (fabs(SPHP(place).DivVel) + SPHP(place).CurlVel +
         0.0001 * soundspeed_i / P[place].Hsml / fac_mu);

#else
    input->F1 = SPHP(place).DivVel;
#endif


#ifdef PARTICLE_DEBUG
    input->ID = P[place].ID;
#endif

}

static void hydro_reduce(int place, struct hydrodata_out * result, int mode) {
#define REDUCE(A, B) (A) = (mode==0)?(B):((A) + (B))
    int k;

    for(k = 0; k < 3; k++)
    {
        REDUCE(SPHP(place).HydroAccel[k], result->Acc[k]);
    }

    REDUCE(SPHP(place).DtEntropy, result->DtEntropy);

#ifdef HYDRO_COST_FACTOR
    P[place].GravCost += HYDRO_COST_FACTOR * All.cf.a * result->Ninteractions;
#endif

    if(mode == 0 || SPHP(place).MaxSignalVel < result->MaxSignalVel)
        SPHP(place).MaxSignalVel = result->MaxSignalVel;

}


/*! This function is the 'core' of the SPH force computation. A target
 *  particle is specified which may either be local, or reside in the
 *  communication buffer.
 */
static int hydro_evaluate(int target, int mode,
        struct hydrodata_in * I,
        struct hydrodata_out * O,
        LocalEvaluator * lv)
{
    int startnode, numngb, listindex = 0;
    int j, n;

    int ninteractions = 0;
    int nnodesinlist = 0;

    double p_over_rho2_i, p_over_rho2_j, soundspeed_i, soundspeed_j;

    DensityKernel kernel_i;
    DensityKernel kernel_j;

    startnode = I->NodeList[0];
    listindex ++;
    startnode = Nodes[startnode].u.d.nextnode;	/* open it */

#ifdef DENSITY_INDEPENDENT_SPH
    soundspeed_i = sqrt(GAMMA * I->Pressure / I->EgyRho);
#else
    soundspeed_i = sqrt(GAMMA * I->Pressure / I->Density);
#endif

    /* initialize variables before SPH loop is started */

    O->Acc[0] = O->Acc[1] = O->Acc[2] = O->DtEntropy = 0;
    density_kernel_init(&kernel_i, I->Hsml);

#ifdef DENSITY_INDEPENDENT_SPH
    p_over_rho2_i = I->Pressure / (I->EgyRho * I->EgyRho);
#else
    p_over_rho2_i = I->Pressure / (I->Density * I->Density);
#endif

    O->MaxSignalVel = soundspeed_i;


    /* Now start the actual SPH computation for this particle */

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb =
                ngb_treefind_threads(I->Pos, I->Hsml, target, &startnode,
                        mode, lv, NGB_TREEFIND_SYMMETRIC, 1); /* gas only 1 << 0 */

            if(numngb < 0)
                return numngb;

            for(n = 0; n < numngb; n++)
            {
                j = lv->ngblist[n];

                ninteractions++;

#ifdef BLACK_HOLES
                if(P[j].Mass == 0)
                    continue;
#endif

#ifdef WINDS
#ifdef NOWINDTIMESTEPPING
                if(HAS(All.WindModel, WINDS_DECOUPLE_SPH)) {
                    if(P[j].Type == 0)
                        if(SPHP(j).DelayTime > 0)	/* ignore the wind particles */
                            continue;
                }
#endif
#endif
                double dx = I->Pos[0] - P[j].Pos[0];
                double dy = I->Pos[1] - P[j].Pos[1];
                double dz = I->Pos[2] - P[j].Pos[2];

                dx = NEAREST(dx);
                dy = NEAREST(dy);
                dz = NEAREST(dz);
                double r2 = dx * dx + dy * dy + dz * dz;
                density_kernel_init(&kernel_j, P[j].Hsml);
                if(r2 > 0 && (r2 < kernel_i.HH || r2 < kernel_j.HH))
                {
                    double r = sqrt(r2);
                    p_over_rho2_j = SPHP(j).Pressure / (SPHP(j).EOMDensity * SPHP(j).EOMDensity);

#ifdef DENSITY_INDEPENDENT_SPH
                    soundspeed_j = sqrt(GAMMA * SPHP(j).Pressure / SPHP(j).EOMDensity);
#else
                    soundspeed_j = sqrt(GAMMA * p_over_rho2_j * SPHP(j).Density);
#endif

                    double dvx = I->Vel[0] - SPHP(j).VelPred[0];
                    double dvy = I->Vel[1] - SPHP(j).VelPred[1];
                    double dvz = I->Vel[2] - SPHP(j).VelPred[2];
                    double vdotr = dx * dvx + dy * dvy + dz * dvz;
                    double rho_ij = 0.5 * (I->Density + SPHP(j).Density);
                    double vdotr2 = vdotr + All.cf.hubble_a2 * r2;

                    double dwk_i = density_kernel_dwk(&kernel_i, r * kernel_i.Hinv);
                    double dwk_j = density_kernel_dwk(&kernel_j, r * kernel_j.Hinv);

                    double vsig = soundspeed_i + soundspeed_j;


                    if(vsig > O->MaxSignalVel)
                        O->MaxSignalVel = vsig;

                    double visc = 0;

                    if(vdotr2 < 0)	/* ... artificial viscosity visc is 0 by default*/
                    {
#ifndef ALTVISCOSITY
#ifndef CONVENTIONAL_VISCOSITY
                        double mu_ij = fac_mu * vdotr2 / r;	/* note: this is negative! */
#else
                        double c_ij = 0.5 * (soundspeed_i + soundspeed_j);
                        double h_ij = 0.5 * (I->Hsml + P[j].Hsml);
                        double mu_ij = fac_mu * h_ij * vdotr2 / (r2 + 0.0001 * h_ij * h_ij);
#endif
                        vsig -= 3 * mu_ij;


                        if(vsig > O->MaxSignalVel)
                            O->MaxSignalVel = vsig;

                        double f2 =
                            fabs(SPHP(j).DivVel) / (fabs(SPHP(j).DivVel) + SPHP(j).CurlVel +
                                    0.0001 * soundspeed_j / fac_mu / P[j].Hsml);

                        double BulkVisc_ij = All.ArtBulkViscConst;

#ifndef CONVENTIONAL_VISCOSITY
                        visc = 0.25 * BulkVisc_ij * vsig * (-mu_ij) / rho_ij * (I->F1 + f2);
#else
                        visc =
                            (-BulkVisc_ij * mu_ij * c_ij + 2 * BulkVisc_ij * mu_ij * mu_ij) /
                            rho_ij * (I->F1 + f2) * 0.5;
#endif

#else /* start of ALTVISCOSITY block */
                        double mu_i;
                        if(I->F1 < 0)
                            mu_i = I->Hsml * fabs(I->F1);	/* f1 hold here the velocity divergence of particle i */
                        else
                            mu_i = 0;
                        if(SPHP(j).DivVel < 0)
                            mu_j = P[j].Hsml * fabs(SPHP(j).DivVel);
                        else
                            mu_j = 0;
                        visc = All.ArtBulkViscConst * ((soundspeed_i + mu_i) * mu_i / I->Density +
                                (soundspeed_j + mu_j) * mu_j / SPHP(j).Density);
#endif /* end of ALTVISCOSITY block */


                        /* .... end artificial viscosity evaluation */
                        /* now make sure that viscous acceleration is not too large */

#ifndef NOVISCOSITYLIMITER
                        double dt =
                            2 * IMAX(I->Timestep,
                                    (P[j].TimeBin ? (1 << P[j].TimeBin) : 0)) * All.Timebase_interval;
                        if(dt > 0 && (dwk_i + dwk_j) < 0)
                        {
#ifdef BLACK_HOLES
                            if((I->Mass + P[j].Mass) > 0)
#endif
                                visc = DMIN(visc, 0.5 * fac_vsic_fix * vdotr2 /
                                        (0.5 * (I->Mass + P[j].Mass) * (dwk_i + dwk_j) * r * dt));
                        }
#endif
                    }
                    double hfc_visc = 0.5 * P[j].Mass * visc * (dwk_i + dwk_j) / r;
#ifdef DENSITY_INDEPENDENT_SPH
                    double hfc = hfc_visc;
                    /* leading-order term */
                    hfc += P[j].Mass *
                        (dwk_i*p_over_rho2_i*SPHP(j).EntVarPred/I->EntVarPred +
                         dwk_j*p_over_rho2_j*I->EntVarPred/SPHP(j).EntVarPred) / r;

                    /* enable grad-h corrections only if contrastlimit is non negative */
                    if(All.DensityContrastLimit >= 0) {
                        double r1 = I->EgyRho / I->Density;
                        double r2 = SPHP(j).EgyWtDensity / SPHP(j).Density;
                        if(All.DensityContrastLimit > 0) {
                            /* apply the limit if it is enabled > 0*/
                            if(r1 > All.DensityContrastLimit) {
                                r1 = All.DensityContrastLimit;
                            }
                            if(r2 > All.DensityContrastLimit) {
                                r2 = All.DensityContrastLimit;
                            }
                        }
                        /* grad-h corrections */
                        /* I->DhsmlDensityFactor is actually EgyDensityFactor */
                        hfc += P[j].Mass *
                            (dwk_i*p_over_rho2_i*r1*I->DhsmlDensityFactor +
                             dwk_j*p_over_rho2_j*r2*SPHP(j).DhsmlEgyDensityFactor) / r;
                    }
#else
                    /* Formulation derived from the Lagrangian */
                    double hfc = hfc_visc + P[j].Mass * (p_over_rho2_i *I->DhsmlDensityFactor * dwk_i
                            + p_over_rho2_j * SPHP(j).DhsmlDensityFactor * dwk_j) / r;
#endif

#ifdef WINDS
                    if(HAS(All.WindModel, WINDS_DECOUPLE_SPH)) {
                        if(P[j].Type == 0)
                            if(SPHP(j).DelayTime > 0)	/* No force by wind particles */
                            {
                                hfc = hfc_visc = 0;
                            }
                    }
#endif

#ifndef NOACCEL
                    O->Acc[0] += (-hfc * dx);
                    O->Acc[1] += (-hfc * dy);
                    O->Acc[2] += (-hfc * dz);
#endif

                    O->DtEntropy += (0.5 * hfc_visc * vdotr2);

                }
            }
        }

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
#ifdef HYDRO_COST_FACTOR
        O->Ninteractions = ninteractions;
#endif

    /* some performance measures not currently used */
    lv->Ninteractions += ninteractions;
    lv->Nnodesinlist += nnodesinlist;

    return 0;
}

static int hydro_isactive(int i) {
    return P[i].Type == 0;
}

static void hydro_post_process(int i) {
    if(P[i].Type == 0)
    {
        /* Translate energy change rate into entropy change rate */
        SPHP(i).DtEntropy *= GAMMA_MINUS1 / (All.cf.hubble_a2 * pow(SPHP(i).EOMDensity, GAMMA_MINUS1));

#ifdef WINDS
        /* if we have winds, we decouple particles briefly if delaytime>0 */
        if(HAS(All.WindModel, WINDS_DECOUPLE_SPH)) {
            if(SPHP(i).DelayTime > 0)
            {
                int k;
                for(k = 0; k < 3; k++)
                    SPHP(i).HydroAccel[k] = 0;

                SPHP(i).DtEntropy = 0;

#ifdef NOWINDTIMESTEPPING
                SPHP(i).MaxSignalVel = 2 * sqrt(GAMMA * SPHP(i).Pressure / SPHP(i).Density);
#else
                double windspeed = All.WindSpeed * All.cf.a;
                windspeed *= fac_mu;
                double hsml_c = pow(All.WindFreeTravelDensFac * All.PhysDensThresh /
                        (SPHP(i).Density * All.cf.a3inv), (1. / 3.));
                SPHP(i).MaxSignalVel = hsml_c * DMAX((2 * windspeed), SPHP(i).MaxSignalVel);
#endif
            }
        }
#endif
    }
}
