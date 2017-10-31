#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>


#include "allvars.h"
#include "garbage.h"
#include "treewalk.h"
#include "densitykernel.h"
#include "mymalloc.h"
#include "timestep.h"
#include "endrun.h"

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
typedef struct {
    TreeWalkQueryBase base;
#ifdef DENSITY_INDEPENDENT_SPH
    MyFloat EgyRho;
    MyFloat EntVarPred;
#endif

    MyFloat Vel[3];
    MyFloat Hsml;
    MyFloat Mass;
    MyFloat Density;
    MyFloat Pressure;
    MyFloat F1;
    MyFloat DhsmlDensityFactor;
    signed char TimeBin;

} TreeWalkQueryHydro;

typedef struct {
    TreeWalkResultBase base;
    MyFloat Acc[3];
    MyFloat DtEntropy;
    MyFloat MaxSignalVel;
    int Ninteractions;
} TreeWalkResultHydro;

typedef struct {
    TreeWalkNgbIterBase base;
    double p_over_rho2_i;
    double soundspeed_i;

    DensityKernel kernel_i;
} TreeWalkNgbIterHydro;

static int
hydro_haswork(int n, TreeWalk * tw);

static void
hydro_postprocess(int i, TreeWalk * tw);

static void
hydro_ngbiter(
    TreeWalkQueryHydro * I,
    TreeWalkResultHydro * O,
    TreeWalkNgbIterHydro * iter,
    LocalTreeWalk * lv
   );

static void
hydro_copy(int place, TreeWalkQueryHydro * input, TreeWalk * tw);

static void
hydro_reduce(int place, TreeWalkResultHydro * result, enum TreeWalkReduceMode mode, TreeWalk * tw);

/*! This function is the driver routine for the calculation of hydrodynamical
 *  force and rate of change of entropy due to shock heating for all active
 *  particles .
 */
void hydro_force(void)
{
    if(!All.HydroOn)
        return;
    TreeWalk tw[1] = {0};

    tw->ev_label = "HYDRO";
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw->ngbiter = (TreeWalkNgbIterFunction) hydro_ngbiter;
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterHydro);
    tw->haswork = hydro_haswork;
    tw->fill = (TreeWalkFillQueryFunction) hydro_copy;
    tw->reduce = (TreeWalkReduceResultFunction) hydro_reduce;
    tw->postprocess = (TreeWalkProcessFunction) hydro_postprocess;
    tw->UseNodeList = 0;
    tw->query_type_elsize = sizeof(TreeWalkQueryHydro);
    tw->result_type_elsize = sizeof(TreeWalkResultHydro);

    double timeall = 0, timenetwork = 0;
    double timecomp, timecomm, timewait;

    walltime_measure("/Misc");

    /* allocate buffers to arrange communication */

    walltime_measure("/SPH/Hydro/Init");

    treewalk_run(tw, ActiveParticle, NumActiveParticle);

    /* collect some timing information */

    timeall += walltime_measure(WALLTIME_IGNORE);

    timecomp = tw->timecomp1 + tw->timecomp2 + tw->timecomp3;
    timewait = tw->timewait1 + tw->timewait2;
    timecomm = tw->timecommsumm1 + tw->timecommsumm2;

    walltime_add("/SPH/Hydro/Compute", timecomp);
    walltime_add("/SPH/Hydro/Wait", timewait);
    walltime_add("/SPH/Hydro/Comm", timecomm);
    walltime_add("/SPH/Hydro/Misc", timeall - (timecomp + timewait + timecomm + timenetwork));
}

static void
hydro_copy(int place, TreeWalkQueryHydro * input, TreeWalk * tw)
{
    double soundspeed_i;
    const double fac_mu = pow(All.cf.a, 3 * (GAMMA - 1) / 2) / All.cf.a;
    /*Compute predicted velocity*/
    sph_VelPred(place, input->Vel);
    input->Hsml = P[place].Hsml;
    input->Mass = P[place].Mass;
    input->Density = SPHP(place).Density;
#ifdef DENSITY_INDEPENDENT_SPH
    input->EgyRho = SPHP(place).EgyWtDensity;
    input->EntVarPred = EntropyPred(place);
    input->DhsmlDensityFactor = SPHP(place).DhsmlEgyDensityFactor;
#else
    input->DhsmlDensityFactor = SPHP(place).DhsmlDensityFactor;
#endif

    input->Pressure = PressurePred(place);
    input->TimeBin = P[place].TimeBin;
    /* calculation of F1 */
    soundspeed_i = sqrt(GAMMA * input->Pressure / SPHP(place).EOMDensity);
    input->F1 = fabs(SPHP(place).DivVel) /
        (fabs(SPHP(place).DivVel) + SPHP(place).CurlVel +
         0.0001 * soundspeed_i / P[place].Hsml / fac_mu);
}

static void
hydro_reduce(int place, TreeWalkResultHydro * result, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    int k;

    for(k = 0; k < 3; k++)
    {
        TREEWALK_REDUCE(SPHP(place).HydroAccel[k], result->Acc[k]);
    }

    TREEWALK_REDUCE(SPHP(place).DtEntropy, result->DtEntropy);

    P[place].GravCost += All.HydroCostFactor * All.cf.a * result->Ninteractions;

    if(mode == TREEWALK_PRIMARY || SPHP(place).MaxSignalVel < result->MaxSignalVel)
        SPHP(place).MaxSignalVel = result->MaxSignalVel;

}

/*! This function is the 'core' of the SPH force computation. A target
 *  particle is specified which may either be local, or reside in the
 *  communication buffer.
 */
static void
hydro_ngbiter(
    TreeWalkQueryHydro * I,
    TreeWalkResultHydro * O,
    TreeWalkNgbIterHydro * iter,
    LocalTreeWalk * lv
   )
{
    if(iter->base.other == -1) {
        iter->base.Hsml = I->Hsml;
        iter->base.mask = 1;
        iter->base.symmetric = NGB_TREEFIND_SYMMETRIC;

    #ifdef DENSITY_INDEPENDENT_SPH
        iter->soundspeed_i = sqrt(GAMMA * I->Pressure / I->EgyRho);
    #else
        iter->soundspeed_i = sqrt(GAMMA * I->Pressure / I->Density);
    #endif

        /* initialize variables before SPH loop is started */

        O->Acc[0] = O->Acc[1] = O->Acc[2] = O->DtEntropy = 0;
        density_kernel_init(&iter->kernel_i, I->Hsml);

    #ifdef DENSITY_INDEPENDENT_SPH
        iter->p_over_rho2_i = I->Pressure / (I->EgyRho * I->EgyRho);
    #else
        iter->p_over_rho2_i = I->Pressure / (I->Density * I->Density);
    #endif

        O->MaxSignalVel = iter->soundspeed_i;
        return;
    }

    int other = iter->base.other;
    double r2 = iter->base.r2;
    double * dist = iter->base.dist;
    double r = iter->base.r;

    if(P[other].Mass == 0) {
        endrun(-1, "Encoutered zero mass particle during hydro;"
                  " We haven't implemented tracer particles and this shall not happen\n");
    }

#ifdef SFR
#ifdef NOWINDTIMESTEPPING
    if(All.WindOn && HAS(All.WindModel, WIND_DECOUPLE_SPH)) {
        if(P[other].Type == 0)
            if(SPHP(other).DelayTime > 0)  /* ignore the wind particles */
                return;
    }
#endif
#endif
    DensityKernel kernel_j;

    density_kernel_init(&kernel_j, P[other].Hsml);

    if(r2 > 0 && (r2 < iter->kernel_i.HH || r2 < kernel_j.HH))
    {
        double Pressure_j = PressurePred(other);
        double p_over_rho2_j = Pressure_j / (SPHP(other).EOMDensity * SPHP(other).EOMDensity);
        double soundspeed_j;

#ifdef DENSITY_INDEPENDENT_SPH
        soundspeed_j = sqrt(GAMMA * Pressure_j / SPHP(other).EOMDensity);
#else
        soundspeed_j = sqrt(GAMMA * p_over_rho2_j * SPHP(other).Density);
#endif

        double dv[3];
        int d;
        sph_VelPred(other, dv);
        for(d = 0; d < 3; d++) {
            dv[d] = I->Vel[d] - dv[d];
        }

        double vdotr = dotproduct(dist, dv);

        double rho_ij = 0.5 * (I->Density + SPHP(other).Density);
        double vdotr2 = vdotr + All.cf.hubble_a2 * r2;

        double dwk_i = density_kernel_dwk(&iter->kernel_i, r * iter->kernel_i.Hinv);
        double dwk_j = density_kernel_dwk(&kernel_j, r * kernel_j.Hinv);

        double vsig = iter->soundspeed_i + soundspeed_j;


        if(vsig > O->MaxSignalVel)
            O->MaxSignalVel = vsig;

        double visc = 0;

        if(vdotr2 < 0)	/* ... artificial viscosity visc is 0 by default*/
        {
            /*See Gadget-2 paper: eq. 13*/
            const double fac_mu = pow(All.cf.a, 3 * (GAMMA - 1) / 2) / All.cf.a;
            double mu_ij = fac_mu * vdotr2 / r;	/* note: this is negative! */
            vsig -= 3 * mu_ij;


            if(vsig > O->MaxSignalVel)
                O->MaxSignalVel = vsig;

            double f2 =
                fabs(SPHP(other).DivVel) / (fabs(SPHP(other).DivVel) + SPHP(other).CurlVel +
                        0.0001 * soundspeed_j / fac_mu / P[other].Hsml);

            double BulkVisc_ij = All.ArtBulkViscConst;

            /*Gadget-2 paper, eq. 14*/
            visc = 0.25 * BulkVisc_ij * vsig * (-mu_ij) / rho_ij * (I->F1 + f2);
            /* .... end artificial viscosity evaluation */
            /* now make sure that viscous acceleration is not too large */

#ifndef NOVISCOSITYLIMITER
            /*XXX: why is this dloga ?*/
            double dloga = 2 * get_dloga_for_bin(IMAX(I->TimeBin, P[other].TimeBin));
            if(dloga > 0 && (dwk_i + dwk_j) < 0)
            {
                if((I->Mass + P[other].Mass) > 0) {
                    double fac_vsic_fix = All.cf.hubble * pow(All.cf.a, 3 * GAMMA_MINUS1);
                    visc = DMIN(visc, 0.5 * fac_vsic_fix * vdotr2 /
                            (0.5 * (I->Mass + P[other].Mass) * (dwk_i + dwk_j) * r * dloga));
                }
            }
#endif
        }
        double hfc_visc = 0.5 * P[other].Mass * visc * (dwk_i + dwk_j) / r;
#ifdef DENSITY_INDEPENDENT_SPH
        double hfc = hfc_visc;
        /* leading-order term */
        double EntPred = EntropyPred(other);
        hfc += P[other].Mass *
            (dwk_i*iter->p_over_rho2_i*EntPred/I->EntVarPred +
             dwk_j*p_over_rho2_j*I->EntVarPred/EntPred) / r;

        /* enable grad-h corrections only if contrastlimit is non negative */
        if(All.DensityContrastLimit >= 0) {
            double r1 = I->EgyRho / I->Density;
            double r2 = SPHP(other).EgyWtDensity / SPHP(other).Density;
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
            hfc += P[other].Mass *
                (dwk_i*iter->p_over_rho2_i*r1*I->DhsmlDensityFactor +
                 dwk_j*p_over_rho2_j*r2*SPHP(other).DhsmlEgyDensityFactor) / r;
        }
#else
        /* Formulation derived from the Lagrangian */
        double hfc = hfc_visc + P[other].Mass * (iter->p_over_rho2_i *I->DhsmlDensityFactor * dwk_i
                + p_over_rho2_j * SPHP(other).DhsmlDensityFactor * dwk_j) / r;
#endif

#ifdef SFR
        if(All.WindOn && HAS(All.WindModel, WIND_DECOUPLE_SPH)) {
            if(P[other].Type == 0)
                if(SPHP(other).DelayTime > 0)	/* No force by wind particles */
                {
                    hfc = hfc_visc = 0;
                }
        }
#endif

#ifndef NOACCEL
        for(d = 0; d < 3; d ++)
            O->Acc[d] += (-hfc * dist[d]);
#endif

        O->DtEntropy += (0.5 * hfc_visc * vdotr2);

    }
    O->Ninteractions++;
}

static int
hydro_haswork(int i, TreeWalk * tw)
{
    return P[i].Type == 0;
}

static void
hydro_postprocess(int i, TreeWalk * tw)
{
    if(P[i].Type == 0)
    {
        /* Translate energy change rate into entropy change rate */
        SPHP(i).DtEntropy *= GAMMA_MINUS1 / (All.cf.hubble_a2 * pow(SPHP(i).EOMDensity, GAMMA_MINUS1));

#ifdef SFR
        /* if we have winds, we decouple particles briefly if delaytime>0 */
        if(All.WindOn && HAS(All.WindModel, WIND_DECOUPLE_SPH)) {
            if(SPHP(i).DelayTime > 0)
            {
                int k;
                for(k = 0; k < 3; k++)
                    SPHP(i).HydroAccel[k] = 0;

                SPHP(i).DtEntropy = 0;

#ifdef NOWINDTIMESTEPPING
                SPHP(i).MaxSignalVel = 2 * sqrt(GAMMA * PressurePred(i) / SPHP(i).Density);
#else
                double windspeed = All.WindSpeed * All.cf.a;
                const double fac_mu = pow(All.cf.a, 3 * (GAMMA - 1) / 2) / All.cf.a;
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
