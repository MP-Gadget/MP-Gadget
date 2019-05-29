#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>


#include "allvars.h"
#include "slotsmanager.h"
#include "treewalk.h"
#include "densitykernel.h"
#include "timestep.h"
#include "hydra.h"
#include "winds.h"
#include "utils.h"

/*! \file hydra.c
 *  \brief Computation of SPH forces and rate of entropy generation
 *
 *  This file contains the "second SPH loop", where the SPH forces are
 *  computed, and where the rate of change of entropy due to the shock heating
 *  (via artificial viscosity) is computed.
 */

MyFloat SPH_EOMDensity(int i)
{
    if(All.DensityIndependentSphOn)
        return SPHP(i).EgyWtDensity;
    else
        return SPHP(i).Density;
}

MyFloat SPH_DhsmlDensityFactor(int i)
{
    if(All.DensityIndependentSphOn)
        return SPHP(i).DhsmlEgyDensityFactor;
    else
        return SPHP(i).DhsmlDensityFactor;
}

double
PressurePred(int PI)
{
    MyFloat EOMDensity;
    if(All.DensityIndependentSphOn)
        EOMDensity = SphP[PI].EgyWtDensity;
    else
        EOMDensity = SphP[PI].Density;
    return pow(SphP[PI].EntVarPred * EOMDensity, GAMMA);
}

struct HydraPriv {
    double * PressurePred;
};

#define HYDRA_GET_PRIV(tw) ((struct HydraPriv*) ((tw)->priv))

typedef struct {
    TreeWalkQueryBase base;
    /* These are only used for DensityIndependentSphOn*/
    MyFloat EgyRho;
    MyFloat EntVarPred;

    double Vel[3];
    MyFloat Hsml;
    MyFloat Mass;
    MyFloat Density;
    MyFloat Pressure;
    MyFloat F1;
    MyFloat SPH_DhsmlDensityFactor;
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

/* Time-dependent constant factors, brought out here because
 * they need an expensive pow().*/
static double fac_mu;
static double fac_vsic_fix;

/*! This function is the driver routine for the calculation of hydrodynamical
 *  force and rate of change of entropy due to shock heating for all active
 *  particles .
 */
void hydro_force(ForceTree * tree)
{
    int i;
    if(!All.HydroOn)
        return;
    TreeWalk tw[1] = {{0}};

    struct HydraPriv priv[1];

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
    tw->tree = tree;
    tw->priv = priv;

    /* Cache the pressure for speed*/
    HYDRA_GET_PRIV(tw)->PressurePred = (double *) mymalloc("PressurePred", SlotsManager->info[0].size * sizeof(double));

    #pragma omp parallel for
    for(i = 0; i < SlotsManager->info[0].size; i++)
        HYDRA_GET_PRIV(tw)->PressurePred[i] = PressurePred(i);

    double timeall = 0, timenetwork = 0;
    double timecomp, timecomm, timewait;

    walltime_measure("/Misc");

    /* Initialize some time factors*/
    fac_mu = pow(All.cf.a, 3 * (GAMMA - 1) / 2) / All.cf.a;
    fac_vsic_fix = All.cf.hubble * pow(All.cf.a, 3 * GAMMA_MINUS1);

    treewalk_run(tw, ActiveParticle, NumActiveParticle);

    myfree(HYDRA_GET_PRIV(tw)->PressurePred);
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
    /*Compute predicted velocity*/
    input->Vel[0] = SPHP(place).VelPred[0];
    input->Vel[1] = SPHP(place).VelPred[1];
    input->Vel[2] = SPHP(place).VelPred[2];
    input->Hsml = P[place].Hsml;
    input->Mass = P[place].Mass;
    input->Density = SPHP(place).Density;

    if(All.DensityIndependentSphOn) {
        input->EgyRho = SPHP(place).EgyWtDensity;
        input->EntVarPred = SPHP(place).EntVarPred;
    }

    input->SPH_DhsmlDensityFactor = SPH_DhsmlDensityFactor(place);

    input->Pressure = HYDRA_GET_PRIV(tw)->PressurePred[P[place].PI];
    input->TimeBin = P[place].TimeBin;
    /* calculation of F1 */
    soundspeed_i = sqrt(GAMMA * input->Pressure / SPH_EOMDensity(place));
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

        if(All.DensityIndependentSphOn)
            iter->soundspeed_i = sqrt(GAMMA * I->Pressure / I->EgyRho);
        else
            iter->soundspeed_i = sqrt(GAMMA * I->Pressure / I->Density);

        /* initialize variables before SPH loop is started */

        O->Acc[0] = O->Acc[1] = O->Acc[2] = O->DtEntropy = 0;
        density_kernel_init(&iter->kernel_i, I->Hsml);

        if(All.DensityIndependentSphOn)
            iter->p_over_rho2_i = I->Pressure / (I->EgyRho * I->EgyRho);
        else
            iter->p_over_rho2_i = I->Pressure / (I->Density * I->Density);

        O->MaxSignalVel = iter->soundspeed_i;
        return;
    }

    int other = iter->base.other;
    double r2 = iter->base.r2;
    double * dist = iter->base.dist;
    double r = iter->base.r;

    if(P[other].Mass == 0) {
        endrun(12, "Encountered zero mass particle during hydro;"
                  " We haven't implemented tracer particles and this shall not happen\n");
    }

    DensityKernel kernel_j;

    density_kernel_init(&kernel_j, P[other].Hsml);

    if(r2 > 0 && (r2 < iter->kernel_i.HH || r2 < kernel_j.HH))
    {
        double Pressure_j = HYDRA_GET_PRIV(lv->tw)->PressurePred[P[other].PI];
        double p_over_rho2_j = Pressure_j / (SPH_EOMDensity(other) * SPH_EOMDensity(other));
        double soundspeed_j = sqrt(GAMMA * Pressure_j / SPH_EOMDensity(other));

        double dv[3];
        int d;
        for(d = 0; d < 3; d++) {
            dv[d] = I->Vel[d] - SPHP(other).VelPred[d];
        }

        double vdotr = dotproduct(dist, dv);
        double vdotr2 = vdotr + All.cf.hubble_a2 * r2;

        double dwk_i = density_kernel_dwk(&iter->kernel_i, r * iter->kernel_i.Hinv);
        double dwk_j = density_kernel_dwk(&kernel_j, r * kernel_j.Hinv);

        double visc = 0;

        if(vdotr2 < 0)	/* ... artificial viscosity visc is 0 by default*/
        {
            /*See Gadget-2 paper: eq. 13*/
            const double mu_ij = fac_mu * vdotr2 / r;	/* note: this is negative! */
            const double rho_ij = 0.5 * (I->Density + SPHP(other).Density);
            double vsig = iter->soundspeed_i + soundspeed_j;

            vsig -= 3 * mu_ij;

            if(vsig > O->MaxSignalVel)
                O->MaxSignalVel = vsig;

            const double f2 = fabs(SPHP(other).DivVel) / (fabs(SPHP(other).DivVel) +
                    SPHP(other).CurlVel + 0.0001 * soundspeed_j / fac_mu / P[other].Hsml);

            /*Gadget-2 paper, eq. 14*/
            visc = 0.25 * All.ArtBulkViscConst * vsig * (-mu_ij) / rho_ij * (I->F1 + f2);
            /* .... end artificial viscosity evaluation */
            /* now make sure that viscous acceleration is not too large */

            /*XXX: why is this dloga ?*/
            double dloga = 2 * get_dloga_for_bin(IMAX(I->TimeBin, P[other].TimeBin));
            if(dloga > 0 && (dwk_i + dwk_j) < 0)
            {
                if((I->Mass + P[other].Mass) > 0) {
                    visc = DMIN(visc, 0.5 * fac_vsic_fix * vdotr2 /
                            (0.5 * (I->Mass + P[other].Mass) * (dwk_i + dwk_j) * r * dloga));
                }
            }
        }
        double hfc_visc = 0.5 * P[other].Mass * visc * (dwk_i + dwk_j) / r;
        double hfc = hfc_visc;
        double r1 = 1, r2 = 1;

        if(All.DensityIndependentSphOn) {
            /*This enables the grad-h corrections*/
            r1 = 0, r2 = 0;
            /* leading-order term */
            double EntOther = SPHP(other).EntVarPred;

            hfc += P[other].Mass *
                (dwk_i*iter->p_over_rho2_i*EntOther/I->EntVarPred +
                dwk_j*p_over_rho2_j*I->EntVarPred/EntOther) / r;

            /* enable grad-h corrections only if contrastlimit is non negative */
            if(All.DensityContrastLimit >= 0) {
                r1 = I->EgyRho / I->Density;
                r2 = SPHP(other).EgyWtDensity / SPHP(other).Density;
                if(All.DensityContrastLimit > 0) {
                    /* apply the limit if it is enabled > 0*/
                    r1 = DMIN(r1, All.DensityContrastLimit);
                    r2 = DMIN(r2, All.DensityContrastLimit);
                }
            }
        }

        /* grad-h corrections: enabled if DensityIndependentSphOn = 0, or DensityConstrastLimit >= 0 */
        /* Formulation derived from the Lagrangian */
        hfc += P[other].Mass * (iter->p_over_rho2_i*I->SPH_DhsmlDensityFactor * dwk_i * r1
                 + p_over_rho2_j*SPH_DhsmlDensityFactor(other) * dwk_j * r2) / r;

        /* No force by wind particles */
        if(All.WindOn && winds_is_particle_decoupled(other)) {
            hfc = hfc_visc = 0;
        }

        for(d = 0; d < 3; d ++)
            O->Acc[d] += (-hfc * dist[d]);

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
        SPHP(i).DtEntropy *= GAMMA_MINUS1 / (All.cf.hubble_a2 * pow(SPH_EOMDensity(i), GAMMA_MINUS1));

        /* if we have winds, we decouple particles briefly if delaytime>0 */
        if(All.WindOn && winds_is_particle_decoupled(i))
        {
            winds_decoupled_hydro(i, All.cf.a);
        }
    }
}
