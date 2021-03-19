#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "physconst.h"
#include "walltime.h"
#include "slotsmanager.h"
#include "treewalk.h"
#include "density.h"
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

static struct hydro_params
{
    /* Enables density independent (Pressure-entropy) SPH */
    int DensityIndependentSphOn;
    /* limit of density contrast ratio for hydro force calculation (only effective with Density Indep. Sph) */
    double DensityContrastLimit;
    /*!< Sets the parameter \f$\alpha\f$ of the artificial viscosity */
    double ArtBulkViscConst;
} HydroParams;

/*Set the parameters of the hydro module*/
void
set_hydro_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        HydroParams.ArtBulkViscConst = param_get_double(ps, "ArtBulkViscConst");
        HydroParams.DensityContrastLimit = param_get_double(ps, "DensityContrastLimit");
        HydroParams.DensityIndependentSphOn= param_get_int(ps, "DensityIndependentSphOn");
    }
    MPI_Bcast(&HydroParams, sizeof(struct hydro_params), MPI_BYTE, 0, MPI_COMM_WORLD);
}

int DensityIndependentSphOn(void)
{
    return HydroParams.DensityIndependentSphOn;
}

MyFloat SPH_EOMDensity(const struct sph_particle_data * const pi)
{
    if(HydroParams.DensityIndependentSphOn)
        return pi->EgyWtDensity;
    else
        return pi->Density;
}

/* Compute pressure using the predicted density (EgyWtDensity if Pressure-Entropy SPH,
 * Density otherwise) and predicted entropy*/
static double
PressurePred(MyFloat EOMDensityPred, double EntVarPred)
{
    return pow(EntVarPred * EOMDensityPred, GAMMA);
}

struct HydraPriv {
    double * PressurePred;
    struct sph_pred_data * SPH_predicted;
    /* Time-dependent constant factors, brought out here because
     * they need an expensive pow().*/
    double fac_mu;
    double fac_vsic_fix;
    double hubble_a2;
    double atime;
    int WindOn;
    DriftKickTimes const * times;
    double MinEgySpec;
    double FgravkickB;
    double gravkicks[TIMEBINS+1];
    double hydrokicks[TIMEBINS+1];
    double drifts[TIMEBINS+1];
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
    MyFloat dloga;
} TreeWalkQueryHydro;

typedef struct {
    TreeWalkResultBase base;
    MyFloat Acc[3];
    MyFloat DtEntropy;
    MyFloat MaxSignalVel;
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
void
hydro_force(const ActiveParticles * act, int WindOn, const double hubble, const double atime, struct sph_pred_data * SPH_predicted, double MinEgySpec, const DriftKickTimes times,  Cosmology * CP, const ForceTree * const tree)
{
    int i;
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
    tw->query_type_elsize = sizeof(TreeWalkQueryHydro);
    tw->result_type_elsize = sizeof(TreeWalkResultHydro);
    tw->tree = tree;
    tw->priv = priv;

    if(!tree->hmax_computed_flag)
        endrun(5, "Hydro called before hmax computed\n");
    /* Cache the pressure for speed*/
    HYDRA_GET_PRIV(tw)->SPH_predicted = SPH_predicted;
    HYDRA_GET_PRIV(tw)->PressurePred = (double *) mymalloc("PressurePred", SlotsManager->info[0].size * sizeof(double));
    if(SPH_predicted->store_inactive_predict) {
        memset(HYDRA_GET_PRIV(tw)->PressurePred, 0, SlotsManager->info[0].size * sizeof(double));
    }
    /* Compute pressure for active particles*/
    if(act->ActiveParticle) {
        #pragma omp parallel for
        for(i = 0; i < act->NumActiveParticle; i++) {
            int p_i = act->ActiveParticle[i];
            if(P[p_i].Type != 0)
                continue;
            int pi = P[p_i].PI;
            HYDRA_GET_PRIV(tw)->PressurePred[pi] = PressurePred(SPH_EOMDensity(&SphP[pi]), SPH_predicted->EntVarPred[pi]);
        }
    }
    else{
        /* Do it in slot order for memory locality*/
        #pragma omp parallel for
        for(i = 0; i < SlotsManager->info[0].size; i++)
            HYDRA_GET_PRIV(tw)->PressurePred[i] = PressurePred(SPH_EOMDensity(&SphP[i]), SPH_predicted->EntVarPred[i]);
    }


    double timeall = 0, timenetwork = 0;
    double timecomp, timecomm, timewait;

    walltime_measure("/SPH/Hydro/Init");

    /* Initialize some time factors*/
    HYDRA_GET_PRIV(tw)->fac_mu = pow(atime, 3 * (GAMMA - 1) / 2) / atime;
    HYDRA_GET_PRIV(tw)->fac_vsic_fix = hubble * pow(atime, 3 * GAMMA_MINUS1);
    HYDRA_GET_PRIV(tw)->WindOn = WindOn;
    HYDRA_GET_PRIV(tw)->atime = atime;
    HYDRA_GET_PRIV(tw)->hubble_a2 = hubble * atime * atime;
    priv->MinEgySpec = MinEgySpec;
    priv->times = &times;
    priv->FgravkickB = get_exact_gravkick_factor(CP, times.PM_kick, times.Ti_Current);
    memset(priv->gravkicks, 0, sizeof(priv->gravkicks[0])*(TIMEBINS+1));
    memset(priv->hydrokicks, 0, sizeof(priv->hydrokicks[0])*(TIMEBINS+1));
    memset(priv->drifts, 0, sizeof(priv->drifts[0])*(TIMEBINS+1));
    /* Compute the factors to move a current kick times velocity to the drift time velocity.
     * We need to do the computation for all timebins up to the maximum because even inactive
     * particles may have interactions. */
    #pragma omp parallel for
    for(i = times.mintimebin; i <= TIMEBINS; i++)
    {
        priv->gravkicks[i] = get_exact_gravkick_factor(CP, times.Ti_kick[i], times.Ti_Current);
        priv->hydrokicks[i] = get_exact_hydrokick_factor(CP, times.Ti_kick[i], times.Ti_Current);
        /* For density: last active drift time is Ti_kick - 1/2 timestep as the kick time is half a timestep ahead.
         * For active particles no density drift is needed.*/
        if(!is_timebin_active(i, times.Ti_Current))
            priv->drifts[i] = get_exact_drift_factor(CP, times.Ti_lastactivedrift[i], times.Ti_Current);
    }

    treewalk_run(tw, act->ActiveParticle, act->NumActiveParticle);

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
    MyFloat * velpred = HYDRA_GET_PRIV(tw)->SPH_predicted->VelPred;
    /*Compute predicted velocity*/
    input->Vel[0] = velpred[3 * P[place].PI];
    input->Vel[1] = velpred[3 * P[place].PI + 1];
    input->Vel[2] = velpred[3 * P[place].PI + 2];
    input->Hsml = P[place].Hsml;
    input->Mass = P[place].Mass;
    input->Density = SPHP(place).Density;

    if(HydroParams.DensityIndependentSphOn) {
        input->EgyRho = SPHP(place).EgyWtDensity;
        MyFloat * entvarpred = HYDRA_GET_PRIV(tw)->SPH_predicted->EntVarPred;

        input->EntVarPred = entvarpred[P[place].PI];
    }

    input->SPH_DhsmlDensityFactor = SPHP(place).DhsmlEgyDensityFactor;

    input->Pressure = HYDRA_GET_PRIV(tw)->PressurePred[P[place].PI];
    input->dloga = get_dloga_for_bin(P[place].TimeBin, HYDRA_GET_PRIV(tw)->times->Ti_Current);
    /* calculation of F1 */
    soundspeed_i = sqrt(GAMMA * input->Pressure / SPH_EOMDensity(&SPHP(place)));
    input->F1 = fabs(SPHP(place).DivVel) /
        (fabs(SPHP(place).DivVel) + SPHP(place).CurlVel +
         0.0001 * soundspeed_i / P[place].Hsml / HYDRA_GET_PRIV(tw)->fac_mu);
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

    if(mode == TREEWALK_PRIMARY || SPHP(place).MaxSignalVel < result->MaxSignalVel)
        SPHP(place).MaxSignalVel = result->MaxSignalVel;

}

/* Find the density predicted forward to the current drift time.
 * The Density in the SPHP struct is evaluated at the last time
 * the particle was active. Good for both EgyWtDensity and Density,
 * cube of the change in Hsml in drift.c. */
double
SPH_DensityPred(MyFloat Density, MyFloat DivVel, double dtdrift)
{
    /* Note minus sign!*/
    return Density - DivVel * Density * dtdrift;
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

        if(HydroParams.DensityIndependentSphOn)
            iter->soundspeed_i = sqrt(GAMMA * I->Pressure / I->EgyRho);
        else
            iter->soundspeed_i = sqrt(GAMMA * I->Pressure / I->Density);

        /* initialize variables before SPH loop is started */

        O->Acc[0] = O->Acc[1] = O->Acc[2] = O->DtEntropy = 0;
        density_kernel_init(&iter->kernel_i, I->Hsml, GetDensityKernelType());

        if(HydroParams.DensityIndependentSphOn)
            iter->p_over_rho2_i = I->Pressure / (I->EgyRho * I->EgyRho);
        else
            iter->p_over_rho2_i = I->Pressure / (I->Density * I->Density);

        O->MaxSignalVel = iter->soundspeed_i;
        return;
    }

    int other = iter->base.other;
    double rsq = iter->base.r2;
    double * dist = iter->base.dist;
    double r = iter->base.r;

    if(P[other].Mass == 0) {
        endrun(12, "Encountered zero mass particle during hydro;"
                  " We haven't implemented tracer particles and this shall not happen\n");
    }

    /* Wind particles do not interact hydrodynamically: don't produce hydro acceleration
     * or change the signalvel.*/
    if(HYDRA_GET_PRIV(lv->tw)->WindOn && winds_is_particle_decoupled(other))
        return;

    DensityKernel kernel_j;

    density_kernel_init(&kernel_j, P[other].Hsml, GetDensityKernelType());

    /* Check we are within the density kernel*/
    if(rsq <= 0 || !(rsq < iter->kernel_i.HH || rsq < kernel_j.HH))
        return;

    struct HydraPriv * priv = HYDRA_GET_PRIV(lv->tw);

    /* Predict densities. Note that for active timebins the density is up to date so SPH_DensityPred is just returns the current densities.
     * This improves on the technique used in Gadget-2 by being a linear prediction that does not become pathological in deep timebins.*/
    const int bin = P[other].TimeBin;
    const double density_j = SPH_DensityPred(SPHP(other).Density, SPHP(other).DivVel, priv->drifts[bin]);
    const double eomdensity = SPH_DensityPred(SPH_EOMDensity(&SPHP(other)), SPHP(other).DivVel, priv->drifts[bin]);;

    double EntVarPred, Pressure_j;
    MyFloat VelPred[3];

    if(priv->SPH_predicted->store_inactive_predict) {
        #pragma omp atomic read
        EntVarPred = priv->SPH_predicted->EntVarPred[P[other].PI];
        /* Lazily compute the predicted quantities. We need to do this again here, even though we do it in density,
        * because this treewalk is symmetric and that one is asymmetric. In density() hmax has not been computed
        * yet so we cannot merge them. We can do this
        * with minimal locking since nothing happens should we compute them twice.
        * Zero can be the special value since there should never be zero entropy.*/
        if(EntVarPred == 0) {
            double a3inv = pow(priv->atime, -3);
            double dloga = dloga_from_dti(priv->times->Ti_Current - priv->times->Ti_kick[bin], priv->times->Ti_Current);
            EntVarPred = SPH_EntVarPred(P[other].PI, priv->MinEgySpec, a3inv, dloga);
            SPH_VelPred(other, VelPred, priv->FgravkickB, priv->gravkicks[bin], priv->hydrokicks[bin]);
            /* Note this goes first to avoid threading issues: EntVarPred will only be set after this is done.
                * The worst that can happen is that some data points get copied twice.*/
            int i;
            for(i = 0; i < 3; i++) {
                #pragma omp atomic write
                priv->SPH_predicted->VelPred[3 * P[other].PI + i] = VelPred[i];
            }
            #pragma omp atomic write
            priv->SPH_predicted->EntVarPred[P[other].PI] = EntVarPred;
        }
        else {
            int i;
            for(i = 0; i < 3; i++)
                VelPred[i] = priv->SPH_predicted->VelPred[3 * P[other].PI + i];
        }
        /* Compute pressure lazily*/
        #pragma omp atomic read
        Pressure_j = HYDRA_GET_PRIV(lv->tw)->PressurePred[P[other].PI];
        if(Pressure_j == 0) {
            Pressure_j = PressurePred(eomdensity, EntVarPred);
            #pragma omp atomic write
            priv->PressurePred[P[other].PI] = Pressure_j;
        }
    }
    else {
        double a3inv = pow(priv->atime, -3);
        double dloga = dloga_from_dti(priv->times->Ti_Current - priv->times->Ti_kick[bin], priv->times->Ti_Current);
        SPH_VelPred(other, VelPred, priv->FgravkickB, priv->gravkicks[bin], priv->hydrokicks[bin]);
        EntVarPred = SPH_EntVarPred(P[other].PI, priv->MinEgySpec, a3inv, dloga);
        Pressure_j = PressurePred(eomdensity, EntVarPred);
    }

    double p_over_rho2_j = Pressure_j / (eomdensity * eomdensity);
    double soundspeed_j = sqrt(GAMMA * Pressure_j / eomdensity);

    double dv[3];
    int d;
    for(d = 0; d < 3; d++) {
        dv[d] = I->Vel[d] - VelPred[d];
    }

    double vdotr = dotproduct(dist, dv);
    double vdotr2 = vdotr + HYDRA_GET_PRIV(lv->tw)->hubble_a2 * rsq;

    double dwk_i = density_kernel_dwk(&iter->kernel_i, r * iter->kernel_i.Hinv);
    double dwk_j = density_kernel_dwk(&kernel_j, r * kernel_j.Hinv);

    double visc = 0;

    if(vdotr2 < 0)	/* ... artificial viscosity visc is 0 by default*/
    {
        /*See Gadget-2 paper: eq. 13*/
        const double mu_ij = HYDRA_GET_PRIV(lv->tw)->fac_mu * vdotr2 / r;	/* note: this is negative! */
        const double rho_ij = 0.5 * (I->Density + density_j);
        double vsig = iter->soundspeed_i + soundspeed_j;

        vsig -= 3 * mu_ij;

        if(vsig > O->MaxSignalVel)
            O->MaxSignalVel = vsig;

        /* Note this uses the CurlVel of an inactive particle, which is not at the present drift time*/
        const double f2 = fabs(SPHP(other).DivVel) / (fabs(SPHP(other).DivVel) +
                SPHP(other).CurlVel + 0.0001 * soundspeed_j / HYDRA_GET_PRIV(lv->tw)->fac_mu / P[other].Hsml);

        /*Gadget-2 paper, eq. 14*/
        visc = 0.25 * HydroParams.ArtBulkViscConst * vsig * (-mu_ij) / rho_ij * (I->F1 + f2);
        /* .... end artificial viscosity evaluation */
        /* now make sure that viscous acceleration is not too large */

        /*XXX: why is this dloga ?*/
        double dloga = 2 * DMAX(I->dloga, get_dloga_for_bin(P[other].TimeBin, HYDRA_GET_PRIV(lv->tw)->times->Ti_Current));
        if(dloga > 0 && (dwk_i + dwk_j) < 0)
        {
            if((I->Mass + P[other].Mass) > 0) {
                visc = DMIN(visc, 0.5 * HYDRA_GET_PRIV(lv->tw)->fac_vsic_fix * vdotr2 /
                        (0.5 * (I->Mass + P[other].Mass) * (dwk_i + dwk_j) * r * dloga));
            }
        }
    }
    const double hfc_visc = 0.5 * P[other].Mass * visc * (dwk_i + dwk_j) / r;
    double hfc = hfc_visc;
    double rr1 = 1, rr2 = 1;

    if(HydroParams.DensityIndependentSphOn) {
        /*This enables the grad-h corrections*/
        rr1 = 0, rr2 = 0;
        hfc += P[other].Mass *
            (dwk_i*iter->p_over_rho2_i*EntVarPred/I->EntVarPred +
            dwk_j*p_over_rho2_j*I->EntVarPred/EntVarPred) / r;

        /* enable grad-h corrections only if contrastlimit is non negative */
        if(HydroParams.DensityContrastLimit >= 0) {
            rr1 = I->EgyRho / I->Density;
            rr2 = eomdensity / density_j;
            if(HydroParams.DensityContrastLimit > 0) {
                /* apply the limit if it is enabled > 0*/
                rr1 = DMIN(rr1, HydroParams.DensityContrastLimit);
                rr2 = DMIN(rr2, HydroParams.DensityContrastLimit);
            }
        }
    }

    /* grad-h corrections: enabled if DensityIndependentSphOn = 0, or DensityConstrastLimit >= 0 */
    /* Formulation derived from the Lagrangian */
    hfc += P[other].Mass * (iter->p_over_rho2_i*I->SPH_DhsmlDensityFactor * dwk_i * rr1
                + p_over_rho2_j*SPHP(other).DhsmlEgyDensityFactor * dwk_j * rr2) / r;

    for(d = 0; d < 3; d ++)
        O->Acc[d] += (-hfc * dist[d]);

    O->DtEntropy += (0.5 * hfc_visc * vdotr2);

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
        SPHP(i).DtEntropy *= GAMMA_MINUS1 / (HYDRA_GET_PRIV(tw)->hubble_a2 * pow(SPH_EOMDensity(&SPHP(i)), GAMMA_MINUS1));

        /* if we have winds, we decouple particles briefly if delaytime>0 */
        if(HYDRA_GET_PRIV(tw)->WindOn && winds_is_particle_decoupled(i))
        {
            winds_decoupled_hydro(i, HYDRA_GET_PRIV(tw)->atime);
        }
    }
}
