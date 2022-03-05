#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <omp.h>

#include "physconst.h"
#include "walltime.h"
#include "cooling.h"
#include "density.h"
#include "treewalk.h"
#include "timefac.h"
#include "slotsmanager.h"
#include "timestep.h"
#include "utils.h"
#include "gravity.h"
#include "winds.h"

static struct density_params DensityParams;

/*Set cooling module parameters from a cooling_params struct for the tests*/
void
set_densitypar(struct density_params dp)
{
    DensityParams = dp;
}

/*Set the parameters of the density module*/
void
set_density_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        DensityParams.DensityKernelType = param_get_enum(ps, "DensityKernelType");
        DensityParams.MaxNumNgbDeviation = param_get_double(ps, "MaxNumNgbDeviation");
        DensityParams.DensityResolutionEta = param_get_double(ps, "DensityResolutionEta");
        DensityParams.MinGasHsmlFractional = param_get_double(ps, "MinGasHsmlFractional");

        DensityKernel kernel;
        density_kernel_init(&kernel, 1.0, DensityParams.DensityKernelType);
        message(1, "The Density Kernel type is %s\n", kernel.name);
        message(1, "The Density resolution is %g * mean separation, or %g neighbours\n",
                    DensityParams.DensityResolutionEta, GetNumNgb(GetDensityKernelType()));
        /*These two look like black hole parameters but they are really neighbour finding parameters*/
        DensityParams.BlackHoleNgbFactor = param_get_double(ps, "BlackHoleNgbFactor");
        DensityParams.BlackHoleMaxAccretionRadius = param_get_double(ps, "BlackHoleMaxAccretionRadius");
    }
    MPI_Bcast(&DensityParams, sizeof(struct density_params), MPI_BYTE, 0, MPI_COMM_WORLD);
}

double
GetNumNgb(enum DensityKernelType KernelType)
{
    DensityKernel kernel;
    density_kernel_init(&kernel, 1.0, KernelType);
    return density_kernel_desnumngb(&kernel, DensityParams.DensityResolutionEta);
}

enum DensityKernelType
GetDensityKernelType(void)
{
    return DensityParams.DensityKernelType;
}

/* The evolved entropy at drift time: evolved dlog a.
 * Used to predict pressure and entropy for SPH */
MyFloat
SPH_EntVarPred(int PI, double MinEgySpec, double a3inv, double dloga)
{
        double EntVarPred = SphP[PI].Entropy + SphP[PI].DtEntropy * dloga;
        /*Entropy limiter for the predicted entropy: makes sure entropy stays positive. */
        if(dloga > 0 && EntVarPred < 0.5*SphP[PI].Entropy)
            EntVarPred = 0.5 * SphP[PI].Entropy;
        const double enttou = pow(SphP[PI].Density * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1;
        if(EntVarPred < MinEgySpec / enttou)
            EntVarPred = MinEgySpec / enttou;
        EntVarPred = pow(EntVarPred, 1/GAMMA);
        return EntVarPred;
}

/* Get the predicted velocity for a particle
 * at the current Force computation time ti,
 * which always coincides with the Drift inttime.
 * For hydro forces.*/
void
SPH_VelPred(int i, MyFloat * VelPred, const double FgravkickB, double gravkick, double hydrokick)
{
    int j;
    for(j = 0; j < 3; j++) {
        VelPred[j] = P[i].Vel[j] + gravkick * P[i].GravAccel[j]
            + P[i].GravPM[j] * FgravkickB + hydrokick * SPHP(i).HydroAccel[j];
    }
}

/*! Structure for communication during the density computation. Holds data that is sent to other processors.
*/
typedef struct {
    TreeWalkNgbIterBase base;
    DensityKernel kernel;
    double kernel_volume;
} TreeWalkNgbIterDensity;

typedef struct
{
    TreeWalkQueryBase base;
    double Vel[3];
    MyFloat Hsml;
    int Type;
    int alignment;
} TreeWalkQueryDensity;

typedef struct {
    TreeWalkResultBase base;

    /*These are only used for density independent SPH*/
    MyFloat EgyRho;
    MyFloat DhsmlEgyDensity;

    MyFloat Rho;
    MyFloat DhsmlDensity;
    MyFloat Ngb;
    MyFloat Div;
    MyFloat Rot[3];
    /*Only used if sfr_need_to_compute_sph_grad_rho is true*/
    MyFloat GradRho[3];
} TreeWalkResultDensity;

struct DensityPriv {
    /* Predicted quantities computed during for density and reused during hydro.*/
    struct sph_pred_data * SPH_predicted;
    /* The gradient of the density, used sometimes during star formation.
     * May be NULL.*/
    MyFloat * GradRho;
    /* Current number of neighbours*/
    MyFloat *NumNgb;
    /* Lower and upper bounds on smoothing length*/
    MyFloat *Left, *Right;
    MyFloat (*Rot)[3];
    /* This is the DhsmlDensityFactor for the pure density,
     * not the entropy weighted density.
     * If DensityIndependentSphOn = 0 then DhsmlEgyDensityFactor and DhsmlDensityFactor
     * are the same and this is not used.
     * If DensityIndependentSphOn = 1 then this is used to set DhsmlEgyDensityFactor.*/
    MyFloat * DhsmlDensityFactor;
    int update_hsml;
    int DoEgyDensity;
    /*!< Desired number of SPH neighbours */
    double DesNumNgb;
    /*!< minimum allowed SPH smoothing length */
    double MinGasHsml;
    /* Are there potentially black holes?*/
    int BlackHoleOn;

    /* For computing the predicted quantities dynamically during the treewalk.*/
    double a3inv;
    double MinEgySpec;
    DriftKickTimes const * times;
    double FgravkickB;
    double gravkicks[TIMEBINS+1];
    double hydrokicks[TIMEBINS+1];
};

#define DENSITY_GET_PRIV(tw) ((struct DensityPriv*) ((tw)->priv))

static void
density_ngbiter(
        TreeWalkQueryDensity * I,
        TreeWalkResultDensity * O,
        TreeWalkNgbIterDensity * iter,
        LocalTreeWalk * lv);

static int density_haswork(int n, TreeWalk * tw);
static void density_postprocess(int i, TreeWalk * tw);
static void density_check_neighbours(int i, TreeWalk * tw);

static void density_reduce(int place, TreeWalkResultDensity * remote, enum TreeWalkReduceMode mode, TreeWalk * tw);
static void density_copy(int place, TreeWalkQueryDensity * I, TreeWalk * tw);

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
void
density(const ActiveParticles * act, int update_hsml, int DoEgyDensity, int BlackHoleOn, double MinEgySpec, const DriftKickTimes times, Cosmology * CP, struct sph_pred_data * SPH_predicted, MyFloat * GradRho, const ForceTree * const tree)
{
    TreeWalk tw[1] = {{0}};
    struct DensityPriv priv[1];

    tw->ev_label = "DENSITY";
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_nolist_ngbiter;
    tw->NoNgblist = 1;
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterDensity);
    tw->ngbiter = (TreeWalkNgbIterFunction) density_ngbiter;
    tw->haswork = density_haswork;
    tw->fill = (TreeWalkFillQueryFunction) density_copy;
    tw->reduce = (TreeWalkReduceResultFunction) density_reduce;
    tw->postprocess = (TreeWalkProcessFunction) density_postprocess;
    tw->query_type_elsize = sizeof(TreeWalkQueryDensity);
    tw->result_type_elsize = sizeof(TreeWalkResultDensity);
    tw->priv = priv;
    tw->tree = tree;

    int i;

    double timeall = 0;
    double timecomp, timecomm, timewait;
    const double atime = exp(loga_from_ti(times.Ti_Current));

    walltime_measure("/Misc");

    DENSITY_GET_PRIV(tw)->Left = (MyFloat *) mymalloc("DENS_PRIV->Left", PartManager->NumPart * sizeof(MyFloat));
    DENSITY_GET_PRIV(tw)->Right = (MyFloat *) mymalloc("DENS_PRIV->Right", PartManager->NumPart * sizeof(MyFloat));
    DENSITY_GET_PRIV(tw)->NumNgb = (MyFloat *) mymalloc("DENS_PRIV->NumNgb", PartManager->NumPart * sizeof(MyFloat));
    DENSITY_GET_PRIV(tw)->Rot = (MyFloat (*) [3]) mymalloc("DENS_PRIV->Rot", SlotsManager->info[0].size * sizeof(priv->Rot[0]));
    /* This one stores the gradient for h finding. The factor stored in SPHP->DhsmlEgyDensityFactor depends on whether PE SPH is enabled.*/
    DENSITY_GET_PRIV(tw)->DhsmlDensityFactor = (MyFloat *) mymalloc("DENSITY_GET_PRIV(tw)->DhsmlDensity", PartManager->NumPart * sizeof(MyFloat));

    DENSITY_GET_PRIV(tw)->update_hsml = update_hsml;
    DENSITY_GET_PRIV(tw)->DoEgyDensity = DoEgyDensity;

    DENSITY_GET_PRIV(tw)->DesNumNgb = GetNumNgb(DensityParams.DensityKernelType);
    DENSITY_GET_PRIV(tw)->MinGasHsml = DensityParams.MinGasHsmlFractional * (FORCE_SOFTENING(1, 1)/2.8);

    DENSITY_GET_PRIV(tw)->BlackHoleOn = BlackHoleOn;
    DENSITY_GET_PRIV(tw)->SPH_predicted = SPH_predicted;
    DENSITY_GET_PRIV(tw)->GradRho = GradRho;

    /* Init Left and Right: this has to be done before treewalk */
    #pragma omp parallel for
    for(i = 0; i < act->NumActiveParticle; i++)  {
        int p_i = act->ActiveParticle ? act->ActiveParticle[i] : i;
        /* We only really need active particles with work
         * but I don't want to read the particle table here*/
        DENSITY_GET_PRIV(tw)->Right[p_i] = tree->BoxSize;
        DENSITY_GET_PRIV(tw)->NumNgb[p_i] = 0;
        DENSITY_GET_PRIV(tw)->Left[p_i] = 0;
    }

    priv->a3inv = pow(atime, -3);
    priv->MinEgySpec = MinEgySpec;
    /* Factor this out since all particles have the same drift time*/
    priv->FgravkickB = get_exact_gravkick_factor(CP, times.PM_kick, times.Ti_Current);
    memset(priv->gravkicks, 0, sizeof(priv->gravkicks[0])*(TIMEBINS+1));
    memset(priv->hydrokicks, 0, sizeof(priv->hydrokicks[0])*(TIMEBINS+1));
    /* Compute the factors to move a current kick times velocity to the drift time velocity.
     * We need to do the computation for all timebins up to the maximum because even inactive
     * particles may have interactions. */
    #pragma omp parallel for
    for(i = times.mintimebin; i <= TIMEBINS; i++)
    {
        priv->gravkicks[i] = get_exact_gravkick_factor(CP, times.Ti_kick[i], times.Ti_Current);
        priv->hydrokicks[i] = get_exact_hydrokick_factor(CP, times.Ti_kick[i], times.Ti_Current);
    }
    priv->times = &times;

    #pragma omp parallel for
    for(i = 0; i < act->NumActiveParticle; i++)
    {
        int p_i = act->ActiveParticle ? act->ActiveParticle[i] : i;
        if(P[p_i].Type == 0 && !P[p_i].IsGarbage) {
            int bin = P[p_i].TimeBin;
            double dloga = dloga_from_dti(priv->times->Ti_Current - priv->times->Ti_kick[bin], priv->times->Ti_Current);
            priv->SPH_predicted->EntVarPred[P[p_i].PI] = SPH_EntVarPred(P[p_i].PI, priv->MinEgySpec, priv->a3inv, dloga);
            SPH_VelPred(p_i, priv->SPH_predicted->VelPred + 3 * P[p_i].PI, priv->FgravkickB, priv->gravkicks[bin], priv->hydrokicks[bin]);
        }
    }

    /* allocate buffers to arrange communication */

    walltime_measure("/SPH/Density/Init");

    /* Do the treewalk with looping for hsml*/
    treewalk_do_hsml_loop(tw, act->ActiveParticle, act->NumActiveParticle, update_hsml);

    myfree(DENSITY_GET_PRIV(tw)->DhsmlDensityFactor);
    myfree(DENSITY_GET_PRIV(tw)->Rot);
    myfree(DENSITY_GET_PRIV(tw)->NumNgb);
    myfree(DENSITY_GET_PRIV(tw)->Right);
    myfree(DENSITY_GET_PRIV(tw)->Left);


    /* collect some timing information */

    timeall = walltime_measure(WALLTIME_IGNORE);

    timecomp = tw->timecomp3 + tw->timecomp1 + tw->timecomp2;
    timewait = tw->timewait1 + tw->timewait2;
    timecomm = tw->timecommsumm1 + tw->timecommsumm2;

    walltime_add("/SPH/Density/Compute", timecomp);
    walltime_add("/SPH/Density/Wait", timewait);
    walltime_add("/SPH/Density/Comm", timecomm);
    walltime_add("/SPH/Density/Misc", timeall - (timecomp + timewait + timecomm));
}

static void
density_copy(int place, TreeWalkQueryDensity * I, TreeWalk * tw)
{
    I->Hsml = P[place].Hsml;

    I->Type = P[place].Type;

    if(P[place].Type != 0)
    {
        I->Vel[0] = P[place].Vel[0];
        I->Vel[1] = P[place].Vel[1];
        I->Vel[2] = P[place].Vel[2];
    }
    else
    {
        MyFloat * velpred = DENSITY_GET_PRIV(tw)->SPH_predicted->VelPred;
        I->Vel[0] = velpred[3 * P[place].PI];
        I->Vel[1] = velpred[3 * P[place].PI + 1];
        I->Vel[2] = velpred[3 * P[place].PI + 2];
    }

}

static void
density_reduce(int place, TreeWalkResultDensity * remote, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    TREEWALK_REDUCE(DENSITY_GET_PRIV(tw)->NumNgb[place], remote->Ngb);
    TREEWALK_REDUCE(DENSITY_GET_PRIV(tw)->DhsmlDensityFactor[place], remote->DhsmlDensity);

    if(P[place].Type == 0)
    {
        TREEWALK_REDUCE(SPHP(place).Density, remote->Rho);

        TREEWALK_REDUCE(SPHP(place).DivVel, remote->Div);
        int pi = P[place].PI;
        TREEWALK_REDUCE(DENSITY_GET_PRIV(tw)->Rot[pi][0], remote->Rot[0]);
        TREEWALK_REDUCE(DENSITY_GET_PRIV(tw)->Rot[pi][1], remote->Rot[1]);
        TREEWALK_REDUCE(DENSITY_GET_PRIV(tw)->Rot[pi][2], remote->Rot[2]);

        MyFloat * gradrho = DENSITY_GET_PRIV(tw)->GradRho;

        if(gradrho) {
            TREEWALK_REDUCE(gradrho[3*pi], remote->GradRho[0]);
            TREEWALK_REDUCE(gradrho[3*pi+1], remote->GradRho[1]);
            TREEWALK_REDUCE(gradrho[3*pi+2], remote->GradRho[2]);
        }

        /*Only used for density independent SPH*/
        if(DENSITY_GET_PRIV(tw)->DoEgyDensity) {
            TREEWALK_REDUCE(SPHP(place).EgyWtDensity, remote->EgyRho);
            TREEWALK_REDUCE(SPHP(place).DhsmlEgyDensityFactor, remote->DhsmlEgyDensity);
        }
    }
    else if(P[place].Type == 5)
    {
        TREEWALK_REDUCE(BHP(place).Density, remote->Rho);
    }
}

/******
 *
 *  This function represents the core of the SPH density computation.
 *
 *  The neighbours of the particle in the Query are enumerated, and results
 *  are stored into the Result object.
 *
 *  Upon start-up we initialize the iterator with the density kernels used in
 *  the computation. The assumption is the density kernels are slow to
 *  initialize.
 *
 */

static void
density_ngbiter(
        TreeWalkQueryDensity * I,
        TreeWalkResultDensity * O,
        TreeWalkNgbIterDensity * iter,
        LocalTreeWalk * lv)
{
    if(iter->base.other == -1) {
        const double h = I->Hsml;
        density_kernel_init(&iter->kernel, h, DensityParams.DensityKernelType);
        iter->kernel_volume = density_kernel_volume(&iter->kernel);

        iter->base.Hsml = h;
        iter->base.mask = 1; /* gas only */
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        return;
    }
    const int other = iter->base.other;
    const double r = iter->base.r;
    const double r2 = iter->base.r2;
    const double * dist = iter->base.dist;

    if(P[other].Mass == 0) {
        endrun(12, "Density found zero mass particle %d type %d id %ld pos %g %g %g\n",
               other, P[other].Type, P[other].ID, P[other].Pos[0], P[other].Pos[1], P[other].Pos[2]);
    }

    if(r2 < iter->kernel.HH)
    {
        /* For the BH we wish to exclude wind particles from the density,
         * because they are excluded from the accretion treewalk.*/
        if(I->Type == 5 && winds_is_particle_decoupled(other))
            return;

        const double u = r * iter->kernel.Hinv;
        const double wk = density_kernel_wk(&iter->kernel, u);
        O->Ngb += wk * iter->kernel_volume;

        const double dwk = density_kernel_dwk(&iter->kernel, u);

        const double mass_j = P[other].Mass;

        O->Rho += (mass_j * wk);

        /* Hinv is here because O->DhsmlDensity is drho / dH.
         * nothing to worry here */
        double density_dW = density_kernel_dW(&iter->kernel, u, wk, dwk);
        O->DhsmlDensity += mass_j * density_dW;

        /* For the BH and stars only density and dhsmldensity is used.*/
        if(I->Type != 0)
            return;

        struct sph_pred_data * SphP_scratch = DENSITY_GET_PRIV(lv->tw)->SPH_predicted;

        double EntVarPred;
        MyFloat VelPred[3];
        #pragma omp atomic read
        EntVarPred = SphP_scratch->EntVarPred[P[other].PI];
        /* Lazily compute the predicted quantities. We can do this
         * with minimal locking since nothing happens should we compute them twice.
         * Zero can be the special value since there should never be zero entropy.*/
        if(EntVarPred == 0) {
            struct DensityPriv * priv = DENSITY_GET_PRIV(lv->tw);
            int bin = P[other].TimeBin;
            double dloga = dloga_from_dti(priv->times->Ti_Current - priv->times->Ti_kick[bin], priv->times->Ti_Current);
            EntVarPred = SPH_EntVarPred(P[other].PI, priv->MinEgySpec, priv->a3inv, dloga);
            SPH_VelPred(other, VelPred, priv->FgravkickB, priv->gravkicks[bin], priv->hydrokicks[bin]);
            /* Note this goes first to avoid threading issues: EntVarPred will only be set after this is done.
             * The worst that can happen is that some data points get copied twice.*/
            int i;
            for(i = 0; i < 3; i++) {
                #pragma omp atomic write
                SphP_scratch->VelPred[3 * P[other].PI + i] = VelPred[i];
            }
            #pragma omp atomic write
            SphP_scratch->EntVarPred[P[other].PI] = EntVarPred;
        }
        else {
            int i;
            for(i = 0; i < 3; i++) {
                #pragma omp atomic read
                VelPred[i] = SphP_scratch->VelPred[3 * P[other].PI + i];
            }
        }
        if(DENSITY_GET_PRIV(lv->tw)->DoEgyDensity) {
            O->EgyRho += mass_j * EntVarPred * wk;
            O->DhsmlEgyDensity += mass_j * EntVarPred * density_dW;
        }

        if(r > 0)
        {
            double fac = mass_j * dwk / r;
            double dv[3];
            double rot[3];
            int d;
            for(d = 0; d < 3; d ++) {
                dv[d] = I->Vel[d] - VelPred[d];
            }
            O->Div += -fac * dotproduct(dist, dv);

            crossproduct(dv, dist, rot);
            for(d = 0; d < 3; d ++) {
                O->Rot[d] += fac * rot[d];
            }
            if(DENSITY_GET_PRIV(lv->tw)->GradRho) {
                for (d = 0; d < 3; d ++)
                    O->GradRho[d] += fac * dist[d];
            }
        }
    }
}

static int
density_haswork(int n, TreeWalk * tw)
{
    /* Don't want a density for swallowed black hole particles*/
    if(P[n].Swallowed)
        return 0;
    if(P[n].Type == 0 || P[n].Type == 5)
        return 1;
    return 0;
}

static void
density_postprocess(int i, TreeWalk * tw)
{
    MyFloat * DhsmlDens = &(DENSITY_GET_PRIV(tw)->DhsmlDensityFactor[i]);
    double density = -1;
    if(P[i].Type == 0)
        density = SPHP(i).Density;
    else if(P[i].Type == 5)
        density = BHP(i).Density;
    if(density <= 0 && DENSITY_GET_PRIV(tw)->NumNgb[i] > 0) {
        endrun(12, "Particle %d type %d has bad density: %g\n", i, P[i].Type, density);
    }
    *DhsmlDens *= P[i].Hsml / (NUMDIMS * density);
    *DhsmlDens = 1 / (1 + *DhsmlDens);

    /* Uses DhsmlDensityFactor and changes Hsml, hence the location.*/
    if(DENSITY_GET_PRIV(tw)->update_hsml)
        density_check_neighbours(i, tw);

    if(P[i].Type == 0)
    {
        int PI = P[i].PI;
        /*Compute the EgyWeight factors, which are only useful for density independent SPH */
        if(DENSITY_GET_PRIV(tw)->DoEgyDensity) {
            struct sph_pred_data * SphP_scratch = DENSITY_GET_PRIV(tw)->SPH_predicted;
            const double EntPred = SphP_scratch->EntVarPred[P[i].PI];
            if(EntPred <= 0 || SPHP(i).EgyWtDensity <=0)
                endrun(12, "Particle %d has bad predicted entropy: %g or EgyWtDensity: %g\n", i, EntPred, SPHP(i).EgyWtDensity);
            SPHP(i).DhsmlEgyDensityFactor *= P[i].Hsml/ (NUMDIMS * SPHP(i).EgyWtDensity);
            SPHP(i).DhsmlEgyDensityFactor *= - (*DhsmlDens);
            SPHP(i).EgyWtDensity /= EntPred;
        }
        else
            SPHP(i).DhsmlEgyDensityFactor = *DhsmlDens;

        MyFloat * Rot = DENSITY_GET_PRIV(tw)->Rot[PI];
        SPHP(i).CurlVel = sqrt(Rot[0] * Rot[0] + Rot[1] * Rot[1] + Rot[2] * Rot[2]) / SPHP(i).Density;

        SPHP(i).DivVel /= SPHP(i).Density;
        P[i].DtHsml = (1.0 / NUMDIMS) * SPHP(i).DivVel * P[i].Hsml;
    }
}

void density_check_neighbours (int i, TreeWalk * tw)
{
    /* now check whether we had enough neighbours */
    int tid = omp_get_thread_num();
    double desnumngb = DENSITY_GET_PRIV(tw)->DesNumNgb;

    if(DENSITY_GET_PRIV(tw)->BlackHoleOn && P[i].Type == 5)
        desnumngb = desnumngb * DensityParams.BlackHoleNgbFactor;

    MyFloat * Left = DENSITY_GET_PRIV(tw)->Left;
    MyFloat * Right = DENSITY_GET_PRIV(tw)->Right;
    MyFloat * NumNgb = DENSITY_GET_PRIV(tw)->NumNgb;

    if(NumNgb[i] < (desnumngb - DensityParams.MaxNumNgbDeviation) ||
            (NumNgb[i] > (desnumngb + DensityParams.MaxNumNgbDeviation)))
    {
        /* This condition is here to prevent the density code looping forever if it encounters
         * multiple particles at the same position. If this happens you likely have worse
         * problems anyway, so warn also. */
        if((Right[i] - Left[i]) < 1.0e-3 * Left[i])
        {
            /* If this happens probably the exchange is screwed up and all your particles have moved to (0,0,0)*/
            message(1, "Very tight Hsml bounds for i=%d ID=%lu Hsml=%g Left=%g Right=%g Ngbs=%g Right-Left=%g pos=(%g|%g|%g)\n",
             i, P[i].ID, P[i].Hsml, Left[i], Right[i], NumNgb[i], Right[i] - Left[i], P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
            P[i].Hsml = Right[i];
            return;
        }

        /* If we need more neighbours, move the lower bound up. If we need fewer, move the upper bound down.*/
        if(NumNgb[i] < desnumngb) {
                Left[i] = P[i].Hsml;
        } else {
                Right[i] = P[i].Hsml;
        }

        /* Next step is geometric mean of previous. */
        if((Right[i] < tw->tree->BoxSize && Left[i] > 0) || (P[i].Hsml * 1.26 > 0.99 * tw->tree->BoxSize))
            P[i].Hsml = pow(0.5 * (pow(Left[i], 3) + pow(Right[i], 3)), 1.0 / 3);
        else
        {
            if(!(Right[i] < tw->tree->BoxSize) && Left[i] == 0)
                endrun(8188, "Cannot occur. Check for memory corruption: i=%d L = %g R = %g N=%g. Type %d, Pos %g %g %g hsml %g Box %g\n", i, Left[i], Right[i], NumNgb[i], P[i].Type, P[i].Pos[0], P[i].Pos[1], P[i].Pos[2], P[i].Hsml, tw->tree->BoxSize);

            MyFloat DensFac = DENSITY_GET_PRIV(tw)->DhsmlDensityFactor[i];
            double fac = 1.26;
            if(NumNgb[i] > 0)
                fac = 1 - (NumNgb[i] - desnumngb) / (NUMDIMS * NumNgb[i]) * DensFac;

            /* Find the initial bracket using the kernel gradients*/
            if(Right[i] > 0.99 * tw->tree->BoxSize && Left[i] > 0)
                if(DensFac <= 0 || fabs(NumNgb[i] - desnumngb) >= 0.5 * desnumngb || fac > 1.26)
                    fac = 1.26;

            if(Right[i] < 0.99*tw->tree->BoxSize && Left[i] == 0)
                if(DensFac <=0 || fac < 1./3)
                    fac = 1./3;

            P[i].Hsml *= fac;
        }

        if(DENSITY_GET_PRIV(tw)->BlackHoleOn && P[i].Type == 5)
            if(Left[i] > DensityParams.BlackHoleMaxAccretionRadius)
            {
                P[i].Hsml = DensityParams.BlackHoleMaxAccretionRadius;
                return;
            }

        if(Right[i] < DENSITY_GET_PRIV(tw)->MinGasHsml) {
            P[i].Hsml = DENSITY_GET_PRIV(tw)->MinGasHsml;
            return;
        }
        /* More work needed: add this particle to the redo queue*/
        tw->NPRedo[tid][tw->NPLeft[tid]] = i;
        tw->NPLeft[tid] ++;
        if(tw->NPLeft[tid] > tw->Redo_thread_alloc)
            endrun(5, "Particle %d on thread %d exceeded allocated size of redo queue %ld\n", tw->NPLeft[tid], tid, tw->Redo_thread_alloc);
    }
    else {
        /* We might have got here by serendipity, without bounding.*/
        if(DENSITY_GET_PRIV(tw)->BlackHoleOn && P[i].Type == 5)
            if(P[i].Hsml > DensityParams.BlackHoleMaxAccretionRadius)
                P[i].Hsml = DensityParams.BlackHoleMaxAccretionRadius;
        if(P[i].Hsml < DENSITY_GET_PRIV(tw)->MinGasHsml)
            P[i].Hsml = DENSITY_GET_PRIV(tw)->MinGasHsml;
    }
    if(tw->maxnumngb[tid] < NumNgb[i])
        tw->maxnumngb[tid] = NumNgb[i];
    if(tw->minnumngb[tid] > NumNgb[i])
        tw->minnumngb[tid] = NumNgb[i];

    if(tw->Niteration >= MAXITER - 10)
    {
         message(1, "i=%d ID=%lu Hsml=%g Left=%g Right=%g Ngbs=%g Right-Left=%g\n   pos=(%g|%g|%g)\n",
             i, P[i].ID, P[i].Hsml, Left[i], Right[i],
             NumNgb[i], Right[i] - Left[i], P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
    }
}


struct sph_pred_data
slots_allocate_sph_pred_data(int nsph)
{
    struct sph_pred_data sph_scratch;
    /*Data is allocated high so that we can free the tree around it*/
    sph_scratch.EntVarPred = (MyFloat *) mymalloc2("EntVarPred", sizeof(MyFloat) * nsph);
    memset(sph_scratch.EntVarPred, 0, sizeof(sph_scratch.EntVarPred[0]) * nsph);
    sph_scratch.VelPred = (MyFloat *) mymalloc2("VelPred", sizeof(MyFloat) * 3 * nsph);
    return sph_scratch;
}

void
slots_free_sph_pred_data(struct sph_pred_data * sph_scratch)
{
    myfree(sph_scratch->VelPred);
    sph_scratch->VelPred = NULL;
    myfree(sph_scratch->EntVarPred);
    sph_scratch->EntVarPred = NULL;
}
