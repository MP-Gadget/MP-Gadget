#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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
        DensityParams.DensityKernelType = (enum DensityKernelType) param_get_enum(ps, "DensityKernelType");
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
SPH_EntVarPred(const int p_i, const DriftKickTimes * times)
{
        const int bin = P[p_i].TimeBinHydro;
        const int PI = P[p_i].PI;
        const double dloga = dloga_from_dti(times->Ti_Current - times->Ti_kick[bin], times->Ti_Current);
        double EntVarPred = SphP[PI].Entropy + SphP[PI].DtEntropy * dloga;
        /*Entropy limiter for the predicted entropy: makes sure entropy stays positive. */
        if(EntVarPred < 0.05*SphP[PI].Entropy)
            EntVarPred = 0.05 * SphP[PI].Entropy;
        /* Just in case*/
        if(EntVarPred <= 0)
            return 0;
        EntVarPred = exp(1./GAMMA * log(EntVarPred));
//         EntVarPred = pow(EntVarPred, 1/GAMMA);
        return EntVarPred;
}

/* Get the predicted velocity for a particle
 * at the current Force computation time ti,
 * which always coincides with the Drift inttime.
 * For hydro forces.*/
void
SPH_VelPred(int i, MyFloat * VelPred, const struct kick_factor_data * kf)
{
    int j;
    /* Notice that the kick time for gravity and hydro may be different! So the prediction is also different*/
    for(j = 0; j < 3; j++) {
        VelPred[j] = P[i].Vel[j] + kf->gravkicks[P[i].TimeBinGravity] * P[i].FullTreeGravAccel[j]
            + P[i].GravPM[j] * kf->FgravkickB + kf->hydrokicks[P[i].TimeBinHydro] * SPHP(i).HydroAccel[j];
    }
}

/* Get the predicted velocity for a particle
 * at the current Force computation time ti,
 * which always coincides with the Drift inttime.
 * For hydro forces.*/
void
DM_VelPred(int i, MyFloat * VelPred, const struct kick_factor_data * kf)
{
    int j;
    for(j = 0; j < 3; j++)
        VelPred[j] = P[i].Vel[j] + kf->gravkicks[P[i].TimeBinGravity] * P[i].FullTreeGravAccel[j]+ P[i].GravPM[j] * kf->FgravkickB;
}

/* Initialise the grav and hydrokick arrays for the current kick times.*/
void
init_kick_factor_data(struct kick_factor_data * kf, const DriftKickTimes * const times, Cosmology * CP)
{
    int i;
    /* Factor this out since all particles have the same PM kick time*/
    kf->FgravkickB = get_exact_gravkick_factor(CP, times->PM_kick, times->Ti_Current);
    memset(kf->gravkicks, 0, sizeof(kf->gravkicks[0])*(TIMEBINS+1));
    memset(kf->hydrokicks, 0, sizeof(kf->hydrokicks[0])*(TIMEBINS+1));
    /* Compute the factors to move a current kick times velocity to the drift time velocity.
     * We need to do the computation for all timebins up to the maximum because even inactive
     * particles may have interactions. */
    #pragma omp parallel for
    for(i = times->mintimebin; i <= TIMEBINS; i++)
    {
        kf->gravkicks[i] = get_exact_gravkick_factor(CP, times->Ti_kick[i], times->Ti_Current);
        kf->hydrokicks[i] = get_exact_hydrokick_factor(CP, times->Ti_kick[i], times->Ti_Current);
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
    DriftKickTimes const * times;
    struct kick_factor_data kf;
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
static int density_check_neighbours(int i, TreeWalk * tw);

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
density(const ActiveParticles * act, int update_hsml, int DoEgyDensity, int BlackHoleOn, const DriftKickTimes times, Cosmology * CP, struct sph_pred_data * SPH_predicted, MyFloat * GradRho_mag, const ForceTree * const tree)
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

    DENSITY_GET_PRIV(tw)->Left = (MyFloat *) mymalloc("DENS_PRIV->Left", PartManager->NumPart * sizeof(MyFloat));
    DENSITY_GET_PRIV(tw)->Right = (MyFloat *) mymalloc("DENS_PRIV->Right", PartManager->NumPart * sizeof(MyFloat));
    DENSITY_GET_PRIV(tw)->NumNgb = (MyFloat *) mymalloc("DENS_PRIV->NumNgb", PartManager->NumPart * sizeof(MyFloat));
    DENSITY_GET_PRIV(tw)->Rot = (MyFloat (*) [3]) mymalloc("DENS_PRIV->Rot", SlotsManager->info[0].size * sizeof(priv->Rot[0]));
    /* This one stores the gradient for h finding. The factor stored in SPHP->DhsmlEgyDensityFactor depends on whether PE SPH is enabled.*/
    DENSITY_GET_PRIV(tw)->DhsmlDensityFactor = (MyFloat *) mymalloc("DENSITY_GET_PRIV(tw)->DhsmlDensity", PartManager->NumPart * sizeof(MyFloat));

    DENSITY_GET_PRIV(tw)->update_hsml = update_hsml;
    DENSITY_GET_PRIV(tw)->DoEgyDensity = DoEgyDensity;

    DENSITY_GET_PRIV(tw)->DesNumNgb = GetNumNgb(DensityParams.DensityKernelType);
    DENSITY_GET_PRIV(tw)->MinGasHsml = DensityParams.MinGasHsmlFractional * (FORCE_SOFTENING()/2.8);

    DENSITY_GET_PRIV(tw)->BlackHoleOn = BlackHoleOn;
    DENSITY_GET_PRIV(tw)->SPH_predicted = SPH_predicted;
    if(GradRho_mag)
        DENSITY_GET_PRIV(tw)->GradRho = (MyFloat *) mymalloc("SPH_GradRho", sizeof(MyFloat) * 3 * SlotsManager->info[0].size);
    else
        DENSITY_GET_PRIV(tw)->GradRho = NULL;

    int i;
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

    init_kick_factor_data(&priv->kf, &times, CP);
    priv->times = &times;

    /* If all particles are active, easiest to compute all the predicted velocities immediately*/
    if(!act->ActiveParticle || act->NumActiveHydro > 0.1 * (SlotsManager->info[0].size + SlotsManager->info[5].size)) {
        priv->SPH_predicted->EntVarPred = (MyFloat *) mymalloc2("EntVarPred", sizeof(MyFloat) * SlotsManager->info[0].size);
        #pragma omp parallel for
        for(i = 0; i < PartManager->NumPart; i++)
            if(P[i].Type == 0 && !P[i].IsGarbage)
                priv->SPH_predicted->EntVarPred[P[i].PI] = SPH_EntVarPred(i, priv->times);
    }
    /* But if only some particles are active, the pow function in EntVarPred is slow and we have a lot of overhead, because we are doing 5500^3 exps for 5 particles.
     * So instead we compute it for active particles and use an atomic to guard the changes inside the loop.
     * For sufficiently small particle numbers the memset dominates and it is fastest to just compute each predicted entropy as we need it.*/
    else if(act->NumActiveHydro > 0.0001 * (SlotsManager->info[0].size + SlotsManager->info[5].size)){
        priv->SPH_predicted->EntVarPred = (MyFloat *) mymalloc2("EntVarPred", sizeof(MyFloat) * SlotsManager->info[0].size);
        memset(priv->SPH_predicted->EntVarPred, 0, sizeof(priv->SPH_predicted->EntVarPred[0]) * SlotsManager->info[0].size);
        #pragma omp parallel for
        for(i = 0; i < act->NumActiveParticle; i++)
        {
            int p_i = act->ActiveParticle ? act->ActiveParticle[i] : i;
            if(P[p_i].Type == 0 && !P[p_i].IsGarbage)
                priv->SPH_predicted->EntVarPred[P[p_i].PI] = SPH_EntVarPred(p_i, priv->times);
        }
    }

    /* allocate buffers to arrange communication */

    walltime_measure("/SPH/Density/Init");

    /* Do the treewalk with looping for hsml*/
    treewalk_do_hsml_loop(tw, act->ActiveParticle, act->NumActiveParticle, update_hsml);

    if(GradRho_mag) {
        #pragma omp parallel for
        for(i = 0; i < SlotsManager->info[0].size; i++)
        {
            MyFloat * gr = DENSITY_GET_PRIV(tw)->GradRho + (3*i);
            GradRho_mag[i] = sqrt(gr[0]*gr[0] + gr[1] * gr[1] + gr[2] * gr[2]);
        }
    }

    if(DENSITY_GET_PRIV(tw)->GradRho)
        myfree(DENSITY_GET_PRIV(tw)->GradRho);
    myfree(DENSITY_GET_PRIV(tw)->DhsmlDensityFactor);
    myfree(DENSITY_GET_PRIV(tw)->Rot);
    myfree(DENSITY_GET_PRIV(tw)->NumNgb);
    myfree(DENSITY_GET_PRIV(tw)->Right);
    myfree(DENSITY_GET_PRIV(tw)->Left);


    /* collect some timing information */

    double timeall = walltime_measure(WALLTIME_IGNORE);

    double timecomp = tw->timecomp0 + tw->timecomp3 + tw->timecomp1 + tw->timecomp2;
    walltime_add("/SPH/Density/WalkTop", tw->timecomp0);
    walltime_add("/SPH/Density/WalkPrim", tw->timecomp1);
    walltime_add("/SPH/Density/WalkSec", tw->timecomp2);
    walltime_add("/SPH/Density/PostPre", tw->timecomp3);
    // walltime_add("/SPH/Density/Compute", timecomp);
    walltime_add("/SPH/Density/Wait", tw->timewait1);
    walltime_add("/SPH/Density/Reduce", tw->timecommsumm);
    walltime_add("/SPH/Density/Misc", timeall - (timecomp + tw->timewait1 + tw->timecommsumm));
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
        SPH_VelPred(place, I->Vel, &DENSITY_GET_PRIV(tw)->kf);
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
        TREEWALK_REDUCE(BHP(place).DivVel, remote->Div);
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
        iter->base.mask = GASMASK; /* gas only */
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

        double EntVarPred;
        MyFloat VelPred[3];
        struct DensityPriv * priv = DENSITY_GET_PRIV(lv->tw);
        SPH_VelPred(other, VelPred, &priv->kf);

        if(priv->SPH_predicted->EntVarPred) {
            #pragma omp atomic read
            EntVarPred = priv->SPH_predicted->EntVarPred[P[other].PI];
            /* Lazily compute the predicted quantities. We can do this
            * with minimal locking since nothing happens should we compute them twice.
            * Zero can be the special value since there should never be zero entropy.*/
            if(EntVarPred == 0) {
                EntVarPred = SPH_EntVarPred(other, priv->times);
                #pragma omp atomic write
                priv->SPH_predicted->EntVarPred[P[other].PI] = EntVarPred;
            }
        }
        else
            EntVarPred = SPH_EntVarPred(other, priv->times);

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
    if(DENSITY_GET_PRIV(tw)->update_hsml) {
        int done = density_check_neighbours(i, tw);
        /* If we are done repeating, update the hmax in the parent node,
         * if that type is in the tree.*/
        if(done && (tw->tree->mask & (1<<P[i].Type)))
            update_tree_hmax_father(tw->tree, i, P[i].Pos, P[i].Hsml);
    }

    if(P[i].Type == 0)
    {
        int PI = P[i].PI;
        /*Compute the EgyWeight factors, which are only useful for density independent SPH */
        if(DENSITY_GET_PRIV(tw)->DoEgyDensity) {
            double EntPred;
            if(DENSITY_GET_PRIV(tw)->SPH_predicted->EntVarPred)
                EntPred = DENSITY_GET_PRIV(tw)->SPH_predicted->EntVarPred[P[i].PI];
            else
                EntPred = SPH_EntVarPred(i, DENSITY_GET_PRIV(tw)->times);
            if(EntPred <= 0 || SPHP(i).EgyWtDensity <=0)
                endrun(12, "Particle %d has bad predicted entropy: %g or EgyWtDensity: %g, Particle ID = %ld, pos %g %g %g, vel %g %g %g, mass = %g, density = %g, MaxSignalVel = %g, Entropy = %g, DtEntropy = %g \n", i, EntPred, SPHP(i).EgyWtDensity, P[i].ID, P[i].Pos[0], P[i].Pos[1], P[i].Pos[2], P[i].Vel[0], P[i].Vel[1], P[i].Vel[2], P[i].Mass, SPHP(i).Density, SPHP(i).MaxSignalVel, SPHP(i).Entropy, SPHP(i).DtEntropy);
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
    else if(P[i].Type == 5)
    {
        BHP(i).DivVel /= BHP(i).Density;
        P[i].DtHsml = (1.0 / NUMDIMS) * BHP(i).DivVel * P[i].Hsml;
    }
}

/* Returns 1 if we are done and do not need to loop. 0 if we need to repeat.*/
int
density_check_neighbours (int i, TreeWalk * tw)
{
    /* now check whether we had enough neighbours */
    int tid = omp_get_thread_num();
    double desnumngb = DENSITY_GET_PRIV(tw)->DesNumNgb;

    if(DENSITY_GET_PRIV(tw)->BlackHoleOn && P[i].Type == 5)
        desnumngb = desnumngb * DensityParams.BlackHoleNgbFactor;

    MyFloat * Left = DENSITY_GET_PRIV(tw)->Left;
    MyFloat * Right = DENSITY_GET_PRIV(tw)->Right;
    MyFloat * NumNgb = DENSITY_GET_PRIV(tw)->NumNgb;

    if(tw->maxnumngb[tid] < NumNgb[i])
        tw->maxnumngb[tid] = NumNgb[i];
    if(tw->minnumngb[tid] > NumNgb[i])
        tw->minnumngb[tid] = NumNgb[i];

    if(tw->Niteration >= MAXITER - 5)
    {
         message(1, "i=%d ID=%lu Hsml=%g Left=%g Right=%g Ngbs=%g Right-Left=%g\n   pos=(%g|%g|%g)\n",
             i, P[i].ID, P[i].Hsml, Left[i], Right[i],
             NumNgb[i], Right[i] - Left[i], P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
    }

    if(NumNgb[i] < (desnumngb - DensityParams.MaxNumNgbDeviation) ||
            (NumNgb[i] > (desnumngb + DensityParams.MaxNumNgbDeviation)))
    {
        /* This condition is here to prevent the density code looping forever if it encounters
         * multiple particles at the same position. If this happens you likely have worse
         * problems anyway, so warn also. */
        if((Right[i] - Left[i]) < 1.0e-5 * Left[i])
        {
            /* If this happens probably the exchange is screwed up and all your particles have moved to (0,0,0)*/
            message(1, "Very tight Hsml bounds for i=%d ID=%lu Hsml=%g Left=%g Right=%g Ngbs=%g Right-Left=%g pos=(%g|%g|%g)\n",
             i, P[i].ID, P[i].Hsml, Left[i], Right[i], NumNgb[i], Right[i] - Left[i], P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
            P[i].Hsml = Right[i];
            return 1;
        }

        /* If we need more neighbours, move the lower bound up. If we need fewer, move the upper bound down.*/
        if(NumNgb[i] < desnumngb) {
                Left[i] = P[i].Hsml;
        } else {
                Right[i] = P[i].Hsml;
        }

        /* Next step is geometric mean of previous. */
        if((Right[i] < tw->tree->BoxSize && Left[i] > 0) || (P[i].Hsml * 1.26 > 0.99 * tw->tree->BoxSize))
            P[i].Hsml = cbrt(0.5 * (pow(Left[i], 3) + pow(Right[i], 3)));
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
                return 1;
            }

        if(Right[i] < DENSITY_GET_PRIV(tw)->MinGasHsml) {
            P[i].Hsml = DENSITY_GET_PRIV(tw)->MinGasHsml;
            return 1;
        }
        /* More work needed: add this particle to the redo queue*/
        tw->NPRedo[tid][tw->NPLeft[tid]] = i;
        tw->NPLeft[tid] ++;
        if(tw->NPLeft[tid] > tw->Redo_thread_alloc)
            endrun(5, "Particle %ld on thread %d exceeded allocated size of redo queue %ld\n", tw->NPLeft[tid], tid, tw->Redo_thread_alloc);
        return 0;
    }
    else {
        /* We might have got here by serendipity, without bounding.*/
        if(DENSITY_GET_PRIV(tw)->BlackHoleOn && P[i].Type == 5)
            if(P[i].Hsml > DensityParams.BlackHoleMaxAccretionRadius)
                P[i].Hsml = DensityParams.BlackHoleMaxAccretionRadius;
        if(P[i].Hsml < DENSITY_GET_PRIV(tw)->MinGasHsml)
            P[i].Hsml = DENSITY_GET_PRIV(tw)->MinGasHsml;
        return 1;
    }
}

void
slots_free_sph_pred_data(struct sph_pred_data * sph_scratch)
{
    if(sph_scratch->EntVarPred)
        myfree(sph_scratch->EntVarPred);
    sph_scratch->EntVarPred = NULL;
}

/* Set the initial smoothing length for gas and BH*/
void
set_init_hsml(ForceTree * tree, DomainDecomp * ddecomp, const double MeanGasSeparation)
{
    /* Need moments because we use them to set Hsml*/
    force_tree_calc_moments(tree, ddecomp);
    if(!tree->Father)
        endrun(5, "tree Father array not allocated at initial hsml!\n");
    const double DesNumNgb = GetNumNgb(GetDensityKernelType());
    int i;
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)
    {
        /* These initial smoothing lengths are only used for SPH-like particles.*/
        if(P[i].Type != 0 && P[i].Type != 5)
            continue;

        if(P[i].IsGarbage)
            continue;
        int no = i;

        do {
            int p = force_get_father(no, tree);

            if(p < tree->firstnode)
                break;

            /* Check that we didn't somehow get a bad set of nodes*/
            if(p > tree->numnodes + tree->firstnode)
                endrun(5, "Bad init father: i=%d, mass = %g type %d hsml %g no %d len %g father %d, numnodes %ld firstnode %ld\n",
                    i, P[i].Mass, P[i].Type, P[i].Hsml, no, tree->Nodes[no].len, p, tree->numnodes, tree->firstnode);
            no = p;
        } while(10 * DesNumNgb * P[i].Mass > tree->Nodes[no].mom.mass);

        /* Validate the tree node contents*/
        if(tree->Nodes[no].len > tree->BoxSize || tree->Nodes[no].mom.mass < P[i].Mass)
            endrun(5, "Bad tree moments: i=%d, mass = %g type %d hsml %g no %d len %g treemass %g\n",
                    i, P[i].Mass, P[i].Type, P[i].Hsml, no, tree->Nodes[no].len, tree->Nodes[no].mom.mass);
        P[i].Hsml = MeanGasSeparation;
        if(no >= tree->firstnode) {
            double testhsml = tree->Nodes[no].len * pow(3.0 / (4 * M_PI) * DesNumNgb * P[i].Mass / tree->Nodes[no].mom.mass, 1.0 / 3);
            /* recover from a poor initial guess */
            if (testhsml < 500. * MeanGasSeparation)
                P[i].Hsml = testhsml;
        }

        if(P[i].Hsml <= 0)
            endrun(5, "Bad hsml guess: i=%d, mass = %g type %d hsml %g no %d len %g treemass %g\n",
                    i, P[i].Mass, P[i].Type, P[i].Hsml, no, tree->Nodes[no].len, tree->Nodes[no].mom.mass);
    }
}
