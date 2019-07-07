#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "cooling.h"
#include "densitykernel.h"
#include "treewalk.h"
#include "timefac.h"
#include "slotsmanager.h"
#include "timestep.h"
#include "winds.h"
#include "utils.h"

#define MAXITER 400

/* The evolved entropy at drift time: evolved dlog a.
 * Used to predict pressure and entropy for SPH */
static MyFloat
SPH_EntVarPred(int i)
{
        double dloga = dloga_from_dti(P[i].Ti_drift - P[i].Ti_kick);
        double EntVarPred = SPHP(i).Entropy + SPHP(i).DtEntropy * dloga;
        /*Entropy limiter for the predicted entropy: makes sure entropy stays positive. */
        if(dloga > 0 && EntVarPred < 0.5*SPHP(i).Entropy)
            EntVarPred = 0.5 * SPHP(i).Entropy;
        EntVarPred = pow(EntVarPred, 1/GAMMA);
        return EntVarPred;
}

/* Get the predicted velocity for a particle
 * at the Force computation time, which always coincides with the Drift inttime.
 * for gravity and hydro forces.*/
static void
SPH_VelPred(int i, MyFloat * VelPred)
{
    const int ti = P[i].Ti_drift;
    const double Fgravkick2 = get_gravkick_factor(P[i].Ti_kick, ti);
    const double Fhydrokick2 = get_hydrokick_factor(P[i].Ti_kick, ti);
    const double FgravkickB = get_gravkick_factor(PM.Ti_kick, ti);
    int j;
    for(j = 0; j < 3; j++) {
        VelPred[j] = P[i].Vel[j] + Fgravkick2 * P[i].GravAccel[j]
            + P[i].GravPM[j] * FgravkickB + Fhydrokick2 * SPHP(i).HydroAccel[j];
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
    /*Wind model*/
    MyFloat DelayTime;
    int Type;
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

    int Ninteractions;

    /*Only used if sfr_need_to_compute_sph_grad_rho is true*/
    MyFloat GradRho[3];
} TreeWalkResultDensity;

struct DensityPriv {
    MyFloat *Left, *Right;
    MyFloat (*Rot)[3];
    /* This is the DhsmlDensityFactor for the pure density,
     * not the entropy weighted density.
     * If DensityIndependentSphOn = 0 then DhsmlEgyDensityFactor and DhsmlDensityFactor
     * are the same and this is not used.
     * If DensityIndependentSphOn = 1 then this is used to set DhsmlEgyDensityFactor.*/
    MyFloat * DhsmlDensityFactor;
    int NIteration;
    int *NPLeft;
    int update_hsml;
    int DoEgyDensity;
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
density(int update_hsml, int DoEgyDensity, ForceTree * tree)
{
    if(!All.DensityOn)
        return;

    TreeWalk tw[1] = {{0}};
    struct DensityPriv priv[1];

    tw->ev_label = "DENSITY";
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterDensity);
    tw->ngbiter = (TreeWalkNgbIterFunction) density_ngbiter;
    tw->haswork = density_haswork;
    tw->fill = (TreeWalkFillQueryFunction) density_copy;
    tw->reduce = (TreeWalkReduceResultFunction) density_reduce;
    tw->postprocess = (TreeWalkProcessFunction) density_postprocess;
    tw->UseNodeList = 1;
    tw->query_type_elsize = sizeof(TreeWalkQueryDensity);
    tw->result_type_elsize = sizeof(TreeWalkResultDensity);
    tw->priv = priv;
    tw->tree = tree;

    int i;
    int64_t ntot = 0;

    double timeall = 0;
    double timecomp, timecomm, timewait;

    walltime_measure("/Misc");

    DENSITY_GET_PRIV(tw)->Left = (MyFloat *) mymalloc("DENSITY_GET_PRIV(tw)->Left", PartManager->NumPart * sizeof(MyFloat));
    DENSITY_GET_PRIV(tw)->Right = (MyFloat *) mymalloc("DENSITY_GET_PRIV(tw)->Right", PartManager->NumPart * sizeof(MyFloat));
    DENSITY_GET_PRIV(tw)->Rot = (MyFloat (*) [3]) mymalloc("DENSITY_GET_PRIV(tw)->Rot", SlotsManager->info[0].size * sizeof(priv->Rot[0]));
    if(DoEgyDensity)
        DENSITY_GET_PRIV(tw)->DhsmlDensityFactor = (MyFloat *) mymalloc("DENSITY_GET_PRIV(tw)->DhsmlDensity", SlotsManager->info[0].size * sizeof(MyFloat));
    else
        DENSITY_GET_PRIV(tw)->DhsmlDensityFactor = NULL;

    DENSITY_GET_PRIV(tw)->update_hsml = update_hsml;
    DENSITY_GET_PRIV(tw)->DoEgyDensity = DoEgyDensity;

    DENSITY_GET_PRIV(tw)->NIteration = 0;

    /* this has to be done before treewalk so that
     * all particles are ran for the first loop.
     * The iteration will gradually turn DensityIterationDone on more particles.
     * */
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)
    {
        P[i].DensityIterationDone = 0;
        P[i].NumNgb = 0;
        DENSITY_GET_PRIV(tw)->Left[i] = 0;
        DENSITY_GET_PRIV(tw)->Right[i] = All.BoxSize;
        if(P[i].Type == 0) {
            SphP_scratch->EntVarPred[P[i].PI] = SPH_EntVarPred(i);
            SPH_VelPred(i, SphP_scratch->VelPred + 3 * P[i].PI);
        }
    }

    /* allocate buffers to arrange communication */

    walltime_measure("/SPH/Density/Init");

    DENSITY_GET_PRIV(tw)->NPLeft = ta_malloc("NPLeft", int, All.NumThreads);

    /* we will repeat the whole thing for those particles where we didn't find enough neighbours */
    do {
        memset(DENSITY_GET_PRIV(tw)->NPLeft, 0, sizeof(int)*All.NumThreads);

        treewalk_run(tw, ActiveParticle, NumActiveParticle);

        int Nleft = 0;

        for(i = 0; i< All.NumThreads; i++)
            Nleft += DENSITY_GET_PRIV(tw)->NPLeft[i];

        sumup_large_ints(1, &Nleft, &ntot);

        if(ntot == 0) break;

        DENSITY_GET_PRIV(tw)->NIteration ++;
        /*
        if(ntot < 1 ) {
            foreach(ActiveParticle)
            {
                if(density_haswork(i) && !P[i].DensityIterationDone) {
                    MyFloat Left = DENSITY_GET_PRIV(tw)->Left[i];
                    MyFloat Right = DENSITY_GET_PRIV(tw)->Right[i];
                    message (1, "i=%d task=%d ID=%llu type=%d, Hsml=%g Left=%g Right=%g Ngbs=%g Right-Left=%g\n   pos=(%g|%g|%g)\n",
                         i, ThisTask, P[i].ID, P[i].Type, P[i].Hsml, Left, Right,
                         (float) P[i].NumNgb, Right - Left, P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
                }
            }

        }
        */

        if(DENSITY_GET_PRIV(tw)->NIteration > 0) {
            message(0, "ngb iteration %d: need to repeat for %ld particles.\n", DENSITY_GET_PRIV(tw)->NIteration, ntot);
        }

        if(DENSITY_GET_PRIV(tw)->NIteration > MAXITER) {
            endrun(1155, "failed to converge in neighbour iteration in density()\n");
        }
    } while(1);

    ta_free(DENSITY_GET_PRIV(tw)->NPLeft);
    if(DoEgyDensity)
        myfree(DENSITY_GET_PRIV(tw)->DhsmlDensityFactor);
    myfree(DENSITY_GET_PRIV(tw)->Rot);
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
        I->DelayTime = 0;
    }
    else
    {
        I->Vel[0] = SphP_scratch->VelPred[3 * P[place].PI];
        I->Vel[1] = SphP_scratch->VelPred[3 * P[place].PI + 1];
        I->Vel[2] = SphP_scratch->VelPred[3 * P[place].PI + 2];
        I->DelayTime = SPHP(place).DelayTime;
    }

}

static void
density_reduce(int place, TreeWalkResultDensity * remote, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    TREEWALK_REDUCE(P[place].NumNgb, remote->Ngb);

    /* these will be added */
    P[place].GravCost += All.HydroCostFactor * All.cf.a * remote->Ninteractions;

    if(P[place].Type == 0)
    {
        TREEWALK_REDUCE(SPHP(place).Density, remote->Rho);

        TREEWALK_REDUCE(SPHP(place).DivVel, remote->Div);
        int pi = P[place].PI;
        TREEWALK_REDUCE(DENSITY_GET_PRIV(tw)->Rot[pi][0], remote->Rot[0]);
        TREEWALK_REDUCE(DENSITY_GET_PRIV(tw)->Rot[pi][1], remote->Rot[1]);
        TREEWALK_REDUCE(DENSITY_GET_PRIV(tw)->Rot[pi][2], remote->Rot[2]);

        if(SphP_scratch->GradRho) {
            TREEWALK_REDUCE(SphP_scratch->GradRho[3*pi], remote->GradRho[0]);
            TREEWALK_REDUCE(SphP_scratch->GradRho[3*pi+1], remote->GradRho[1]);
            TREEWALK_REDUCE(SphP_scratch->GradRho[3*pi+2], remote->GradRho[2]);
        }

        /*Only used for density independent SPH*/
        if(DENSITY_GET_PRIV(tw)->DoEgyDensity) {
            TREEWALK_REDUCE(SPHP(place).EgyWtDensity, remote->EgyRho);
            TREEWALK_REDUCE(SPHP(place).DhsmlEgyDensityFactor, remote->DhsmlEgyDensity);
            TREEWALK_REDUCE(DENSITY_GET_PRIV(tw)->DhsmlDensityFactor[pi], remote->DhsmlDensity);
        }
        else
            TREEWALK_REDUCE(SPHP(place).DhsmlEgyDensityFactor, remote->DhsmlDensity);
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
        density_kernel_init(&iter->kernel, h);
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

    if(All.WindOn) {
        if(winds_is_particle_decoupled(other))
            if(!(I->Type == 0 && I->DelayTime > 0))	/* if I'm not wind, then ignore the wind particle */
                return;
    }

    if(P[other].Mass == 0) {
        endrun(12, "Encountered zero mass particle during density;"
                  " We haven't implemented tracer particles and this shall not happen\n");
    }

    /* some performance measures*/
    O->Ninteractions ++;

    if(r2 < iter->kernel.HH)
    {
        const double u = r * iter->kernel.Hinv;
        const double wk = density_kernel_wk(&iter->kernel, u);
        O->Ngb += wk * iter->kernel_volume;

        /* For the BH only Ngb is used. BH density is
         * computed during accretion.*/
        if(I->Type == 5)
            return;

        const double dwk = density_kernel_dwk(&iter->kernel, u);

        const double mass_j = P[other].Mass;

        O->Rho += (mass_j * wk);

        /* Hinv is here because O->DhsmlDensity is drho / dH.
         * nothing to worry here */
        double density_dW = density_kernel_dW(&iter->kernel, u, wk, dwk);
        O->DhsmlDensity += mass_j * density_dW;

        if(DENSITY_GET_PRIV(lv->tw)->DoEgyDensity) {
            const double EntPred = SphP_scratch->EntVarPred[P[other].PI];
            O->EgyRho += mass_j * EntPred * wk;
            O->DhsmlEgyDensity += mass_j * EntPred * density_dW;
        }


        if(SphP_scratch->GradRho) {
            if(r > 0)
            {
                int d;
                for (d = 0; d < 3; d ++) {
                    O->GradRho[d] += mass_j * dwk * dist[d] / r;
                }
            }
        }

        if(r > 0)
        {
            double fac = mass_j * dwk / r;
            double dv[3];
            double rot[3];
            int d;
            for(d = 0; d < 3; d ++) {
                dv[d] = I->Vel[d] - SphP_scratch->VelPred[3 * P[other].PI + d];
            }
            O->Div += -fac * dotproduct(dist, dv);

            crossproduct(dv, dist, rot);
            for(d = 0; d < 3; d ++) {
                O->Rot[d] += fac * rot[d];
            }
        }
    }
}

static int
density_haswork(int n, TreeWalk * tw)
{
    if(P[n].DensityIterationDone) return 0;

    if(P[n].TimeBin < 0) {
        endrun(9999, "TimeBin negative!\n use DensityIterationDone flag");
    }

    if(P[n].Type == 0 || P[n].Type == 5)
        return 1;

    return 0;
}

static void
density_postprocess(int i, TreeWalk * tw)
{
    if(P[i].Type == 0)
    {
        if(SPHP(i).Density > 0)
        {
            int PI = P[i].PI;
            MyFloat * DhsmlDens;
            if(DENSITY_GET_PRIV(tw)->DoEgyDensity)
                DhsmlDens = &(DENSITY_GET_PRIV(tw)->DhsmlDensityFactor[PI]);
            else
                DhsmlDens = &(SPHP(i).DhsmlEgyDensityFactor);

            *DhsmlDens *= P[i].Hsml / (NUMDIMS * SPHP(i).Density);
            *DhsmlDens = 1 / (1 + *DhsmlDens);

            /*Compute the EgyWeight factors, which are only useful for density independent SPH */
            if(DENSITY_GET_PRIV(tw)->DoEgyDensity) {
                const double EntPred = SphP_scratch->EntVarPred[P[i].PI];
                if(EntPred <= 0 || SPHP(i).EgyWtDensity <=0)
                    endrun(12, "Particle %d has bad predicted entropy: %g or EgyWtDensity: %g\n", i, EntPred, SPHP(i).EgyWtDensity);
                SPHP(i).DhsmlEgyDensityFactor *= P[i].Hsml/ (NUMDIMS * SPHP(i).EgyWtDensity);
                SPHP(i).DhsmlEgyDensityFactor *= - (*DhsmlDens);
                SPHP(i).EgyWtDensity /= EntPred;
            }

            MyFloat * Rot = DENSITY_GET_PRIV(tw)->Rot[PI];
            SPHP(i).CurlVel = sqrt(Rot[0] * Rot[0] + Rot[1] * Rot[1] + Rot[2] * Rot[2]) / SPHP(i).Density;

            SPHP(i).DivVel /= SPHP(i).Density;
        }
        else if(P[i].NumNgb > 0) {
            endrun(12, "Particle %d has bad density: %g\n", i, SPHP(i).Density);
        }
    }

    /* This is slightly more complicated so we put it in a different function */
    if(DENSITY_GET_PRIV(tw)->update_hsml)
        density_check_neighbours(i, tw);
}

void density_check_neighbours (int i, TreeWalk * tw) {
    /* now check whether we had enough neighbours */

    double desnumngb = All.DesNumNgb;

    if(All.BlackHoleOn && P[i].Type == 5)
        desnumngb = All.DesNumNgb * All.BlackHoleNgbFactor;

    MyFloat * Left = DENSITY_GET_PRIV(tw)->Left;
    MyFloat * Right = DENSITY_GET_PRIV(tw)->Right;

    if(P[i].NumNgb < (desnumngb - All.MaxNumNgbDeviation) ||
            (P[i].NumNgb > (desnumngb + All.MaxNumNgbDeviation)))
    {
        /* need to redo this particle */
        if(P[i].DensityIterationDone) {
            /* should have been 0*/
            endrun(999993, "Already has DensityIterationDone set, bad memory intialization.");
        }

        /* This condition is here to prevent the density code looping forever if it encounters
         * multiple particles at the same position. If this happens you likely have worse
         * problems anyway, so warn also. */
        if((Right[i] - Left[i]) < 1.0e-3 * Left[i])
        {
            /* If this happens probably the exchange is screwed up and all your particles have moved to (0,0,0)*/
            message(1, "Very tight Hsml bounds for i=%d ID=%lu Hsml=%g Left=%g Right=%g Ngbs=%g Right-Left=%g pos=(%g|%g|%g)\n",
             i, P[i].ID, P[i].Hsml, Left[i], Right[i], P[i].NumNgb, Right[i] - Left[i], P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
            P[i].Hsml = Right[i];
            P[i].DensityIterationDone = 1;
            return;
        }

        /* If we need more neighbours, move the lower bound up. If we need fewer, move the upper bound down.*/
        if(P[i].NumNgb < desnumngb) {
                Left[i] = P[i].Hsml;
        } else {
                Right[i] = P[i].Hsml;
        }

        /* Next step is geometric mean of previous. */
        if(Right[i] < 0.99 * All.BoxSize && Left[i] > 0)
            P[i].Hsml = pow(0.5 * (pow(Left[i], 3) + pow(Right[i], 3)), 1.0 / 3);
        else
        {
            if(Right[i] > 0.99 * All.BoxSize && Left[i] == 0)
                endrun(8188, "Cannot occur. Check for memory corruption: L = %g R = %g N=%g.", Left[i], Right[i], P[i].NumNgb);

            /* If this is the first step we can be faster by increasing or decreasing current Hsml by a constant factor*/
            if(Right[i] > 0.99 * All.BoxSize && Left[i] > 0)
            {
                if(P[i].Type == 0 && fabs(P[i].NumNgb - desnumngb) < 0.5 * desnumngb)
                {
                    MyFloat DensFac;
                    if(DENSITY_GET_PRIV(tw)->DoEgyDensity)
                        DensFac = DENSITY_GET_PRIV(tw)->DhsmlDensityFactor[P[i].PI];
                    else
                        DensFac = SPHP(i).DhsmlEgyDensityFactor;
                    double fac = 1 - (P[i].NumNgb - desnumngb) / (NUMDIMS * P[i].NumNgb) * DensFac;

                    if(fac < 1.26)
                        P[i].Hsml *= fac;
                    else
                        P[i].Hsml *= 1.26;
                }
                else
                    P[i].Hsml *= 1.26;
            }

            if(Right[i] < 0.99*All.BoxSize && Left[i] == 0)
            {
                if(P[i].Type == 0 && fabs(P[i].NumNgb - desnumngb) < 0.5 * desnumngb)
                {
                    MyFloat DensFac;
                    if(DENSITY_GET_PRIV(tw)->DoEgyDensity)
                        DensFac = DENSITY_GET_PRIV(tw)->DhsmlDensityFactor[P[i].PI];
                    else
                        DensFac = SPHP(i).DhsmlEgyDensityFactor;

                    double fac = 1 - (P[i].NumNgb - desnumngb) / (NUMDIMS * P[i].NumNgb) * DensFac;

                    if(fac > 1 / 1.26)
                        P[i].Hsml *= fac;
                    else
                        P[i].Hsml /= 1.26;
                }
                else
                    P[i].Hsml /= 1.26;
            }
        }

        if(All.BlackHoleOn && P[i].Type == 5)
            if(Left[i] > All.BlackHoleMaxAccretionRadius)
            {
                P[i].Hsml = All.BlackHoleMaxAccretionRadius;
                P[i].DensityIterationDone = 1;
            }

        if(Right[i] < All.MinGasHsml) {
            P[i].Hsml = All.MinGasHsml;
            P[i].DensityIterationDone = 1;
        }
    }
    else {
        /* We might have got here by serendipity, without bounding.*/
        if(All.BlackHoleOn && P[i].Type == 5)
            if(P[i].Hsml > All.BlackHoleMaxAccretionRadius)
                P[i].Hsml = All.BlackHoleMaxAccretionRadius;
        if(P[i].Hsml < All.MinGasHsml)
            P[i].Hsml = All.MinGasHsml;
        P[i].DensityIterationDone = 1;
    }

    if(DENSITY_GET_PRIV(tw)->NIteration >= MAXITER - 10)
    {
         message(1, "i=%d ID=%lu Hsml=%g Left=%g Right=%g Ngbs=%g Right-Left=%g\n   pos=(%g|%g|%g)\n",
             i, P[i].ID, P[i].Hsml, Left[i], Right[i],
             P[i].NumNgb, Right[i] - Left[i], P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
    }

    if(!P[i].DensityIterationDone) {
        int tid = omp_get_thread_num();
        DENSITY_GET_PRIV(tw)->NPLeft[tid] ++;
    }
}

