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
#include "mymalloc.h"
#include "endrun.h"
#include "timestep.h"
#include "system.h"

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
    MyFloat Vel[3];
    MyFloat Hsml;
#ifdef WINDS
    MyFloat DelayTime;
#endif
    int Type;
} TreeWalkQueryDensity;

typedef struct {
    TreeWalkResultBase base;
#ifdef DENSITY_INDEPENDENT_SPH
    MyFloat EgyRho;
    MyFloat DhsmlEgyDensity;
#endif
    MyFloat Rho;
    MyFloat DhsmlDensity;
    MyFloat Ngb;
    MyFloat Div;
    MyFloat Rot[3];

    int Ninteractions;


#ifdef SPH_GRAD_RHO
    MyFloat GradRho[3];
#endif
} TreeWalkResultDensity;

struct DensityPriv {
    /* placeholder empty to serve as an example */
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

void density(void)
{
    if(!All.DensityOn)
        return;

    TreeWalk tw[1] = {0};
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

    int i;
    int64_t ntot = 0;

    double timeall = 0;
    double timecomp, timecomm, timewait;

    walltime_measure("/Misc");

    treewalk_run(tw, ActiveParticle, NumActiveParticle);

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
        sph_VelPred(place, I->Vel);
    }

#ifdef WINDS
    I->DelayTime = SPHP(place).DelayTime;
#endif

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
        TREEWALK_REDUCE(SPHP(place).DhsmlDensityFactor, remote->DhsmlDensity);
#ifdef DENSITY_INDEPENDENT_SPH
        TREEWALK_REDUCE(SPHP(place).EgyWtDensity, remote->EgyRho);
        TREEWALK_REDUCE(SPHP(place).DhsmlEgyDensityFactor, remote->DhsmlEgyDensity);
#endif

        TREEWALK_REDUCE(SPHP(place).DivVel, remote->Div);
        TREEWALK_REDUCE(SPHP(place).Rot[0], remote->Rot[0]);
        TREEWALK_REDUCE(SPHP(place).Rot[1], remote->Rot[1]);
        TREEWALK_REDUCE(SPHP(place).Rot[2], remote->Rot[2]);

#ifdef SPH_GRAD_RHO
        TREEWALK_REDUCE(SPHP(place).GradRho[0], remote->GradRho[0]);
        TREEWALK_REDUCE(SPHP(place).GradRho[1], remote->GradRho[1]);
        TREEWALK_REDUCE(SPHP(place).GradRho[2], remote->GradRho[2]);
#endif

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
    int other = iter->base.other;
    double r = iter->base.r;
    double r2 = iter->base.r2;
    double * dist = iter->base.dist;

#ifdef WINDS
    if(HAS(All.WindModel, WINDS_DECOUPLE_SPH)) {
        if(SPHP(other).DelayTime > 0)	/* partner is a wind particle */
            if(!(I->DelayTime > 0))	/* if I'm not wind, then ignore the wind particle */
                return;
    }
#endif

    if(P[other].Mass == 0)
        return;

    if(r2 < iter->kernel.HH)
    {

        double u = r * iter->kernel.Hinv;
        double wk = density_kernel_wk(&iter->kernel, u);
        double dwk = density_kernel_dwk(&iter->kernel, u);

        double mass_j = P[other].Mass;

        O->Rho += (mass_j * wk);
        O->Ngb += wk * iter->kernel_volume;

        /* Hinv is here because O->DhsmlDensity is drho / dH.
         * nothing to worry here */
        O->DhsmlDensity += mass_j * density_kernel_dW(&iter->kernel, u, wk, dwk);

#ifdef DENSITY_INDEPENDENT_SPH
        double EntPred = EntropyPred(other);
        O->EgyRho += mass_j * EntPred * wk;
        O->DhsmlEgyDensity += mass_j * EntPred * density_kernel_dW(&iter->kernel, u, wk, dwk);
#endif

#ifdef SPH_GRAD_RHO
        if(r > 0)
        {
            int d;
            for (d = 0; d < 3; d ++) {
                O->GradRho[d] += mass_j * dwk * dist[d] / r;
            }
        }
#endif

        if(r > 0)
        {
            double fac = mass_j * dwk / r;
            double dv[3];
            double rot[3];
            int d;
            sph_VelPred(other, dv);
            for(d = 0; d < 3; d ++) {
                dv[d] = I->Vel[d] - dv[d];
            }
            O->Div += -fac * dotproduct(dist, dv);

            crossproduct(dv, dist, rot);
            for(d = 0; d < 3; d ++) {
                O->Rot[d] += fac * rot[d];
            }
        }
    }

    /* some performance measures not currently used */
    O->Ninteractions ++;
}

static int
density_haswork(int n, TreeWalk * tw)
{
    /* density of BH is computed for bh feedback */
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
            SPHP(i).DhsmlDensityFactor *= P[i].Hsml / (NUMDIMS * SPHP(i).Density);
            if(SPHP(i).DhsmlDensityFactor > -0.9)	/* note: this would be -1 if only a single particle at zero lag is found */
                SPHP(i).DhsmlDensityFactor = 1 / (1 + SPHP(i).DhsmlDensityFactor);
            else
                SPHP(i).DhsmlDensityFactor = 1;

#ifdef DENSITY_INDEPENDENT_SPH
            const double EntPred = EntropyPred(i);
            if((EntPred > 0) && (SPHP(i).EgyWtDensity>0))
            {
                SPHP(i).DhsmlEgyDensityFactor *= P[i].Hsml/ (NUMDIMS * SPHP(i).EgyWtDensity);
                SPHP(i).DhsmlEgyDensityFactor *= -SPHP(i).DhsmlDensityFactor;
                SPHP(i).EgyWtDensity /= EntPred;
            } else {
                SPHP(i).DhsmlEgyDensityFactor=0;
                SPHP(i).EgyWtDensity=0;
            }
#endif

            SPHP(i).CurlVel = sqrt(SPHP(i).Rot[0] * SPHP(i).Rot[0] +
                    SPHP(i).Rot[1] * SPHP(i).Rot[1] +
                    SPHP(i).Rot[2] * SPHP(i).Rot[2]) / SPHP(i).Density;

            SPHP(i).DivVel /= SPHP(i).Density;

        }
    }

}
