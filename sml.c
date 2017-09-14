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
} TreeWalkNgbIterSml;

typedef struct
{
    TreeWalkQueryBase base;
    MyFloat Vel[3];
    MyFloat Hsml;
#ifdef WINDS
    MyFloat DelayTime;
#endif
    int Type;
} TreeWalkQuerySml;

typedef struct {
    TreeWalkResultBase base;
    MyFloat Rho;
    MyFloat DhsmlDensity;
    MyFloat Ngb;
    int Ninteractions;

} TreeWalkResultSml;

struct SmlPriv {
    double *Left, *Right;
    int NIteration;
    int NPLeft;
};

#define SML_GET_PRIV(tw) ((struct SmlPriv*) ((tw)->priv))

static void
sml_ngbiter(
        TreeWalkQuerySml * I,
        TreeWalkResultSml * O,
        TreeWalkNgbIterSml * iter,
        LocalTreeWalk * lv);

static int sml_haswork(int n, TreeWalk * tw);
static void sml_postprocess(int i, TreeWalk * tw);
static void sml_check_neighbours(int i, TreeWalk * tw);

static void sml_reduce(int place, TreeWalkResultSml * remote, enum TreeWalkReduceMode mode, TreeWalk * tw);
static void sml_copy(int place, TreeWalkQuerySml * I, TreeWalk * tw);

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

void compute_sml(void)
{
    if(!All.DensityOn) return;

    TreeWalk tw[1] = {0};
    struct SmlPriv priv[1];

    tw->ev_label = "SML";
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterSml);
    tw->ngbiter = (TreeWalkNgbIterFunction) sml_ngbiter;
    tw->haswork = sml_haswork;
    tw->fill = (TreeWalkFillQueryFunction) sml_copy;
    tw->reduce = (TreeWalkReduceResultFunction) sml_reduce;
    tw->postprocess = (TreeWalkProcessFunction) sml_postprocess;

    tw->UseNodeList = 1;
    tw->query_type_elsize = sizeof(TreeWalkQuerySml);
    tw->result_type_elsize = sizeof(TreeWalkResultSml);
    tw->priv = priv;

    int i;
    int64_t ntot = 0;

    double timeall = 0;
    double timecomp, timecomm, timewait;

    walltime_measure("/Misc");

    SML_GET_PRIV(tw)->Left = (double *) mymalloc("SML_GET_PRIV(tw)->Left", NumPart * sizeof(double));
    SML_GET_PRIV(tw)->Right = (double *) mymalloc("SML_GET_PRIV(tw)->Right", NumPart * sizeof(double));

    SML_GET_PRIV(tw)->NIteration = 0;

    /* this has to be done before treewalk so that
     * all particles are ran for the first loop.
     * The iteration will gradually turn SmlIterationDone on more particles.
     * */
    #pragma omp parallel for
    for(i = 0; i < NumActiveParticle; i++)
    {
        const int p_i = ActiveParticle[i];
        P[p_i].SmlIterationDone = 0;
        SML_GET_PRIV(tw)->Left[p_i] = 0;
        SML_GET_PRIV(tw)->Right[p_i] = 0;

    }

    /* allocate buffers to arrange communication */

    walltime_measure("/SPH/Sml/Init");

    /* we will repeat the whole thing for those particles where we didn't find enough neighbours */
    do {
        SML_GET_PRIV(tw)->NPLeft = 0;

        treewalk_run(tw, ActiveParticle, NumActiveParticle);
        sumup_large_ints(1, &SML_GET_PRIV(tw)->NPLeft, &ntot);

        if(ntot == 0) break;

        SML_GET_PRIV(tw)->NIteration ++;
        /*
        if(ntot < 1 ) {
            foreach(ActiveParticle)
            {
                if(sml_haswork(i) && !P[i].SmlIterationDone) {
                    message
                        (1, "i=%d task=%d ID=%llu type=%d, Hsml=%g SML_GET_PRIV(tw)->Left=%g SML_GET_PRIV(tw)->Right=%g Ngbs=%g SML_GET_PRIV(tw)->Right-SML_GET_PRIV(tw)->Left=%g\n   pos=(%g|%g|%g)\n",
                         i, ThisTask, P[i].ID, P[i].Type, P[i].Hsml, SML_GET_PRIV(tw)->Left[i], SML_GET_PRIV(tw)->Right[i],
                         (float) P[i].NumNgb, SML_GET_PRIV(tw)->Right[i] - SML_GET_PRIV(tw)->Left[i], P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
                }
            }

        }
        */

        if(SML_GET_PRIV(tw)->NIteration > 0) {
            message(0, "ngb iteration %d: need to repeat for %ld particles.\n", SML_GET_PRIV(tw)->NIteration, ntot);
        }

        if(SML_GET_PRIV(tw)->NIteration > MAXITER) {
            endrun(1155, "failed to converge in neighbour iteration in density()\n");
        }
    } while(1);

    myfree(SML_GET_PRIV(tw)->Right);
    myfree(SML_GET_PRIV(tw)->Left);


    /* collect some timing information */

    timeall = walltime_measure(WALLTIME_IGNORE);

    timecomp = tw->timecomp3 + tw->timecomp1 + tw->timecomp2;
    timewait = tw->timewait1 + tw->timewait2;
    timecomm = tw->timecommsumm1 + tw->timecommsumm2;

    walltime_add("/SPH/Sml/Compute", timecomp);
    walltime_add("/SPH/Sml/Wait", timewait);
    walltime_add("/SPH/Sml/Comm", timecomm);
    walltime_add("/SPH/Sml/Misc", timeall - (timecomp + timewait + timecomm));
}

static void
sml_copy(int place, TreeWalkQuerySml * I, TreeWalk * tw)
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
sml_reduce(int place, TreeWalkResultSml * remote, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    TREEWALK_REDUCE(P[place].NumNgb, remote->Ngb);

    /* these will be added */
    P[place].GravCost += All.HydroCostFactor * All.cf.a * remote->Ninteractions;

    if(P[place].Type == 0)
    {
        TREEWALK_REDUCE(SPHP(place).Density, remote->Rho);
        TREEWALK_REDUCE(SPHP(place).DhsmlDensityFactor, remote->DhsmlDensity);
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
sml_ngbiter(
        TreeWalkQuerySml * I,
        TreeWalkResultSml * O,
        TreeWalkNgbIterSml * iter,
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
    }

    /* some performance measures not currently used */
    O->Ninteractions ++;
}

static int
sml_haswork(int n, TreeWalk * tw)
{
    if(P[n].SmlIterationDone) return 0;

    if(P[n].TimeBin < 0) {
        endrun(9999, "TimeBin negative!\n use SmlIterationDone flag");
    }

    if(P[n].Type == 0 || P[n].Type == 5)
        return 1;

    return 0;
}

static void
sml_postprocess(int i, TreeWalk * tw)
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

        }

    }

    /* This is slightly more complicated so we put it in a different function */
    /* FIXME: It may make sense to have a seperate tree walk that calculates Hsml only. */
    sml_check_neighbours(i, tw);
}

void sml_check_neighbours (int i, TreeWalk * tw) {
    /* now check whether we had enough neighbours */

    double desnumngb = All.DesNumNgb;

#ifdef BLACK_HOLES
    if(P[i].Type == 5)
        desnumngb = All.DesNumNgb * All.BlackHoleNgbFactor;
#endif

    if(P[i].NumNgb < (desnumngb - All.MaxNumNgbDeviation) ||
            (P[i].NumNgb > (desnumngb + All.MaxNumNgbDeviation)
             && P[i].Hsml > (1.01 * All.MinGasHsml)))
    {
        /* need to redo this particle */
        if(P[i].SmlIterationDone) {
            /* should have been 0*/
            endrun(999993, "Already has SmlIterationDone set, bad memory intialization.");
        }

        if(SML_GET_PRIV(tw)->Left[i] > 0 && SML_GET_PRIV(tw)->Right[i] > 0)
            if((SML_GET_PRIV(tw)->Right[i] - SML_GET_PRIV(tw)->Left[i]) < 1.0e-3 * SML_GET_PRIV(tw)->Left[i])
            {
                /* this one should be ok */
                P[i].SmlIterationDone = 1;
                return;
            }

        if(P[i].NumNgb < (desnumngb - All.MaxNumNgbDeviation))
            SML_GET_PRIV(tw)->Left[i] = DMAX(P[i].Hsml, SML_GET_PRIV(tw)->Left[i]);
        else
        {
            if(SML_GET_PRIV(tw)->Right[i] != 0)
            {
                if(P[i].Hsml < SML_GET_PRIV(tw)->Right[i])
                    SML_GET_PRIV(tw)->Right[i] = P[i].Hsml;
            }
            else
                SML_GET_PRIV(tw)->Right[i] = P[i].Hsml;
        }

        if(SML_GET_PRIV(tw)->Right[i] > 0 && SML_GET_PRIV(tw)->Left[i] > 0)
            P[i].Hsml = pow(0.5 * (pow(SML_GET_PRIV(tw)->Left[i], 3) + pow(SML_GET_PRIV(tw)->Right[i], 3)), 1.0 / 3);
        else
        {
            if(SML_GET_PRIV(tw)->Right[i] == 0 && SML_GET_PRIV(tw)->Left[i] == 0)
                endrun(8188, "Cannot occur. Check for memory corruption.");	/* can't occur */

            if(SML_GET_PRIV(tw)->Right[i] == 0 && SML_GET_PRIV(tw)->Left[i] > 0)
            {
                if(P[i].Type == 0 && fabs(P[i].NumNgb - desnumngb) < 0.5 * desnumngb)
                {
                    double fac = 1 - (P[i].NumNgb -
                            desnumngb) / (NUMDIMS * P[i].NumNgb) *
                        SPHP(i).DhsmlDensityFactor;

                    if(fac < 1.26)
                        P[i].Hsml *= fac;
                    else
                        P[i].Hsml *= 1.26;
                }
                else
                    P[i].Hsml *= 1.26;
            }

            if(SML_GET_PRIV(tw)->Right[i] > 0 && SML_GET_PRIV(tw)->Left[i] == 0)
            {
                if(P[i].Type == 0 && fabs(P[i].NumNgb - desnumngb) < 0.5 * desnumngb)
                {
                    double fac = 1 - (P[i].NumNgb -
                            desnumngb) / (NUMDIMS * P[i].NumNgb) *
                        SPHP(i).DhsmlDensityFactor;

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
            if(SML_GET_PRIV(tw)->Left[i] > All.BlackHoleMaxAccretionRadius)
            {
                /* this will stop the search for a new BH smoothing length in the next iteration */
                P[i].Hsml = SML_GET_PRIV(tw)->Left[i] = SML_GET_PRIV(tw)->Right[i] = All.BlackHoleMaxAccretionRadius;
            }
#endif

    }
    else {
        P[i].SmlIterationDone = 1;
    }

    if(SML_GET_PRIV(tw)->NIteration >= MAXITER - 10)
    {
         message(1, "i=%d task=%d ID=%lu Hsml=%g SML_GET_PRIV(tw)->Left=%g SML_GET_PRIV(tw)->Right=%g Ngbs=%g SML_GET_PRIV(tw)->Right-SML_GET_PRIV(tw)->Left=%g\n   pos=(%g|%g|%g)\n",
             i, ThisTask, P[i].ID, P[i].Hsml, SML_GET_PRIV(tw)->Left[i], SML_GET_PRIV(tw)->Right[i],
             (float) P[i].NumNgb, SML_GET_PRIV(tw)->Right[i] - SML_GET_PRIV(tw)->Left[i], P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
    }

    if(!P[i].SmlIterationDone) {
#pragma omp atomic
        SML_GET_PRIV(tw)->NPLeft ++;
    }
}

