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
#include "metal_return.h"
#include "densitykernel.h"
#include "density.h"
#include "winds.h"
#include "utils/spinlocks.h"

/*! \file metal_return.c
 *  \brief Compute the mass return rate of metals from stellar evolution.
 *
 *  This file returns metals from stars with some delay.
 *  Delayed sources followed are AGB stars, SNII and Sn1a.
 *  Mass from each type of star is stored as a separate value.
 *  Since the species-specific yields do not affect anything
 *  (the cooling function depends only on total metallicity),
 *  actual species-specific yields are *not* specified and can
 *  be given in post-processing.
 */

static struct metal_return_params
{
} MetalParams;

/*Set the parameters of the hydro module*/
void
set_metal_return_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
    }
    MPI_Bcast(&MetalParams, sizeof(struct metal_return_params), MPI_BYTE, 0, MPI_COMM_WORLD);
}

struct MetalReturnPriv {
    double atime;
    struct SpinLocks * spin;
//    struct Yields
};

#define METALS_GET_PRIV(tw) ((struct MetalReturnPriv*) ((tw)->priv))

typedef struct {
    TreeWalkQueryBase base;
    MyFloat Metallicity[NMETALS];
    MyFloat Mass;
} TreeWalkQueryMetals;

typedef struct {
    TreeWalkResultBase base;
    MyFloat MassReturn[NMETALS];
    int Ninteractions;
} TreeWalkResultMetals;

typedef struct {
    TreeWalkNgbIterBase base;
} TreeWalkNgbIterMetals;

static int
metal_return_haswork(int n, TreeWalk * tw);

static void
metal_return_ngbiter(
    TreeWalkQueryMetals * I,
    TreeWalkResultMetals * O,
    TreeWalkNgbIterMetals * iter,
    LocalTreeWalk * lv
   );

static void
metal_return_copy(int place, TreeWalkQueryMetals * input, TreeWalk * tw);

static void
metal_return_reduce(int place, TreeWalkResultMetals * result, enum TreeWalkReduceMode mode, TreeWalk * tw);

/*! This function is the driver routine for the calculation of metal return. */
void
metal_return(const ActiveParticles * act, const double atime, const ForceTree * const tree)
{
    TreeWalk tw[1] = {{0}};

    struct MetalReturnPriv priv[1];

    tw->ev_label = "METALS";
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw->ngbiter = (TreeWalkNgbIterFunction) metal_return_ngbiter;
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterMetals);
    tw->haswork = metal_return_haswork;
    tw->fill = (TreeWalkFillQueryFunction) metal_return_copy;
    tw->reduce = (TreeWalkReduceResultFunction) metal_return_reduce;
    tw->postprocess = NULL;
    tw->query_type_elsize = sizeof(TreeWalkQueryMetals);
    tw->result_type_elsize = sizeof(TreeWalkResultMetals);
    tw->tree = tree;
    tw->priv = priv;

    if(!tree->hmax_computed_flag)
        endrun(5, "Metal called before hmax computed\n");
    /* Initialize some time factors*/
    METALS_GET_PRIV(tw)->atime = atime;

    priv->spin = init_spinlocks(SlotsManager->info[0].size);
    treewalk_run(tw, act->ActiveParticle, act->NumActiveParticle);
    free_spinlocks(priv->spin);

    /* collect some timing information */
    walltime_measure("/SPH/Metals");
}

static void
metal_return_copy(int place, TreeWalkQueryMetals * input, TreeWalk * tw)
{
    int j;
    for(j = 0; j< NMETALS; j++)
        input->Metallicity[j] = STARP(place).Metallicity[j];
    input->Mass = P[place].Mass;
}

static void
metal_return_reduce(int place, TreeWalkResultMetals * result, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    int j;
    /* Conserve mass returned*/
    P[place].Mass -= result->MassReturn[Total];
    /* TODO: What to do about the enrichment of the star particle?*/
    for(j = 0; j < NMETALS; j++)
        STARP(place).Metallicity[j] += result->MassReturn[j] / P[place].Mass;
}

/*! This function is the 'core' of the SPH force computation. A target
 *  particle is specified which may either be local, or reside in the
 *  communication buffer.
 */
static void
metal_return_ngbiter(
    TreeWalkQueryMetals * I,
    TreeWalkResultMetals * O,
    TreeWalkNgbIterMetals * iter,
    LocalTreeWalk * lv
   )
{
    if(iter->base.other == -1) {
        /* Only return metals to gas*/
        iter->base.mask = 1;
        /* Use symmetric treewalk because in practice we ignore the Hsml of the star.
         * So I-> Hsml = 0*/
        iter->base.Hsml = 0;
        iter->base.symmetric = NGB_TREEFIND_SYMMETRIC;

        /* initialize variables before SPH loop is started */
        int j;
        for(j = 0; j < NMETALS; j++)
            O->MassReturn[j] = 0;
        return;
    }

    int other = iter->base.other;
    double r2 = iter->base.r2;

    if(P[other].Mass == 0) {
        endrun(12, "Encountered zero mass particle during hydro;"
                  " We haven't implemented tracer particles and this shall not happen\n");
    }

    /* Wind particles do not interact hydrodynamically: don't receive metal mass.*/
    if(winds_is_particle_decoupled(other))
        return;

    DensityKernel kernel_j;

    density_kernel_init(&kernel_j, P[other].Hsml, GetDensityKernelType());

    if(r2 > 0 && r2 < kernel_j.HH)
    {
        /* Compute the yields*/
        /* If this is the closest star, add the metals*/
        lock_spinlock(other, METALS_GET_PRIV(lv->tw)->spin);
        /* Add metals weighted by SPH kernel*/
        unlock_spinlock(other, METALS_GET_PRIV(lv->tw)->spin);
    }
    O->Ninteractions++;
}

/* Only stars return metals to the gas*/
static int
metal_return_haswork(int i, TreeWalk * tw)
{
    return P[i].Type == 4;
}
