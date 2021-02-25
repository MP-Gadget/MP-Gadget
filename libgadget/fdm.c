#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_errno.h>
#include <omp.h>

#include "allvars.h"
#include "physconst.h"
#include "walltime.h"
#include "slotsmanager.h"
#include "treewalk.h"
#include "fdm.h"
#include "densitykernel.h"
#include "density.h"
#include "cosmology.h"
#include "utils/spinlocks.h"

#define FDMMAXITER 400

static struct fdm_params
{
    double FDM22;
    double FDMMaxNgbDeviation;
} FdmParams;


/*Set the parameters of the fdm module*/
void
set_fdm_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        FdmParams.FDM22 = param_get_double(ps, "FDM22");
        FdmParams.FDMMaxNgbDeviation = param_get_double(ps, "FDMMaxNgbDeviation");
    }
    MPI_Bcast(&FdmParams, sizeof(struct fdm_params), MPI_BYTE, 0, MPI_COMM_WORLD);
}

typedef struct {
    TreeWalkNgbIterBase base;
    DensityKernel kernel;
    double kernel_volume;
} TreeWalkNgbIterDMDensity;

typedef struct
{
    TreeWalkQueryBase base;
    MyFloat Hsml;
} TreeWalkQueryDMDensity;

typedef struct {
    TreeWalkResultBase base;
    MyFloat Ngb;
    MyFloat Rho;
    MyFloat DhsmlDensity;
} TreeWalkResultDMDensity;

struct DMDensityPriv {
    /* Current number of neighbours*/
    MyFloat *NumNgb;
    /* Lower and upper bounds on smoothing length*/
    MyFloat *Left, *Right, *DhsmlDensity;
    size_t *NPLeft;
    int **NPRedo;
    /*!< Desired number of SPH neighbours */
    double DesNumNgb;
};

#define DM_DENSITY_GET_PRIV(tw) ((struct DMDensityPriv*) ((tw)->priv))

static int
dm_density_haswork(int i, TreeWalk * tw)
{
    if(P[i].Type == 1)
        return 1;
    return 0;
}

static void
dm_density_copy(int place, TreeWalkQueryDMDensity * I, TreeWalk * tw)
{
    I->Hsml = P[place].Hsml;
}

static void
dm_density_reduce(int place, TreeWalkResultDMDensity * remote, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    int pi = P[place].PI;
    TREEWALK_REDUCE(DM_DENSITY_GET_PRIV(tw)->NumNgb[pi], remote->Ngb);
    TREEWALK_REDUCE(DM_DENSITY_GET_PRIV(tw)->DhsmlDensity[pi], remote->DhsmlDensity);
    TREEWALK_REDUCE(FDMP(place).Density, remote->Rho);
}

void dm_density_check_neighbours (int i, TreeWalk * tw)
{
    /* now check whether we had enough neighbours */

    double desnumngb = DM_DENSITY_GET_PRIV(tw)->DesNumNgb;

    MyFloat * Left = DM_DENSITY_GET_PRIV(tw)->Left;
    MyFloat * Right = DM_DENSITY_GET_PRIV(tw)->Right;
    MyFloat * NumNgb = DM_DENSITY_GET_PRIV(tw)->NumNgb;
    MyFloat * DhsmlDensity = DM_DENSITY_GET_PRIV(tw)->DhsmlDensity;

    int pi = P[i].PI;
    int tid = omp_get_thread_num();   

    if(NumNgb[pi] < (desnumngb - FdmParams.FDMMaxNgbDeviation) || (NumNgb[pi] > (desnumngb + FdmParams.FDMMaxNgbDeviation)))
    {

        /* This condition is here to prevent the density code looping forever if it encounters
         * multiple particles at the same position. If this happens you likely have worse
         * problems anyway, so warn also. */
        if((Right[pi] - Left[pi]) < 1.0e-3 * Left[pi])
        {
            /* If this happens probably the exchange is screwed up and all your particles have moved to (0,0,0)*/
            message(1, "Very tight Hsml bounds for i=%d ID=%lu type %d Hsml=%g Left=%g Right=%g Ngbs=%g des = %g Right-Left=%g pos=(%g|%g|%g) deviation=%g \n",
             i, P[i].ID, P[i].Type, P[i].Hsml, Left[pi], Right[pi], NumNgb[pi], desnumngb, Right[pi] - Left[pi], P[i].Pos[0], P[i].Pos[1], P[i].Pos[2], FdmParams.FDMMaxNgbDeviation);
            P[i].Hsml = Right[pi];
            return;
        }
        
        DhsmlDensity[pi] *= P[i].Hsml / (NUMDIMS * (FDMP(i).Density));
        DhsmlDensity[pi] = 1 / (1 + DhsmlDensity[pi]);
        /* We will also use this in quantum pressure calculation*/
        FDMP(i).DhsmlDensityFactor = DhsmlDensity[pi];

        /* If we need more neighbours, move the lower bound up. If we need fewer, move the upper bound down.*/
        if(NumNgb[pi] < desnumngb) {
                Left[pi] = P[i].Hsml;
        } else {
                Right[pi] = P[i].Hsml;
        }

        /* Next step is geometric mean of previous. */
        if((Right[pi] < tw->tree->BoxSize && Left[pi] > 0) || (P[i].Hsml * 1.26 > 0.99 * tw->tree->BoxSize))
            P[i].Hsml = pow(0.5 * (pow(Left[pi], 3) + pow(Right[pi], 3)), 1.0 / 3);
        else
        {
            if(!(Right[pi] < tw->tree->BoxSize) && Left[pi] == 0)
                endrun(8188, "Cannot occur. Check for memory corruption: i=%d pi %d L = %g R = %g N=%g. Type %d, Pos %g %g %g",
                       i, pi, Left[pi], Right[pi], NumNgb[pi], P[i].Type, P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
            
            double fac = 1.26;
            if(NumNgb[pi] > 0)
                fac = 1 - (NumNgb[pi] - desnumngb) / (NUMDIMS * NumNgb[pi]) * DhsmlDensity[pi];
            
            /* If this is the first step we can be faster by increasing or decreasing current Hsml by a constant factor*/
            if(Right[pi] > 0.99 * tw->tree->BoxSize && Left[pi] > 0) {
                if(fac > 1.26)
                    fac = 1.26;
            }
            if(Right[pi] < 0.99*tw->tree->BoxSize && Left[pi] == 0) {
                if(fac < 0.33)
                    fac = 0.33;
            }
                
            P[i].Hsml *= fac;
        }
        /* More work needed: add this particle to the redo queue*/
        DM_DENSITY_GET_PRIV(tw)->NPRedo[tid][DM_DENSITY_GET_PRIV(tw)->NPLeft[tid]] = i;
        DM_DENSITY_GET_PRIV(tw)->NPLeft[tid] ++;
    }

    if(tw->Niteration >= FDMMAXITER-5)
    {
         message(1, "FDM iter=%d i=%d ID=%lu type=%d Hsml=%g dens=%g DhsmlDensity=%g Left=%g Right=%g Ngbs=%g targetNgb=%g\n",
             tw->Niteration, i, P[i].ID, P[i].Type, P[i].Hsml, FDMP(i).Density, DhsmlDensity[pi],Left[pi], Right[pi],
             NumNgb[pi],desnumngb);
    }
}

static void
dm_density_ngbiter(
        TreeWalkQueryDMDensity * I,
        TreeWalkResultDMDensity * O,
        TreeWalkNgbIterDMDensity * iter,
        LocalTreeWalk * lv)
{
    if(iter->base.other == -1) {
        const double h = I->Hsml;
        density_kernel_init(&iter->kernel, h, GetDensityKernelType());
        iter->kernel_volume = density_kernel_volume(&iter->kernel);

        iter->base.Hsml = h;
        iter->base.mask = 1+2; /* dm-only ? */
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        return;
    }
    const int other = iter->base.other;
    const double r = iter->base.r;
    const double r2 = iter->base.r2;

    if(P[other].Type == 1 && r2 < iter->kernel.HH)
    {
        const double u = r * iter->kernel.Hinv;
        double wk = density_kernel_wk(&iter->kernel, u);
        O->Ngb += wk * iter->kernel_volume;
        /* Hinv is here because O->DhsmlDensity is drho / dH.
         * nothing to worry here */
        O->Rho += P[other].Mass * wk;
        const double dwk = density_kernel_dwk(&iter->kernel, u);
        double density_dW = density_kernel_dW(&iter->kernel, u, wk, dwk);
        O->DhsmlDensity += P[other].Mass * density_dW;
    }
}

void
dm_density(const ActiveParticles * act, const ForceTree * const tree)
{
    TreeWalk tw[1] = {{0}};
    struct DMDensityPriv priv[1];

    tw->ev_label = "DM_DENSITY";
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterDMDensity);
    tw->ngbiter = (TreeWalkNgbIterFunction) dm_density_ngbiter;
    tw->haswork = dm_density_haswork;
    tw->fill = (TreeWalkFillQueryFunction) dm_density_copy;
    tw->reduce = (TreeWalkReduceResultFunction) dm_density_reduce;
    tw->postprocess = (TreeWalkProcessFunction) dm_density_check_neighbours;
    tw->query_type_elsize = sizeof(TreeWalkQueryDMDensity);
    tw->result_type_elsize = sizeof(TreeWalkResultDMDensity);
    tw->priv = priv;
    tw->tree = tree;

    int i;
    int64_t ntot = 0;

    priv->Left = (MyFloat *) mymalloc("DENS_PRIV->Left", SlotsManager->info[1].size * sizeof(MyFloat));
    priv->Right = (MyFloat *) mymalloc("DENS_PRIV->Right", SlotsManager->info[1].size * sizeof(MyFloat));
    priv->NumNgb = (MyFloat *) mymalloc("DENS_PRIV->NumNgb", SlotsManager->info[1].size * sizeof(MyFloat));
    priv->DhsmlDensity = (MyFloat *) mymalloc("DENS_PRIV->DhsmlDensity", SlotsManager->info[1].size * sizeof(MyFloat));
    priv->DesNumNgb = GetNumNgb(GetDensityKernelType());

    /* Init Left and Right: this has to be done before treewalk */
    memset(priv->NumNgb, 0, SlotsManager->info[1].size * sizeof(MyFloat));
    memset(priv->Left, 0, SlotsManager->info[1].size * sizeof(MyFloat));
    #pragma omp parallel for
    for(i = 0; i < SlotsManager->info[1].size; i++)
        priv->Right[i] = tree->BoxSize;
    
    /* allocate buffers to arrange communication */
    
    walltime_measure("/FDM/Density/Init");
    
    int NumThreads = omp_get_max_threads();
    priv->NPLeft = ta_malloc("NPLeft", size_t, NumThreads);
    priv->NPRedo = ta_malloc("NPRedo", int *, NumThreads);
    int alloc_high = 0;
    int * ReDoQueue = act->ActiveParticle;
    int size = act->NumActiveParticle;

    /* we will repeat the whole thing for those particles where we didn't find enough neighbours */
    do {
        /* The RedoQueue needs enough memory to store every particle on every thread, because
         * we cannot guarantee that the sph particles are evenly spread across threads!*/
        int * CurQueue = ReDoQueue;
        /* The ReDoQueue swaps between high and low allocations so we can have two allocated alternately*/
        if(!alloc_high) {
            ReDoQueue = (int *) mymalloc2("ReDoQueue", size * sizeof(int) * NumThreads);
            alloc_high = 1;
        }
        else {
            ReDoQueue = (int *) mymalloc("ReDoQueue", size * sizeof(int) * NumThreads);
            alloc_high = 0;
        }
        
        gadget_setup_thread_arrays(ReDoQueue, priv->NPRedo, priv->NPLeft, size, NumThreads);
        treewalk_run(tw, CurQueue, size);

        tw->haswork = NULL;
        /* Now done with the current queue*/
        if(tw->Niteration > 1)
            myfree(CurQueue);

        /* Set up the next queue*/
        size = gadget_compact_thread_arrays(ReDoQueue, priv->NPRedo, priv->NPLeft, NumThreads);

        sumup_large_ints(1, &size, &ntot);
        if(ntot == 0){
            myfree(ReDoQueue);
            break;
        }

        /*Shrink memory*/
        ReDoQueue = myrealloc(ReDoQueue, sizeof(int) * size);

#ifdef DEBUG
        if(ntot == 1 && size > 0 && tw->Niteration > 20 ) {
            int pp = ReDoQueue[0];
            message(1, "Remaining i=%d, t %d, pos %g %g %g, hsml: %g ngb: %g\n", pp, P[pp].Type, P[pp].Pos[0], P[pp].Pos[1], P[pp].Pos[2], P[pp].Hsml, priv->NumNgb[pp]);
        }
#endif
        if(tw->Niteration > FDMMAXITER) {
            endrun(1155, "failed to converge in neighbour iteration in density()\n");
        }
    } while(1);

    ta_free(priv->NPRedo);
    ta_free(priv->NPLeft);
    
    myfree(priv->DhsmlDensity);
    myfree(priv->NumNgb);
    myfree(priv->Right);
    myfree(priv->Left);

    double timeall = walltime_measure(WALLTIME_IGNORE);

    double timecomp = tw->timecomp3 + tw->timecomp1 + tw->timecomp2;
    double timewait = tw->timewait1 + tw->timewait2;
    double timecomm = tw->timecommsumm1 + tw->timecommsumm2;
    walltime_add("/FDM/Density/Compute", timecomp);
    walltime_add("/FDM/Density/Wait", timewait);
    walltime_add("/FDM/Density/Comm", timecomm);
    walltime_add("/FDM/Density/Misc", timeall - (timecomp + timewait + timecomm));

    return;
}

/***************************************************************************************************/
/* Now we start to do the QP iterations*/

typedef struct{
    TreeWalkQueryBase base;
    MyFloat Hsml;
    MyFloat Density;
} TreeWalkQueryDerivative;

typedef struct{
    TreeWalkResultBase base;
    MyFloat GradDensity[3];
    MyFloat LapDensity;
} TreeWalkResultDerivative;

typedef struct {
    TreeWalkNgbIterBase base;
    DensityKernel kernel_i;
} TreeWalkNgbIterDerivative;


static void
densderiv_copy(int place, TreeWalkQueryDerivative * I, TreeWalk * tw)
{
    I->Hsml = P[place].Hsml;
    I->Density = FDMP(place).Density;
}

static void
densderiv_reduce(int place, TreeWalkResultDerivative * remote, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    TREEWALK_REDUCE(FDMP(place).LapDensity, remote->LapDensity);
    int k;
    for(k=0;k<3;k++){
        TREEWALK_REDUCE(FDMP(place).GradDensity[k], remote->GradDensity[k]);
    }
}

static void
densderiv_ngbiter(
        TreeWalkQueryDerivative * I,
        TreeWalkResultDerivative * O,
        TreeWalkNgbIterDerivative * iter,
        LocalTreeWalk * lv)
{
    if(iter->base.other == -1) {
        iter->base.Hsml = I->Hsml;
        iter->base.mask = 2; /* dm-only ? */
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        
        density_kernel_init(&iter->kernel_i, I->Hsml, GetDensityKernelType());
        return;
    }
    
    const int other = iter->base.other;
    const double r = iter->base.r;
    const double r2 = iter->base.r2;
    const double * dist = iter->base.dist;
    
    if(P[other].Type != 1 || r2 <=0 || !(r2 < iter->kernel_i.HH))
        return;
    
    const double u = r * iter->kernel_i.Hinv;
    const double dwk = density_kernel_dwk(&iter->kernel_i, u);
    const double ddwk = density_kernel_ddwk(&iter->kernel_i, u);
    const double mass_j = P[other].Mass;
    
    double fac = (FDMP(other).Density - I->Density)/sqrt(I->Density*FDMP(other).Density);
    O->LapDensity += mass_j * fac * (ddwk + 3*dwk/r);
    int d;
    for(d = 0; d < 3; d++){
        O->GradDensity[d] += mass_j * dwk * fac * dist[d]/r;
    }
}

static void
densderiv_postprocess(int i, TreeWalk * tw)
{
    if(P[i].Type == 1){
        double gradsq = 0;
        int k;
        for(k = 0; k < 3; k++){
            gradsq += pow(FDMP(i).GradDensity[k],2);
        }
        FDMP(i).LapDensity -= gradsq/FDMP(i).Density;
    }
//     message(0,"deriv,ID=%lu,LapDensity=%g,gradsq=%g,density=%g",P[i].ID,FDMP(i).LapDensity,
//             FDMP(i).GradDensity[0],FDMP(i).Density);
}

/*************************************************************************/
typedef struct{
    TreeWalkQueryBase base;
    MyFloat Hsml;
} TreeWalkQueryQP;

typedef struct{
    TreeWalkResultBase base;
    MyFloat Acc[3];
} TreeWalkResultQP;

typedef struct {
    TreeWalkNgbIterBase base;
    DensityKernel kernel_i;
} TreeWalkNgbIterQP;

static void
qp_copy(int place, TreeWalkQueryQP * I, TreeWalk * tw)
{
    I->Hsml = P[place].Hsml;
}

static void
qp_reduce(int place, TreeWalkResultQP * remote, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    int k;
    for(k=0;k<3;k++){
        TREEWALK_REDUCE(FDMP(place).QPAccel[k], remote->Acc[k]);
    }
//     message(0,"Reduced,ID=%lu,QPAccel=(%g|%g|%g),GravAccel=(%g|%g|%g)",P[place].ID,FDMP(place).QPAccel[0],FDMP(place).QPAccel[1],
//             FDMP(place).QPAccel[2],P[place].GravAccel[0],P[place].GravAccel[1],P[place].GravAccel[2]);
}

static void
qp_ngbiter(
        TreeWalkQueryQP * I,
        TreeWalkResultQP * O,
        TreeWalkNgbIterQP * iter,
        LocalTreeWalk * lv)
{
    if(iter->base.other == -1) {
        iter->base.Hsml = I->Hsml;
        iter->base.mask = 1+2; /* dm-only ? */
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        
        O->Acc[0] = O->Acc[1] = O->Acc[2] = 0;
        density_kernel_init(&iter->kernel_i, I->Hsml, GetDensityKernelType());
        return;
    }
    
    const int other = iter->base.other;
    const double r = iter->base.r;
    const double r2 = iter->base.r2;
    const double * dist = iter->base.dist;
    
    if(P[other].Type != 1 || r2 <=0 || !(r2 < iter->kernel_i.HH))
        return;
    
    const double u = r * iter->kernel_i.Hinv;
    const double dwk = density_kernel_dwk(&iter->kernel_i, u);
    const double mass_j = P[other].Mass;
    const double rho_j = FDMP(other).Density;
    
    double h_bar = PLANCK/(2*M_PI); /* in cgs unit */
    double m_boson = FdmParams.FDM22*1.0e-22*eVinergs/(LIGHTCGS*LIGHTCGS); /*to cgs*/
    double prefac = pow((h_bar/m_boson),2)/2.0;
    /* convert to internal unit*/
    prefac *= 1.0/pow((All.UnitVelocity_in_cm_per_s*All.UnitLength_in_cm),2);
    
    double gradsq = 0;
    int d;
    for(d = 0; d < 3; d++){
        gradsq += pow(FDMP(other).GradDensity[d],2);
    }
    double fac = FDMP(other).LapDensity/(2*rho_j) - gradsq/(4*rho_j*rho_j);
    fac *= prefac*(mass_j/rho_j)*FDMP(other).DhsmlDensityFactor;
    fac /= (All.cf.a*All.cf.a);
    for(d = 0; d < 3; d++){
        O->Acc[d] += fac * dwk * dist[d]/r;
    }
//     if(fac1*dwk>1e20 || fac1*dwk<1e-20)
//         endrun(5,"caught error, prefac = %g, fac0 = %g, fac1 = %g, dwk = %g, gradsq = %g, dhdrho = %g",prefac,fac0,fac1,
//                dwk,gradsq,FDMP(other).DhsmlDensityFactor);
}


void
quantum_pressure(const ActiveParticles * act, const ForceTree * const tree)
{
    /*************************************************************************/
    TreeWalk tw_deriv[1] = {{0}};

    tw_deriv->ev_label = "DM_DensDeriv";
    tw_deriv->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw_deriv->ngbiter_type_elsize = sizeof(TreeWalkNgbIterDerivative);
    tw_deriv->ngbiter = (TreeWalkNgbIterFunction) densderiv_ngbiter;
    tw_deriv->haswork = dm_density_haswork;
    tw_deriv->fill = (TreeWalkFillQueryFunction) densderiv_copy;
    tw_deriv->reduce = (TreeWalkReduceResultFunction) densderiv_reduce;
    tw_deriv->postprocess = (TreeWalkProcessFunction) densderiv_postprocess;
    tw_deriv->query_type_elsize = sizeof(TreeWalkQueryDerivative);
    tw_deriv->result_type_elsize = sizeof(TreeWalkResultDerivative);
    tw_deriv->tree = tree;
    /*************************************************************************/
    
    TreeWalk tw_qp[1] = {{0}};
    
    tw_qp->ev_label = "DM_QP";
    tw_qp->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw_qp->ngbiter_type_elsize = sizeof(TreeWalkNgbIterQP);
    tw_qp->ngbiter = (TreeWalkNgbIterFunction) qp_ngbiter;
    tw_qp->haswork = dm_density_haswork;
    tw_qp->fill = (TreeWalkFillQueryFunction) qp_copy;
    tw_qp->reduce = (TreeWalkReduceResultFunction) qp_reduce;
    tw_qp->postprocess = (TreeWalkProcessFunction) NULL;
    tw_qp->query_type_elsize = sizeof(TreeWalkQueryQP);
    tw_qp->result_type_elsize = sizeof(TreeWalkResultQP);
    tw_qp->tree = tree;
    /*************************************************************************/
    
    double timeall = 0, timenetwork = 0;
    double timecomp, timecomm, timewait;
    
    walltime_measure("/FDM/DensDeriv/Init");
    treewalk_run(tw_deriv, act->ActiveParticle, act->NumActiveParticle);
    
    timeall += walltime_measure(WALLTIME_IGNORE);

    timecomp = tw_deriv->timecomp1 + tw_deriv->timecomp2 + tw_deriv->timecomp3;
    timewait = tw_deriv->timewait1 + tw_deriv->timewait2;
    timecomm = tw_deriv->timecommsumm1 + tw_deriv->timecommsumm2;

    walltime_add("/FDM/DensDeriv/Compute", timecomp);
    walltime_add("/FDM/DensDeriv/Wait", timewait);
    walltime_add("/FDM/DensDeriv/Comm", timecomm);
    walltime_add("/FDM/DensDeriv/Misc", timeall - (timecomp + timewait + timecomm + timenetwork));
    
    /*************************************************************************/
    walltime_measure("/FDM/DensQP/Init");
    treewalk_run(tw_qp, act->ActiveParticle, act->NumActiveParticle);
    
    timeall += walltime_measure(WALLTIME_IGNORE);

    timecomp = tw_qp->timecomp1 + tw_qp->timecomp2 + tw_qp->timecomp3;
    timewait = tw_qp->timewait1 + tw_qp->timewait2;
    timecomm = tw_qp->timecommsumm1 + tw_qp->timecommsumm2;

    walltime_add("/FDM/DensQP/Compute", timecomp);
    walltime_add("/FDM/DensQP/Wait", timewait);
    walltime_add("/FDM/DensQP/Comm", timecomm);
    walltime_add("/FDM/DensQP/Misc", timeall - (timecomp + timewait + timecomm + timenetwork));
    
}
