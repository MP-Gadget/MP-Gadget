#include <math.h>
#include "treewalk.h"
#include "densitykernel.h"
#include "bhdynfric.h"
#include "density.h"

static struct BlackholeDynFricParams
{
    int BH_DynFrictionMethod;/*0 for off; 1 for Star Only; 2 for DM+Star; 3 for DM+Star+Gas */
    int BH_DFBoostFactor; /*Optional boost factor for DF */
    double BH_DFbmax; /* the maximum impact range, in physical unit of kpc. */
} blackhole_dynfric_params;

/*Set the parameters of the BH module*/
void set_blackhole_dynfric_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        blackhole_dynfric_params.BH_DynFrictionMethod = param_get_int(ps, "BH_DynFrictionMethod");
        blackhole_dynfric_params.BH_DFBoostFactor = param_get_int(ps, "BH_DFBoostFactor");
        blackhole_dynfric_params.BH_DFbmax = param_get_double(ps, "BH_DFbmax");
    }
    MPI_Bcast(&blackhole_dynfric_params, sizeof(struct BlackholeDynFricParams), MPI_BYTE, 0, MPI_COMM_WORLD);
}

#define BHDYN_GET_PRIV(tw) ((struct BHDynFricPriv *) (tw->priv))

/*****************************************************************************/
typedef struct {
    TreeWalkQueryBase base;
    MyFloat Hsml;
} TreeWalkQueryBHDynfric;

typedef struct {
    TreeWalkResultBase base;

    MyFloat SurroundingVel[3];
    MyFloat SurroundingDensity;
    int SurroundingParticles;
    MyFloat SurroundingRmsVel;

} TreeWalkResultBHDynfric;

typedef struct {
    TreeWalkNgbIterBase base;
    DensityKernel dynfric_kernel;
} TreeWalkNgbIterBHDynfric;

int
blackhole_dynfric_treemask(void)
{
    /* dynamical friction uses: stars, DM if BH_DynFrictionMethod > 1, gas if BH_DynFrictionMethod  == 3.*/
    int treemask = STARMASK;
    /* Don't necessarily need dark matter */
    if(blackhole_dynfric_params.BH_DynFrictionMethod > 1)
        treemask += DMMASK;
    if(blackhole_dynfric_params.BH_DynFrictionMethod > 2)
        treemask += GASMASK;
    return treemask;
}

/*************************************************************************************/
/* DF routines */
static void
blackhole_dynfric_postprocess(int n, TreeWalk * tw){

    int PI = P[n].PI;
    int j;

    /***********************************************************************************/
    /* This is Gizmo's implementation of dynamic friction                              */
    /* c.f. section 3.1 in http://www.tapir.caltech.edu/~phopkins/public/notes_blackholes.pdf */
    /* Compute dynamic friction accel when DF turned on                                */
    /* averaged value for colomb logarithm and integral over the distribution function */
    /* acc_friction = -4*pi*G^2 * Mbh * log(lambda) * rho * f_of_x * bhvel / |bhvel^3| */
    /*       f_of_x = [erf(x) - 2*x*exp(-x^2)/sqrt(pi)]                                */
    /*       lambda = b_max * v^2 / G / (M+m)                                          */
    /*        b_max = Size of system (e.g. Rvir)                                       */
    /*            v = Relative velocity of BH with respect to the environment          */
    /*            M = Mass of BH                                                       */
    /*            m = individual mass elements composing the large system (e.g. m<<M)  */
    /*            x = v/sqrt(2)/sigma                                                  */
    /*        sigma = width of the max. distr. of the host system                      */
    /*                (e.g. sigma = v_disp / 3                                         */

    if(BHDYN_GET_PRIV(tw)->BH_SurroundingDensity[PI] > 0){
        double bhvel;
        double lambda, x, f_of_x;
        const double a_erf = 8 * (M_PI - 3) / (3 * M_PI * (4. - M_PI));

        /* normalize velocity/dispersion */
        BHDYN_GET_PRIV(tw)->BH_SurroundingRmsVel[PI] /= BHDYN_GET_PRIV(tw)->BH_SurroundingDensity[PI];
        BHDYN_GET_PRIV(tw)->BH_SurroundingRmsVel[PI] = sqrt(BHDYN_GET_PRIV(tw)->BH_SurroundingRmsVel[PI]);
        for(j = 0; j < 3; j++)
            BHDYN_GET_PRIV(tw)->BH_SurroundingVel[PI][j] /= BHDYN_GET_PRIV(tw)->BH_SurroundingDensity[PI];

        /* Calculate Coulumb Logarithm */
        bhvel = 0;
        for(j = 0; j < 3; j++)
        {
            bhvel += pow(P[n].Vel[j] - BHDYN_GET_PRIV(tw)->BH_SurroundingVel[PI][j], 2);
        }
        bhvel = sqrt(bhvel);

        /* There is no parameter in physical unit, so I kept everything in code unit */

        x = bhvel / sqrt(2) / (BHDYN_GET_PRIV(tw)->BH_SurroundingRmsVel[PI] / 3);
        /* First term is aproximation of the error function */
        f_of_x = x / fabs(x) * sqrt(1 - exp(-x * x * (4 / M_PI + a_erf * x * x)
            / (1 + a_erf * x * x))) - 2 * x / sqrt(M_PI) * exp(-x * x);
        /* Floor at zero */
        if (f_of_x < 0)
            f_of_x = 0;

        lambda = 1. + blackhole_dynfric_params.BH_DFbmax * pow((bhvel/BHDYN_GET_PRIV(tw)->atime),2) / BHDYN_GET_PRIV(tw)->CP->GravInternal / P[n].Mass;

        for(j = 0; j < 3; j++)
        {
            /* prevent DFAccel from exploding */
            if(bhvel > 0){
                BHP(n).DFAccel[j] = - 4. * M_PI * BHDYN_GET_PRIV(tw)->CP->GravInternal * BHDYN_GET_PRIV(tw)->CP->GravInternal * P[n].Mass * BHDYN_GET_PRIV(tw)->BH_SurroundingDensity[PI] *
                log(lambda) * f_of_x * (P[n].Vel[j] - BHDYN_GET_PRIV(tw)->BH_SurroundingVel[PI][j]) / pow(bhvel, 3);
                BHP(n).DFAccel[j] *= BHDYN_GET_PRIV(tw)->atime;  // convert to code unit of acceleration
                BHP(n).DFAccel[j] *= blackhole_dynfric_params.BH_DFBoostFactor; // Add a boost factor
            }
            else{
                BHP(n).DFAccel[j] = 0;
            }
        }
#ifdef DEBUG
        message(2,"x=%e, log(lambda)=%e, fof_x=%e, Mbh=%e, ratio=%e \n",
           x,log(lambda),f_of_x,P[n].Mass,BHP(n).DFAccel[0]/P[n].FullTreeGravAccel[0]);
#endif
    }
    else
    {
        message(2, "Dynamic Friction density is zero for BH %ld. Surroundingpart %d, mass %g, hsml %g, dens %g, pos %g %g %g.\n",
            P[n].ID, BHDYN_GET_PRIV(tw)->BH_SurroundingParticles[PI], BHP(n).Mass, P[n].Hsml, BHP(n).Density, P[n].Pos[0], P[n].Pos[1], P[n].Pos[2]);
        for(j = 0; j < 3; j++)
        {
            BHP(n).DFAccel[j] = 0;
        }
    }
}

static void
blackhole_dynfric_reduce(int place, TreeWalkResultBHDynfric * remote, enum TreeWalkReduceMode mode, TreeWalk * tw){
    int PI = P[place].PI;

    TREEWALK_REDUCE(BHDYN_GET_PRIV(tw)->BH_SurroundingDensity[PI], remote->SurroundingDensity);
    TREEWALK_REDUCE(BHDYN_GET_PRIV(tw)->BH_SurroundingParticles[PI], remote->SurroundingParticles);
    TREEWALK_REDUCE(BHDYN_GET_PRIV(tw)->BH_SurroundingVel[PI][0], remote->SurroundingVel[0]);
    TREEWALK_REDUCE(BHDYN_GET_PRIV(tw)->BH_SurroundingVel[PI][1], remote->SurroundingVel[1]);
    TREEWALK_REDUCE(BHDYN_GET_PRIV(tw)->BH_SurroundingVel[PI][2], remote->SurroundingVel[2]);
    TREEWALK_REDUCE(BHDYN_GET_PRIV(tw)->BH_SurroundingRmsVel[PI], remote->SurroundingRmsVel);

}

static void
blackhole_dynfric_copy(int place, TreeWalkQueryBHDynfric * I, TreeWalk * tw){
    /* SPH kernel width should be the only thing needed */
    I->Hsml = P[place].Hsml;
}


static void
blackhole_dynfric_ngbiter(TreeWalkQueryBHDynfric * I,
        TreeWalkResultBHDynfric * O,
        TreeWalkNgbIterBHDynfric * iter,
        LocalTreeWalk * lv){

   if(iter->base.other == -1) {
        iter->base.mask = blackhole_dynfric_treemask();
        iter->base.Hsml = I->Hsml;
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        density_kernel_init(&iter->dynfric_kernel, I->Hsml, GetDensityKernelType());
        return;
    }

    int other = iter->base.other;
    double r = iter->base.r;
    double r2 = iter->base.r2;

    /* Collect Star/+DM/+Gas density/velocity for DF computation */
    if(P[other].Type == 4 || (P[other].Type == 1 && blackhole_dynfric_params.BH_DynFrictionMethod > 1) ||
        (P[other].Type == 0 && blackhole_dynfric_params.BH_DynFrictionMethod == 3) ){
        if(r2 < iter->dynfric_kernel.HH) {
            double u = r * iter->dynfric_kernel.Hinv;
            double wk = density_kernel_wk(&iter->dynfric_kernel, u);
            float mass_j = P[other].Mass;
            int k;
            O->SurroundingParticles += 1;
            O->SurroundingDensity += (mass_j * wk);
            MyFloat VelPred[3];
            if(P[other].Type == 0)
                SPH_VelPred(other, VelPred, BHDYN_GET_PRIV(lv->tw)->kf);
            else {
                DM_VelPred(other, VelPred, BHDYN_GET_PRIV(lv->tw)->kf);
            }
            for (k = 0; k < 3; k++){
                O->SurroundingVel[k] += (mass_j * wk * VelPred[k]);
                O->SurroundingRmsVel += (mass_j * wk * pow(VelPred[k], 2));
            }
        }
    }
}

void blackhole_dynpriv_free(struct BHDynFricPriv * dynpriv)
{
    myfree(dynpriv->BH_SurroundingDensity);
    myfree(dynpriv->BH_SurroundingParticles);
    myfree(dynpriv->BH_SurroundingVel);
    myfree(dynpriv->BH_SurroundingRmsVel);
}

void
blackhole_dynfric(int * ActiveBlackHoles, int64_t NumActiveBlackHoles, ForceTree * tree, struct BHDynFricPriv * priv)
{
    
    if (blackhole_dynfric_params.BH_DynFrictionMethod == 0) {
        /* Environment variables for DF */
        /* Guard against memory allocation issue when collecting bhinfo in case df is turned off */
        /* we could also just not allocate this at all and set bhinfo to zero, but we need to also be careful with freeing the mem in this case */
        /* for now we put a place-holder here */
        priv->BH_SurroundingRmsVel = (MyFloat *) mymalloc("BH_SurroundingRmsVel", SlotsManager->info[5].size * sizeof(priv->BH_SurroundingRmsVel));
        priv->BH_SurroundingVel = (MyFloat (*) [3]) mymalloc("BH_SurroundingVel", 3* SlotsManager->info[5].size * sizeof(priv->BH_SurroundingVel[0]));
        priv->BH_SurroundingParticles = (int *)mymalloc("BH_SurroundingParticles", SlotsManager->info[5].size * sizeof(priv->BH_SurroundingParticles));
        priv->BH_SurroundingDensity = (MyFloat *) mymalloc("BH_SurroundingDensity", SlotsManager->info[5].size * sizeof(priv->BH_SurroundingDensity));
        
        for(int i = 0; i < NumActiveBlackHoles; i++)
        {
            const int p_i = ActiveBlackHoles ? ActiveBlackHoles[i] : i;
            if(p_i < 0 || p_i > PartManager->NumPart)
                endrun(1, "Bad index %d in black hole with %d active, %ld total\n", p_i, NumActiveBlackHoles, PartManager->NumPart);
            if(P[p_i].Type != 5)
                endrun(1, "Supposed BH %d of %d has type %d\n", p_i, NumActiveBlackHoles, P[p_i].Type);
            int PI = P[p_i].PI;
            priv->BH_SurroundingRmsVel[PI] = 0;
            priv->BH_SurroundingVel[PI][0] = 0;
            priv->BH_SurroundingVel[PI][1] = 0;
            priv->BH_SurroundingVel[PI][2] = 0;
            priv->BH_SurroundingParticles[PI] = 0;
            priv->BH_SurroundingDensity[PI] = 0;
        }
        return;
    }
        

    if(!(tree->mask & GASMASK) || !(tree->mask & STARMASK))
        endrun(5, "Error: BH tree types GAS: %d STAR %d\n", tree->mask & GASMASK, tree->mask & STARMASK);

    /*************************************************************************/
    /* Environment variables for DF */
    priv->BH_SurroundingRmsVel = (MyFloat *) mymalloc("BH_SurroundingRmsVel", SlotsManager->info[5].size * sizeof(priv->BH_SurroundingRmsVel));
    priv->BH_SurroundingVel = (MyFloat (*) [3]) mymalloc("BH_SurroundingVel", 3* SlotsManager->info[5].size * sizeof(priv->BH_SurroundingVel[0]));
    priv->BH_SurroundingParticles = (int *)mymalloc("BH_SurroundingParticles", SlotsManager->info[5].size * sizeof(priv->BH_SurroundingParticles));
    priv->BH_SurroundingDensity = (MyFloat *) mymalloc("BH_SurroundingDensity", SlotsManager->info[5].size * sizeof(priv->BH_SurroundingDensity));

    TreeWalk tw_dynfric[1] = {{0}};
    tw_dynfric->ev_label = "BH_DYNFRIC";
    tw_dynfric->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw_dynfric->ngbiter_type_elsize = sizeof(TreeWalkNgbIterBHDynfric);
    tw_dynfric->ngbiter = (TreeWalkNgbIterFunction) blackhole_dynfric_ngbiter;
    tw_dynfric->postprocess = (TreeWalkProcessFunction) blackhole_dynfric_postprocess;
    tw_dynfric->fill = (TreeWalkFillQueryFunction) blackhole_dynfric_copy;
    tw_dynfric->reduce = (TreeWalkReduceResultFunction) blackhole_dynfric_reduce;
    tw_dynfric->query_type_elsize = sizeof(TreeWalkQueryBHDynfric);
    tw_dynfric->result_type_elsize = sizeof(TreeWalkResultBHDynfric);
    tw_dynfric->tree = tree;
    tw_dynfric->priv = priv;
    tw_dynfric->haswork = NULL;

    treewalk_run(tw_dynfric, ActiveBlackHoles, NumActiveBlackHoles);

}
