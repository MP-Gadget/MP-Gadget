#include <math.h>
#include <omp.h>
#include "veldisp.h"
#include "treewalk.h"
#include "walltime.h"
#include "sfr_eff.h"

/* Compute the DM velocity dispersion for black holes*/
static void blackhole_veldisp(const ActiveParticles * act, Cosmology * CP, ForceTree * tree, struct kick_factor_data * kf);

/* For the wind hsml loop*/
#define NWINDHSML 5 /* Number of densities to evaluate for wind weight ngbiter*/
#define NUMDMNGB 40 /*Number of DM ngb to evaluate vel dispersion */
#define MAXDMDEVIATION 1


/* Computes the BH velocity dispersion for kinetic feedback*/

struct BHVelDispPriv {
    /* temporary computed for kinetic feedback energy threshold*/
    MyFloat * NumDM;
    MyFloat (*V1sumDM)[3];
    MyFloat * V2sumDM;
    /* Time factors*/
    struct kick_factor_data * kf;
};
#define BHVDISP_GET_PRIV(tw) ((struct BHVelDispPriv *) (tw->priv))

typedef struct {
    TreeWalkQueryBase base;
    MyFloat Hsml;
    MyFloat Vel[3];
} TreeWalkQueryBHVelDisp;

typedef struct {
    TreeWalkResultBase base;
    /* used for AGN kinetic feedback */
    MyFloat V2sumDM;
    MyFloat V1sumDM[3];
    MyFloat NumDM;
} TreeWalkResultBHVelDisp;

typedef struct {
    TreeWalkNgbIterBase base;
    DensityKernel feedback_kernel;
} TreeWalkNgbIterBHVelDisp;

/*******************************************************************/
static int
blackhole_dynfric_haswork(int n, TreeWalk * tw){
    /*Black hole not being swallowed*/
    return (P[n].Type == 5) && (!P[n].Swallowed);
}

static void
blackhole_veldisp_postprocess(int i, TreeWalk * tw)
{
    int PI = P[i].PI;
    /*************************************************************************/
    /* decide whether to release KineticFdbkEnergy*/
    double vdisp = 0;
    double numdm = BHVDISP_GET_PRIV(tw)->NumDM[PI];
    if (numdm>0){
        vdisp = BHVDISP_GET_PRIV(tw)->V2sumDM[PI]/numdm;
        int d;
        for(d = 0; d<3; d++){
            vdisp -= pow(BHVDISP_GET_PRIV(tw)->V1sumDM[PI][d]/numdm,2);
        }
        if(vdisp > 0)
            BHP(i).VDisp = sqrt(vdisp / 3);
    }
}

static void
blackhole_veldisp_ngbiter(TreeWalkQueryBHVelDisp * I,
        TreeWalkResultBHVelDisp * O,
        TreeWalkNgbIterBHVelDisp * iter,
        LocalTreeWalk * lv)
{
    if(iter->base.other == -1) {
        iter->base.mask = DMMASK;
        iter->base.Hsml = I->Hsml;
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        density_kernel_init(&iter->feedback_kernel, I->Hsml, GetDensityKernelType());
        return;
    }

    int other = iter->base.other;
    double r2 = iter->base.r2;

    /* collect info for sigmaDM and Menc for kinetic feedback */
    if(P[other].Type == 1 && r2 < iter->feedback_kernel.HH){
        O->NumDM += 1;
        MyFloat VelPred[3];
        DM_VelPred(other, VelPred, BHVDISP_GET_PRIV(lv->tw)->kf);
        int d;
        for(d = 0; d < 3; d++){
            double vel = VelPred[d] - I->Vel[d];
            O->V1sumDM[d] += vel;
            O->V2sumDM += vel * vel;
        }
    }
}

static void
blackhole_veldisp_reduce(int place, TreeWalkResultBHVelDisp * remote, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    int k;

    int PI = P[place].PI;
    for (k = 0; k < 3; k++){
        TREEWALK_REDUCE(BHVDISP_GET_PRIV(tw)->V1sumDM[PI][k], remote->V1sumDM[k]);
    }
    TREEWALK_REDUCE(BHVDISP_GET_PRIV(tw)->NumDM[PI], remote->NumDM);
    TREEWALK_REDUCE(BHVDISP_GET_PRIV(tw)->V2sumDM[PI], remote->V2sumDM);
}

static void
blackhole_veldisp_copy(int place, TreeWalkQueryBHVelDisp * I, TreeWalk * tw)
{
    int k;
    for(k = 0; k < 3; k++)
        I->Vel[k] = P[place].Vel[k];
    I->Hsml = P[place].Hsml;
}

static void
blackhole_veldisp(const ActiveParticles * act, Cosmology * CP, ForceTree * tree, struct kick_factor_data * kf)
{
    struct BHVelDispPriv priv[1] = {0};

    TreeWalk tw_veldisp[1] = {{0}};
    tw_veldisp->ev_label = "BH_VDISP";
    tw_veldisp->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw_veldisp->ngbiter_type_elsize = sizeof(TreeWalkNgbIterBHVelDisp);
    tw_veldisp->ngbiter = (TreeWalkNgbIterFunction) blackhole_veldisp_ngbiter;
    tw_veldisp->haswork = blackhole_dynfric_haswork;
    tw_veldisp->postprocess = (TreeWalkProcessFunction) blackhole_veldisp_postprocess;
    tw_veldisp->fill = (TreeWalkFillQueryFunction) blackhole_veldisp_copy;
    tw_veldisp->reduce = (TreeWalkReduceResultFunction) blackhole_veldisp_reduce;
    tw_veldisp->query_type_elsize = sizeof(TreeWalkQueryBHVelDisp);
    tw_veldisp->result_type_elsize = sizeof(TreeWalkResultBHVelDisp);
    tw_veldisp->tree = tree;
    tw_veldisp->priv = priv;

    /* This treewalk uses only DM */
    if(!tree->tree_allocated_flag)
        endrun(0, "DM Tree not allocated for veldisp\n");

    /* For AGN kinetic feedback */
    priv->NumDM = mymalloc("NumDM", SlotsManager->info[5].size * sizeof(MyFloat));
    priv->V2sumDM = mymalloc("V2sumDM", SlotsManager->info[5].size * sizeof(MyFloat));
    priv->V1sumDM = (MyFloat (*) [3]) mymalloc("V1sumDM", 3* SlotsManager->info[5].size * sizeof(priv->V1sumDM[0]));
    priv->kf = kf;

    /* This allocates memory*/
    treewalk_run(tw_veldisp, act->ActiveParticle, act->NumActiveParticle);

    walltime_measure("/BH/VDisp");

    myfree(priv->V1sumDM);
    myfree(priv->V2sumDM);
    myfree(priv->NumDM);
}


/* Code to compute velocity dispersions*/

typedef struct {
    TreeWalkQueryBase base;
    double Mass;
    double DMRadius[NWINDHSML];
    double Vel[3];
} TreeWalkQueryWindVDisp;

typedef struct {
    TreeWalkResultBase base;
    double V1sum[NWINDHSML][3];
    double V2sum[NWINDHSML];
    double Ngb[NWINDHSML];
    int alignment; /* Ensure alignment*/
    int maxcmpte;
} TreeWalkResultWindVDisp;

struct WindVDispPriv {
    double Time;
    double hubble;
    struct kick_factor_data kf;
    double ddrift;
    /* Lower and upper bounds on smoothing length*/
    MyFloat *Left, *Right, *DMRadius;
    /* Maximum index where NumNgb is valid. */
    int * maxcmpte;
    MyFloat (* V2sum)[NWINDHSML];
    MyFloat (* V1sum)[NWINDHSML][3];
    MyFloat (* Ngb) [NWINDHSML];
};
#define WINDV_GET_PRIV(tw) ((struct WindVDispPriv *) (tw->priv))

static inline double
vdispeffdmradius(int place, int i, TreeWalk * tw)
{
    double left = WINDV_GET_PRIV(tw)->Left[place];
    double right = WINDV_GET_PRIV(tw)->Right[place];
    /*The asymmetry is because it is free to compute extra densities for h < Hsml, but not for h > Hsml*/
    if (right > 0.99*tw->tree->BoxSize){
        right = WINDV_GET_PRIV(tw)->DMRadius[place];
    }
    if(left == 0)
        left = 0.1 * WINDV_GET_PRIV(tw)->DMRadius[place];
    /*Evenly split in volume*/
    double rvol = pow(right, 3);
    double lvol = pow(left, 3);
    return pow((1.0*i+1)/(1.0*NWINDHSML+1) * (rvol - lvol) + lvol, 1./3);
}

static void
wind_vdisp_copy(int place, TreeWalkQueryWindVDisp * input, TreeWalk * tw)
{
    input->Mass = P[place].Mass;
    int i;
    for(i=0; i<3; i++)
        input->Vel[i] = P[place].Vel[i];
    for(i = 0; i<NWINDHSML; i++){
        input->DMRadius[i]=vdispeffdmradius(place,i,tw);
    }
}

static void
wind_vdisp_ngbiter(TreeWalkQueryWindVDisp * I,
        TreeWalkResultWindVDisp * O,
        TreeWalkNgbIterBase * iter,
        LocalTreeWalk * lv)
{
    /* this evaluator walks the tree and sums the total mass of surrounding gas
     * particles as described in VS08. */
    /* it also calculates the velocity dispersion of the nearest 40 DM or gas particles */
    if(iter->other == -1) {
        iter->Hsml = I->DMRadius[NWINDHSML-1];
        iter->mask = DMMASK; /* gas and dm */
        iter->symmetric = NGB_TREEFIND_ASYMMETRIC;
        O->maxcmpte = NWINDHSML;
        return;
    }

    int other = iter->other;
    double r = iter->r;
    double * dist = iter->dist;

    int i;
    const double atime = WINDV_GET_PRIV(lv->tw)->Time;
    for (i = 0; i < O->maxcmpte; i++) {
        if(r < I->DMRadius[i]) {
            O->Ngb[i] += 1;
            int d;
            MyFloat VelPred[3];
            DM_VelPred(other, VelPred, &WINDV_GET_PRIV(lv->tw)->kf);
            for(d = 0; d < 3; d ++) {
                /* Add hubble flow to relative velocity. Use predicted velocity to current time.
                 * The I particle is active so always at current time.*/
                double vel = VelPred[d] - I->Vel[d] + WINDV_GET_PRIV(lv->tw)->hubble * atime * atime * dist[d];
                O->V1sum[i][d] += vel;
                O->V2sum[i] += vel * vel;
            }
        }
    }

    for(i = 0; i<NWINDHSML; i++){
        if(O->Ngb[i] > NUMDMNGB){
            O->maxcmpte = i+1;
            iter->Hsml = I->DMRadius[i];
            break;
        }
    }
    /*
    message(1, "ThisTask = %d %ld ngb=%d NGB=%d TotalWeight=%g V2sum=%g V1sum=%g %g %g\n",
    ThisTask, I->ID, numngb, O->Ngb, O->TotalWeight, O->V2sum,
    O->V1sum[0], O->V1sum[1], O->V1sum[2]);
    */
}

static void
wind_vdisp_reduce(int place, TreeWalkResultWindVDisp * O, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    int pi = P[place].PI;
    int i;
    if(mode == TREEWALK_PRIMARY || WINDV_GET_PRIV(tw)->maxcmpte[pi] > O->maxcmpte)
        WINDV_GET_PRIV(tw)->maxcmpte[pi] = O->maxcmpte;
    int k;
    for (i = 0; i < O->maxcmpte; i++){
        TREEWALK_REDUCE(WINDV_GET_PRIV(tw)->Ngb[pi][i], O->Ngb[i]);
        TREEWALK_REDUCE(WINDV_GET_PRIV(tw)->V2sum[pi][i], O->V2sum[i]);
        for(k = 0; k < 3; k ++) {
            TREEWALK_REDUCE(WINDV_GET_PRIV(tw)->V1sum[pi][i][k], O->V1sum[i][k]);
        }
    }
//     message(1, "Reduce ID=%ld, NGB_first=%d NGB_last=%d maxcmpte = %d, left = %g, right = %g\n",
//             P[place].ID, O->Ngb[0],O->Ngb[O->maxcmpte-1],WINDP(place, Windd).maxcmpte,WINDP(place, Windd).Left,WINDP(place, Windd).Right);
}

static void
wind_vdisp_postprocess(const int i, TreeWalk * tw)
{
    const int pi = P[i].PI;
    const int maxcmpt = WINDV_GET_PRIV(tw)->maxcmpte[pi];
    int j;
    double evaldmradius[NWINDHSML];
    for(j = 0; j < maxcmpt; j++){
        evaldmradius[j] = vdispeffdmradius(i,j,tw);
    }
    int close = 0;
    WINDV_GET_PRIV(tw)->DMRadius[pi] = ngb_narrow_down(&WINDV_GET_PRIV(tw)->Right[pi], &WINDV_GET_PRIV(tw)->Left[pi], evaldmradius, WINDV_GET_PRIV(tw)->Ngb[pi], maxcmpt, NUMDMNGB, &close, tw->tree->BoxSize);
    double numngb = WINDV_GET_PRIV(tw)->Ngb[pi][close];

    int tid = omp_get_thread_num();
    /*  If we have 40 neighbours, or if DMRadius is narrow, set vdisp. Otherwise add to redo queue */
    if((numngb < (NUMDMNGB - MAXDMDEVIATION) || numngb > (NUMDMNGB + MAXDMDEVIATION)) &&
        (WINDV_GET_PRIV(tw)->Right[pi] - WINDV_GET_PRIV(tw)->Left[pi] > 5e-6 * WINDV_GET_PRIV(tw)->Left[pi])) {
        /* More work needed: add this particle to the redo queue*/
        tw->NPRedo[tid][tw->NPLeft[tid]] = i;
        tw->NPLeft[tid] ++;
    }
    else{
        if((WINDV_GET_PRIV(tw)->Right[pi] - WINDV_GET_PRIV(tw)->Left[pi] < 5e-6 * WINDV_GET_PRIV(tw)->Left[pi]))
            message(1, "Tight dm hsml for id %ld ngb %g radius %g\n",P[i].ID, numngb, evaldmradius[close]);

        double vdisp = WINDV_GET_PRIV(tw)->V2sum[pi][close] / numngb;
        int d;
        for(d = 0; d < 3; d++){
            vdisp -= pow(WINDV_GET_PRIV(tw)->V1sum[pi][close][d] / numngb,2);
        }
        if(vdisp > 0) {
            if(P[i].Type == 0)
                SPHP(i).VDisp = sqrt(vdisp / 3);
        }
    }

    if(tw->maxnumngb[tid] < numngb)
        tw->maxnumngb[tid] = numngb;
    if(tw->minnumngb[tid] > numngb)
        tw->minnumngb[tid] = numngb;
}

static int
winds_veldisp_haswork(int n, TreeWalk * tw)
{
    /* Don't want a density for swallowed particles*/
    if(P[n].Swallowed || P[n].IsGarbage)
        return 0;
    /* Only want gas*/
    if(P[n].Type != 0)
        return 0;
    /* We only want VDisp for gas particles that may be star-forming over the next PM timestep.
     * Use DtHsml.*/
    double densfac = (P[n].Hsml + P[n].DtHsml * WINDV_GET_PRIV(tw)->ddrift)/P[n].Hsml;
    if(densfac > 1)
        densfac = 1;
    if(SPHP(n).Density/(densfac * densfac * densfac) < 0.1 * sfr_density_threshold(WINDV_GET_PRIV(tw)->Time))
        return 0;
    /* Update veldisp only on a gravitationally active timestep,
     * since this is the frequency at which the gravitational acceleration,
     * which is the only thing DM contributes to, could change. This is probably
     * overly conservative, because most of the acceleration will be from other stars. */
//     if(!is_timebin_active(P[n].TimeBinGravity, P[n].Ti_drift))
//         return 0;
    return 1;
}

/* Find the 1D DM velocity dispersion of all gas particles by running a density loop.
 * Stores it in VDisp in the slots structure.*/
void
winds_find_vel_disp(const ActiveParticles * act, const double Time, const double hubble, Cosmology * CP, DriftKickTimes * times, DomainDecomp * ddecomp)
{
    TreeWalk tw[1] = {0};
    struct WindVDispPriv priv[1] = {0};
    ForceTree tree[1] = {0};
    /* Types used: gas*/
    tw->ev_label = "WIND_VDISP";
    tw->fill = (TreeWalkFillQueryFunction) wind_vdisp_copy;
    tw->reduce = (TreeWalkReduceResultFunction) wind_vdisp_reduce;
    tw->query_type_elsize = sizeof(TreeWalkQueryWindVDisp);
    tw->result_type_elsize = sizeof(TreeWalkResultWindVDisp);
    tw->tree = tree;

    /* sum the total weight of surrounding gas */
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterBase);
    tw->ngbiter = (TreeWalkNgbIterFunction) wind_vdisp_ngbiter;

    tw->haswork = winds_veldisp_haswork;
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_nolist_ngbiter;
    tw->postprocess = (TreeWalkProcessFunction) wind_vdisp_postprocess;

    priv[0].Time = Time;
    priv[0].hubble = hubble;
    priv[0].ddrift = get_exact_drift_factor(CP, times->Ti_Current, times->Ti_Current + times->PM_length);
    tw->priv = priv;

    /* Build the queue to check that we have something to do before we rebuild the tree.*/
    treewalk_build_queue(tw, act->ActiveParticle, act->NumActiveParticle, 0);

    int * ActiveVDisp = tw->WorkSet;
    int64_t NumVDisp = tw->WorkSetSize;
    int64_t totvdisp, totbh;
    /* Check for black holes*/
    MPI_Allreduce(&SlotsManager->info[5].size, &totbh, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    /* If this queue is empty, nothing to do for winds.*/
    MPI_Allreduce(&NumVDisp, &totvdisp, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

    if(totvdisp > 0 || totbh > 0) {
        force_tree_rebuild_mask(tree, ddecomp, DMMASK, NULL);
        tw->haswork = NULL;
        init_kick_factor_data(&priv->kf, times, CP);
    }

    /* Compute the black hole velocity dispersions if needed*/
    if(totbh)
        blackhole_veldisp(act, CP, tree, &priv->kf);

    if(totvdisp == 0) {
        force_tree_free(tree);
        myfree(ActiveVDisp);
        return;
    }

    priv->Left = (MyFloat *) mymalloc("VDISP->Left", SlotsManager->info[0].size * sizeof(MyFloat));
    priv->Right = (MyFloat *) mymalloc("VDISP->Right", SlotsManager->info[0].size * sizeof(MyFloat));
    priv->DMRadius = (MyFloat *) mymalloc("VDISP->DMRadius", SlotsManager->info[0].size * sizeof(MyFloat));
    priv->Ngb = (MyFloat (*) [NWINDHSML]) mymalloc("VDISP->NumNgb", SlotsManager->info[0].size * sizeof(priv->Ngb[0]));
    priv->V1sum = (MyFloat (*) [NWINDHSML][3]) mymalloc("VDISP->V1Sum", SlotsManager->info[0].size * sizeof(priv->V1sum[0]));
    priv->V2sum = (MyFloat (*) [NWINDHSML]) mymalloc("VDISP->V2Sum", SlotsManager->info[0].size * sizeof(priv->V2sum[0]));
    priv->maxcmpte = (int *) mymalloc("maxcmpte", SlotsManager->info[0].size * sizeof(int));
    report_memory_usage("WIND_VDISP");

    int i;
    /*Initialise the WINDP array*/
    #pragma omp parallel for
    for (i = 0; i < act->NumActiveParticle; i++) {
        const int n = act->ActiveParticle ? act->ActiveParticle[i] : i;
        if(P[n].Type == 0) {
            const int pi = P[n].PI;
            priv->DMRadius[pi] = P[n].Hsml;
            priv->Left[pi] = 0;
            priv->Right[pi] = tree->BoxSize;
            priv->maxcmpte[pi] = NUMDMNGB;
        }
    }
    /* Find densities*/
    treewalk_do_hsml_loop(tw, ActiveVDisp, NumVDisp, 1);
    /* Free memory*/
    myfree(priv->maxcmpte);
    myfree(priv->V2sum);
    myfree(priv->V1sum);
    myfree(priv->Ngb);
    myfree(priv->DMRadius);
    myfree(priv->Right);
    myfree(priv->Left);

    force_tree_free(tree);
    myfree(ActiveVDisp);
    walltime_measure("/Cooling/VDisp");

}
