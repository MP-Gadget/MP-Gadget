#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#include "allvars.h"
#include "proto.h"
#include "forcetree.h"
#include "treewalk.h"
#include "mymalloc.h"
#include "domain.h"
#include "endrun.h"

/*! \file gravtree.c
 *  \brief main driver routines for gravitational (short-range) force computation
 *
 *  This file contains the code for the gravitational force computation by
 *  means of the tree algorithm. To this end, a tree force is computed for all
 *  active local particles, and particles are exported to other processors if
 *  needed, where they can receive additional force contributions. If the
 *  TreePM algorithm is enabled, the force computed will only be the
 *  short-range part.
 */

/* According to upstream P-GADGET3
 * correct workcount slows it down and yields little benefits in load balancing
 *
 * YF: anything we shall do about this?
 * */

static int gravtree_isactive(int i);
void gravtree_copy(int place, struct gravitydata_in * input) ;
void gravtree_reduce(int place, struct gravitydata_out * result, int mode);
static void gravtree_post_process(int i);

/*! This function computes the gravitational forces for all active particles.
 *  If needed, a new tree is constructed, otherwise the dynamically updated
 *  tree is used.  Particles are only exported to other processors when really
 *  needed, thereby allowing a good use of the communication buffer.
 */
void gravity_tree(void)
{
    double Ewaldcount, Costtotal;
    int64_t N_nodesinlist;

    int64_t n_exported = 0;
    int i, maxnumnodes, iter = 0;
    double timeall = 0, timetree1 = 0, timetree2 = 0;
    double timetree, timewait, timecomm;
    double timecommsumm1 = 0, timecommsumm2 = 0, timewait1 = 0, timewait2 = 0;
    double sum_costtotal, ewaldtot;
    double maxt, sumt, maxt1, sumt1, maxt2, sumt2, sumcommall, sumwaitall;
    double plb, plb_max;

    TreeWalk ev[1] = {0};

    ev[0].ev_label = "FORCETREE_SHORTRANGE";
    ev[0].ev_evaluate = (ev_ev_func) force_treeev_shortrange;
    ev[0].ev_isactive = gravtree_isactive;
    ev[0].ev_reduce = (ev_reduce_func) gravtree_reduce;
    ev[0].UseNodeList = 1;

    ev[0].ev_datain_elsize = sizeof(struct gravitydata_in);
    ev[0].ev_dataout_elsize = sizeof(struct gravitydata_out);
    ev[0].ev_copy = (ev_copy_func) gravtree_copy;

    walltime_measure("/Misc");

    /* set new softening lengths */

    set_softenings();

    /* allocate buffers to arrange communication */
    message(0, "Begin tree force.  (presently allocated=%g MB)\n", AllocatedBytes / (1024.0 * 1024.0));

    walltime_measure("/Misc");

    ev_run(&ev[0]);
    iter += ev[0].Niterations;
    n_exported += ev[0].Nexport_sum;
    N_nodesinlist += ev[0].Nnodesinlist;

    Ewaldcount = ev[0].Ninteractions;

    if(All.TypeOfOpeningCriterion == 1)
        All.ErrTolTheta = 0;	/* This will switch to the relative opening criterion for the following force computations */


    Costtotal = 0;
    /* now add things for comoving integration */

    int Nactive;
    /* doesn't matter which ev to use, they have the same ev_active*/
    int * queue = ev_get_queue(&ev[0], &Nactive);
#pragma omp parallel for if(Nactive > 32)
    for(i = 0; i < Nactive; i++) {
        gravtree_post_process(queue[i]);
        /* this shall agree with sum of Ninteractions in all ev[..] need to
         * check it*/
#pragma omp atomic
        Costtotal += P[i].GravCost;
    }
    myfree(queue);

    message(0, "tree is done.\n");

    /* This code is removed for now gravity_static_potential(); */

    /* Now the force computation is finished */


    /*  gather some diagnostic information */
    timetree1 += ev[0].timecomp1;
    timetree2 += ev[0].timecomp2;
    timewait1 += ev[0].timewait1;
    timewait2 += ev[0].timewait2;
    timecommsumm1 += ev[0].timecommsumm1 ;
    timecommsumm2 += ev[0].timecommsumm2;

    timetree = timetree1 + timetree2;
    timewait = timewait1 + timewait2;
    timecomm= timecommsumm1 + timecommsumm2;

    MPI_Reduce(&timetree, &sumt, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timetree, &maxt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timetree1, &sumt1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timetree1, &maxt1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timetree2, &sumt2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timetree2, &maxt2, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timewait, &sumwaitall, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timecomm, &sumcommall, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Costtotal, &sum_costtotal, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Ewaldcount, &ewaldtot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    sumup_longs(1, &n_exported, &n_exported);
    sumup_longs(1, &N_nodesinlist, &N_nodesinlist);

    All.TotNumOfForces += GlobNumForceUpdate;

    plb = (NumPart / ((double) All.TotNumPart)) * NTask;
    MPI_Reduce(&plb, &plb_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Numnodestree, &maxnumnodes, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    walltime_add("/Tree/Walk1", timetree1);
    walltime_add("/Tree/Walk2", timetree2);
    walltime_add("/Tree/Send", timecommsumm1);
    walltime_add("/Tree/Recv", timecommsumm2);
    walltime_add("/Tree/Wait1", timewait1);
    walltime_add("/Tree/Wait2", timewait2);

    timeall = walltime_measure(WALLTIME_IGNORE);
    walltime_add("/Tree/Misc", timeall - (timetree + timewait + timecomm));

    if(ThisTask == 0)
    {
        fprintf(FdTimings, "Step= %d  t= %g  dt= %g \n", All.NumCurrentTiStep, All.Time, All.TimeStep);
        fprintf(FdTimings, "Nf= %013ld  total-Nf= %013ld  ex-frac= %g (%g) iter= %d\n",
                GlobNumForceUpdate, All.TotNumOfForces,
                n_exported / ((double) GlobNumForceUpdate), N_nodesinlist / ((double) n_exported + 1.0e-10),
                iter);
        /* note: on Linux, the 8-byte integer could be printed with the format identifier "%qd", but doesn't work on AIX */

        fprintf(FdTimings, "work-load balance: %g (%g %g) rel1to2=%g   max=%g avg=%g\n",
                maxt / (1.0e-6 + sumt / NTask), maxt1 / (1.0e-6 + sumt1 / NTask),
                maxt2 / (1.0e-6 + sumt2 / NTask), sumt1 / (1.0e-6 + sumt1 + sumt2), maxt, sumt / NTask);
        fprintf(FdTimings, "particle-load balance: %g\n", plb_max);
        fprintf(FdTimings, "max. nodes: %d, filled: %g\n", maxnumnodes,
                maxnumnodes / (All.TreeAllocFactor * All.MaxPart + NTopnodes));
        fprintf(FdTimings, "part/sec=%g | %g  ia/part=%g (%g)\n", GlobNumForceUpdate / (sumt + 1.0e-20),
                GlobNumForceUpdate / (1.0e-6 + maxt * NTask),
                ((double) (sum_costtotal)) / (1.0e-20 + GlobNumForceUpdate),
                ((double) ewaldtot) / (1.0e-20 + GlobNumForceUpdate));
        fprintf(FdTimings, "\n");

        fflush(FdTimings);
    }

    walltime_measure("/Tree/Timing");
}

void gravtree_copy(int place, struct gravitydata_in * input) {
    int k;
    for(k = 0; k < 3; k++)
        input->Pos[k] = P[place].Pos[k];

    input->Type = P[place].Type;
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
    if(P[place].Type == 0)
        input->Soft = P[place].Hsml;
#endif
    input->OldAcc = P[place].OldAcc;

}

void gravtree_reduce(int place, struct gravitydata_out * result, int mode) {
#define REDUCE(A, B) (A) = (mode==0)?(B):((A) + (B))
    int k;
    for(k = 0; k < 3; k++)
        REDUCE(P[place].GravAccel[k], result->Acc[k]);

    REDUCE(P[place].GravCost, result->Ninteractions);
    REDUCE(P[place].Potential, result->Potential);
}

static int gravtree_isactive(int i) {
    int isactive = 1;
#ifdef BLACK_HOLES
    /* blackhole has not gravity, they move along to pot minimium */
    isactive = isactive && (P[i].Type != 5);
#endif
    return isactive;
}

static void gravtree_post_process(int i)
{
    int j;

    double ax, ay, az;
    ax = P[i].GravAccel[0] + P[i].GravPM[0] / All.G;
    ay = P[i].GravAccel[1] + P[i].GravPM[1] / All.G;
    az = P[i].GravAccel[2] + P[i].GravPM[2] / All.G;

    P[i].OldAcc = sqrt(ax * ax + ay * ay + az * az);
    for(j = 0; j < 3; j++)
        P[i].GravAccel[j] *= All.G;

    /* calculate the potential */
    /* remove self-potential */
    P[i].Potential += P[i].Mass / All.SofteningTable[P[i].Type];

    P[i].Potential -= 2.8372975 * pow(P[i].Mass, 2.0 / 3) *
        pow(All.CP.Omega0 * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.G), 1.0 / 3);

    P[i].Potential *= All.G;

    P[i].Potential += P[i].PM_Potential;	/* add in long-range potential */

}

/*! This function sets the (comoving) softening length of all particle
 *  types in the table All.SofteningTable[...].  We check that the physical
 *  softening length is bounded by the Softening-MaxPhys values.
 */
void set_softenings(void)
{
    int i;

    if(All.SofteningGas * All.Time > All.SofteningGasMaxPhys)
        All.SofteningTable[0] = All.SofteningGasMaxPhys / All.Time;
    else
        All.SofteningTable[0] = All.SofteningGas;

    if(All.SofteningHalo * All.Time > All.SofteningHaloMaxPhys)
        All.SofteningTable[1] = All.SofteningHaloMaxPhys / All.Time;
    else
        All.SofteningTable[1] = All.SofteningHalo;

    if(All.SofteningDisk * All.Time > All.SofteningDiskMaxPhys)
        All.SofteningTable[2] = All.SofteningDiskMaxPhys / All.Time;
    else
        All.SofteningTable[2] = All.SofteningDisk;

    if(All.SofteningBulge * All.Time > All.SofteningBulgeMaxPhys)
        All.SofteningTable[3] = All.SofteningBulgeMaxPhys / All.Time;
    else
        All.SofteningTable[3] = All.SofteningBulge;

    if(All.SofteningStars * All.Time > All.SofteningStarsMaxPhys)
        All.SofteningTable[4] = All.SofteningStarsMaxPhys / All.Time;
    else
        All.SofteningTable[4] = All.SofteningStars;

    if(All.SofteningBndry * All.Time > All.SofteningBndryMaxPhys)
        All.SofteningTable[5] = All.SofteningBndryMaxPhys / All.Time;
    else
        All.SofteningTable[5] = All.SofteningBndry;

    for(i = 0; i < 6; i++)
        All.ForceSoftening[i] = 2.8 * All.SofteningTable[i];

    All.MinGasHsml = All.MinGasHsmlFractional * All.ForceSoftening[0];
}
