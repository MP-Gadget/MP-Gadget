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

/*! length of lock-up table for short-range force kernel in TreePM algorithm */
#define NTAB 1000
/*! variables for short-range lookup table */
static float shortrange_table[NTAB], shortrange_table_potential[NTAB], shortrange_table_tidal[NTAB];

/*! toggles after first tree-memory allocation, has only influence on log-files */
static int first_flag = 0;


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

typedef struct
{
    TreeWalkQueryBase base;
    int Type;
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
    MyFloat Soft;
#endif
    MyFloat OldAcc;
} TreeWalkQueryGravity;

typedef struct
{
    TreeWalkResultBase base;
    MyDouble Acc[3];
    MyDouble Potential;
    int Ninteractions;
} TreeWalkResultGravity;


int force_treeev_shortrange(const int target,
        TreeWalkQueryGravity * input,
        TreeWalkResultGravity * output,
        LocalTreeWalk * lv);

static void fill_ntab()
{
    if(first_flag == 0)
    {
        first_flag = 1;
        int i;
        for(i = 0; i < NTAB; i++)
        {
            double u = 3.0 / NTAB * (i + 0.5);
            shortrange_table[i] = erfc(u) + 2.0 * u / sqrt(M_PI) * exp(-u * u);
            shortrange_table_potential[i] = erfc(u);
            shortrange_table_tidal[i] = 4.0 * u * u * u / sqrt(M_PI) * exp(-u * u);
        }
    }

}

static int gravtree_isactive(int i);
void gravtree_copy(int place, TreeWalkQueryGravity * input) ;
void gravtree_reduce(int place, TreeWalkResultGravity * result, enum TreeWalkReduceMode mode);
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

    ev[0].query_type_elsize = sizeof(TreeWalkQueryGravity);
    ev[0].result_type_elsize = sizeof(TreeWalkResultGravity);
    ev[0].ev_copy = (ev_copy_func) gravtree_copy;

    walltime_measure("/Misc");

    /* set new softening lengths */
    fill_ntab();

    set_softenings();

    /* allocate buffers to arrange communication */
    message(0, "Begin tree force.  (presently allocated=%g MB)\n", AllocatedBytes / (1024.0 * 1024.0));

    walltime_measure("/Misc");

    treewalk_run(&ev[0]);
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
    int * queue = treewalk_get_queue(&ev[0], &Nactive);
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

void gravtree_copy(int place, TreeWalkQueryGravity * input) {
    input->Type = P[place].Type;
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
    if(P[place].Type == 0)
        input->Soft = P[place].Hsml;
#endif
    input->OldAcc = P[place].OldAcc;

}

void gravtree_reduce(int place, TreeWalkResultGravity * result, enum TreeWalkReduceMode mode) {
    int k;
    for(k = 0; k < 3; k++)
        TREEWALK_REDUCE(P[place].GravAccel[k], result->Acc[k]);

    TREEWALK_REDUCE(P[place].GravCost, result->Ninteractions);
    TREEWALK_REDUCE(P[place].Potential, result->Potential);
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

/*! In the TreePM algorithm, the tree is walked only locally around the
 *  target coordinate.  Tree nodes that fall outside a box of half
 *  side-length Rcut= RCUT*ASMTH*MeshSize can be discarded. The short-range
 *  potential is modified by a complementary error function, multiplied
 *  with the Newtonian form. The resulting short-range suppression compared
 *  to the Newtonian force is tabulated, because looking up from this table
 *  is faster than recomputing the corresponding factor, despite the
 *  memory-access panelty (which reduces cache performance) incurred by the
 *  table.
 */
int force_treeev_shortrange(int target,
        TreeWalkQueryGravity * input,
        TreeWalkResultGravity * output,
        LocalTreeWalk * lv)
{
    struct NODE *nop = 0;
    int no, ptype, tabindex, listindex = 0;
    int nnodesinlist = 0, ninteractions = 0;
    double r2, dx, dy, dz, mass, r, fac, u, h, h_inv, h3_inv;
    double pos_x, pos_y, pos_z, aold;
    double eff_dist;
    double rcut, asmth, asmthfac, rcut2, dist;
    MyDouble acc_x, acc_y, acc_z;

#ifdef ADAPTIVE_GRAVSOFT_FORGAS
    double soft = 0;
#endif
    double wp, facpot;
    MyDouble pot;

    pot = 0;

    acc_x = 0;
    acc_y = 0;
    acc_z = 0;
    ninteractions = 0;
    nnodesinlist = 0;

    rcut = All.Rcut[0];
    asmth = All.Asmth[0];

    no = input->base.NodeList[0];
    listindex ++;
    no = Nodes[no].u.d.nextnode;	/* open it */

    pos_x = input->base.Pos[0];
    pos_y = input->base.Pos[1];
    pos_z = input->base.Pos[2];
    ptype = input->Type;

    aold = All.ErrTolForceAcc * input->OldAcc;
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
    if(ptype == 0)
        soft = input->Soft;
#endif
    rcut2 = rcut * rcut;

    asmthfac = 0.5 / asmth * (NTAB / 3.0);

    while(no >= 0)
    {
        while(no >= 0)
        {
            if(no < All.MaxPart)
            {
                /* the index of the node is the index of the particle */
                drift_particle(no, All.Ti_Current);

                dx = P[no].Pos[0] - pos_x;
                dy = P[no].Pos[1] - pos_y;
                dz = P[no].Pos[2] - pos_z;

                dx = NEAREST(dx);
                dy = NEAREST(dy);
                dz = NEAREST(dz);

                r2 = dx * dx + dy * dy + dz * dz;

                mass = P[no].Mass;

#ifdef ADAPTIVE_GRAVSOFT_FORGAS
                if(ptype == 0)
                    h = soft;
                else
                    h = All.ForceSoftening[ptype];

                if(P[no].Type == 0)
                {
                    if(h < P[no].Hsml)
                        h = P[no].Hsml;
                }
                else
                {
                    if(h < All.ForceSoftening[P[no].Type])
                        h = All.ForceSoftening[P[no].Type];
                }
#else
                h = All.ForceSoftening[ptype];
                if(h < All.ForceSoftening[P[no].Type])
                    h = All.ForceSoftening[P[no].Type];
#endif
                no = Nextnode[no];
            }
            else			/* we have an  internal node */
            {
                if(no >= All.MaxPart + MaxNodes)	/* pseudo particle */
                {
                    if(lv->mode == 0)
                    {
                        if(-1 == ev_export_particle(lv, target, no))
                            return -1;
                    }
                    no = Nextnode[no - MaxNodes];
                    continue;
                }

                nop = &Nodes[no];

                if(lv->mode == 1)
                {
                    if(nop->u.d.bitflags & (1 << BITFLAG_TOPLEVEL))	/* we reached a top-level node again, which means that we are done with the branch */
                    {
                        no = -1;
                        continue;
                    }
                }

                if(!(nop->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
                {
                    /* open cell */
                    no = nop->u.d.nextnode;
                    continue;
                }

                force_drift_node(no, All.Ti_Current);

                mass = nop->u.d.mass;

                dx = nop->u.d.s[0] - pos_x;
                dy = nop->u.d.s[1] - pos_y;
                dz = nop->u.d.s[2] - pos_z;

                dx = NEAREST(dx);
                dy = NEAREST(dy);
                dz = NEAREST(dz);
                r2 = dx * dx + dy * dy + dz * dz;

                if(r2 > rcut2)
                {
                    /* check whether we can stop walking along this branch */
                    eff_dist = rcut + 0.5 * nop->len;
                    dist = NEAREST(nop->center[0] - pos_x);

                    if(dist < -eff_dist || dist > eff_dist)
                    {
                        no = nop->u.d.sibling;
                        continue;
                    }
                    dist = NEAREST(nop->center[1] - pos_y);

                    if(dist < -eff_dist || dist > eff_dist)
                    {
                        no = nop->u.d.sibling;
                        continue;
                    }
                    dist = NEAREST(nop->center[2] - pos_z);

                    if(dist < -eff_dist || dist > eff_dist)
                    {
                        no = nop->u.d.sibling;
                        continue;
                    }
                }


                if(All.ErrTolTheta)	/* check Barnes-Hut opening criterion */
                {
                    if(nop->len * nop->len > r2 * All.ErrTolTheta * All.ErrTolTheta)
                    {
                        /* open cell */
                        no = nop->u.d.nextnode;
                        continue;
                    }
                }
                else		/* check relative opening criterion */
                {
                    if(mass * nop->len * nop->len > r2 * r2 * aold)
                    {
                        /* open cell */
                        no = nop->u.d.nextnode;
                        continue;
                    }

                    /* check in addition whether we lie inside the cell */

                    if(fabs(nop->center[0] - pos_x) < 0.60 * nop->len)
                    {
                        if(fabs(nop->center[1] - pos_y) < 0.60 * nop->len)
                        {
                            if(fabs(nop->center[2] - pos_z) < 0.60 * nop->len)
                            {
                                no = nop->u.d.nextnode;
                                continue;
                            }
                        }
                    }
                }

#ifndef ADAPTIVE_GRAVSOFT_FORGAS
                h = All.ForceSoftening[ptype];
                if(h < All.ForceSoftening[extract_max_softening_type(nop->u.d.bitflags)])
                {
                    h = All.ForceSoftening[extract_max_softening_type(nop->u.d.bitflags)];
                    if(r2 < h * h)
                    {
                        if(maskout_different_softening_flag(nop->u.d.bitflags))	/* bit-5 signals that there are particles of different softening in the node */
                        {
                            no = nop->u.d.nextnode;

                            continue;
                        }
                    }
                }
#else
                if(ptype == 0)
                    h = soft;
                else
                    h = All.ForceSoftening[ptype];

                if(h < nop->maxsoft)
                {
                    h = nop->maxsoft;
                    if(r2 < h * h)
                    {
                        no = nop->u.d.nextnode;
                        continue;
                    }
                }
#endif
                no = nop->u.d.sibling;	/* ok, node can be used */

            }

            r = sqrt(r2);

            if(r >= h)
            {
                fac = mass / (r2 * r);
                facpot = -mass / r;
            }
            else
            {
                h_inv = 1.0 / h;
                h3_inv = h_inv * h_inv * h_inv;
                u = r * h_inv;
                if(u < 0.5)
                    fac = mass * h3_inv * (10.666666666667 + u * u * (32.0 * u - 38.4));
                else
                    fac =
                        mass * h3_inv * (21.333333333333 - 48.0 * u +
                                38.4 * u * u - 10.666666666667 * u * u * u - 0.066666666667 / (u * u * u));
                if(u < 0.5)
                    wp = -2.8 + u * u * (5.333333333333 + u * u * (6.4 * u - 9.6));
                else
                    wp =
                        -3.2 + 0.066666666667 / u + u * u * (10.666666666667 +
                                u * (-16.0 + u * (9.6 - 2.133333333333 * u)));

                facpot = mass * h_inv * wp;

            }

            tabindex = (int) (asmthfac * r);

            if(tabindex < NTAB)
            {
                fac *= shortrange_table[tabindex];

                acc_x += (dx * fac);
                acc_y += (dy * fac);
                acc_z += (dz * fac);
                pot += (facpot * shortrange_table_potential[tabindex]);
                ninteractions++;
            }

        }

        if(listindex < NODELISTLENGTH)
        {
            no = input->base.NodeList[listindex];
            if(no >= 0)
            {
                no = Nodes[no].u.d.nextnode;	/* open it */
                nnodesinlist++;
                listindex++;
            }
        }
    }

        output->Acc[0] = acc_x;
        output->Acc[1] = acc_y;
        output->Acc[2] = acc_z;
        output->Ninteractions = ninteractions;
        output->Potential = pot;

    lv->Ninteractions = ninteractions;
    lv->Nnodesinlist = nnodesinlist;
    return ninteractions;
}


