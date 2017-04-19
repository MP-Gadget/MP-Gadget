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

#include "gravshort.h"

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

int force_treeev_shortrange(TreeWalkQueryGravShort * input,
        TreeWalkResultGravShort * output,
        LocalTreeWalk * lv);

/*! This function computes the gravitational forces for all active particles.
 *  If needed, a new tree is constructed, otherwise the dynamically updated
 *  tree is used.  Particles are only exported to other processors when really
 *  needed, thereby allowing a good use of the communication buffer.
 */
void grav_short_tree(void)
{
    double timeall = 0;
    double timetree, timewait, timecomm;
    if(!All.TreeGravOn)
        return;

    TreeWalk tw[1] = {0};

    tw->ev_label = "FORCETREE_SHORTRANGE";
    tw->visit = (TreeWalkVisitFunction) force_treeev_shortrange;
    tw->isactive = grav_short_isactive;
    tw->reduce = (TreeWalkReduceResultFunction) grav_short_reduce;
    tw->postprocess = (TreeWalkProcessFunction) grav_short_postprocess;
    tw->UseNodeList = 1;

    tw->query_type_elsize = sizeof(TreeWalkQueryGravShort);
    tw->result_type_elsize = sizeof(TreeWalkResultGravShort);
    tw->fill = (TreeWalkFillQueryFunction) grav_short_copy;

    walltime_measure("/Misc");

    /* allocate buffers to arrange communication */
    message(0, "Begin tree force.  (presently allocated=%g MB)\n", AllocatedBytes / (1024.0 * 1024.0));

    walltime_measure("/Misc");

    treewalk_run(tw);

    if(All.TypeOfOpeningCriterion == 1) {
        /* This will switch to the relative opening criterion for the following force computations */
        All.ErrTolTheta = 0;
    }

    /* now add things for comoving integration */

    message(0, "tree is done.\n");

    /* Now the force computation is finished */

    /*  gather some diagnostic information */

    timetree = tw->timecomp1 + tw->timecomp2 + tw->timecomp3;
    timewait = tw->timewait1 + tw->timewait2;
    timecomm= tw->timecommsumm1 + tw->timecommsumm2;

    All.TotNumOfForces += GlobNumForceUpdate;

    walltime_add("/Tree/Walk1", tw->timecomp1);
    walltime_add("/Tree/Walk2", tw->timecomp2);
    walltime_add("/Tree/PostProcess", tw->timecomp3);
    walltime_add("/Tree/Send", tw->timecommsumm1);
    walltime_add("/Tree/Recv", tw->timecommsumm2);
    walltime_add("/Tree/Wait1", tw->timewait1);
    walltime_add("/Tree/Wait2", tw->timewait2);

    timeall = walltime_measure(WALLTIME_IGNORE);

    walltime_add("/Tree/Misc", timeall - (timetree + timewait + timecomm));

}

/*! In the TreePM algorithm, the tree is walked only locally around the
 *  target coordinate.  Tree nodes that fall outside a box of half
 *  side-length Rcut= RCUT*ASMTH*MeshSize can be discarded. The short-range
 *  potential is modified by a complementary error function, multiplied
 *  with the Newtonian form. The resulting short-range suppression compared
 *  to the Newtonian force is tabulated, because looking up from this table
 *  is faster than recomputing the corresponding factor, despite the
 *  memory-access penalty (which reduces cache performance) incurred by the
 *  table.
 */
int force_treeev_shortrange(TreeWalkQueryGravShort * input,
        TreeWalkResultGravShort * output,
        LocalTreeWalk * lv)
{
    struct NODE *nop = 0;
    int no, ptype, listindex = 0;
    int nnodesinlist = 0, ninteractions = 0;
    double r2, dx, dy, dz, mass, r, fac, u, h, h_inv, h3_inv;
    double pos_x, pos_y, pos_z, aold;
    double eff_dist;
    double rcut, rcut2, dist;
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

    rcut = RCUT * ASMTH * All.BoxSize / All.Nmesh;

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
                        if(-1 == treewalk_export_particle(lv, no))
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

            if(0 == grav_apply_short_range_window(r, &fac, &facpot)) {
                acc_x += (dx * fac);
                acc_y += (dy * fac);
                acc_z += (dz * fac);
                pot += facpot;
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


