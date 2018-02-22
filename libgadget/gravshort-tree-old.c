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
#include "drift.h"
#include "forcetree.h"
#include "treewalk.h"
#include "timestep.h"
#include "mymalloc.h"
#include "domain.h"
#include "endrun.h"

#include "gravshort.h"

void set_softenings(void);
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

/* intentially duplicated from those in longrange.c */
/* length of lock-up table for short-range force kernel in TreePM algorithm */
#define NTAB 1000
/* variables for short-range lookup table */
static float shortrange_table[NTAB], shortrange_table_potential[NTAB], shortrange_table_tidal[NTAB];

static void
fill_ntab()
{
    int i;
    for(i = 0; i < NTAB; i++)
    {
        double u = 3.0 / NTAB * (i + 0.5);
        shortrange_table[i] = erfc(u) + 2.0 * u / sqrt(M_PI) * exp(-u * u);
        shortrange_table_potential[i] = erfc(u);
        shortrange_table_tidal[i] = 4.0 * u * u * u / sqrt(M_PI) * exp(-u * u);
    }
}

static int force_treeevaluate_shortrange(TreeWalkQueryGravShort * input,
        TreeWalkResultGravShort * output,
        LocalTreeWalk * lv);


/*! This function computes the gravitational forces for all active particles.
 *  If needed, a new tree is constructed, otherwise the dynamically updated
 *  tree is used.  Particles are only exported to other processors when really
 *  needed, thereby allowing a good use of the communication buffer.
 */
void grav_short_tree_old(void)
{
    double timeall = 0;
    double timetree, timewait, timecomm;

    fill_ntab();

    set_softenings();
    if(!All.TreeGravOn)
        return;

    TreeWalk tw[1] = {0};

    tw->ev_label = "FORCETREE_SHORTRANGE";
    tw->visit = (TreeWalkVisitFunction) force_treeevaluate_shortrange;
    tw->haswork = grav_short_haswork;
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

    treewalk_run(tw, ActiveParticle, NumActiveParticle);

    /* now add things for comoving integration */

    message(0, "tree is done.\n");

    /* Now the force computation is finished */

    /*  gather some diagnostic information */

    timetree = tw->timecomp1 + tw->timecomp2 + tw->timecomp3;
    timewait = tw->timewait1 + tw->timewait2;
    timecomm= tw->timecommsumm1 + tw->timecommsumm2;

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
 *  memory-access panelty (which reduces cache performance) incurred by the
 *  table.
 */
#define PERIODIC
#define EVALPOTENTIAL
#define UNEQUALSOFTENINGS

static int
force_treeevaluate_shortrange(TreeWalkQueryGravShort * input,
        TreeWalkResultGravShort * output,
        LocalTreeWalk * lv)
{
    struct NODE *nop = 0;
    int no, ptype, tabindex, listindex = 0;
    int nnodesinlist = 0, ninteractions = 0;
    double r2, dx, dy, dz, mass, r, fac, u, h, h_inv, h3_inv;
    double pos_x, pos_y, pos_z, aold;
    double eff_dist;
    const double asmth = All.Asmth * All.BoxSize / All.Nmesh;
    const double rcut = RCUT * asmth;
    double asmthfac, rcut2, dist;
    double acc_x, acc_y, acc_z;

#ifdef DISTORTIONTENSORPS
    int i1, i2;
    double fac2, h5_inv;
    double fac_tidal;
    MyDouble tidal_tensorps[3][3];
#endif

#ifdef SCALARFIELD
    double dx_dm = 0, dy_dm = 0, dz_dm = 0, mass_dm = 0;
#endif
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
    double soft = 0;
#endif
#ifdef EVALPOTENTIAL
    double wp, facpot;
    double pot;

    pot = 0;
#endif

#ifdef DISTORTIONTENSORPS
    for(i1 = 0; i1 < 3; i1++)
        for(i2 = 0; i2 < 3; i2++)
            tidal_tensorps[i1][i2] = 0.0;
#endif


    acc_x = 0;
    acc_y = 0;
    acc_z = 0;
    ninteractions = 0;
    nnodesinlist = 0;

    no = input->base.NodeList[0];
    listindex ++;
    no = Nodes[no].u.d.nextnode;	/* open it */

    pos_x = input->base.Pos[0];
    pos_y = input->base.Pos[1];
    pos_z = input->base.Pos[2];
#if defined(UNEQUALSOFTENINGS) || defined(SCALARFIELD)
    ptype = input->Type;
#else
    ptype = P[0].Type;
#endif
    aold = All.ErrTolForceAcc * input->OldAcc;
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
    if(ptype == 0)
        soft = input->Soft;
#endif

    rcut2 = rcut * rcut;

    asmthfac = 0.5 / asmth * (NTAB / 3.0);

#ifndef UNEQUALSOFTENINGS
    h = All.ForceSoftening[ptype];
    h_inv = 1.0 / h;
    h3_inv = h_inv * h_inv * h_inv;
#ifdef DISTORTIONTENSORPS
    h5_inv = h_inv * h_inv * h_inv * h_inv * h_inv;
#endif
#endif

    while(no >= 0)
    {
        while(no >= 0)
        {
            if(node_is_particle(no))
            {
                /* the index of the node is the index of the particle */
                drift_particle(no, All.Ti_Current);

                dx = P[no].Pos[0] - pos_x;
                dy = P[no].Pos[1] - pos_y;
                dz = P[no].Pos[2] - pos_z;
#ifdef PERIODIC
                dx = NEAREST(dx);
                dy = NEAREST(dy);
                dz = NEAREST(dz);
#endif
                r2 = dx * dx + dy * dy + dz * dz;

                mass = P[no].Mass;
#ifdef SCALARFIELD
                if(ptype != 0)	/* we have a dark matter particle as target */
                {
                    if(P[no].Type == 1)
                    {
                        dx_dm = dx;
                        dy_dm = dy;
                        dz_dm = dz;
                        mass_dm = mass;
                    }
                    else
                    {
                        mass_dm = 0;
                        dx_dm = dy_dm = dz_dm = 0;
                    }
                }
#endif
#ifdef UNEQUALSOFTENINGS
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
#endif
                no = Nextnode[no];
            }
            else			/* we have an  internal node */
            {
                if(node_is_pseudo_particle(no))	/* pseudo particle */
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
                    if(nop->f.TopLevel)	/* we reached a top-level node again, which means that we are done with the branch */
                    {
                        no = -1;
                        continue;
                    }
                }

                mass = nop->u.d.mass;

                dx = nop->u.d.s[0] - pos_x;
                dy = nop->u.d.s[1] - pos_y;
                dz = nop->u.d.s[2] - pos_z;

#ifdef SCALARFIELD
                if(ptype != 0)	/* we have a dark matter particle as target */
                {
                    dx_dm = nop->s_dm[0] - pos_x;
                    dy_dm = nop->s_dm[1] - pos_y;
                    dz_dm = nop->s_dm[2] - pos_z;
                    mass_dm = nop->mass_dm;
                }
                else
                {
                    mass_dm = 0;
                    dx_dm = dy_dm = dz_dm = 0;
                }
#endif

#ifdef PERIODIC
                dx = NEAREST(dx);
                dy = NEAREST(dy);
                dz = NEAREST(dz);
#endif
                r2 = dx * dx + dy * dy + dz * dz;

                if(r2 > rcut2)
                {
                    /* check whether we can stop walking along this branch */
                    eff_dist = rcut + 0.5 * nop->len;
#ifdef PERIODIC
                    dist = NEAREST(nop->center[0] - pos_x);
#else
                    dist = nop->center[0] - pos_x;
#endif
                    if(dist < -eff_dist || dist > eff_dist)
                    {
                        no = nop->u.d.sibling;
                        continue;
                    }
#ifdef PERIODIC
                    dist = NEAREST(nop->center[1] - pos_y);
#else
                    dist = nop->center[1] - pos_y;
#endif
                    if(dist < -eff_dist || dist > eff_dist)
                    {
                        no = nop->u.d.sibling;
                        continue;
                    }
#ifdef PERIODIC
                    dist = NEAREST(nop->center[2] - pos_z);
#else
                    dist = nop->center[2] - pos_z;
#endif
                    if(dist < -eff_dist || dist > eff_dist)
                    {
                        no = nop->u.d.sibling;
                        continue;
                    }
                }


                /* check relative opening criterion */
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

#ifdef UNEQUALSOFTENINGS
#ifndef ADAPTIVE_GRAVSOFT_FORGAS
                h = All.ForceSoftening[ptype];
                if(h < All.ForceSoftening[nop->f.MaxSofteningType])
                {
                    h = All.ForceSoftening[nop->f.MaxSofteningType];
                    if(r2 < h * h)
                    {
                        if(nop->f.MixedSofteningsInNode)
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
#endif
                no = nop->u.d.sibling;	/* ok, node can be used */

            }

            r = sqrt(r2);

            if(r >= h)
            {
                fac = mass / (r2 * r);
#ifdef DISTORTIONTENSORPS
                /* second derivative of potential needs this factor */
                fac2 = 3.0 * mass / (r2 * r2 * r);
#endif
#ifdef EVALPOTENTIAL
                facpot = -mass / r;
#endif
            }
            else
            {
#ifdef UNEQUALSOFTENINGS
                h_inv = 1.0 / h;
                h3_inv = h_inv * h_inv * h_inv;
#ifdef DISTORTIONTENSORPS
                h5_inv = h_inv * h_inv * h_inv * h_inv * h_inv;
#endif
#endif
                u = r * h_inv;
                if(u < 0.5)
                    fac = mass * h3_inv * (10.666666666667 + u * u * (32.0 * u - 38.4));
                else
                    fac =
                        mass * h3_inv * (21.333333333333 - 48.0 * u +
                                38.4 * u * u - 10.666666666667 * u * u * u - 0.066666666667 / (u * u * u));
#ifdef EVALPOTENTIAL
                if(u < 0.5)
                    wp = -2.8 + u * u * (5.333333333333 + u * u * (6.4 * u - 9.6));
                else
                    wp =
                        -3.2 + 0.066666666667 / u + u * u * (10.666666666667 +
                                u * (-16.0 + u * (9.6 - 2.133333333333 * u)));

                facpot = mass * h_inv * wp;
#endif
#ifdef DISTORTIONTENSORPS
                /*second derivates needed -> calculate them from softend potential,
                  (see Gadget 1 paper and there g2 function). SIGN?! */
                if(u < 0.5)
                    fac2 = mass * h5_inv * (76.8 - 96.0 * u);
                else
                    fac2 = mass * h5_inv * (-0.2 / (u * u * u * u * u) + 48.0 / u - 76.8 + 32.0 * u);
#endif

            }

            tabindex = (int) (asmthfac * r);

            if(tabindex < NTAB)
            {
#ifdef DISTORTIONTENSORPS
                /* save original fac without shortrange_table facor (needed for tidal field calculation) */
                fac_tidal = fac;
#endif
                fac *= shortrange_table[tabindex];

                acc_x += (dx * fac);
                acc_y += (dy * fac);
                acc_z += (dz * fac);
#ifdef DISTORTIONTENSORPS
                /*
                   tidal_tensorps[][] = Matrix of second derivatives of grav. potential, symmetric:
                   |Txx Txy Txz|   |tidal_tensorps[0][0] tidal_tensorps[0][1] tidal_tensorps[0][2]|
                   |Tyx Tyy Tyz| = |tidal_tensorps[1][0] tidal_tensorps[1][1] tidal_tensorps[1][2]| 
                   |Tzx Tzy Tzz|   |tidal_tensorps[2][0] tidal_tensorps[2][1] tidal_tensorps[2][2]|
                   */

                tidal_tensorps[0][0] += ((-fac_tidal + dx * dx * fac2) * shortrange_table[tabindex]) +
                    dx * dx * fac2 / 3.0 * shortrange_table_tidal[tabindex];
                tidal_tensorps[0][1] += ((dx * dy * fac2) * shortrange_table[tabindex]) +
                    dx * dy * fac2 / 3.0 * shortrange_table_tidal[tabindex];
                tidal_tensorps[0][2] += ((dx * dz * fac2) * shortrange_table[tabindex]) +
                    dx * dz * fac2 / 3.0 * shortrange_table_tidal[tabindex];
                tidal_tensorps[1][1] += ((-fac_tidal + dy * dy * fac2) * shortrange_table[tabindex]) +
                    dy * dy * fac2 / 3.0 * shortrange_table_tidal[tabindex];
                tidal_tensorps[1][2] += ((dy * dz * fac2) * shortrange_table[tabindex]) +
                    dy * dz * fac2 / 3.0 * shortrange_table_tidal[tabindex];
                tidal_tensorps[2][2] += ((-fac_tidal + dz * dz * fac2) * shortrange_table[tabindex]) +
                    dz * dz * fac2 / 3.0 * shortrange_table_tidal[tabindex];
                tidal_tensorps[1][0] = tidal_tensorps[0][1];
                tidal_tensorps[2][0] = tidal_tensorps[0][2];
                tidal_tensorps[2][1] = tidal_tensorps[1][2];
#endif
#ifdef EVALPOTENTIAL
                pot += (facpot * shortrange_table_potential[tabindex]);
#endif
                ninteractions++;
            }


#ifdef SCALARFIELD
            if(ptype != 0)	/* we have a dark matter particle as target */
            {
#ifdef PERIODIC
                dx_dm = NEAREST(dx_dm);
                dy_dm = NEAREST(dy_dm);
                dz_dm = NEAREST(dz_dm);
#endif
                r2 = dx_dm * dx_dm + dy_dm * dy_dm + dz_dm * dz_dm;
                r = sqrt(r2);
                if(r >= h)
                    fac = mass_dm / (r2 * r);
                else
                {
#ifdef UNEQUALSOFTENINGS
                    h_inv = 1.0 / h;
                    h3_inv = h_inv * h_inv * h_inv;
#endif
                    u = r * h_inv;
                    if(u < 0.5)
                        fac = mass_dm * h3_inv * (10.666666666667 + u * u * (32.0 * u - 38.4));
                    else
                        fac =
                            mass_dm * h3_inv * (21.333333333333 - 48.0 * u +
                                    38.4 * u * u - 10.666666666667 * u * u * u -
                                    0.066666666667 / (u * u * u));
                }

                /* assemble force with strength, screening length, and target charge.  */

                fac *=
                    All.ScalarBeta * (1 + r / All.ScalarScreeningLength) * exp(-r / All.ScalarScreeningLength);
                tabindex = (int) (asmthfac * r);
                if(tabindex < NTAB)
                {
                    fac *= shortrange_table[tabindex];
                    acc_x += FLT(dx_dm * fac);
                    acc_y += FLT(dy_dm * fac);
                    acc_z += FLT(dz_dm * fac);
                }
            }
#endif
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
#ifdef EVALPOTENTIAL
        output->Potential = pot;
#endif
#ifdef DISTORTIONTENSORPS
        for(i1 = 0; i1 < 3; i1++)
            for(i2 = 0; i2 < 3; i2++)
                output->tidal_tensorps[i1][i2] = tidal_tensorps[i1][i2];
#endif

    lv->Ninteractions = ninteractions;
    lv->Nnodesinlist = nnodesinlist;
    return ninteractions;
}
