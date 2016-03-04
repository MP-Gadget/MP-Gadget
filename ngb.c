#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "allvars.h"
#include "proto.h"

/*! \file ngb.c
 *  \brief neighbour search by means of the tree
 *
 *  This file contains routines for neighbour finding.  We use the
 *  gravity-tree and a range-searching technique to find neighbours.
 */


/*! This routine finds all neighbours `j' that can interact with the
 *  particle `i' in the communication buffer.
 *
 *  Note that an interaction can take place if 
 *  \f$ r_{ij} < h_i \f$  OR if  \f$ r_{ij} < h_j \f$. 
 * 
 *  In the range-search this is taken into account, i.e. it is guaranteed that
 *  all particles are found that fulfil this condition, including the (more
 *  difficult) second part of it. For this purpose, each node knows the
 *  maximum h occuring among the particles it represents.
 */
int ngb_treefind_pairs(MyDouble searchcenter[3], MyFloat hsml, int target, int *startnode,
        int mode, int *nexport, int *nsend_local)
{
    int no, p, numngb, task, nexport_save;
    MyDouble dist, dx, dy, dz;
    struct NODE *current;

    nexport_save = *nexport;

    numngb = 0;

    no = *startnode;

    while(no >= 0)
    {
        if(no < All.MaxPart)	/* single particle */
        {
            p = no;
            no = Nextnode[no];

            if(P[p].Type > 0)
                continue;

            drift_particle(p, All.Ti_Current);

            dist = DMAX(P[p].Hsml, hsml);

            dx = NEAREST(P[p].Pos[0] - searchcenter[0]);
            if(dx > dist)
                continue;
            dy = NEAREST(P[p].Pos[1] - searchcenter[1]);
            if(dy > dist)
                continue;
            dz = NEAREST(P[p].Pos[2] - searchcenter[2]);
            if(dz > dist)
                continue;
            if(dx * dx + dy * dy + dz * dz > dist * dist)
                continue;

            Ngblist[numngb++] = p;	/* Note: unlike in previous versions of the code, the buffer 
                                       can hold up to all particles */
        }
        else
        {
            if(no >= All.MaxPart + MaxNodes)	/* pseudo particle */
            {
                if(mode == 1)
                    endrun(23131);

                if(Exportflag[task = DomainTask[no - (All.MaxPart + MaxNodes)]] != target)
                {
                    Exportflag[task] = target;
                    Exportnodecount[task] = NODELISTLENGTH;
                }

                if(Exportnodecount[task] == NODELISTLENGTH)
                {
                    if(*nexport >= All.BunchSize)
                    {
                        *nexport = nexport_save;
                        if(nexport_save == 0)
                            endrun(13003);	/* in this case, the buffer is too small to process even a single particle */
                        for(task = 0; task < NTask; task++)
                            nsend_local[task] = 0;
                        for(no = 0; no < nexport_save; no++)
                            nsend_local[DataIndexTable[no].Task]++;
                        return -1;
                    }
                    Exportnodecount[task] = 0;
                    Exportindex[task] = *nexport;
                    DataIndexTable[*nexport].Task = task;
                    DataIndexTable[*nexport].Index = target;
                    DataIndexTable[*nexport].IndexGet = *nexport;
                    *nexport = *nexport + 1;
                    nsend_local[task]++;
                }

                DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]++] =
                    DomainNodeIndex[no - (All.MaxPart + MaxNodes)];

                if(Exportnodecount[task] < NODELISTLENGTH)
                    DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]] = -1;

                no = Nextnode[no - MaxNodes];
                continue;
            }

            current = &Nodes[no];

            if(mode == 1)
            {
                if(current->u.d.bitflags & (1 << BITFLAG_TOPLEVEL))	/* we reached a top-level node again, which means that we are done with the branch */
                {
                    *startnode = -1;
                    return numngb;
                }
            }

            force_drift_node(no, All.Ti_Current);

            if(!(current->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
            {
                if(current->u.d.mass)	/* open cell */
                {
                    no = current->u.d.nextnode;
                    continue;
                }
            }

            dist = DMAX(Extnodes[no].hmax, hsml) + 0.5 * current->len;

            no = current->u.d.sibling;	/* in case the node can be discarded */

            dx = NEAREST(current->center[0] - searchcenter[0]);
            if(dx > dist)
                continue;
            dy = NEAREST(current->center[1] - searchcenter[1]);
            if(dy > dist)
                continue;
            dz = NEAREST(current->center[2] - searchcenter[2]);
            if(dz > dist)
                continue;
            /* now test against the minimal sphere enclosing everything */
            dist += FACT1 * current->len;
            if(dx * dx + dy * dy + dz * dz > dist * dist)
                continue;

            no = current->u.d.nextnode;	/* ok, we need to open the node */
        }
    }


    *startnode = -1;
    return numngb;
}

/* this is the internal code that looks for particles in the ngb tree from
 * searchcenter upto hsml. if symmetric is NGB_TREE_FIND_SYMMETRIC, then upto
 * max(P[i].Hsml, hsml). 
 *
 * the particle at target are marked for export. 
 * nodes are exported too if ev->UseNodeList is True.
 *
 * ptypemask is the sum of 1 << type of particle types that are returned. 
 *
 * */
/*! This function returns neighbours with distance <= hsml and returns them in
 *  Ngblist. Actually, particles in a box of half side length hsml are
 *  returned, i.e. the reduction to a sphere still needs to be done in the
 *  calling routine.
 */
int ngb_treefind_threads(MyDouble searchcenter[3], MyFloat hsml, int target, int *startnode,
        int mode, LocalEvaluator * lv,
        int *ngblist, enum NgbTreeFindSymmetric symmetric, int ptypemask)
{
    int no, p, numngb;
    MyDouble dist, dx, dy, dz;
    struct NODE *current;

    /* for now always blocking */
    int blocking = 1;
    int donotusenodelist = ! lv->ev->UseNodeList;
    numngb = 0;

    no = *startnode;

    while(no >= 0)
    {
        if(no < All.MaxPart)	/* single particle */
        {
            p = no;
            no = Nextnode[no];

            if(!((1<<P[p].Type) & ptypemask))
                continue;

            if(drift_particle_full(p, All.Ti_Current, blocking) < 0) {
                return -2;
            }

            if(symmetric == NGB_TREEFIND_SYMMETRIC) {
                dist = DMAX(P[p].Hsml, hsml);
            } else {
                dist = hsml;
            }
            dx = NEAREST(P[p].Pos[0] - searchcenter[0]);
            if(dx > dist)
                continue;
            dy = NEAREST(P[p].Pos[1] - searchcenter[1]);
            if(dy > dist)
                continue;
            dz = NEAREST(P[p].Pos[2] - searchcenter[2]);
            if(dz > dist)
                continue;
            if(dx * dx + dy * dy + dz * dz > dist * dist)
                continue;

            ngblist[numngb++] = p;	
            /* Note: unlike in previous versions of the code, the buffer 
                                       can hold up to all particles */
        }
        else
        {
            if(no >= All.MaxPart + MaxNodes)	/* pseudo particle */
            {
                if(mode == 1)
                {
                    if(donotusenodelist) {
                        no = Nextnode[no - MaxNodes];
                        continue;
                    } else {
                        endrun(12312);
                    }
                }
                if(target >= 0)	/* if no target is given, export will not occur */
                {
                    if(-1 == ev_export_particle(lv, target, no))
                        return -1;
                }

                no = Nextnode[no - MaxNodes];
                continue;

            }

            current = &Nodes[no];

            if(mode == 1)
            {
                if (!donotusenodelist) {
                    if(current->u.d.bitflags & (1 << BITFLAG_TOPLEVEL))	/* we reached a top-level node again, which means that we are done with the branch */
                    {
                        *startnode = -1;
                        return numngb;
                    }
                }
            }

            if(force_drift_node_full(no, All.Ti_Current, blocking) < 0) {
                return -2;
            }

            if(!(current->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
            {
                if(current->u.d.mass)	/* open cell */
                {
                    no = current->u.d.nextnode;
                    continue;
                }
            }

            if(symmetric == NGB_TREEFIND_SYMMETRIC) {
                dist = DMAX(Extnodes[no].hmax, hsml) + 0.5 * current->len;
            } else {
                dist = hsml + 0.5 * current->len;;
            }
            no = current->u.d.sibling;	/* in case the node can be discarded */

            dx = NEAREST(current->center[0] - searchcenter[0]);
            if(dx > dist)
                continue;
            dy = NEAREST(current->center[1] - searchcenter[1]);
            if(dy > dist)
                continue;
            dz = NEAREST(current->center[2] - searchcenter[2]);
            if(dz > dist)
                continue;
            /* now test against the minimal sphere enclosing everything */
            dist += FACT1 * current->len;
            if(dx * dx + dy * dy + dz * dz > dist * dist)
                continue;

            no = current->u.d.nextnode;	/* ok, we need to open the node */
        }
    }

    *startnode = -1;
    return numngb;
}





/*! This function returns neighbours with distance <= hsml and returns them in
 *  Ngblist. Actually, particles in a box of half side length hsml are
 *  returned, i.e. the reduction to a sphere still needs to be done in the
 *  calling routine.
 */
int ngb_treefind_variable(MyDouble searchcenter[3], MyFloat hsml, int target, int *startnode, int mode,
        int *nexport, int *nsend_local)
{
    int numngb, no, p, task, nexport_save;
    struct NODE *current;
    MyDouble dx, dy, dz, dist;

    nexport_save = *nexport;

    numngb = 0;
    no = *startnode;

    while(no >= 0)
    {
        if(no < All.MaxPart)	/* single particle */
        {
            p = no;
            no = Nextnode[no];

            if(P[p].Type > 0)
                continue;

            drift_particle(p, All.Ti_Current);

            dist = hsml;
            dx = NEAREST(P[p].Pos[0] - searchcenter[0]);
            if(dx > dist)
                continue;
            dy = NEAREST(P[p].Pos[1] - searchcenter[1]);
            if(dy > dist)
                continue;
            dz = NEAREST(P[p].Pos[2] - searchcenter[2]);
            if(dz > dist)
                continue;
            if(dx * dx + dy * dy + dz * dz > dist * dist)
                continue;

            Ngblist[numngb++] = p;
        }
        else
        {
            if(no >= All.MaxPart + MaxNodes)	/* pseudo particle */
            {
                if(mode == 1)
                    endrun(12312);

                if(target >= 0)	/* if no target is given, export will not occur */
                {
                    if(Exportflag[task = DomainTask[no - (All.MaxPart + MaxNodes)]] != target)
                    {
                        Exportflag[task] = target;
                        Exportnodecount[task] = NODELISTLENGTH;
                    }

                    if(Exportnodecount[task] == NODELISTLENGTH)
                    {
                        if(*nexport >= All.BunchSize)
                        {
                            *nexport = nexport_save;
                            if(nexport_save == 0)
                                endrun(13004);	/* in this case, the buffer is too small to process even a single particle */
                            for(task = 0; task < NTask; task++)
                                nsend_local[task] = 0;
                            for(no = 0; no < nexport_save; no++)
                                nsend_local[DataIndexTable[no].Task]++;
                            return -1;
                        }
                        Exportnodecount[task] = 0;
                        Exportindex[task] = *nexport;
                        DataIndexTable[*nexport].Task = task;
                        DataIndexTable[*nexport].Index = target;
                        DataIndexTable[*nexport].IndexGet = *nexport;
                        *nexport = *nexport + 1;
                        nsend_local[task]++;
                    }

                    DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]++] =
                        DomainNodeIndex[no - (All.MaxPart + MaxNodes)];

                    if(Exportnodecount[task] < NODELISTLENGTH)
                        DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]] = -1;
                }

                no = Nextnode[no - MaxNodes];
                continue;
            }

            current = &Nodes[no];

            if(mode == 1)
            {
                if(current->u.d.bitflags & (1 << BITFLAG_TOPLEVEL))	/* we reached a top-level node again, which means that we are done with the branch */
                {
                    *startnode = -1;
                    return numngb;
                }
            }

            force_drift_node(no, All.Ti_Current);

            if(!(current->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
            {
                if(current->u.d.mass)	/* open cell */
                {
                    no = current->u.d.nextnode;
                    continue;
                }
            }

            no = current->u.d.sibling;	/* in case the node can be discarded */

            dist = hsml + 0.5 * current->len;;
            dx = NEAREST(current->center[0] - searchcenter[0]);
            if(dx > dist)
                continue;
            dy = NEAREST(current->center[1] - searchcenter[1]);
            if(dy > dist)
                continue;
            dz = NEAREST(current->center[2] - searchcenter[2]);
            if(dz > dist)
                continue;
            /* now test against the minimal sphere enclosing everything */
            dist += FACT1 * current->len;
            if(dx * dx + dy * dy + dz * dz > dist * dist)
                continue;

            no = current->u.d.nextnode;	/* ok, we need to open the node */
        }
    }

    *startnode = -1;
    return numngb;
}



/*! Allocates memory for the neighbour list buffer.
*/
void ngb_init(void)
{

}


/*! This function constructs the neighbour tree. To this end, we actually need
 *  to construct the gravitational tree, because we use it now for the
 *  neighbour search.
 */
void ngb_treebuild(void)
{
    if(ThisTask == 0)
        printf("Begin Ngb-tree construction.\n");

    force_treeallocate((int) (All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);

    walltime_measure("/Misc");

#ifdef DENSITY_INDEPENDENT_SPH_DEBUG
    force_treebuild(N_sph, NULL);
#else
    force_treebuild(NumPart, NULL);
#endif
    walltime_measure("/Tree/Build");

    if(ThisTask == 0)
        printf("Ngb-Tree contruction finished \n");
}



int ngb_treefind_fof_primary(MyDouble searchcenter[3], MyFloat hsml, int target, int *startnode, int mode,
        int *nexport, int *nsend_local)
{
    int numngb, no, p, task, nexport_save;
    struct NODE *current;
    MyDouble dx, dy, dz, dist, r2;

    nexport_save = *nexport;

    numngb = 0;
    no = *startnode;

    while(no >= 0)
    {
        if(no < All.MaxPart)	/* single particle */
        {
            p = no;
            no = Nextnode[no];

            if(!((1 << P[p].Type) & (FOF_PRIMARY_LINK_TYPES)))
                continue;

            if(mode == 0)
                continue;

            dist = hsml;
            dx = NEAREST(P[p].Pos[0] - searchcenter[0]);
            if(dx > dist)
                continue;
            dy = NEAREST(P[p].Pos[1] - searchcenter[1]);
            if(dy > dist)
                continue;
            dz = NEAREST(P[p].Pos[2] - searchcenter[2]);
            if(dz > dist)
                continue;
            if(dx * dx + dy * dy + dz * dz > dist * dist)
                continue;

            Ngblist[numngb++] = p;
        }
        else
        {
            if(no >= All.MaxPart + MaxNodes)	/* pseudo particle */
            {
                if(mode == 1)
                    endrun(12312);

                if(mode == 0)
                {
                    if(Exportflag[task = DomainTask[no - (All.MaxPart + MaxNodes)]] != target)
                    {
                        Exportflag[task] = target;
                        Exportnodecount[task] = NODELISTLENGTH;
                    }

                    if(Exportnodecount[task] == NODELISTLENGTH)
                    {
                        if(*nexport >= All.BunchSize)
                        {
                            *nexport = nexport_save;
                            if(nexport_save == 0)
                                endrun(13005);	/* in this case, the buffer is too small to process even a single particle */
                            for(task = 0; task < NTask; task++)
                                nsend_local[task] = 0;
                            for(no = 0; no < nexport_save; no++)
                                nsend_local[DataIndexTable[no].Task]++;
                            return -1;
                        }
                        Exportnodecount[task] = 0;
                        Exportindex[task] = *nexport;
                        DataIndexTable[*nexport].Task = task;
                        DataIndexTable[*nexport].Index = target;
                        DataIndexTable[*nexport].IndexGet = *nexport;
                        *nexport = *nexport + 1;
                        nsend_local[task]++;
                    }

                    DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]++] =
                        DomainNodeIndex[no - (All.MaxPart + MaxNodes)];

                    if(Exportnodecount[task] < NODELISTLENGTH)
                        DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]] = -1;
                }

                if(mode == -1)
                {
                    *nexport = 1;
                }

                no = Nextnode[no - MaxNodes];
                continue;

            }

            current = &Nodes[no];

            if(mode == 1)
            {
                if(current->u.d.bitflags & (1 << BITFLAG_TOPLEVEL))	/* we reached a top-level node again, which means that we are done with the branch */
                {
                    *startnode = -1;
                    return numngb;
                }
            }

            if(mode == 0)
            {
                if(!(current->u.d.bitflags & (1 << BITFLAG_TOPLEVEL)))	/* we have a node with only local particles, can skip branch */
                {
                    no = current->u.d.sibling;
                    continue;
                }
            }

            no = current->u.d.sibling;	/* in case the node can be discarded */

            dist = hsml + 0.5 * current->len;;
            dx = NEAREST(current->center[0] - searchcenter[0]);
            if(dx > dist)
                continue;
            dy = NEAREST(current->center[1] - searchcenter[1]);
            if(dy > dist)
                continue;
            dz = NEAREST(current->center[2] - searchcenter[2]);
            if(dz > dist)
                continue;
            /* now test against the minimal sphere enclosing everything */
            dist += FACT1 * current->len;
            if((r2 = (dx * dx + dy * dy + dz * dz)) > dist * dist)
                continue;

            if((current->u.d.bitflags & ((1 << BITFLAG_TOPLEVEL) + (1 << BITFLAG_DEPENDS_ON_LOCAL_MASS))) == 0)	/* only use fully local nodes */
            {
                /* test whether the node is contained within the sphere */
                dist = hsml - FACT2 * current->len;
                if(dist > 0)
                    if(r2 < dist * dist)
                    {
                        if(current->u.d.bitflags & (1 << BITFLAG_INSIDE_LINKINGLENGTH))	/* already flagged */
                        {
                            /* sufficient to return only one particle inside this cell */

                            p = current->u.d.nextnode;
                            while(p >= 0)
                            {
                                if(p < All.MaxPart)
                                {
                                    if(((1 << P[p].Type) & (FOF_PRIMARY_LINK_TYPES)))
                                    {
                                        dx = NEAREST(P[p].Pos[0] - searchcenter[0]);
                                        dy = NEAREST(P[p].Pos[1] - searchcenter[1]);
                                        dz = NEAREST(P[p].Pos[2] - searchcenter[2]);
                                        if(dx * dx + dy * dy + dz * dz > hsml * hsml)
                                            break;

                                        Ngblist[numngb++] = p;
                                        break;
                                    }
                                    p = Nextnode[p];
                                }
                                else if(p >= All.MaxPart + MaxNodes)
                                    p = Nextnode[p - MaxNodes];
                                else
                                    p = Nodes[p].u.d.nextnode;
                            }
                            continue;
                        }
                        else
                        {
                            /* flag it now */
                            current->u.d.bitflags |= (1 << BITFLAG_INSIDE_LINKINGLENGTH);
                        }
                    }
            }

            no = current->u.d.nextnode;	/* ok, we need to open the node */
        }
    }

    *startnode = -1;
    return numngb;
}



/* find all particles of type FOF_PRIMARY_LINK_TYPES in smoothing length in order to find nearest one */
int ngb_treefind_fof_nearest(MyDouble searchcenter[3], MyFloat hsml, int target, int *startnode, int mode,
        int *nexport, int *nsend_local)
{
    int numngb, no, p, task, nexport_save;
    struct NODE *current;
    MyDouble dx, dy, dz, dist, r2;

#define FACT2 0.86602540
    nexport_save = *nexport;

    numngb = 0;
    no = *startnode;

    while(no >= 0)
    {
        if(no < All.MaxPart)	/* single particle */
        {
            p = no;
            no = Nextnode[no];

            if(!((1 << P[p].Type) & (FOF_PRIMARY_LINK_TYPES)))
                continue;

            dist = hsml;
            dx = NEAREST(P[p].Pos[0] - searchcenter[0]);
            if(dx > dist)
                continue;
            dy = NEAREST(P[p].Pos[1] - searchcenter[1]);
            if(dy > dist)
                continue;
            dz = NEAREST(P[p].Pos[2] - searchcenter[2]);
            if(dz > dist)
                continue;
            if(dx * dx + dy * dy + dz * dz > dist * dist)
                continue;

            Ngblist[numngb++] = p;
        }
        else
        {
            if(no >= All.MaxPart + MaxNodes)	/* pseudo particle */
            {
                if(mode == 1)
                    endrun(123192);

                if(Exportflag[task = DomainTask[no - (All.MaxPart + MaxNodes)]] != target)
                {
                    Exportflag[task] = target;
                    Exportnodecount[task] = NODELISTLENGTH;
                }

                if(Exportnodecount[task] == NODELISTLENGTH)
                {
                    if(*nexport >= All.BunchSize)
                    {
                        *nexport = nexport_save;
                        if(nexport_save == 0)
                            endrun(13005);	/* in this case, the buffer is too small to process even a single particle */
                        for(task = 0; task < NTask; task++)
                            nsend_local[task] = 0;
                        for(no = 0; no < nexport_save; no++)
                            nsend_local[DataIndexTable[no].Task]++;
                        return -1;
                    }
                    Exportnodecount[task] = 0;
                    Exportindex[task] = *nexport;
                    DataIndexTable[*nexport].Task = task;
                    DataIndexTable[*nexport].Index = target;
                    DataIndexTable[*nexport].IndexGet = *nexport;
                    *nexport = *nexport + 1;
                    nsend_local[task]++;
                }

                DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]++] =
                    DomainNodeIndex[no - (All.MaxPart + MaxNodes)];

                if(Exportnodecount[task] < NODELISTLENGTH)
                    DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]] = -1;

                no = Nextnode[no - MaxNodes];
                continue;
            }

            current = &Nodes[no];

            if(mode == 1)
            {
                if(current->u.d.bitflags & (1 << BITFLAG_TOPLEVEL))	/* we reached a top-level node again, which means that we are done with the branch */
                {
                    *startnode = -1;
                    return numngb;
                }
            }

            no = current->u.d.sibling;	/* in case the node can be discarded */

            dist = hsml + 0.5 * current->len;;
            dx = NEAREST(current->center[0] - searchcenter[0]);
            if(dx > dist)
                continue;
            dy = NEAREST(current->center[1] - searchcenter[1]);
            if(dy > dist)
                continue;
            dz = NEAREST(current->center[2] - searchcenter[2]);
            if(dz > dist)
                continue;
            /* now test against the minimal sphere enclosing everything */
            dist += FACT1 * current->len;
            if((r2 = (dx * dx + dy * dy + dz * dz)) > dist * dist)
                continue;

            no = current->u.d.nextnode;	/* ok, we need to open the node */
        }
    }

    *startnode = -1;
    return numngb;
}






