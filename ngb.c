#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "allvars.h"
#include "treewalk.h"
#include "forcetree.h"
#include "proto.h"
#include "domain.h"
#include "endrun.h"

/*! \file ngb.c
 *  \brief neighbour search by means of the tree
 *
 *  This file contains routines for neighbour finding.  We use the
 *  gravity-tree and a range-searching technique to find neighbours.
 */


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
        int mode, LocalTreeWalk * lv, enum NgbTreeFindSymmetric symmetric, int ptypemask)
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

            lv->ngblist[numngb++] = p;	
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
                        endrun(12312, "using node list but fell into mode 1. Why shall fail in this case?");
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


/*! This function constructs the neighbour tree. To this end, we actually need
 *  to construct the gravitational tree, because we use it now for the
 *  neighbour search.
 */
void ngb_treebuild(void)
{
    message(0, "Begin Ngb-tree construction.\n");

    force_treeallocate((int) (All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);

    walltime_measure("/Misc");

#ifdef DENSITY_INDEPENDENT_SPH_DEBUG
    force_treebuild(N_sph, NULL);
#else
    force_treebuild(NumPart, NULL);
#endif
    walltime_measure("/Tree/Build");

    message(0, "Ngb-Tree contruction finished \n");
}

