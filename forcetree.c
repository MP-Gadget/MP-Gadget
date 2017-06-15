#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "allvars.h"
#include "proto.h"
#include "domain.h"
#include "forcetree.h"
#include "mymalloc.h"
#include "endrun.h"

/*! \file forcetree.c
 *  \brief gravitational tree and code for Ewald correction
 *
 *  This file contains the computation of the gravitational force by means
 *  of a tree. The type of tree implemented is a geometrical oct-tree,
 *  starting from a cube encompassing all particles. This cube is
 *  automatically found in the domain decomposition, which also splits up
 *  the global "top-level" tree along node boundaries, moving the particles
 *  of different parts of the tree to separate processors. Tree nodes can
 *  be dynamically updated in drift/kick operations to avoid having to
 *  reconstruct the tree every timestep.
 */

struct NODE *Nodes_base,	/*!< points to the actual memory allocated for the nodes */
 *Nodes;			/*!< this is a pointer used to access the nodes which is shifted such that Nodes[All.MaxPart]
				   gives the first allocated node */


int MaxNodes;			/*!< maximum allowed number of internal nodes */
int Numnodestree;		/*!< number of (internal) nodes in each tree */


int *Nextnode;			/*!< gives next node in tree walk  (nodes array) */
int *Father;			/*!< gives parent node in tree (Prenodes array) */

/*! auxiliary variable used to set-up non-recursive walk */
static int last;


static int tree_allocated_flag = 0;


static int force_tree_build(int npart);
static int force_tree_build_single(int npart);
static void
force_treeallocate(int maxnodes, int maxpart);

static void
force_flag_localnodes(void);

static void
force_treeupdate_pseudos(int);

static void
force_update_node_recursive(int no, int sib, int father);

static void
force_create_empty_nodes(int no, int topnode, int bits, int x, int y, int z, int *nodecount, int *nextfree);

static void
force_exchange_pseudodata(void);

static void
force_insert_pseudo_particles(void);

int
force_tree_allocated()
{
    return tree_allocated_flag;
}

void
force_tree_rebuild()
{
    message(0, "Tree construction.  (presently allocated=%g MB)\n", AllocatedBytes / (1024.0 * 1024.0));

    if(force_tree_allocated()) {
        force_tree_free();
    }
    /* construct tree if needed */
    /* the tree is used in grav dens, hydro, bh and sfr */
    force_treeallocate((int) (All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);

    walltime_measure("/Misc");

    force_tree_build(NumPart);

    walltime_measure("/Tree/Build");

    message(0, "Tree construction done.\n");

}

/*! This function is a driver routine for constructing the gravitational
 *  oct-tree, which is done by calling a small number of other functions.
 */
int force_tree_build(int npart)
{
    int flag;

    do
    {
        Numnodestree = force_tree_build_single(npart);

        MPI_Allreduce(&Numnodestree, &flag, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if(flag == -1)
        {
            force_tree_free();

            message(0, "Increasing TreeAllocFactor=%g to %g", All.TreeAllocFactor, All.TreeAllocFactor*1.15);

            All.TreeAllocFactor *= 1.15;

            if(All.TreeAllocFactor > 5.0)
            {
                message(1, "An excessively large number of tree nodes were required, stopping with particle dump.\n");
                savepositions(999999, 0);
                endrun(1, "Too many tree nodes, snapshot saved.");
            }

            force_treeallocate((int) (All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);
        }
    }
    while(flag == -1);

    force_flag_localnodes();

    force_exchange_pseudodata();

    force_treeupdate_pseudos(All.MaxPart);

    return Numnodestree;
}

/* Get the subnode for a given particle and parent node.
 * This splits a parent node into 8 subregions depending on the particle position.
 * node is the parent node to split, p_i is the index of the particle we
 * are currently inserting, and shift denotes the level of the
 * tree we are currently at.
 * Returns a value between 0 and 7. If particles are very close,
 * the tree subnode is randomised.
 * */
int get_subnode(const struct NODE * node, const int nodepos, const int p_i, const int shift)
{
    int subnode=0;
    if(shift >= 0)
    {
        const peanokey morton = MORTON(p_i);
        /* Shift morton key to the right by shift bits,
         * cutting the key at the correct tree level*/
        subnode = ((morton >> shift) & 7);
    }
    else
    {
        if(P[p_i].Pos[0] > node->center[0])
            subnode += 1;
        if(P[p_i].Pos[1] > node->center[1])
            subnode += 2;
        if(P[p_i].Pos[2] > node->center[2])
            subnode += 4;
    }

#ifndef NOTREERND
    if(node->len < 1.0e-3 * All.ForceSoftening[P[p_i].Type])
    {
        /* seems like we're dealing with particles at identical (or extremely close)
         * locations. Randomize subnode index to allow tree construction. Note: Multipole moments
         * of tree are still correct, but this will only happen well below gravitational softening
         * length-scale anyway.
         */
        return (int) (7.99 * get_random_number(P[p_i].ID+nodepos));
    }
#endif
    return subnode;
}

/*! Constructs the gravitational oct-tree.
 *
 *  The index convention for accessing tree nodes is the following: the
 *  indices 0...NumPart-1 reference single particles, the indices
 *  All.MaxPart.... All.MaxPart+nodes-1 reference tree nodes. `Nodes_base'
 *  points to the first tree node, while `nodes' is shifted such that
 *  nodes[All.MaxPart] gives the first tree node. Finally, node indices
 *  with values 'All.MaxPart + MaxNodes' and larger indicate "pseudo
 *  particles", i.e. multipole moments of top-level nodes that lie on
 *  different CPUs. If such a node needs to be opened, the corresponding
 *  particle must be exported to that CPU. */
int force_tree_build_single(int npart)
{
    int i;
    int nfree = All.MaxPart;		/* index of first free node */

    /* create an empty root node  */
    {
        struct NODE *nfreep = &Nodes[nfree];	/* select first node */

        nfreep->len = DomainLen;
        for(i = 0; i < 3; i++)
            nfreep->center[i] = DomainCenter[i];
        for(i = 0; i < 8; i++)
            nfreep->u.suns[i] = -1;
        nfree++;
        /* create a set of empty nodes corresponding to the top-level domain
         * grid. We need to generate these nodes first to make sure that we have a
         * complete top-level tree which allows the easy insertion of the
         * pseudo-particles in the right place */
        int numnodes = 1;
        force_create_empty_nodes(All.MaxPart, 0, 1, 0, 0, 0, &numnodes, &nfree);
    }

    /* now we insert all particles */
    #pragma omp parallel for
    for(i = 0; i < npart; i++)
    {
        int shift = 3 * (BITS_PER_DIMENSION - 1);

        int no = 0;
        /*Can't break from openmp for*/
        if(nfree >= All.MaxPart + MaxNodes)
            continue;
        while(TopNodes[no].Daughter >= 0)
        {
            const peanokey key = P[i].Key;
            no = TopNodes[no].Daughter + (key - TopNodes[no].StartKey) / (TopNodes[no].Size / 8);
            shift -= 3;
        }

        no = TopNodes[no].Leaf;
        int th = DomainNodeIndex[no];
        int parent = -1;			/* note: will not be used below before it is changed */
        int subnode = 0;

        while(1)
        {
            if(th >= All.MaxPart)	/* we are dealing with an internal node */
            {
                int nn;
                subnode = get_subnode(&Nodes[th], th, i, shift);

                /*Protect this access as we will be changing the value in the same loop.*/
                #pragma omp atomic read
                nn = Nodes[th].u.suns[subnode];

                shift -= 3;

                if(nn >= 0)	/* ok, something is in the daughter slot already, need to continue */
                {
                    parent = th;
                    th = nn;
                }
                else
                {
                    /* here we have found an empty slot where we can attach
                     * the new particle as a leaf.
                     */
                    /*Protect write modification to Node*/
                    #pragma omp atomic write
                    Nodes[th].u.suns[subnode] = i;
                    break;	/* done for this particle */
                }
            }
            else
            {
                /* We try to insert into a leaf with a single particle.  Need
                 * to generate a new internal node at this point.
                 */
                /*Get node index to insert and mark it taken*/
                const int ninsert = atomic_fetch_and_add(&nfree, 1);
                if(ninsert >= All.MaxPart + MaxNodes)
                {
                    message(1, "maximum number %d of tree-nodes reached for particle %d.\n", MaxNodes, i);
                    break;
                }
                struct NODE *nfreep = &Nodes[ninsert];	/* select desired node */
                int j;
                const MyFloat lenhalf = 0.25 * Nodes[parent].len;

                nfreep->len = 0.5 * Nodes[parent].len;

                for(j = 0; j < 3; j++) {
                    /* Detect which quadrant we are in by testing the bits of subnode:
                     * if (subnode & [1,2,4]) is true we add lenhalf, otherwise subtract lenhalf*/
                    const int sign = (subnode & (1 << j)) ? 1 : -1;
                    nfreep->center[j] = Nodes[parent].center[j] + sign*lenhalf;
                }
                for(j = 0; j < 8; j++)
                    nfreep->u.suns[j] = -1;

                int parent_subnode = subnode;
                subnode = get_subnode(nfreep, parent, th, shift);

                nfreep->u.suns[subnode] = th;

                th = ninsert;	/* resume trying to insert the new particle at
                             * the newly created internal node */

                /* Mark this node in the parent: this goes last
                 * so that we don't access the child before it is constructed.
                 * Protect write modification to Node*/
                #pragma omp atomic write
                Nodes[parent].u.suns[parent_subnode] = ninsert;
            }
        }
    }
    if(nfree >= All.MaxPart + MaxNodes)
    {
        return -1;
    }

    /* insert the pseudo particles that represent the mass distribution of other domains */
    force_insert_pseudo_particles();


    /* now compute the multipole moments recursively */
    last = -1;

    force_update_node_recursive(All.MaxPart, -1, -1);

    if(last >= All.MaxPart)
    {
        if(last >= All.MaxPart + MaxNodes)	/* a pseudo-particle */
            Nextnode[last - MaxNodes] = -1;
        else
            Nodes[last].u.d.nextnode = -1;
    }
    else
        Nextnode[last] = -1;

    return nfree - All.MaxPart;
}



/*! This function recursively creates a set of empty tree nodes which
 *  corresponds to the top-level tree for the domain grid. This is done to
 *  ensure that this top-level tree is always "complete" so that we can easily
 *  associate the pseudo-particles of other CPUs with tree-nodes at a given
 *  level in the tree, even when the particle population is so sparse that
 *  some of these nodes are actually empty.
 */
void force_create_empty_nodes(int no, int topnode, int bits, int x, int y, int z, int *nodecount,
        int *nextfree)
{
    int i, j, k, n, sub, count;
    MyFloat lenhalf;

    if(TopNodes[topnode].Daughter >= 0)
    {
        for(i = 0; i < 2; i++)
            for(j = 0; j < 2; j++)
                for(k = 0; k < 2; k++)
                {
                    sub = 7 & peano_hilbert_key((x << 1) + i, (y << 1) + j, (z << 1) + k, bits);

                    count = i + 2 * j + 4 * k;

                    Nodes[no].u.suns[count] = *nextfree;

                    lenhalf = 0.25 * Nodes[no].len;
                    Nodes[*nextfree].len = 0.5 * Nodes[no].len;
                    Nodes[*nextfree].center[0] = Nodes[no].center[0] + (2 * i - 1) * lenhalf;
                    Nodes[*nextfree].center[1] = Nodes[no].center[1] + (2 * j - 1) * lenhalf;
                    Nodes[*nextfree].center[2] = Nodes[no].center[2] + (2 * k - 1) * lenhalf;

                    for(n = 0; n < 8; n++)
                        Nodes[*nextfree].u.suns[n] = -1;

                    if(TopNodes[TopNodes[topnode].Daughter + sub].Daughter == -1)
                        DomainNodeIndex[TopNodes[TopNodes[topnode].Daughter + sub].Leaf] = *nextfree;

                    *nextfree = *nextfree + 1;
                    *nodecount = *nodecount + 1;

                    if((*nodecount) >= MaxNodes || (*nodecount) >= MaxTopNodes)
                    {
                        endrun(11, "maximum number MaxNodes=%d of tree-nodes reached."
                                "MaxTopNodes=%d NTopnodes=%d NTopleaves=%d nodecount=%d\n",
                                MaxNodes, MaxTopNodes, NTopnodes, NTopleaves, *nodecount);
                    }

                    force_create_empty_nodes(*nextfree - 1, TopNodes[topnode].Daughter + sub,
                            bits + 1, 2 * x + i, 2 * y + j, 2 * z + k, nodecount, nextfree);
                }
    }
}



/*! this function inserts pseudo-particles which will represent the mass
 *  distribution of the other CPUs. Initially, the mass of the
 *  pseudo-particles is set to zero, and their coordinate is set to the
 *  center of the domain-cell they correspond to. These quantities will be
 *  updated later on.
 */
void force_insert_pseudo_particles(void)
{
    int i, index;

    for(i = 0; i < NTopleaves; i++)
    {
        index = DomainNodeIndex[i];

        if(DomainTask[i] != ThisTask)
            Nodes[index].u.suns[0] = All.MaxPart + MaxNodes + i;
    }
}


/*! this routine determines the multipole moments for a given internal node
 *  and all its subnodes using a recursive computation.  The result is
 *  stored in the Nodes[] structure in the sequence of this tree-walk.
 *
 *  Note that the bitflags-variable for each node is used to store in the
 *  lowest bits some special information: Bit 0 flags whether the node
 *  belongs to the top-level tree corresponding to the domain
 *  decomposition, while Bit 1 signals whether the top-level node is
 *  dependent on local mass.
 *
 *  bits 2-4 give the particle type with
 *  the maximum softening among the particles in the node, and bit 5
 *  flags whether the node contains any particles with lower softening
 *  than that.
 */
void force_update_node_recursive(int no, int sib, int father)
{
    int j, jj, p, pp, nextsib, suns[8], count_particles, multiple_flag;
    MyFloat hmax;
    MyFloat s[3], mass;
    struct particle_data *pa;

#ifndef ADAPTIVE_GRAVSOFT_FORGAS
    int maxsofttype, current_maxsofttype, diffsoftflag;
#else
    MyFloat maxsoft;
#endif

    if(no >= All.MaxPart && no < All.MaxPart + MaxNodes)	/* internal node */
    {
        for(j = 0; j < 8; j++)
        suns[j] = Nodes[no].u.suns[j];	/* this "backup" is necessary because the nextnode entry will overwrite one element (union!) */
        if(last >= 0)
        {
            if(last >= All.MaxPart)
            {
                if(last >= All.MaxPart + MaxNodes)	/* a pseudo-particle */
                    Nextnode[last - MaxNodes] = no;
                else
                    Nodes[last].u.d.nextnode = no;
            }
            else
                Nextnode[last] = no;
        }

        last = no;

        mass = 0;
        s[0] = 0;
        s[1] = 0;
        s[2] = 0;
        hmax = 0;
        count_particles = 0;

#ifndef ADAPTIVE_GRAVSOFT_FORGAS
        maxsofttype = 7;
        diffsoftflag = 0;
#else
        maxsoft = 0;
#endif

        for(j = 0; j < 8; j++)
        {
            if((p = suns[j]) >= 0)
            {
                /* check if we have a sibling on the same level */
                for(jj = j + 1; jj < 8; jj++)
                    if((pp = suns[jj]) >= 0)
                        break;

                if(jj < 8)	/* yes, we do */
                    nextsib = pp;
                else
                    nextsib = sib;

                force_update_node_recursive(p, nextsib, no);

                if(p >= All.MaxPart)	/* an internal node or pseudo particle */
                {
                    if(p >= All.MaxPart + MaxNodes)	/* a pseudo particle */
                    {
                        /* nothing to be done here because the mass of the
                         * pseudo-particle is still zero. This will be changed
                         * later.
                         */
                    }
                    else
                    {
                        mass += (Nodes[p].u.d.mass);
                        s[0] += (Nodes[p].u.d.mass * Nodes[p].u.d.s[0]);
                        s[1] += (Nodes[p].u.d.mass * Nodes[p].u.d.s[1]);
                        s[2] += (Nodes[p].u.d.mass * Nodes[p].u.d.s[2]);

                        if(Nodes[p].u.d.mass > 0)
                        {
                            if(Nodes[p].u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES))
                                count_particles += 2;
                            else
                                count_particles++;
                        }

                        if(Nodes[p].hmax > hmax)
                            hmax = Nodes[p].hmax;

#ifndef ADAPTIVE_GRAVSOFT_FORGAS
                        diffsoftflag |= maskout_different_softening_flag(Nodes[p].u.d.bitflags);

                        if(maxsofttype == 7)
                            maxsofttype = extract_max_softening_type(Nodes[p].u.d.bitflags);
                        else
                        {
                            current_maxsofttype = extract_max_softening_type(Nodes[p].u.d.bitflags);
                            if(current_maxsofttype != 7)
                            {
                                if(All.ForceSoftening[current_maxsofttype] > All.ForceSoftening[maxsofttype])
                                {
                                    maxsofttype = current_maxsofttype;
                                    diffsoftflag = (1 << BITFLAG_MIXED_SOFTENINGS_IN_NODE);
                                }
                                else
                                {
                                    if(All.ForceSoftening[current_maxsofttype] <
                                            All.ForceSoftening[maxsofttype])
                                        diffsoftflag = (1 << BITFLAG_MIXED_SOFTENINGS_IN_NODE);
                                }
                            }
                        }
#else
                        if(Nodes[p].maxsoft > maxsoft)
                            maxsoft = Nodes[p].maxsoft;
#endif
                    }
                }
                else		/* a particle */
                {
                    count_particles++;

                    pa = &P[p];

                    mass += (pa->Mass);
                    s[0] += (pa->Mass * pa->Pos[0]);
                    s[1] += (pa->Mass * pa->Pos[1]);
                    s[2] += (pa->Mass * pa->Pos[2]);

                    if(pa->Type == 0)
                    {
                        if(P[p].Hsml > hmax)
                            hmax = P[p].Hsml;
                    }
#ifndef ADAPTIVE_GRAVSOFT_FORGAS
                    if(maxsofttype == 7)
                        maxsofttype = pa->Type;
                    else
                    {
                        if(All.ForceSoftening[pa->Type] > All.ForceSoftening[maxsofttype])
                        {
                            maxsofttype = pa->Type;
                            diffsoftflag = (1 << BITFLAG_MIXED_SOFTENINGS_IN_NODE);
                        }
                        else
                        {
                            if(All.ForceSoftening[pa->Type] < All.ForceSoftening[maxsofttype])
                                diffsoftflag = (1 << BITFLAG_MIXED_SOFTENINGS_IN_NODE);
                        }
                    }
#else
                    if(pa->Type == 0)
                    {
                        if(P[p].Hsml > maxsoft)
                            maxsoft = P[p].Hsml;
                    }
                    else
                    {
                        if(All.ForceSoftening[pa->Type] > maxsoft)
                            maxsoft = All.ForceSoftening[pa->Type];
                    }
#endif
                }
            }
        }


        if(mass)
        {
            s[0] /= mass;
            s[1] /= mass;
            s[2] /= mass;
        }
        else
        {
            s[0] = Nodes[no].center[0];
            s[1] = Nodes[no].center[1];
            s[2] = Nodes[no].center[2];
        }


        Nodes[no].u.d.mass = mass;
        Nodes[no].u.d.s[0] = s[0];
        Nodes[no].u.d.s[1] = s[1];
        Nodes[no].u.d.s[2] = s[2];

        Nodes[no].hmax = hmax;

        if(count_particles > 1)	/* this flags that the node represents more than one particle */
            multiple_flag = (1 << BITFLAG_MULTIPLEPARTICLES);
        else
            multiple_flag = 0;

        Nodes[no].u.d.bitflags = multiple_flag;

#ifndef ADAPTIVE_GRAVSOFT_FORGAS
        Nodes[no].u.d.bitflags |= diffsoftflag + (maxsofttype << BITFLAG_MAX_SOFTENING_TYPE);
#else
        Nodes[no].maxsoft = maxsoft;
#endif
        Nodes[no].u.d.sibling = sib;
        Nodes[no].u.d.father = father;
    }
    else				/* single particle or pseudo particle */
    {
        if(last >= 0)
        {
            if(last >= All.MaxPart)
            {
                if(last >= All.MaxPart + MaxNodes)	/* a pseudo-particle */
                    Nextnode[last - MaxNodes] = no;
                else
                    Nodes[last].u.d.nextnode = no;
            }
            else
                Nextnode[last] = no;
        }

        last = no;

        if(no < All.MaxPart)	/* only set it for single particles */
            Father[no] = father;
    }
}




/*! This function communicates the values of the multipole moments of the
 *  top-level tree-nodes of the domain grid.  This data can then be used to
 *  update the pseudo-particles on each CPU accordingly.
 */
void force_exchange_pseudodata(void)
{
    int i, no, m, ta, recvTask;
    int *recvcounts, *recvoffset;
    struct DomainNODE
    {
        MyFloat s[3];
        MyFloat mass;
        MyFloat hmax;
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
        MyFloat maxsoft;
#endif

        unsigned int bitflags;
    }
    *DomainMoment;


    DomainMoment = (struct DomainNODE *) mymalloc("DomainMoment", NTopleaves * sizeof(struct DomainNODE));
    memset(&DomainMoment[0], 0, sizeof(DomainMoment[0]) * NTopleaves);

    for(m = 0; m < All.DomainOverDecompositionFactor; m++)
        for(i = DomainStartList[ThisTask * All.DomainOverDecompositionFactor + m];
                i <= DomainEndList[ThisTask * All.DomainOverDecompositionFactor + m]; i++)
        {
            no = DomainNodeIndex[i];

            /* read out the multipole moments from the local base cells */
            DomainMoment[i].s[0] = Nodes[no].u.d.s[0];
            DomainMoment[i].s[1] = Nodes[no].u.d.s[1];
            DomainMoment[i].s[2] = Nodes[no].u.d.s[2];
            DomainMoment[i].mass = Nodes[no].u.d.mass;
            DomainMoment[i].hmax = Nodes[no].hmax;
            DomainMoment[i].bitflags = Nodes[no].u.d.bitflags;
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
            DomainMoment[i].maxsoft = Nodes[no].maxsoft;
#endif
        }

    /* share the pseudo-particle data accross CPUs */

    recvcounts = (int *) mymalloc("recvcounts", sizeof(int) * NTask);
    recvoffset = (int *) mymalloc("recvoffset", sizeof(int) * NTask);

    for(m = 0; m < All.DomainOverDecompositionFactor; m++)
    {
        for(recvTask = 0; recvTask < NTask; recvTask++)
        {
            recvcounts[recvTask] =
                (DomainEndList[recvTask * All.DomainOverDecompositionFactor + m]
               - DomainStartList[recvTask * All.DomainOverDecompositionFactor + m] +
                 1)
             * sizeof(struct DomainNODE);
            recvoffset[recvTask] = DomainStartList[recvTask * All.DomainOverDecompositionFactor + m] * sizeof(struct DomainNODE);
        }

        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                &DomainMoment[0], recvcounts, recvoffset,
                MPI_BYTE, MPI_COMM_WORLD);
    }

    myfree(recvoffset);
    myfree(recvcounts);


    for(ta = 0; ta < NTask; ta++)
        if(ta != ThisTask)
            for(m = 0; m < All.DomainOverDecompositionFactor; m++)
                for(i = DomainStartList[ta * All.DomainOverDecompositionFactor + m]; i <= DomainEndList[ta * All.DomainOverDecompositionFactor + m]; i++)
                {
                    no = DomainNodeIndex[i];

                    Nodes[no].u.d.s[0] = DomainMoment[i].s[0];
                    Nodes[no].u.d.s[1] = DomainMoment[i].s[1];
                    Nodes[no].u.d.s[2] = DomainMoment[i].s[2];
                    Nodes[no].u.d.mass = DomainMoment[i].mass;
                    Nodes[no].hmax = DomainMoment[i].hmax;
                    Nodes[no].u.d.bitflags =
                        (Nodes[no].u.d.bitflags & (~BITFLAG_MASK)) | (DomainMoment[i].bitflags & BITFLAG_MASK);
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
                    Nodes[no].maxsoft = DomainMoment[i].maxsoft;
#endif
                }

    myfree(DomainMoment);
}


/*! This function updates the top-level tree after the multipole moments of
 *  the pseudo-particles have been updated.
 */
void force_treeupdate_pseudos(int no)
{
    int j, p, count_particles, multiple_flag;
    MyFloat hmax;
    MyFloat s[3], mass;

#ifndef ADAPTIVE_GRAVSOFT_FORGAS
    int maxsofttype, diffsoftflag, current_maxsofttype;
#else
    MyFloat maxsoft;
#endif

    mass = 0;
    s[0] = 0;
    s[1] = 0;
    s[2] = 0;
    hmax = 0;
    count_particles = 0;
#ifndef ADAPTIVE_GRAVSOFT_FORGAS
    maxsofttype = 7;
    diffsoftflag = 0;
#else
    maxsoft = 0;
#endif

    p = Nodes[no].u.d.nextnode;

    for(j = 0; j < 8; j++)	/* since we are dealing with top-level nodes, we now that there are 8 consecutive daughter nodes */
    {
        if(p >= All.MaxPart && p < All.MaxPart + MaxNodes)	/* internal node */
        {
            if(Nodes[p].u.d.bitflags & (1 << BITFLAG_INTERNAL_TOPLEVEL))
                force_treeupdate_pseudos(p);

            mass += (Nodes[p].u.d.mass);
            s[0] += (Nodes[p].u.d.mass * Nodes[p].u.d.s[0]);
            s[1] += (Nodes[p].u.d.mass * Nodes[p].u.d.s[1]);
            s[2] += (Nodes[p].u.d.mass * Nodes[p].u.d.s[2]);

            if(Nodes[p].hmax > hmax)
                hmax = Nodes[p].hmax;

            if(Nodes[p].u.d.mass > 0)
            {
                if(Nodes[p].u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES))
                    count_particles += 2;
                else
                    count_particles++;
            }

#ifndef ADAPTIVE_GRAVSOFT_FORGAS
            diffsoftflag |= maskout_different_softening_flag(Nodes[p].u.d.bitflags);

            if(maxsofttype == 7)
                maxsofttype = extract_max_softening_type(Nodes[p].u.d.bitflags);
            else
            {
                current_maxsofttype = extract_max_softening_type(Nodes[p].u.d.bitflags);
                if(current_maxsofttype != 7)
                {
                    if(All.ForceSoftening[current_maxsofttype] > All.ForceSoftening[maxsofttype])
                    {
                        maxsofttype = current_maxsofttype;
                        diffsoftflag = (1 << BITFLAG_MIXED_SOFTENINGS_IN_NODE);
                    }
                    else
                    {
                        if(All.ForceSoftening[current_maxsofttype] < All.ForceSoftening[maxsofttype])
                            diffsoftflag = (1 << BITFLAG_MIXED_SOFTENINGS_IN_NODE);
                    }
                }
            }
#else
            if(Nodes[p].maxsoft > maxsoft)
                maxsoft = Nodes[p].maxsoft;
#endif
        }
        else
            endrun(6767, "may not happen");		/* may not happen */

        p = Nodes[p].u.d.sibling;
    }

    if(mass)
    {
        s[0] /= mass;
        s[1] /= mass;
        s[2] /= mass;
    }
    else
    {
        s[0] = Nodes[no].center[0];
        s[1] = Nodes[no].center[1];
        s[2] = Nodes[no].center[2];
    }


    Nodes[no].u.d.s[0] = s[0];
    Nodes[no].u.d.s[1] = s[1];
    Nodes[no].u.d.s[2] = s[2];
    Nodes[no].u.d.mass = mass;

    Nodes[no].hmax = hmax;

    if(count_particles > 1)
        multiple_flag = (1 << BITFLAG_MULTIPLEPARTICLES);
    else
        multiple_flag = 0;

    Nodes[no].u.d.bitflags &= (~BITFLAG_MASK);	/* this clears the bits */

    Nodes[no].u.d.bitflags |= multiple_flag;

#ifndef ADAPTIVE_GRAVSOFT_FORGAS
    Nodes[no].u.d.bitflags |= diffsoftflag + (maxsofttype << BITFLAG_MAX_SOFTENING_TYPE);
#else
    Nodes[no].maxsoft = maxsoft;
#endif
}



/*! This function flags nodes in the top-level tree that are dependent on
 *  local particle data.
 */
static void
force_flag_localnodes(void)
{
    int no, i, m;

    /* mark all top-level nodes */

    for(i = 0; i < NTopleaves; i++)
    {
        no = DomainNodeIndex[i];

        while(no >= 0)
        {
            if(Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL))
                break;

            Nodes[no].u.d.bitflags |= (1 << BITFLAG_TOPLEVEL);

            no = Nodes[no].u.d.father;
        }

        /* mark also internal top level nodes */

        no = DomainNodeIndex[i];
        no = Nodes[no].u.d.father;

        while(no >= 0)
        {
            if(Nodes[no].u.d.bitflags & (1 << BITFLAG_INTERNAL_TOPLEVEL))
                break;

            Nodes[no].u.d.bitflags |= (1 << BITFLAG_INTERNAL_TOPLEVEL);

            no = Nodes[no].u.d.father;
        }
    }

    /* mark top-level nodes that contain local particles */

    for(m = 0; m < All.DomainOverDecompositionFactor; m++)
        for(i = DomainStartList[ThisTask * All.DomainOverDecompositionFactor + m];
                i <= DomainEndList[ThisTask * All.DomainOverDecompositionFactor + m]; i++)
        {
            no = DomainNodeIndex[i];

            if(DomainTask[i] != ThisTask)
                endrun(131231231, "DomainTask struct is corrupted");

            while(no >= 0)
            {
                if(Nodes[no].u.d.bitflags & (1 << BITFLAG_DEPENDS_ON_LOCAL_MASS))
                    break;

                Nodes[no].u.d.bitflags |= (1 << BITFLAG_DEPENDS_ON_LOCAL_MASS);

                no = Nodes[no].u.d.father;
            }
        }
}

/*! This function updates the hmax-values in tree nodes that hold SPH
 *  particles. These values are needed to find all neighbors in the
 *  hydro-force computation.  Since the Hsml-values are potentially changed
 *  in the SPH-density computation, force_update_hmax() should be carried
 *  out just before the hydrodynamical SPH forces are computed, i.e. after
 *  density().
 */
void force_update_hmax(void)
{
    int i, no, ta, totDomainNumChanged;
    int *domainList_all;
    int *counts, *offset_list;
    MyFloat *domainHmax_loc, *domainHmax_all;
    int *DomainList, DomainNumChanged;

    walltime_measure("/Misc");

    DomainNumChanged = 0;
    DomainList = (int *) mymalloc("DomainList", NTopleaves * sizeof(int));

    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
        if(P[i].Type == 0)
        {
            no = Father[i];

            while(no >= 0)
            {
                if(P[i].Hsml > Nodes[no].hmax)
                {
                    if(P[i].Hsml > Nodes[no].hmax)
                        Nodes[no].hmax = P[i].Hsml;

                    if(Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL))	/* we reached a top-level node */
                    {
                        DomainList[DomainNumChanged++] = no;
                        break;
                    }
                }
                else
                    break;

                no = Nodes[no].u.d.father;
            }
        }

    /* share the hmax-data of the pseudo-particles accross CPUs */

    counts = (int *) mymalloc("counts", sizeof(int) * NTask);
    offset_list = (int *) mymalloc("offset_list", sizeof(int) * NTask);

    domainHmax_loc = (MyFloat *) mymalloc("domainHmax_loc", DomainNumChanged * sizeof(MyFloat));

    for(i = 0; i < DomainNumChanged; i++)
    {
        domainHmax_loc[i] = Nodes[DomainList[i]].hmax;
    }


    MPI_Allgather(&DomainNumChanged, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);

    for(ta = 0, totDomainNumChanged = 0, offset_list[0] = 0; ta < NTask; ta++)
    {
        totDomainNumChanged += counts[ta];
        if(ta > 0)
        {
            offset_list[ta] = offset_list[ta - 1] + counts[ta - 1];
        }
    }

    message(0, "Hmax exchange: %d topleaves out of %d\n", totDomainNumChanged, NTopleaves);

    domainHmax_all = (MyFloat *) mymalloc("domainHmax_all", totDomainNumChanged * sizeof(MyFloat));
    domainList_all = (int *) mymalloc("domainList_all", totDomainNumChanged * sizeof(int));

    MPI_Allgatherv(DomainList, DomainNumChanged, MPI_INT,
            domainList_all, counts, offset_list, MPI_INT, MPI_COMM_WORLD);

    for(ta = 0; ta < NTask; ta++) {
        counts[ta] *= sizeof(MyFloat);
        offset_list[ta] *= sizeof(MyFloat);
    }

    MPI_Allgatherv(domainHmax_loc, DomainNumChanged * sizeof(MyFloat), MPI_BYTE,
            domainHmax_all, counts, offset_list, MPI_BYTE, MPI_COMM_WORLD);


    for(i = 0; i < totDomainNumChanged; i++)
    {
        no = domainList_all[i];

        if(Nodes[no].u.d.bitflags & (1 << BITFLAG_DEPENDS_ON_LOCAL_MASS))	/* to avoid that the hmax is updated twice */
            no = Nodes[no].u.d.father;

        while(no >= 0)
        {
            if(domainHmax_all[i] > Nodes[no].hmax)
            {
                    Nodes[no].hmax = domainHmax_all[i];
            }
            else
                break;

            no = Nodes[no].u.d.father;
        }
    }


    myfree(domainList_all);
    myfree(domainHmax_all);
    myfree(domainHmax_loc);
    myfree(offset_list);
    myfree(counts);
    myfree(DomainList);

    walltime_measure("/Tree/HmaxUpdate");
}

/*! This function allocates the memory used for storage of the tree and of
 *  auxiliary arrays needed for tree-walk and link-lists.  Usually,
 *  maxnodes approximately equal to 0.7*maxpart is sufficient to store the
 *  tree for up to maxpart particles.
 */
void force_treeallocate(int maxnodes, int maxpart)
{
    size_t bytes;
    double allbytes = 0, allbytes_topleaves = 0;

    tree_allocated_flag = 1;
    DomainNodeIndex = (int *) mymalloc("DomainNodeIndex", bytes = NTopleaves * sizeof(int));
    allbytes_topleaves += bytes;
    MaxNodes = maxnodes;
    if(!(Nodes_base = (struct NODE *) mymalloc("Nodes_base", bytes = (MaxNodes + 1) * sizeof(struct NODE))))
    {
        endrun(3, "failed to allocate memory for %d tree-nodes (%g MB).\n", MaxNodes, bytes / (1024.0 * 1024.0));
    }
    allbytes += bytes;
    allbytes += bytes;
    Nodes = Nodes_base - All.MaxPart;
    if(!(Nextnode = (int *) mymalloc("Nextnode", bytes = (maxpart + NTopnodes) * sizeof(int))))
    {
        endrun(1, "Failed to allocate %d spaces for 'Nextnode' array (%g MB)\n",
                maxpart + NTopnodes, bytes / (1024.0 * 1024.0));
    }
    allbytes += bytes;
    if(!(Father = (int *) mymalloc("Father", bytes = (maxpart) * sizeof(int))))
    {
        endrun(1, "Failed to allocate %d spaces for 'Father' array (%g MB)\n", maxpart, bytes / (1024.0 * 1024.0));
    }
    allbytes += bytes;

    message(0, "Allocated %g MByte for BH-tree, and %g Mbyte for top-leaves.  (presently allocated %g MB)\n",
             allbytes / (1024.0 * 1024.0), allbytes_topleaves / (1024.0 * 1024.0),
             AllocatedBytes / (1024.0 * 1024.0));
}


/*! This function frees the memory allocated for the tree, i.e. it frees
 *  the space allocated by the function force_treeallocate().
 */
void force_tree_free(void)
{
    myfree(Father);
    myfree(Nextnode);
    myfree(Nodes_base);
    myfree(DomainNodeIndex);
    tree_allocated_flag = 0;
}

