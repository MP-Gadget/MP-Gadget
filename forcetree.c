#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "allvars.h"
#include "proto.h"
#include "domain.h"

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

/*! auxialiary variable used to set-up non-recursive walk */
static int last;



/*! length of lock-up table for short-range force kernel in TreePM algorithm */
#define NTAB 1000
/*! variables for short-range lookup table */
static float tabfac, shortrange_table[NTAB], shortrange_table_potential[NTAB], shortrange_table_tidal[NTAB];

/*! toggles after first tree-memory allocation, has only influence on log-files */
static int first_flag = 0;

static int tree_allocated_flag = 0;


void force_treebuild_simple() {
    /* construct tree if needed */
    /* the tree is used in grav dens, hydro, bh and sfr */
    force_treeallocate((int) (All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);

    if(ThisTask == 0)
        printf("Tree construction.  (presently allocated=%g MB)\n", AllocatedBytes / (1024.0 * 1024.0));

    walltime_measure("/Misc");

#if defined(SFR) || defined(BLACK_HOLES)
    rearrange_particle_sequence();
#endif

    force_treebuild(NumPart, NULL);

    walltime_measure("/Tree/Build");

    if(ThisTask == 0)
        printf("Tree construction done.\n");

}

/*! This function is a driver routine for constructing the gravitational
 *  oct-tree, which is done by calling a small number of other functions.
 */
int force_treebuild(int npart, struct unbind_data *mp)
{
    int flag;

    do
    {
        Numnodestree = force_treebuild_single(npart, mp);

        MPI_Allreduce(&Numnodestree, &flag, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if(flag == -1)
        {
            force_treefree();

            if(ThisTask == 0)
                printf("Increasing TreeAllocFactor=%g", All.TreeAllocFactor);

            All.TreeAllocFactor *= 1.15;

            if(ThisTask == 0)
                printf("new value=%g\n", All.TreeAllocFactor);

            force_treeallocate((int) (All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);
        }
    }
    while(flag == -1);

    force_flag_localnodes();

    force_exchange_pseudodata();

    force_treeupdate_pseudos(All.MaxPart);

    TimeOfLastTreeConstruction = All.Time;

    return Numnodestree;
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
 *  particle must be exported to that CPU. The 'Extnodes' structure
 *  parallels that of 'Nodes'. Its information is only needed for the SPH
 *  part of the computation. (The data is split onto these two structures
 *  as a tuning measure.  If it is merged into 'Nodes' a somewhat bigger
 *  size of the nodes also for gravity would result, which would reduce
 *  cache utilization slightly.
 */
int force_treebuild_single(int npart, struct unbind_data *mp)
{
    int i, j, k, subnode = 0, shift, parent, numnodes, rep;
    int nfree, th, nn, no;
    struct NODE *nfreep;
    MyFloat lenhalf;
    peanokey key, morton, th_key, *morton_list;


    /* create an empty root node  */
    nfree = All.MaxPart;		/* index of first free node */
    nfreep = &Nodes[nfree];	/* select first node */

    nfreep->len = DomainLen;
    for(j = 0; j < 3; j++)
        nfreep->center[j] = DomainCenter[j];
    for(j = 0; j < 8; j++)
        nfreep->u.suns[j] = -1;


    numnodes = 1;
    nfreep++;
    nfree++;

    /* create a set of empty nodes corresponding to the top-level domain
     * grid. We need to generate these nodes first to make sure that we have a
     * complete top-level tree which allows the easy insertion of the
     * pseudo-particles at the right place 
     */

    force_create_empty_nodes(All.MaxPart, 0, 1, 0, 0, 0, &numnodes, &nfree);

    /* if a high-resolution region in a global tree is used, we need to generate
     * an additional set empty nodes to make sure that we have a complete
     * top-level tree for the high-resolution inset
     */

    nfreep = &Nodes[nfree];
    parent = -1;			/* note: will not be used below before it is changed */

    morton_list = (peanokey *) mymalloc("morton_list", NumPart * sizeof(peanokey));

    /* now we insert all particles */
    for(k = 0; k < npart; k++)
    {
        if(mp)
            i = mp[k].index;
        else
            i = k;

#ifdef NEUTRINOS
        if(P[i].Type == 2)
            continue;
#endif

        rep = 0;

        key = peano_and_morton_key((int) ((P[i].Pos[0] - DomainCorner[0]) * DomainFac),
                (int) ((P[i].Pos[1] - DomainCorner[1]) * DomainFac),
                (int) ((P[i].Pos[2] - DomainCorner[2]) * DomainFac), BITS_PER_DIMENSION,
                &morton);
        morton_list[i] = morton;

        shift = 3 * (BITS_PER_DIMENSION - 1);

        no = 0;
        while(TopNodes[no].Daughter >= 0)
        {
            no = TopNodes[no].Daughter + (key - TopNodes[no].StartKey) / (TopNodes[no].Size / 8);
            shift -= 3;
            rep++;
        }

        no = TopNodes[no].Leaf;
        th = DomainNodeIndex[no];

        while(1)
        {
            if(th >= All.MaxPart)	/* we are dealing with an internal node */
            {
                if(shift >= 0)
                {
                    subnode = ((morton >> shift) & 7);
                }
                else
                {
                    subnode = 0;
                    if(P[i].Pos[0] > Nodes[th].center[0])
                        subnode += 1;
                    if(P[i].Pos[1] > Nodes[th].center[1])
                        subnode += 2;
                    if(P[i].Pos[2] > Nodes[th].center[2])
                        subnode += 4;
                }

#ifndef NOTREERND
                if(Nodes[th].len < 1.0e-3 * All.ForceSoftening[P[i].Type])
                {
                    /* seems like we're dealing with particles at identical (or extremely close)
                     * locations. Randomize subnode index to allow tree construction. Note: Multipole moments
                     * of tree are still correct, but this will only happen well below gravitational softening
                     * length-scale anyway.
                     */
                    subnode = (int) (8.0 * get_random_number((P[i].ID + rep) % (RNDTABLE + (rep & 3))));

                    if(subnode >= 8)
                        subnode = 7;
                }
#endif

                nn = Nodes[th].u.suns[subnode];

                shift -= 3;

                if(nn >= 0)	/* ok, something is in the daughter slot already, need to continue */
                {
                    parent = th;
                    th = nn;
                    rep++;
                }
                else
                {
                    /* here we have found an empty slot where we can attach
                     * the new particle as a leaf.
                     */
                    Nodes[th].u.suns[subnode] = i;
                    break;	/* done for this particle */
                }
            }
            else
            {
                /* We try to insert into a leaf with a single particle.  Need
                 * to generate a new internal node at this point.
                 */
                Nodes[parent].u.suns[subnode] = nfree;

                nfreep->len = 0.5 * Nodes[parent].len;
                lenhalf = 0.25 * Nodes[parent].len;

                if(subnode & 1)
                    nfreep->center[0] = Nodes[parent].center[0] + lenhalf;
                else
                    nfreep->center[0] = Nodes[parent].center[0] - lenhalf;

                if(subnode & 2)
                    nfreep->center[1] = Nodes[parent].center[1] + lenhalf;
                else
                    nfreep->center[1] = Nodes[parent].center[1] - lenhalf;

                if(subnode & 4)
                    nfreep->center[2] = Nodes[parent].center[2] + lenhalf;
                else
                    nfreep->center[2] = Nodes[parent].center[2] - lenhalf;

                nfreep->u.suns[0] = -1;
                nfreep->u.suns[1] = -1;
                nfreep->u.suns[2] = -1;
                nfreep->u.suns[3] = -1;
                nfreep->u.suns[4] = -1;
                nfreep->u.suns[5] = -1;
                nfreep->u.suns[6] = -1;
                nfreep->u.suns[7] = -1;

                if(shift >= 0)
                {
                    th_key = morton_list[th];
                    subnode = ((th_key >> shift) & 7);
                }
                else
                {
                    subnode = 0;
                    if(P[th].Pos[0] > nfreep->center[0])
                        subnode += 1;
                    if(P[th].Pos[1] > nfreep->center[1])
                        subnode += 2;
                    if(P[th].Pos[2] > nfreep->center[2])
                        subnode += 4;
                }

#ifndef NOTREERND
                if(nfreep->len < 1.0e-3 * All.ForceSoftening[P[th].Type])
                {
                    /* seems like we're dealing with particles at identical (or extremely close)
                     * locations. Randomize subnode index to allow tree construction. Note: Multipole moments
                     * of tree are still correct, but this will only happen well below gravitational softening
                     * length-scale anyway.
                     */
                    subnode = (int) (8.0 * get_random_number((P[th].ID + rep) % (RNDTABLE + (rep & 3))));

                    if(subnode >= 8)
                        subnode = 7;
                }
#endif
                nfreep->u.suns[subnode] = th;

                th = nfree;	/* resume trying to insert the new particle at
                             * the newly created internal node
                             */

                numnodes++;
                nfree++;
                nfreep++;

                if((numnodes) >= MaxNodes)
                {
                    printf("task %d: maximum number %d of tree-nodes reached for particle %d.\n", ThisTask,
                            MaxNodes, i);

                    if(All.TreeAllocFactor > 5.0)
                    {
                        printf
                            ("task %d: looks like a serious problem for particle %d, stopping with particle dump.\n",
                             ThisTask, i);
                        dump_particles();
                        endrun(1);
                    }
                    else
                    {
                        myfree(morton_list);
                        return -1;
                    }
                }
            }
        }
    }

    myfree(morton_list);


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

    return numnodes;
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
                        printf("task %d: maximum number MaxNodes=%d of tree-nodes reached."
                                "MaxTopNodes=%d NTopnodes=%d NTopleaves=%d nodecount=%d\n",
                                ThisTask, MaxNodes, MaxTopNodes, NTopnodes, NTopleaves, *nodecount);
                        printf("in create empty nodes\n");
                        dump_particles();
                        endrun(11);
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
 *  If UNEQUALSOFTENINGS is set, bits 2-4 give the particle type with
 *  the maximum softening among the particles in the node, and bit 5
 *  flags whether the node contains any particles with lower softening
 *  than that.  
 */
void force_update_node_recursive(int no, int sib, int father)
{
    int j, jj, k, p, pp, nextsib, suns[8], count_particles, multiple_flag;
    MyFloat hmax, vmax, v, divVmax;
    MyFloat s[3], vs[3], mass;
    struct particle_data *pa;

#ifdef SCALARFIELD
    MyFloat s_dm[3], vs_dm[3], mass_dm;
#endif
#ifdef RADTRANSFER
    MyFloat stellar_mass;
    MyFloat stellar_s[3];

#ifdef RT_RAD_PRESSURE
    MyFloat bh_mass;
    MyFloat bh_s[3];
#endif
#endif
#ifdef UNEQUALSOFTENINGS
#ifndef ADAPTIVE_GRAVSOFT_FORGAS
    int maxsofttype, current_maxsofttype, diffsoftflag;
#else
    MyFloat maxsoft;
#endif
#endif

    if(no >= All.MaxPart && no < All.MaxPart + MaxNodes)	/* internal node */
    {
        for(j = 0; j < 8; j++)
#ifdef GRAVITY_CENTROID
            suns[j] = Extnodes[no].suns[j] = Nodes[no].u.suns[j];
#else
        suns[j] = Nodes[no].u.suns[j];	/* this "backup" is necessary because the nextnode entry will overwrite one element (union!) */
#endif
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

#ifdef RADTRANSFER
        stellar_mass = 0;
        stellar_s[0] = 0;
        stellar_s[1] = 0;
        stellar_s[2] = 0;
#ifdef RT_RAD_PRESSURE
        bh_mass = bh_s[0] = bh_s[1] = bh_s[2] = 0.0;
#endif
#endif
#ifdef SCALARFIELD
        mass_dm = 0;
        s_dm[0] = vs_dm[0] = 0;
        s_dm[1] = vs_dm[1] = 0;
        s_dm[2] = vs_dm[2] = 0;
#endif
        mass = 0;
        s[0] = 0;
        s[1] = 0;
        s[2] = 0;
        vs[0] = 0;
        vs[1] = 0;
        vs[2] = 0;
        hmax = 0;
        vmax = 0;
        divVmax = 0;
        count_particles = 0;

#ifdef UNEQUALSOFTENINGS
#ifndef ADAPTIVE_GRAVSOFT_FORGAS
        maxsofttype = 7;
        diffsoftflag = 0;
#else
        maxsoft = 0;
#endif
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
                        vs[0] += (Nodes[p].u.d.mass * Extnodes[p].vs[0]);
                        vs[1] += (Nodes[p].u.d.mass * Extnodes[p].vs[1]);
                        vs[2] += (Nodes[p].u.d.mass * Extnodes[p].vs[2]);
#ifdef RADTRANSFER
                        stellar_s[0] += (Nodes[p].stellar_mass * Nodes[p].stellar_s[0]);
                        stellar_s[1] += (Nodes[p].stellar_mass * Nodes[p].stellar_s[1]);
                        stellar_s[2] += (Nodes[p].stellar_mass * Nodes[p].stellar_s[2]);
                        stellar_mass += (Nodes[p].stellar_mass);
#ifdef RT_RAD_PRESSURE
                        bh_s[0] += (Nodes[p].bh_mass * Nodes[p].bh_s[0]);
                        bh_s[1] += (Nodes[p].bh_mass * Nodes[p].bh_s[1]);
                        bh_s[2] += (Nodes[p].bh_mass * Nodes[p].bh_s[2]);
                        bh_mass += (Nodes[p].bh_mass);
#endif
#endif
#ifdef SCALARFIELD
                        mass_dm += (Nodes[p].mass_dm);
                        s_dm[0] += (Nodes[p].mass_dm * Nodes[p].s_dm[0]);
                        s_dm[1] += (Nodes[p].mass_dm * Nodes[p].s_dm[1]);
                        s_dm[2] += (Nodes[p].mass_dm * Nodes[p].s_dm[2]);
                        vs_dm[0] += (Nodes[p].mass_dm * Extnodes[p].vs_dm[0]);
                        vs_dm[1] += (Nodes[p].mass_dm * Extnodes[p].vs_dm[1]);
                        vs_dm[2] += (Nodes[p].mass_dm * Extnodes[p].vs_dm[2]);
#endif
                        if(Nodes[p].u.d.mass > 0)
                        {
                            if(Nodes[p].u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES))
                                count_particles += 2;
                            else
                                count_particles++;
                        }

                        if(Extnodes[p].hmax > hmax)
                            hmax = Extnodes[p].hmax;

                        if(Extnodes[p].vmax > vmax)
                            vmax = Extnodes[p].vmax;

                        if(Extnodes[p].divVmax > divVmax)
                            divVmax = Extnodes[p].divVmax;

#ifdef UNEQUALSOFTENINGS
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
#endif
                    }
                }
                else		/* a particle */
                {
                    count_particles++;

                    pa = &P[p];

                    mass += (pa->Mass);
#ifdef GRAVITY_CENTROID
                    if(P[p].Type == 0)
                    {
                        s[0] += (pa->Mass * SPHP(p).Center[0]);
                        s[1] += (pa->Mass * SPHP(p).Center[1]);
                        s[2] += (pa->Mass * SPHP(p).Center[2]);
                    }
                    else
                    {
                        s[0] += (pa->Mass * pa->Pos[0]);
                        s[1] += (pa->Mass * pa->Pos[1]);
                        s[2] += (pa->Mass * pa->Pos[2]);
                    }
#else
                    s[0] += (pa->Mass * pa->Pos[0]);
                    s[1] += (pa->Mass * pa->Pos[1]);
                    s[2] += (pa->Mass * pa->Pos[2]);
#endif
                    vs[0] += (pa->Mass * pa->Vel[0]);
                    vs[1] += (pa->Mass * pa->Vel[1]);
                    vs[2] += (pa->Mass * pa->Vel[2]);
#ifdef RADTRANSFER
#ifdef EDDINGTON_TENSOR_STARS
                    if(pa->Type == 4 || pa->Type == 2 || pa->Type == 3)
                    {
                        stellar_s[0] += (pa->Mass * pa->Pos[0]);
                        stellar_s[1] += (pa->Mass * pa->Pos[1]);
                        stellar_s[2] += (pa->Mass * pa->Pos[2]);
                        stellar_mass += (pa->Mass);
                    }
#endif
#ifdef EDDINGTON_TENSOR_GAS
                    if(pa->Type == 0)
                    {
                        stellar_s[0] += (pa->Mass * pa->Pos[0]);
                        stellar_s[1] += (pa->Mass * pa->Pos[1]);
                        stellar_s[2] += (pa->Mass * pa->Pos[2]);
                        stellar_mass += (pa->Mass);
                    }
#endif

#if defined(SFR) && defined(EDDINGTON_TENSOR_SFR)
                    if(pa->Type == 0)
                    {
                        if((SPHP(p).d.Density * All.cf.a3inv) >= All.PhysDensThresh)
                        {
                            stellar_s[0] += (pa->Mass * pa->Pos[0]);
                            stellar_s[1] += (pa->Mass * pa->Pos[1]);
                            stellar_s[2] += (pa->Mass * pa->Pos[2]);
                            stellar_mass += (pa->Mass);
                        }
                    }
#endif
#if defined(BLACK_HOLES) && defined(EDDINGTON_TENSOR_BH)
                    if(pa->Type == 5)
                    {
                        stellar_s[0] += (pa->Mass * pa->Pos[0]);
                        stellar_s[1] += (pa->Mass * pa->Pos[1]);
                        stellar_s[2] += (pa->Mass * pa->Pos[2]);
                        stellar_mass += (pa->Mass);

#ifdef RT_RAD_PRESSURE
                        bh_s[0] += (pa->Mass * pa->Pos[0]);
                        bh_s[1] += (pa->Mass * pa->Pos[1]);
                        bh_s[2] += (pa->Mass * pa->Pos[2]);
                        bh_mass += (pa->Mass);
#endif

                    }
#endif
#endif

#ifdef SCALARFIELD
                    if(pa->Type != 0)
                    {
                        mass_dm += (pa->Mass);
                        s_dm[0] += (pa->Mass * pa->Pos[0]);
                        s_dm[1] += (pa->Mass * pa->Pos[1]);
                        s_dm[2] += (pa->Mass * pa->Pos[2]);
                        vs_dm[0] += (pa->Mass * pa->Vel[0]);
                        vs_dm[1] += (pa->Mass * pa->Vel[1]);
                        vs_dm[2] += (pa->Mass * pa->Vel[2]);
                    }
#endif
                    if(pa->Type == 0)
                    {
                        if(P[p].Hsml > hmax)
                            hmax = P[p].Hsml;

                        if(SPHP(p).DivVel > divVmax)
                            divVmax = SPHP(p).DivVel;
                    }

                    for(k = 0; k < 3; k++)
                        if((v = fabs(pa->Vel[k])) > vmax)
                            vmax = v;

#ifdef UNEQUALSOFTENINGS
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
#endif
                }
            }
        }


        if(mass)
        {
            s[0] /= mass;
            s[1] /= mass;
            s[2] /= mass;
            vs[0] /= mass;
            vs[1] /= mass;
            vs[2] /= mass;
        }
        else
        {
            s[0] = Nodes[no].center[0];
            s[1] = Nodes[no].center[1];
            s[2] = Nodes[no].center[2];
            vs[0] = 0;
            vs[1] = 0;
            vs[2] = 0;
        }

#ifdef RADTRANSFER
        if(stellar_mass)
        {
            stellar_s[0] /= stellar_mass;
            stellar_s[1] /= stellar_mass;
            stellar_s[2] /= stellar_mass;
        }
        else
        {
            stellar_s[0] = Nodes[no].center[0];
            stellar_s[1] = Nodes[no].center[1];
            stellar_s[2] = Nodes[no].center[2];
        }
#ifdef RT_RAD_PRESSURE
        if(bh_mass)
        {
            bh_s[0] /= bh_mass;
            bh_s[1] /= bh_mass;
            bh_s[2] /= bh_mass;
        }
        else
        {
            bh_s[0] = Nodes[no].center[0];
            bh_s[1] = Nodes[no].center[1];
            bh_s[2] = Nodes[no].center[2];
        }
#endif
#endif
#ifdef SCALARFIELD
        if(mass_dm)
        {
            s_dm[0] /= mass_dm;
            s_dm[1] /= mass_dm;
            s_dm[2] /= mass_dm;
            vs_dm[0] /= mass_dm;
            vs_dm[1] /= mass_dm;
            vs_dm[2] /= mass_dm;
        }
        else
        {
            s_dm[0] = Nodes[no].center[0];
            s_dm[1] = Nodes[no].center[1];
            s_dm[2] = Nodes[no].center[2];
            vs_dm[0] = 0;
            vs_dm[1] = 0;
            vs_dm[2] = 0;
        }
#endif


        Nodes[no].Ti_current = All.Ti_Current;
        Nodes[no].u.d.mass = mass;
        Nodes[no].u.d.s[0] = s[0];
        Nodes[no].u.d.s[1] = s[1];
        Nodes[no].u.d.s[2] = s[2];
#ifdef RADTRANSFER
        Nodes[no].stellar_s[0] = stellar_s[0];
        Nodes[no].stellar_s[1] = stellar_s[1];
        Nodes[no].stellar_s[2] = stellar_s[2];
        Nodes[no].stellar_mass = stellar_mass;
#ifdef RT_RAD_PRESSURE
        Nodes[no].bh_s[0] = bh_s[0];
        Nodes[no].bh_s[1] = bh_s[1];
        Nodes[no].bh_s[2] = bh_s[2];
        Nodes[no].bh_mass = bh_mass;
#endif
#endif
#ifdef SCALARFIELD
        Nodes[no].s_dm[0] = s_dm[0];
        Nodes[no].s_dm[1] = s_dm[1];
        Nodes[no].s_dm[2] = s_dm[2];
        Nodes[no].mass_dm = mass_dm;
        Extnodes[no].vs_dm[0] = vs_dm[0];
        Extnodes[no].vs_dm[1] = vs_dm[1];
        Extnodes[no].vs_dm[2] = vs_dm[2];
        Extnodes[no].dp_dm[0] = 0;
        Extnodes[no].dp_dm[1] = 0;
        Extnodes[no].dp_dm[2] = 0;
#endif

        Extnodes[no].Ti_lastkicked = All.Ti_Current;
        Extnodes[no].Flag = GlobFlag;
        Extnodes[no].vs[0] = vs[0];
        Extnodes[no].vs[1] = vs[1];
        Extnodes[no].vs[2] = vs[2];
        Extnodes[no].hmax = hmax;
        Extnodes[no].vmax = vmax;
        Extnodes[no].divVmax = divVmax;
        Extnodes[no].dp[0] = 0;
        Extnodes[no].dp[1] = 0;
        Extnodes[no].dp[2] = 0;

        if(count_particles > 1)	/* this flags that the node represents more than one particle */
            multiple_flag = (1 << BITFLAG_MULTIPLEPARTICLES);
        else
            multiple_flag = 0;

        Nodes[no].u.d.bitflags = multiple_flag;

#ifdef UNEQUALSOFTENINGS
#ifndef ADAPTIVE_GRAVSOFT_FORGAS
        Nodes[no].u.d.bitflags |= diffsoftflag + (maxsofttype << BITFLAG_MAX_SOFTENING_TYPE);
#else
        Nodes[no].maxsoft = maxsoft;
#endif
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
        MyFloat vs[3];
        MyFloat mass;
        MyFloat hmax;
        MyFloat vmax;
        MyFloat divVmax;
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
        MyFloat maxsoft;
#endif
#ifdef RADTRANSFER
        MyFloat stellar_mass;
        MyFloat stellar_s[3];
#ifdef RT_RAD_PRESSURE
        MyFloat bh_mass;
        MyFloat bh_s[3];
#endif
#endif
#ifdef SCALARFIELD
        MyFloat s_dm[3];
        MyFloat vs_dm[3];
        MyFloat mass_dm;
#endif
        unsigned int bitflags;
    }
    *DomainMoment;


    DomainMoment = (struct DomainNODE *) mymalloc("DomainMoment", NTopleaves * sizeof(struct DomainNODE));

    for(m = 0; m < MULTIPLEDOMAINS; m++)
        for(i = DomainStartList[ThisTask * MULTIPLEDOMAINS + m];
                i <= DomainEndList[ThisTask * MULTIPLEDOMAINS + m]; i++)
        {
            no = DomainNodeIndex[i];

            /* read out the multipole moments from the local base cells */
            DomainMoment[i].s[0] = Nodes[no].u.d.s[0];
            DomainMoment[i].s[1] = Nodes[no].u.d.s[1];
            DomainMoment[i].s[2] = Nodes[no].u.d.s[2];
            DomainMoment[i].vs[0] = Extnodes[no].vs[0];
            DomainMoment[i].vs[1] = Extnodes[no].vs[1];
            DomainMoment[i].vs[2] = Extnodes[no].vs[2];
            DomainMoment[i].mass = Nodes[no].u.d.mass;
            DomainMoment[i].hmax = Extnodes[no].hmax;
            DomainMoment[i].vmax = Extnodes[no].vmax;
            DomainMoment[i].divVmax = Extnodes[no].divVmax;
            DomainMoment[i].bitflags = Nodes[no].u.d.bitflags;
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
            DomainMoment[i].maxsoft = Nodes[no].maxsoft;
#endif
#ifdef RADTRANSFER
            DomainMoment[i].stellar_s[0] = Nodes[no].stellar_s[0];
            DomainMoment[i].stellar_s[1] = Nodes[no].stellar_s[1];
            DomainMoment[i].stellar_s[2] = Nodes[no].stellar_s[2];
            DomainMoment[i].stellar_mass = Nodes[no].stellar_mass;
#ifdef RT_RAD_PRESSURE
            DomainMoment[i].bh_s[0] = Nodes[no].bh_s[0];
            DomainMoment[i].bh_s[1] = Nodes[no].bh_s[1];
            DomainMoment[i].bh_s[2] = Nodes[no].bh_s[2];
            DomainMoment[i].bh_mass = Nodes[no].bh_mass;
#endif
#endif
#ifdef SCALARFIELD
            DomainMoment[i].s_dm[0] = Nodes[no].s_dm[0];
            DomainMoment[i].s_dm[1] = Nodes[no].s_dm[1];
            DomainMoment[i].s_dm[2] = Nodes[no].s_dm[2];
            DomainMoment[i].mass_dm = Nodes[no].mass_dm;
            DomainMoment[i].vs_dm[0] = Extnodes[no].vs_dm[0];
            DomainMoment[i].vs_dm[1] = Extnodes[no].vs_dm[1];
            DomainMoment[i].vs_dm[2] = Extnodes[no].vs_dm[2];
#endif
        }

    /* share the pseudo-particle data accross CPUs */

    recvcounts = (int *) mymalloc("recvcounts", sizeof(int) * NTask);
    recvoffset = (int *) mymalloc("recvoffset", sizeof(int) * NTask);

    for(m = 0; m < MULTIPLEDOMAINS; m++)
    {
        for(recvTask = 0; recvTask < NTask; recvTask++)
        {
            recvcounts[recvTask] =
                (DomainEndList[recvTask * MULTIPLEDOMAINS + m] - DomainStartList[recvTask * MULTIPLEDOMAINS + m] +
                 1) * sizeof(struct DomainNODE);
            recvoffset[recvTask] = DomainStartList[recvTask * MULTIPLEDOMAINS + m] * sizeof(struct DomainNODE);
        }

        MPI_Allgatherv(&DomainMoment[DomainStartList[ThisTask * MULTIPLEDOMAINS + m]], recvcounts[ThisTask],
                MPI_BYTE, &DomainMoment[0], recvcounts, recvoffset, MPI_BYTE, MPI_COMM_WORLD);
    }

    myfree(recvoffset);
    myfree(recvcounts);


    for(ta = 0; ta < NTask; ta++)
        if(ta != ThisTask)
            for(m = 0; m < MULTIPLEDOMAINS; m++)
                for(i = DomainStartList[ta * MULTIPLEDOMAINS + m]; i <= DomainEndList[ta * MULTIPLEDOMAINS + m]; i++)
                {
                    no = DomainNodeIndex[i];

                    Nodes[no].u.d.s[0] = DomainMoment[i].s[0];
                    Nodes[no].u.d.s[1] = DomainMoment[i].s[1];
                    Nodes[no].u.d.s[2] = DomainMoment[i].s[2];
                    Extnodes[no].vs[0] = DomainMoment[i].vs[0];
                    Extnodes[no].vs[1] = DomainMoment[i].vs[1];
                    Extnodes[no].vs[2] = DomainMoment[i].vs[2];
                    Nodes[no].u.d.mass = DomainMoment[i].mass;
                    Extnodes[no].hmax = DomainMoment[i].hmax;
                    Extnodes[no].vmax = DomainMoment[i].vmax;
                    Extnodes[no].divVmax = DomainMoment[i].divVmax;
                    Nodes[no].u.d.bitflags =
                        (Nodes[no].u.d.bitflags & (~BITFLAG_MASK)) | (DomainMoment[i].bitflags & BITFLAG_MASK);
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
                    Nodes[no].maxsoft = DomainMoment[i].maxsoft;
#endif
#ifdef RADTRANSFER
                    Nodes[no].stellar_s[0] = DomainMoment[i].stellar_s[0];
                    Nodes[no].stellar_s[1] = DomainMoment[i].stellar_s[1];
                    Nodes[no].stellar_s[2] = DomainMoment[i].stellar_s[2];
                    Nodes[no].stellar_mass = DomainMoment[i].stellar_mass;
#ifdef RT_RAD_PRESSURE
                    Nodes[no].bh_s[0] = DomainMoment[i].bh_s[0];
                    Nodes[no].bh_s[1] = DomainMoment[i].bh_s[1];
                    Nodes[no].bh_s[2] = DomainMoment[i].bh_s[2];
                    Nodes[no].bh_mass = DomainMoment[i].bh_mass;
#endif
#endif
#ifdef SCALARFIELD
                    Nodes[no].s_dm[0] = DomainMoment[i].s_dm[0];
                    Nodes[no].s_dm[1] = DomainMoment[i].s_dm[1];
                    Nodes[no].s_dm[2] = DomainMoment[i].s_dm[2];
                    Nodes[no].mass_dm = DomainMoment[i].mass_dm;
                    Extnodes[no].vs_dm[0] = DomainMoment[i].vs_dm[0];
                    Extnodes[no].vs_dm[1] = DomainMoment[i].vs_dm[1];
                    Extnodes[no].vs_dm[2] = DomainMoment[i].vs_dm[2];
#endif
                }

    myfree(DomainMoment);
}



#ifdef GRAVITY_CENTROID
void force_update_node_center_of_mass_recursive(int no, int sib, int father)
{
    int j, jj, p, pp, nextsib, suns[8], count_particles;

    //   int k, multiple_flag; 
    //   MyFloat hmax, vmax, v, divVmax;
    MyFloat s[3], mass;

    //   MyFloat vs[3];
    struct particle_data *pa;


    if(no >= All.MaxPart && no < All.MaxPart + MaxNodes)	/* internal node */
    {
        for(j = 0; j < 8; j++)
            suns[j] = Extnodes[no].suns[j];	/* this "backup" is necessary because the nextnode entry will */

        mass = 0;
        s[0] = 0;
        s[1] = 0;
        s[2] = 0;

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

                force_update_node_center_of_mass_recursive(p, nextsib, no);

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


                    }
                }
                else		/* a particle */
                {
                    count_particles++;

                    pa = &P[p];

                    mass += (pa->Mass);

                    if(P[p].Type == 0)
                    {
                        s[0] += (pa->Mass * SPHP(p).Center[0]);
                        s[1] += (pa->Mass * SPHP(p).Center[1]);
                        s[2] += (pa->Mass * SPHP(p).Center[2]);
                    }
                    else
                    {
                        s[0] += (pa->Mass * pa->Pos[0]);
                        s[1] += (pa->Mass * pa->Pos[1]);
                        s[2] += (pa->Mass * pa->Pos[2]);
                    }
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

        Nodes[no].u.d.s[0] = s[0];
        Nodes[no].u.d.s[1] = s[1];
        Nodes[no].u.d.s[2] = s[2];
    }
}
#endif


/*! This function updates the top-level tree after the multipole moments of
 *  the pseudo-particles have been updated.
 */
void force_treeupdate_pseudos(int no)
{
    int j, p, count_particles, multiple_flag;
    MyFloat hmax, vmax, divVmax;
    MyFloat s[3], vs[3], mass;

#ifdef RADTRANSFER
    MyFloat stellar_mass;
    MyFloat stellar_s[3];

#ifdef RT_RAD_PRESSURE
    MyFloat bh_mass;
    MyFloat bh_s[3];
#endif
#endif
#ifdef SCALARFIELD
    MyFloat s_dm[3], vs_dm[3], mass_dm;
#endif

#ifdef UNEQUALSOFTENINGS
#ifndef ADAPTIVE_GRAVSOFT_FORGAS
    int maxsofttype, diffsoftflag, current_maxsofttype;
#else
    MyFloat maxsoft;
#endif
#endif

#ifdef RADTRANSFER
    stellar_mass = 0;
    stellar_s[0] = 0;
    stellar_s[1] = 0;
    stellar_s[2] = 0;
#ifdef RT_RAD_PRESSURE
    bh_mass = bh_s[0] = bh_s[1] = bh_s[2] = 0.0;
#endif
#endif
#ifdef SCALARFIELD
    mass_dm = 0;
    s_dm[0] = vs_dm[0] = 0;
    s_dm[1] = vs_dm[1] = 0;
    s_dm[2] = vs_dm[2] = 0;
#endif
    mass = 0;
    s[0] = 0;
    s[1] = 0;
    s[2] = 0;
    vs[0] = 0;
    vs[1] = 0;
    vs[2] = 0;
    hmax = 0;
    vmax = 0;
    divVmax = 0;
    count_particles = 0;
#ifdef UNEQUALSOFTENINGS
#ifndef ADAPTIVE_GRAVSOFT_FORGAS
    maxsofttype = 7;
    diffsoftflag = 0;
#else
    maxsoft = 0;
#endif
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
#ifdef RADTRANSFER
            stellar_mass += (Nodes[p].stellar_mass);
            stellar_s[0] += (Nodes[p].stellar_mass * Nodes[p].stellar_s[0]);
            stellar_s[1] += (Nodes[p].stellar_mass * Nodes[p].stellar_s[1]);
            stellar_s[2] += (Nodes[p].stellar_mass * Nodes[p].stellar_s[2]);
#ifdef RT_RAD_PRESSURE
            bh_mass += (Nodes[p].bh_mass);
            bh_s[0] += (Nodes[p].bh_mass * Nodes[p].bh_s[0]);
            bh_s[1] += (Nodes[p].bh_mass * Nodes[p].bh_s[1]);
            bh_s[2] += (Nodes[p].bh_mass * Nodes[p].bh_s[2]);
#endif
#endif
#ifdef SCALARFIELD
            mass_dm += (Nodes[p].mass_dm);
            s_dm[0] += (Nodes[p].mass_dm * Nodes[p].s_dm[0]);
            s_dm[1] += (Nodes[p].mass_dm * Nodes[p].s_dm[1]);
            s_dm[2] += (Nodes[p].mass_dm * Nodes[p].s_dm[2]);
            vs_dm[0] += (Nodes[p].mass_dm * Extnodes[p].vs_dm[0]);
            vs_dm[1] += (Nodes[p].mass_dm * Extnodes[p].vs_dm[1]);
            vs_dm[2] += (Nodes[p].mass_dm * Extnodes[p].vs_dm[2]);
#endif
            vs[0] += (Nodes[p].u.d.mass * Extnodes[p].vs[0]);
            vs[1] += (Nodes[p].u.d.mass * Extnodes[p].vs[1]);
            vs[2] += (Nodes[p].u.d.mass * Extnodes[p].vs[2]);

            if(Extnodes[p].hmax > hmax)
                hmax = Extnodes[p].hmax;
            if(Extnodes[p].vmax > vmax)
                vmax = Extnodes[p].vmax;
            if(Extnodes[p].divVmax > divVmax)
                divVmax = Extnodes[p].divVmax;

            if(Nodes[p].u.d.mass > 0)
            {
                if(Nodes[p].u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES))
                    count_particles += 2;
                else
                    count_particles++;
            }

#ifdef UNEQUALSOFTENINGS
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
#endif
        }
        else
            endrun(6767);		/* may not happen */

        p = Nodes[p].u.d.sibling;
    }

    if(mass)
    {
        s[0] /= mass;
        s[1] /= mass;
        s[2] /= mass;
        vs[0] /= mass;
        vs[1] /= mass;
        vs[2] /= mass;
    }
    else
    {
        s[0] = Nodes[no].center[0];
        s[1] = Nodes[no].center[1];
        s[2] = Nodes[no].center[2];
        vs[0] = 0;
        vs[1] = 0;
        vs[2] = 0;
    }

#ifdef RADTRANSFER
    if(stellar_mass)
    {
        stellar_s[0] /= stellar_mass;
        stellar_s[1] /= stellar_mass;
        stellar_s[2] /= stellar_mass;
    }
    else
    {
        stellar_s[0] = Nodes[no].center[0];
        stellar_s[1] = Nodes[no].center[1];
        stellar_s[2] = Nodes[no].center[2];
    }
#ifdef RT_RAD_PRESSURE
    if(bh_mass)
    {
        bh_s[0] /= bh_mass;
        bh_s[1] /= bh_mass;
        bh_s[2] /= bh_mass;
    }
    else
    {
        bh_s[0] = Nodes[no].center[0];
        bh_s[1] = Nodes[no].center[1];
        bh_s[2] = Nodes[no].center[2];
    }
#endif
#endif
#ifdef SCALARFIELD
    if(mass_dm)
    {
        s_dm[0] /= mass_dm;
        s_dm[1] /= mass_dm;
        s_dm[2] /= mass_dm;
        vs_dm[0] /= mass_dm;
        vs_dm[1] /= mass_dm;
        vs_dm[2] /= mass_dm;
    }
    else
    {
        s_dm[0] = Nodes[no].center[0];
        s_dm[1] = Nodes[no].center[1];
        s_dm[2] = Nodes[no].center[2];
        vs_dm[0] = 0;
        vs_dm[1] = 0;
        vs_dm[2] = 0;
    }
#endif


    Nodes[no].u.d.s[0] = s[0];
    Nodes[no].u.d.s[1] = s[1];
    Nodes[no].u.d.s[2] = s[2];
    Extnodes[no].vs[0] = vs[0];
    Extnodes[no].vs[1] = vs[1];
    Extnodes[no].vs[2] = vs[2];
    Nodes[no].u.d.mass = mass;
#ifdef RADTRANSFER
    Nodes[no].stellar_s[0] = stellar_s[0];
    Nodes[no].stellar_s[1] = stellar_s[1];
    Nodes[no].stellar_s[2] = stellar_s[2];
    Nodes[no].stellar_mass = stellar_mass;
#ifdef RT_RAD_PRESSURE
    Nodes[no].bh_s[0] = bh_s[0];
    Nodes[no].bh_s[1] = bh_s[1];
    Nodes[no].bh_s[2] = bh_s[2];
    Nodes[no].bh_mass = bh_mass;
#endif
#endif
#ifdef SCALARFIELD
    Nodes[no].s_dm[0] = s_dm[0];
    Nodes[no].s_dm[1] = s_dm[1];
    Nodes[no].s_dm[2] = s_dm[2];
    Nodes[no].mass_dm = mass_dm;
    Extnodes[no].vs_dm[0] = vs_dm[0];
    Extnodes[no].vs_dm[1] = vs_dm[1];
    Extnodes[no].vs_dm[2] = vs_dm[2];
#endif

    Extnodes[no].hmax = hmax;
    Extnodes[no].vmax = vmax;
    Extnodes[no].divVmax = divVmax;

    Extnodes[no].Flag = GlobFlag;

    if(count_particles > 1)
        multiple_flag = (1 << BITFLAG_MULTIPLEPARTICLES);
    else
        multiple_flag = 0;

    Nodes[no].u.d.bitflags &= (~BITFLAG_MASK);	/* this clears the bits */

    Nodes[no].u.d.bitflags |= multiple_flag;

#ifdef UNEQUALSOFTENINGS
#ifndef ADAPTIVE_GRAVSOFT_FORGAS
    Nodes[no].u.d.bitflags |= diffsoftflag + (maxsofttype << BITFLAG_MAX_SOFTENING_TYPE);
#else
    Nodes[no].maxsoft = maxsoft;
#endif
#endif
}



/*! This function flags nodes in the top-level tree that are dependent on
 *  local particle data.
 */
void force_flag_localnodes(void)
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

    for(m = 0; m < MULTIPLEDOMAINS; m++)
        for(i = DomainStartList[ThisTask * MULTIPLEDOMAINS + m];
                i <= DomainEndList[ThisTask * MULTIPLEDOMAINS + m]; i++)
        {
            no = DomainNodeIndex[i];

            if(DomainTask[i] != ThisTask)
                endrun(131231231);

            while(no >= 0)
            {
                if(Nodes[no].u.d.bitflags & (1 << BITFLAG_DEPENDS_ON_LOCAL_MASS))
                    break;

                Nodes[no].u.d.bitflags |= (1 << BITFLAG_DEPENDS_ON_LOCAL_MASS);

                no = Nodes[no].u.d.father;
            }
        }
}

static void real_force_drift_node(int no, int time1);
void force_drift_node(int no, int time1) {
    force_drift_node_full(no, time1, 1);
}

int force_drift_node_full(int no, int time1, int blocking) {
    if(time1 == Nodes[no].Ti_current)
        return 0;
#pragma omp atomic
        TotalNodeDrifts ++;

#ifdef OPENMP_USE_SPINLOCK
    int lockstate;
    if (blocking) {
        lockstate = pthread_spin_lock(&Nodes[no].SpinLock);
    } else {
        lockstate = pthread_spin_trylock(&Nodes[no].SpinLock);
    }
    if(0 == lockstate) {
        if(time1 != Nodes[no].Ti_current) {
            real_force_drift_node(no, time1);
#pragma omp flush
        } else {
#pragma omp atomic
            BlockedNodeDrifts ++;
        }
        pthread_spin_unlock(&Nodes[no].SpinLock); 
        return 0;
    } else {
        if(blocking) {
            endrun(99999);
        }
        return -1;
    }
#else
    /* do not use spinlock */
#pragma omp critical (_driftnode_)
    {
        if(time1 != Nodes[no].Ti_current) {
            real_force_drift_node(no, time1);
        }  else {
            BlockedNodeDrifts ++;
        }
    }
    return 0;
#endif
}

static void real_force_drift_node(int no, int time1)
{
    int j;
    double dt_drift, dt_drift_hmax, fac;
    if(time1 == Nodes[no].Ti_current) return;

    if(Nodes[no].u.d.bitflags & (1 << BITFLAG_NODEHASBEENKICKED))
    {
        if(Extnodes[no].Ti_lastkicked != Nodes[no].Ti_current)
        {
            printf("Task=%d Extnodes[no].Ti_lastkicked=%d  Nodes[no].Ti_current=%d\n", 
                    ThisTask, Extnodes[no].Ti_lastkicked, Nodes[no].Ti_current);
            terminate("inconsistency in drift node");
        }

        if(Nodes[no].u.d.mass)
            fac = 1 / Nodes[no].u.d.mass;
        else
            fac = 0;

#ifdef SCALARFIELD
        double fac_dm;

        if(Nodes[no].mass_dm)
            fac_dm = 1 / Nodes[no].mass_dm;
        else
            fac_dm = 0;
#endif

        for(j = 0; j < 3; j++)
        {
            Extnodes[no].vs[j] += fac * (Extnodes[no].dp[j]);
            Extnodes[no].dp[j] = 0;
#ifdef SCALARFIELD
            Extnodes[no].vs_dm[j] += fac_dm * (Extnodes[no].dp_dm[j]);
            Extnodes[no].dp_dm[j] = 0;
#endif

        }
        Nodes[no].u.d.bitflags &= (~(1 << BITFLAG_NODEHASBEENKICKED));
    }

    if(All.ComovingIntegrationOn)
    {
        dt_drift_hmax = get_drift_factor(Nodes[no].Ti_current, time1);
        dt_drift = dt_drift_hmax;
    }
    else
    {
        dt_drift_hmax = (time1 - Nodes[no].Ti_current) * All.Timebase_interval;
        dt_drift = dt_drift_hmax;
    }

    for(j = 0; j < 3; j++)
        Nodes[no].u.d.s[j] += Extnodes[no].vs[j] * dt_drift;
    Nodes[no].len += 2 * Extnodes[no].vmax * dt_drift;

#ifdef SCALARFIELD
    for(j = 0; j < 3; j++)
        Nodes[no].s_dm[j] += Extnodes[no].vs_dm[j] * dt_drift;
#endif

    //  Extnodes[no].hmax *= exp(0.333333333333 * Extnodes[no].divVmax * dt_drift_hmax);

    Nodes[no].Ti_current = time1;
}


void force_kick_node(int i, MyFloat * dv)
{
    int j, no;
    MyFloat dp[3], v, vmax;

#ifdef SCALARFIELD
    MyFloat dp_dm[3];
#endif

#ifdef NEUTRINOS
    if(P[i].Type == 2)
        return;
#endif

    for(j = 0; j < 3; j++)
    {
        dp[j] = P[i].Mass * dv[j];
#ifdef SCALARFIELD
        if(P[i].Type != 0)
            dp_dm[j] = P[i].Mass * dv[j];
        else
            dp_dm[j] = 0;
#endif
    }

    for(j = 0, vmax = 0; j < 3; j++)
        if((v = fabs(P[i].Vel[j])) > vmax)
            vmax = v;

    no = Father[i];

    while(no >= 0)
    {
        real_force_drift_node(no, All.Ti_Current);

        for(j = 0; j < 3; j++)
        {
            Extnodes[no].dp[j] += dp[j];
#ifdef SCALARFIELD
            Extnodes[no].dp_dm[j] += dp_dm[j];
#endif
        }

        if(Extnodes[no].vmax < vmax)
            Extnodes[no].vmax = vmax;

        Nodes[no].u.d.bitflags |= (1 << BITFLAG_NODEHASBEENKICKED);

        Extnodes[no].Ti_lastkicked = All.Ti_Current;

        if(Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL))	/* top-level tree-node reached */
        {
            if(Extnodes[no].Flag != GlobFlag)
            {
                Extnodes[no].Flag = GlobFlag;
                DomainList[DomainNumChanged++] = no;
            }
            break;
        }

        no = Nodes[no].u.d.father;
    }
}

void force_finish_kick_nodes(void)
{
    int i, j, no, ta, totDomainNumChanged;
    int *domainList_all;
    int *counts, *counts_dp, *offset_list, *offset_dp, *offset_vmax;
    MyDouble *domainDp_loc, *domainDp_all;

#ifdef SCALARFIELD
    MyDouble *domainDp_dm_loc, *domainDp_dm_all;
#endif
    MyFloat *domainVmax_loc, *domainVmax_all;

    /* share the momentum-data of the pseudo-particles accross CPUs */

    counts = (int *) mymalloc("counts", sizeof(int) * NTask);
    counts_dp = (int *) mymalloc("counts_dp", sizeof(int) * NTask);
    offset_list = (int *) mymalloc("offset_list", sizeof(int) * NTask);
    offset_dp = (int *) mymalloc("offset_dp", sizeof(int) * NTask);
    offset_vmax = (int *) mymalloc("offset_vmax", sizeof(int) * NTask);

    domainDp_loc = (MyDouble *) mymalloc("domainDp_loc", DomainNumChanged * 3 * sizeof(MyDouble));
#ifdef SCALARFIELD
    domainDp_dm_loc = (MyDouble *) mymalloc("domainDp_dm_loc", DomainNumChanged * 3 * sizeof(MyDouble));
#endif
    domainVmax_loc = (MyFloat *) mymalloc("domainVmax_loc", DomainNumChanged * sizeof(MyFloat));

    for(i = 0; i < DomainNumChanged; i++)
    {
        for(j = 0; j < 3; j++)
        {
            domainDp_loc[i * 3 + j] = Extnodes[DomainList[i]].dp[j];
#ifdef SCALARFIELD
            domainDp_dm_loc[i * 3 + j] = Extnodes[DomainList[i]].dp_dm[j];
#endif
        }
        domainVmax_loc[i] = Extnodes[DomainList[i]].vmax;
    }

    MPI_Allgather(&DomainNumChanged, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);

    for(ta = 0, totDomainNumChanged = 0, offset_list[0] = 0, offset_dp[0] = 0, offset_vmax[0] = 0; ta < NTask;
            ta++)
    {
        totDomainNumChanged += counts[ta];
        if(ta > 0)
        {
            offset_list[ta] = offset_list[ta - 1] + counts[ta - 1];
            offset_dp[ta] = offset_dp[ta - 1] + counts[ta - 1] * 3 * sizeof(MyDouble);
            offset_vmax[ta] = offset_vmax[ta - 1] + counts[ta - 1] * sizeof(MyFloat);
        }
    }

    if(ThisTask == 0)
    {
        printf("I exchange kick momenta for %d top-level nodes out of %d\n", totDomainNumChanged, NTopleaves);
    }

    domainDp_all = (MyDouble *) mymalloc("domainDp_all", totDomainNumChanged * 3 * sizeof(MyDouble));
#ifdef SCALARFIELD
    domainDp_dm_all =
        (MyDouble *) mymalloc("domainDp_dm_all", totDomainNumChanged * 3 * sizeof(MyDouble));
#endif
    domainVmax_all = (MyFloat *) mymalloc("domainVmax_all", totDomainNumChanged * sizeof(MyFloat));

    domainList_all = (int *) mymalloc("domainList_all", totDomainNumChanged * sizeof(int));

    MPI_Allgatherv(DomainList, DomainNumChanged, MPI_INT,
            domainList_all, counts, offset_list, MPI_INT, MPI_COMM_WORLD);

    for(ta = 0; ta < NTask; ta++)
    {
        counts_dp[ta] = counts[ta] * 3 * sizeof(MyDouble);
        counts[ta] *= sizeof(MyFloat);
    }


    MPI_Allgatherv(domainDp_loc, DomainNumChanged * 3 * sizeof(MyDouble), MPI_BYTE,
            domainDp_all, counts_dp, offset_dp, MPI_BYTE, MPI_COMM_WORLD);

#ifdef SCALARFIELD
    MPI_Allgatherv(domainDp_dm_loc, DomainNumChanged * 3 * sizeof(MyDouble), MPI_BYTE,
            domainDp_dm_all, counts_dp, offset_dp, MPI_BYTE, MPI_COMM_WORLD);
#endif

    MPI_Allgatherv(domainVmax_loc, DomainNumChanged * sizeof(MyFloat), MPI_BYTE,
            domainVmax_all, counts, offset_vmax, MPI_BYTE, MPI_COMM_WORLD);


    /* construct momentum kicks in top-level tree */
    for(i = 0; i < totDomainNumChanged; i++)
    {
        no = domainList_all[i];

        if(Nodes[no].u.d.bitflags & (1 << BITFLAG_DEPENDS_ON_LOCAL_MASS))	/* to avoid that the local one is kicked twice */
            no = Nodes[no].u.d.father;

        while(no >= 0)
        {
            real_force_drift_node(no, All.Ti_Current);

            for(j = 0; j < 3; j++)
            {
                Extnodes[no].dp[j] += domainDp_all[3 * i + j];
#ifdef SCALARFIELD
                Extnodes[no].dp_dm[j] += domainDp_dm_all[3 * i + j];
#endif
            }

            if(Extnodes[no].vmax < domainVmax_all[i])
                Extnodes[no].vmax = domainVmax_all[i];

            Nodes[no].u.d.bitflags |= (1 << BITFLAG_NODEHASBEENKICKED);
            Extnodes[no].Ti_lastkicked = All.Ti_Current;

            no = Nodes[no].u.d.father;
        }
    }

    myfree(domainList_all);
    myfree(domainVmax_all);
#ifdef SCALARFIELD
    myfree(domainDp_dm_all);
#endif
    myfree(domainDp_all);
    myfree(domainVmax_loc);
#ifdef SCALARFIELD
    myfree(domainDp_dm_loc);
#endif
    myfree(domainDp_loc);
    myfree(offset_vmax);
    myfree(offset_dp);
    myfree(offset_list);
    myfree(counts_dp);
    myfree(counts);
}


/*! This function updates the hmax-values in tree nodes that hold SPH
 *  particles. These values are needed to find all neighbors in the
 *  hydro-force computation.  Since the Hsml-values are potentially changed
 *  in the SPH-denity computation, force_update_hmax() should be carried
 *  out just before the hydrodynamical SPH forces are computed, i.e. after
 *  density().
 */
void force_update_hmax(void)
{
    int i, no, ta, totDomainNumChanged;
    int *domainList_all;
    int *counts, *offset_list, *offset_hmax;
    MyFloat *domainHmax_loc, *domainHmax_all;

    walltime_measure("/Misc");
    GlobFlag++;

    DomainNumChanged = 0;
    DomainList = (int *) mymalloc("DomainList", NTopleaves * sizeof(int));

    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
        if(P[i].Type == 0)
        {
            no = Father[i];

            while(no >= 0)
            {
                real_force_drift_node(no, All.Ti_Current);

                if(P[i].Hsml > Extnodes[no].hmax || SPHP(i).DivVel > Extnodes[no].divVmax)
                {
                    if(P[i].Hsml > Extnodes[no].hmax)
                        Extnodes[no].hmax = P[i].Hsml;

                    if(SPHP(i).DivVel > Extnodes[no].divVmax)
                        Extnodes[no].divVmax = SPHP(i).DivVel;

                    if(Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL))	/* we reached a top-level node */
                    {
                        if(Extnodes[no].Flag != GlobFlag)
                        {
                            Extnodes[no].Flag = GlobFlag;
                            DomainList[DomainNumChanged++] = no;
                        }
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
    offset_hmax = (int *) mymalloc("offset_hmax", sizeof(int) * NTask);

    domainHmax_loc = (MyFloat *) mymalloc("domainHmax_loc", DomainNumChanged * 2 * sizeof(MyFloat));

    for(i = 0; i < DomainNumChanged; i++)
    {
        domainHmax_loc[2 * i] = Extnodes[DomainList[i]].hmax;
        domainHmax_loc[2 * i + 1] = Extnodes[DomainList[i]].divVmax;
    }


    MPI_Allgather(&DomainNumChanged, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);

    for(ta = 0, totDomainNumChanged = 0, offset_list[0] = 0, offset_hmax[0] = 0; ta < NTask; ta++)
    {
        totDomainNumChanged += counts[ta];
        if(ta > 0)
        {
            offset_list[ta] = offset_list[ta - 1] + counts[ta - 1];
            offset_hmax[ta] = offset_hmax[ta - 1] + counts[ta - 1] * 2 * sizeof(MyFloat);
        }
    }

    if(ThisTask == 0)
        printf("Hmax exchange: %d topleaves out of %d\n", totDomainNumChanged, NTopleaves);

    domainHmax_all = (MyFloat *) mymalloc("domainHmax_all", totDomainNumChanged * 2 * sizeof(MyFloat));
    domainList_all = (int *) mymalloc("domainList_all", totDomainNumChanged * sizeof(int));

    MPI_Allgatherv(DomainList, DomainNumChanged, MPI_INT,
            domainList_all, counts, offset_list, MPI_INT, MPI_COMM_WORLD);

    for(ta = 0; ta < NTask; ta++)
        counts[ta] *= 2 * sizeof(MyFloat);

    MPI_Allgatherv(domainHmax_loc, 2 * DomainNumChanged * sizeof(MyFloat), MPI_BYTE,
            domainHmax_all, counts, offset_hmax, MPI_BYTE, MPI_COMM_WORLD);


    for(i = 0; i < totDomainNumChanged; i++)
    {
        no = domainList_all[i];

        if(Nodes[no].u.d.bitflags & (1 << BITFLAG_DEPENDS_ON_LOCAL_MASS))	/* to avoid that the hmax is updated twice */
            no = Nodes[no].u.d.father;

        while(no >= 0)
        {
            real_force_drift_node(no, All.Ti_Current);

            if(domainHmax_all[2 * i] > Extnodes[no].hmax || domainHmax_all[2 * i + 1] > Extnodes[no].divVmax)
            {
                if(domainHmax_all[2 * i] > Extnodes[no].hmax)
                    Extnodes[no].hmax = domainHmax_all[2 * i];

                if(domainHmax_all[2 * i + 1] > Extnodes[no].divVmax)
                    Extnodes[no].divVmax = domainHmax_all[2 * i + 1];
            }
            else
                break;

            no = Nodes[no].u.d.father;
        }
    }


    myfree(domainList_all);
    myfree(domainHmax_all);
    myfree(domainHmax_loc);
    myfree(offset_hmax);
    myfree(offset_list);
    myfree(counts);
    myfree(DomainList);

    walltime_measure("/Tree/HmaxUpdate");
}





/*! This routine computes the gravitational force for a given local
 *  particle, or for a particle in the communication buffer. Depending on
 *  the value of TypeOfOpeningCriterion, either the geometrical BH
 *  cell-opening criterion, or the `relative' opening criterion is used.
 */
int force_treeevaluate(int target, int mode, 
        struct gravitydata_in  * input,
        struct gravitydata_out  * output,
        LocalEvaluator * lv, void * unused)
{

    struct NODE *nop = 0;
    int no, ptype, listindex = 0;
    int nnodesinlist = 0, ninteractions = 0; 
    double r2, dx, dy, dz, mass, r, fac, u, h, h_inv, h3_inv;
    double pos_x, pos_y, pos_z, aold;
    MyDouble acc_x, acc_y, acc_z;

#ifdef SCALARFIELD
    double dx_dm = 0, dy_dm = 0, dz_dm = 0, mass_dm = 0;
#endif

    double wp;
    MyDouble pot;

    pot = 0.0;

#ifdef ADAPTIVE_GRAVSOFT_FORGAS
    double soft = 0;
#endif

    acc_x = 0;
    acc_y = 0;
    acc_z = 0;

    no = input->NodeList[0];
    listindex ++;
    no = Nodes[no].u.d.nextnode;	/* open it */

    pos_x = input->Pos[0];
    pos_y = input->Pos[1];
    pos_z = input->Pos[2];
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


#ifndef UNEQUALSOFTENINGS
    h = All.ForceSoftening[ptype];
    h_inv = 1.0 / h;
    h3_inv = h_inv * h_inv * h_inv;
#endif

    while(no >= 0)
    {
        while(no >= 0)
        {
            if(no < All.MaxPart)	/* single particle */
            {
                /* the index of the node is the index of the particle */
                /* observe the sign */

                drift_particle(no, All.Ti_Current);

#ifdef GRAVITY_CENTROID
                if(P[no].Type == 0)
                {
                    dx = SPHP(no).Center[0] - pos_x;
                    dy = SPHP(no).Center[1] - pos_y;
                    dz = SPHP(no).Center[2] - pos_z;

                }
                else
                {
                    dx = P[no].Pos[0] - pos_x;
                    dy = P[no].Pos[1] - pos_y;
                    dz = P[no].Pos[2] - pos_z;
                }
#else
                dx = P[no].Pos[0] - pos_x;
                dy = P[no].Pos[1] - pos_y;
                dz = P[no].Pos[2] - pos_z;
#endif

                mass = P[no].Mass;
#ifdef SCALARFIELD
                if(ptype != 0)	/* we have a dark matter particle as target */
                {
                    if(P[no].Type != 0)
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
            }
            else
            {
                if(no >= All.MaxPart + MaxNodes)	/* pseudo particle */
                {
                    if(mode == 0)
                    {
                        if(-1 == ev_export_particle(lv, target, no))
                            return -1;
                    }
                    no = Nextnode[no - MaxNodes];
                    continue;
                }

                nop = &Nodes[no];

                if(mode == 1)
                {
                    if(nop->u.d.bitflags & (1 << BITFLAG_TOPLEVEL))	/* we reached a top-level node again, which means that we are done with the branch */
                    {
                        no = -1;
                        continue;
                    }
                }

                mass = nop->u.d.mass;

                if(!(nop->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
                {
                    /* open cell */
                    if(mass)
                    {
                        no = nop->u.d.nextnode;
                        continue;
                    }
                }

                force_drift_node(no, All.Ti_Current);

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
            }

#if defined(PERIODIC) && !defined(GRAVITY_NOT_PERIODIC)
            dx = NEAREST(dx);
            dy = NEAREST(dy);
            dz = NEAREST(dz);
#endif
            r2 = dx * dx + dy * dy + dz * dz;


            if(no < All.MaxPart)
            {
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
            else			/* we have an  internal node. Need to check opening criterion */
            {

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

#ifdef UNEQUALSOFTENINGS
#ifndef ADAPTIVE_GRAVSOFT_FORGAS
                h = All.ForceSoftening[ptype];
                if(h < All.ForceSoftening[extract_max_softening_type(nop->u.d.bitflags)])
                {
                    h = All.ForceSoftening[extract_max_softening_type(nop->u.d.bitflags)];
                    if(r2 < h * h)
                    {
                        if(maskout_different_softening_flag(nop->u.d.bitflags))	/* signals that there are particles of different softening in the node */
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
                pot += (-mass / r);
            }
            else
            {
#ifdef UNEQUALSOFTENINGS
                h_inv = 1.0 / h;
                h3_inv = h_inv * h_inv * h_inv;
#endif
                u = r * h_inv;
                if(u < 0.5)
                    fac = mass * h3_inv * (10.666666666667 + u * u * (32.0 * u - 38.4));
                else
                    fac =
                        mass * h3_inv * (21.333333333333 - 48.0 * u +
                                38.4 * u * u - 10.666666666667 * u * u * u - 0.066666666667 / (u * u * u));

                /* now the potential */
                if(u < 0.5)
                    wp = -2.8 + u * u * (5.333333333333 + u * u * (6.4 * u - 9.6));
                else
                    wp =
                        -3.2 + 0.066666666667 / u + u * u * (10.666666666667 +
                                u * (-16.0 + u * (9.6 - 2.133333333333 * u)));
                pot += (mass * h_inv * wp);
            }

            acc_x += (dx * fac);
            acc_y += (dy * fac);
            acc_z += (dz * fac);


            if(mass > 0)
                ninteractions++;

#ifdef SCALARFIELD
            if(ptype != 0)	/* we have a dark matter particle as target */
            {
#if defined(PERIODIC) && !defined(GRAVITY_NOT_PERIODIC)
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

                acc_x += (dx_dm * fac);
                acc_y += (dy_dm * fac);
                acc_z += (dz_dm * fac);
            }
#endif


        }
        if(listindex < NODELISTLENGTH)
        {
            no = input->NodeList[listindex];
            if(no >= 0) {
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

    /* store result at the proper place */
    lv->Ninteractions = ninteractions;
    lv->Nnodesinlist = nnodesinlist;
    return ninteractions;
}

#ifdef PETAPM
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
int force_treeev_shortrange(int target, int mode, 
        struct gravitydata_in * input, 
        struct gravitydata_out * output, 
        LocalEvaluator * lv, void * unused)
{
    struct NODE *nop = 0;
    int no, ptype, tabindex, listindex = 0;
    int nnodesinlist = 0, ninteractions = 0;
    double r2, dx, dy, dz, mass, r, fac, u, h, h_inv, h3_inv;
    double pos_x, pos_y, pos_z, aold;
    double eff_dist;
    double rcut, asmth, asmthfac, rcut2, dist;
    MyDouble acc_x, acc_y, acc_z;

#ifdef SCALARFIELD
    double dx_dm = 0, dy_dm = 0, dz_dm = 0, mass_dm = 0;
#endif
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

    no = input->NodeList[0];
    listindex ++;
    no = Nodes[no].u.d.nextnode;	/* open it */

    pos_x = input->Pos[0];
    pos_y = input->Pos[1];
    pos_z = input->Pos[2];
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
#ifdef PLACEHIGHRESREGION
    if(pmforce_is_particle_high_res(ptype, input->Pos))
    {
        rcut = All.Rcut[1];
        asmth = All.Asmth[1];
    }
#endif

    rcut2 = rcut * rcut;

    asmthfac = 0.5 / asmth * (NTAB / 3.0);

#ifndef UNEQUALSOFTENINGS
    h = All.ForceSoftening[ptype];
    h_inv = 1.0 / h;
    h3_inv = h_inv * h_inv * h_inv;
#endif

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
                if(no >= All.MaxPart + MaxNodes)	/* pseudo particle */
                {
                    if(mode == 0)
                    {
                        if(-1 == ev_export_particle(lv, target, no))
                            return -1;
                    }
                    no = Nextnode[no - MaxNodes];
                    continue;
                }

                nop = &Nodes[no];

                if(mode == 1)
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

#ifdef UNEQUALSOFTENINGS
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
#ifdef UNEQUALSOFTENINGS
                h_inv = 1.0 / h;
                h3_inv = h_inv * h_inv * h_inv;
#endif
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
                    acc_x += (dx_dm * fac);
                    acc_y += (dy_dm * fac);
                    acc_z += (dz_dm * fac);
                }
            }
#endif
        }

        if(listindex < NODELISTLENGTH)
        {
            no = input->NodeList[listindex];
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

#endif


/*! This function allocates the memory used for storage of the tree and of
 *  auxiliary arrays needed for tree-walk and link-lists.  Usually,
 *  maxnodes approximately equal to 0.7*maxpart is sufficient to store the
 *  tree for up to maxpart particles.
 */
void force_treeallocate(int maxnodes, int maxpart)
{
    int i;
    size_t bytes;
    double allbytes = 0, allbytes_topleaves = 0;
    double u;

    tree_allocated_flag = 1;
    DomainNodeIndex = (int *) mymalloc("DomainNodeIndex", bytes = NTopleaves * sizeof(int));
    allbytes_topleaves += bytes;
    MaxNodes = maxnodes;
    if(!(Nodes_base = (struct NODE *) mymalloc("Nodes_base", bytes = (MaxNodes + 1) * sizeof(struct NODE))))
    {
        printf("failed to allocate memory for %d tree-nodes (%g MB).\n", MaxNodes, bytes / (1024.0 * 1024.0));
        endrun(3);
    }
    allbytes += bytes;
    if(!
            (Extnodes_base =
             (struct extNODE *) mymalloc("Extnodes_base", bytes = (MaxNodes + 1) * sizeof(struct extNODE))))
    {
        printf("failed to allocate memory for %d tree-extnodes (%g MB).\n",
                MaxNodes, bytes / (1024.0 * 1024.0));
        endrun(3);
    }
#ifdef OPENMP_USE_SPINLOCK
    {
        int i;
        for (i = 0; i < MaxNodes + 1; i ++) {
            /* Maybe we can directly set these guys to one
             *
             * at least with the glibc spinlock implementation.
             * */
            pthread_spin_init(&Nodes_base[i].SpinLock, 0);
        }
    }
#endif
    allbytes += bytes;
    Nodes = Nodes_base - All.MaxPart;
    Extnodes = Extnodes_base - All.MaxPart;
    if(!(Nextnode = (int *) mymalloc("Nextnode", bytes = (maxpart + NTopnodes) * sizeof(int))))
    {
        printf("Failed to allocate %d spaces for 'Nextnode' array (%g MB)\n",
                maxpart + NTopnodes, bytes / (1024.0 * 1024.0));
        exit(0);
    }
    allbytes += bytes;
    if(!(Father = (int *) mymalloc("Father", bytes = (maxpart) * sizeof(int))))
    {
        printf("Failed to allocate %d spaces for 'Father' array (%g MB)\n", maxpart, bytes / (1024.0 * 1024.0));
        exit(0);
    }
    allbytes += bytes;
    if(first_flag == 0)
    {
        first_flag = 1;
        if(ThisTask == 0)
            printf
                ("\nAllocated %g MByte for BH-tree, and %g Mbyte for top-leaves.  (presently allocted %g MB)\n\n",
                 allbytes / (1024.0 * 1024.0), allbytes_topleaves / (1024.0 * 1024.0),
                 AllocatedBytes / (1024.0 * 1024.0));
        tabfac = NTAB / 3.0;
        for(i = 0; i < NTAB; i++)
        {
            u = 3.0 / NTAB * (i + 0.5);
            shortrange_table[i] = erfc(u) + 2.0 * u / sqrt(M_PI) * exp(-u * u);
            shortrange_table_potential[i] = erfc(u);
            shortrange_table_tidal[i] = 4.0 * u * u * u / sqrt(M_PI) * exp(-u * u);
        }
    }
}


/*! This function frees the memory allocated for the tree, i.e. it frees
 *  the space allocated by the function force_treeallocate().
 */
void force_treefree(void)
{
    if(tree_allocated_flag)
    {
        myfree(Father);
        myfree(Nextnode);
        myfree(Extnodes_base);
        myfree(Nodes_base);
        myfree(DomainNodeIndex);
        tree_allocated_flag = 0;
    }
}




/*! This function does the force computation with direct summation for the
 *  specified particle in the communication buffer. This can be useful for
 *  debugging purposes, in particular for explicit checks of the force
 *  accuracy.
 */
#ifdef FORCETEST
int force_treeev_direct(int target, int mode)
{
    double epsilon;
    double h, h_inv, dx, dy, dz, r, r2, u, r_inv, fac;
    int i, ptype;
    double pos_x, pos_y, pos_z;
    double acc_x, acc_y, acc_z;

#ifdef PERIODIC
    double fcorr[3];
#endif
#ifdef PERIODIC
    double boxsize, boxhalf;

    boxsize = All.BoxSize;
    boxhalf = 0.5 * All.BoxSize;
#endif
    acc_x = 0;
    acc_y = 0;
    acc_z = 0;
    if(mode == 0)
    {
        pos_x = P[target].Pos[0];
        pos_y = P[target].Pos[1];
        pos_z = P[target].Pos[2];
        ptype = P[target].Type;
    }
    else
    {
        pos_x = GravDataGet[target].Pos[0];
        pos_y = GravDataGet[target].Pos[1];
        pos_z = GravDataGet[target].Pos[2];
#if defined(UNEQUALSOFTENINGS) || defined(SCALARFIELD)
        ptype = GravDataGet[target].Type;
#else
        ptype = P[0].Type;
#endif
    }


    for(i = 0; i < NumPart; i++)
    {
        epsilon = DMAX(All.ForceSoftening[P[i].Type], All.ForceSoftening[ptype]);
        h = epsilon;
        h_inv = 1 / h;
        dx = P[i].Pos[0] - pos_x;
        dy = P[i].Pos[1] - pos_y;
        dz = P[i].Pos[2] - pos_z;
#ifdef PERIODIC
        while(dx > boxhalf)
            dx -= boxsize;
        while(dy > boxhalf)
            dy -= boxsize;
        while(dz > boxhalf)
            dz -= boxsize;
        while(dx < -boxhalf)
            dx += boxsize;
        while(dy < -boxhalf)
            dy += boxsize;
        while(dz < -boxhalf)
            dz += boxsize;
#endif
        r2 = dx * dx + dy * dy + dz * dz;
        r = sqrt(r2);
        u = r * h_inv;
        if(u >= 1)
        {
            r_inv = 1 / r;
            fac = P[i].Mass * r_inv * r_inv * r_inv;
        }
        else
        {
            if(u < 0.5)
                fac = P[i].Mass * h_inv * h_inv * h_inv * (10.666666666667 + u * u * (32.0 * u - 38.4));
            else
                fac =
                    P[i].Mass * h_inv * h_inv * h_inv * (21.333333333333 -
                            48.0 * u + 38.4 * u * u -
                            10.666666666667 * u * u *
                            u - 0.066666666667 / (u * u * u));
        }

        acc_x += dx * fac;
        acc_y += dy * fac;
        acc_z += dz * fac;
#ifdef PERIODIC
        if(u > 1.0e-5)
        {
            ewald_corr(dx, dy, dz, fcorr);
            acc_x += P[i].Mass * fcorr[0];
            acc_y += P[i].Mass * fcorr[1];
            acc_z += P[i].Mass * fcorr[2];
        }
#endif
    }


    if(mode == 0)
    {
        P[target].GravAccelDirect[0] = acc_x;
        P[target].GravAccelDirect[1] = acc_y;
        P[target].GravAccelDirect[2] = acc_z;
    }
    else
    {
        GravDataResult[target].Acc[0] = acc_x;
        GravDataResult[target].Acc[1] = acc_y;
        GravDataResult[target].Acc[2] = acc_z;
    }


    return NumPart;
}
#endif


/*! This function dumps some of the basic particle data to a file. In case
 *  the tree construction fails, it is called just before the run
 *  terminates with an error message. Examination of the generated file may
 *  then give clues to what caused the problem.
 */
void dump_particles(void)
{
    FILE *fd;
    char buffer[200];
    int i;

    sprintf(buffer, "particles%d.dat", ThisTask);
    fd = fopen(buffer, "w");
    my_fwrite(&NumPart, 1, sizeof(int), fd);
    for(i = 0; i < NumPart; i++)
        my_fwrite(&P[i].Pos[0], 3, sizeof(MyFloat), fd);
    for(i = 0; i < NumPart; i++)
        my_fwrite(&P[i].Vel[0], 3, sizeof(MyFloat), fd);
    for(i = 0; i < NumPart; i++)
        my_fwrite(&P[i].ID, 1, sizeof(int), fd);
    fclose(fd);
}



