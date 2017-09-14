#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "allvars.h"
#include "proto.h"
#include "domain.h"
#include "openmpsort.h"
#include "forcetree.h"
#include "mymalloc.h"
#include "endrun.h"
#include "system.h"

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

/*The node index is an integer with unusual properties:
 * no = 0..All.MaxPart (firstnode in internal functions) corresponds to a particle.
 * no = All.MaxPart..All.MaxPart + MaxNodes (firstnode..lastnode) corresponds to actual tree nodes,
 * and is the only memory allocated in Nodes_base.
 * no > All.MaxPart + MaxNodes (lastnode) means a pseudo particle on another processor*/
struct NODE *Nodes_base,	/*!< points to the actual memory allocated for the nodes */
 *Nodes;			/*!< this is a pointer used to access the nodes which is shifted such that Nodes[All.MaxPart]
				   gives the first allocated node */


int MaxNodes;                  /*!< maximum allowed number of internal nodes */


int *Nextnode;			/*!< gives next node in tree walk  (nodes array) */
int *Father;			/*!< gives parent node in tree (nodes array) */

static int tree_allocated_flag = 0;

static int force_tree_build(int npart);

static int
force_tree_build_single(const int firstnode, const int lastnode, const int npart);

/*Next three are not static as tested.*/
int
force_tree_create_nodes(const int firstnode, const int lastnode, const int npart);

size_t
force_treeallocate(int maxnodes, int maxpart, int first_node_offset);

int
force_update_node_recursive(int no, int sib, int tail, const int firstnode, const int lastnode);

static void
force_treeupdate_pseudos(int no, int firstnode, int lastnode);

static void
force_create_node_for_topnode(int no, int topnode, int bits, int x, int y, int z, int *nextfree, const int lastnode);

static void
force_exchange_pseudodata(void);

static void
force_insert_pseudo_particles(int firstpseudo);

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
    int Numnodestree;
    int flag;
    int maxnodes;

    do
    {
        maxnodes = All.TreeAllocFactor * All.MaxPart + NTopNodes;
        /* construct tree if needed */
        /* the tree is used in grav dens, hydro, bh and sfr */
        size_t allbytes = force_treeallocate(maxnodes, All.MaxPart, All.MaxPart);

        message(0, "Allocated %g MByte for BH-tree, (presently allocated %g MB)\n",
             allbytes / (1024.0 * 1024.0),
             AllocatedBytes / (1024.0 * 1024.0));

        Numnodestree = force_tree_build_single(All.MaxPart, All.MaxPart + maxnodes, npart);
        if(Numnodestree < 0)
            message(1, "Not enough tree nodes (%d) for %d particles.\n", maxnodes, npart);

        MPI_Allreduce(&Numnodestree, &flag, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if(flag == -1)
        {
            force_tree_free();

            message(0, "TreeAllocFactor from %g to %g\n", All.TreeAllocFactor, All.TreeAllocFactor*1.15);

            All.TreeAllocFactor *= 1.15;

            if(All.TreeAllocFactor > 5.0)
            {
                message(0, "An excessively large number of tree nodes were required, stopping with particle dump.\n");
                savepositions(999999, 0);
                endrun(0, "Too many tree nodes, snapshot saved.");
            }
        }
    }
    while(flag == -1);

    force_exchange_pseudodata();

    force_treeupdate_pseudos(All.MaxPart, All.MaxPart, All.MaxPart + maxnodes);

    return Numnodestree;
}

/* Get the subnode for a given particle and parent node.
 * This splits a parent node into 8 subregions depending on the particle position.
 * node is the parent node to split, p_i is the index of the particle we
 * are currently inserting, and shift denotes the level of the
 * tree we are currently at.
 * Returns a value between 0 and 7. If particles are very close,
 * the tree subnode is randomised.
 *
 * shift is the level of the subnode to be returned, not the level of the parent node.
 * */
int get_subnode(const struct NODE * node, const int p_i)
{
    int subnode = 0;
    int k;
    for(k=0; k<3; k++)
        subnode |= (P[p_i].Pos[k] > node->center[k]) << k;
    return subnode;
}

/*Check whether a particle is inside the volume covered by a node*/
static inline int inside_node(const struct NODE * node, const int p_i)
{
    int k;
    for(k=0; k<3; k++) {
        double pdiff = 2*(P[p_i].Pos[k] - node->center[k]);
        pdiff = pdiff > 0 ? pdiff : -1*pdiff;
        if(pdiff > node->len)
            return 0;
    }
    return 1;
}

/*Initialise an internal node at nfreep. The parent is assumed to be locked, and
 * we have assured that nothing else will change nfreep while we are here.*/
static void init_internal_node(struct NODE *nfreep, struct NODE *parent, int subnode)
{
    int j;
    const MyFloat lenhalf = 0.25 * parent->len;

    nfreep->len = 0.5 * parent->len;
    nfreep->f.TopLevel = 0;
    nfreep->f.InternalTopLevel = 0;

    for(j = 0; j < 3; j++) {
        /* Detect which quadrant we are in by testing the bits of subnode:
         * if (subnode & [1,2,4]) is true we add lenhalf, otherwise subtract lenhalf*/
        const int sign = (subnode & (1 << j)) ? 1 : -1;
        nfreep->center[j] = parent->center[j] + sign*lenhalf;
    }
    for(j = 0; j < 8; j++)
        nfreep->u.suns[j] = -1;
}

/* Size of the free Node thread cache.
 * 100 was found to be optimal for an Intel skylake with 4 threads.*/
#define NODECACHE_SIZE 100

/*Get a pointer to memory for a free node, from our node cache.
 * If there is no memory left, return NULL.*/
int get_freenode(int * nnext, int *nnext_thread, int *nrem_thread)
{
    /*Get memory for an extra node from our cache.*/
    if(*nrem_thread == 0) {
        *nnext_thread = atomic_fetch_and_add(nnext, NODECACHE_SIZE);
        *nrem_thread = NODECACHE_SIZE;
    }
    const int ninsert = (*nnext_thread)++;
    (*nrem_thread)--;
    return ninsert;
}

/* Parent is a node where the subnode we want to add a particle to is filled.
 * We add a new internal node at this subnode and try to add both the old and new particles to it.
 * Parent is assumed to be locked.*/
int insert_internal_node(int parent, int subnode, int p_child, int p_toplace, const int lastnode, int *nnext, int *nnext_thread, int *nrem_thread, double minlen)
{
    /*Get memory for an extra node from our cache.*/
    int ninsert = get_freenode(nnext, nnext_thread, nrem_thread);
    /*If we already have too many nodes, exit loop.*/
    if(*nnext_thread >= lastnode)
        return 1;

    struct NODE *nfreep = &Nodes[ninsert];
    struct NODE *nprnt = &Nodes[parent];
    /* We create a new internal node with empty subnodes at the end of the array, and
     * use it to replace the particle in the parent's subnode.*/
    init_internal_node(nfreep, nprnt, subnode);
    /*Set father of new node*/
    nfreep->father = parent;

    /* The new internal node replaced a particle in the parent.
     * Re-add that particle to the child.*/
    const int child_subnode = get_subnode(nfreep, p_child);

    int new_subnode = get_subnode(nfreep, p_toplace);

    /* If the node is very small, just add the particle to the next subnode.
     * This ensures that we can always construct the tree, and only happens far
     * below the softening length. This happens occasionally, but if it is
     * frequent increase the force softening.
     */
    if(nfreep->len < minlen && new_subnode == child_subnode) {
        if(child_subnode < 7)
            new_subnode = child_subnode + 1;
        else
            new_subnode = child_subnode - 1;
        /* Warn about this happening.*/
        message(1,"Close particles: %d @ [%g, %g, %g] and %d @ [%g, %g, %g]. "
                "Attached to node %d, subnode %d, at [%g, %g, %g] (len %g).\n",
                p_toplace, P[p_toplace].Pos[0], P[p_toplace].Pos[1], P[p_toplace].Pos[2],
                p_child, P[p_child].Pos[0], P[p_child].Pos[1], P[p_child].Pos[2],
                nfreep, child_subnode, nprnt->center[0], nprnt->center[1], nprnt->center[2], nprnt->len);
    }

    int ret = 0;
    /*If these two are different, great! Attach both particles to this new node*/
    if(child_subnode != new_subnode) {
        Father[p_child] = ninsert;
        Father[p_toplace] = ninsert;
        nfreep->u.suns[child_subnode] = p_child;
        nfreep->u.suns[new_subnode] = p_toplace;
    }
    /*Otherwise recurse and create a new node*/
    else {
        ret = insert_internal_node(ninsert, new_subnode, p_child, p_toplace, lastnode, nnext, nnext_thread, nrem_thread, minlen);
    }

    /* Mark this node in the parent: this goes last
     * so that we don't access the child before it is constructed.*/
    #pragma omp atomic write
    nprnt->u.suns[subnode] = ninsert;
    return ret;
}

/*! Does initial creation of the nodes for the gravitational oct-tree.
 **/
int force_tree_create_nodes(const int firstnode, const int lastnode, const int npart)
{
    int i;
    int nnext = firstnode;		/* index of first free node */

    /* create an empty root node  */
    {
        struct NODE *nfreep = &Nodes[nnext];	/* select first node */

        nfreep->len = All.BoxSize*1.001;
        for(i = 0; i < 3; i++)
            nfreep->center[i] = All.BoxSize/2.;
        for(i = 0; i < 8; i++)
            nfreep->u.suns[i] = -1;
        nfreep->father = -1;
        nfreep->f.TopLevel = 1;
        nfreep->f.InternalTopLevel = 0;
        nnext++;
        /* create a set of empty nodes corresponding to the top-level domain
         * grid. We need to generate these nodes first to make sure that we have a
         * complete top-level tree which allows the easy insertion of the
         * pseudo-particles in the right place */
        force_create_node_for_topnode(firstnode, 0, 1, 0, 0, 0, &nnext, lastnode);
    }

    /* This implements a small thread-local free Node cache.
     * The cache ensures that Nodes from the same (or close) particles
     * are created close to each other on the Node list and thus
     * helps cache locality. In my tests without this list the
     * reduction in cache performance destroyed the benefit of
     * parallelizing this loop!*/
    int nnext_thread=nnext;
    int nrem_thread=0;
    /* Stores the last-seen node on this thread.
     * Since most particles are close to each other, this should save a number of tree walks.*/
    int this_acc = firstnode;
    /* now we insert all particles */
#ifdef OPENMP_USE_SPINLOCK
    /*Initialise some spinlocks off*/
    pthread_spinlock_t * SpinLocks = mymalloc("NodeSpinlocks", (lastnode - firstnode)*sizeof(pthread_spinlock_t));
    for(i=0; i < lastnode - firstnode; i++) {
        pthread_spin_init(&SpinLocks[i],PTHREAD_PROCESS_PRIVATE);
    }
    #pragma omp parallel for firstprivate(nnext_thread, nrem_thread, this_acc)
#endif
    for(i = 0; i < npart; i++)
    {
        /*Can't break from openmp for*/
        if(nnext_thread >= lastnode-1)
            continue;

        /*First find the Node for the TopLeaf */

        int this;
        if(inside_node(&Nodes[this_acc], i)) {
            this = this_acc;
        } else {
            const int topleaf = domain_get_topleaf(P[i].Key);
            this = TopLeaves[topleaf].treenode;
        }
        int child, subnode;

        /*Walk the main tree until we get something that isn't an internal node.*/
        do
        {
            /*We will always start with an internal node: find the desired subnode.*/
            subnode = get_subnode(&Nodes[this], i);

            /*No lock needed: if we have an internal node here it will be stable*/
            #pragma omp atomic read
            child = Nodes[this].u.suns[subnode];

            if(child > lastnode)
                endrun(1,"Corruption in tree build: N[%d].[%d] = %d > lastnode (%d)\n",this, subnode, child, lastnode);
            /* If we found an internal node keep walking*/
            else if(child >= firstnode) {
                this = child;
            }
        }
        while(child >= firstnode);

#ifdef OPENMP_USE_SPINLOCK
        /*Now lock this node.*/
        pthread_spin_lock(&SpinLocks[this-firstnode]);
#endif

        /*Check nothing changed when we took the lock*/
        #pragma omp atomic read
        child = Nodes[this].u.suns[subnode];
        /*If it did, walk again*/
        while(child >= firstnode)
        {
#ifdef OPENMP_USE_SPINLOCK
            /*Move the lock to the child*/
            pthread_spin_lock(&SpinLocks[child-firstnode]);
            pthread_spin_unlock(&SpinLocks[this-firstnode]);
#endif
            this = child;
            /*New subnode*/
            subnode = get_subnode(&Nodes[this], i);
            #pragma omp atomic read
            child = Nodes[this].u.suns[subnode];
        }
        /*Update last-used cache*/
        this_acc = this;
        /*Now we have something that isn't an internal node, and we have a lock on the parent,
         * so we know it won't change. We can place the particle!*/
        /* The easy case: we found an empty slot on this node,
         * so attach this particle.*/
        if(child < 0) {
            Father[i] = this;
            #pragma omp atomic write
            Nodes[this].u.suns[subnode] = i;
        }
        /*The slot we wanted to fill up contains a particle.
         * We must insert a new internal node here.*/
        else {
            /*When we get here we have reached a leaf of the tree. We have an internal node in this,
             * with a full (positive) subnode in subnode, containing a real particle.
             * We split this node (making it an internal node) and try to add our particle to the new split node.*/
            const double minlen = 1.0e-3 * All.ForceSoftening[1];
            insert_internal_node(this, subnode, child, i, lastnode, &nnext, &nnext_thread, &nrem_thread, minlen);
        }
#ifdef OPENMP_USE_SPINLOCK
        /*Unlock the parent*/
        pthread_spin_unlock(&SpinLocks[this - firstnode]);
#endif
    }

#ifdef OPENMP_USE_SPINLOCK
    for(i=0; i < lastnode - firstnode; i++)
            pthread_spin_destroy(&SpinLocks[i]);
    /*Avoid a warning about discarding volatile*/
    int * ss = (int *) SpinLocks;
    myfree(ss);
#endif
    return nnext - firstnode;
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
static int
force_tree_build_single(const int firstnode, const int lastnode, const int npart)
{
    int nnext = force_tree_create_nodes(firstnode, lastnode, npart);
    if(nnext >= lastnode - firstnode)
    {
        return -1;
    }

    /* insert the pseudo particles that represent the mass distribution of other domains */
    force_insert_pseudo_particles(lastnode);

    /* now compute the multipole moments recursively */
    int tail = force_update_node_recursive(firstnode, -1, -1, firstnode, lastnode);

    force_set_next_node(tail, -1, firstnode, lastnode);

    return nnext;
}



/*! This function recursively creates a set of empty tree nodes which
 *  corresponds to the top-level tree for the domain grid. This is done to
 *  ensure that this top-level tree is always "complete" so that we can easily
 *  associate the pseudo-particles of other CPUs with tree-nodes at a given
 *  level in the tree, even when the particle population is so sparse that
 *  some of these nodes are actually empty.
 */
void force_create_node_for_topnode(int no, int topnode, int bits, int x, int y, int z, int *nextfree, const int lastnode)
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
                    /*We are an internal top level node as we now have a child top level.*/
                    Nodes[no].f.InternalTopLevel = 1;

                    lenhalf = 0.25 * Nodes[no].len;
                    Nodes[*nextfree].len = 0.5 * Nodes[no].len;
                    Nodes[*nextfree].center[0] = Nodes[no].center[0] + (2 * i - 1) * lenhalf;
                    Nodes[*nextfree].center[1] = Nodes[no].center[1] + (2 * j - 1) * lenhalf;
                    Nodes[*nextfree].center[2] = Nodes[no].center[2] + (2 * k - 1) * lenhalf;
                    Nodes[*nextfree].father = no;
                    /*All nodes here are top level nodes*/
                    Nodes[*nextfree].f.TopLevel = 1;
                    Nodes[*nextfree].f.InternalTopLevel = 0;

                    for(n = 0; n < 8; n++)
                        Nodes[*nextfree].u.suns[n] = -1;
                    Nodes[*nextfree].u.d.hmax = 0;

                    if(TopNodes[TopNodes[topnode].Daughter + sub].Daughter == -1)
                        TopLeaves[TopNodes[TopNodes[topnode].Daughter + sub].Leaf].treenode = *nextfree;

                    (*nextfree)++;

                    if(*nextfree >= lastnode)
                        endrun(11, "Not enough force nodes to topnode grid: need %d\n",lastnode);

                    force_create_node_for_topnode(*nextfree - 1, TopNodes[topnode].Daughter + sub,
                            bits + 1, 2 * x + i, 2 * y + j, 2 * z + k, nextfree, lastnode);
                }
    }
}



/*! this function inserts pseudo-particles which will represent the mass
 *  distribution of the other CPUs. Initially, the mass of the
 *  pseudo-particles is set to zero, and their coordinate is set to the
 *  center of the domain-cell they correspond to. These quantities will be
 *  updated later on.
 */
void force_insert_pseudo_particles(const int firstpseudo)
{
    int i, index;

    for(i = 0; i < NTopLeaves; i++)
    {
        index = TopLeaves[i].treenode;

        if(TopLeaves[i].Task != ThisTask)
            Nodes[index].u.suns[0] = firstpseudo + i;
    }
}

int
force_get_next_node(int no)
{
    if(no >= All.MaxPart && no < All.MaxPart + MaxNodes) {
        /* internal node */
        return Nodes[no].u.d.nextnode;
    }
    if(no < All.MaxPart) {
        /* Particle */
        return Nextnode[no];
    }
    else { //if(no >= All.MaxPart + MaxNodes) {
        /* Pseudo Particle */
        return Nextnode[no - MaxNodes];
    }
}

int
force_set_next_node(int no, int next, const int firstnode, const int lastnode)
{
    if(no < 0) return next;
    if(no >= firstnode && no < lastnode) {
        /* internal node */
        Nodes[no].u.d.nextnode = next;
    }
    if(no < firstnode) {
        /* Particle */
        Nextnode[no] = next;
    }
    if(no >= lastnode) {
        /* Pseudo Particle */
        Nextnode[no - (lastnode - firstnode)] = next;
    }

    return next;
}

int
force_get_prev_node(int no)
{
    if(no < All.MaxPart) {
        /* Particle */
        int t = Father[no];
        int next = force_get_next_node(t);
        while(next != no) {
            t = next;
            next = force_get_next_node(t);
        }
        return t;
    } else {
        /* not implemented yet */
        endrun(1, "get_prev_node on non particles is not implemented yet\n");
        return 0;
    }
}

/*Sets the node softening on a node.*/
static void
force_set_node_softening(struct NODE * pnode, const int new_type, const double hsml)
{
    if(pnode->f.MaxSofteningType == 7)
        pnode->f.MaxSofteningType = new_type;
    else
    {
        if(All.ForceSoftening[new_type] > All.ForceSoftening[pnode->f.MaxSofteningType])
            pnode->f.MaxSofteningType = new_type;
        if((All.ForceSoftening[new_type] != All.ForceSoftening[pnode->f.MaxSofteningType])
                || (All.AdaptiveGravsoftForGas && new_type == 0))
            pnode->f.MixedSofteningsInNode = 1;
    }
}

static void
add_particle_moment_to_node(struct NODE * pnode, const struct particle_data * pa)
{
            int k;
            pnode->u.d.mass += (pa->Mass);
            for(k=0; k<3; k++)
                pnode->u.d.s[k] += (pa->Mass * pa->Pos[k]);

            if(pa->Type == 0)
            {
                if(pa->Hsml > pnode->u.d.hmax)
                    pnode->u.d.hmax = pa->Hsml;
            }

            force_set_node_softening(pnode, pa->Type, pa->Hsml);
}

/*! this routine determines the multipole moments for a given internal node
 *  and all its subnodes using a recursive computation.  The result is
 *  stored in the Nodes[] structure in the sequence of this tree-walk.
 *
 *  Note that the bitflags-struct for each node is used to store in the
 *  lowest bits some special information: Bit 0 flags whether the node
 *  belongs to the top-level tree corresponding to the domain
 *  decomposition, while Bit 1 signals whether the top-level node is
 *  dependent on local mass.
 *
 *  bits 2-4 give the particle type with
 *  the maximum softening among the particles in the node, and bit 5
 *  flags whether the node contains any particles with lower softening
 *  than that.
 *
 *  The function also builds the NextNode linked list. The return value
 *  and argument tail is the current tail of the NextNode linked list.
 */

int
force_update_node_recursive(int no, int sib, int tail, const int firstnode, const int lastnode)
{
    /*Set NextNode for this node*/
    tail = force_set_next_node(tail, no, firstnode, lastnode);
    /*For particles and pseudo particles we have nothing to update*/
    if(no < firstnode || no >= lastnode)
        return tail;

    Nodes[no].f.MaxSofteningType=7;
    int j, suns[8];
    /* this "backup" is necessary because the nextnode
     * entry will overwrite one element (union!) */
    for(j = 0; j < 8; j++)
        suns[j] = Nodes[no].u.suns[j];

    memset(&Nodes[no].u.d.s,0,3*sizeof(MyFloat));
    Nodes[no].u.d.mass=0;
    Nodes[no].u.d.hmax=0;
    Nodes[no].f.DependsOnLocalMass=0;
    Nodes[no].f.MixedSofteningsInNode=0;

    for(j = 0; j < 8; j++)
    {
        int nextsib = sib;
        const int p = suns[j];
        int jj;
        /*Empty slot*/
        if(p < 0)
            continue;

        /* check if we have a sibling on the same level */
        for(jj = j + 1; jj < 8; jj++)
            if(suns[jj] >= 0) {
                nextsib = suns[jj];
                break;
            }

        tail = force_update_node_recursive(p, nextsib, tail, firstnode, lastnode);

        if(p >= lastnode)	/* a pseudo particle */
        {
            /* nothing to be done here because the mass of the
             * pseudo-particle is still zero. This will be changed
             * later.
             */
        }
        else if(p < lastnode && p >= firstnode)	/* an internal node or pseudo particle */
        {
            Nodes[no].u.d.mass += (Nodes[p].u.d.mass);
            Nodes[no].u.d.s[0] += (Nodes[p].u.d.mass * Nodes[p].u.d.s[0]);
            Nodes[no].u.d.s[1] += (Nodes[p].u.d.mass * Nodes[p].u.d.s[1]);
            Nodes[no].u.d.s[2] += (Nodes[p].u.d.mass * Nodes[p].u.d.s[2]);

            if(Nodes[p].u.d.hmax > Nodes[no].u.d.hmax)
                Nodes[no].u.d.hmax = Nodes[p].u.d.hmax;

            force_set_node_softening(&Nodes[no], Nodes[p].f.MaxSofteningType, Nodes[p].u.d.hmax);
        }
        else		/* a particle */
        {
            add_particle_moment_to_node(&Nodes[no], &P[p]);
        }
    }

    const double mass = Nodes[no].u.d.mass;
    if(mass)
    {
        Nodes[no].u.d.s[0] /= mass;
        Nodes[no].u.d.s[1] /= mass;
        Nodes[no].u.d.s[2] /= mass;
    }
    /*This only happens for a pseudo particle*/
    else
    {
        Nodes[no].u.d.s[0] = Nodes[no].center[0];
        Nodes[no].u.d.s[1] = Nodes[no].center[1];
        Nodes[no].u.d.s[2] = Nodes[no].center[2];
    }

    Nodes[no].u.d.sibling = sib;

    return tail;
}




/*! This function communicates the values of the multipole moments of the
 *  top-level tree-nodes of the domain grid.  This data can then be used to
 *  update the pseudo-particles on each CPU accordingly.
 */
void force_exchange_pseudodata(void)
{
    int i, no, ta, recvTask;
    int *recvcounts, *recvoffset;
    struct topleaf_momentsdata
    {
        MyFloat s[3];
        MyFloat mass;
        MyFloat hmax;
        struct {
            unsigned int MaxSofteningType :3; /* bits 2-4 */
            unsigned int MixedSofteningsInNode :1;
        };
    }
    *TopLeafMoments;


    TopLeafMoments = (struct topleaf_momentsdata *) mymalloc("TopLeafMoments", NTopLeaves * sizeof(TopLeafMoments[0]));
    memset(&TopLeafMoments[0], 0, sizeof(TopLeafMoments[0]) * NTopLeaves);

    for(i = Tasks[ThisTask].StartLeaf; i < Tasks[ThisTask].EndLeaf; i ++) {
        no = TopLeaves[i].treenode;
        if(TopLeaves[i].Task != ThisTask)
            endrun(131231231, "TopLeave's Task table is corrupted");

        /* read out the multipole moments from the local base cells */
        TopLeafMoments[i].s[0] = Nodes[no].u.d.s[0];
        TopLeafMoments[i].s[1] = Nodes[no].u.d.s[1];
        TopLeafMoments[i].s[2] = Nodes[no].u.d.s[2];
        TopLeafMoments[i].mass = Nodes[no].u.d.mass;
        TopLeafMoments[i].hmax = Nodes[no].u.d.hmax;
        TopLeafMoments[i].MaxSofteningType = Nodes[no].f.MaxSofteningType;
        TopLeafMoments[i].MixedSofteningsInNode = Nodes[no].f.MixedSofteningsInNode;

        /*Set the local base nodes dependence on local mass*/
        while(no >= 0)
        {
            if(Nodes[no].f.DependsOnLocalMass)
                break;

            Nodes[no].f.DependsOnLocalMass = 1;

            no = Nodes[no].father;
        }
    }

    /* share the pseudo-particle data accross CPUs */

    recvcounts = (int *) mymalloc("recvcounts", sizeof(int) * NTask);
    recvoffset = (int *) mymalloc("recvoffset", sizeof(int) * NTask);

    for(recvTask = 0; recvTask < NTask; recvTask++)
    {
        recvoffset[recvTask] = Tasks[recvTask].StartLeaf * sizeof(TopLeafMoments[0]);
        recvcounts[recvTask] = (Tasks[recvTask].EndLeaf - Tasks[recvTask].StartLeaf) * sizeof(TopLeafMoments[0]);
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
            &TopLeafMoments[0], recvcounts, recvoffset,
            MPI_BYTE, MPI_COMM_WORLD);

    myfree(recvoffset);
    myfree(recvcounts);


    for(ta = 0; ta < NTask; ta++) {
        if(ta == ThisTask) continue; /* bypass ThisTask since it is already up to date */

        for(i = Tasks[ta].StartLeaf; i < Tasks[ta].EndLeaf; i ++) {
            no = TopLeaves[i].treenode;

            Nodes[no].u.d.s[0] = TopLeafMoments[i].s[0];
            Nodes[no].u.d.s[1] = TopLeafMoments[i].s[1];
            Nodes[no].u.d.s[2] = TopLeafMoments[i].s[2];
            Nodes[no].u.d.mass = TopLeafMoments[i].mass;
            Nodes[no].u.d.hmax = TopLeafMoments[i].hmax;
            Nodes[no].f.MaxSofteningType = TopLeafMoments[i].MaxSofteningType;
            Nodes[no].f.MixedSofteningsInNode = TopLeafMoments[i].MixedSofteningsInNode;
         }
    }
    myfree(TopLeafMoments);
}


/*! This function updates the top-level tree after the multipole moments of
 *  the pseudo-particles have been updated.
 */
void force_treeupdate_pseudos(int no, const int firstnode, const int lastnode)
{
    int j, p;
    MyFloat hmax;
    MyFloat s[3], mass;

    int maxsofttype;

    mass = 0;
    s[0] = 0;
    s[1] = 0;
    s[2] = 0;
    hmax = 0;
    maxsofttype = 7;

    p = Nodes[no].u.d.nextnode;

    /* since we are dealing with top-level nodes, we know that there are 8 consecutive daughter nodes */
    for(j = 0; j < 8; j++)
    {
        /*This may not happen as we are an internal top level node*/
        if(p < firstnode || p >= lastnode)
            endrun(6767, "Updating pseudos: %d -> %d which is not an internal node between %d and %d.",no, p, firstnode, lastnode);

        if(Nodes[p].f.InternalTopLevel)
            force_treeupdate_pseudos(p, firstnode, lastnode);

        mass += (Nodes[p].u.d.mass);
        s[0] += (Nodes[p].u.d.mass * Nodes[p].u.d.s[0]);
        s[1] += (Nodes[p].u.d.mass * Nodes[p].u.d.s[1]);
        s[2] += (Nodes[p].u.d.mass * Nodes[p].u.d.s[2]);

        if(Nodes[p].u.d.hmax > hmax)
            hmax = Nodes[p].u.d.hmax;

        Nodes[no].f.MixedSofteningsInNode = Nodes[p].f.MixedSofteningsInNode;

        if(maxsofttype == 7)
            maxsofttype = Nodes[p].f.MaxSofteningType;
        else
        {
            int current_maxsofttype = Nodes[p].f.MaxSofteningType;
            if(current_maxsofttype != 7)
            {
                if(All.ForceSoftening[current_maxsofttype] > All.ForceSoftening[maxsofttype])
                    maxsofttype = current_maxsofttype;
                if(All.ForceSoftening[current_maxsofttype] != All.ForceSoftening[maxsofttype])
                    Nodes[no].f.MixedSofteningsInNode = 1;
            }
        }
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

    Nodes[no].u.d.hmax = hmax;

    Nodes[no].f.MaxSofteningType = maxsofttype;
}

/*! This function updates the hmax-values in tree nodes that hold SPH
 *  particles. These values are needed to find all neighbors in the
 *  hydro-force computation.  Since the Hsml-values are potentially changed
 *  in the SPH-density computation, force_update_hmax() should be carried
 *  out just before the hydrodynamical SPH forces are computed, i.e. after
 *  density().
 */
void force_update_hmax(int * activeset, int size)
{
    int i, ta; 
    int *counts, *offsets;
    struct dirty_node_data {
        int treenode;
        MyFloat hmax;
    } * DirtyTopLevelNodes;

    int NumDirtyTopLevelNodes;

    walltime_measure("/Misc");

    NumDirtyTopLevelNodes = 0;

    /* At most NTopLeaves are dirty, since we are only concerned with TOPLEVEL nodes */
    DirtyTopLevelNodes = (struct dirty_node_data*) mymalloc("DirtyTopLevelNodes", NTopLeaves * sizeof(DirtyTopLevelNodes[0]));

    char * NodeIsDirty = (char *) mymalloc("NodeIsDirty", MaxNodes * sizeof(char));

    /* FIXME: actually only TOPLEVEL nodes contains the local mass can potentially be dirty,
     *  we may want to save a list of them to speed this up.
     * */
    for(i = All.MaxPart; i < All.MaxPart + MaxNodes; i ++) {
        NodeIsDirty[i - All.MaxPart] = 0;
    }

    for(i = 0; i < size; i++)
    {
        const int p_i = activeset[i];

        if(P[p_i].Type != 0)
            continue;

        int no = Father[p_i];

        while(no >= 0)
        {
            if(P[p_i].Hsml <= Nodes[no].u.d.hmax) break;

            Nodes[no].u.d.hmax = P[p_i].Hsml;

            if(Nodes[no].f.TopLevel) /* we reached a top-level node */
            {
                if (!NodeIsDirty[no - All.MaxPart]) {
                    NodeIsDirty[no - All.MaxPart] = 1;
                    DirtyTopLevelNodes[NumDirtyTopLevelNodes].treenode = no;
                    DirtyTopLevelNodes[NumDirtyTopLevelNodes].hmax = Nodes[no].u.d.hmax;
                    NumDirtyTopLevelNodes ++;
                }
                break;
            }

            no = Nodes[no].father;
        }
    }

    /* share the hmax-data of the dirty nodes accross CPUs */

    counts = (int *) mymalloc("counts", sizeof(int) * NTask);
    offsets = (int *) mymalloc("offsets", sizeof(int) * (NTask + 1));

    MPI_Allgather(&NumDirtyTopLevelNodes, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);

    for(ta = 0, offsets[0] = 0; ta < NTask; ta++)
    {
        offsets[ta + 1] = offsets[ta] + counts[ta];
    }

    message(0, "Hmax exchange: %d toplevel tree nodes out of %d\n", offsets[NTask], NTopLeaves);

    /* move to the right place for MPI_INPLACE*/
    memmove(&DirtyTopLevelNodes[offsets[ThisTask]], &DirtyTopLevelNodes[0], NumDirtyTopLevelNodes * sizeof(DirtyTopLevelNodes[0]));

    MPI_Datatype MPI_TYPE_DIRTY_NODES;
    MPI_Type_contiguous(sizeof(DirtyTopLevelNodes[0]), MPI_BYTE, &MPI_TYPE_DIRTY_NODES);
    MPI_Type_commit(&MPI_TYPE_DIRTY_NODES);

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
            DirtyTopLevelNodes, counts, offsets, MPI_TYPE_DIRTY_NODES, MPI_COMM_WORLD);

    MPI_Type_free(&MPI_TYPE_DIRTY_NODES);

    for(i = 0; i < offsets[NTask]; i++)
    {
        int no = DirtyTopLevelNodes[i].treenode;

        /* FIXME: why does this matter? The logic is simpler if we just blindly update them all.
            ::: to avoid that the hmax is updated twice :::*/
        if(Nodes[no].f.DependsOnLocalMass)
            no = Nodes[no].father;

        while(no >= 0)
        {
            if(DirtyTopLevelNodes[i].hmax <= Nodes[no].u.d.hmax) break;

            Nodes[no].u.d.hmax = DirtyTopLevelNodes[i].hmax;

            no = Nodes[no].father;
        }
    }

    myfree(offsets);
    myfree(counts);
    myfree(NodeIsDirty);
    myfree(DirtyTopLevelNodes);

    walltime_measure("/Tree/HmaxUpdate");
}

/*! This function allocates the memory used for storage of the tree and of
 *  auxiliary arrays needed for tree-walk and link-lists.  Usually,
 *  maxnodes approximately equal to 0.7*maxpart is sufficient to store the
 *  tree for up to maxpart particles.
 */
size_t force_treeallocate(int maxnodes, int maxpart, int first_node_offset)
{
    size_t bytes;
    size_t allbytes = 0;

    tree_allocated_flag = 1;
    MaxNodes = maxnodes;
    message(0, "Allocating memory for %d tree-nodes (MaxPart=%d).\n", maxnodes, maxpart);
    Nodes_base = (struct NODE *) mymalloc("Nodes_base", bytes = (MaxNodes + 1) * sizeof(struct NODE));
    allbytes += bytes;
    Nodes = Nodes_base - first_node_offset;
    Nextnode = (int *) mymalloc("Nextnode", bytes = (maxpart + NTopNodes) * sizeof(int));
    allbytes += bytes;
    Father = (int *) mymalloc("Father", bytes = (maxpart) * sizeof(int));
    allbytes += bytes;
    return allbytes;
}

/*! This function frees the memory allocated for the tree, i.e. it frees
 *  the space allocated by the function force_treeallocate().
 */
void force_tree_free(void)
{
    myfree(Father);
    myfree(Nextnode);
    myfree(Nodes_base);
    tree_allocated_flag = 0;
}
