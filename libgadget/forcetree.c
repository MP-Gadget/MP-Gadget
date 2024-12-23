#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "domain.h"
#include "forcetree.h"
#include "checkpoint.h"
#include "walltime.h"
#include "slotsmanager.h"
#include "partmanager.h"
#include "utils/endrun.h"
#include "utils/system.h"
#include "utils/mymalloc.h"

/*! \file forcetree.c
 *  \brief gravitational tree
 *
 *  This file contains the computation of the gravitational force by means
 *  of a tree. The type of tree implemented is a geometrical oct-tree,
 *  starting from a cube encompassing all particles. This cube is
 *  automatically found in the ddecomp decomposition, which also splits up
 *  the global "top-level" tree along node boundaries, moving the particles
 *  of different parts of the tree to separate processors.
 *
 * Local naming convention: once built it is ForceTree * tree, passed by reference.
 * While the nodes are still being added it is ForceTree tb, passed by value.
 */

static struct forcetree_params
{
    /* Each processor allocates a number of nodes which is TreeAllocFactor times
       the maximum(!) number of particles.  Note: A typical local tree for N
       particles needs usually about ~0.65*N nodes.
       If the allocated memory is not sufficient, this parameter will be increased.*/
    double TreeAllocFactor;
} ForceTreeParams;

void
init_forcetree_params(const double treeallocfactor)
{
    /* This was increased due to the extra nodes created by subtrees*/
    ForceTreeParams.TreeAllocFactor = treeallocfactor;
}

static ForceTree
force_tree_build(int mask, DomainDecomp * ddecomp, const ActiveParticles * act, const int DoMoments, const int alloc_father, const char * EmergencyOutputDir);

static void
force_treeupdate_pseudos(const int no, const int level, const ForceTree * const tree);

static void
force_create_node_for_topnode(int no, int topnode, struct NODE * Nodes, const DomainDecomp * ddecomp, const int bits, const int x, const int y, const int z, int *nextfree, const int lastnode);

static void
force_exchange_pseudodata(const ForceTree * const tree, const DomainDecomp * const ddecomp);

static void
add_particle_moment_to_node(struct NODE * pnode, const struct particle_data * const part);

#ifdef DEBUG
/* Walk the constructed tree, validating sibling and nextnode as we go*/
static void force_validate_nextlist(const ForceTree * tree)
{
    int no = tree->firstnode;
    while(no != -1)
    {
        struct NODE * current = &tree->Nodes[no];
        if(current->sibling != -1 && !node_is_node(current->sibling, tree))
            endrun(5, "Node %d (type %d) has sibling %d next %d father %d first %ld final %ld last %ld ntop %d\n", no, current->f.ChildType, current->sibling, current->s.suns[0], current->father, tree->firstnode, tree->firstnode + tree->numnodes, tree->lastnode, tree->NTopLeaves);

        if(current->f.ChildType == PSEUDO_NODE_TYPE) {
            /* pseudo particle: nextnode should be a pseudo particle, sibling should be a node. */
            if(!node_is_pseudo_particle(current->s.suns[0], tree))
                endrun(5, "Pseudo Node %d has next node %d sibling %d father %d first %ld final %ld last %ld ntop %d\n", no, current->s.suns[0], current->sibling, current->father, tree->firstnode, tree->firstnode + tree->numnodes, tree->lastnode, tree->NTopLeaves);
        }
        else if(current->f.ChildType == NODE_NODE_TYPE) {
            /* Next node should be another node */
            if(!node_is_node(current->s.suns[0], tree))
                endrun(5, "Node Node %d has next node which is particle %d sibling %d father %d first %ld final %ld last %ld ntop %d\n", no, current->s.suns[0], current->sibling, current->father, tree->firstnode, tree->firstnode + tree->numnodes, tree->lastnode, tree->NTopLeaves);
            no = current->s.suns[0];
            continue;
        }
        no = current->sibling;
    }
    /* Every node should have a valid father: collect those that do not.*/
    for(no = tree->firstnode; no < tree->firstnode + tree->numnodes; no++)
    {
        if(!node_is_node(tree->Nodes[no].father, tree) && tree->Nodes[no].father >= 0) {
            struct NODE *current = &tree->Nodes[no];
            message(1, "Danger! no %d has father %d, next %d sib %d, (ptype = %d) len %g center (%g %g %g) mass %g cofm %g %g %g TL %d DLM %d ITL %d nocc %d suns %d %d %d %d\n", no, current->father, current->s.suns[0], current->sibling, current->f.ChildType,
                current->len, current->center[0], current->center[1], current->center[2],
                current->mom.mass, current->mom.cofm[0], current->mom.cofm[1], current->mom.cofm[2],
                current->f.TopLevel, current->f.DependsOnLocalMass, current->f.InternalTopLevel, current->s.noccupied,
                current->s.suns[0], current->s.suns[1], current->s.suns[2], current->s.suns[3]);
        }
    }
    walltime_measure("/Tree/Build/Validate");
}
#endif

int
force_tree_allocated(const ForceTree * tree)
{
    return tree->tree_allocated_flag;
}

/* Build a tree structure using all particles, compute moments and allocate a father array.
 * This is the fattest tree constructor, allows moments and walking up and down.*/
void
force_tree_full(ForceTree * tree, DomainDecomp * ddecomp, const int HybridNuTracer, const char * EmergencyOutputDir)
{
    if(force_tree_allocated(tree)) {
        force_tree_free(tree);
    }
    walltime_measure("/Misc");

    /* Build for all particles*/
    ActiveParticles act = init_empty_active_particles(PartManager);
    int mask = ALLMASK;
    if(HybridNuTracer)
        mask = GASMASK + DMMASK + STARMASK + BHMASK;

    /*No father array by default, only need it for hmax. We want moments.*/
    *tree = force_tree_build(mask, ddecomp, &act, 1, 1, EmergencyOutputDir);
    /* This is all particles (even if there are neutrinos)*/
    tree->full_particle_tree_flag = 1;
}

void
force_tree_active_moments(ForceTree * tree, DomainDecomp * ddecomp, const ActiveParticles *act, const int HybridNuTracer, const int alloc_father, const char * EmergencyOutputDir)
{
    //message(0, "Tree construction.  (presently allocated=%g MB)\n", mymalloc_usedbytes() / (1024.0 * 1024.0));

    if(force_tree_allocated(tree)) {
        force_tree_free(tree);
    }
    walltime_measure("/Misc");
    int mask = ALLMASK;
    // Exclude neutrinos
    if(HybridNuTracer)
        mask = GASMASK + DMMASK + STARMASK + BHMASK;

    /*No father array by default, only need it for hmax. We want moments.*/
    *tree = force_tree_build(mask, ddecomp, act, 1, alloc_father, EmergencyOutputDir);
    /* This is all particles if active particle is null (even if there are neutrinos)*/
    if(!act->ActiveParticle)
        tree->full_particle_tree_flag = 1;
}

void
force_tree_rebuild_mask(ForceTree * tree, DomainDecomp * ddecomp, int mask, const char * EmergencyOutputDir)
{
    message(0, "Tree construction for types: %d.\n", mask);

    if(force_tree_allocated(tree)) {
        force_tree_free(tree);
    }

    /* Build for all particles*/
    ActiveParticles act = init_empty_active_particles(PartManager);
    /* No moments, but need father for hmax. The hybridnugrav only affects moments, so isn't needed.*/
    *tree = force_tree_build(mask, ddecomp, &act, 0, 1, EmergencyOutputDir);
    if(mask == ALLMASK)
        tree->full_particle_tree_flag = 1;
}


/* Compute the multipole moments recursively*/
void
force_tree_calc_moments(ForceTree * tree, DomainDecomp * ddecomp)
{
    force_update_node_parallel(tree, ddecomp);
    /* Exchange the pseudo-data*/
    force_exchange_pseudodata(tree, ddecomp);
    #pragma omp parallel
    #pragma omp single nowait
    {
        force_treeupdate_pseudos(tree->firstnode, 1, tree);
    }
    tree->moments_computed_flag = 1;
    tree->hmax_computed_flag = 1;
}

/*! Constructs the gravitational oct-tree.
 *
 *  The index convention for accessing tree nodes is the following: the
 *  indices 0...NumPart-1 reference single particles, the indices
 *  firstnode....firstnode +nodes-1 reference tree nodes. `Nodes_base'
 *  points to the first tree node, while `nodes' is shifted such that
 *  nodes[firstnode] gives the first tree node. Finally, node indices
 *  with values 'tb.lastnode' and larger indicate "pseudo
 *  particles", i.e. multipole moments of top-level nodes that lie on
 *  different CPUs. If such a node needs to be opened, the corresponding
 *  particle must be exported to that CPU. */
ForceTree
force_tree_build(int mask, DomainDecomp * ddecomp, const ActiveParticles *act, const int DoMoments, const int alloc_father, const char * EmergencyOutputDir)
{
    ForceTree tree;
    int64_t maxnodes = ForceTreeParams.TreeAllocFactor * PartManager->NumPart + ddecomp->NTopNodes;
    /* int64_t maxmaxnodes;
     MPI_Reduce(&maxnodes, &maxmaxnodes, 1, MPI_INT64, MPI_MAX,0, MPI_COMM_WORLD);
    message(0, "Treebuild: Largest is %g MByte for %ld tree nodes. firstnode %ld. (presently allocated %g MB)\n",
         maxmaxnodes * sizeof(struct NODE) / (1024.0 * 1024.0), maxmaxnodes, PartManager->MaxPart,
         mymalloc_usedbytes() / (1024.0 * 1024.0));*/

    do
    {
        /* Allocate memory: note that because node numbers are passed around between ranks,
         * this has to be something which is the same on all ranks. */
        tree = force_treeallocate(maxnodes, PartManager->MaxPart, ddecomp, alloc_father, 0);
        tree.mask = mask;
        tree.BoxSize = PartManager->BoxSize;
        force_tree_create_nodes(&tree, act, mask, ddecomp);
        if(tree.numnodes >= tree.lastnode - tree.firstnode)
        {
            message(1, "Not enough tree nodes (%ld) for %ld particles. Created %ld\n", maxnodes, act->NumActiveParticle, tree.numnodes);
            force_tree_free(&tree);
            ForceTreeParams.TreeAllocFactor *= 1.15;
            if(ForceTreeParams.TreeAllocFactor > 3.0) {
#ifndef DEBUG
                endrun(2, "TreeAllocFactor is %g nodes, which is too large!\n", ForceTreeParams.TreeAllocFactor);
#else
                break;
#endif
            }
            maxnodes = ForceTreeParams.TreeAllocFactor * PartManager->NumPart + ddecomp->NTopNodes;
            message(1, "TreeAllocFactor from %g to %g now %ld tree nodes\n", ForceTreeParams.TreeAllocFactor, ForceTreeParams.TreeAllocFactor*1.15, maxnodes);
        }
    }
    while(tree.numnodes >= tree.lastnode - tree.firstnode);

#ifdef DEBUG
    if(MPIU_Any(ForceTreeParams.TreeAllocFactor > 3.0, MPI_COMM_WORLD)) {
        /* Assume scale factor = 1 for dump as position is not affected.*/
        if(EmergencyOutputDir) {
            Cosmology CP = {0};
            CP.Omega0 = 0.3;
            CP.OmegaLambda = 0.7;
            CP.HubbleParam = 0.7;
            dump_snapshot("FORCETREE-DUMP", 1, &CP, EmergencyOutputDir);
        }
        endrun(2, "Required too many nodes, snapshot dumped\n");
    }
#endif
    report_memory_usage("FORCETREE");
    tree.Nodes_base = (struct NODE *) myrealloc(tree.Nodes_base, (tree.numnodes +1) * sizeof(struct NODE));

    /*Update the oct-tree struct so it knows about the memory change*/
    tree.Nodes = tree.Nodes_base - tree.firstnode;

    tree.moments_computed_flag = 0;

    if(DoMoments) {
        walltime_measure("/Tree/Build/Nodes");
        force_tree_calc_moments(&tree, ddecomp);
        walltime_measure("/Tree/Build/Moments");
    }

    int64_t allact = tree.NumParticles;
    int64_t maxnumnodes = tree.numnodes;
#ifdef DEBUG
    force_validate_nextlist(&tree);
    MPI_Reduce(&tree.NumParticles, &allact, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tree.numnodes, &maxnumnodes, 1, MPI_INT64, MPI_MAX, 0, MPI_COMM_WORLD);
#endif
    message(0, "Tree constructed (type mask: %d moments: %d) with %ld particles. First node %ld, num nodes %ld, first pseudo %ld. NTopLeaves %d\n",
            mask, tree.moments_computed_flag, allact, tree.firstnode, maxnumnodes, tree.lastnode, tree.NTopLeaves);
    return tree;
}

/* Get the subnode for a given particle and parent node.
 * This splits a parent node into 8 subregions depending on the particle position.
 * node is the parent node to split, p_i is the index of the particle we
 * are currently inserting
 * Returns a value between 0 and 7.
 * */
int get_subnode(const struct NODE * node, const double Pos[3])
{
    /*Loop is unrolled to help out the compiler,which normally only manages it at -O3*/
     return (Pos[0] > node->center[0]) +
            ((Pos[1] > node->center[1]) << 1) +
            ((Pos[2] > node->center[2]) << 2);
}

/*Check whether a particle is inside the volume covered by a node,
 * by checking whether each dimension is close enough to center (L1 metric).*/
static inline int inside_node(const struct NODE * node, const double Pos[3])
{
    /*One can also use a loop, but the compiler unrolls it only at -O3,
     *so this is a little faster*/
    int inside =
        (fabs(2*(Pos[0] - node->center[0])) <= node->len) *
        (fabs(2*(Pos[1] - node->center[1])) <= node->len) *
        (fabs(2*(Pos[2] - node->center[2])) <= node->len);
    return inside;
}

/*Initialise an internal node at nfreep. The parent is assumed to be locked, and
 * we have assured that nothing else will change nfreep while we are here.*/
static void init_internal_node(struct NODE *nfreep, struct NODE *parent, int subnode)
{
    int j;
    const MyFloat lenhalf = 0.25 * parent->len;
    nfreep->len = 0.5 * parent->len;
    nfreep->sibling = -10;
    nfreep->father = -10;
    nfreep->f.TopLevel = 0;
    nfreep->f.InternalTopLevel = 0;
    nfreep->f.DependsOnLocalMass = 0;
    nfreep->f.ChildType = PARTICLE_NODE_TYPE;
    nfreep->f.unused = 0;

    for(j = 0; j < 3; j++) {
        /* Detect which quadrant we are in by testing the bits of subnode:
         * if (subnode & [1,2,4]) is true we add lenhalf, otherwise subtract lenhalf*/
        const int sign = (subnode & (1 << j)) ? 1 : -1;
        nfreep->center[j] = parent->center[j] + sign*lenhalf;
    }
    for(j = 0; j < NMAXCHILD; j++)
        nfreep->s.suns[j] = -1;
    nfreep->s.noccupied = 0;
    memset(&(nfreep->mom.cofm),0,3*sizeof(MyFloat));
    nfreep->mom.mass = 0;
    nfreep->mom.hmax = 0;
}

/* Size of the free Node thread cache.
 * 12 8-node rows (works out at 8kB) was found
 * to be optimal for an Intel skylake and
 * an AMD Zen2 with 12 threads.*/
#define NODECACHE_SIZE (8*12)

/*Structure containing thread-local parameters of the tree build*/
struct NodeCache {
    int nnext_thread;
    int nrem_thread;
};

/*Get a pointer to memory for 8 free nodes, from our node cache. */
int get_freenode(int * nnext, struct NodeCache *nc)
{
    /*Get memory for an extra node from our cache.*/
    if(nc->nrem_thread < 8) {
        nc->nnext_thread = atomic_fetch_and_add(nnext, NODECACHE_SIZE);
        nc->nrem_thread = NODECACHE_SIZE;
    }
    const int ninsert = nc->nnext_thread;
    nc->nnext_thread += 8;
    nc->nrem_thread -= 8;
    return ninsert;
}

/* Add a particle to a node in a known empty location.
 * Parent is assumed to be locked.*/
static int
modify_internal_node(int parent, int subnode, int p_toplace, const ForceTree tb)
{
    if(tb.Father)
        tb.Father[p_toplace] = parent;
    tb.Nodes[parent].s.suns[subnode] = p_toplace;
    add_particle_moment_to_node(&tb.Nodes[parent], &P[p_toplace]);
    return 0;
}


/* Create a new layer of nodes beneath the current node, and place the particle.
 * Must have node lock.*/
static int
create_new_node_layer(int firstparent, int p_toplace, const ForceTree tb, int *nnext, struct NodeCache *nc)
{
    /* This is so we can defer changing
     * the type of the existing node until the end.*/
    int parent = firstparent;

    do {
        int i;
        struct NODE *nprnt = &tb.Nodes[parent];

        /* Braces to scope oldsuns and newsuns*/
        {
        int newsuns[NMAXCHILD];

        int * oldsuns = nprnt->s.suns;

        /*We have two particles here, so create a new child node to store them both.*/
        /* if we are here the node must be large enough, thus contain exactly one child. */
        /* The parent is already a leaf, need to split */
        /* Get memory for 8 extra nodes from our cache.*/
        newsuns[0] = get_freenode(nnext, nc);
        /*If we already have too many nodes, exit loop.*/
        if(nc->nnext_thread >= tb.lastnode) {
            /* This means that we have > NMAXCHILD particles in the same place,
            * which usually indicates a bug in the particle evolution. Print some helpful debug information.*/
            message(1, "Failed placing %d at %g %g %g, type %d, ID %ld. Others were %d (%g %g %g, t %d ID %ld) and %d (%g %g %g, t %d ID %ld).\n",
                p_toplace, P[p_toplace].Pos[0], P[p_toplace].Pos[1], P[p_toplace].Pos[2], P[p_toplace].Type, P[p_toplace].ID,
                oldsuns[0], P[oldsuns[0]].Pos[0], P[oldsuns[0]].Pos[1], P[oldsuns[0]].Pos[2], P[oldsuns[0]].Type, P[oldsuns[0]].ID,
                oldsuns[1], P[oldsuns[1]].Pos[0], P[oldsuns[1]].Pos[1], P[oldsuns[1]].Pos[2], P[oldsuns[1]].Type, P[oldsuns[1]].ID
            );
            nc->nnext_thread = tb.lastnode + 10 * NODECACHE_SIZE;
            /* If this is not the first layer created,
                * we need to mark the overall parent as a node node
                * while marking this one as a particle node */
            if(firstparent != parent)
            {
                nprnt->f.ChildType = PARTICLE_NODE_TYPE;
                nprnt->s.noccupied = NMAXCHILD;
                tb.Nodes[firstparent].f.ChildType = NODE_NODE_TYPE;
                tb.Nodes[firstparent].s.noccupied = NODEFULL;
            }
            return 1;
        }
        for(i=0; i<8; i++) {
            newsuns[i] = newsuns[0] + i;
            struct NODE *nfreep = &tb.Nodes[newsuns[i]];
            /* We create a new leaf node.*/
            init_internal_node(nfreep, nprnt, i);
            /*Set father of new node*/
            nfreep->father = parent;
        }
        /*Initialize the remaining entries to empty*/
        for(i=8; i<NMAXCHILD;i++)
            newsuns[i] = -1;

        for(i=0; i < NMAXCHILD; i++) {
            /* Re-attach each particle to the appropriate new leaf.
            * Notice that since we have NMAXCHILD slots on each child and NMAXCHILD particles,
            * we will always have a free slot. */
            int subnode = get_subnode(nprnt, P[oldsuns[i]].Pos);
            int child = newsuns[subnode];
            struct NODE * nchild = &tb.Nodes[child];
            modify_internal_node(child, nchild->s.noccupied, oldsuns[i], tb);
            nchild->s.noccupied++;
        }
        /* Copy the new node array into the node*/
        memcpy(nprnt->s.suns, newsuns, NMAXCHILD * sizeof(int));
        } /* After this brace oldsuns and newsuns are invalid*/

        /* Set sibling for the new rank. Since empty at this point, point onwards.*/
        for(i=0; i<7; i++) {
            int child = nprnt->s.suns[i];
            struct NODE * nchild = &tb.Nodes[child];
            nchild->sibling = nprnt->s.suns[i+1];
        }
        /* Final child needs special handling: set to the parent's sibling.*/
        tb.Nodes[nprnt->s.suns[7]].sibling = nprnt->sibling;
        /* Zero the momenta for the parent*/
        memset(&nprnt->mom, 0, sizeof(nprnt->mom));

        /* Now try again to add the new particle*/
        int subnode = get_subnode(nprnt, P[p_toplace].Pos);
        int child = nprnt->s.suns[subnode];
        struct NODE * nchild = &tb.Nodes[child];
        if(nchild->s.noccupied < NMAXCHILD) {
            modify_internal_node(child, nchild->s.noccupied, p_toplace, tb);
            nchild->s.noccupied++;
            break;
        }
        /* The attached particles are already within one subnode of the new node.
         * Iterate, creating a new layer beneath.*/
        else {
            /* The current child is going to have new nodes created beneath it,
             * so mark it a Node-containing node. It cannot be accessed until
             * we mark the top-level parent, so no need for atomics.*/
            tb.Nodes[child].f.ChildType = NODE_NODE_TYPE;
            tb.Nodes[child].s.noccupied = NODEFULL;
            parent = child;
        }
    } while(1);

    /* A new node is created. Mark the (original) parent as an internal node with node children.
     * This goes last so that we don't access the child before it is constructed.*/
    tb.Nodes[firstparent].f.ChildType = NODE_NODE_TYPE;
    tb.Nodes[firstparent].s.noccupied = NODEFULL;
    return 0;
}

/* Add a particle to the tree, extending the tree as necessary. Locking is done,
 * so may be called from a threaded context*/
int add_particle_to_tree(int i, int cur_start, const ForceTree tb, struct NodeCache *nc, int* nnext)
{
    int child, nocc;
    int cur = cur_start;
    /*Walk the main tree until we get something that isn't an internal node.*/
    do
    {
        /*No lock needed: if we have an internal node here it will be stable*/
        nocc = tb.Nodes[cur].s.noccupied;

        /* This node still has space for a particle (or needs conversion)*/
        if(nocc < NODEFULL)
            break;

        /* This node has child subnodes: find them.*/
        int subnode = get_subnode(&tb.Nodes[cur], P[i].Pos);
        /*No lock needed: if we have an internal node here it will be stable*/
        child = tb.Nodes[cur].s.suns[subnode];

        if(child > tb.lastnode || child < tb.firstnode)
            endrun(1,"Corruption in tree build: N[%d].[%d] = %d > lastnode (%ld)\n",cur, subnode, child, tb.lastnode);
        cur = child;
    }
    while(child >= tb.firstnode);

    /* We have a guaranteed spot.*/
    nocc = tb.Nodes[cur].s.noccupied;
    tb.Nodes[cur].s.noccupied++;

    /* Now we have something that isn't an internal node. We can place the particle! */
    if(nocc < NMAXCHILD)
        modify_internal_node(cur, nocc, i, tb);
    /* In this case we need to create a new layer of nodes beneath this one*/
    else if(nocc < NODEFULL) {
        if(create_new_node_layer(cur, i, tb, nnext, nc))
            return -1;
    } else
        endrun(2, "Tried to convert already converted node %d with nocc = %d\n", cur, nocc);
    return cur;
}

/* Merge two partial trees together. Trees are walked simultaneously.
 * A merge is done when a particle node is encountered in one of the side trees.
 * The merge rule is that the node node is attached to the
 * left-most old parent and the particles are re-attached to the node node*/
int
merge_partial_force_trees(int left, int right, struct NodeCache * nc, int * nnext, const struct ForceTree tb)
{
    int this_left = left;
    int this_right = right;
    const int left_end = tb.Nodes[left].sibling;
    const int right_end = tb.Nodes[right].sibling;
//     message(5, "Ends: %d %d\n", left_end, right_end);
    while(this_left != left_end && this_right != right_end)
    {
        if(this_left < tb.firstnode || this_right < tb.firstnode)
            endrun(10, "Encountered invalid node: %d %d < first %ld\n", this_left, this_right, tb.firstnode);
        struct NODE * nleft = &tb.Nodes[this_left];
        struct NODE * nright = &tb.Nodes[this_right];
        if(nc->nnext_thread >= tb.lastnode)
            return 1;
#ifdef DEBUG
        /* Stop when we reach another topnode*/
        if((nleft->f.TopLevel && this_left != left) || (nright->f.TopLevel && this_right != right))
            endrun(6, "Encountered another topnode: left %d == right %d! type %d\n", this_left, this_right, nleft->f.ChildType);
        if(this_left == this_right)
            endrun(6, "Odd: left %d == right %d! type %d\n", this_left, this_right, nleft->f.ChildType);
//         message(1, "left %d right %d\n", this_left, this_right);
        /* Trees should be synced*/
        if(fabs(nleft->len / nright->len-1) > 1e-6)
            endrun(6, "Merge unsynced trees: %d %d len %g %g\n", this_left, this_right, nleft->len, nright->len);
#endif
        /* Two node nodes: keep walking down*/
        if(nleft->f.ChildType == NODE_NODE_TYPE && nright->f.ChildType == NODE_NODE_TYPE) {
            if(tb.Nodes[nleft->s.suns[0]].father < 0 || tb.Nodes[nright->s.suns[0]].father < 0)
                endrun(7, "Walking to nodes (%d %d) from (%d %d) fathers (%d %d)\n",
                       nleft->s.suns[0], nright->s.suns[0], this_left, this_right, tb.Nodes[nleft->s.suns[0]].father, tb.Nodes[nright->s.suns[0]].father);
            this_left = nleft->s.suns[0];
            this_right = nright->s.suns[0];
            continue;
        }
        /* If the right node has particles, add them to the left node, go to sibling on right and left.*/
        else if(nright->f.ChildType == PARTICLE_NODE_TYPE) {
            int i;
            for(i = 0; i < nright->s.noccupied; i++) {
                if(nright->s.suns[i] >= tb.firstnode)
                    endrun(8, "Bad child %d of %d in %d\n", i, nright->s.suns[i], this_right);
                if(add_particle_to_tree(nright->s.suns[i], this_left, tb, nc, nnext) < 0)
                    return 1;
            }
            /* Make sure that nodes which have
             * this_right as a sibling (there will
             * be a max of one, as it is a particle node
             * with no children) point to the replacement
             * on the left*/
            /* This condition is checking for the root node, which has no siblings*/
            if(this_right > right) {
                /* Find the father, then the next child*/
                struct NODE * fat = &tb.Nodes[nright->father];
                /* Find the position of this child in the father*/
                int sunloc = 0;
                for(i = 0; i < 8; i++)
                {
                    if(fat->s.suns[i] == this_right) {
                        sunloc = i;
                        break;
                    }
                }
                /* Change the sibling of the child next to this one*/
                if(sunloc > 0) {
                    if(tb.Nodes[fat->s.suns[i-1]].sibling == this_right)
                        tb.Nodes[fat->s.suns[i-1]].sibling = this_left;
                }
            }
            /* Mark the right node as now invalid*/
            nright->father = -5;
            /* Now go to sibling*/
            this_left = nleft->sibling;
            this_right = nright->sibling;
            continue;
        }
        /* If the left node has particles, add them to the right node,
         * then copy the right node over the left node and go to (old) sibling on right and left.*/
        else if(nleft->f.ChildType == PARTICLE_NODE_TYPE && nright->f.ChildType == NODE_NODE_TYPE) {
            /* Add the left particles to the right*/
            int i;
            for(i = 0; i < nleft->s.noccupied; i++) {
                if(nleft->s.suns[i] >= tb.firstnode)
                    endrun(8, "Bad child %d of %d in left %d\n", i, nleft->s.suns[i], this_left);
                if(add_particle_to_tree(nleft->s.suns[i], this_right, tb, nc, nnext) < 0)
                    return 1;
            }
            /* Copy the right node over the left*/
            memmove(&nleft->s, &nright->s, sizeof(nleft->s));
            nleft->f.ChildType = NODE_NODE_TYPE;
            /* Zero the momenta for the parent*/
            memset(&nleft->mom, 0, sizeof(nleft->mom));
            /* Reset children to the new parent:
             * this assumes nright is a NODE NODE*/
            for(i = 0; i < 8; i++) {
                int child = nleft->s.suns[i];
                tb.Nodes[child].father = this_left;
            }
            /* Make sure final child points to the parent's sibling.*/
#ifdef DEBUG
            int oldsib = tb.Nodes[nleft->s.suns[7]].sibling;
#endif
            /* Walk downwards making sure all the children point to the new sibling.
             * Note also changes last particle node child. */
            int nn = this_left;
            while(tb.Nodes[nn].f.ChildType == NODE_NODE_TYPE) {
                nn = tb.Nodes[nn].s.suns[7];
#ifdef DEBUG
                if(tb.Nodes[nn].sibling != oldsib)
                    endrun(20, "Not the expected sibling %d != %d\n",tb.Nodes[nn].sibling, oldsib);
#endif
                tb.Nodes[nn].sibling = nleft->sibling;
            }
            /* Mark the right node as now invalid*/
            nright->father = -5;
            /* Next iteration is going to sibling*/
            this_left = nleft->sibling;
            this_right = nright->sibling;
            continue;
        }
        else
            endrun(6, "Nodes %d %d have unexpected type %d %d\n", this_left, this_right, nleft->f.ChildType, nright->f.ChildType);
    }
    return 0;
}


/* create an empty root node  */
int
force_tree_create_topnodes(ForceTree * tree, DomainDecomp * ddecomp)
{
    int nnext = tree->firstnode;       /* index of first free node */
    int i;
    struct NODE *nfreep = &tree->Nodes[nnext];	/* select first node */
    MPI_Comm_rank(MPI_COMM_WORLD, &tree->ThisTask);

    nfreep->len = PartManager->BoxSize*1.001;
    for(i = 0; i < 3; i++)
        nfreep->center[i] = PartManager->BoxSize/2.;
    for(i = 0; i < NMAXCHILD; i++)
        nfreep->s.suns[i] = -1;
    nfreep->s.noccupied = 0;
    nfreep->father = -1;
    nfreep->sibling = -1;
    nfreep->f.TopLevel = 1;
    nfreep->f.InternalTopLevel = 0;
    nfreep->f.DependsOnLocalMass = 0;
    nfreep->f.ChildType = PARTICLE_NODE_TYPE;
    nfreep->f.unused = 0;
    memset(&(nfreep->mom.cofm),0,3*sizeof(MyFloat));
    nfreep->mom.mass = 0;
    nfreep->mom.hmax = 0;
    nnext++;
    /* Set the treenode for this node*/
    ddecomp->TopLeaves[0].treenode = tree->firstnode;
    /* create a set of empty nodes corresponding to the top-level ddecomp
        * grid. We need to generate these nodes first to make sure that we have a
        * complete top-level tree which allows the easy insertion of the
        * pseudo-particles in the right place */
    force_create_node_for_topnode(tree->firstnode, 0, tree->Nodes, ddecomp, 1, 0, 0, 0, &nnext, tree->lastnode);
    return nnext;
}

/*! Constructs the topnodes for the gravitational oct-tree.
 *  Only toptree nodes are made. Particles are not added to the tree.
 *  The indices firstnode....firstnode +nodes-1 reference tree nodes. `Nodes_base'
 *  points to the first tree node, while `nodes' is shifted such that
 *  nodes[firstnode] gives the first tree node. Past tb.lastnode we have the
 *  pseudo particles, whose indices correspond to the indices in TopLeaves.*/
ForceTree
force_tree_top_build(DomainDecomp * ddecomp, const int alloc_high)
{
    /* Allocate memory. Two extra for the first node and for a sentinel*/
    ForceTree tree = force_treeallocate(ddecomp->NTopNodes+2, 0, ddecomp, 0, alloc_high);
    tree.mask = ALLMASK;
    tree.BoxSize = PartManager->BoxSize;
    // message(1, "Building toptree first %d last %d, topnodes %d\n", tree.firstnode, tree.lastnode, ddecomp->NTopNodes);
    force_tree_create_topnodes(&tree, ddecomp);
    return tree;
}

int
force_tree_find_topnode(const double * const pos, const ForceTree * const tree)
{
    /*Walk the main tree until we get something that is a topnode leaf.*/
    int no = tree->firstnode;
    while(tree->Nodes[no].f.InternalTopLevel && tree->Nodes[no].f.TopLevel)
    {
        /* This node has child subnodes: find them.*/
        int subnode = get_subnode(&tree->Nodes[no], pos);
        no = tree->Nodes[no].s.suns[subnode];
    }
#ifdef DEBUG
    if(!tree->Nodes[no].f.TopLevel || tree->Nodes[no].f.InternalTopLevel || no < tree->firstnode)
        endrun(7, "Topnode %d not topleaf, tl = %d itl = %d fn %ld\n", no, tree->Nodes[no].f.TopLevel, tree->Nodes[no].f.InternalTopLevel, tree->firstnode);
#endif
    return no;
}
/*! Does initial creation of the nodes for the gravitational oct-tree.
 * mask is a bitfield: Only types whose bit is set are added.
 **/
void
force_tree_create_nodes(ForceTree * tree, const ActiveParticles * act, int mask, DomainDecomp * ddecomp)
{
    int nnext = force_tree_create_topnodes(tree, ddecomp);

    /* Set up thread-local copies of the topnodes to anchor the subtrees. */
    int ThisTask, j, t;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    const int StartLeaf = ddecomp->Tasks[ThisTask].StartLeaf;
    const int EndLeaf = ddecomp->Tasks[ThisTask].EndLeaf;
    const int nthr = omp_get_max_threads();
    int * topnodes = ta_malloc("topnodes", int, (EndLeaf - StartLeaf) * nthr);
    /* Topnodes for each thread. For tid 0, just use the real tree. Saves copying the tree back.*/
    for(j = 0; j < EndLeaf - StartLeaf; j++)
        topnodes[j] = ddecomp->TopLeaves[j + StartLeaf].treenode;
    /* Other threads need a copy*/
    for(t = 1; t < nthr; t++) {
        for(j = 0; j < EndLeaf - StartLeaf; j++) {
            /* Make a local copy*/
            topnodes[j + t * (EndLeaf - StartLeaf)] = nnext;
            memmove(&tree->Nodes[nnext], &tree->Nodes[topnodes[j]], sizeof(struct NODE));
            nnext++;
        }
    }

/*     double tstart = second(); */
    const int first_free = nnext;
    /* Increment nnext for the threads we are about to initialise.*/
    nnext += NODECACHE_SIZE * nthr;
    /* now we insert all particles */
    int numparticles=0;

    #pragma omp parallel
    {
        int j;
        /* Local topnodes*/
        int tid = omp_get_thread_num();
        const int * const local_topnodes = topnodes + tid * (EndLeaf - StartLeaf);

        /* This implements a small thread-local free Node cache.
         * The cache ensures that Nodes from the same (or close) particles
         * are created close to each other on the Node list and thus
         * helps cache locality. I tried each thread getting a separate
         * part of the tree, and it wasted too much memory. */
        struct NodeCache nc;
        nc.nnext_thread = first_free + tid * NODECACHE_SIZE;
        nc.nrem_thread = NODECACHE_SIZE;

        /* Stores the last-seen node on this thread.
         * Since most particles are close to each other, this should save a number of tree walks.*/
        int this_acc = local_topnodes[0];
        // message(1, "Topnodes %d real %d\n", local_topnodes[0], topnodes[0]);

        /* The default schedule is static with a chunk 1/4 the total.
         * However, particles are sorted by type and then by peano order.
         * Since we need to merge trees, it is advantageous to have all particles
         * spatially close be processed by the same thread. This means that the threads should
         * process particles at a constant offset from the start of the type.
         * We do this with a static schedule. */
        int chnksz = PartManager->NumPart/nthr;
        if(SlotsManager->info[0].enabled && SlotsManager->info[0].size > 0)
            chnksz = SlotsManager->info[0].size/nthr;
        if(chnksz < 1000)
            chnksz = 1000;
        #pragma omp for schedule(static, chnksz) reduction(+: numparticles)
        for(j = 0; j < act->NumActiveParticle; j++)
        {
            /*Can't break from openmp for*/
            if(nc.nnext_thread >= tree->lastnode)
                continue;

            /* Pick the next particle from the active list if there is one*/
            const int i = act->ActiveParticle ? act->ActiveParticle[j] : j;

            /* Do not add types that do not have their mask bit set.*/
            if(!((1<<P[i].Type) & mask)) {
                continue;
            }
            /* Do not add garbage/swallowed particles to the tree*/
            if(P[i].IsGarbage || (P[i].Swallowed && P[i].Type==5))
                continue;

            if(P[i].Mass <= 0)
                endrun(12, "Zero mass particle %d m %g type %d id %ld pos %g %g %g\n", i, P[i].Mass, P[i].Type, P[i].ID, P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
            /*First find the Node for the TopLeaf */
            int cur;
            if(inside_node(&tree->Nodes[this_acc], P[i].Pos)) {
                cur = this_acc;
            } else {
                /* Get the topnode to which a particle belongs. Each local tree
                 * has a local set of treenodes copying the global topnodes, except tid 0
                 * which has the real topnodes.*/
                const int topleaf = P[i].TopLeaf;
                if(topleaf < StartLeaf || topleaf >= EndLeaf)
                    endrun(5, "Bad topleaf %d start %d end %d type %d ID %ld\n", topleaf, StartLeaf, EndLeaf, P[i].Type, P[i].ID);
                //int treenode = ddecomp->TopLeaves[topleaf].treenode;
                cur = local_topnodes[topleaf - StartLeaf];
            }
            numparticles++;
            this_acc = add_particle_to_tree(i, cur, *tree, &nc, &nnext);
        }
        /* The implicit omp-barrier is important here!*/
/*         double tend = second(); */
/*         message(0, "Initial insertion: %.3g ms. First node %d\n", (tend - tstart)*1000, local_topnodes[0]); */

        /* Merge each topnode separately, using a for loop.
         * This wastes threads if NTHREAD > NTOPNODES, but it
         * means only one merge is done per subtree and
         * it requires no locking.*/
        #pragma omp for schedule(static, 1)
        for(j = 0; j < EndLeaf - StartLeaf; j++) {
            int t;
            /* These are the addresses of the real topnodes*/
            const int target = topnodes[j];
            if(nc.nnext_thread >= tree->lastnode)
                continue;
            for(t = 1; t < nthr; t++) {
                const int righttop = topnodes[j + t * (EndLeaf - StartLeaf)];
//                  message(1, "tid = %d i = %d t = %d Merging %d to %d addresses are %lx - %lx end is %lx\n", omp_get_thread_num(), i, t, righttop, target, &tb.Nodes[righttop], &tb.Nodes[target], &tb.Nodes[nnext]);
                if(merge_partial_force_trees(target, righttop, &nc, &nnext, *tree))
                    break;
            }
        }
    }
    tree->NumParticles = numparticles;
    tree->numnodes = nnext - tree->firstnode;
    ta_free(topnodes);
    return;
}

/*! This function recursively creates a set of empty tree nodes which
 *  corresponds to the top-level tree for the ddecomp grid. This is done to
 *  ensure that this top-level tree is always "complete" so that we can easily
 *  associate the pseudo-particles of other CPUs with tree-nodes at a given
 *  level in the tree, even when the particle population is so sparse that
 *  some of these nodes are actually empty.
 */
void force_create_node_for_topnode(int no, int topnode, struct NODE * Nodes, const DomainDecomp * ddecomp, const int bits, const int x, const int y, const int z, int *nextfree, const int lastnode)
{
    int i, j, k;

    /*We reached the leaf of the toptree*/
    const int curdaughter = ddecomp->TopNodes[topnode].Daughter;
    if(curdaughter < 0)
        return;

    for(i = 0; i < 2; i++)
        for(j = 0; j < 2; j++)
            for(k = 0; k < 2; k++)
            {
                int sub = 7 & peano_hilbert_key((x << 1) + i, (y << 1) + j, (z << 1) + k, bits);

                int count = i + 2 * j + 4 * k;

                Nodes[no].s.suns[count] = *nextfree;
                /*We are an internal top level node as we now have a child top level.*/
                Nodes[no].f.InternalTopLevel = 1;
                Nodes[no].f.ChildType = NODE_NODE_TYPE;
                Nodes[no].s.noccupied = NODEFULL;

                /* We create a new leaf node.*/
                init_internal_node(&Nodes[*nextfree], &Nodes[no], count);
                /*Set father of new node*/
                Nodes[*nextfree].father = no;
                /*All nodes here are top level nodes*/
                Nodes[*nextfree].f.TopLevel = 1;

                if(curdaughter + sub >= ddecomp->NTopNodes)
                    endrun(5, "Invalid topnode: daughter %d sub %d > topnodes %d\n", curdaughter, sub, ddecomp->NTopNodes);
                const struct topnode_data curtopnode = ddecomp->TopNodes[curdaughter + sub];
                if(curtopnode.Daughter == -1) {
                    int ThisTask;
                    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
                    ddecomp->TopLeaves[curtopnode.Leaf].treenode = *nextfree;
                    /* We set the first child as a pointer to the topleaf, essentially constructing the pseudoparticles early.
                     * We do not set nocc, so this first child will be over-written on local nodes when we construct the full tree.*/
                    Nodes[*nextfree].s.suns[0] = curtopnode.Leaf + lastnode;
                    if(ddecomp->TopLeaves[curtopnode.Leaf].Task != ThisTask)
                        Nodes[*nextfree].f.ChildType = PSEUDO_NODE_TYPE;
                }

                (*nextfree)++;

                if(*nextfree >= lastnode)
                    endrun(11, "Not enough force nodes to topnode grid: need %d\n",lastnode);
            }
    /* Set sibling on the child*/
    for(j=0; j<7; j++) {
        int chld = Nodes[no].s.suns[j];
        Nodes[chld].sibling = Nodes[no].s.suns[j+1];
    }
    Nodes[Nodes[no].s.suns[7]].sibling = Nodes[no].sibling;
    for(i = 0; i < 2; i++)
        for(j = 0; j < 2; j++)
            for(k = 0; k < 2; k++)
            {
                int sub = 7 & peano_hilbert_key((x << 1) + i, (y << 1) + j, (z << 1) + k, bits);
                int count = i + 2 * j + 4 * k;
                force_create_node_for_topnode(Nodes[no].s.suns[count], ddecomp->TopNodes[topnode].Daughter + sub, Nodes, ddecomp,
                        bits + 1, 2 * x + i, 2 * y + j, 2 * z + k, nextfree, lastnode);
            }

}

int
force_get_father(int no, const ForceTree * tree)
{
    if(no >= tree->firstnode)
        return tree->Nodes[no].father;
    else if(tree->Father)
        return tree->Father[no];
    else
        return -1;
}

static void
add_particle_moment_to_node(struct NODE * pnode, const struct particle_data * const part)
{
    int k;
    pnode->mom.mass += (part->Mass);
    for(k=0; k<3; k++)
        pnode->mom.cofm[k] += (part->Mass * part->Pos[k]);

    /* We do not add active particles to the hmax here.
     * The active particles will have hsml updated in density_postprocess instead, often to a smaller value.*/
    if((part->Type == 0 || part->Type == 5 )&& !is_timebin_active(part->TimeBinHydro, part->Ti_drift))
    {
        int j;
        /* Maximal distance any of the member particles peek out from the side of the node.
         * May be at most hsml, as |Pos - Center| < len/2.*/
        for(j = 0; j < 3; j++) {
            pnode->mom.hmax = DMAX(pnode->mom.hmax, fabs(part->Pos[j] - pnode->center[j]) + part->Hsml - pnode->len/2.);
        }
    }
}

/*Get the sibling of a node, using the suns array. Only to be used in the tree build, before update_node_recursive is called.*/
static int
force_get_sibling(const int sib, const int j, const int * suns)
{
    /* check if we have a sibling on the same level */
    int jj;
    int nextsib = sib;
    for(jj = j + 1; jj < 8; jj++) {
        if(suns[jj] >= 0) {
            nextsib = suns[jj];
            break;
        }
    }
    return nextsib;
}

/* Set the center of mass of the current node*/
static void
force_update_particle_node(int no, const ForceTree * tree)
{
#ifdef DEBUG
    if(tree->Nodes[no].f.ChildType != PARTICLE_NODE_TYPE)
        endrun(3, "force_update_particle_node called on node %d of wrong type!\n", no);
#endif
    int j;
    /*Set the center of mass moments*/
    const double mass = tree->Nodes[no].mom.mass;
    /* Be careful about empty nodes*/
    if(mass > 0) {
        for(j = 0; j < 3; j++)
            tree->Nodes[no].mom.cofm[j] /= mass;
    }
    else {
        for(j = 0; j < 3; j++)
            tree->Nodes[no].mom.cofm[j] = tree->Nodes[no].center[j];
    }
}

/*! this routine determines the multipole moments for a given internal node
 *  and all its subnodes using a recursive computation.  The result is
 *  stored in tb.Nodes in the sequence of this tree-walk.
 *
 *  The function also computes the NextNode and sibling linked lists.
 *  The return value is the current tail of the NextNode linked list.
 *
 *  This function is called recursively using openmp tasks.
 *  We spawn a new task for a fixed number of levels of the tree.
 *
 */
static int
force_update_node_recursive(const int no, const int sib, const int level, const ForceTree * const tree)
{
#ifdef DEBUG
    if(tree->Nodes[no].f.ChildType != NODE_NODE_TYPE)
        endrun(3, "force_update_node_recursive called on node %d of type %d != %d!\n", no, tree->Nodes[no].f.ChildType, NODE_NODE_TYPE);
#endif
    int j;
    int * const suns = tree->Nodes[no].s.suns;

    int childcnt = 0;
    /* Remove any empty children, moving the suns array around
     * so non-empty entries are contiguous at the beginning of the array.
     * This sharply reduces the size of the tree.
     * Also count the node children for thread balancing.*/
    int jj = 0;
    for(j=0; j < 8; j++, jj++) {
        /* Never remove empty top-level nodes so we don't
         * mess up the pseudo-data exchange.
         * This may happen for a pseudo particle host or, in very rare cases,
         * when one of the local domains is empty. */
        while(jj < 8 && !tree->Nodes[suns[jj]].f.TopLevel &&
            tree->Nodes[suns[jj]].f.ChildType == PARTICLE_NODE_TYPE &&
            tree->Nodes[suns[jj]].s.noccupied == 0) {
                    jj++;
        }
        if(jj < 8)
            suns[j] = suns[jj];
        else
            suns[j] = -1;
        if(suns[j] >= 0 && tree->Nodes[suns[j]].f.ChildType == NODE_NODE_TYPE)
            childcnt++;
    }

    /*First do the children*/
    for(j = 0; j < 8; j++)
    {
        const int p = suns[j];
        /*Empty slot*/
        if(p < 0)
            continue;
        const int nextsib = force_get_sibling(sib, j, suns);
        /* This is set in create_nodes but needed because we may remove empty nodes above.*/
        tree->Nodes[p].sibling = nextsib;
        /* Nodes containing particles or pseudo-particles*/
        if(tree->Nodes[p].f.ChildType == PARTICLE_NODE_TYPE)
            force_update_particle_node(p, tree);
        if(tree->Nodes[p].f.ChildType == NODE_NODE_TYPE) {
            /* Don't spawn a new task if we are deep enough that we already spawned a lot.*/
            if(childcnt > 1 && level < 512) {
                const int newlevel = level * childcnt;
                /* Firstprivate for const variables should be optimised out*/
                #pragma omp task default(none) firstprivate(nextsib, p, newlevel, tree)
                force_update_node_recursive(p, nextsib, newlevel, tree);
            }
            else
                force_update_node_recursive(p, nextsib, level, tree);
        }
    }

    /*Make sure all child nodes are done*/
    #pragma omp taskwait

    /*Now we do the moments*/
    for(j = 0; j < 8; j++)
    {
        const int p = suns[j];
        if(p < 0)
            continue;
        tree->Nodes[no].mom.mass += (tree->Nodes[p].mom.mass);
        tree->Nodes[no].mom.cofm[0] += (tree->Nodes[p].mom.mass * tree->Nodes[p].mom.cofm[0]);
        tree->Nodes[no].mom.cofm[1] += (tree->Nodes[p].mom.mass * tree->Nodes[p].mom.cofm[1]);
        tree->Nodes[no].mom.cofm[2] += (tree->Nodes[p].mom.mass * tree->Nodes[p].mom.cofm[2]);
        if(tree->Nodes[p].mom.hmax > tree->Nodes[no].mom.hmax)
            tree->Nodes[no].mom.hmax = tree->Nodes[p].mom.hmax;
    }

    /*Set the center of mass moments*/
    const double mass = tree->Nodes[no].mom.mass;
    /* In principle all the children could be pseudo-particles*/
    if(mass > 0) {
        tree->Nodes[no].mom.cofm[0] /= mass;
        tree->Nodes[no].mom.cofm[1] /= mass;
        tree->Nodes[no].mom.cofm[2] /= mass;
    }

    return -1;
}

/*! This routine determines the multipole moments for a given internal node
 *  and all its subnodes in parallel, assigning the recursive algorithm to different threads using openmp's task api.
 *  The result is stored in tb.Nodes in the sequence of this tree-walk.
 *
 * - A new task is spawned from each  down from each local topleaf. Local topleaves are used
 * so that we do not waste time trying moment calculation with pseudoparticles.
 * - Each internal node found at that level is added to a list, together with its sibling.
 * - Each node in this list then has the recursive moment calculation called on it.
 * Note: If the tree is very unbalanced and one branch much deeper than the others, this will not be efficient.
 * - Once each tree's recursive moment is generated in parallel, the tail value from each recursion is stored, and the node marked as done.
 * - A final recursive moment calculation is run in serial for the top 3 levels of the tree. When it encounters one of the pre-computed nodes, it
 * searches the list of pre-computed tail values to set the next node as if it had recursed and continues.
 */
void force_update_node_parallel(const ForceTree * const tree, const DomainDecomp * const ddecomp)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

#pragma omp parallel
#pragma omp single nowait
    {
        int i;
        for(i = ddecomp->Tasks[ThisTask].StartLeaf; i < ddecomp->Tasks[ThisTask].EndLeaf; i ++) {
            const int no = ddecomp->TopLeaves[i].treenode;
            /* Set local mass dependence*/
            tree->Nodes[no].f.DependsOnLocalMass = 1;
            /* Nodes containing other nodes: the overwhelmingly likely case.*/
            if(tree->Nodes[no].f.ChildType == NODE_NODE_TYPE) {
                #pragma omp task default(none) firstprivate(no, tree)
                force_update_node_recursive(no, tree->Nodes[no].sibling, 1, tree);
            }
            else if(tree->Nodes[no].f.ChildType == PARTICLE_NODE_TYPE)
                force_update_particle_node(no, tree);
            else if(tree->Nodes[no].f.ChildType == PSEUDO_NODE_TYPE)
                endrun(5, "Error, found pseudo node %d but domain entry %d says on task %d\n", no, i, ThisTask);
        }
    }
}

struct topleaf_momentsdata
{
    MyFloat s[3];
    MyFloat mass;
    MyFloat hmax;
};

/*! This function communicates the values of the multipole moments of the
 *  top-level tree-nodes of the ddecomp grid.  This data can then be used to
 *  update the pseudo-particles on each CPU accordingly.
 */
static void force_exchange_pseudodata(const ForceTree * const tree, const DomainDecomp * const ddecomp)
{
    int i;

    struct topleaf_momentsdata * TopLeafMoments = (struct topleaf_momentsdata *) mymalloc("TopLeafMoments", ddecomp->NTopLeaves * sizeof(TopLeafMoments[0]));

    #pragma omp parallel for
    for(i = ddecomp->Tasks[tree->ThisTask].StartLeaf; i < ddecomp->Tasks[tree->ThisTask].EndLeaf; i ++) {
        int no = ddecomp->TopLeaves[i].treenode;
        if(ddecomp->TopLeaves[i].Task != tree->ThisTask)
            endrun(131231231, "TopLeaf %d Task table is corrupted: task is %d\n", i, ddecomp->TopLeaves[i].Task);
        /* read out the multipole moments from the local base cells */
        TopLeafMoments[i].s[0] = tree->Nodes[no].mom.cofm[0];
        TopLeafMoments[i].s[1] = tree->Nodes[no].mom.cofm[1];
        TopLeafMoments[i].s[2] = tree->Nodes[no].mom.cofm[2];
        TopLeafMoments[i].mass = tree->Nodes[no].mom.mass;
        TopLeafMoments[i].hmax = tree->Nodes[no].mom.hmax;
    }

    /* share the pseudo-particle data across CPUs */
    int NTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);

    int * recvcounts = (int *) mymalloc("recvcounts", sizeof(int) * NTask);
    int * recvoffset = (int *) mymalloc("recvoffset", sizeof(int) * NTask);
    int recvTask;

    for(recvTask = 0; recvTask < NTask; recvTask++)
    {
        recvoffset[recvTask] = ddecomp->Tasks[recvTask].StartLeaf * sizeof(TopLeafMoments[0]);
        recvcounts[recvTask] = (ddecomp->Tasks[recvTask].EndLeaf - ddecomp->Tasks[recvTask].StartLeaf) * sizeof(TopLeafMoments[0]);
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
            &TopLeafMoments[0], recvcounts, recvoffset,
            MPI_BYTE, MPI_COMM_WORLD);

    myfree(recvoffset);
    myfree(recvcounts);

    int ta;
    #pragma omp parallel for
    for(ta = 0; ta < NTask; ta++) {
        if(ta == tree->ThisTask)
            continue; /* bypass ThisTask since it is already up to date */

        for(i = ddecomp->Tasks[ta].StartLeaf; i < ddecomp->Tasks[ta].EndLeaf; i ++) {
            int no = ddecomp->TopLeaves[i].treenode;
            tree->Nodes[no].mom.cofm[0] = TopLeafMoments[i].s[0];
            tree->Nodes[no].mom.cofm[1] = TopLeafMoments[i].s[1];
            tree->Nodes[no].mom.cofm[2] = TopLeafMoments[i].s[2];
            tree->Nodes[no].mom.mass = TopLeafMoments[i].mass;
            tree->Nodes[no].mom.hmax = TopLeafMoments[i].hmax;
         }
    }
    myfree(TopLeafMoments);
}


/*! This function updates the top-level tree after the multipole moments of
 *  the pseudo-particles have been updated.
 */
void
force_treeupdate_pseudos(const int no, const int level, const ForceTree * const tree)
{
    /* This happens if we have a trivial domain with only one entry*/
    if(!tree->Nodes[no].f.InternalTopLevel)
        return;

    int j;

    /* since we are dealing with top-level nodes, we know that there are 8 consecutive daughter nodes */
    for(j = 0; j < 8; j++)
    {
        const int p = tree->Nodes[no].s.suns[j];

        /*This may not happen as we are an internal top level node*/
        if(p < tree->firstnode || p >= tree->lastnode)
            endrun(6767, "Updating pseudos: %d -> %d which is not an internal node between %ld and %ld\n",no, p, tree->firstnode, tree->lastnode);
#ifdef DEBUG
        /* Check we don't move to another part of the tree*/
        if(tree->Nodes[p].father != no)
            endrun(6767, "Tried to update toplevel node %d with parent %d != expected %d\n", p, tree->Nodes[p].father, no);
#endif

        if(tree->Nodes[p].f.InternalTopLevel) {
            if(level < 512) {
                #pragma omp task default(none) firstprivate(p, level, tree)
                force_treeupdate_pseudos(p, level*8, tree);
            }
            else {
                force_treeupdate_pseudos(p, level, tree);
            }
        }
    }
    /* Zero the moments*/
    tree->Nodes[no].mom.mass = 0;
    tree->Nodes[no].mom.cofm[0] = 0;
    tree->Nodes[no].mom.cofm[1] = 0;
    tree->Nodes[no].mom.cofm[2] = 0;
    tree->Nodes[no].mom.hmax = 0;

    /*Make sure all child nodes are done*/
    #pragma omp taskwait

    for(j = 0; j < 8; j++)
    {
        const int p = tree->Nodes[no].s.suns[j];

        tree->Nodes[no].mom.mass += (tree->Nodes[p].mom.mass);
        tree->Nodes[no].mom.cofm[0] += (tree->Nodes[p].mom.mass * tree->Nodes[p].mom.cofm[0]);
        tree->Nodes[no].mom.cofm[1] += (tree->Nodes[p].mom.mass * tree->Nodes[p].mom.cofm[1]);
        tree->Nodes[no].mom.cofm[2] += (tree->Nodes[p].mom.mass * tree->Nodes[p].mom.cofm[2]);

        if(tree->Nodes[p].mom.hmax > tree->Nodes[no].mom.hmax)
            tree->Nodes[no].mom.hmax = tree->Nodes[p].mom.hmax;
        if(tree->Nodes[p].f.DependsOnLocalMass)
            tree->Nodes[no].f.DependsOnLocalMass = 1;
    }

    if(tree->Nodes[no].mom.mass)
    {
        tree->Nodes[no].mom.cofm[0] /= tree->Nodes[no].mom.mass;
        tree->Nodes[no].mom.cofm[1] /= tree->Nodes[no].mom.mass;
        tree->Nodes[no].mom.cofm[2] /= tree->Nodes[no].mom.mass;
    }
    else
    {
        tree->Nodes[no].mom.cofm[0] = tree->Nodes[no].center[0];
        tree->Nodes[no].mom.cofm[1] = tree->Nodes[no].center[1];
        tree->Nodes[no].mom.cofm[2] = tree->Nodes[no].center[2];
    }
}

/* Update the hmax in the parent node of the particle p_i*/
void
update_tree_hmax_father(const ForceTree * const tree, const int p_i, const double Pos[3], const double Hsml)
{
    if(!tree->Father)
        endrun(4, "Father not allocated in tree_hmax_father\n");
    const int no = tree->Father[p_i];
#ifdef DEBUG
    if(no < 0)
        endrun(5, "Father for particle %d pos %g %g %g hsml %g not initialised, likely not in tree\n", p_i, Pos[0], Pos[1], Pos[2], Hsml);
#endif
    struct NODE * node = &tree->Nodes[no];
    /* How much does this particle peek beyond this node?
        * Note len does not change so we can read it without a lock or atomic. */

    MyFloat newhmax = 0;
    int j;
    for(j = 0; j < 3; j++)
        newhmax = DMAX(newhmax, fabs(Pos[j] - node->center[j]) + Hsml - node->len/2.);

    MyFloat readhmax;
    #pragma omp atomic read
    readhmax = node->mom.hmax;

    do {
        if (newhmax <= readhmax)
            break;
        /* Swap in the new hmax only if the old one hasn't changed. */
    } while(!__atomic_compare_exchange(&(node->mom.hmax), &readhmax, &newhmax, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
}

/*! This function updates the hmax-values in tree nodes that hold SPH
 *  particles. This is no longer called. The density() code updates hmax for the parent node
 *  of each active particle.
 *
 *  The purpose of the hmax node is for a symmetric treewalk (currently only the hydro).
 *  Particles where P[i].Pos + Hsml pokes beyond the exterior of the tree node may mean
 *  that a tree node should be included when it would normally be culled. Therefore we don't really
 *  want hmax, we want the maximum amount P[i].Pos + Hsml pokes beyond the tree node.
 */
void force_update_hmax(ActiveParticles * act, ForceTree * tree, DomainDecomp * ddecomp)
{
    int i;

    if(!tree->Father)
        endrun(1, "Father array not allocated, needed for hmax!\n");

    if(!(tree->mask & GASMASK))
        endrun(1, "tree mask is %d, does not contain gas (%d)! Cannot compute hmax.\n", tree->mask, GASMASK);

    int tree_has_bh = 0;
    if(tree->mask & BHMASK)
        tree_has_bh = 1;

    /* Adjust the base particle containing nodes*/
    #pragma omp parallel for
    for(i = 0; i < act->NumActiveParticle; i++)
    {
        const int p_i = act->ActiveParticle ? act->ActiveParticle[i] : i;
        /* This is only gas particles or BH.*/
        struct particle_data * pp = &act->Particles[p_i];
        if((pp->Type != 0 && pp->Type != 5) || pp->IsGarbage || pp->Swallowed)
            continue;
        /* Can't do tree for BH if BH not in tree*/
        if(!tree_has_bh && pp->Type == 5)
            continue;

        update_tree_hmax_father(tree, p_i, pp->Pos, pp->Hsml);
    }

    /* Calculate moments to propagate everything upwards. */
    force_tree_calc_moments(tree, ddecomp);

    walltime_measure("/SPH/HmaxUpdate");
    int64_t totnumparticles;
    MPI_Reduce(&tree->NumParticles, &totnumparticles, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
    message(0, "Root hmax: %lg Tree Mean IPS: %lg\n", tree->Nodes[tree->firstnode].mom.hmax, tree->BoxSize / cbrt(totnumparticles));
}

/*! This function allocates the memory used for storage of the tree and of
 *  auxiliary arrays needed for tree-walk and link-lists.  Usually,
 *  maxnodes approximately equal to 0.7*maxpart is sufficient to store the
 *  tree for up to maxpart particles.
 */
ForceTree force_treeallocate(const int64_t maxnodes, const int64_t maxpart, const DomainDecomp * ddecomp, const int alloc_father, const int alloc_high)
{
    ForceTree tb = {0};

    if(alloc_father) {
        tb.Father = (int *) mymalloc("Father", maxpart * sizeof(int));
        tb.nfather = maxpart;
#ifdef DEBUG
        memset(tb.Father, -1, maxpart * sizeof(int));
#endif
    }
    if(alloc_high)
        tb.Nodes_base = (struct NODE *) mymalloc2("Nodes_base", (maxnodes + 1) * sizeof(struct NODE));
    else
        tb.Nodes_base = (struct NODE *) mymalloc("Nodes_base", (maxnodes + 1) * sizeof(struct NODE));
#ifdef DEBUG
    memset(tb.Nodes_base, -1, (maxnodes + 1) * sizeof(struct NODE));
#endif
    tb.firstnode = maxpart;
    tb.lastnode = tb.firstnode + maxnodes;
    if(tb.lastnode >= (1L<<30) + (1L<<29))
        endrun(5, "Size of tree overflowed for maxpart = %ld, maxnodes = %ld!\n", maxpart, maxnodes);
    tb.numnodes = 0;
    tb.Nodes = tb.Nodes_base - tb.firstnode;
    tb.tree_allocated_flag = 1;
    tb.NTopLeaves = ddecomp->NTopLeaves;
    tb.TopLeaves = ddecomp->TopLeaves;
    return tb;
}

/*! This function frees the memory allocated for the tree, i.e. it frees
 *  the space allocated by the function force_treeallocate().
 */
void force_tree_free(ForceTree * tree)
{
    if(!force_tree_allocated(tree))
        return;
    myfree(tree->Nodes_base);
    if(tree->Father)
        myfree(tree->Father);
    /* Zero everything, especially the allocation flag*/
    memset(tree, 0, sizeof(ForceTree));
    tree->tree_allocated_flag = 0;
}
