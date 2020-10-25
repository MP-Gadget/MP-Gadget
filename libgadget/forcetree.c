#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "slotsmanager.h"
#include "partmanager.h"
#include "domain.h"
#include "forcetree.h"
#include "checkpoint.h"
#include "walltime.h"
#include "utils.h"

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
    /*!< flags the particle species which will be excluded from the tree if the HybridNuGrav parameter is set.*/
    int FastParticleType;
} ForceTreeParams;

void
init_forcetree_params(const int FastParticleType)
{
    ForceTreeParams.TreeAllocFactor = 0.7;
    ForceTreeParams.FastParticleType = FastParticleType;
}

static ForceTree
force_tree_build(int npart, DomainDecomp * ddecomp, const double BoxSize, const int HybridNuGrav, const int DoMoments, const char * EmergencyOutputDir);

/*Next three are not static as tested.*/
int
force_tree_create_nodes(const ForceTree tb, const int npart, DomainDecomp * ddecomp, const double BoxSize, const int HybridNuGrav);

ForceTree
force_treeallocate(int maxnodes, int maxpart, DomainDecomp * ddecomp);

void
force_update_node_parallel(const ForceTree * tree);

static void
force_treeupdate_pseudos(int no, const ForceTree * tree);

static void
force_create_node_for_topnode(int no, int topnode, struct NODE * Nodes, const DomainDecomp * ddecomp, int bits, int x, int y, int z, int *nextfree, const int lastnode);

static void
force_exchange_pseudodata(ForceTree * tree, const DomainDecomp * ddecomp);

static void
force_insert_pseudo_particles(const ForceTree * tree, const DomainDecomp * ddecomp);

static void
add_particle_moment_to_node(struct NODE * pnode, int i);

#ifdef DEBUG
/* Walk the constructed tree, validating sibling and nextnode as we go*/
static void force_validate_nextlist(const ForceTree * tree)
{
    int no = tree->firstnode;
    while(no != -1)
    {
        struct NODE * current = &tree->Nodes[no];
        if(current->sibling != -1 && !node_is_node(current->sibling, tree))
            endrun(5, "Node %d (type %d) has sibling %d next %d father %d first %d final %d last %d ntop %d\n", no, current->f.ChildType, current->sibling, current->s.suns[0], current->father, tree->firstnode, tree->firstnode + tree->numnodes, tree->lastnode, tree->NTopLeaves);

        if(current->f.ChildType == PSEUDO_NODE_TYPE) {
            /* pseudo particle: nextnode should be a pseudo particle, sibling should be a node. */
            if(!node_is_pseudo_particle(current->s.suns[0], tree))
                endrun(5, "Pseudo Node %d has next node %d sibling %d father %d first %d final %d last %d ntop %d\n", no, current->s.suns[0], current->sibling, current->father, tree->firstnode, tree->firstnode + tree->numnodes, tree->lastnode, tree->NTopLeaves);
        }
        else if(current->f.ChildType == NODE_NODE_TYPE) {
            /* Next node should be another node */
            if(!node_is_node(current->s.suns[0], tree))
                endrun(5, "Node Node %d has next node which is particle %d sibling %d father %d first %d final %d last %d ntop %d\n", no, current->s.suns[0], current->sibling, current->father, tree->firstnode, tree->firstnode + tree->numnodes, tree->lastnode, tree->NTopLeaves);
            no = current->s.suns[0];
            continue;
        }
        no = current->sibling;
    }
    /* Every node should have a valid father: collect those that do not.*/
    for(no = tree->firstnode; no < tree->firstnode + tree->numnodes; no++)
    {
        if(!node_is_node(tree->Nodes[no].father, tree) && tree->Nodes[no].father != -1) {
            struct NODE *current = &tree->Nodes[no];
            message(1, "Danger! no %d has father %d, next %d sib %d, (ptype = %d) len %g center (%g %g %g) mass %g cofm %g %g %g TL %d DLM %d ITL %d nocc %d suns %d %d %d %d\n", no, current->father, current->s.suns[0], current->sibling, current->f.ChildType,
                current->len, current->center[0], current->center[1], current->center[2],
                current->mom.mass, current->mom.cofm[0], current->mom.cofm[1], current->mom.cofm[2],
                current->f.TopLevel, current->f.DependsOnLocalMass, current->f.InternalTopLevel, current->s.noccupied,
                current->s.suns[0], current->s.suns[1], current->s.suns[2], current->s.suns[3]);
        }
    }

}
#endif

static int
force_tree_eh_slots_fork(EIBase * event, void * userdata)
{
    /* after a fork, we will attach the new particle to the force tree. */
    EISlotsFork * ev = (EISlotsFork*) event;
    int parent = ev->parent;
    int child = ev->child;
    ForceTree * tree = (ForceTree * ) userdata;
    int no = force_get_father(parent, tree);
    struct NODE * nop = &tree->Nodes[no];
    /* FIXME: We lose particles if the node is full.
     * At the moment this does not matter, because
     * the only new particles are stars, which do not
     * participate in the SPH tree walk.*/
    if(nop->s.noccupied < NMAXCHILD) {
       nop->s.suns[nop->s.noccupied] = child;
       nop->s.Types += P[child].Type << (3*nop->s.noccupied);
       nop->s.noccupied++;
    }
    tree->Father[child] = no;
    return 0;
}

int
force_tree_allocated(const ForceTree * tree)
{
    return tree->tree_allocated_flag;
}

void
force_tree_rebuild(ForceTree * tree, DomainDecomp * ddecomp, const double BoxSize, const int HybridNuGrav, const int DoMoments, const char * EmergencyOutputDir)
{
    MPIU_Barrier(MPI_COMM_WORLD);
    message(0, "Tree construction.  (presently allocated=%g MB)\n", mymalloc_usedbytes() / (1024.0 * 1024.0));

    if(force_tree_allocated(tree)) {
        force_tree_free(tree);
    }
    walltime_measure("/Misc");

    *tree = force_tree_build(PartManager->NumPart, ddecomp, BoxSize, HybridNuGrav, DoMoments, EmergencyOutputDir);

    event_listen(&EventSlotsFork, force_tree_eh_slots_fork, tree);
    walltime_measure("/Tree/Build/Moments");

    message(0, "Tree constructed (moments: %d). First node %d, number of nodes %d, first pseudo %d. NTopLeaves %d\n",
            tree->moments_computed_flag, tree->firstnode, tree->numnodes, tree->lastnode, tree->NTopLeaves);
    MPIU_Barrier(MPI_COMM_WORLD);
}

/*! Constructs the gravitational oct-tree.
 *
 *  The index convention for accessing tree nodes is the following: the
 *  indices 0...NumPart-1 reference single particles, the indices
 *  PartManager->MaxPart.... PartManager->MaxPart+nodes-1 reference tree nodes. `Nodes_base'
 *  points to the first tree node, while `nodes' is shifted such that
 *  nodes[PartManager->MaxPart] gives the first tree node. Finally, node indices
 *  with values 'PartManager->MaxPart + tb.lastnode' and larger indicate "pseudo
 *  particles", i.e. multipole moments of top-level nodes that lie on
 *  different CPUs. If such a node needs to be opened, the corresponding
 *  particle must be exported to that CPU. */
ForceTree force_tree_build(int npart, DomainDecomp * ddecomp, const double BoxSize, const int HybridNuGrav, const int DoMoments, const char * EmergencyOutputDir)
{
    ForceTree tree;

    int TooManyNodes = 0;

    do
    {
        int maxnodes = ForceTreeParams.TreeAllocFactor * PartManager->NumPart + ddecomp->NTopNodes;
        /* Allocate memory. */
        tree = force_treeallocate(maxnodes, PartManager->MaxPart, ddecomp);

        tree.BoxSize = BoxSize;
        tree.numnodes = force_tree_create_nodes(tree, npart, ddecomp, BoxSize, HybridNuGrav);
        if(tree.numnodes >= tree.lastnode - tree.firstnode)
        {
            message(1, "Not enough tree nodes (%d) for %d particles. Created %d\n", maxnodes, npart, tree.numnodes);
            force_tree_free(&tree);
            message(1, "TreeAllocFactor from %g to %g\n", ForceTreeParams.TreeAllocFactor, ForceTreeParams.TreeAllocFactor*1.15);
            ForceTreeParams.TreeAllocFactor *= 1.15;
            if(ForceTreeParams.TreeAllocFactor > 3.0) {
                TooManyNodes = 1;
                break;
            }
        }
    }
    while(tree.numnodes >= tree.lastnode - tree.firstnode);

    if(MPIU_Any(TooManyNodes, MPI_COMM_WORLD)) {
        if(EmergencyOutputDir)
            dump_snapshot("FORCETREE-DUMP", EmergencyOutputDir);
        endrun(2, "Required too many nodes, snapshot dumped\n");
    }
    walltime_measure("/Tree/Build/Nodes");
#ifdef DEBUG
    force_validate_nextlist(&tree);
#endif
    /* insert the pseudo particles that represent the mass distribution of other ddecomps */
    force_insert_pseudo_particles(&tree, ddecomp);
#ifdef DEBUG
    force_validate_nextlist(&tree);
#endif

    tree.moments_computed_flag = 0;

    if(DoMoments) {
        /* now compute the multipole moments recursively */
        force_update_node_parallel(&tree);
#ifdef DEBUG
        force_validate_nextlist(&tree);
#endif

        /* Exchange the pseudo-data*/
        force_exchange_pseudodata(&tree, ddecomp);

        force_treeupdate_pseudos(PartManager->MaxPart, &tree);
        tree.moments_computed_flag = 1;
        tree.hmax_computed_flag = 1;
    }
    tree.Nodes_base = myrealloc(tree.Nodes_base, (tree.numnodes +1) * sizeof(struct NODE));

    /*Update the oct-tree struct so it knows about the memory change*/
    tree.Nodes = tree.Nodes_base - tree.firstnode;
#ifdef DEBUG
        force_validate_nextlist(&tree);
#endif
    return tree;
}

/* Get the subnode for a given particle and parent node.
 * This splits a parent node into 8 subregions depending on the particle position.
 * node is the parent node to split, p_i is the index of the particle we
 * are currently inserting
 * Returns a value between 0 and 7.
 * */
int get_subnode(const struct NODE * node, const int p_i)
{
    /*Loop is unrolled to help out the compiler,which normally only manages it at -O3*/
     return (P[p_i].Pos[0] > node->center[0]) +
            ((P[p_i].Pos[1] > node->center[1]) << 1) +
            ((P[p_i].Pos[2] > node->center[2]) << 2);
}

/*Check whether a particle is inside the volume covered by a node,
 * by checking whether each dimension is close enough to center (L1 metric).*/
static inline int inside_node(const struct NODE * node, const int p_i)
{
    /*One can also use a loop, but the compiler unrolls it only at -O3,
     *so this is a little faster*/
    int inside =
        (fabs(2*(P[p_i].Pos[0] - node->center[0])) <= node->len) *
        (fabs(2*(P[p_i].Pos[1] - node->center[1])) <= node->len) *
        (fabs(2*(P[p_i].Pos[2] - node->center[2])) <= node->len);
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
    nfreep->s.Types = 0;
    memset(&(nfreep->mom.cofm),0,3*sizeof(MyFloat));
    nfreep->mom.mass = 0;
    nfreep->mom.hmax = 0;
}

/* Size of the free Node thread cache.
 * 12 8-node rows (works out at 8kB) was found
 * to be optimal for an Intel skylake with 4 threads.*/
#define NODECACHE_SIZE (8*12)

/*Structure containing thread-local parameters of the tree build*/
struct NodeCache {
    int nnext_thread;
    int nrem_thread;
};

/*Get a pointer to memory for a free node, from our node cache.
 * If there is no memory left, return NULL.*/
int get_freenode(int * nnext, struct NodeCache *nc)
{
    /*Get memory for an extra node from our cache.*/
    if(nc->nrem_thread == 0) {
        nc->nnext_thread = atomic_fetch_and_add(nnext, NODECACHE_SIZE);
        nc->nrem_thread = NODECACHE_SIZE;
    }
    const int ninsert = (nc->nnext_thread)++;
    (nc->nrem_thread)--;
    return ninsert;
}

/* Add a particle to a node in a known empty location.
 * Parent is assumed to be locked.*/
static int
modify_internal_node(int parent, int subnode, int p_toplace, const ForceTree tb, const int HybridNuGrav)
{
    tb.Father[p_toplace] = parent;
    tb.Nodes[parent].s.suns[subnode] = p_toplace;
    /* Encode the type in the Types array*/
    tb.Nodes[parent].s.Types += P[p_toplace].Type << (3*subnode);
    if(!HybridNuGrav || P[p_toplace].Type != ForceTreeParams.FastParticleType)
        add_particle_moment_to_node(&tb.Nodes[parent], p_toplace);
    return 0;
}


/* Create a new layer of nodes beneath the current node, and place the particle.
 * Must have node lock.*/
static int
create_new_node_layer(int firstparent, int p_toplace,
        const int HybridNuGrav, const ForceTree tb, int *nnext, struct NodeCache *nc)
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
        for(i=0; i<8; i++) {
            /* Get memory for an extra node from our cache.*/
            newsuns[i] = get_freenode(nnext, nc);
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
                return 1;
            }
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
            int subnode = get_subnode(nprnt, oldsuns[i]);
            int child = newsuns[subnode];
            struct NODE * nchild = &tb.Nodes[child];
            modify_internal_node(child, nchild->s.noccupied, oldsuns[i], tb, HybridNuGrav);
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
        int subnode = get_subnode(nprnt, p_toplace);
        int child = nprnt->s.suns[subnode];
        struct NODE * nchild = &tb.Nodes[child];
        if(nchild->s.noccupied < NMAXCHILD) {
            modify_internal_node(child, nchild->s.noccupied, p_toplace, tb, HybridNuGrav);
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
            tb.Nodes[child].s.noccupied = (1<<16);
            parent = child;
        }
    } while(1);

    /* A new node is created. Mark the (original) parent as an internal node with node children.
     * This goes last so that we don't access the child before it is constructed.*/
    tb.Nodes[firstparent].f.ChildType = NODE_NODE_TYPE;
    #pragma omp atomic write
    tb.Nodes[firstparent].s.noccupied = (1<<16);
    return 0;
}

/*! Does initial creation of the nodes for the gravitational oct-tree.
 **/
int force_tree_create_nodes(const ForceTree tb, const int npart, DomainDecomp * ddecomp, const double BoxSize, const int HybridNuGrav)
{
    int i;
    int nnext = tb.firstnode;		/* index of first free node */

    /* create an empty root node  */
    {
        struct NODE *nfreep = &tb.Nodes[nnext];	/* select first node */

        nfreep->len = BoxSize*1.001;
        for(i = 0; i < 3; i++)
            nfreep->center[i] = BoxSize/2.;
        for(i = 0; i < NMAXCHILD; i++)
            nfreep->s.suns[i] = -1;
        nfreep->s.Types = 0;
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
        /* create a set of empty nodes corresponding to the top-level ddecomp
         * grid. We need to generate these nodes first to make sure that we have a
         * complete top-level tree which allows the easy insertion of the
         * pseudo-particles in the right place */
        force_create_node_for_topnode(tb.firstnode, 0, tb.Nodes, ddecomp, 1, 0, 0, 0, &nnext, tb.lastnode);
    }

    /* This implements a small thread-local free Node cache.
     * The cache ensures that Nodes from the same (or close) particles
     * are created close to each other on the Node list and thus
     * helps cache locality. In my tests without this list the
     * reduction in cache performance destroyed the benefit of
     * parallelizing this loop!*/
    struct NodeCache nc;
    nc.nnext_thread = nnext;
    nc.nrem_thread = 0;
    /* Stores the last-seen node on this thread.
     * Since most particles are close to each other, this should save a number of tree walks.*/
    int this_acc = tb.firstnode;
    /*Initialise some spinlocks off*/
    struct SpinLocks * spin = init_spinlocks(tb.lastnode - tb.firstnode);

    /* now we insert all particles */
    #pragma omp parallel for firstprivate(nc, this_acc)
    for(i = 0; i < npart; i++)
    {
        /*Can't break from openmp for*/
        if(nc.nnext_thread >= tb.lastnode)
            continue;

        /* Do not add garbage/swallowed particles to the tree*/
        if(P[i].IsGarbage || (P[i].Swallowed && P[i].Type==5))
            continue;

        /*First find the Node for the TopLeaf */
        int this;
        if(inside_node(&tb.Nodes[this_acc], i)) {
            this = this_acc;
        } else {
            const int topleaf = domain_get_topleaf(P[i].Key, ddecomp);
            this = ddecomp->TopLeaves[topleaf].treenode;
        }
        int child;
        int nocc;

        /*Walk the main tree until we get something that isn't an internal node.*/
        do
        {
            /*No lock needed: if we have an internal node here it will be stable*/
            #pragma omp atomic read
            nocc = tb.Nodes[this].s.noccupied;

            /* This node still has space for a particle (or needs conversion)*/
            if(nocc < (1 << 16))
                break;

            /* This node has child subnodes: find them.*/
            int subnode = get_subnode(&tb.Nodes[this], i);
            /*No lock needed: if we have an internal node here it will be stable*/
            child = tb.Nodes[this].s.suns[subnode];

            if(child > tb.lastnode || child < tb.firstnode)
                endrun(1,"Corruption in tree build: N[%d].[%d] = %d > lastnode (%d)\n",this, subnode, child, tb.lastnode);
            this = child;
        }
        while(child >= tb.firstnode);

        /*Now lock this node.*/
        lock_spinlock(this-tb.firstnode, spin);
        /* We have a guaranteed spot.*/
        nocc = atomic_fetch_and_add(&tb.Nodes[this].s.noccupied, 1);

        /* Check whether there is now a new layer of nodes and if so walk down until there isn't.*/
        if(nocc >= (1<<16)) {
            /* This node has child subnodes: find them.*/
            int subnode = get_subnode(&tb.Nodes[this], i);
            child = tb.Nodes[this].s.suns[subnode];
            while(child >= tb.firstnode)
            {
                /*Move the lock to the child*/
                lock_spinlock(child-tb.firstnode, spin);
                unlock_spinlock(this-tb.firstnode, spin);
                this = child;

                /*No lock needed: if we have an internal node here it will be stable*/
                #pragma omp atomic read
                nocc = tb.Nodes[this].s.noccupied;
                /* This node still has space for a particle (or needs conversion)*/
                if(nocc < (1 << 16))
                    break;

                /* This node has child subnodes: find them.*/
                subnode = get_subnode(&tb.Nodes[this], i);
                /*No lock needed: if we have an internal node here it will be stable*/
                child = tb.Nodes[this].s.suns[subnode];
            }
            /* Get the free spot under the lock.*/
            nocc = atomic_fetch_and_add(&tb.Nodes[this].s.noccupied, 1);
        }

        /*Update last-used cache*/
        this_acc = this;

        /* Now we have something that isn't an internal node, and we have a lock on it,
         * so we know it won't change. We can place the particle! */
        if(nocc < NMAXCHILD)
            modify_internal_node(this, nocc, i, tb, HybridNuGrav);
        /* In this case we need to create a new layer of nodes beneath this one*/
        else if(nocc < 1<<16)
            create_new_node_layer(this, i, HybridNuGrav, tb, &nnext, &nc);
        else
            endrun(2, "Tried to convert already converted node %d with nocc = %d\n", this, nocc);

        /*Unlock the parent*/
        unlock_spinlock(this - tb.firstnode, spin);
    }
    free_spinlocks(spin);

    return nnext - tb.firstnode;
}

/*! This function recursively creates a set of empty tree nodes which
 *  corresponds to the top-level tree for the ddecomp grid. This is done to
 *  ensure that this top-level tree is always "complete" so that we can easily
 *  associate the pseudo-particles of other CPUs with tree-nodes at a given
 *  level in the tree, even when the particle population is so sparse that
 *  some of these nodes are actually empty.
 */
void force_create_node_for_topnode(int no, int topnode, struct NODE * Nodes, const DomainDecomp * ddecomp, int bits, int x, int y, int z, int *nextfree, const int lastnode)
{
    int i, j, k;

    /*We reached the leaf of the toptree*/
    if(ddecomp->TopNodes[topnode].Daughter < 0)
        return;

    for(i = 0; i < 2; i++)
        for(j = 0; j < 2; j++)
            for(k = 0; k < 2; k++)
            {
                int sub = 7 & peano_hilbert_key((x << 1) + i, (y << 1) + j, (z << 1) + k, bits);

                int count = i + 2 * j + 4 * k;

                Nodes[no].s.Types = 0;
                Nodes[no].s.suns[count] = *nextfree;
                /*We are an internal top level node as we now have a child top level.*/
                Nodes[no].f.InternalTopLevel = 1;
                Nodes[no].f.ChildType = NODE_NODE_TYPE;
                Nodes[no].s.noccupied = (1<<16);

                /* We create a new leaf node.*/
                init_internal_node(&Nodes[*nextfree], &Nodes[no], count);
                /*Set father of new node*/
                Nodes[*nextfree].father = no;
                /*All nodes here are top level nodes*/
                Nodes[*nextfree].f.TopLevel = 1;

                const struct topnode_data curtopnode = ddecomp->TopNodes[ddecomp->TopNodes[topnode].Daughter + sub];
                if(curtopnode.Daughter == -1) {
                    ddecomp->TopLeaves[curtopnode.Leaf].treenode = *nextfree;
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

/*! this function inserts pseudo-particles which will represent the mass
 *  distribution of the other CPUs. Initially, the mass of the
 *  pseudo-particles is set to zero, and their coordinate is set to the
 *  center of the ddecomp-cell they correspond to. These quantities will be
 *  updated later on.
 */
static void
force_insert_pseudo_particles(const ForceTree * tree, const DomainDecomp * ddecomp)
{
    int i, index;
    const int firstpseudo = tree->lastnode;
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    for(i = 0; i < ddecomp->NTopLeaves; i++)
    {
        index = ddecomp->TopLeaves[i].treenode;
        if(ddecomp->TopLeaves[i].Task != ThisTask) {
            if(tree->Nodes[index].s.noccupied != 0)
                endrun(5, "In node %d, overwriting %d child particles (i = %d etc) with pseudo particle %d\n",
                       index, tree->Nodes[index].s.noccupied, tree->Nodes[index].s.suns[0], i);
            tree->Nodes[index].f.ChildType = PSEUDO_NODE_TYPE;
            /* This node points to the pseudo particle*/
            tree->Nodes[index].s.suns[0] = firstpseudo + i;
        }
    }
}

int
force_get_father(int no, const ForceTree * tree)
{
    if(no >= tree->firstnode)
        return tree->Nodes[no].father;
    else
        return tree->Father[no];
}

static void
add_particle_moment_to_node(struct NODE * pnode, int i)
{
    int k;
    pnode->mom.mass += (P[i].Mass);
    for(k=0; k<3; k++)
        pnode->mom.cofm[k] += (P[i].Mass * P[i].Pos[k]);

    if(P[i].Type == 0)
    {
        int j;
        /* Maximal distance any of the member particles peek out from the side of the node.
         * May be at most hmax, as |Pos - Center| < len.*/
        for(j = 0; j < 3; j++) {
            pnode->mom.hmax = DMAX(pnode->mom.hmax, fabs(P[i].Pos[j] - pnode->center[j]) + P[i].Hsml - pnode->len);
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
force_update_node_recursive(int no, int sib, int level, const ForceTree * tree)
{
#ifdef DEBUG
    if(tree->Nodes[no].f.ChildType != NODE_NODE_TYPE)
        endrun(3, "force_update_node_recursive called on node %d of type %d != %d!\n", no, tree->Nodes[no].f.ChildType, NODE_NODE_TYPE);
#endif
    int j;
    int * suns = tree->Nodes[no].s.suns;

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
        int p = suns[j];
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
            if(childcnt > 1 && level < 64) {
                #pragma omp task default(none) shared(level, childcnt, tree) firstprivate(nextsib, p)
                force_update_node_recursive(p, nextsib, level*childcnt, tree);
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
void force_update_node_parallel(const ForceTree * tree)
{
#pragma omp parallel
#pragma omp single nowait
    {
        /* Nodes containing other nodes: the overwhelmingly likely case.*/
        if(tree->Nodes[tree->firstnode].f.ChildType == NODE_NODE_TYPE)
            force_update_node_recursive(tree->firstnode, -1, 1, tree);
        else if(tree->Nodes[tree->firstnode].f.ChildType == PARTICLE_NODE_TYPE)
            force_update_particle_node(tree->firstnode, tree);
    }
}

/*! This function communicates the values of the multipole moments of the
 *  top-level tree-nodes of the ddecomp grid.  This data can then be used to
 *  update the pseudo-particles on each CPU accordingly.
 */
void force_exchange_pseudodata(ForceTree * tree, const DomainDecomp * ddecomp)
{
    int NTask, ThisTask;
    int i, no, ta, recvTask;
    int *recvcounts, *recvoffset;
    struct topleaf_momentsdata
    {
        MyFloat s[3];
        MyFloat mass;
        MyFloat hmax;
    }
    *TopLeafMoments;

    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    TopLeafMoments = (struct topleaf_momentsdata *) mymalloc("TopLeafMoments", ddecomp->NTopLeaves * sizeof(TopLeafMoments[0]));
    memset(&TopLeafMoments[0], 0, sizeof(TopLeafMoments[0]) * ddecomp->NTopLeaves);

    for(i = ddecomp->Tasks[ThisTask].StartLeaf; i < ddecomp->Tasks[ThisTask].EndLeaf; i ++) {
        no = ddecomp->TopLeaves[i].treenode;
        if(ddecomp->TopLeaves[i].Task != ThisTask)
            endrun(131231231, "TopLeaf %d Task table is corrupted: task is %d\n", i, ddecomp->TopLeaves[i].Task);

        /* read out the multipole moments from the local base cells */
        TopLeafMoments[i].s[0] = tree->Nodes[no].mom.cofm[0];
        TopLeafMoments[i].s[1] = tree->Nodes[no].mom.cofm[1];
        TopLeafMoments[i].s[2] = tree->Nodes[no].mom.cofm[2];
        TopLeafMoments[i].mass = tree->Nodes[no].mom.mass;
        TopLeafMoments[i].hmax = tree->Nodes[no].mom.hmax;

        /*Set the local base nodes dependence on local mass*/
        while(no >= 0)
        {
#ifdef DEBUG
            if(tree->Nodes[no].f.ChildType == PSEUDO_NODE_TYPE)
                endrun(333, "Pseudo node %d parent of a leaf on this processor %d\n", no, ddecomp->TopLeaves[i].treenode);
#endif
            if(tree->Nodes[no].f.DependsOnLocalMass)
                break;

            tree->Nodes[no].f.DependsOnLocalMass = 1;

            no = tree->Nodes[no].father;
        }
    }

    /* share the pseudo-particle data across CPUs */

    recvcounts = (int *) mymalloc("recvcounts", sizeof(int) * NTask);
    recvoffset = (int *) mymalloc("recvoffset", sizeof(int) * NTask);

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


    for(ta = 0; ta < NTask; ta++) {
        if(ta == ThisTask) continue; /* bypass ThisTask since it is already up to date */

        for(i = ddecomp->Tasks[ta].StartLeaf; i < ddecomp->Tasks[ta].EndLeaf; i ++) {
            no = ddecomp->TopLeaves[i].treenode;

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
void force_treeupdate_pseudos(int no, const ForceTree * tree)
{
    int j, p;
    MyFloat hmax;
    MyFloat s[3], mass;

    mass = 0;
    s[0] = 0;
    s[1] = 0;
    s[2] = 0;
    hmax = 0;

    /* This happens if we have a trivial domain with only one entry*/
    if(!tree->Nodes[no].f.InternalTopLevel)
        return;

    p = tree->Nodes[no].s.suns[0];

    /* since we are dealing with top-level nodes, we know that there are 8 consecutive daughter nodes */
    for(j = 0; j < 8; j++)
    {
        /*This may not happen as we are an internal top level node*/
        if(p < tree->firstnode || p >= tree->lastnode)
            endrun(6767, "Updating pseudos: %d -> %d which is not an internal node between %d and %d.",no, p, tree->firstnode, tree->lastnode);
#ifdef DEBUG
        /* Check we don't move to another part of the tree*/
        if(tree->Nodes[p].father != no)
            endrun(6767, "Tried to update toplevel node %d with parent %d != expected %d\n", p, tree->Nodes[p].father, no);
#endif

        if(tree->Nodes[p].f.InternalTopLevel)
            force_treeupdate_pseudos(p, tree);

        mass += (tree->Nodes[p].mom.mass);
        s[0] += (tree->Nodes[p].mom.mass * tree->Nodes[p].mom.cofm[0]);
        s[1] += (tree->Nodes[p].mom.mass * tree->Nodes[p].mom.cofm[1]);
        s[2] += (tree->Nodes[p].mom.mass * tree->Nodes[p].mom.cofm[2]);

        if(tree->Nodes[p].mom.hmax > hmax)
            hmax = tree->Nodes[p].mom.hmax;

        p = tree->Nodes[p].sibling;
    }

    if(mass)
    {
        s[0] /= mass;
        s[1] /= mass;
        s[2] /= mass;
    }
    else
    {
        s[0] = tree->Nodes[no].center[0];
        s[1] = tree->Nodes[no].center[1];
        s[2] = tree->Nodes[no].center[2];
    }

    tree->Nodes[no].mom.cofm[0] = s[0];
    tree->Nodes[no].mom.cofm[1] = s[1];
    tree->Nodes[no].mom.cofm[2] = s[2];
    tree->Nodes[no].mom.mass = mass;

    tree->Nodes[no].mom.hmax = hmax;
}

/*! This function updates the hmax-values in tree nodes that hold SPH
 *  particles. Since the Hsml-values are potentially changed for active particles
 *  in the SPH-density computation, force_update_hmax() should be carried
 *  out just before the hydrodynamical SPH forces are computed, i.e. after
 *  density().
 *
 *  The purpose of the hmax node is for a symmetric treewalk (currently only the hydro).
 *  Particles where P[i].Pos + Hsml pokes beyond the exterior of the tree node may mean
 *  that a tree node should be included when it would normally be culled. Therefore we don't really
 *  want hmax, we want the maximum amount P[i].Pos + Hsml pokes beyond the tree node.
 */
void force_update_hmax(int * activeset, int size, ForceTree * tree, DomainDecomp * ddecomp)
{
    int NTask, ThisTask, recvTask;
    int i, ta;
    int *recvcounts, *recvoffset;

    walltime_measure("/Misc");

    /* If hmax has not yet been computed, do all particles*/
    if(!tree->hmax_computed_flag) {
        activeset = NULL;
        size = PartManager->NumPart;
    }
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    #pragma omp parallel for
    for(i = 0; i < size; i++)
    {
        const int p_i = activeset ? activeset[i] : i;

        if(P[p_i].Type != 0 || P[p_i].IsGarbage)
            continue;

        int no = tree->Father[p_i];

        while(no >= 0)
        {
            /* How much does this particle peek beyond this node?
             * Note len does not change so we can read it without a lock or atomic. */
            MyFloat readhmax, newhmax = 0;
            int j, done = 0;
            for(j = 0; j < 3; j++) {
                /* Compute each direction independently and take the maximum.
                 * This is the largest possible distance away from node center within a cube bounding hsml.
                 * Note that because Pos - Center < len, the maximum value this can have is Hsml.*/
                newhmax = DMAX(newhmax, fabs(P[p_i].Pos[j] - tree->Nodes[no].center[j]) + P[p_i].Hsml - tree->Nodes[no].len);
            }
            /* Most particles will lie fully inside a node. No need then for the atomic! */
            if(newhmax <= 0)
                break;

            #pragma omp atomic read
            readhmax = tree->Nodes[no].mom.hmax;
            do {
                if(newhmax <= readhmax) {
                    done = 1;
                    break;
                }
                /* Swap in the new hmax only if the old one hasn't changed. */
            } while(!__atomic_compare_exchange(&(tree->Nodes[no].mom.hmax), &readhmax, &newhmax, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));

            if(done)
                break;
            no = tree->Nodes[no].father;
        }
    }

    double * TopLeafhmax = (double *) mymalloc("TopLeafMoments", ddecomp->NTopLeaves * sizeof(double));
    memset(&TopLeafhmax[0], 0, sizeof(double) * ddecomp->NTopLeaves);

    for(i = ddecomp->Tasks[ThisTask].StartLeaf; i < ddecomp->Tasks[ThisTask].EndLeaf; i ++) {
        int no = ddecomp->TopLeaves[i].treenode;
        TopLeafhmax[i] = tree->Nodes[no].mom.hmax;
    }

    /* share the hmax-data of the dirty nodes accross CPUs */
    recvcounts = (int *) mymalloc("recvcounts", sizeof(int) * NTask);
    recvoffset = (int *) mymalloc("recvoffset", sizeof(int) * NTask);

    for(recvTask = 0; recvTask < NTask; recvTask++)
    {
        recvoffset[recvTask] = ddecomp->Tasks[recvTask].StartLeaf;
        recvcounts[recvTask] = ddecomp->Tasks[recvTask].EndLeaf - ddecomp->Tasks[recvTask].StartLeaf;
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
            &TopLeafhmax[0], recvcounts, recvoffset,
            MPI_DOUBLE, MPI_COMM_WORLD);

    myfree(recvoffset);
    myfree(recvcounts);

    for(ta = 0; ta < NTask; ta++) {
        if(ta == ThisTask)
            continue; /* bypass ThisTask since it is already up to date */
        for(i = ddecomp->Tasks[ta].StartLeaf; i < ddecomp->Tasks[ta].EndLeaf; i ++) {
            int no = ddecomp->TopLeaves[i].treenode;
            tree->Nodes[no].mom.hmax = TopLeafhmax[i];

            while(no >= 0)
            {
                if(TopLeafhmax[i] <= tree->Nodes[no].mom.hmax)
                    break;
                tree->Nodes[no].mom.hmax = TopLeafhmax[i];
                no = tree->Nodes[no].father;
            }
         }
    }
    myfree(TopLeafhmax);

    tree->hmax_computed_flag = 1;
    walltime_measure("/Tree/HmaxUpdate");
}

/*! This function allocates the memory used for storage of the tree and of
 *  auxiliary arrays needed for tree-walk and link-lists.  Usually,
 *  maxnodes approximately equal to 0.7*maxpart is sufficient to store the
 *  tree for up to maxpart particles.
 */
ForceTree force_treeallocate(int maxnodes, int maxpart, DomainDecomp * ddecomp)
{
    size_t bytes;
    size_t allbytes = 0;
    ForceTree tb;

    tb.Father = (int *) mymalloc("Father", bytes = (maxpart) * sizeof(int));
#ifdef DEBUG
    memset(tb.Father, -1, bytes);
#endif
    allbytes += bytes;
    tb.Nodes_base = (struct NODE *) mymalloc("Nodes_base", bytes = (maxnodes + 1) * sizeof(struct NODE));
#ifdef DEBUG
    memset(tb.Nodes_base, -1, bytes);
#endif
    allbytes += bytes;
    tb.firstnode = maxpart;
    tb.lastnode = maxpart + maxnodes;
    if(tb.lastnode < 0)
        endrun(5, "Size of tree overflowed for maxpart = %d, maxnodes = %d!\n", maxpart, maxnodes);
    tb.numnodes = 0;
    tb.Nodes = tb.Nodes_base - maxpart;
    tb.tree_allocated_flag = 1;
    tb.NTopLeaves = ddecomp->NTopLeaves;
    tb.TopLeaves = ddecomp->TopLeaves;
    message(0, "Allocated %g MByte for %d tree nodes. firstnode %d. (presently allocated %g MB)\n",
         allbytes / (1024.0 * 1024.0), maxnodes, maxpart,
         mymalloc_usedbytes() / (1024.0 * 1024.0));
    return tb;
}

/*! This function frees the memory allocated for the tree, i.e. it frees
 *  the space allocated by the function force_treeallocate().
 */
void force_tree_free(ForceTree * tree)
{
    event_unlisten(&EventSlotsFork, force_tree_eh_slots_fork, tree);

    if(!force_tree_allocated(tree))
        return;
    myfree(tree->Nodes_base);
    myfree(tree->Father);
    tree->tree_allocated_flag = 0;
}
