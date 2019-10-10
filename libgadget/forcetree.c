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
init_forcetree_params(const int FastParticleType, const double * GravitySofteningTable)
{
    ForceTreeParams.TreeAllocFactor = 0.7;
    ForceTreeParams.FastParticleType = FastParticleType;
}

static ForceTree
force_tree_build(int npart, DomainDecomp * ddecomp, const double BoxSize, const int HybridNuGrav);

/*Next three are not static as tested.*/
int
force_tree_create_nodes(const ForceTree tb, const int npart, DomainDecomp * ddecomp, const double BoxSize);

ForceTree
force_treeallocate(int maxnodes, int maxpart, DomainDecomp * ddecomp);

int
force_update_node_parallel(const ForceTree * tree, const int HybridNuGrav);

static void
force_treeupdate_pseudos(int no, const ForceTree * tree);

static void
force_create_node_for_topnode(int no, int topnode, struct NODE * Nodes, const DomainDecomp * ddecomp, int bits, int x, int y, int z, int *nextfree, const int lastnode);

static void
force_exchange_pseudodata(ForceTree * tree, const DomainDecomp * ddecomp);

static void
force_insert_pseudo_particles(const ForceTree * tree, const DomainDecomp * ddecomp);

static int
force_tree_eh_slots_fork(EIBase * event, void * userdata)
{
    /* after a fork, we will attach the new particle to the force tree. */
    EISlotsFork * ev = (EISlotsFork*) event;
    int parent = ev->parent;
    int child = ev->child;
    int no;
    ForceTree * tree = (ForceTree * ) userdata;
    no = tree->Nextnode[parent];
    tree->Nextnode[parent] = child;
    tree->Nextnode[child] = no;
    tree->Father[child] = tree->Father[parent];

    return 0;
}

int
force_tree_allocated(const ForceTree * tree)
{
    return tree->tree_allocated_flag;
}

void
force_tree_rebuild(ForceTree * tree, DomainDecomp * ddecomp, const double BoxSize, const int HybridNuGrav)
{
    MPIU_Barrier(MPI_COMM_WORLD);
    message(0, "Tree construction.  (presently allocated=%g MB)\n", mymalloc_usedbytes() / (1024.0 * 1024.0));

    if(force_tree_allocated(tree)) {
        force_tree_free(tree);
    }
    walltime_measure("/Misc");

    *tree = force_tree_build(PartManager->NumPart, ddecomp, BoxSize, HybridNuGrav);

    event_listen(&EventSlotsFork, force_tree_eh_slots_fork, tree);

    MPIU_Barrier(MPI_COMM_WORLD);
    message(0, "Tree construction done.\n");
    walltime_measure("/Tree/Build/Moments");
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
ForceTree force_tree_build(int npart, DomainDecomp * ddecomp, const double BoxSize, const int HybridNuGrav)
{
    int Numnodestree;
    ForceTree tree;

    int TooManyNodes = 0;

    do
    {
        int maxnodes = ForceTreeParams.TreeAllocFactor * PartManager->MaxPart + ddecomp->NTopNodes;
        /* Allocate memory. */
        tree = force_treeallocate(maxnodes, PartManager->MaxPart, ddecomp);

        tree.BoxSize = BoxSize;
        Numnodestree = force_tree_create_nodes(tree, npart, ddecomp, BoxSize);
        if(Numnodestree >= tree.lastnode - tree.firstnode)
        {
            message(1, "Not enough tree nodes (%d) for %d particles.\n", maxnodes, npart);
            force_tree_free(&tree);
            message(1, "TreeAllocFactor from %g to %g\n", ForceTreeParams.TreeAllocFactor, ForceTreeParams.TreeAllocFactor*1.15);
            ForceTreeParams.TreeAllocFactor *= 1.15;
            if(ForceTreeParams.TreeAllocFactor > 3.0) {
                TooManyNodes = 1;
                break;
            }
        }
    }
    while(Numnodestree >= tree.lastnode - tree.firstnode);

    if(MPIU_Any(TooManyNodes, MPI_COMM_WORLD)) {
        dump_snapshot();
        endrun(2, "Required too many nodes, snapshot dumped\n");
    }
    walltime_measure("/Tree/Build/Nodes");

    /* insert the pseudo particles that represent the mass distribution of other ddecomps */
    force_insert_pseudo_particles(&tree, ddecomp);

    /* now compute the multipole moments recursively */
    force_update_node_parallel(&tree, HybridNuGrav);

    /* Exchange the pseudo-data*/
    force_exchange_pseudodata(&tree, ddecomp);

    force_treeupdate_pseudos(PartManager->MaxPart, &tree);

    tree.Nodes_base = myrealloc(tree.Nodes_base, (Numnodestree +1) * sizeof(struct NODE));

    /*Update the oct-tree struct so it knows about the memory change*/
    tree.numnodes = Numnodestree;
    tree.Nodes = tree.Nodes_base - tree.firstnode;

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
    nfreep->f.TopLevel = 0;
    nfreep->f.InternalTopLevel = 0;
    nfreep->f.ChildType = PARTICLE_NODE_TYPE;

    for(j = 0; j < 3; j++) {
        /* Detect which quadrant we are in by testing the bits of subnode:
         * if (subnode & [1,2,4]) is true we add lenhalf, otherwise subtract lenhalf*/
        const int sign = (subnode & (1 << j)) ? 1 : -1;
        nfreep->center[j] = parent->center[j] + sign*lenhalf;
    }
    for(j = 0; j < NMAXCHILD; j++)
        nfreep->u.s.suns[j] = -1;
    nfreep->u.s.noccupied = 0;
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
modify_internal_node(int parent, int subnode, int p_toplace, const ForceTree tb)
{
    tb.Father[p_toplace] = parent;
    tb.Nodes[parent].u.s.suns[subnode] = p_toplace;
    return 0;
}


/* Create a new layer of nodes beneath the current node, and place the particle.
 * Must have node lock.*/
static int
create_new_node_layer(int firstparent, int p_toplace,
        const ForceTree tb, int *nnext, struct NodeCache *nc)
{
    /* This is so we can defer changing
     * the type of the existing node until the end.*/
    int parent = firstparent;

    do {
        int i;
        int oldsuns[NMAXCHILD];

        struct NODE *nprnt = &tb.Nodes[parent];

        /* Copy the old particles and a new one into a temporary array*/
        memcpy(oldsuns, nprnt->u.s.suns, NMAXCHILD * sizeof(int));

        /*We have two particles here, so create a new child node to store them both.*/
        /* if we are here the node must be large enough, thus contain exactly one child. */
        /* The parent is already a leaf, need to split */
        for(i=0; i<8; i++) {
            /* Get memory for an extra node from our cache.*/
            nprnt->u.s.suns[i] = get_freenode(nnext, nc);
            /*If we already have too many nodes, exit loop.*/
            if(nc->nnext_thread >= tb.lastnode) {
                /* This means that we have > NMAXCHILD particles in the same place,
                * which usually indicates a bug in the particle evolution. Print some helpful debug information.*/
                message(1, "Failed placing %d at %g %g %g, type %d, ID %ld. Others were %d (%g %g %g, t %d ID %ld) and %d (%g %g %g, t %d ID %ld).\n",
                    p_toplace, P[p_toplace].Pos[0], P[p_toplace].Pos[1], P[p_toplace].Pos[2], P[p_toplace].Type, P[p_toplace].ID,
                    oldsuns[0], P[oldsuns[0]].Pos[0], P[oldsuns[0]].Pos[1], P[oldsuns[0]].Pos[2], P[oldsuns[0]].Type, P[oldsuns[0]].ID,
                    oldsuns[1], P[oldsuns[1]].Pos[0], P[oldsuns[1]].Pos[1], P[oldsuns[1]].Pos[2], P[oldsuns[1]].Type, P[oldsuns[1]].ID
                );
                return 1;
            }
            struct NODE *nfreep = &tb.Nodes[nprnt->u.s.suns[i]];
            /* We create a new leaf node.*/
            init_internal_node(nfreep, nprnt, i);
            /*Set father of new node*/
            nfreep->father = parent;
        }
        /*Initialize the remaining entries to empty*/
        for(i=8; i<NMAXCHILD;i++)
            nprnt->u.s.suns[i] = -1;

        for(i=0; i < NMAXCHILD; i++) {
            /* Re-attach each particle to the appropriate new leaf.
            * Notice that since we have NMAXCHILD slots on each child and NMAXCHILD particles,
            * we will always have a free slot. */
            int subnode = get_subnode(nprnt, oldsuns[i]);
            int child = nprnt->u.s.suns[subnode];
            struct NODE * nchild = &tb.Nodes[child];
            modify_internal_node(child, nchild->u.s.noccupied, oldsuns[i], tb);
            nchild->u.s.noccupied++;
        }
        /* Now try again to add the new particle*/
        int subnode = get_subnode(nprnt, p_toplace);
        int child = nprnt->u.s.suns[subnode];
        struct NODE * nchild = &tb.Nodes[child];
        if(nchild->u.s.noccupied < NMAXCHILD) {
            modify_internal_node(child, nchild->u.s.noccupied, p_toplace, tb);
            nchild->u.s.noccupied++;
            break;
        }
        /* The attached particles are already within one subnode of the new node.
         * Iterate, creating a new layer beneath.*/
        else {
            /* The current child is going to have new nodes created beneath it,
             * so mark it a Node-containing node. It cannot be accessed until
             * we mark the top-level parent, so no need for atomics.*/
            tb.Nodes[child].f.ChildType = NODE_NODE_TYPE;
            tb.Nodes[child].u.s.noccupied = (1<<16);
            parent = child;
        }
    } while(1);

    /* A new node is created. Mark the (original) parent as an internal node with node children.
     * This goes last so that we don't access the child before it is constructed.*/
    tb.Nodes[firstparent].f.ChildType = NODE_NODE_TYPE;
    #pragma omp atomic write
    tb.Nodes[firstparent].u.s.noccupied = (1<<16);
    return 0;
}

/*! Does initial creation of the nodes for the gravitational oct-tree.
 **/
int force_tree_create_nodes(const ForceTree tb, const int npart, DomainDecomp * ddecomp, const double BoxSize)
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
            nfreep->u.s.suns[i] = -1;
        nfreep->u.s.noccupied = 0;
        nfreep->father = -1;
        nfreep->f.TopLevel = 1;
        nfreep->f.InternalTopLevel = 0;
        nfreep->f.ChildType = PARTICLE_NODE_TYPE;
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
        if(nc.nnext_thread >= tb.lastnode-1)
            continue;

        /* Do not add garbage particles to the tree*/
        if(P[i].IsGarbage)
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
            nocc = tb.Nodes[this].u.s.noccupied;

            /* This node still has space for a particle (or needs conversion)*/
            if(nocc < (1 << 16))
                break;

            /* This node has child subnodes: find them.*/
            int subnode = get_subnode(&tb.Nodes[this], i);
            /*No lock needed: if we have an internal node here it will be stable*/
            child = tb.Nodes[this].u.s.suns[subnode];

            if(child > tb.lastnode || child < tb.firstnode)
                endrun(1,"Corruption in tree build: N[%d].[%d] = %d > lastnode (%d)\n",this, subnode, child, tb.lastnode);
            this = child;
        }
        while(child >= tb.firstnode);

        /*Now lock this node.*/
        lock_spinlock(this-tb.firstnode, spin);
        /* We have a guaranteed spot.*/
        nocc = atomic_fetch_and_add(&tb.Nodes[this].u.s.noccupied, 1);

        /* Check whether there is now a new layer of nodes and if so walk down until there isn't.*/
        if(nocc >= (1<<16)) {
            /* This node has child subnodes: find them.*/
            int subnode = get_subnode(&tb.Nodes[this], i);
            child = tb.Nodes[this].u.s.suns[subnode];
            while(child >= tb.firstnode)
            {
                /*Move the lock to the child*/
                lock_spinlock(child-tb.firstnode, spin);
                unlock_spinlock(this-tb.firstnode, spin);
                this = child;

                /*No lock needed: if we have an internal node here it will be stable*/
                #pragma omp atomic read
                nocc = tb.Nodes[this].u.s.noccupied;
                /* This node still has space for a particle (or needs conversion)*/
                if(nocc < (1 << 16))
                    break;

                /* This node has child subnodes: find them.*/
                subnode = get_subnode(&tb.Nodes[this], i);
                /*No lock needed: if we have an internal node here it will be stable*/
                child = tb.Nodes[this].u.s.suns[subnode];
            }
            /* Get the free spot under the lock.*/
            nocc = atomic_fetch_and_add(&tb.Nodes[this].u.s.noccupied, 1);
        }

        /*Update last-used cache*/
        this_acc = this;

        /* Now we have something that isn't an internal node, and we have a lock on it,
         * so we know it won't change. We can place the particle! */
        if(nocc < NMAXCHILD)
            modify_internal_node(this, nocc, i, tb);
        /* In this case we need to create a new layer of nodes beneath this one*/
        else if(nocc < 1<<16)
            create_new_node_layer(this, i, tb, &nnext, &nc);
        else
            endrun(2, "Tried to convert already converted node %d with nocc = %d\n", this, nocc);

        /* Add an explicit flush because we are not using openmp's critical sections */
        #pragma omp flush

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

                Nodes[no].u.s.suns[count] = *nextfree;
                /*We are an internal top level node as we now have a child top level.*/
                Nodes[no].f.InternalTopLevel = 1;
                Nodes[no].f.ChildType = NODE_NODE_TYPE;
                Nodes[no].u.s.noccupied = (1<<16);

                const MyFloat lenhalf = 0.25 * Nodes[no].len;
                Nodes[*nextfree].len = 0.5 * Nodes[no].len;
                Nodes[*nextfree].center[0] = Nodes[no].center[0] + (2 * i - 1) * lenhalf;
                Nodes[*nextfree].center[1] = Nodes[no].center[1] + (2 * j - 1) * lenhalf;
                Nodes[*nextfree].center[2] = Nodes[no].center[2] + (2 * k - 1) * lenhalf;
                Nodes[*nextfree].father = no;
                /*All nodes here are top level nodes*/
                Nodes[*nextfree].f.TopLevel = 1;
                Nodes[*nextfree].f.InternalTopLevel = 0;
                Nodes[*nextfree].f.ChildType = PARTICLE_NODE_TYPE;

                int n;
                for(n = 0; n < NMAXCHILD; n++)
                    Nodes[*nextfree].u.s.suns[n] = -1;
                Nodes[*nextfree].u.s.noccupied = 0;

                const struct topnode_data curtopnode = ddecomp->TopNodes[ddecomp->TopNodes[topnode].Daughter + sub];
                if(curtopnode.Daughter == -1) {
                    ddecomp->TopLeaves[curtopnode.Leaf].treenode = *nextfree;
                }

                (*nextfree)++;

                if(*nextfree >= lastnode)
                    endrun(11, "Not enough force nodes to topnode grid: need %d\n",lastnode);

                force_create_node_for_topnode(*nextfree - 1, ddecomp->TopNodes[topnode].Daughter + sub, Nodes, ddecomp,
                        bits + 1, 2 * x + i, 2 * y + j, 2 * z + k, nextfree, lastnode);
            }
}

/* This function zeros the parts of a node in a union with the suns array*/
static void
force_zero_union(struct NODE * node)
{
    memset(&(node->u.d.s),0,3*sizeof(MyFloat));
    node->u.d.mass = 0;
    node->u.d.hmax = 0;
    node->u.d.MaxSoftening = -1;
    node->f.DependsOnLocalMass = 0;
    node->f.MixedSofteningsInNode = 0;
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
            if(tree->Nodes[index].u.s.noccupied != 0)
                endrun(5, "In node %d, overwriting %d child particles (i = %d etc) with pseudo particle %d (%d)\n",
                       index, tree->Nodes[index].u.s.noccupied, tree->Nodes[index].u.s.suns[0], i);
            force_zero_union(&tree->Nodes[index]);
            tree->Nodes[index].f.ChildType = PSEUDO_NODE_TYPE;
            force_set_next_node(index, firstpseudo + i, tree);
            force_set_next_node(firstpseudo + i, -1, tree);
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

int
force_get_next_node(int no, const ForceTree * tree)
{
    if(no >= tree->firstnode && no < tree->lastnode) {
        /* internal node */
        return tree->Nodes[no].u.d.nextnode;
    }
    if(no < tree->firstnode) {
        /* Particle */
        return tree->Nextnode[no];
    }
    else { //if(no >= tb.lastnode) {
        /* Pseudo Particle */
        return tree->Nextnode[no - (tree->lastnode - tree->firstnode)];
    }
}

int
force_set_next_node(int no, int next, const ForceTree * tree)
{
    if(no < 0) return next;
    if(no >= tree->firstnode && no < tree->lastnode) {
        /* internal node */
        tree->Nodes[no].u.d.nextnode = next;
    }
    if(no < tree->firstnode) {
        /* Particle */
        tree->Nextnode[no] = next;
    }
    if(no >= tree->lastnode) {
        /* Pseudo Particle */
        tree->Nextnode[no - (tree->lastnode - tree->firstnode)] = next;
    }

    return next;
}

int
force_get_prev_node(int no, const ForceTree * tree)
{
    if(node_is_particle(no, tree)) {
        /* Particle */
        int t = tree->Father[no];
        int next = force_get_next_node(t, tree);
        while(next != no) {
            t = next;
            next = force_get_next_node(t, tree);
        }
        return t;
    } else {
        /* not implemented yet */
        endrun(1, "get_prev_node on non particles is not implemented yet\n");
        return 0;
    }
}

/* Sets the node softening on a node.
 *
 * */
static void
force_adjust_node_softening(struct NODE * pnode, double MaxSoftening, int mixed)
{

    if(pnode->u.d.MaxSoftening > 0) {
        /* already set? mark MixedSoftenings */
        if(MaxSoftening != pnode->u.d.MaxSoftening) {
            pnode->f.MixedSofteningsInNode = 1;
        }
    }
    if(MaxSoftening > pnode->u.d.MaxSoftening) {
        pnode->u.d.MaxSoftening = MaxSoftening;
    }
    if(mixed) {
        pnode->f.MixedSofteningsInNode = 1;
    }
}

static void
add_particle_moment_to_node(struct NODE * pnode, int i)
{
    int k;
    pnode->u.d.mass += (P[i].Mass);
    for(k=0; k<3; k++)
        pnode->u.d.s[k] += (P[i].Mass * P[i].Pos[k]);

    if(P[i].Type == 0)
    {
        if(P[i].Hsml > pnode->u.d.hmax)
            pnode->u.d.hmax = P[i].Hsml;
    }

    force_adjust_node_softening(pnode, FORCE_SOFTENING(i), 0);
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

/* Very little to be done for a pseudo particle because the mass of the
* pseudo-particle is still zero. The node attributes will be changed
* later when we exchange the pseudo-particles.*/
static int
force_update_pseudo_node(int no, int sib, const ForceTree * tree)
{
    if(tree->Nodes[no].f.ChildType != PSEUDO_NODE_TYPE)
        endrun(3, "force_update_pseudo_node called on node %d of wrong type!\n", no);

    tree->Nodes[no].u.d.sibling = sib;

    /*The pseudo-particle is the return value of this function.*/
    return force_get_next_node(no, tree);
}

static int
force_update_particle_node(int no, int sib, const ForceTree * tree, const int HybridNuGrav)
{
    if(tree->Nodes[no].f.ChildType != PARTICLE_NODE_TYPE)
        endrun(3, "force_update_particle_node called on node %d of wrong type!\n", no);
    /*Last value of tails is the return value of this function*/
    int j, suns[NMAXCHILD];

    /* this "backup" is necessary because the nextnode
     * entry will overwrite one element (union!) */
    int noccupied = tree->Nodes[no].u.s.noccupied;
    for(j = 0; j < noccupied; j++) {
        suns[j] = tree->Nodes[no].u.s.suns[j];
    }

    /*After this point the suns array is invalid!*/
    force_zero_union(&tree->Nodes[no]);
    tree->Nodes[no].u.d.sibling = sib;

    int tail = no;
    /*Now we do the moments*/
    for(j = 0; j < noccupied; j++) {
        const int p = suns[j];
        /*Hybrid particle neutrinos do not gravitate at early times.
            * So do not add their masses to the node*/
        if(!HybridNuGrav || P[p].Type != ForceTreeParams.FastParticleType)
            add_particle_moment_to_node(&tree->Nodes[no], p);
        /*This loop sets the next node value for the row we just computed.
         * Note that tails[i] is the next node for suns[i-1].
         * The last tail needs to be the return value of this function.*/
        force_set_next_node(tail, p, tree);
        tail = p;
    }

    /*Set the center of mass moments*/
    const double mass = tree->Nodes[no].u.d.mass;
    /* Be careful about empty nodes*/
    if(mass > 0) {
        for(j = 0; j < 3; j++)
            tree->Nodes[no].u.d.s[j] /= mass;
    }
    else {
        for(j = 0; j < 3; j++)
            tree->Nodes[no].u.d.s[j] = tree->Nodes[no].center[j];
    }

    return tail;
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
force_update_node_recursive(int no, int sib, int level, const ForceTree * tree, const int HybridNuGrav)
{
#ifdef DEBUG
    if(tree->Nodes[no].f.ChildType != NODE_NODE_TYPE)
        endrun(3, "force_update_node_recursive called on node %d of type %d != %d!\n", no, tree->Nodes[no].f.ChildType, NODE_NODE_TYPE);
#endif

    /*Last value of tails is the return value of this function*/
    int j, suns[8], tails[8];

    /* this "backup" is necessary because the nextnode
     * entry will overwrite one element (union!) */
    memcpy(suns, tree->Nodes[no].u.s.suns, 8 * sizeof(int));

    int childcnt = 0;
    /* Remove any empty children.
     * This sharply reduces the size of the tree.
     * Also count the node children for thread balancing.*/
    for(j=0; j < 8; j++) {
        /* Never remove empty top-level nodes so we don't
         * mess up the pseudo-data exchange.
         * This may happen for a pseudo particle host or, in very rare cases,
         * when one of the local domains is empty. */
        if(!tree->Nodes[suns[j]].f.TopLevel &&
            tree->Nodes[suns[j]].f.ChildType == PARTICLE_NODE_TYPE &&
            tree->Nodes[suns[j]].u.s.noccupied == 0) {
                /* In principle the removed node will be
                 * disconnected and never used, but zero it just in case.*/
                force_zero_union(&tree->Nodes[suns[j]]);
                suns[j] = -1;
        }
        else if(tree->Nodes[suns[j]].f.ChildType == NODE_NODE_TYPE)
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
        /* Nodes containing particles or pseudo-particles*/
        if(tree->Nodes[p].f.ChildType == PARTICLE_NODE_TYPE)
            tails[j] = force_update_particle_node(p, nextsib, tree, HybridNuGrav);
        else if(tree->Nodes[p].f.ChildType == PSEUDO_NODE_TYPE)
            tails[j] = force_update_pseudo_node(p, nextsib, tree);
        /*Don't spawn a new task if we are deep enough that we already spawned a lot.
        Note: final clause is much slower for some reason. */
        else if(childcnt > 1 && level < 256) {
            /* We cannot use default(none) here because we need a const (HybridNuGrav),
            * which for gcc < 9 is default shared (and thus cannot be explicitly shared
            * without error) and for gcc == 9 must be explicitly shared. The other solution
            * is to make it firstprivate which I think will be excessively expensive for a
            * recursive call like this. See:
            * https://www.gnu.org/software/gcc/gcc-9/porting_to.html */
            #pragma omp task shared(tails, level, childcnt, tree) firstprivate(j, nextsib, p)
            tails[j] = force_update_node_recursive(p, nextsib, level*childcnt, tree, HybridNuGrav);
        }
        else
            tails[j] = force_update_node_recursive(p, nextsib, level, tree, HybridNuGrav);
    }

    /*After this point the suns array is invalid!*/
    force_zero_union(&tree->Nodes[no]);
    tree->Nodes[no].u.d.sibling = sib;

    /*Make sure all child nodes are done*/
    #pragma omp taskwait

    /*Now we do the moments*/
    for(j = 0; j < 8; j++)
    {
        const int p = suns[j];
        if(p < 0)
            continue;
        tree->Nodes[no].u.d.mass += (tree->Nodes[p].u.d.mass);
        tree->Nodes[no].u.d.s[0] += (tree->Nodes[p].u.d.mass * tree->Nodes[p].u.d.s[0]);
        tree->Nodes[no].u.d.s[1] += (tree->Nodes[p].u.d.mass * tree->Nodes[p].u.d.s[1]);
        tree->Nodes[no].u.d.s[2] += (tree->Nodes[p].u.d.mass * tree->Nodes[p].u.d.s[2]);
        if(tree->Nodes[p].u.d.hmax > tree->Nodes[no].u.d.hmax)
            tree->Nodes[no].u.d.hmax = tree->Nodes[p].u.d.hmax;

        force_adjust_node_softening(&tree->Nodes[no], tree->Nodes[p].u.d.MaxSoftening, tree->Nodes[p].f.MixedSofteningsInNode);
    }

    /*Set the center of mass moments*/
    const double mass = tree->Nodes[no].u.d.mass;
    /* In principle all the children could be pseudo-particles*/
    if(mass > 0) {
        tree->Nodes[no].u.d.s[0] /= mass;
        tree->Nodes[no].u.d.s[1] /= mass;
        tree->Nodes[no].u.d.s[2] /= mass;
    }

    /*This loop sets the next node value for the row we just computed.
      Note that tails[i] is the next node for suns[i-1].
      The the last tail needs to be the return value of this function.*/
    int tail = no;
    for(j = 0; j < 8; j++)
    {
        if(suns[j] < 0)
            continue;
        /*Set NextNode for this node*/
        force_set_next_node(tail, suns[j], tree);
        tail = tails[j];
    }
    return tail;
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
int
force_update_node_parallel(const ForceTree * tree, const int HybridNuGrav)
{
    int tail;
#pragma omp parallel
#pragma omp single nowait
    {
        /* Nodes containing other nodes: the overwhelmingly likely case.*/
        if(tree->Nodes[tree->firstnode].f.ChildType == NODE_NODE_TYPE)
            tail = force_update_node_recursive(tree->firstnode, -1, 1, tree, HybridNuGrav);
        else if(tree->Nodes[tree->firstnode].f.ChildType == PARTICLE_NODE_TYPE)
            tail = force_update_particle_node(tree->firstnode, -1, tree, HybridNuGrav);
        else
            tail = force_update_pseudo_node(tree->firstnode, -1, tree);
    }

    force_set_next_node(tail, -1, tree);

    return tail;
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
        struct {
            unsigned int MixedSofteningsInNode :1;
        };
        MyFloat MaxSoftening;
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
        TopLeafMoments[i].s[0] = tree->Nodes[no].u.d.s[0];
        TopLeafMoments[i].s[1] = tree->Nodes[no].u.d.s[1];
        TopLeafMoments[i].s[2] = tree->Nodes[no].u.d.s[2];
        TopLeafMoments[i].mass = tree->Nodes[no].u.d.mass;
        TopLeafMoments[i].hmax = tree->Nodes[no].u.d.hmax;
        TopLeafMoments[i].MaxSoftening = tree->Nodes[no].u.d.MaxSoftening;
        TopLeafMoments[i].MixedSofteningsInNode = tree->Nodes[no].f.MixedSofteningsInNode;

        /*Set the local base nodes dependence on local mass*/
        while(no >= 0)
        {
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

            tree->Nodes[no].u.d.s[0] = TopLeafMoments[i].s[0];
            tree->Nodes[no].u.d.s[1] = TopLeafMoments[i].s[1];
            tree->Nodes[no].u.d.s[2] = TopLeafMoments[i].s[2];
            tree->Nodes[no].u.d.mass = TopLeafMoments[i].mass;
            tree->Nodes[no].u.d.hmax = TopLeafMoments[i].hmax;
            tree->Nodes[no].u.d.MaxSoftening = TopLeafMoments[i].MaxSoftening;
            tree->Nodes[no].f.MixedSofteningsInNode = TopLeafMoments[i].MixedSofteningsInNode;
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

    tree->Nodes[no].u.d.MaxSoftening = -1;
    tree->Nodes[no].f.MixedSofteningsInNode = 0;

    /* This happens if we have a trivial domain with only one entry*/
    if(!tree->Nodes[no].f.InternalTopLevel)
        return;

    p = tree->Nodes[no].u.d.nextnode;

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

        mass += (tree->Nodes[p].u.d.mass);
        s[0] += (tree->Nodes[p].u.d.mass * tree->Nodes[p].u.d.s[0]);
        s[1] += (tree->Nodes[p].u.d.mass * tree->Nodes[p].u.d.s[1]);
        s[2] += (tree->Nodes[p].u.d.mass * tree->Nodes[p].u.d.s[2]);

        if(tree->Nodes[p].u.d.hmax > hmax)
            hmax = tree->Nodes[p].u.d.hmax;

        force_adjust_node_softening(&tree->Nodes[no], tree->Nodes[p].u.d.MaxSoftening, tree->Nodes[p].f.MixedSofteningsInNode);

        p = tree->Nodes[p].u.d.sibling;
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

    tree->Nodes[no].u.d.s[0] = s[0];
    tree->Nodes[no].u.d.s[1] = s[1];
    tree->Nodes[no].u.d.s[2] = s[2];
    tree->Nodes[no].u.d.mass = mass;

    tree->Nodes[no].u.d.hmax = hmax;
}

/*! This function updates the hmax-values in tree nodes that hold SPH
 *  particles. These values are needed to find all neighbors in the
 *  hydro-force computation.  Since the Hsml-values are potentially changed
 *  in the SPH-density computation, force_update_hmax() should be carried
 *  out just before the hydrodynamical SPH forces are computed, i.e. after
 *  density().
 */
void force_update_hmax(int * activeset, int size, ForceTree * tree)
{
    int NTask, ThisTask;
    int i, ta;
    int *counts, *offsets;
    struct dirty_node_data {
        int treenode;
        MyFloat hmax;
    } * DirtyTopLevelNodes;

    int NumDirtyTopLevelNodes;

    walltime_measure("/Misc");

    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    NumDirtyTopLevelNodes = 0;

    /* At most NTopLeaves are dirty, since we are only concerned with TOPLEVEL nodes */
    DirtyTopLevelNodes = (struct dirty_node_data*) mymalloc("DirtyTopLevelNodes", tree->NTopLeaves * sizeof(DirtyTopLevelNodes[0]));

    /* FIXME: actually only TOPLEVEL nodes contains the local mass can potentially be dirty,
     *  we may want to save a list of them to speed this up.
     * */
    for(i = tree->firstnode; i < tree->firstnode + tree->numnodes; i ++) {
        tree->Nodes[i].f.NodeIsDirty = 0;
    }

    for(i = 0; i < size; i++)
    {
        const int p_i = activeset ? activeset[i] : i;

        if(P[p_i].Type != 0)
            continue;

        int no = tree->Father[p_i];

        while(no >= 0)
        {
            if(P[p_i].Hsml <= tree->Nodes[no].u.d.hmax) break;

            tree->Nodes[no].u.d.hmax = P[p_i].Hsml;

            if(tree->Nodes[no].f.TopLevel) /* we reached a top-level node */
            {
                if (!tree->Nodes[no].f.NodeIsDirty) {
                    tree->Nodes[no].f.NodeIsDirty = 1;
                    DirtyTopLevelNodes[NumDirtyTopLevelNodes].treenode = no;
                    DirtyTopLevelNodes[NumDirtyTopLevelNodes].hmax = tree->Nodes[no].u.d.hmax;
                    NumDirtyTopLevelNodes ++;
                }
                break;
            }

            no = tree->Nodes[no].father;
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

    message(0, "Hmax exchange: %d toplevel tree nodes out of %d\n", offsets[NTask], tree->NTopLeaves);

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

        /* Why does this matter? The logic is simpler if we just blindly update them all.
            ::: to avoid that the hmax is updated twice :::*/
        if(tree->Nodes[no].f.DependsOnLocalMass)
            no = tree->Nodes[no].father;

        while(no >= 0)
        {
            if(DirtyTopLevelNodes[i].hmax <= tree->Nodes[no].u.d.hmax) break;

            tree->Nodes[no].u.d.hmax = DirtyTopLevelNodes[i].hmax;

            no = tree->Nodes[no].father;
        }
    }

    myfree(offsets);
    myfree(counts);
    myfree(DirtyTopLevelNodes);

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

    message(0, "Allocating memory for %d tree-nodes (MaxPart=%d).\n", maxnodes, maxpart);
    tb.Nnextnode = maxpart + ddecomp->NTopNodes;
    tb.Nextnode = (int *) mymalloc("Nextnode", bytes = tb.Nnextnode * sizeof(int));
    tb.Father = (int *) mymalloc("Father", bytes = (maxpart) * sizeof(int));
    allbytes += bytes;
    tb.Nodes_base = (struct NODE *) mymalloc("Nodes_base", bytes = (maxnodes + 1) * sizeof(struct NODE));
    allbytes += bytes;
    tb.firstnode = maxpart;
    tb.lastnode = maxpart + maxnodes;
    tb.numnodes = maxnodes;
    tb.Nodes = tb.Nodes_base - maxpart;
    tb.tree_allocated_flag = 1;
    tb.NTopLeaves = ddecomp->NTopLeaves;
    tb.TopLeaves = ddecomp->TopLeaves;

    allbytes += bytes;
    message(0, "Allocated %g MByte for BH-tree, (presently allocated %g MB)\n",
         allbytes / (1024.0 * 1024.0),
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
    myfree(tree->Nextnode);
    tree->tree_allocated_flag = 0;
}
