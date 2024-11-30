#ifndef FORCETREE_H
#define FORCETREE_H

#include "types.h"
#include "domain.h"
#include "timestep.h"
/*
 * Variables for Tree
 * ------------------
 */

/* Total allowed number of particle children for a node*/
#define NMAXCHILD 8
#define NODEFULL (1<<16)

/* Defines for the type of node, classified by type of children.*/
#define PARTICLE_NODE_TYPE 0
#define NODE_NODE_TYPE 1
#define PSEUDO_NODE_TYPE 2

/* Define to build a tree containing all types of particles*/
#define ALLMASK (1<<6)-1
#define GASMASK (1)
#define DMMASK (2)
#define NUMASK (1<<2)
#define STARMASK (1<<4)
#define BHMASK (1<<5)

struct NodeChild
{
    /*!< pointers to daughter nodes or daughter particles. */
    int suns[NMAXCHILD];
    /* Number of daughter particles if node contains particles.
     * During treebuild >= (1<<16) if node contains nodes.*/
    int noccupied;
};

struct NODE
{
    int sibling;		/*!< this gives the next node in the walk in case the current node can be used */
    int father;		/*!< this gives the parent node of each node (or -1 if we have the root node) */
    MyFloat len;			/*!< sidelength of treenode */
    MyFloat center[3];		/*!< geometrical center of node */

    struct {
        MyFloat cofm[3];		/*!< center of mass of node */
        MyFloat mass;		/*!< mass of node */
        MyFloat hmax;           /*!< maximum amount by which Pos + Hsml of all gas particles in the node exceeds len for this node. */
    } mom;

    /* If the current node needs to be opened, go to the first element of this array.
     * In principle storing the full array wastes memory, because we only use it for the leaf nodes.
     * However, in practice the wasted memory is fairly small: there are sum(1/8^n) ~ 0.15 internal nodes
     * for each leaf node, and we are losing 30% of the memory per node, so the total lost is 5%.
     * Any attempt to get it back by using a separate allocation means we lost the ability to resize
     * the Nodes array and that is always worse.*/
    struct NodeChild s;
    struct {
        unsigned int InternalTopLevel :1; /* TopLevel and has a child which is also TopLevel*/
        unsigned int TopLevel :1; /* Node corresponding to a toplevel node */
        unsigned int DependsOnLocalMass :1;  /* Intersects with local mass */
        unsigned int ChildType :2; /* Specify the type of children this node has: particles, other nodes, or pseudo-particles.
                                    * (should be an enum, but not standard in C).*/
        unsigned int unused : 3; /* Spare bits*/
    } f;
};

/*Structure containing the Node pointer, and various Tree metadata.*/
/*The node index is an integer with unusual properties:
 * no = 0..ForceTree.firstnode  corresponds to a particle.
 * no = ForceTree.firstnode..ForceTree.lastnode corresponds to actual tree nodes,
 * and is the only memory allocated in ForceTree.Nodes_base. After the tree is built this becomes
 * no = ForceTree.firstnode..ForceTree.numnodes which is the only allocated memory.
 * no > ForceTree.lastnode means a pseudo particle on another processor*/
typedef struct ForceTree {
    /*Is 1 if the tree is allocated. Only used inside force_tree_allocated() and when allocating.*/
    int tree_allocated_flag;
    /* Flags that hmax has been computed for this tree*/
    int hmax_computed_flag;
    /* Flags that the tree has fully computed and exchanged mass moments*/
    int moments_computed_flag;
    /* Flags that the tree contains all active particles*/
    int full_particle_tree_flag;
    /*Index of first internal node. Difference between Nodes and Nodes_base. == MaxPart*/
    int64_t firstnode;
    /*Index of first pseudo-particle node*/
    int64_t lastnode;
    /* Number of actually allocated nodes*/
    int64_t numnodes;
    /* Types which are included have their bits set to 1*/
    int mask;
    /* Number of particles stored in this tree*/
    int64_t NumParticles;
    /*Pointer to the TopLeaves struct imported from Domain. Sets up the pseudo particles.*/
    struct topleaf_data * TopLeaves;
    /*Number of TopLeaves*/
    int NTopLeaves;
    /* Index of current task*/
    int ThisTask;
    /*!< this is a pointer used to access the nodes which is shifted such that Nodes[firstnode]
     *   gives the first allocated node */
    struct NODE *Nodes;
    /* The following pointers should only be used via accessors or inside of forcetree.c.
     * The exception is the crazy memory shifting done in sfr_eff.c*/
    /*This points to the actual memory allocated for the nodes.*/
    struct NODE * Nodes_base;
    /*!< gives parent node in tree for every particle */
    int *Father;
    int64_t nfather;
    /*!< Store the size of the box used to build the tree, for periodic walking.*/
    double BoxSize;
} ForceTree;

/*Initialize the internal parameters of the forcetree module*/
void init_forcetree_params(const double treeallocfactor);

int force_tree_allocated(const ForceTree * tt);

/* This function propagates changed SPH smoothing lengths up the tree*/
void force_update_hmax(ActiveParticles * act, ForceTree * tt, DomainDecomp * ddecomp);

/* Update the hmax in the parent node of a single particle at p_i*/
void update_tree_hmax_father(const ForceTree * const tree, const int p_i, const double Pos[3], const double Hsml);

/* Build a tree structure using all particles, compute moments and allocate a father array.
 * This is the fattest tree constructor, allows moments and walking up and down.*/
void force_tree_full(ForceTree * tree, DomainDecomp * ddecomp, const int HybridNuTracer, const char * EmergencyOutputDir);

/* Build a tree structure using only the active particles and compute moments.
 * This variant is for the gravity code*/
void force_tree_active_moments(ForceTree * tree, DomainDecomp * ddecomp, const ActiveParticles *act, const int HybridNuTracer, const int alloc_father, const char * EmergencyOutputDir);

/* Main constructor with a mask argument.
 * Mask is a bitfield, specified as 1 for each type that should be included. Use ALLMASK for all particle types.
 * This is much faster than _full: because the particles are sorted by type the merge step is much faster than
 * with all particle types, and of course the tree is smaller.*/
void force_tree_rebuild_mask(ForceTree * tree, DomainDecomp * ddecomp, int mask, const char * EmergencyOutputDir);

/* Just construct a toptree for domain exchange. If alloc_high is true, allocate the toptree at the upper memory range. */
ForceTree force_tree_top_build(DomainDecomp * ddecomp, const int alloc_high);
/* Find the topnode leaf in the tree that the current particle is attached to*/
int force_tree_find_topnode(const double * const pos, const ForceTree * const tree);

/* Compute moments of the force tree, recursively, and update hmax.*/
void force_tree_calc_moments(ForceTree * tree, DomainDecomp * ddecomp);

/*Free the memory associated with the tree*/
void   force_tree_free(ForceTree * tt);

static inline int
node_is_pseudo_particle(int no, const ForceTree * tree)
{
    return no >= tree->lastnode;
}

static inline int
node_is_particle(int no, const ForceTree * tree)
{
    return no >= 0 && no < tree->firstnode;
}

static inline int
node_is_node(int no, const ForceTree * tree)
{
    return (no >= tree->firstnode) && (no < tree->firstnode + tree->numnodes);
}

int
force_get_father(int no, const ForceTree * tt);

/*Internal API, exposed for tests*/
void
force_tree_create_nodes(ForceTree * tree, const ActiveParticles * act, int mask, DomainDecomp * ddecomp);

ForceTree
force_treeallocate(const int64_t maxnodes, const int64_t maxpart, const DomainDecomp * ddecomp, const int alloc_father, const int alloc_high);

void
force_update_node_parallel(const ForceTree * tree, const DomainDecomp * ddecomp);


#endif



