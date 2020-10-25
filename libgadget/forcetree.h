#ifndef FORCETREE_H
#define FORCETREE_H

#include "types.h"
#include "domain.h"
/*
 * Variables for Tree
 * ------------------
 */

/* Total allowed number of particle children for a node*/
#define NMAXCHILD 8

/* Defines for the type of node, classified by type of children.*/
#define PARTICLE_NODE_TYPE 0
#define NODE_NODE_TYPE 1
#define PSEUDO_NODE_TYPE 2

struct NodeChild
{
    /* Stores the types of the child particles. Uses a bit field, 3 bits per particle.
     */
    unsigned int Types;
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
        unsigned int InternalTopLevel :1; /* TopLevel and has a child which is also TopLevel*/
        unsigned int TopLevel :1; /* Node corresponding to a toplevel node */
        unsigned int DependsOnLocalMass :1;  /* Intersects with local mass */
        unsigned int ChildType :2; /* Specify the type of children this node has: particles, other nodes, or pseudo-particles.
                                    * (should be an enum, but not standard in C).*/
        unsigned int unused : 3; /* Spare bits*/
    } f;

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
    /*Index of first internal node. Difference between Nodes and Nodes_base. == MaxPart*/
    int firstnode;
    /*Index of first pseudo-particle node*/
    int lastnode;
    /* Number of actually allocated nodes*/
    int numnodes;
    /*Pointer to the TopLeaves struct imported from Domain. Sets up the pseudo particles.*/
    struct topleaf_data * TopLeaves;
    /*Number of TopLeaves*/
    int NTopLeaves;
    /*!< this is a pointer used to access the nodes which is shifted such that Nodes[firstnode]
     *   gives the first allocated node */
    struct NODE *Nodes;
    /* The following pointers should only be used via accessors or inside of forcetree.c.
     * The exception is the crazy memory shifting done in sfr_eff.c*/
    /*This points to the actual memory allocated for the nodes.*/
    struct NODE * Nodes_base;
    /*!< gives parent node in tree for every particle */
    int *Father;
    /*!< Store the size of the box used to build the tree, for periodic walking.*/
    double BoxSize;
} ForceTree;

/*Initialize the internal parameters of the forcetree module*/
void init_forcetree_params(const int FastParticleType);

int force_tree_allocated(const ForceTree * tt);

/* This function propagates changed SPH smoothing lengths up the tree*/
void force_update_hmax(int * activeset, int size, ForceTree * tt, DomainDecomp * ddecomp);

/* This is the main constructor for the tree structure.
   The tree shall be either zero-filled, so that force_tree_allocated = 0, or a valid ForceTree.
*/
void force_tree_rebuild(ForceTree * tree, DomainDecomp * ddecomp, const double BoxSize, const int HybridNuGrav, const int DoMoments, const char * EmergencyOutputDir);

/*Free the memory associated with the tree*/
void   force_tree_free(ForceTree * tt);
void   dump_particles(void);

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

#endif



