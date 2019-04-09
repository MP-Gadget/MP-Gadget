#ifndef FORCETREE_H
#define FORCETREE_H

#include "types.h"
#include "domain.h"
/*
 * Variables for Tree
 * ------------------
 */

struct NODE
{
    MyFloat len;			/*!< sidelength of treenode */
    MyFloat center[3];		/*!< geometrical center of node */

    int father;		/*!< this gives the parent node of each node (or -1 if we have the root node) */
    struct {
        unsigned int InternalTopLevel :1; /* TopLevel and has a child which is also TopLevel*/
        unsigned int TopLevel :1; /* Node corresponding to a toplevel node */
        unsigned int DependsOnLocalMass :1;  /* Intersects with local mass */
        unsigned int MixedSofteningsInNode:1;  /* Softening is mixed, need to open the node */
        unsigned int NodeIsDirty :1; /*Node is a toplevel node containing local mass, and its moments need updating*/
    } f;
    union
    {
        int suns[8];		/*!< temporary pointers to daughter nodes */
        struct
        {
            MyFloat s[3];		/*!< center of mass of node */
            MyFloat mass;		/*!< mass of node */
            int sibling;		/*!< this gives the next node in the walk in case the current node can be used */
            int nextnode;		/*!< this gives the next node in case the current node needs to be opened */
            MyFloat hmax;			/*!< maximum SPH smoothing length in node. Only used for gas particles */
            MyFloat MaxSoftening;  /* Stores the largest softening in the node. The short-range
                                 * gravitational force solver will check this and use it
                                 * open the node if a particle is closer.*/
        }
        d;
    }
    u;
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
    /* Gives next node in the tree walk for particles and pseudo particles.
     * next node for the actual nodes is stored in Nodes*/
    int * Nextnode;
    /*Allocated length of the Nextnode array*/
    int Nnextnode;
    /*!< gives parent node in tree for every particle */
    int *Father;
} ForceTree;

int force_tree_allocated(const ForceTree * tt);

/* This function propagates changed SPH smoothing lengths up the tree*/
void force_update_hmax(int * activeset, int size, ForceTree * tt);

/*This is the main constructor for the tree structure. Pass in something empty.*/
void force_tree_rebuild(ForceTree * tree, DomainDecomp * ddecomp);

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
    return no < tree->firstnode;
}

static inline int
node_is_node(int no, const ForceTree * tree)
{
    return (no >= tree->firstnode) && (no < tree->lastnode);
}

int
force_get_prev_node(int no, const ForceTree * tb);

int
force_get_next_node(int no, const ForceTree * tb);

int
force_set_next_node(int no, int next, const ForceTree * tb);

int
force_get_father(int no, const ForceTree * tt);

#endif



