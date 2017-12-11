#ifndef FORCETREE_H
#define FORCETREE_H

#include "types.h"
/*
 * Variables for Tree
 * ------------------
 */

/*Used in treewalk.c*/
extern struct NODE
{
    MyFloat len;			/*!< sidelength of treenode */
    MyFloat center[3];		/*!< geometrical center of node */

    int father;		/*!< this gives the parent node of each node (or -1 if we have the root node) */
    struct {
        unsigned int InternalTopLevel :1; /* TopLevel and has a child which is also TopLevel*/
        unsigned int TopLevel :1; /* Node corresponding to a toplevel node */
        unsigned int DependsOnLocalMass :1;  /* Intersects with local mass */
        unsigned int MixedSofteningsInNode:1;  /* Softening is mixed, need to open the node */
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
}
*Nodes_base,			/*!< points to the actual memory allocated for the nodes */
    *Nodes;			/*!< this is a pointer used to access the nodes which is shifted such that Nodes[RootNode]
                      gives the first allocated node */

/*Structure containing the Node pointer, and the first and last entries*/
struct TreeBuilder {
    /*Index of first internal node*/
    int firstnode;
    /*Index of first pseudo-particle node*/
    int lastnode;
    /*!< this is a pointer used to access the nodes which is shifted such that Nodes[firstnode]
     *   gives the first allocated node */
    struct NODE *Nodes; 
};

extern int MaxNodes;		/*!< maximum allowed number of internal nodes */
extern int RootNode;      /*!< Index of the first node. Difference between Nodes and Nodes_base. == MaxPart*/

/*Used in domain.c*/
extern int *Nextnode;		/*!< gives next node in tree walk  (nodes array) */
extern int *Father;		/*!< gives parent node in tree (Prenodes array) */

int force_tree_allocated();

void force_update_hmax(int * activeset, int size);
void force_tree_rebuild();

void   force_tree_free(void);
void   dump_particles(void);

static inline int
node_is_pseudo_particle(int no)
{
    return no >= RootNode + MaxNodes;
}

static inline int
node_is_particle(int no)
{
    return no < RootNode;
}

static inline int
node_is_node(int no)
{
    return (no >= RootNode) && (no < RootNode + MaxNodes);
}

int
force_get_prev_node(int no, const struct TreeBuilder tb);

int
force_get_next_node(int no, const struct TreeBuilder tb);

int
force_set_next_node(int no, int next, const struct TreeBuilder tb);

#endif



