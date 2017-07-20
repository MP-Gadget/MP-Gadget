#ifndef FORCETREE_H
#define FORCETREE_H
#include "treewalk.h"

/*
 * Variables for Tree
 * ------------------
 */

/*Used in treewalk.c*/
extern struct NODE
{
    MyFloat len;			/*!< sidelength of treenode */
    MyFloat center[3];		/*!< geometrical center of node */

    morton_t morton; /* shifting morton by (shift + 3) is zero if particle is in the node */
    int shift; /* shifting morton & 7 will give the child position. */

#ifdef ADAPTIVE_GRAVSOFT_FORGAS
    MyFloat maxsoft;		/*!< hold the maximum gravitational softening of particle in the
                              node if the ADAPTIVE_GRAVSOFT_FORGAS option is selected */
#endif
    MyFloat hmax;			/*!< maximum SPH smoothing length in node. Only used for gas particles */
    union
    {
        int suns[8];		/*!< temporary pointers to daughter nodes */
        struct
        {
            MyFloat s[3];		/*!< center of mass of node */
            MyFloat mass;		/*!< mass of node */
            unsigned int bitflags;	/*!< flags certain node properties */
            int sibling;		/*!< this gives the next node in the walk in case the current node can be used */
            int nextnode;		/*!< this gives the next node in case the current node needs to be opened */
            int father;		/*!< this gives the parent node of each node (or -1 if we have the root node) */
        }
        d;
    }
    u;
}
*Nodes_base,			/*!< points to the actual memory allocated for the nodes */
    *Nodes;			/*!< this is a pointer used to access the nodes which is shifted such that Nodes[All.MaxPart]
                      gives the first allocated node */

extern int MaxNodes;		/*!< maximum allowed number of internal nodes */

/*Used in domain.c*/
extern int *Nextnode;		/*!< gives next node in tree walk  (nodes array) */
extern int *Father;		/*!< gives parent node in tree (Prenodes array) */

#define BITFLAG_TOPLEVEL                   0
#define BITFLAG_DEPENDS_ON_LOCAL_MASS      1  /* Intersects with local mass */
#define BITFLAG_MAX_SOFTENING_TYPE         2  /* bits 2-4 */
#define BITFLAG_MIXED_SOFTENINGS_IN_NODE   5
#define BITFLAG_INTERNAL_TOPLEVEL          6  /* INTERNAL tree nodes and toplevel*/
#define BITFLAG_MULTIPLEPARTICLES          7
#define BITFLAG_INSIDE_LINKINGLENGTH       9

#define BITFLAG_MASK  ((1 << BITFLAG_MULTIPLEPARTICLES) + (1 << BITFLAG_MIXED_SOFTENINGS_IN_NODE) + (7 << BITFLAG_MAX_SOFTENING_TYPE))
#define maskout_different_softening_flag(x) (x & (1 << BITFLAG_MIXED_SOFTENINGS_IN_NODE))
#define extract_max_softening_type(x) ((x >> BITFLAG_MAX_SOFTENING_TYPE) & 7)

int force_tree_allocated();

void force_update_hmax(int * activeset, int size);
void force_tree_rebuild();

void   force_tree_free(void);
void   dump_particles(void);

int
force_find_enclosing_node(int i);

int
force_get_prev_node(int no);

int
force_get_next_node(int no);

void
force_remove_node(int no);

int
force_set_next_node(int no, int next);

void
force_insert_particle(int i);

#endif



