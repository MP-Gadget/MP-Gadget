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

#ifdef ADAPTIVE_GRAVSOFT_FORGAS
    MyFloat maxsoft;		/*!< hold the maximum gravitational softening of particle in the
                              node if the ADAPTIVE_GRAVSOFT_FORGAS option is selected */
#endif
    int father;		/*!< this gives the parent node of each node (or -1 if we have the root node) */
    union
    {
        int suns[8];		/*!< temporary pointers to daughter nodes */
        struct
        {
            MyFloat s[3];		/*!< center of mass of node */
            MyFloat mass;		/*!< mass of node */
            struct {
                unsigned int TopLevel :1;
                unsigned int DependsOnLocalMass :1;  /* Intersects with local mass */
                unsigned int MaxSofteningType :3; /* bits 2-4 */
                unsigned int MixedSofteningsInNode :1;
                unsigned int InternalTopLevel :1; /* INTERNAL tree nodes and toplevel*/
                unsigned int MultipleParticles :1;
            };
            int sibling;		/*!< this gives the next node in the walk in case the current node can be used */
            int nextnode;		/*!< this gives the next node in case the current node needs to be opened */
            MyFloat hmax;			/*!< maximum SPH smoothing length in node. Only used for gas particles */
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

int force_tree_allocated();

void force_update_hmax(int * activeset, int size);
void force_tree_rebuild();

void   force_tree_free(void);
void   dump_particles(void);

int
force_get_prev_node(int no);

int
force_get_next_node(int no);

int
force_set_next_node(int no, int next, const int firstnode, const int lastnode);

#endif



