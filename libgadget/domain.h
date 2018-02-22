#ifndef DOMAIN_H
#define DOMAIN_H

/*These variables are used externally in forcetree.c.
 * DomainTask is also used in treewalk and NTopLeaves is used in gravpm.c*/

extern struct topnode_data
{
    peano_t StartKey;		/*!< first Peano-Hilbert key in top-level node */
    int Daughter;			/*!< index of first daughter cell (out of 8) of top-level node */
    short int Shift;    /*!< level = log 2 of number of Peano-Hilbert mesh-cells represented by top-level node */
    int Leaf ;			/*!< if the node is a leaf, this gives its index in topleaf_data */
} *TopNodes;

extern struct topleaf_data {
    int Task;
    union {
        int topnode; /* used during domain_decompose for balancing the decomposition */
        int treenode; /* used during life span of the tree for looking up in the tree Nodes */
    };
} * TopLeaves;

extern struct task_data {
    int StartLeaf;
    int EndLeaf;
} * Tasks;

extern int MaxTopNodes;	        /*!< Maximum number of nodes in the top-level tree used for domain decomposition */

extern int NTopNodes, NTopLeaves;

void domain_decompose_full(void);
void domain_maintain(void);

int
domain_get_topleaf(const peano_t key);

#endif
