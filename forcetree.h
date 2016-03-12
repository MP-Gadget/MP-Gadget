#ifndef FORCETREE_H
#define FORCETREE_H
#include "evaluator.h"

/*
 * Variables for Tree
 * ------------------
 */

/*Used in restart.c*/
extern struct NODE
{
#ifdef OPENMP_USE_SPINLOCK
    pthread_spinlock_t SpinLock;
#endif

    MyFloat len;			/*!< sidelength of treenode */
    MyFloat center[3];		/*!< geometrical center of node */

#ifdef ADAPTIVE_GRAVSOFT_FORGAS
    MyFloat maxsoft;		/*!< hold the maximum gravitational softening of particle in the
                              node if the ADAPTIVE_GRAVSOFT_FORGAS option is selected */
#endif
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
    int Ti_current;
}
*Nodes_base,			/*!< points to the actual memory allocted for the nodes */
    *Nodes;			/*!< this is a pointer used to access the nodes which is shifted such that Nodes[All.MaxPart]
                      gives the first allocated node */

/*Used in restart.c*/
extern struct extNODE
{
    MyDouble dp[3];
    MyFloat vs[3];
    MyFloat vmax;
    MyFloat divVmax;
    MyFloat hmax;			/*!< maximum SPH smoothing length in node. Only used for gas particles */
    int Ti_lastkicked;
    int Flag;
}
*Extnodes, *Extnodes_base;


extern int MaxNodes;		/*!< maximum allowed number of internal nodes */
/*Used in restart.c*/
extern int Numnodestree;	/*!< number of (internal) nodes in each tree */

/*Used in domain.c*/
extern int *Nextnode;		/*!< gives next node in tree walk  (nodes array) */
extern int *Father;		/*!< gives parent node in tree (Prenodes array) */

#define BITFLAG_TOPLEVEL                   0
#define BITFLAG_DEPENDS_ON_LOCAL_MASS      1
#define BITFLAG_MAX_SOFTENING_TYPE         2	/* bits 2-4 */
#define BITFLAG_MIXED_SOFTENINGS_IN_NODE   5
#define BITFLAG_INTERNAL_TOPLEVEL          6
#define BITFLAG_MULTIPLEPARTICLES          7
#define BITFLAG_NODEHASBEENKICKED          8
#define BITFLAG_INSIDE_LINKINGLENGTH       9

#define BITFLAG_MASK  ((1 << BITFLAG_MULTIPLEPARTICLES) + (1 << BITFLAG_MIXED_SOFTENINGS_IN_NODE) + (7 << BITFLAG_MAX_SOFTENING_TYPE))
#define maskout_different_softening_flag(x) (x & (1 << BITFLAG_MIXED_SOFTENINGS_IN_NODE))
#define extract_max_softening_type(x) ((x >> BITFLAG_MAX_SOFTENING_TYPE) & 7)

struct gravitydata_in
{
    int NodeList[NODELISTLENGTH];
    MyFloat Pos[3];
    int Type;
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
    MyFloat Soft;
#endif
    MyFloat OldAcc;
} ;

struct gravitydata_out
{
    MyDouble Acc[3];
    MyDouble Potential;
    int Ninteractions;
};


void force_flag_localnodes(void);

void *gravity_primary_loop(void *p);
void *gravity_secondary_loop(void *p);

int force_treeev_shortrange(int target, int mode, 
        struct gravitydata_in * input,
        struct gravitydata_out * output,
        LocalEvaluator * lv, void * unused);


int force_treeev_potential(int target, int type, int *nexport, int *nsend_local);
int force_treeev_potential_shortrange(int target, int mode, int *nexport, int *nsend_local);


void force_drift_node(int no, int time1);
int force_drift_node_full(int no, int time1, int blocking);
     
void force_tree_discardpartials(void);
void force_treeupdate_pseudos(int);
void force_update_pseudoparticles(void);

void force_kick_node(int i, MyFloat *dv);

void force_dynamic_update(void);
void force_dynamic_update_node(int no, int mode, MyFloat *minbound, MyFloat *maxbound);

void force_update_hmax(void);
void force_update_hmax_of_node(int no, int mode);

void force_finish_kick_nodes(void);

void force_create_empty_nodes(int no, int topnode, int bits, int x, int y, int z, int *nodecount, int *nextfree);

void force_exchange_pseudodata(void);

void force_insert_pseudo_particles(void);

void   force_costevaluate(void);
int    force_getcost_single(void);
int    force_getcost_quadru(void);
void   force_resetcost(void);
void   force_setupnonrecursive(int no);
void   force_treeallocate(int maxnodes, int maxpart);  
int    force_treebuild(int npart, struct unbind_data *mp);
int    force_treebuild_single(int npart, struct unbind_data *mp);

void force_treebuild_simple();

int    force_treeev_direct(int target, int mode);


void   force_treefree(void);
void   force_update_node(int no, int flag);

void   force_update_node_recursive(int no, int sib, int father);

void   force_update_size_of_parent_node(int no);

void   dump_particles(void);

MyFloat  ngb_select_closest(int k, int n, MyFloat *arr, int *ind);
void   ngb_treeallocate(int npart);
void   ngb_treebuild(void);


void   ngb_treefree(void);
void   ngb_treesearch(int);
void   ngb_treesearch_pairs(int);
void   ngb_update_nodes(void);
void   ngb_treesearch_notsee(int no);

int ngb_treefind_fof_primary(MyDouble searchcenter[3], MyFloat hsml, int target, int *startnode, int mode,
			    int *nexport, int *nsend_local);

enum NgbTreeFindSymmetric {
    NGB_TREEFIND_SYMMETRIC,
    NGB_TREEFIND_ASYMMETRIC,
};

int ngb_treefind_threads(MyDouble searchcenter[3], MyFloat hsml, int target, int *startnode,
		       int mode, LocalEvaluator * lv, enum NgbTreeFindSymmetric symmetric, int ptypemask);

#endif



