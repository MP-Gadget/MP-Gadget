#ifndef FORCETREE_H
#define FORCETREE_H
#include "evaluator.h"

#ifndef INLINE_FUNC
#ifdef INLINE
#define INLINE_FUNC inline
#else
#define INLINE_FUNC
#endif
#endif


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
#if defined(UNEQUALSOFTENINGS) || defined(SCALARFIELD)
    int Type;
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
    MyFloat Soft;
#endif
#endif
    MyFloat OldAcc;
} ;

struct gravitydata_out
{
    MyLongDouble Acc[3];
#ifdef EVALPOTENTIAL
    MyLongDouble Potential;
#endif
#ifdef DISTORTIONTENSORPS
    MyLongDouble tidal_tensorps[3][3];
#endif
    int Ninteractions;
};


void force_flag_localnodes(void);

void *gravity_primary_loop(void *p);
void *gravity_secondary_loop(void *p);


int force_treeevaluate(int target, int mode, 
        struct gravitydata_in * input,
        struct gravitydata_out * output,
        LocalEvaluator * lv, void * unused);
int force_treeevaluate_ewald_correction(int target, int mode, 
        struct gravitydata_in * input,
        struct gravitydata_out * output,
        LocalEvaluator * lv, void * unused);
int force_treeevaluate_shortrange(int target, int mode, 
        struct gravitydata_in * input,
        struct gravitydata_out * output,
        LocalEvaluator * lv, void * unused);


int force_treeevaluate_potential(int target, int type, int *nexport, int *nsend_local);
int force_treeevaluate_potential_shortrange(int target, int mode, int *nexport, int *nsend_local);


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

void force_add_star_to_tree(int igas, int istar);

void   force_costevaluate(void);
int    force_getcost_single(void);
int    force_getcost_quadru(void);
void   force_resetcost(void);
void   force_setupnonrecursive(int no);
void   force_treeallocate(int maxnodes, int maxpart);  
int    force_treebuild(int npart, struct unbind_data *mp);
int    force_treebuild_single(int npart, struct unbind_data *mp);

int    force_treeevaluate_direct(int target, int mode);


void   force_treefree(void);
void   force_update_node(int no, int flag);

void   force_update_node_recursive(int no, int sib, int father);

void   force_update_size_of_parent_node(int no);

void   dump_particles(void);

MyFloat  INLINE_FUNC ngb_periodic(MyFloat x);
MyFloat  INLINE_FUNC ngb_periodic_longbox(MyFloat x);
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
int ngb_clear_buf(MyDouble searchcenter[3], MyFloat hguess, int numngb);
void ngb_treefind_flagexport(MyDouble searchcenter[3], MyFloat hguess);

int ngb_treefind_blackhole(MyDouble searchcenter[3], MyFloat hsml, int target, int *startnode, int mode,
			  int *nexport, int *nsend_local);


int ngb_treefind_pairs(MyDouble searchcenter[3], MyFloat hsml, int target, int *startnode,
		       int mode, int *nexport, int *nsend_local);
int ngb_treefind_variable(MyDouble searchcenter[3], MyFloat hsml, int target, int *startnode, int mode,
 			  int *nexport, int *nsend_local);

enum NgbTreeFindSymmetric {
    NGB_TREEFIND_SYMMETRIC,
    NGB_TREEFIND_ASYMMETRIC,
};

int ngb_treefind_threads(MyDouble searchcenter[3], MyFloat hsml, int target, int *startnode,
		       int mode, LocalEvaluator * lv, int *ngblist, enum NgbTreeFindSymmetric symmetric, int ptypemask);

#endif



