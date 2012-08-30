#ifndef SUBFIND_H
#define SUBFIND_H




typedef struct
{
  double r;
  double mass;
}
sort_r2list;

void subfind_col_find_candidates(int totgrouplen);
void subfind_reshuffle_free(void);

void read_hsml_files(float *Values, int count, enum iofields blocknr, long long nskip);
void read_hsml_files_voronoi(float *Values, int count, enum iofields blocknr, long long nskip);

void subfind_distlinklist_get_two_heads(long long ngb_index1, long long ngb_index2, 
					long long *head,      long long *head_attach);

int subfind_reshuffle_compare_ID_list_ID(const void *a, const void *b);
void  read_hsml_table(void);
void get_hsml_file(long long nskip, int count, int *filenr, int *n_to_read, int *n_to_skip);
void read_subfind_ids(void);
int subfind_reshuffle_compare_ID_list_ID(const void *a, const void *b);

int subfind_distlinklist_get_tail_set_tail_increaselen(long long index, long long *tail, long long newtail);

void subfind_exchange(void);
void subfind_col_save_candidates_task(int totgrouplen, int num);
void subfind_col_load_candidates(int num);

void subfind(int num);
int subfind_overdensity_evaluate_dispersion(int target, int mode, int *nexport, int *nsend_local);
int subfind_contamination_treefind(MyDouble *searchcenter, MyFloat hsml, int target, int *startnode,
                                      int mode, int *nexport, int *nsend_local, double *Mass);
int subfind_ovderdens_treefind_dispersion(MyDouble searchcenter[3], MyFloat hsml, int target, int *startnode,
                                          int mode, int *nexport, int *nsend_local);
int subfind_contamination_evaluate(int target, int mode, int *nexport, int *nsend_local);
void subfind_contamination(void);
int subfind_force_treeevaluate_potential(int target, int mode, int *nexport, int *nsend_local);
#ifdef DENSITY_SPLIT_BY_TYPE
void subfind_density(int j);
#else
void subfind_density(void);
#endif
void subfind_overdensity(void);
int subfind_overdensity_evaluate(int target, int mode, int *nexport, int *nsend_local);
double subfind_ovderdens_treefind(MyDouble *searchcenter, MyFloat hsml, int target, int *startnode,
				  int mode, int *nexport, int *nsend_local);
void subfind_save_densities(int num);
void subfind_save_local_densities(int num);
#ifdef DENSITY_SPLIT_BY_TYPE
void subfind_setup_smoothinglengths(int j);
int subfind_density_evaluate(int target, int mode, int *nexport, int *nsend_local, int tp);
int subfind_ngb_treefind_linkpairs(MyDouble *searchcenter, double hsml, int target, int *startnode, int mode,
                                 double *hmax, int *nexport, int *nsend_local);
#else
void subfind_setup_smoothinglengths(void);
int subfind_density_evaluate(int target, int mode, int *nexport, int *nsend_local);
#endif
void subfind_save_local_catalogue(int num);
void subfind_save_final(int num);
int subfind_linkngb_evaluate(int target, int mode, int *nexport, int *nsend_local);
int subfind_ngb_treefind_linkngb(MyDouble *searchcenter, double hsml, int target, int *startnode, int mode,
                                 double *hmax, int *nexport, int *nsend_local);
int subfind_ngb_treefind_nearesttwo(MyDouble *searchcenter, double hsml, int target, int *startnode, int mode,
                                    double *hmax, int *nexport, int *nsend_local);
void subfind_distribute_particles(int mode);
void subfind_unbind_independent_ones(int count);
void subfind_distribute_groups(void);
void subfind_potential_compute(int num, struct unbind_data * d, int phase, double weakly_bound_limit);
void subfind_process_group_collectively(int num);
int subfind_col_unbind(struct unbind_data *d, int num);
void subfind_col_determine_sub_halo_properties(struct unbind_data *d, int num, double *mass, 
					       double *pos, double *vel, double *cm, double *veldisp,
					       double *vmax, double *vmaxrad, double *spin, MyIDType *mostboundid,
					       double *halfmassrad, double *submasstab);
void subfind_col_determine_R200(double hmr, double center[3],
				double *m_Mean200, double *r_Mean200,
				double *m_Crit200, double *r_Crit200, double *m_TopHat, double *r_TopHat);
void subfind_find_linkngb(void);
int subfind_loctree_treebuild(int npart, struct unbind_data *mp);
void subfind_loctree_update_node_recursive(int no, int sib, int father);
double subfind_loctree_treeevaluate_potential(int target);
void subfind_loctree_copyExtent(void);
double subfind_locngb_treefind(MyDouble *xyz, int desngb, double hguess);
void subfind_loctree_findExtent(int npart, struct unbind_data *mp);
int subfind_locngb_treefind_variable(MyDouble *searchcenter, double hguess);
size_t subfind_loctree_treeallocate(int maxnodes, int maxpart);
void subfind_loctree_treefree(void);
void subfind_find_nearesttwo(void);
int subfind_nearesttwo_evaluate(int target, int mode, int *nexport, int *nsend_local);
int subfind_process_group_serial(int gr, int offset);
int subfind_unbind(struct unbind_data *ud, int len);
void subfind_determine_sub_halo_properties(struct unbind_data *ud, int num, double *mass, 
					   double *pos, double *vel, double *cm, double *veldisp,
					   double *vmax, double *vmaxrad, double *spin, MyIDType *mostboundid, 
                                           double *halfmassrad, double *mass_tab);
int subfind_compare_P_origindex(const void *a, const void *b);
int subfind_compare_P_GrNr_DM_Density(const void *a, const void *b);
int subfind_compare_P_GrNrGrNr(const void *a, const void *b);
int subfind_locngb_compare_key(const void *a, const void *b);
int subfind_compare_serial_candidates_subnr(const void *a, const void *b);
int subfind_compare_serial_candidates_rank(const void *a, const void *b);
int subfind_compare_dens(const void *a, const void *b);
int subfind_compare_energy(const void *a, const void *b);
int subfind_compare_grp_particles(const void *a, const void *b);
int subfind_compare_candidates_boundlength(const void *a, const void *b);
int subfind_compare_candidates_nsubs(const void *a, const void *b);
int subfind_compare_serial_candidates_boundlength(const void *a, const void *b);
int subfind_compare_P_submark(const void *a, const void *b);
int subfind_compare_dist_rotcurve(const void *a, const void *b);
int subfind_compare_binding_energy(const void *a, const void *b);
int subfind_compare_unbind_data_Potential(const void *a, const void *b);
int subfind_compare_unbind_data_seqnr(const void *a, const void *b);
int subfind_compare_densities(const void *a, const void *b);
int subfind_compare_candidates_rank(const void *a, const void *b);
int subfind_ngb_compare_dist(const void *a, const void *b);
int subfind_compare_hsml_data(const void *a, const void *b);
int subfind_compare_ID_list(const void *a, const void *b);
int subfind_compare_SubGroup_GrNr_SubNr(const void *a, const void *b);
int subfind_compare_candidates_subnr(const void *a, const void *b);
void subfind_poll_for_requests(void);
long long subfind_distlinklist_setrank_and_get_next(long long index, long long *rank);
long long subfind_distlinklist_get_rank(long long index);
void subfind_distlinklist_set_next(long long index, long long next);
void subfind_distlinklist_add_particle(long long index);
void subfind_distlinklist_add_bound_particles(long long index, int nsub);
void subfind_distlinklist_mark_particle(long long index, int target, int submark);
long long subfind_distlinklist_get_next(long long index);
long long subfind_distlinklist_get_head(long long index);
void subfind_distlinklist_set_headandnext(long long index, long long head, long long next);
void subfind_distlinklist_set_tailandlen(long long index, long long tail, int len);
void subfind_distlinklist_get_tailandlen(long long index, long long *tail, int *len);
void subfind_distlinklist_set_all(long long index, long long head, long long tail, int len, long long next);
int subfind_distlinklist_get_ngb_count(long long index, long long *ngb_index1, long long *ngb_index2);
long long subfind_distlinklist_set_head_get_next(long long index, long long head);


extern int Ncollective;
extern int MaxNsubgroups;
extern int Nsubgroups;
extern int TotNsubgroups;

extern struct subgroup_properties
{
  int Len;
  int GrNr;
  int SubNr;
  int SubParent;    
  unsigned int Offset;
  MyIDType SubMostBoundID;
  double Mass;
  double SubVelDisp;
  double SubVmax;
  double SubVmaxRad;
  double SubHalfMass;
  double Pos[3];
  double CM[3];
  double Vel[3];
  double Spin[3];
#ifdef SAVE_MASS_TAB
  double MassTab[6];
#endif
} *SubGroup;


extern struct nearest_r2_data
{
  double dist[2];
}
*R2Loc;

extern struct nearest_ngb_data
{
  long long index[2];
  int count;
}
*NgbLoc;


extern struct r2data
{
  MyFloat r2;
  int   index;
}
*R2list;

extern double *Dist2list;



#endif
