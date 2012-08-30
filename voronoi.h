#ifndef HAVE_H_VORONOI
#define HAVE_H_VORONOI

#include <gmp.h>


#define STACKSIZE_TETRA        10000
#define MIN_ALLOC_NUMBER       1000
#define ALLOC_INCREASE_FACTOR  1.1
#define ALLOC_DECREASE_FACTOR  0.7
#define MAX_VORONOI_ITERATIONS 500

#define USEDBITS 52

#if USEDBITS > 31
typedef signed long long int IntegerMapType;
void MY_mpz_set_si(mpz_t dest, signed long long int val);
void MY_mpz_mul_si(mpz_t prod, mpz_t mult, signed long long int val);
void MY_mpz_sub_ui(mpz_t prod, mpz_t mult, unsigned long long int val);
#else
typedef signed long int IntegerMapType;
#define MY_mpz_set_si mpz_set_si
#define MY_mpz_mul_si mpz_mul_si
#define MY_mpz_sub_ui mpz_sub_ui
#endif

#define DOUBLE_to_VORONOIINT(y)   ((IntegerMapType)(((*((long long *) &y)) & 0xFFFFFFFFFFFFFllu) >> (52 - USEDBITS)))

/*
    Prerequisites for this function:
      sizeof(double)==sizeof(unsigned long long)
      doubles must be stored according to IEEE 754
*/
static inline IntegerMapType double_to_voronoiint(double d)
{
   union { double d; unsigned long long ull; } u;
   u.d=d;
   return (u.ull&0xFFFFFFFFFFFFFllu) >> (52 - USEDBITS);
}

static inline double mask_voronoi_int(double x)
{
   union { double d; unsigned long long ull; } u;
   u.d = x;
   u.ull = u.ull & (~((1llu << (52 - USEDBITS)) - 1));
   return u.d;
}

#ifndef TWODIMS

#define EDGE_0 1     /* points 0-1 */
#define EDGE_1 2     /* points 0-2 */
#define EDGE_2 4     /* points 0-3 */
#define EDGE_3 8     /* points 1-2 */
#define EDGE_4 16    /* points 1-3 */
#define EDGE_5 32    /* points 2-3 */
#define EDGE_ALL 63

#define SHAPE_FAC  0.23090085  /* (3.0/5/pow(4pi/3,2.0/3) */

#else

#define EDGE_0 1     /* points 1-2 */
#define EDGE_1 2     /* points 0-2 */
#define EDGE_2 4     /* points 0-1 */
#define EDGE_ALL 7

#define SHAPE_FAC  0.15915494  /* 1/(2*PI) */

#endif

#define HSML_INCREASE_FACTOR 1.3


#ifdef TWODIMS    /* will only be compiled in 2D case */
#define DIMS 2
#else
#define DIMS 3
#endif





extern struct grad_data
{
  double drho[3];
  double dvel[3][3];
  double dpress[3];

}
 *GradExch;

extern struct primexch
{
  MyFloat Pressure;
  MyFloat VelPred[3];
  MyFloat Density;
  MyFloat Mass;
  MyFloat Entropy;
  MyFloat Center[3];
  MyFloat Volume;
#ifdef VORONOI_SHAPESCHEME
  MyFloat W;
#endif
  int task, index;
}
*PrimExch;

typedef struct
{
  double x, y, z;
  double xx, yy, zz;
  int task, index, inactiveflag;
  int ID;
  IntegerMapType ix, iy, iz;
}
point;


typedef struct tetra_data
{
  int     p[DIMS+1];  /* oriented tetrahedron points */
  int     t[DIMS+1];  /* adjacent tetrahedrons, always opposite to corresponding point */
  unsigned char    s[DIMS+1];  /* gives the index of the point in the adjacent tetrahedron that
                                  lies opposite to the common face */

  /* Note: if t[0] == -1, the tetrahedron has been deleted */
}
tetra;

typedef struct tetra_center_data
{
  double  cx, cy, cz;  /* describes circumcircle center */
}
tetra_center;



extern unsigned char *Edge_visited;

typedef struct face_data
{
  int p1, p2;
  double area;
  double cx, cy, cz;  /* center-of-mass of face */

#ifdef VORONOI_SHAPESCHEME
  double T_xx, T_yy, T_zz, T_xy, T_xz, T_yz;
#ifndef TWODIMS
  double g_x, g_y, g_z;
#endif
#endif
}
face;





extern struct list_export_data
{
  unsigned int image_bits;
  int origin, index;
  int nextexport;
}
*ListExports;

extern int Ninlist, MaxNinlist;

extern struct list_P_data
{
  int firstexport, currentexport;

} *List_P;

extern int CountInSphereTests, CountInSphereTestsExact;
extern int CountConvexEdgeTest, CountConvexEdgeTestExact;
extern int CountFlips, Count_1_to_3_Flips2d, Count_2_to_4_Flips2d;
extern int Count_1_to_4_Flips, Count_2_to_3_Flips, Count_3_to_2_Flips, Count_4_to_4_Flips;
extern int Count_EdgeSplits, Count_FaceSplits;
extern int Count_InTetra, Count_InTetraExact;
extern int Largest_N_DP_Buffer;

extern int Ninlist, MaxNinlist;

extern int FlagCompleteMesh;

extern int Ndp;			        /* number of delaunay points */
extern int MaxNdp;			/* maximum number of delaunay points */
extern point *DP;			/* delaunay points */

extern int Ndt;
extern int MaxNdt;			/* number of delaunary tetrahedra */
extern tetra *DT;			/* Delaunay tetrahedra */
extern tetra_center *DTC;               /* circumcenters of delaunay tetrahedra */
extern char *DTF;


extern int Nvf;			/* number of Voronoi faces */
extern int MaxNvf;			/* maximum number of Voronoi faces */
extern face *VF;			/* Voronoi faces */


extern int DPinfinity;

extern double CentralOffsetX, CentralOffsetY, CentralOffsetZ, ConversionFac;

void do_gravity_hydro_half_step(double dt);
void check_for_cut(double pp[3][3], int xaxis, int yaxis, int zaxis, double zval);
void drift_voronoi_face_half_a_step(void);
void recompute_circumcircles_and_faces(void);

void reprocess_edge_faces(tetra * t, int nr);

void make_3d_voronoi_listfaces_check_for_cut(int tt, int nr, int xaxis, int yaxis, int zaxis, double zval);
void calculate_volume_changes(void);


void save_mass_flux_list(void);
void do_gravity_massflux_based_gravitywork(void);

void voronoi_save_old_gravity_forces(void);

void calculate_volume_changes_and_correct(void);

void free_voronoi_mesh(void);

void isothermal_function(double rhostar, double rho, double *F, double *FD);


void make_2d_voronoi_image(int num, int pixels_x, int pixels_y);

void drift_mesh_generator(int i, int time1, double hubble_fac);

void voronoi_mesh(void);
void write_voronoi_mesh(char *fname, int writeTask, int lastTask);

void do_hydro_step(void);
void write_voronoi_mesh(char *fname, int writeTask, int lastTask);
void finalize_output_of_voronoi_geometry(void);
void prepare_output_of_voronoi_geometry(void);
void initialize_and_create_first_tetra(void);
void compute_voronoi_faces_and_volumes(void);
void process_edge_faces_and_volumes(int tt, int nr);

int insert_point(int pp, int ttstart);


void make_an_edge_split(int tt0, int edge_nr, int count, int pp, int *ttlist);

void make_a_face_split(int tt0, int face_nr, int pp, int tt1, int tt2, int qq1, int qq2);


double calculate_tetra_volume(int pp0, int pp1, int pp2, int pp3);
void make_a_4_to_4_flip(int tt, int tip_index, int edge_nr);


void make_a_1_to_4_flip(int pp, int tt0, int tt1, int tt2, int tt3);

void make_a_3_to_2_flip(int tt0, int tt1, int tt2, int tip, int edge, int bottom);

void make_a_2_to_3_flip(int tt0, int tip, int tt1, int bottom, int qq, int tt2);

int Orient3d(int pp0, int pp1, int pp2, int pp3);

int get_tetra(int pp, int *moves, int ttstart, int *flag, int *edgeface_nr);

int InTetra(int tt, int pp, int *edgeface_nr, int *nexttetra);
double InSphere(point * p0, point * p1, point * p2, point * p3, point * p);
void update_circumcircle(int tt);
int test_tetra_orientation(int pp0, int pp1, int pp2, int pp3);
int test_intersect_triangle(point * p0, point * p1, point * p2, point * q, point * s);
double deter4(point * p0, point * p1, point * p2, point * p3);
double deter3(point * p0, point * p1, point * p2);
double deter4_orient(point * p0, point * p1, point * p2, point * p3);
double determinante3(double *a, double *b, double *c);
double deter_special(double *a, double *b, double *c, double *d);
int voronoi_ghost_search_alternative(void);
int voronoi_exchange_evaluate(int target, int mode, int *nexport, int *nsend_local);
int ngb_treefind_voronoi(MyDouble searchcenter[3], MyFloat hsml, int target, int origin,
			 int *startnode, int mode, int *nexport, int *nsend_local, int id);
void compute_circumcircles(void);
int compute_max_delaunay_radius(void);
void check_for_min_distance(void);
void check_links(void);
void check_orientations(void);
void check_tetras(int npoints);
int voronoi_get_local_particles(void);
void check_for_vertex_crossings(void);
void voronoi_calculate_gravity_work_from_potential(int mode);
int convex_edge_test(int tt, int tip, int *edgenr);
void voronoi_update_ghost_potential(void);

void do_hydro_calculations(void);
void update_primitive_variables(void);
void update_cells_with_fluxes(void);
void gradient_init( MyFloat *addr, MyFloat *addr_exch, double *addr_grad, int type );
void scalar_init( MyFloat * addr, MyFloat * addr_mass, int type );
void calculate_green_gauss_gradients(void);
void compute_interface_fluxes(void);
void half_step_evolution(void);
void limit_gradient(double *d, double phi, double min_phi, double max_phi, double *dphi);




void voronoi_exchange_primitive_variables(void);
void voronoi_update_primitive_variables_and_exchange_gradients(void);
void voronoi_exchange_vertex_velocities(void);

int compare_primexch(const void *a, const void *b);



/* 2D voronoi routines */
void check_edge_and_flip_if_needed(int ip, int it);
int get_triangle(int pp, int *moves, int *degenerate_flag, int ttstart);
int InTriangle(point * p0, point * p1, point * p2, point * p);
double InCircle(point * p0, point * p1, point * p2, point * p);
double v2d_deter3(point * p0, point * p1, point * p2);
void make_a_1_to_3_flip(int pp, int tt0, int tt1, int tt2);


double test_triangle_orientation(int pp0, int pp1, int pp2);
void get_circle_center(point * a, point * b, point * c, double *x, double *y);
void do_special_dump(int num);

void make_a_2_to_4_flip(int pp, int tt0, int tt1, int tt2, int tt3, int i0, int j0);

void set_vertex_velocities(void);
void dump_points(void);

void set_integers_for_point(int pp);

int solve_linear_equations(double *m, double *res);


void check_triangles(int npoints);


int InCircle_Quick(int pp0, int pp1, int pp2, int pp);
int InCircle_Errorbound(int pp0, int pp1, int pp2, int pp);

int InCircle_Exact(int pp0, int pp1, int pp2, int pp);

int Orient2d_Exact(int pp0, int pp1, int pp2);
int Orient2d_Quick(int pp0, int pp1, int pp2);


int FindTriangle(int tt, int pp, int *degnerate_flag, int *nexttetra);

int InSphere_Exact(int pp0, int pp1, int pp2, int pp3, int pp);
int InSphere_Quick(int pp0, int pp1, int pp2, int pp3, int pp);
int InSphere_Gauss(int pp0, int pp1, int pp2, int pp3, int pp);
int InSphere_Errorbound(int pp0, int pp1, int pp2, int pp3, int pp);

int Orient3d_Exact(int  pp0, int pp1, int pp2, int pp3);

int Orient3d_Quick(int  pp0, int pp1, int pp2, int pp3);

void make_3d_voronoi_listfaces(int num, int xaxis, int yaxis, int zaxis, double zval);

int count_undecided_tetras(void);
int ngb_treefind_ghost_search(MyDouble searchcenter[3], MyDouble refpos[3],
                              MyFloat hsml, MyFloat maxdist, int target, int origin, int *startnode, int bitflags,
                              int mode, int *nexport, int *nsend_local);
int voronoi_ghost_search_evaluate(int target, int mode, int q, int *nexport, int *nsend_local);
int voronoi_ghost_search(void);
void apply_flux_list(void);
int flux_list_data_compare(const void *a, const void *b);
void voronoi_exchange_ghost_variables(void);
void voronoi_density(void);
void voronoi_hydro_force(void);

#endif


