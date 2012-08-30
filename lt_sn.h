
#define Z_ROUND_OFF_ERROR (1e-7)
#define qag_INT_KEY 4


/* ...........: useful quantities :...................... */

int    FirstChemActiveParticle, *NextChemActiveParticle;

static double UnitMassFact;
static int    SEvInfo_grain;

#ifdef LT_TRACK_CONTRIBUTES
Contrib       contrib;
float         IIcontrib[LT_NMetP], Iacontrib[LT_NMetP], AGBcontrib[LT_NMetP];

#define NULL_CONTRIB(c)     {bzero((void*)c, sizeof(Contrib));       }
#define NULL_EXPCONTRIB(c)  {bzero((void*)c, sizeof(float)*LT_NMetP);}
#endif


#define TIME_INTERP_SIZE 10000
static double aarray[TIME_INTERP_SIZE], tarray[TIME_INTERP_SIZE];

#define agefact 1000000000
#define max(x, y) (((x) > (y)) ? (x) : (y))
#define min(x, y) (((x) < (y)) ? (x) : (y))


/* :::............................................................... */

#define IaY( set, element, metallicity, mass) (SnIaYields[set][element][metallicity*IaMbins_dim[set ] + mass])
#define IIY( set, element, metallicity, mass) (SnIIYields[set][element][metallicity*IIMbins_dim[set ] + mass])
#define AGBY(set, element, metallicity, mass) (AGBYields [set][element][metallicity*AGBMbins_dim[set] + mass])

#define EVOLVE           16

/*
     As long as we use the upper part (> bit 8 so far ) of P[i].Type to store the ordinal number of
     the star, we don't need any hash function.
     NOTE: by this way we are assuming that no more than INT_MAX / 2^8 stars will be active at the
	   same time for each processor ; for 32-bits u-integer this results in ~16.7e6 stars per proc,
	   that will be surely sufficient up to quite big runs.
	   We insert in count_evolving_stars a check on the number of active stars, so that the
	   flagging is stopped if this limit is crossed. The stars excluded will evolve in the next
	   timestep.
*/

#define LOCAL            0
#define BUFFER           1

/* :::............................................................... */

gsl_function F;

#ifdef LT_SEvDbg
int    do_spread_dbg;
double weight_sum, *weightlist;
double ngb_sum, *ngblist;
double metals_spread[LT_NMet], tot_metalsspread[LT_NMet];
#endif


/* ...........: stats and log files related quantities :...................... */

#define INUM          8
#define SN_NeighFind  0
#define SN_NeighCheck 1
#define SN_Comm       2
#define SN_Calc       3
#define SN_Imbalance  4
#define SN_info       5
#define SN_Spread     6

double infos[INUM], sum_infos[INUM];
double SN_Find, sumSN_Find;

long long tot_starsnum, tot_gasnum;
long long tot_allstarsnum;

#ifdef LT_SEv_INFO

#define MIN_INIT_VALUE 1e6

/* Chemical Enrichment history */

#define SPECIES     4
#define SPEC_snIa   0
#define SPEC_snII   2
#define SPEC_agb    4
#define SPEC_IRA    6                                                    /* used if LT_LOCAL_IRA is not defined */

#define SN_INUM     2
#define SN_num      0
#define SN_egy      1

#define AL_INUM_sum 4
#define MEAN_ngb    0
#define MEAN_sl     1
#define NUM_uspread 2
#define NUM_nspread 3

#define AL_INUM_min 2
#define MIN_ngb     0
#define MIN_sl      1

#define AL_INUM_max 2
#define MAX_ngb     0
#define MAX_sl      1

#ifdef LT_SEv_INFO_DETAILS_onSPREAD
#define SP_INUM     7
#define SP_Sum      0
#define SP_W        1
#define SP_N        2
#define SP_W_min    3
#define SP_N_min    4
#define SP_W_max    5
#define SP_N_max    6
#endif

#define GET3d(Dz, A, x, y, z)     A[(SPECIES)*2*(Dz) * (x) + (Dz) * (y) + (z)]
#define SET3d(Dz, A, x, y, z, V) {A[(SPECIES)*2*(Dz) * (x) + (Dz) * (y) + (z)]  = (V);}
#define ADD3d(Dz, A, x, y, z, V) {A[(SPECIES)*2*(Dz) * (x) + (Dz) * (y) + (z)] += (V);}

/* log the ejected mass for each metal by SnIa, SnII and nebulae                    */
/* in [SPEC_XX + 1] there will be the losses (in the case that losses are possible) */
/* prototype: [SFd_dim][SPECIES * 2][LT_NMet]                                       */
/* double *Zmass, *tot_Zmass;  <-- defined as global variables                      */

/* log number and energy released for each specie                                   */
/* double *SNdata, *tot_SNdata; <-- defined as global variables                     */

/* log the number of neighbours and the spreading lenght                            */
/* the min,max and mean will be logged in ALdata_min _max and _sum                  */
double ALdata_sum[AL_INUM_sum], tot_ALdata_sum[AL_INUM_sum];
double ALdata_min[AL_INUM_min], tot_ALdata_min[AL_INUM_min];
double ALdata_max[AL_INUM_max], tot_ALdata_max[AL_INUM_max];


#ifdef LT_SEv_INFO_DETAILS_onSPREAD
double SP[SP_INUM], tot_SP[SP_INUM];
#endif

#define S_INUM_sum (2 * LT_NMet) + 5
#define MEAN_Zs     0                                                    /* mean abundance for each element in gas */
#define MEAN_Z      LT_NMet                                              /* mean metallicity in gas                */
#define MEAN_egyf   MEAN_Z + 1                                           /* mean (internal egy) / (sn egy) value   */
#define NUM_egyf    MEAN_egyf + 1
#define MEAN_Zsstar NUM_egyf + 1                                         /* mean abundance for each element in stars */
#define MEAN_Zstar  MEAN_Zsstar + LT_NMet                                /* mean metallicity in stars              */
#define NUM_star    MEAN_Zstar + 1

#define S_INUM_min  3
#define MIN_Z       0                                                    /* min metallicity in gas                 */
#define MIN_Zstar   1                                                    /* min metallicity in stars               */
#define MIN_egyf    2                                                    /* min (internal egy) / (sn egy) value    */

#define S_INUM_max  3
#define MAX_Z       0                                                    /* max metallicity in gas                 */
#define MAX_Zstar   1                                                    /* min metallicity in stars               */
#define MAX_egyf    2                                                    /* max (internal egy) / (sn egy) value    */

double Stat_sum[S_INUM_sum], tot_Stat_sum[S_INUM_sum];
double Stat_min[S_INUM_min], tot_Stat_min[S_INUM_min];
double Stat_max[S_INUM_max], tot_Stat_max[S_INUM_max];

#if defined(LT_EJECTA_IN_HOTPHASE)
double SpreadEgy[2], tot_SpreadEgy[2];
double SpreadMinMaxEgy[2][2], tot_SpreadMinMaxEgy[2][2];
double AgbFrac[3], tot_AgbFrac[3];
double SpecEgyChange[3], tot_SpecEgyChange[3];
double CFracChange[4], tot_CFracChange[4];
#endif

int    spreading_on;
#endif

#ifdef LT_SEv_INFO_DETAILS
struct details
{
  MyIDType ID;
  char     type;
  float    Data[LT_NMet + 3];                                            /* stores all metals + start_time + end_time */
}
 *Details;

#define DetailsZ 3
int    DetailsPos;
#endif

double dmax1, dmax2;

/* :::............................................................... */



struct metaldata_index
{
  MyFloat Pos[3];
  MyFloat L;
  float   Metals[LT_NMet];
  double  energy;
  double  weight;
  int     Type, Task, Index, IndexGet, SFi;

  int     NodeList[NODELISTLENGTH];
  
#ifdef LT_TRACK_CONTRIBUTES
  Contrib contrib;
#endif

  //#if defined(LT_EJECTA_IN_HOTPHASE) || defined(LT_HOT_EJECTA) || defined(LT_SNegy_IN_HOTPHASE)
  float   LMMass;
  //#endif

#ifdef LT_SEvDbg
  MyIDType ID;
#endif

} *MetalDataIn, *MetalDataGet;

#ifdef LT_SEvDbg_global
struct spreaddata_index
{
  int    Task, Index, IndexGet;
  double weight, numngb;
} *MetalDataSpreadReport;
#endif

/* ...........: declare internal functions :...................... */

int    INLINE_FUNC metaldata_index_compare(const void *, const void *);


#ifdef DOUBLEPRECISION
double INLINE_FUNC myfloor               (double);
#else
float  INLINE_FUNC myfloor               (double);
#endif
void               count_evolving_stars  (int *, int *);
int                build_SN_Stepping     (int);
double INLINE_FUNC lifetime              (double);
int    INLINE_FUNC iterate               (int, int, MyFloat, double *);
double INLINE_FUNC inner_integrand       (double, void *);
double INLINE_FUNC get_da_dota           (double, void *);

/*- stellar evolution -*/
double             perform_stellarevolution_operations(int, int*,  double*, double*);
double             stellarevolution                   (int, double *, double *);
int                spread_evaluate                    (int, int, float*, float, double, int*, int*);

double INLINE_FUNC dying_mass            (double);
double INLINE_FUNC dm_dt                 (double, double);

void               get_metallicity_stat  (void);

/*- snIa -*/
double INLINE_FUNC sec_dist              (double, double);
double             get_SnIa_product      (int, int, double *, double *, double, double);
double INLINE_FUNC nRSnIa                (double, void *);
double INLINE_FUNC mRSnIa                (double, void *);

/*- snII and AGB -*/
double             get_AGB_product       (int, int, double *, double, double, double *);
double             get_SnII_product      (int, int, double *, double *, double, double, double *);
double INLINE_FUNC nRSnII                (double, void *);
double INLINE_FUNC mRSnII                (double time, void *p);
double INLINE_FUNC zmRSnII               (double m, void *p);
double INLINE_FUNC ejectaSnII            (double, void *);
double INLINE_FUNC ztRSnII               (double, void *);

void               get_Egy_and_Beta      (double, double *, double *, SF_Type *);
void               calculate_ShortLiving_related(SF_Type *, int);

/* :::............................................................... */
/* ****************************************************************** */
