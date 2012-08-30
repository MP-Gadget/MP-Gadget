#include "lt.h"

int TimeBinCountStars[TIMEBINS];

FILE *FdSnInit, *FdWarn, *FdSn, *FdSnLost, *FdMetals, *FdSnDetails, *FdTrackW;

double cosmic_time;

float xclouds, Temperature;

int search_for_metalspread;


#if defined(LT_SMOOTH_Z) || defined(LT_METAL_COOLING)
int ignore_failure_in_convert_u;
#endif

int UseSnII, UseSnIa, UseAGB;


gsl_error_handler_t *old_error_handler;

int gsl_status;

gsl_function F;

gsl_integration_workspace *w;

gsl_interp_accel *accel;

gsl_spline *spline;



int UseSnII, UseSnIa, UseAGB;

char *MetNames[LT_NMet];


double MetSolarValues[LT_NMet];

int Hyd, Hel, Oxygen, Iron, FillEl;
#ifdef UM_METAL_COOLING
int Carbon, Nitrogen, Magnesium, Silicon;
#endif


double *PhysDensTh;		/*!< array of density thresholds, one for each "SF type"             */

double *FEVP;			/*!< array of evaporation factors, one for each "SF type"            */

int sfrrate_filenum;

double *sfrrates;		/*!< array of star formation rates for every "SF type"               */

double *totsfrrates;

double *sum_sm, *sum_mass_stars, *total_sm, *total_sum_mass_stars;	/*!< to collect sf rates data                                        */

double MxSfrTimescale_rescale_by_densityth;


int IsThere_TimeDep_SF;

int IsThere_ZDep_SF;

SF_Type *SFs, *SFp, SFu;

int SFi, SFs_dim;

FILE *FdIMFin, *FdIMF;


#ifdef LT_TRACK_CONTRIBUTES
unsigned int Packing_Factor;

unsigned int *Power_Factors, Max_Power, PowerBase;

unsigned int TrackMask;

float Max_Packed_Int, UnPacking_Factor, MaxError, MinPackableFraction, MaxRaisableFraction, PowerBaseLog_inv;

float *save_fractionII, *fractionII, *nfractionII;

float *save_fractionIa, *fractionIa, *nfractionIa;

float *save_fractionAGB, *fractionAGB, *nfractionAGB;

double *dfractionII;

double *dfractionIa;

double *dfractionAGB;

#endif /* LT_TRACK_CONTRIBUTES */


									 /* : ---------------------------------------------- : */
									 /* : Sn related                                     : */
									 /* : ---------------------------------------------- : */

struct SDtype SD;

double ***SNtimesteps;

int *LongLiv_Nsteps, *ShortLiv_Nsteps, *Nsteps;

double ***SnIaYields;

double **IaZbins, **IaMbins;

int *IaZbins_dim, *IaMbins_dim, *SnIaY_dim[LT_NMet];

int *NonProcOn_Ia;


double ***SnIIYields, **SnIIEj;

double ***SnII_ShortLiv_Yields;

double **IIZbins, **IIMbins;

int *IIZbins_dim, *IIMbins_dim, *SnIIY_dim[LT_NMet];

int *NonProcOn_II;


double ***AGBYields, **AGBEj;

double **AGBZbins, **AGBMbins;

int *AGBZbins_dim, *AGBMbins_dim, *AGB_dim[LT_NMet];

int *NonProcOn_AGB;




									 /* : ------------------------------- : */
									 /* :  SEvDbg                         : */
									 /* : ------------------------------- : */

#ifdef LT_SEvDbg
unsigned int FirstID;

int checkFirstID;

int *do_spread_dbg_list;
#endif


									 /* : ------------------------------- : */
									 /* :  SEv_INFO                       : */
									 /* : ------------------------------- : */
#ifdef LT_SEv_INFO

double *Zmass, *tot_Zmass;

double *SNdata, *tot_SNdata;

#ifdef LT_LOG_ENRICH_DETAILS
FILE *FdEnrichDetails_temp, *FdLogEnrichDetails;
#endif

#ifdef LT_SMOOTH_SIZE
int AvgSmoothN;

double AvgSmoothSize;

double MinSmoothSize;

double MaxSmoothSize;

int AvgSmoothNgb;

int MinSmoothNgb;

int MaxSmoothNgb;

/* extern float *HashCorrection;   */
/* extern int   *HashBlock, HashN; */
#endif

#ifdef LT_SEvDbg
FILE *FdMetSumCheck;
#endif

#ifdef WINDS
FILE *FdWinds;
#endif

#ifdef LT_EXTEGY_INFO
FILE *FdExtEgy;
#endif

#ifdef LT_SEv_INFO_DETAILS_onSPREAD
FILE *FdSPinfo;
#endif

#endif /* close SEv_INFO */

#ifdef LT_SEv_INFO_DETAILS
double DetailsW[LT_NMet], DetailsWo[LT_NMet];
#endif


									 /* ------------------------------------------------ */



									  /* : ---------------------------------------- : */
									  /* :  METAL COOLING                           : */
									  /* : ---------------------------------------- : */

int TBins, ZBins;

double TMin, TMax, ZMin, ZMax;

double *CoolTvalue, *CoolZvalue, **CoolingTables;

double *ThInst_onset;

#ifdef LT_DAMP_METALCOOLING
double DAMP_ZC_Fact;
#endif
