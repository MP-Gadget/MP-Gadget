

#ifndef LTH
#define LTH

#include "ImfLib/libimf_vars.h"

                                                                         /* :----------------------------------- : */
                                                                         /* : general variables and defines      : */
                                                                         /* :----------------------------------- : */

#ifndef INLINE_FUNC
#ifdef  INLINE
#define INLINE_FUNC inline
#else
#define INLINE_FUNC
#endif
#endif


#ifndef LT_NMet
#error you MUST define LT_NMet at compilation time
#endif

#ifndef STELLARAGE
#define STELLARAGE
#endif

#ifndef SFR
#define SFR
#endif

#ifndef MOREPARAMS
#define MOREPRAMS
#endif

#ifdef METALS
#error LT_STELLAREVOLUTION is incompatible with METALS
#endif

#ifdef SNIA_HEATING
#error LT_STELLAREVOLUTION is incompatible with SNIA_HEATING
#endif


#define LT_NMetP (LT_NMet-1)                                             /*!< this is the number of elements stored in the particle structure */
#define EVOLVE_SN 0                                                      /*!< used in calling the evolution routine                           */
#define NO_METAL -5.0                                                    /*!< the threshold where the tables are no longer valid              */

extern int TimeBinCountStars[TIMEBINS];


                                                                         /*   auxiliary files                                                  */
extern FILE   *FdSnInit,                                                 /*!<   contains messages & details from sn initialization             */
              *FdWarn,                                                   /*!<   contains warnings from sn initialization                       */
              *FdSn,                                                     /*!<   contains details about sn explosion and metal production       */
              *FdSnLost,                                                 /*!<   log how many sn havn't been able to spread                     */
              *FdMetals,                                                 /*!<   contains details about global enrichment in the box            */
              *FdSnDetails,                                              /*!<   log ALL details about sn explosions                            */
              *FdTrackW;                                                 /*!<   log details about winds                                        */

extern double cosmic_time;                                               /*!< in every timestep this is the current cosmic time in Gyrs        */
extern float  xclouds, Temperature;                                      /*!< at the exit of cooling_and_starformation contain x and T         */
extern int    search_for_metalspread;


#ifdef LT_STARBURSTS
#define SB_DENSITY 1
#define SB_DENTROPY 2
#endif

#if defined(LT_SMOOTH_Z) || defined(LT_METAL_COOLING)
extern int ignore_failure_in_convert_u;                                  /*!< if set do not stop in case of no convergence in find_abundances  */
#endif

/* ------------------- *
 *   gsl integration   *
 * ------------------- */

#define gsl_ws_dim 10000                                                 /*   variables used to interface with gsl                             */
extern gsl_error_handler_t       *old_error_handler;
extern int                       gsl_status;
extern gsl_function              F;

extern gsl_integration_workspace *w;

extern gsl_interp_accel          *accel;
extern gsl_spline                *spline;

                                                                        /* : ------------------------------------------------- : */
                                                                        /* : varialbes related to metal species                : */
                                                                        /* : ------------------------------------------------- : */

extern int    UseSnII, UseSnIa, UseAGB;

extern char   *MetNames[LT_NMet];                                        /*!< contains the labels of metals that will be tracked (read from
                                                                           the file metals.dat                                                 */
                                                                 
extern double MetSolarValues[LT_NMet];
extern int    Hyd,                                                       /*!< the positions of some peculiar elements in the metal array       */
              Hel, 
              Oxygen,
              Iron,
              FillEl;

#ifdef  UM_METAL_COOLING
extern int    Carbon,
              Nitrogen,
              Magnesium,
              Silicon;
#endif


                                                                         /* : -------------------------------------------------- : */
                                                                         /* : variables and structures related to star formation : */
                                                                         /* : -------------------------------------------------- : */


extern double *PhysDensTh;                                               /*!< array of density thresholds, one for each "SF type"             */
extern double *FEVP;                                                     /*!< array of evaporation factors, one for each "SF type"            */
extern int    sfrrate_filenum;

extern double *sfrrates;                                                 /*!< array of star formation rates for every "SF type"               */
extern double *totsfrrates;

extern double *sum_sm, *sum_mass_stars, *total_sm, *total_sum_mass_stars;/*!< to collect sf rates data                                        */

extern double MxSfrTimescale_rescale_by_densityth;


extern int    IsThere_TimeDep_SF;
extern int    IsThere_ZDep_SF;


typedef struct
{
  char   *identifier;
  double referenceZ_toset_SF_DensTh;

  double egyShortLiv_MassTh, egyShortLiv_TimeTh;
  double metShortLiv_MassTh, metShortLiv_TimeTh;

  double ShortLiv_MassTh, ShortLiv_TimeTh;

  double *PhysDensThresh, *FEVP;
  double MaxSfrTimescale_rescale_by_densityth;
  double EgySpecSN, FactorSN, totFactorSN, metFactorSN;
  double MassFrac_inIRA, NumFrac_inIRA, totResFrac;
  double MaxSfrTimescale;

  int    SFTh_Zdep;                                                      /*!< set dependency of rho thresh on metallicity */

  int    SFTh_Tdep;                                                      /*!< set dependency of rho thresh on time */

  double IRA_erg_per_g, TOT_erg_per_g;
  
  double WindEnergy, WindEnergyFraction, WindEfficiency;                 /*!< set quantities for winds */

  int    Generations;                                                    /*!< set # of stellar generations spawned */

  int    nonZeroIRA;                                                     /*!< record whether or not there are metals in IRA */
  
  int    IMFi;
  char   *IMFname;
} SF_Type;

extern SF_Type *SFs, *SFp, SFu;
extern int     SFi, SFs_dim;

extern FILE    *FdIMFin, *FdIMF;


#ifdef LT_TRACK_CONTRIBUTES
  /*
   * LT_N_INT_PACKING is calculated once you declare at
   * compile time both LT_NBits and LT_NIMFs, that are
   * defined here below.
   * LT_N_INT_PACKING_1IMF is also calculated, that is
   * defined as LT_N_INT_PACKING / Number_of_IMFs_Used
   *
   *
   * LT_N_INT_PACKING can be calculated as follows:
   *
   * contributes run from 0 to 1. say that you want to track them
   * with a precision o nbits: then, you can map those number
   * using (number / 2^-nbits) between 0 and 2^nbits.
   * if you multiply (number / 2^-nbits) by 10^exp_max, you will
   * increase the number if significant digits in the integer
   * representation, then reducing the error when reconstructing
   * the original fraction.
   *
   * if you want to use nbits for the significant part and 2 bits
   * for the exponent, the number of bytes that you need is:
   *
   * ((nbits + 2) * N_species * N_elements * N_imf) / 8
   *
   * because you can infer the last species from
   *
   * using N_species = 3 (snIa, CC sn, agb stars) this becomes
   *
   *   >>>> LT_N_INT_PACKING = ((nbits + 2) * 3 * N_elements * N_imf) / 8  <<<<
   *
   * simply using float will put the memory requirement larger
   * by a factor 32 / (nbits+2);
   */
#ifndef LT_Nbits
#define LT_Nbits 20
#endif
#ifndef LT_power10_Nbits
#define LT_power10_Nbits 2
#endif
typedef struct
{
/*   unsigned     II_el0_imf0 : LT_Nbits; */
/*   unsigned  IIexp_el0_imf0 : LT_power10_Nbits; */
  unsigned     II_el1_imf0 : LT_Nbits;
  unsigned  IIexp_el1_imf0 : LT_power10_Nbits;
  unsigned     II_el2_imf0 : LT_Nbits;
  unsigned  IIexp_el2_imf0 : LT_power10_Nbits;
  unsigned     II_el3_imf0 : LT_Nbits;
  unsigned  IIexp_el3_imf0 : LT_power10_Nbits;
  unsigned     II_el4_imf0 : LT_Nbits;
  unsigned  IIexp_el4_imf0 : LT_power10_Nbits;
  unsigned     II_el5_imf0 : LT_Nbits;
  unsigned  IIexp_el5_imf0 : LT_power10_Nbits;
  unsigned     II_el6_imf0 : LT_Nbits;
  unsigned  IIexp_el6_imf0 : LT_power10_Nbits;
  unsigned     II_el7_imf0 : LT_Nbits;
  unsigned  IIexp_el7_imf0 : LT_power10_Nbits;
/*   unsigned     II_el8_imf0 : LT_Nbits; */
/*   unsigned  IIexp_el8_imf0 : LT_power10_Nbits; */

/*   unsigned     Ia_el0_imf0 : LT_Nbits; */
/*   unsigned  Iaexp_el0_imf0 : LT_power10_Nbits; */
  unsigned     Ia_el1_imf0 : LT_Nbits;
  unsigned  Iaexp_el1_imf0 : LT_power10_Nbits;
  unsigned     Ia_el2_imf0 : LT_Nbits;
  unsigned  Iaexp_el2_imf0 : LT_power10_Nbits;
  unsigned     Ia_el3_imf0 : LT_Nbits;
  unsigned  Iaexp_el3_imf0 : LT_power10_Nbits;
  unsigned     Ia_el4_imf0 : LT_Nbits;
  unsigned  Iaexp_el4_imf0 : LT_power10_Nbits;
  unsigned     Ia_el5_imf0 : LT_Nbits;
  unsigned  Iaexp_el5_imf0 : LT_power10_Nbits;
  unsigned     Ia_el6_imf0 : LT_Nbits;
  unsigned  Iaexp_el6_imf0 : LT_power10_Nbits;
  unsigned     Ia_el7_imf0 : LT_Nbits;
  unsigned  Iaexp_el7_imf0 : LT_power10_Nbits;
/*   unsigned     Ia_el8_imf0 : LT_Nbits; */
/*   unsigned  Iaexp_el8_imf0 : LT_power10_Nbits; */

/*   unsigned    AGB_el0_imf0 : LT_Nbits; */
/*   unsigned AGBexp_el0_imf0 : LT_power10_Nbits; */
  unsigned    AGB_el1_imf0 : LT_Nbits;
  unsigned AGBexp_el1_imf0 : LT_power10_Nbits;
  unsigned    AGB_el2_imf0 : LT_Nbits;
  unsigned AGBexp_el2_imf0 : LT_power10_Nbits;
  unsigned    AGB_el3_imf0 : LT_Nbits;
  unsigned AGBexp_el3_imf0 : LT_power10_Nbits;
  unsigned    AGB_el4_imf0 : LT_Nbits;
  unsigned AGBexp_el4_imf0 : LT_power10_Nbits;
  unsigned    AGB_el5_imf0 : LT_Nbits;
  unsigned AGBexp_el5_imf0 : LT_power10_Nbits;
  unsigned    AGB_el6_imf0 : LT_Nbits;
  unsigned AGBexp_el6_imf0 : LT_power10_Nbits;
  unsigned    AGB_el7_imf0 : LT_Nbits;
  unsigned AGBexp_el7_imf0 : LT_power10_Nbits;
/*   unsigned    AGB_el8_imf0 : LT_Nbits; */
/*   unsigned AGBexp_el8_imf0 : LT_power10_Nbits; */

  /* add here below more blocks if more ifms are
   * being used; just change imfX appropriately.
   */

/* /\*   unsigned     II_el0_imf1 : LT_Nbits; *\/ */
/* /\*   unsigned  IIexp_el0_imf1 : LT_power10_Nbits; *\/ */
/*   unsigned     II_el1_imf1 : LT_Nbits; */
/*   unsigned  IIexp_el1_imf1 : LT_power10_Nbits; */
/*   unsigned     II_el2_imf1 : LT_Nbits; */
/*   unsigned  IIexp_el2_imf1 : LT_power10_Nbits; */
/*   unsigned     II_el3_imf1 : LT_Nbits; */
/*   unsigned  IIexp_el3_imf1 : LT_power10_Nbits; */
/*   unsigned     II_el4_imf1 : LT_Nbits; */
/*   unsigned  IIexp_el4_imf1 : LT_power10_Nbits; */
/*   unsigned     II_el5_imf1 : LT_Nbits; */
/*   unsigned  IIexp_el5_imf1 : LT_power10_Nbits; */
/*   unsigned     II_el6_imf1 : LT_Nbits; */
/*   unsigned  IIexp_el6_imf1 : LT_power10_Nbits; */
/*   unsigned     II_el7_imf1 : LT_Nbits; */
/*   unsigned  IIexp_el7_imf1 : LT_power10_Nbits; */
/* /\*   unsigned     II_el8_imf1 : LT_Nbits; *\/ */
/* /\*   unsigned  IIexp_el8_imf1 : LT_power10_Nbits; *\/ */

/*   unsigned     Ia_el1_imf1 : LT_Nbits; */
/*   unsigned  Iaexp_el1_imf1 : LT_power10_Nbits; */
/*   unsigned     Ia_el2_imf1 : LT_Nbits; */
/*   unsigned  Iaexp_el2_imf1 : LT_power10_Nbits; */
/*   unsigned     Ia_el3_imf1 : LT_Nbits; */
/*   unsigned  Iaexp_el3_imf1 : LT_power10_Nbits; */
/*   unsigned     Ia_el4_imf1 : LT_Nbits; */
/*   unsigned  Iaexp_el4_imf1 : LT_power10_Nbits; */
/*   unsigned     Ia_el5_imf1 : LT_Nbits; */
/*   unsigned  Iaexp_el5_imf1 : LT_power10_Nbits; */
/*   unsigned     Ia_el6_imf1 : LT_Nbits; */
/*   unsigned  Iaexp_el6_imf1 : LT_power10_Nbits; */
/*   unsigned     Ia_el7_imf1 : LT_Nbits; */
/*   unsigned  Iaexp_el7_imf1 : LT_power10_Nbits; */

/*   unsigned    AGB_el1_imf1 : LT_Nbits; */
/*   unsigned AGBexp_el1_imf1 : LT_power10_Nbits; */
/*   unsigned    AGB_el2_imf1 : LT_Nbits; */
/*   unsigned AGBexp_el2_imf1 : LT_power10_Nbits; */
/*   unsigned    AGB_el3_imf1 : LT_Nbits; */
/*   unsigned AGBexp_el3_imf1 : LT_power10_Nbits; */
/*   unsigned    AGB_el4_imf1 : LT_Nbits; */
/*   unsigned AGBexp_el4_imf1 : LT_power10_Nbits; */
/*   unsigned    AGB_el5_imf1 : LT_Nbits; */
/*   unsigned AGBexp_el5_imf1 : LT_power10_Nbits; */
/*   unsigned    AGB_el6_imf1 : LT_Nbits; */
/*   unsigned AGBexp_el6_imf1 : LT_power10_Nbits; */
/*   unsigned    AGB_el7_imf1 : LT_Nbits; */
/*   unsigned AGBexp_el7_imf1 : LT_power10_Nbits; */

} Contrib;

extern unsigned int   Packing_Factor;
extern unsigned int   *Power_Factors, Max_Power, PowerBase;
extern unsigned int   TrackMask;
extern float          Max_Packed_Int,
                      UnPacking_Factor,
                      MaxError,
                      MinPackableFraction,
                      MaxRaisableFraction,
                      PowerBaseLog_inv;

extern float          *save_fractionII,  *fractionII , *nfractionII;
extern float          *save_fractionIa,  *fractionIa , *nfractionIa;
extern float          *save_fractionAGB, *fractionAGB, *nfractionAGB;

extern double         *dfractionII;
extern double         *dfractionIa;
extern double         *dfractionAGB;

#endif /* LT_TRACK_CONTRIBUTES */


                                                                         /* : ---------------------------------------------- : */
                                                                         /* : Sn related                                     : */
                                                                         /* : ---------------------------------------------- : */

#define FOREVER -0.1

extern struct SDtype
{
  int    Zbin, Zdim, Mdim, Yset;
  double Zstar;
  double *ZArray, *MArray;
  double *Y;
  int    ExtrDir;
  int    ExtrZbin, ExtrZdim, ExtrMdim, ExtrYset;
  double *ExtrZArray, *ExtrMArray;
  double *ExtrY;
} SD;


extern double ***SNtimesteps;
extern int    *LongLiv_Nsteps, *ShortLiv_Nsteps, *Nsteps;


/*
 * SnIaData defines the yields for SnIa.
 * we define it as a pointer so that in the future we can easily extend the
 * number of parameters on which they depend (e.g. mass of the system, time).
 * currently they are fixed
 */
extern double ***SnIaYields;
/*
 * it is common that yields tables use the same mass array and Z array for
 * all elements. in this case, we simply use IaZbins, IaMbins to store these 
 * common data. otherwise (different mass/Z array for each element), you can
 * just use the already existent Yield structure, just allocating as twice as
 * the amount of memory needed for the yields' value and using half of the
 * memory to store also the array's value. You can also use SnIaY_dim to store
 * the dimensions.
 */
extern double **IaZbins, **IaMbins;
extern int    *IaZbins_dim, *IaMbins_dim, *SnIaY_dim[LT_NMet];
extern int    *NonProcOn_Ia;

/*
 * the same as for SnIa.
 */
extern double ***SnIIYields, **SnIIEj;
extern double ***SnII_ShortLiv_Yields;
extern double **IIZbins, **IIMbins;
extern int    *IIZbins_dim, *IIMbins_dim, *SnIIY_dim[LT_NMet];
extern int    *NonProcOn_II;

extern double ***AGBYields, **AGBEj;
extern double **AGBZbins, **AGBMbins;
extern int    *AGBZbins_dim, *AGBMbins_dim, *AGB_dim[LT_NMet];
extern int    *NonProcOn_AGB;



                                                                         /* : ------------------------------- : */
                                                                         /* :  SEvDbg                         : */
                                                                         /* : ------------------------------- : */

#ifdef LT_SEvDbg
extern unsigned int FirstID;
extern int          checkFirstID;
extern int          *do_spread_dbg_list;
#endif


                                                                         /* : ------------------------------- : */
                                                                         /* :  SEv_INFO                       : */
                                                                         /* : ------------------------------- : */
#ifdef LT_SEv_INFO

#define SEvInfo_GRAIN 15

extern double *Zmass, *tot_Zmass;
extern double *SNdata, *tot_SNdata;

#ifdef LT_LOG_ENRICH_DETAILS
extern FILE   *FdEnrichDetails_temp, *FdLogEnrichDetails;
#endif

#ifdef LT_SMOOTH_SIZE
extern int    AvgSmoothN;
extern double AvgSmoothSize;
extern double MinSmoothSize;
extern double MaxSmoothSize;

extern int    AvgSmoothNgb;
extern int    MinSmoothNgb;
extern int    MaxSmoothNgb;    

/* extern float *HashCorrection;   */
/* extern int   *HashBlock, HashN; */
#endif

#ifdef LT_SEvDbg
extern FILE *FdMetSumCheck;
#endif

#ifdef WINDS
extern FILE *FdWinds;
#endif

#ifdef LT_EXTEGY_INFO
extern FILE *FdExtEgy;
#endif

#ifdef LT_SEv_INFO_DETAILS_onSPREAD
extern FILE *FdSPinfo;
#endif

#endif /* close SEv_INFO */

#ifdef LT_SEv_INFO_DETAILS
extern double DetailsW[LT_NMet], DetailsWo[LT_NMet];
#endif


                                                                         /* ------------------------------------------------ */



                                                                          /* : ---------------------------------------- : */
                                                                          /* :  METAL COOLING                           : */
                                                                          /* : ---------------------------------------- : */   

extern int    TBins, ZBins;
extern double TMin, TMax, ZMin, ZMax;
extern double *CoolTvalue, *CoolZvalue, **CoolingTables;
extern double *ThInst_onset;
#ifdef LT_DAMP_METALCOOLING
extern double DAMP_ZC_Fact;
#endif

#endif
