/* Definitions and declarations for Metallicity Model  */

#ifdef CS_MODEL
#ifndef H_CS_MODEL
#define H_CS_MODEL

#define MAXITER_HOT 50


/* constants */
#define Nab 8      /* Number of Sutherland & Dopita tables  */
#define delta 0.05 /* Interval of cooling tables  */
#define Nmass 10		/* Number of mass intervals on the tables */
#define Nelements 13		/* Number of chemical elements on the tables */


extern int Flag_phase;  

/* input/output */
extern FILE *fpcool;


#ifdef CS_TESTS
extern FILE *FdEgyTest;
extern FILE *FdPromTest;
extern FILE *FdSNTest;
#endif
#if defined(CS_SNI) || defined(CS_SNII)
extern FILE *FdSN;
#endif
#ifdef CS_FEEDBACK
extern FILE *FdPromotion;
extern double SN_Energy;
#endif


#ifdef CS_SNII
extern double Raiteri_COEFF_1;
extern double Raiteri_COEFF_2;
extern double Raiteri_COEFF_3;
extern double Raiteri_COEFF_4;
extern double Raiteri_COEFF_5;
#endif

#ifdef CS_TESTS
extern double InternalEnergy;
extern double Energy_cooling;
#ifdef CS_FEEDBACK
extern double Energy_promotion;
extern double Energy_feedback, Energy_reservoir;
#endif
#endif

/* variables */
extern int *Nlines, cont_sm;
extern  double **logLambda_i;
extern  double **logT_i;
extern double **yy, **yield1, **yield2, **yield3, **yield4, **yield5;
extern double metal, sm2, FeHgas;

extern float **nsimfww;
extern double XH, yhelium;
extern int numenrich;







double cs_CoolingRate_SD(double logT, double rho, double *nelec);

void cs_read_coolrate_table(void);
double cs_get_Lambda_SD(double logT, double abund);
void cs_read_yield_table(void);
double cs_SNII_yields(int index);
void cs_imf(void);

/* function prototypes */
void cs_cooling_and_starformation(void);

void cs_enrichment(void);
int  cs_enrichment_evaluate(int target, int mode, int *nexport, int *nsend_local);
void cs_flag_SN_starparticles(void);
double cs_integrated_time(int indice, double time_hubble_a);
void cs_promotion(void);
double cs_SNI_yields(int indice);
void cs_energy_test(void);
void cs_copy_densities(void);
void cs_find_low_density_tail(void);
int cs_compare_density_values(const void *a, const void *b);


void cs_find_hot_neighbours(void);
int cs_hotngbs_evaluate(int target, int mode, int *nexport, int *nsend_local);


void cs_update_weights(void);
int cs_update_weight_evaluate(int target, int mode, int *nexport, int *nsend_local);

int cs_ngb_treefind_variable_phases(MyDouble searchcenter[3], MyFloat hsml, int target, int *startnode, int mode,
                                    int *nexport, int *nsend_local);
int cs_ngb_treefind_variable_decoupling(MyDouble searchcenter[3], MyFloat hsml, int target,
                                        int *startnode, MyFloat densityold,
                                        MyFloat entropy, MyFloat *vel, 
                                        int mode, int *nexport, int *nsend_local);

int cs_ngb_treefind_variable_decoupling_threads(MyDouble searchcenter[3], MyFloat hsml, int target,
					int *startnode, MyFloat densityold,
					MyFloat entropy, MyFloat * vel,
					int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist);

int cs_ngb_treefind_hotngbs(MyDouble searchcenter[3], MyFloat hsml, int target, int *startnode, MyFloat entropy, 
                            int mode, int *nexport, int *nsend_local);

int cs_ngb_treefind_pairs(MyDouble searchcenter[3], MyFloat hsml, int target, int *startnode,
                          MyFloat density, MyFloat entropy, MyFloat *vel,
                          int mode, int *nexport, int *nsend_local);
int cs_ngb_treefind_pairs_threads(MyDouble searchcenter[3], MyFloat hsml, int target, int *startnode,
                          MyFloat density, MyFloat entropy, MyFloat *vel,
                          int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist);                          
                          
/* functions */
void cs_find_low_density_tail(void);


#endif
#endif      




