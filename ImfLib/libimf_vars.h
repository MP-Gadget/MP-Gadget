#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>


#ifndef LIBIMF_VARS_H
#define LIBIMF_VARS_H


/* /------------------------------------------------\  
 * |                                                |
 * | IMF definition                                 |
 * |                                                |
 * | In this section we actually define what an IMF |
 * | is in this library.                            |
 * \------------------------------------------------/ */

#define LIMF_MAX(a, b)( ((a) > (b)) ? (a) : (b) )
#define LIMF_MIN(a, b)( ((a) < (b)) ? (a) : (b) )

typedef enum {power_law, whatever} IMF_SPEC;
extern char *IMF_Spec_Labels[];
extern IMF_SPEC IMF_Spec;

#define IMF_NSPEC 9 /* defines how many fields must be specified in IMFs' format file for each IMF */
#define INC_BH 0
#define EXC_BH 1

typedef struct
{
  IMF_SPEC type;
  char     *name;
  double (* IMFfunc_byMass)(double, void*);
  double (* IMFfunc_byNum)(double, void*);
  double (* IMFfunc_byEgy)(double, void*);
  double (* getp)(int, double*, double);

  /* Normalization constants; A contains the normalizations 
   * for each segment of a general IMF 
   */
  double Atot, *A; 

  /* [Mm MU] is the mass range over which the IMF is defined */
  double MU, Mm;
  
  /* inf_lifetime = lifetime(MU) 
   * sup_lifetime = lifetime(Mm)
   */
  double inf_lifetime, sup_lifetime;

  /* this array contains the parameters for the IMF */
  double *Params;

  /* This structure contains the slopes of a multi powerlaw IMF */
  struct
  {
    double *masses, *slopes;
  }Slopes;

  /* This strcture contains the kinetic energies of Sn as 
   * a function of mass
   */
  struct
  {
    double *masses, *ekin;
  }EKin;

  /* This structure defines in which interval the stars will not
   * directly end in BH without becoming SN beforehand
   */
  struct
  {
    double *sup, *inf, *list;
  }notBH_ranges;

  int NSlopes,      /* number of segment in the IMF */
    NParams,        /* number of needed parameters  */
    YSet,           /* the yields set */
    NEKin,          /* how many bins for the kinetic energy */
    N_notBH_ranges; /* how many intervals for non-directly-BH */

  /* set if the IMF is timedependent */
  int timedep;
}IMF_Type;

extern IMF_Type *IMFs, *IMFp, IMFu;

/* How many IMFs have been defined */
extern int IMFs_dim;

extern int Nof_TimeDep_IMF;

/* dimension of the integration space */
#define limf_gsl_intspace_dim 10000
extern gsl_integration_workspace *limf_w;

#define LIBIMF_ERR_IMF_FILE        10
#define LIBIMF_ERR_IMF_FILE_FORMAT 11
#define LIBIMF_ERR_IMF_ALLOCATE    12
#define LIBIMF_ERR_INTEGRATION     (-100)

int  read_imfs(char *);
void allocate_IMF_integration_space();

double Renormalize_IMF(IMF_Type *, double, double, int);

void print_IMF(int, char *);
void write_IMF_info(int, FILE*);
void write_SF_info(int, FILE*);

double IntegrateIMF_byNum(double, double, IMF_Type *, int);
double IntegrateIMF_byMass(double, double, IMF_Type *, int);
double IntegrateIMF_byEgy(double, double, IMF_Type *);


double IMFevaluate_byMass_powerlaw(double, void *);
double IMFevaluate_byNum_powerlaw(double, void *);
double IMFevaluate_byEgy_powerlaw(double, void *);

double IMFevaluate_byMass_Larson(double, void *);
double IMFevaluate_byNum_Larson(double, void *);
double IMFevaluate_byEgy_Larson(double, void *);

int get_IMF_SlopeBin(double);

int not_in_BHrange(int, double *);

//typedef EXTERNIMF double (*(double, double, IMF_Type *, int));

typedef double (*EXTERNALIMF)();

extern EXTERNALIMF *externalIMFs_byMass;
extern EXTERNALIMF *externalIMFs_byNum;
extern EXTERNALIMF *externalIMFs_byEgy;
extern char **externalIMFs_names;

extern int NexternalIMFs;


int initialize_externalIMFs(int);
int set_externalIMF(int, char *, EXTERNALIMF, EXTERNALIMF, EXTERNALIMF);
int search_externalIMF_name(char *);

#endif
