/*! \file allvars.h
 *  \brief declares global variables.
 *
 *  This file declares all global variables. Further variables should be added here, and declared as
 *  'extern'. The actual existence of these variables is provided by the file 'allvars.c'. To produce
 *  'allvars.c' from 'allvars.h', do the following:
 *
 *     - Erase all #define statements
 *     - add #include "allvars.h"
 *     - delete all keywords 'extern'
 *     - delete all struct definitions enclosed in {...}, e.g.
 *        "extern struct global_data_all_processes {....} All;"
 *        becomes "struct global_data_all_processes All;"
 */

#include "allvars.h"



#ifdef PERIODIC
MyDouble boxSize, boxHalf, inverse_boxSize;

#ifdef LONG_X
MyDouble boxSize_X, boxHalf_X, inverse_boxSize_X;
#else
#endif
#ifdef LONG_Y
MyDouble boxSize_Y, boxHalf_Y, inverse_boxSize_Y;
#else
#endif
#ifdef LONG_Z
MyDouble boxSize_Z, boxHalf_Z, inverse_boxSize_Z;
#else
#endif
#endif

#ifdef FIX_PATHSCALE_MPI_STATUS_IGNORE_BUG
MPI_Status mpistat;
#endif

/*********************************************************/
/*  Global variables                                     */
/*********************************************************/



int ThisTask;			/*!< the number of the local processor  */
int NTask;			/*!< number of processors */
int PTask;			/*!< note: NTask = 2^PTask */

#ifdef INVARIANCETEST
int World_ThisTask;
int World_NTask;
int Color;
MPI_Comm MPI_CommLocal;
#endif

int NumForceUpdate;		/*!< number of active particles on local processor in current timestep  */
int64_t GlobNumForceUpdate;
int NumSphUpdate;		/*!< number of active SPH particles on local processor in current timestep  */

int MaxTopNodes;		/*!< Maximum number of nodes in the top-level tree used for domain decomposition */

int RestartFlag;		/*!< taken from command line used to start code. 0 is normal start-up from
				   initial conditions, 1 is resuming a run from a set of restart files, while 2
				   marks a restart from a snapshot file. */
int RestartSnapNum;

int *Exportflag;		/*!< Buffer used for flagging whether a particle needs to be exported to another process */
int *Exportnodecount;
int *Exportindex;

int *Send_offset, *Send_count, *Recv_count, *Recv_offset, *Sendcount;

int FirstActiveParticle;
int *NextActiveParticle;

int TimeBinCount[TIMEBINS];
int TimeBinCountSph[TIMEBINS];
int TimeBinActive[TIMEBINS];

int FirstInTimeBin[TIMEBINS];
int LastInTimeBin[TIMEBINS];
int *NextInTimeBin;
int *PrevInTimeBin;

size_t HighMark_run, HighMark_domain, HighMark_gravtree,
  HighMark_pmperiodic, HighMark_pmnonperiodic, HighMark_sphdensity, HighMark_sphhydro;

#ifdef SFR
double TimeBinSfr[TIMEBINS];
#endif

#ifdef BLACK_HOLES
double TimeBin_BH_mass[TIMEBINS];
double TimeBin_BH_dynamicalmass[TIMEBINS];
double TimeBin_BH_Mdot[TIMEBINS];
double TimeBin_BH_Medd[TIMEBINS];
double TimeBin_GAS_Injection[TIMEBINS];
#endif

char DumpFlag = 1;

size_t AllocatedBytes;
size_t HighMarkBytes;
size_t FreeBytes;

int Flag_FullStep;		/*!< Flag used to signal that the current step involves all particles */


int GlobFlag;


int NumPart;			/*!< total number of particles on the LOCAL processor */
int N_sph;			/*!< number of gas particles on the LOCAL processor  */
int N_bh;			/*!< number of bh particles on the LOCAL processor  */

#ifdef SINKS
int NumSinks;
#endif

int64_t Ntype[6];		/*!< total number of particles of each type */
int NtypeLocal[6];		/*!< local number of particles of each type */

gsl_rng *random_generator;	/*!< the random number generator used */


#ifdef SFR
int Stars_converted;		/*!< current number of star particles in gas particle block */
#endif


double TimeOfLastTreeConstruction;	/*!< holds what it says */

int *Ngblist;			/*!< Buffer to hold indices of neighbours retrieved by the neighbour search
				   routines */
double *R2ngblist;

double DomainCorner[3], DomainCenter[3], DomainLen, DomainFac;
int *DomainStartList, *DomainEndList;



double *DomainWork;
int *DomainCount;
int *DomainCountSph;
int *DomainTask;
int *DomainNodeIndex;
int *DomainList, DomainNumChanged;

struct topnode_data *TopNodes;

int NTopnodes, NTopleaves;


double RndTable[RNDTABLE];

/* variables for input/output , usually only used on process 0 */


char ParameterFile[100];	/*!< file name of parameterfile used for starting the simulation */

FILE *FdInfo,			/*!< file handle for info.txt log-file. */
 *FdEnergy,			/*!< file handle for energy.txt log-file. */
 *FdTimings,			/*!< file handle for timings.txt log-file. */
 *FdCPU;			/*!< file handle for cpu.txt log-file. */

#ifdef SCFPOTENTIAL
FILE *FdSCF;
#endif

#ifdef SFR
FILE *FdSfr;			/*!< file handle for sfr.txt log-file. */
FILE *FdSfrDetails;
#endif

#ifdef RADTRANSFER
FILE *FdRad;			/*!< file handle for radtransfer.txt log-file. */
FILE *FdRadNew;			/*!< file handle for radtransferNew.txt log-file. */
#endif

#ifdef DISTORTIONTENSORPS
#ifdef PETAPM
FILE *FdTidaltensor;		/*!< file handle for Tidaltensor.txt log-file. */
#endif
FILE *FdCaustics;		/*!< file handle for Caustics.txt log-file. */
#endif

#ifdef BLACK_HOLES
FILE *FdBlackHoles;		/*!< file handle for blackholes.txt log-file. */
FILE *FdBlackHolesDetails;
#endif


#ifdef FORCETEST
FILE *FdForceTest;		/*!< file handle for forcetest.txt log-file. */
#endif

#ifdef DARKENERGY
FILE *FdDE;			/*!< file handle for darkenergy.txt log-file. */
#endif

#ifdef XXLINFO
FILE *FdXXL;			/*!< file handle for xxl.txt log-file. */

#ifdef MAGNETIC
double MeanB;

#ifdef TRACEDIVB
double MaxDivB;
#endif
#endif
#ifdef TIME_DEP_ART_VISC
double MeanAlpha;
#endif
#endif

/*! This structure contains data which is the SAME for all tasks (mostly code parameters read from the
 * parameter file).  Holding this data in a structure is convenient for writing/reading the restart file, and
 * it allows the introduction of new global variables in a simple way. The only thing to do is to introduce
 * them into this structure.
 */
struct global_data_all_processes All;

#ifdef _OPENMP
uint64_t BlockedParticleDrifts = 0;
uint64_t BlockedNodeDrifts = 0;
uint64_t TotalParticleDrifts = 0;
uint64_t TotalNodeDrifts = 0;
#endif
/*! This structure holds all the information that is
 * stored for each particle of the simulation.
 */
struct particle_data *P;	/*!< holds particle data on local processor */



/* the following struture holds data that is stored for each SPH particle in addition to the collisionless
 * variables.
 */
struct sph_particle_data * SphP;	/*!< holds SPH particle data on local processor */
struct bh_particle_data * BhP;	/*!< holds BH particle data on local processor */

/* global state of system
*/
struct state_of_system SysState, SysStateAtStart, SysStateAtEnd;


/* Various structures for communication during the gravity computation.
 */

struct data_index *DataIndexTable;	/*!< the particles to be exported are grouped
					   by task-number. This table allows the
					   results to be disentangled again and to be
					   assigned to the correct particle */

struct data_nodelist *DataNodeList;


/*! Header for the standard file format.
 */
struct io_header header;	/*!< holds header for snapshot files */





/*
 * Variables for Tree
 * ------------------
 */

struct NODE *Nodes_base,	/*!< points to the actual memory allocted for the nodes */
 *Nodes;			/*!< this is a pointer used to access the nodes which is shifted such that Nodes[All.MaxPart]
				   gives the first allocated node */


struct extNODE *Extnodes, *Extnodes_base;


int MaxNodes;			/*!< maximum allowed number of internal nodes */
int Numnodestree;		/*!< number of (internal) nodes in each tree */


int *Nextnode;			/*!< gives next node in tree walk  (nodes array) */
int *Father;			/*!< gives parent node in tree (Prenodes array) */

#ifdef STATICNFW
double Rs, R200;
double Dc;
double RhoCrit, V200;
double fac;
#endif

#if defined (UM_CHEMISTRY) || defined (UM_METAL_COOLING)
/* --==[ link with LT_ stuffs]==-- */
float *um_ZsPoint, um_FillEl_mu, um_mass;

/* char *PT_Symbols; */
/* double *PT_Masses; */
/* int NPT; */

/* double **II_AvgFillNDens, **IIShLv_AvgFillNDens, **Ia_AvgFillNDens, **AGB_AvgFillNDens; */
#endif

#if defined (CHEMISTRY) || defined (UM_CHEMISTRY)
/* ----- chemistry part ------- */

#define H_number_fraction 0.76
#define He_number_fraction 0.06

/* ----- Tables ------- */
double T[N_T], J0_nu[N_nu], J_nu[N_nu], nu[N_nu];
double k1a[N_T], k2a[N_T], k3a[N_T], k4a[N_T], k5a[N_T], k6a[N_T], k7a[N_T], k8a[N_T], k9a[N_T],
  k10a[N_T], k11a[N_T];
double k12a[N_T], k13a[N_T], k14a[N_T], k15a[N_T], k16a[N_T], k17a[N_T], k18a[N_T], k19a[N_T],
  k20a[N_T], k21a[N_T];
double ciHIa[N_T], ciHeIa[N_T], ciHeIIa[N_T], ciHeISa[N_T], reHIIa[N_T], brema[N_T];
double ceHIa[N_T], ceHeIa[N_T], ceHeIIa[N_T], reHeII1a[N_T], reHeII2a[N_T], reHeIIIa[N_T];

/* cross-sections */
#ifdef RADIATION
double sigma24[N_nu], sigma25[N_nu], sigma26[N_nu], sigma27[N_nu], sigma28[N_nu], sigma29[N_nu],
  sigma30[N_nu], sigma31[N_nu];
#endif
#endif

/* ----- for HD cooling ----- */
#if defined (UM_CHEMISTRY) && defined (UM_HD_COOLING)
double kHD1a[N_T], kHD2a[N_T], kHD3a[N_T], kHD4a[N_T], kHD5a[N_T], kHD6a[N_T];
#endif

/* ---- for HeHII ---- */
#if defined (UM_CHEMISTRY)
double kHeHII1a[N_T], kHeHII2a[N_T], kHeHII3a[N_T];
#endif


#ifdef SCFPOTENTIAL
long scf_seed;
MyDouble *Anltilde, *coeflm, *twoalpha, *c1, *c2, *c3;
MyDouble *cosmphi, *sinmphi;
MyDouble *ultrasp, *ultraspt, *ultrasp1;
MyDouble *dblfact, *plm, *dplm;
MyDouble *sinsum, *cossum;
MyDouble *sinsum_all, *cossum_all;
#endif







