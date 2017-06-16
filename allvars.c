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


/*********************************************************/
/*  Global variables                                     */
/*********************************************************/



int ThisTask;			/*!< the number of the local processor  */
int NTask;			/*!< number of processors */

int MaxTopNodes;		/*!< Maximum number of nodes in the top-level tree used for domain decomposition */

int RestartFlag;		/*!< taken from command line used to start code. 0 is normal start-up from
				   initial conditions, 1 is resuming a run from a set of restart files, while 2
				   marks a restart from a snapshot file. */
int RestartSnapNum;

#ifdef BLACK_HOLES
double Local_BH_mass;
double Local_BH_dynamicalmass;
double Local_BH_Mdot;
double Local_BH_Medd;
#endif

/* Local number of particles; this shall be made into an array */
int NumPart;
int64_t NLocal[6];
int64_t NTotal[6];
int64_t TotNumPart;
int N_bh_slots;
int N_sph_slots;

gsl_rng *random_generator;	/*!< the random number generator used */


double RndTable[RNDTABLE];

/* variables for input/output , usually only used on process 0 */


FILE 			/*!< file handle for info.txt log-file. */
 *FdEnergy,			/*!< file handle for energy.txt log-file. */
 *FdCPU;			/*!< file handle for cpu.txt log-file. */

#ifdef SFR
FILE *FdSfr;			/*!< file handle for sfr.txt log-file. */
FILE *FdSfrDetails;
#endif

#ifdef BLACK_HOLES
FILE *FdBlackHoles;		/*!< file handle for blackholes.txt log-file. */
FILE *FdBlackHolesDetails;
#endif

/*! This structure contains data which is the SAME for all tasks (mostly code parameters read from the
 * parameter file).  Holding this data in a structure is convenient for writing/reading the restart file, and
 * it allows the introduction of new global variables in a simple way. The only thing to do is to introduce
 * them into this structure.
 */
struct global_data_all_processes All;

#ifdef _OPENMP
uint64_t BlockedParticleDrifts = 0;
uint64_t TotalParticleDrifts = 0;
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

