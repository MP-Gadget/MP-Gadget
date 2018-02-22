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

/* variables for input/output , usually only used on process 0 */


FILE 			/*!< file handle for info.txt log-file. */
 *FdEnergy,			/*!< file handle for energy.txt log-file. */
 *FdCPU;			/*!< file handle for cpu.txt log-file. */

#ifdef SFR
FILE *FdSfr;			/*!< file handle for sfr.txt log-file. */
#endif

#ifdef BLACK_HOLES
FILE *FdBlackHoles;		/*!< file handle for blackholes.txt log-file. */
#endif

/*! This structure contains data which is the SAME for all tasks (mostly code parameters read from the
 * parameter file).  Holding this data in a structure is convenient for writing/reading the restart file, and
 * it allows the introduction of new global variables in a simple way. The only thing to do is to introduce
 * them into this structure.
 */
struct global_data_all_processes All;
