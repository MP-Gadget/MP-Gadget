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

/* variables for input/output , usually only used on process 0 */


FILE 			/*!< file handle for info.txt log-file. */
 *FdEnergy,			/*!< file handle for energy.txt log-file. */
 *FdCPU;			/*!< file handle for cpu.txt log-file. */

FILE *FdSfr;			/*!< file handle for sfr.txt log-file. */

FILE *FdBlackHoles;		/*!< file handle for blackholes.txt log-file. */
