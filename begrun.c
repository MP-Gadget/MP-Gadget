#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <gsl/gsl_rng.h>


#include "allvars.h"
#include "param.h"
#include "densitykernel.h"
#include "proto.h"
#include "cosmology.h"
#include "cooling.h"
#include "petaio.h"
#include "mymalloc.h"
#include "endrun.h"
#include "utils-string.h"

/*! \file begrun.c
 *  \brief initial set-up of a simulation run
 *
 *  This file contains various functions to initialize a simulation run. In
 *  particular, the parameterfile is read in and parsed, the initial
 *  conditions or restart files are read, and global variables are initialized
 *  to their proper values.
 */



/*! This function performs the initial set-up of the simulation. First, the
 *  parameterfile is set, then routines for setting units, reading
 *  ICs/restart-files are called, auxialiary memory is allocated, etc.
 */
void begrun(void)
{

    /* n is aligned*/
    size_t n = All.MaxMemSizePerCore * All.NumThreads * ((size_t) 1024 * 1024);

    mymalloc_init(n);
    walltime_init(&All.CT);
    petaio_init();

#ifdef DEBUG
    write_pid_file();
    enable_core_dumps_and_fpu_exceptions();
#endif

    InitCool();

#if defined(SFR)
    init_clouds();
#endif

#ifdef LIGHTCONE
    lightcone_init();
#endif

    random_generator = gsl_rng_alloc(gsl_rng_ranlxd1);

    gsl_rng_set(random_generator, 42);	/* start-up seed */

    if(RestartFlag != 3 && RestartFlag != 4)
        long_range_init();

    All.TimeLastRestartFile = 0;


    set_random_numbers();

    init();			/* ... read in initial model */

    if(RestartFlag >= 3) {
        return;
    }
    open_outputfiles();

    reconstruct_timebins();

#ifdef TWODIMS
    int i;

    for(i = 0; i < NumPart; i++)
    {
        P[i].Pos[2] = 0;
        P[i].Vel[2] = 0;

        P[i].GravAccel[2] = 0;

        if(P[i].Type == 0)
        {
            SPHP(i).VelPred[2] = 0;
            SPHP(i).a.HydroAccel[2] = 0;
        }
    }
#endif


    init_drift_table();

    if(RestartFlag == 2)
        All.Ti_nextoutput = find_next_outputtime(All.Ti_Current + 100);
    else
        All.Ti_nextoutput = find_next_outputtime(All.Ti_Current);


    All.TimeLastRestartFile = 0;
}

/*!  This function opens various log-files that report on the status and
 *   performance of the simulstion. On restart from restart-files
 *   (start-option 1), the code will append to these files.
 */
void open_outputfiles(void)
{
    char mode[2];
    char * buf;
    char * postfix;

    if(RestartFlag == 0 || RestartFlag == 2)
        strcpy(mode, "w");
    else
        strcpy(mode, "a");

    if(RestartFlag == 2) {
        postfix = fastpm_strdup_printf("-R%03d", RestartSnapNum);
    } else {
        postfix = fastpm_strdup_printf("%s", "");
    }

    if(ThisTask != 0) {
        /* only the root processors writes to the log files */
        free(postfix);
        return;
    }

    buf = fastpm_strdup_printf("%s/%s%s", All.OutputDir, All.CpuFile, postfix);
    fastpm_path_ensure_dirname(buf);
    if(!(FdCPU = fopen(buf, mode)))
        endrun(1, "error in opening file '%s'\n", buf);
    free(buf);

    buf = fastpm_strdup_printf("%s/%s%s", All.OutputDir, All.EnergyFile, postfix);
    fastpm_path_ensure_dirname(buf);
    if(!(FdEnergy = fopen(buf, mode)))
        endrun(1, "error in opening file '%s'\n", buf);
    free(buf);

#ifdef SFR
    buf = fastpm_strdup_printf("%s/%s%s", All.OutputDir, "sfr.txt", postfix);
    fastpm_path_ensure_dirname(buf);
    if(!(FdSfr = fopen(buf, mode)))
        endrun(1, "error in opening file '%s'\n", buf);
    free(buf);
#endif

#ifdef BLACK_HOLES
    buf = fastpm_strdup_printf("%s/%s%s", All.OutputDir, "blackholes.txt", postfix);
    fastpm_path_ensure_dirname(buf);
    if(!(FdBlackHoles = fopen(buf, mode)))
        endrun(1, "error in opening file '%s'\n", buf);
    free(buf);
#endif

#ifdef GAL_PART
    buf = fastpm_strdup_printf("%s/%s%s", All.OutputDir, "galaxies.txt", postfix);
    fastpm_path_ensure_dirname(buf);
    if(!(FdGals = fopen(buf, mode)))
        endrun(1, "error in opening file '%s'\n", buf);
    free(buf);
#endif

}




/*!  This function closes the global log-files.
*/
void close_outputfiles(void)
{

    if(ThisTask != 0)		/* only the root processors writes to the log files */
        return;

    fclose(FdCPU);
    fclose(FdEnergy);

#ifdef SFR
    fclose(FdSfr);
#endif

#ifdef BLACK_HOLES
    fclose(FdBlackHoles);
#endif

#ifdef GAL_PART
    fclose(FdGals);
#endif

}


