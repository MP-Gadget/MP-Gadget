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

    set_units();


#ifdef COOLING
    set_global_time(All.TimeBegin);
    InitCool();
#endif

#if defined(SFR)
    init_clouds();
#endif

#ifdef LIGHTCONE
    lightcone_init();
#endif

    boxSize = All.BoxSize;
    boxHalf = 0.5 * All.BoxSize;
    inverse_boxSize = 1. / boxSize;

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




/*! Computes conversion factors between internal code units and the
 *  cgs-system.
 */
void set_units(void)
{
    double meanweight;

    All.UnitVelocity_in_cm_per_s = 1e5; /* 1 km/sec */
    All.UnitLength_in_cm = 3.085678e21; /* 1.0 Kpc /h */
    All.UnitMass_in_g = 1.989e43;       /* 1e10 Msun/h*/

    All.UnitTime_in_s = All.UnitLength_in_cm / All.UnitVelocity_in_cm_per_s;
    All.UnitTime_in_Megayears = All.UnitTime_in_s / SEC_PER_MEGAYEAR;

    All.G = GRAVITY / pow(All.UnitLength_in_cm, 3) * All.UnitMass_in_g * pow(All.UnitTime_in_s, 2);

    All.UnitDensity_in_cgs = All.UnitMass_in_g / pow(All.UnitLength_in_cm, 3);
    All.UnitPressure_in_cgs = All.UnitMass_in_g / All.UnitLength_in_cm / pow(All.UnitTime_in_s, 2);
    All.UnitCoolingRate_in_cgs = All.UnitPressure_in_cgs / All.UnitTime_in_s;
    All.UnitEnergy_in_cgs = All.UnitMass_in_g * pow(All.UnitLength_in_cm, 2) / pow(All.UnitTime_in_s, 2);

    /* convert some physical input parameters to internal units */

    All.Hubble = HUBBLE * All.UnitTime_in_s;

    if(ThisTask == 0)
    {
        printf("\nHubble (internal units) = %g\n", All.Hubble);
        printf("G (internal units) = %g\n", All.G);
        printf("UnitMass_in_g = %g \n", All.UnitMass_in_g);
        printf("UnitTime_in_s = %g \n", All.UnitTime_in_s);
        printf("UnitVelocity_in_cm_per_s = %g \n", All.UnitVelocity_in_cm_per_s);
        printf("UnitDensity_in_cgs = %g \n", All.UnitDensity_in_cgs);
        printf("UnitEnergy_in_cgs = %g \n", All.UnitEnergy_in_cgs);
        printf("Photon density OmegaG = %g\n",All.CP.OmegaG);
        printf("Massless Neutrino density OmegaNu0 = %g\n",All.CP.OmegaNu0);
        printf("Curvature density OmegaK = %g\n",All.CP.OmegaK);
        if(All.CP.RadiationOn) {
            /* note that this value is inaccurate if there is massive neutrino. */
            printf("Radiation is enabled in Hubble(a). \n"
                   "Following CAMB convention: Omega_Tot - 1 = %g\n",
                All.CP.OmegaG + All.CP.OmegaNu0 + All.CP.OmegaK + All.CP.Omega0 + All.CP.OmegaLambda - 1);
        }
        printf("\n");
    }

    meanweight = 4.0 / (1 + 3 * HYDROGEN_MASSFRAC);	/* note: assuming NEUTRAL GAS */

    All.MinEgySpec = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.MinGasTemp;
    All.MinEgySpec *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;


#if defined(SFR)
    set_units_sfr();
#endif
}



/*!  This function opens various log-files that report on the status and
 *   performance of the simulstion. On restart from restart-files
 *   (start-option 1), the code will append to these files.
 */
void open_outputfiles(void)
{
    char mode[2], buf[200];
    char dumpdir[200];
    char postfix[128];

    if(RestartFlag == 0 || RestartFlag == 2)
        strcpy(mode, "w");
    else
        strcpy(mode, "a");

    if(RestartFlag == 2) {
        sprintf(postfix, "-R%03d", RestartSnapNum);
    } else {
        sprintf(postfix, "%s", "");
    }

    /* create spliced dirs */
    int chunk = 10;
    if (NTask > 100) chunk = 100;
    if (NTask > 1000) chunk = 1000;

    sprintf(dumpdir, "%sdumpdir-%d%s/", All.OutputDir, (int)(ThisTask / chunk), postfix);
    mkdir(dumpdir, 02755);

    if(ThisTask != 0)		/* only the root processors writes to the log files */
        return;

    sprintf(buf, "%s%s%s", All.OutputDir, All.CpuFile, postfix);
    if(!(FdCPU = fopen(buf, mode)))
    {
        endrun(1, "error in opening file '%s'\n", buf);
    }

    sprintf(buf, "%s%s%s", All.OutputDir, All.InfoFile, postfix);
    if(!(FdInfo = fopen(buf, mode)))
    {
        endrun(1, "error in opening file '%s'\n", buf);
    }

    sprintf(buf, "%s%s%s", All.OutputDir, All.EnergyFile, postfix);
    if(!(FdEnergy = fopen(buf, mode)))
    {
        endrun(1, "error in opening file '%s'\n", buf);
    }

    sprintf(buf, "%s%s%s", All.OutputDir, All.TimingsFile, postfix);
    if(!(FdTimings = fopen(buf, mode)))
    {
        endrun(1, "error in opening file '%s'\n", buf);
    }

#ifdef SFR
    sprintf(buf, "%s%s%s", All.OutputDir, "sfr.txt", postfix);
    if(!(FdSfr = fopen(buf, mode)))
    {
        endrun(1, "error in opening file '%s'\n", buf);
    }
#endif

#ifdef BLACK_HOLES
    sprintf(buf, "%s%s%s", All.OutputDir, "blackholes.txt", postfix);
    if(!(FdBlackHoles = fopen(buf, mode)))
    {
        endrun(1, "error in opening file '%s'\n", buf);
    }
#endif



}




/*!  This function closes the global log-files.
*/
void close_outputfiles(void)
{

    if(ThisTask != 0)		/* only the root processors writes to the log files */
        return;

    fclose(FdCPU);
    fclose(FdInfo);
    fclose(FdEnergy);
    fclose(FdTimings);

#ifdef SFR
    fclose(FdSfr);
#endif

#ifdef BLACK_HOLES
    fclose(FdBlackHoles);
#endif
}


