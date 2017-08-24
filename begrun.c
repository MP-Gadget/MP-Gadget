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
#include "sfr_eff.h"
#include "cosmology.h"
#include "cooling.h"
#include "petaio.h"
#include "mymalloc.h"
#include "endrun.h"
#include "utils-string.h"
#include "system.h"

/*! \file begrun.c
 *  \brief initial set-up of a simulation run
 *
 *  This file contains various functions to initialize a simulation run. In
 *  particular, the parameterfile is read in and parsed, the initial
 *  conditions or restart files are read, and global variables are initialized
 *  to their proper values.
 */


static void
open_outputfiles(int RestartsnapNum);

static void set_units();


/*! This function performs the initial set-up of the simulation. First, the
 *  parameterfile is set, then routines for setting units, reading
 *  ICs/restart-files are called, auxialiary memory is allocated, etc.
 */
void begrun(int BeginFlag, int RestartSnapNum)
{

    int Nhost = cluster_get_num_hosts();
    size_t n = 1.0 * All.MaxMemSizePerNode * (1.0 * Nhost / NTask) * 1024 * 1024;
    mymalloc_init(n);
    walltime_init(&All.CT);
    petaio_init();

    petaio_read_header(RestartSnapNum);

    set_units();

#ifdef DEBUG
    write_pid_file();
    enable_core_dumps_and_fpu_exceptions();
#endif

    InitCool();

#if defined(SFR)
    init_clouds();
#endif

    random_generator = gsl_rng_alloc(gsl_rng_ranlxd1);

    gsl_rng_set(random_generator, 42);	/* start-up seed */

    if(BeginFlag == 2)
        long_range_init();

    All.TimeLastRestartFile = 0;

    set_random_numbers();

    init(RestartSnapNum);			/* ... read in initial model */

    if(BeginFlag >= 3) {
        return;
    }

#ifdef LIGHTCONE
    lightcone_init(All.Time);
#endif

    open_outputfiles(RestartSnapNum);

#ifdef TWODIMS
    int i;

    for(i = 0; i < NumPart; i++)
    {
        P[i].Pos[2] = 0;
        P[i].Vel[2] = 0;

        P[i].GravAccel[2] = 0;

        if(P[i].Type == 0)
        {
            SPHP(i).a.HydroAccel[2] = 0;
        }
    }
#endif
    All.TimeLastRestartFile = 0;
}

/*!  This function opens various log-files that report on the status and
 *   performance of the simulstion. On restart from restart-files
 *   (start-option 1), the code will append to these files.
 */
static void
open_outputfiles(int RestartSnapNum)
{
    const char mode[3]="a+";
    char * buf;
    char * postfix;

    if(RestartSnapNum != -1) {
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

    if(All.OutputEnergyDebug) {
        buf = fastpm_strdup_printf("%s/%s%s", All.OutputDir, All.EnergyFile, postfix);
        fastpm_path_ensure_dirname(buf);
        if(!(FdEnergy = fopen(buf, mode)))
            endrun(1, "error in opening file '%s'\n", buf);
        free(buf);
    }

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

}




/*!  This function closes the global log-files.
*/
void close_outputfiles(void)
{

    if(ThisTask != 0)		/* only the root processors writes to the log files */
        return;

    fclose(FdCPU);
    if(All.OutputEnergyDebug)
        fclose(FdEnergy);

#ifdef SFR
    fclose(FdSfr);
#endif

#ifdef BLACK_HOLES
    fclose(FdBlackHoles);
#endif
}


/*! Computes conversion factors between internal code units and the
 *  cgs-system.
 */
static void
set_units(void)
{
    double meanweight;

    All.UnitTime_in_s = All.UnitLength_in_cm / All.UnitVelocity_in_cm_per_s;
    All.UnitTime_in_Megayears = All.UnitTime_in_s / SEC_PER_MEGAYEAR;

    All.G = GRAVITY / pow(All.UnitLength_in_cm, 3) * All.UnitMass_in_g * pow(All.UnitTime_in_s, 2);

    All.UnitDensity_in_cgs = All.UnitMass_in_g / pow(All.UnitLength_in_cm, 3);
    All.UnitPressure_in_cgs = All.UnitMass_in_g / All.UnitLength_in_cm / pow(All.UnitTime_in_s, 2);
    All.UnitCoolingRate_in_cgs = All.UnitPressure_in_cgs / All.UnitTime_in_s;
    All.UnitEnergy_in_cgs = All.UnitMass_in_g * pow(All.UnitLength_in_cm, 2) / pow(All.UnitTime_in_s, 2);

    /* convert some physical input parameters to internal units */

    All.CP.Hubble = HUBBLE * All.UnitTime_in_s;
    /*Include massless neutrinos only if we do not have massive neutrino particles*/
    All.CP.MasslessNeutrinosOn = (NTotal[2] == 0);
    init_cosmology(&All.CP);

    meanweight = 4.0 / (1 + 3 * HYDROGEN_MASSFRAC);	/* note: assuming NEUTRAL GAS */

    All.MinEgySpec = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.MinGasTemp;
    All.MinEgySpec *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;

#ifdef SFR

    All.OverDensThresh =
        All.CritOverDensity * All.CP.OmegaBaryon * 3 * All.CP.Hubble * All.CP.Hubble / (8 * M_PI * All.G);

    All.PhysDensThresh = All.CritPhysDensity * PROTONMASS / HYDROGEN_MASSFRAC / All.UnitDensity_in_cgs;

    All.EgySpecCold = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.TempClouds;
    All.EgySpecCold *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;

    meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));	/* note: assuming FULL ionization */

    All.EgySpecSN = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.TempSupernova;
    All.EgySpecSN *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;

    if(HAS(All.WindModel, WINDS_FIXED_EFFICIENCY)) {
        All.WindSpeed = sqrt(2 * All.WindEnergyFraction * All.FactorSN * All.EgySpecSN / (1 - All.FactorSN) / All.WindEfficiency);
        message(1, "Windspeed: %g\n", All.WindSpeed);
    } else {
        All.WindSpeed = sqrt(2 * All.WindEnergyFraction * All.FactorSN * All.EgySpecSN / (1 - All.FactorSN) / 1.0);
        if(All.WindModel != WINDS_NONE)
            message(1, "Reference Windspeed: %g\n", All.WindSigma0 * All.WindSpeedFactor);
    }

#endif

    message(0, "Hubble (internal units) = %g\n", All.CP.Hubble);
    message(0, "G (internal units) = %g\n", All.G);
    message(0, "UnitLengh_in_cm = %g \n", All.UnitLength_in_cm);
    message(0, "UnitMass_in_g = %g \n", All.UnitMass_in_g);
    message(0, "UnitTime_in_s = %g \n", All.UnitTime_in_s);
    message(0, "UnitVelocity_in_cm_per_s = %g \n", All.UnitVelocity_in_cm_per_s);
    message(0, "UnitDensity_in_cgs = %g \n", All.UnitDensity_in_cgs);
    message(0, "UnitEnergy_in_cgs = %g \n", All.UnitEnergy_in_cgs);
    message(0, "Photon density OmegaG = %g\n",All.CP.OmegaG);
    message(0, "Massless Neutrino density OmegaNu0 = %g\n",All.CP.OmegaNu0);
    message(0, "Curvature density OmegaK = %g\n",All.CP.OmegaK);
    if(All.CP.RadiationOn) {
        /* note that this value is inaccurate if there is massive neutrino. */
        message(0, "Radiation is enabled in Hubble(a). "
               "Following CAMB convention: Omega_Tot - 1 = %g\n",
            All.CP.OmegaG + All.CP.OmegaNu0 + All.CP.OmegaK + All.CP.Omega0 + All.CP.OmegaLambda - 1);
    }
    message(0, "\n");
}
