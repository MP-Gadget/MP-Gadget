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

#include "config.h"

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
    if(ThisTask == 0)
    {
        /*    printf("\nThis is P-Gadget, version `%s', svn-revision `%s'.\n", GADGETVERSION, svn_version()); */
        printf("\nThis is P-Gadget, version %s.\n", GADGETVERSION);
        printf("\nRunning on %d MPIs .\n", NTask);
        printf("\nRunning on %d Threads.\n", omp_get_max_threads());
        printf("\nCode was compiled with settings:\n %s\n", COMPILETIMESETTINGS);
        printf("\nSize of particle structure       %td  [bytes]\n",sizeof(struct particle_data));
        printf("\nSize of blackhole structure       %td  [bytes]\n",sizeof(struct bh_particle_data));
        printf("\nSize of sph particle structure   %td  [bytes]\n",sizeof(struct sph_particle_data));

    }

#if defined(X86FIX) && defined(SOFTDOUBLEDOUBLE)
    x86_fix();			/* disable 80bit treatment of internal FPU registers in favour of proper IEEE 64bit double precision arithmetic */
#endif

    read_parameter_file(ParameterFile);	/* ... read in parameters for this run */

    mymalloc_init();
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

    /*With slightly relativistic massive neutrinos, for consistency we need to include radiation.
     * A note on normalisation (as of 08/02/2012):
     * CAMB appears to set Omega_Lambda + Omega_Matter+Omega_K = 1,
     * calculating Omega_K in the code and specifying Omega_Lambda and Omega_Matter in the paramfile.
     * This means that Omega_tot = 1+ Omega_r + Omega_g, effectively
     * making h0 (very) slightly larger than specified, and the Universe is no longer flat!
     */

    All.OmegaCDM = All.Omega0 - All.OmegaBaryon;
    All.OmegaK = 1.0 - All.Omega0 - All.OmegaLambda;

    /* Omega_g = 4 \sigma_B T_{CMB}^4 8 \pi G / (3 c^3 H^2) */

    All.OmegaG = 4 * STEFAN_BOLTZMANN
                  * pow(All.CMBTemperature, 4)
                  * (8 * M_PI * GRAVITY)
                  / (3*C*C*C*HUBBLE*HUBBLE)
                  / (All.HubbleParam*All.HubbleParam);

    /* Neutrino + antineutrino background temperature as a ratio to T_CMB0
     * Note there is a slight correction from 4/11
     * due to the neutrinos being slightly coupled at e+- annihilation.
     * See Mangano et al 2005 (hep-ph/0506164)
     * The correction is (3.046/3)^(1/4), for N_eff = 3.046 */
    double TNu0_TCMB0 = pow(4/11., 1/3.) * 1.00328;

    /* For massless neutrinos,
     * rho_nu/rho_g = 7/8 (T_nu/T_cmb)^4 *N_eff,
     * but we absorbed N_eff into T_nu above. */
    All.OmegaNu = All.OmegaG * 7. / 8 * pow(TNu0_TCMB0, 4) * 3;

    if(ThisTask == 0)
    {
        printf("\nHubble (internal units) = %g\n", All.Hubble);
        printf("G (internal units) = %g\n", All.G);
        printf("UnitMass_in_g = %g \n", All.UnitMass_in_g);
        printf("UnitTime_in_s = %g \n", All.UnitTime_in_s);
        printf("UnitVelocity_in_cm_per_s = %g \n", All.UnitVelocity_in_cm_per_s);
        printf("UnitDensity_in_cgs = %g \n", All.UnitDensity_in_cgs);
        printf("UnitEnergy_in_cgs = %g \n", All.UnitEnergy_in_cgs);
        printf("Photon density OmegaG = %g\n",All.OmegaG);
        printf("Neutrino density OmegaNu = %g\n",All.OmegaNu);
        printf("Curvature density OmegaK = %g\n",All.OmegaK);
        if(All.RadiationOn) {
            /* note that this value is inaccurate if there is massive neutrino. */
            printf("Radiation is enabled in Hubble(a). \n"
                   "Spacetime is approximately flat: Omega-1 = %g\n",
                All.OmegaG + All.OmegaNu + All.OmegaK + All.Omega0 + All.OmegaLambda - 1);
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

#ifdef BLACK_HOLES
static int
BlackHoleFeedbackMethodAction (ParameterSet * ps, char * name, void * data)
{
    int v = param_get_enum(ps, name);
    if(HAS(v, BH_FEEDBACK_TOPHAT) == HAS(v, BH_FEEDBACK_SPLINE)) {
        printf("error BlackHoleFeedbackMethod contains either tophat or spline, but both\n");
        return 1;
    }
    if(HAS(v, BH_FEEDBACK_MASS) ==  HAS(v, BH_FEEDBACK_VOLUME)) {
        printf("error BlackHoleFeedbackMethod contains either volume or mass, but both\n");
        return 1;
    }
    return 0;
}
#endif

#ifdef SFR
static int
StarformationCriterionAction(ParameterSet * ps, char * name, void * data)
{
    int v = param_get_enum(ps, name);
    if(!HAS(v, SFR_CRITERION_DENSITY)) {
        printf("error: At least use SFR_CRITERION_DENSITY\n");
        return 1;
    }
#if ! defined SPH_GRAD_RHO || ! defined METALS
    if(HAS(v, SFR_CRITERION_MOLECULAR_H2)) {
        printf("error: enable SPH_GRAD_RHO to use h2 criterion in sfr \n");
        return 1;
    }
    if(HAS(v, SFR_CRITERION_SELFGRAVITY)) {
        printf("error: enable SPH_GRAD_RHO to use selfgravity in sfr \n");
        return 1;
    }
#endif
    return 0;
}
#endif

int cmp_double(const void * a, const void * b)
{
    return ( *(double*)a - *(double*)b );
}

/*! This function parses a string containing a comma-separated list of variables,
 *  each of which is interpreted as a double.
 *  The purpose is to read an array of output times into the code.
 *  So specifying the output list now looks like:
 *  OutputList  0.1,0.3,0.5,1.0
 *
 *  We sort the input after reading it, so that the initial list need not be sorted.
 *  This function could be repurposed for reading generic arrays in future.
 */

static int
OutputListAction(ParameterSet * ps, char * name, void * data)
{
    char * outputlist = param_get_string(ps, name);
    char * strtmp=strdup(outputlist);
    char * token;
    int count;

    /*First parse the string to get the number of outputs*/
    for(count=0, token=strtok(strtmp,","); token; count++, token=strtok(NULL, ","))
    {}
/*     printf("Found %d times in output list.\n", count); */

    /*Allocate enough memory*/
    All.OutputListLength = count;
    if(All.OutputListLength > sizeof(All.OutputListTimes) / sizeof(All.OutputListTimes[0])) {
        printf("Too many entries (%d) in the OutputList, need to recompile the code. (change All.OutputListTimes in allvars.h \n", 
            All.OutputListLength);
        return 1;
    }
    /*Now read in the values*/
    for(count=0,token=strtok(outputlist,","); count < All.OutputListLength && token; count++, token=strtok(NULL,","))
    {
        All.OutputListTimes[count] = atof(token);
/*         printf("Output at: %g\n", All.OutputListTimes[count]); */
    }
    free(strtmp);

    qsort(All.OutputListTimes, All.OutputListLength, sizeof(double), cmp_double);
    return 0;
}

static char *
fread_all(char * filename)
{
    FILE * fp = fopen(filename, "r");
    if(!fp){
        endrun(1, "Could not open parameter file '%s' for reading\n",filename);
    }
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    char * r = malloc(size + 1);
    fseek(fp, 0, SEEK_SET);
    fread(r, 1, size, fp);
    r[size] = 0;
    fclose(fp);
    return r;
}

/*! This function parses the parameterfile in a simple way.  Each paramater is
 *  defined by a keyword (`tag'), and can be either of type douple, int, or
 *  character string.  The routine makes sure that each parameter appears
 *  exactly once in the parameterfile, otherwise error messages are
 *  produced that complain about the missing parameters.
 */
void read_parameter_file(char *fname)
{
    if(ThisTask == 0) {
        ParameterSet * ps = parameter_set_new();
#ifdef BLACK_HOLES
        param_set_action(ps, "BlackHoleFeedbackMethod", BlackHoleFeedbackMethodAction, NULL);
#endif
#ifdef SFR
        param_set_action(ps, "StarformationCriterion", StarformationCriterionAction, NULL);
#endif
        param_set_action(ps, "OutputList", OutputListAction, NULL);

        char * content = fread_all(fname);
        if(0 != param_parse(ps, content)) {
            endrun(9999, "Parsing failed.");
        }
        if(0 != param_validate(ps)) {
            endrun(9998, "Validation failed.");
        }
        free(content);
        printf("----------- Running with Parameters ----------\n");
        param_dump(ps, stdout);
        printf("----------------------------------------------\n");

        All.NumThreads = omp_get_max_threads();

    /* Start reading the values */
        param_get_string2(ps, "InitCondFile", All.InitCondFile);
        param_get_string2(ps, "OutputDir", All.OutputDir);
        param_get_string2(ps, "TreeCoolFile", All.TreeCoolFile);
        param_get_string2(ps, "MetalCoolFile", All.MetalCoolFile);
        param_get_string2(ps, "UVFluctuationfile", All.UVFluctuationFile);
        param_get_string2(ps, "SnapshotFileBase", All.SnapshotFileBase);
        param_get_string2(ps, "EnergyFile", All.EnergyFile);
        param_get_string2(ps, "CpuFile", All.CpuFile);
        param_get_string2(ps, "InfoFile", All.InfoFile);
        param_get_string2(ps, "TimingsFile", All.TimingsFile);
        param_get_string2(ps, "RestartFile", All.RestartFile);
        param_get_string2(ps, "OutputList", All.OutputList);

        All.DensityKernelType = param_get_enum(ps, "DensityKernelType");
        All.CMBTemperature = param_get_double(ps, "CMBTemperature");

        All.Omega0 = param_get_double(ps, "Omega0");
        All.OmegaBaryon = param_get_double(ps, "OmegaBaryon");
        All.OmegaLambda = param_get_double(ps, "OmegaLambda");
        All.HubbleParam = param_get_double(ps, "HubbleParam");
        All.BoxSize = param_get_double(ps, "BoxSize");

        All.DomainOverDecompositionFactor = param_get_int(ps, "DomainOverDecompositionFactor");
        All.MaxMemSizePerCore = param_get_int(ps, "MaxMemSizePerCore");
        All.CpuTimeBetRestartFile = param_get_double(ps, "CpuTimeBetRestartFile");
        All.TimeBetStatistics = param_get_double(ps, "TimeBetStatistics");
        All.TimeBegin = param_get_double(ps, "TimeBegin");
        All.TimeMax = param_get_double(ps, "TimeMax");
        All.TreeDomainUpdateFrequency = param_get_double(ps, "TreeDomainUpdateFrequency");
        All.ErrTolTheta = param_get_double(ps, "ErrTolTheta");
        All.ErrTolIntAccuracy = param_get_double(ps, "ErrTolIntAccuracy");
        All.ErrTolForceAcc = param_get_double(ps, "ErrTolForceAcc");
        All.Nmesh = param_get_int(ps, "Nmesh");

        All.MinGasHsmlFractional = param_get_double(ps, "MinGasHsmlFractional");
        All.MaxGasVel = param_get_double(ps, "MaxGasVel");
        All.MaxSizeTimestep = param_get_double(ps, "MaxSizeTimestep");

        All.MinSizeTimestep = param_get_double(ps, "MinSizeTimestep");
        All.MaxRMSDisplacementFac = param_get_double(ps, "MaxRMSDisplacementFac");
        All.ArtBulkViscConst = param_get_double(ps, "ArtBulkViscConst");
        All.CourantFac = param_get_double(ps, "CourantFac");
        All.DensityResolutionEta = param_get_double(ps, "DensityResolutionEta");

        All.DensityContrastLimit = param_get_double(ps, "DensityContrastLimit");
        All.MaxNumNgbDeviation = param_get_double(ps, "MaxNumNgbDeviation");

        All.NumFilesPerSnapshot = param_get_int(ps, "NumFilesPerSnapshot");
        All.NumWritersPerSnapshot = param_get_int(ps, "NumWritersPerSnapshot");
        All.NumFilesPerPIG = param_get_int(ps, "NumFilesPerPIG");
        All.NumWritersPerPIG = param_get_int(ps, "NumWritersPerPIG");

        All.CoolingOn = param_get_int(ps, "CoolingOn");
        All.RadiationOn = param_get_int(ps, "RadiationOn");
        All.FastParticleType = param_get_int(ps, "FastParticleType");
        All.NoTreeType = param_get_int(ps, "NoTreeType");
        All.StarformationOn = param_get_int(ps, "StarformationOn");
        All.TypeOfTimestepCriterion = param_get_int(ps, "TypeOfTimestepCriterion");
        All.TypeOfOpeningCriterion = param_get_int(ps, "TypeOfOpeningCriterion");
        All.TimeLimitCPU = param_get_double(ps, "TimeLimitCPU");
        All.SofteningHalo = param_get_double(ps, "SofteningHalo");
        All.SofteningDisk = param_get_double(ps, "SofteningDisk");
        All.SofteningBulge = param_get_double(ps, "SofteningBulge");
        All.SofteningGas = param_get_double(ps, "SofteningGas");
        All.SofteningStars = param_get_double(ps, "SofteningStars");
        All.SofteningBndry = param_get_double(ps, "SofteningBndry");
        All.SofteningHaloMaxPhys= param_get_double(ps, "SofteningHaloMaxPhys");
        All.SofteningDiskMaxPhys= param_get_double(ps, "SofteningDiskMaxPhys");
        All.SofteningBulgeMaxPhys= param_get_double(ps, "SofteningBulgeMaxPhys");
        All.SofteningGasMaxPhys= param_get_double(ps, "SofteningGasMaxPhys");
        All.SofteningStarsMaxPhys= param_get_double(ps, "SofteningStarsMaxPhys");
        All.SofteningBndryMaxPhys= param_get_double(ps, "SofteningBndryMaxPhys");

        All.BufferSize = param_get_double(ps, "BufferSize");
        All.PartAllocFactor = param_get_double(ps, "PartAllocFactor");
        All.TopNodeAllocFactor = param_get_double(ps, "TopNodeAllocFactor");

        All.InitGasTemp = param_get_double(ps, "InitGasTemp");
        All.MinGasTemp = param_get_double(ps, "MinGasTemp");

    #if defined(ADAPTIVE_GRAVSOFT_FORGAS) && !defined(ADAPTIVE_GRAVSOFT_FORGAS_HSML)
        All.ReferenceGasMass = param_get_double(ps, "ReferenceGasMass");
    #endif

    #ifdef FOF
        All.FOFHaloLinkingLength = param_get_double(ps, "FOFHaloLinkingLength");
        All.FOFHaloMinLength = param_get_int(ps, "FOFHaloMinLength");
        All.MinFoFMassForNewSeed = param_get_double(ps, "MinFoFMassForNewSeed");
    #endif

    #ifdef BLACK_HOLES
        All.BlackHoleSoundSpeedFromPressure = 0;

        All.TimeBetweenSeedingSearch = param_get_double(ps, "TimeBetweenSeedingSearch");
        All.BlackHoleAccretionFactor = param_get_double(ps, "BlackHoleAccretionFactor");
        All.BlackHoleEddingtonFactor = param_get_double(ps, "BlackHoleEddingtonFactor");
        All.SeedBlackHoleMass = param_get_double(ps, "SeedBlackHoleMass");

        All.BlackHoleNgbFactor = param_get_double(ps, "BlackHoleNgbFactor");

        All.BlackHoleMaxAccretionRadius = param_get_double(ps, "BlackHoleMaxAccretionRadius");
        All.BlackHoleFeedbackFactor = param_get_double(ps, "BlackHoleFeedbackFactor");
        All.BlackHoleFeedbackRadius = param_get_double(ps, "BlackHoleFeedbackRadius");

        All.BlackHoleFeedbackRadiusMaxPhys = param_get_double(ps, "BlackHoleFeedbackRadiusMaxPhys");

        All.BlackHoleFeedbackMethod = param_get_enum(ps, "BlackHoleFeedbackMethod");

    #endif

    #ifdef SFR
        All.StarformationCriterion = param_get_enum(ps, "StarformationCriterion");
        All.CritOverDensity = param_get_double(ps, "CritOverDensity");
        All.CritPhysDensity = param_get_double(ps, "CritPhysDensity");

        All.FactorSN = param_get_double(ps, "FactorSN");
        All.FactorEVP = param_get_double(ps, "FactorEVP");
        All.TempSupernova = param_get_double(ps, "TempSupernova");
        All.TempClouds = param_get_double(ps, "TempClouds");
        All.MaxSfrTimescale = param_get_double(ps, "MaxSfrTimescale");
        All.WindModel = param_get_enum(ps, "WindModel");

        /* The following two are for VS08 and SH03*/
        All.WindEfficiency = param_get_double(ps, "WindEfficiency");
        All.WindEnergyFraction = param_get_double(ps, "WindEnergyFraction");

        /* The following two are for OFJT10*/
        All.WindSigma0 = param_get_double(ps, "WindSigma0");
        All.WindSpeedFactor = param_get_double(ps, "WindSpeedFactor");

        All.WindFreeTravelLength = param_get_double(ps, "WindFreeTravelLength");
        All.WindFreeTravelDensFac = param_get_double(ps, "WindFreeTravelDensFac");

        All.QuickLymanAlphaProbability = param_get_double(ps, "QuickLymanAlphaProbability");

    #endif

    #ifdef SOFTEREQS
        All.FactorForSofterEQS = param_get_double(ps, "FactorForSofterEQS");
    #endif

        DensityKernel kernel;
        density_kernel_init(&kernel, 1.0);
        printf("The Density Kernel type is %s\n", kernel.name);
        All.DesNumNgb = density_kernel_desnumngb(&kernel, All.DensityResolutionEta);
        printf("The Density resolution is %g * mean separation, or %d neighbours\n",
                All.DensityResolutionEta, All.DesNumNgb);

        parameter_set_free(ps);
    }

    MPI_Bcast(&All, sizeof(All), MPI_BYTE, 0, MPI_COMM_WORLD);

    if(All.TypeOfTimestepCriterion >= 3)
    {
        if(ThisTask == 0)
        {
            endrun(0, "The specified timestep criterion is not valid\n");
        }
    }

#ifdef SFR

    if(All.StarformationOn == 0)
    {
        if(ThisTask == 0)
        {
            printf("StarformationOn is disabled!\n");
        }
    }
    if(All.CoolingOn == 0)
    {
        if(ThisTask == 0)
        {
            endrun(0, "You try to use the code with star formation enabled,\n"
                      "but you did not switch on cooling.\nThis mode is not supported.\n");
        }
    }
#else
    if(All.StarformationOn == 1)
    {
        if(ThisTask == 0)
        {
            endrun(0, "Code was compiled with star formation switched off.\n"
                      "You must set `StarformationOn=0', or recompile the code.\n");
        }
        All.StarformationOn = 0;
    }
#endif





    if(All.NumWritersPerSnapshot > NTask)
    {
       All.NumWritersPerSnapshot = NTask;
    }
    if(All.NumWritersPerPIG > NTask)
    {
       All.NumWritersPerPIG = NTask;
    }


#ifdef METALS
#ifndef SFR
    if(ThisTask == 0)
    {
        endrun(0, "Code was compiled with METALS, but not with SFR.\n"
                  "This is not allowed.\n");
    }
#endif
#endif

}
