#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "allvars.h"
#include "endrun.h"
#include "paramset.h"
#include "densitykernel.h"

#ifdef BLACK_HOLES
static int
BlackHoleFeedbackMethodAction (ParameterSet * ps, char * name, void * data)
{
    int v = param_get_enum(ps, name);
    if(HAS(v, BH_FEEDBACK_TOPHAT) == HAS(v, BH_FEEDBACK_SPLINE)) {
        message(1, "error BlackHoleFeedbackMethod contains either tophat or spline, but both\n");
        return 1;
    }
    if(HAS(v, BH_FEEDBACK_MASS) ==  HAS(v, BH_FEEDBACK_VOLUME)) {
        message(1, "error BlackHoleFeedbackMethod contains either volume or mass, but both\n");
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
        message(1, "error: At least use SFR_CRITERION_DENSITY\n");
        return 1;
    }
#if ! defined SPH_GRAD_RHO || ! defined METALS
    if(HAS(v, SFR_CRITERION_MOLECULAR_H2)) {
        message(1, "error: enable SPH_GRAD_RHO to use h2 criterion in sfr \n");
        return 1;
    }
    if(HAS(v, SFR_CRITERION_SELFGRAVITY)) {
        message(1, "error: enable SPH_GRAD_RHO to use selfgravity in sfr \n");
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
/*     message(1, "Found %d times in output list.\n", count); */

    /*Allocate enough memory*/
    All.OutputListLength = count;
    if(All.OutputListLength > sizeof(All.OutputListTimes) / sizeof(All.OutputListTimes[0])) {
        message(1, "Too many entries (%d) in the OutputList, need to recompile the code. (change All.OutputListTimes in allvars.h \n", 
            All.OutputListLength);
        return 1;
    }
    /*Now read in the values*/
    for(count=0,token=strtok(outputlist,","); count < All.OutputListLength && token; count++, token=strtok(NULL,","))
    {
        All.OutputListTimes[count] = atof(token);
/*         message(1, "Output at: %g\n", All.OutputListTimes[count]); */
    }
    free(strtmp);

    qsort(All.OutputListTimes, All.OutputListLength, sizeof(double), cmp_double);
    return 0;
}

static void set_units();

static ParameterSet *
create_gadget_parameter_set()
{
    ParameterSet * ps = parameter_set_new();

    param_declare_string(ps, "InitCondFile", 1, NULL, "Path to the Initial Condition File");
    param_declare_string(ps, "OutputDir",    1, NULL, "Prefix to the output files");
    param_declare_string(ps, "TreeCoolFile", 0, "", "Path to the Cooling Table");
    param_declare_string(ps, "MetalCoolFile", 0, "", "Path to the Metal Cooling Table. Refer to cooling.c");
    param_declare_string(ps, "UVFluctuationFile", 0, "", "Path to the UVFluctation Table. Refer to cooling.c.");

    static ParameterEnum DensityKernelTypeEnum [] = {
        {"cubic", DENSITY_KERNEL_CUBIC_SPLINE},
        {"quintic", DENSITY_KERNEL_QUINTIC_SPLINE},
        {"quartic", DENSITY_KERNEL_QUARTIC_SPLINE},
        {NULL, DENSITY_KERNEL_QUARTIC_SPLINE},
    } ;
    param_declare_enum(ps,    "DensityKernelType", DensityKernelTypeEnum, 1, 0, "");
    param_declare_string(ps, "SnapshotFileBase", 1, NULL, "");
    param_declare_string(ps, "EnergyFile", 0, "energy.txt", "");
    param_declare_string(ps, "CpuFile", 0, "cpu.txt", "");
    param_declare_string(ps, "InfoFile", 0, "info.txt", "");
    param_declare_string(ps, "TimingsFile", 0, "timings.txt", "");
    param_declare_string(ps, "RestartFile", 0, "restart", "");
    param_declare_string(ps, "OutputList", 1, NULL, "List of output times");

    param_declare_double(ps, "Omega0", 1, 0.2814, "");
    param_declare_double(ps, "CMBTemperature", 0, 2.7255,
            "Present-day CMB temperature in Kelvin, default from Fixsen 2009; affects background if RadiationOn is set.");
    param_declare_double(ps, "OmegaBaryon", 1, 0.0464, "");
    param_declare_double(ps, "OmegaLambda", 1, 0.7186, "");
    param_declare_double(ps, "HubbleParam", 1, 0.697, "");
    param_declare_double(ps, "BoxSize", 1, 32000, "");

    param_declare_int(ps,    "MaxMemSizePerCore", 0, 1200, "");
    param_declare_double(ps, "CpuTimeBetRestartFile", 1, 0, "");

    param_declare_double(ps, "TimeBegin", 1, 0, "");
    param_declare_double(ps, "TimeMax", 0, 1.0, "");
    param_declare_double(ps, "TimeLimitCPU", 1, 0, "");

    param_declare_int   (ps, "DomainOverDecompositionFactor", 0, 1, "Number of sub domains on a MPI rank");
    param_declare_double(ps, "TreeDomainUpdateFrequency", 0, 0.025, "");
    param_declare_double(ps, "ErrTolTheta", 0, 0.5, "");
    param_declare_int(ps,    "TypeOfOpeningCriterion", 0, 1, "");
    param_declare_double(ps, "ErrTolIntAccuracy", 0, 0.02, "");
    param_declare_double(ps, "ErrTolForceAcc", 0, 0.005, "");
    param_declare_int(ps,    "Nmesh", 1, 0, "");

    param_declare_double(ps, "MinGasHsmlFractional", 0, 0, "");
    param_declare_double(ps, "MaxGasVel", 0, 3e5, "");

    param_declare_int(ps,    "TypeOfTimestepCriterion", 0, 0, "Magic numbers!");
    param_declare_double(ps, "MaxSizeTimestep", 0, 0.1, "");
    param_declare_double(ps, "MinSizeTimestep", 0, 0, "");

    param_declare_double(ps, "MaxRMSDisplacementFac", 0, 0.2, "");
    param_declare_double(ps, "ArtBulkViscConst", 0, 0.75, "");
    param_declare_double(ps, "CourantFac", 0, 0.15, "");
    param_declare_double(ps, "DensityResolutionEta", 0, 1.0, "Resolution eta factor (See Price 2008) 1 = 33 for Cubic Spline");

    param_declare_double(ps, "DensityContrastLimit", 0, 100, "Max contrast for hydro force calculation");
    param_declare_double(ps, "MaxNumNgbDeviation", 0, 2, "");

    param_declare_int(ps, "NumPartPerFile", 0, 1024 * 1024 * 128, "number of particles per file");
    param_declare_int(ps, "NumWriters", 0, 0, "Number of concurrent writer processes. 0 implies Number of Tasks ");

    param_declare_int(ps, "CoolingOn", 1, 0, "");
    param_declare_int(ps, "StarformationOn", 1, 0, "");
    param_declare_int(ps, "RadiationOn", 0, 0, "Include radiation density in the background evolution.");
    param_declare_int(ps, "FastParticleType", 0, 2, "Particles of this type will not decrease the timestep. Default neutrinos.");
    param_declare_int(ps, "NoTreeType", 0, 2, "Particles of this type will not produce tree forces. Default neutrinos.");
    param_declare_int(ps, "MeasureFFT", 0, 0, "If enabled, will plan the FFT with FFT_MEASURE. Useful if more than 1000 FFTs will be done in a single submission.");

    param_declare_double(ps, "SofteningHalo", 1, 0, "");
    param_declare_double(ps, "SofteningDisk", 1, 0, "");
    param_declare_double(ps, "SofteningBulge", 1, 0, "");
    param_declare_double(ps, "SofteningGas", 1, 0, "");
    param_declare_double(ps, "SofteningStars", 1, 0, "");
    param_declare_double(ps, "SofteningBndry", 1, 0, "");
    param_declare_double(ps, "SofteningHaloMaxPhys", 1, 0, "");
    param_declare_double(ps, "SofteningDiskMaxPhys", 1, 0, "");
    param_declare_double(ps, "SofteningBulgeMaxPhys", 1, 0, "");
    param_declare_double(ps, "SofteningGasMaxPhys", 1, 0, "");
    param_declare_double(ps, "SofteningStarsMaxPhys", 1, 0, "");
    param_declare_double(ps, "SofteningBndryMaxPhys", 1, 0, "");

    param_declare_double(ps, "BufferSize", 0, 100, "");
    param_declare_double(ps, "PartAllocFactor", 1, 0, "");
    param_declare_double(ps, "TopNodeAllocFactor", 0, 0.5, "");

    param_declare_double(ps, "InitGasTemp", 1, 0, "");
    param_declare_double(ps, "MinGasTemp", 1, 0, "");

#if defined(ADAPTIVE_GRAVSOFT_FORGAS) && !defined(ADAPTIVE_GRAVSOFT_FORGAS_HSML)
    param_declare_double(ps, "ReferenceGasMass", 1, 0, "");
#endif

#ifdef FOF
    param_declare_double(ps, "FOFHaloLinkingLength", 1, 0, "");
    param_declare_int(ps, "FOFHaloMinLength", 0, 32, "");
    param_declare_double(ps, "MinFoFMassForNewSeed", 0, 5e2, "Minimal Mass for seeding tracer particles ");
    param_declare_double(ps, "TimeBetweenSeedingSearch", 0, 1e5, "Time Between Seeding Attempts: default to a a large value, meaning never.");
#endif

#ifdef BLACK_HOLES
    param_declare_double(ps, "BlackHoleAccretionFactor", 0, 100, "");
    param_declare_double(ps, "BlackHoleEddingtonFactor", 0, 3, "");
    param_declare_double(ps, "SeedBlackHoleMass", 1, 0, "");

    param_declare_double(ps, "BlackHoleNgbFactor", 0, 2, "");

    param_declare_double(ps, "BlackHoleMaxAccretionRadius", 0, 99999., "");
    param_declare_double(ps, "BlackHoleFeedbackFactor", 0, 0.05, "");
    param_declare_double(ps, "BlackHoleFeedbackRadius", 1, 0, "");

    param_declare_double(ps, "BlackHoleFeedbackRadiusMaxPhys", 1, 0, "");

    static ParameterEnum BlackHoleFeedbackMethodEnum [] = {
        {"mass", BH_FEEDBACK_MASS},
        {"volume", BH_FEEDBACK_VOLUME},
        {"tophat", BH_FEEDBACK_TOPHAT},
        {"spline", BH_FEEDBACK_SPLINE},
        {NULL, BH_FEEDBACK_SPLINE | BH_FEEDBACK_MASS},
    };
    param_declare_enum(ps, "BlackHoleFeedbackMethod", BlackHoleFeedbackMethodEnum, 1, 0, "");
#endif

#ifdef SFR
    static ParameterEnum StarformationCriterionEnum [] = {
        {"density", SFR_CRITERION_DENSITY},
        {"h2", SFR_CRITERION_MOLECULAR_H2},
        {"selfgravity", SFR_CRITERION_SELFGRAVITY},
        {"convergent", SFR_CRITERION_CONVERGENT_FLOW},
        {"continous", SFR_CRITERION_CONTINUOUS_CUTOFF},
        {NULL, SFR_CRITERION_DENSITY},
    };

    static ParameterEnum WindModelEnum [] = {
        {"subgrid", WINDS_SUBGRID},
        {"decouple", WINDS_DECOUPLE_SPH},
        {"halo", WINDS_USE_HALO},
        {"fixedefficiency", WINDS_FIXED_EFFICIENCY},
        {"sh03", WINDS_SUBGRID | WINDS_DECOUPLE_SPH | WINDS_FIXED_EFFICIENCY} ,
        {"vs08", WINDS_FIXED_EFFICIENCY},
        {"ofjt10", WINDS_USE_HALO | WINDS_DECOUPLE_SPH},
        {"isotropic", WINDS_ISOTROPIC },
        {"nowind", WINDS_NONE},
        {NULL, WINDS_SUBGRID | WINDS_DECOUPLE_SPH | WINDS_FIXED_EFFICIENCY},
    };

    param_declare_enum(ps, "StarformationCriterion", StarformationCriterionEnum, 1, 0, "");

    param_declare_double(ps, "CritOverDensity", 0, 57.7, "");
    param_declare_double(ps, "CritPhysDensity", 0, 0, "");

    param_declare_double(ps, "FactorSN", 0, 0.1, "");
    param_declare_double(ps, "FactorEVP", 0, 1000, "");
    param_declare_double(ps, "TempSupernova", 0, 1e8, "");
    param_declare_double(ps, "TempClouds", 0, 1000, "");
    param_declare_double(ps, "MaxSfrTimescale", 0, 1.5, "");
    param_declare_enum(ps, "WindModel", WindModelEnum, 1, 0, "");

    /* The following two are for VS08 and SH03*/
    param_declare_double(ps, "WindEfficiency", 0, 2.0, "");
    param_declare_double(ps, "WindEnergyFraction", 0, 1.0, "");

    /* The following two are for OFJT10*/
    param_declare_double(ps, "WindSigma0", 0, 353, "");
    param_declare_double(ps, "WindSpeedFactor", 0, 3.7, "");

    param_declare_double(ps, "WindFreeTravelLength", 0, 20, "");
    param_declare_double(ps, "WindFreeTravelDensFac", 0, 0., "");

    param_declare_double(ps, "QuickLymanAlphaProbability", 0, 0, "");

#endif

#ifdef SOFTEREQS
    param_declare_double(ps, "FactorForSofterEQS", 1, 0, "");
#endif

#ifdef BLACK_HOLES
    param_set_action(ps, "BlackHoleFeedbackMethod", BlackHoleFeedbackMethodAction, NULL);
#endif
#ifdef SFR
    param_set_action(ps, "StarformationCriterion", StarformationCriterionAction, NULL);
#endif
    param_set_action(ps, "OutputList", OutputListAction, NULL);

    return ps;
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

        ParameterSet * ps = create_gadget_parameter_set();

        if(0 != param_parse_file(ps, fname)) {
            endrun(1, "Parsing %s failed.", fname);
        }
        if(0 != param_validate(ps)) {
            endrun(1, "Validation of %s failed.", fname);
        }

        message(1, "----------- Running with Parameters ----------\n");
        param_dump(ps, stdout);
        message(1, "----------------------------------------------\n");

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
        All.CP.CMBTemperature = param_get_double(ps, "CMBTemperature");
        All.CP.RadiationOn = param_get_int(ps, "RadiationOn");
        All.CP.Omega0 = param_get_double(ps, "Omega0");
        All.CP.OmegaBaryon = param_get_double(ps, "OmegaBaryon");
        All.CP.OmegaLambda = param_get_double(ps, "OmegaLambda");
        All.CP.HubbleParam = param_get_double(ps, "HubbleParam");

        All.BoxSize = param_get_double(ps, "BoxSize");

        All.DomainOverDecompositionFactor = param_get_int(ps, "DomainOverDecompositionFactor");
        All.MaxMemSizePerCore = param_get_int(ps, "MaxMemSizePerCore");
        All.CpuTimeBetRestartFile = param_get_double(ps, "CpuTimeBetRestartFile");

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

        All.NumPartPerFile = param_get_int(ps, "NumPartPerFile");
        All.NumWriters = param_get_int(ps, "NumWriters");

        All.CoolingOn = param_get_int(ps, "CoolingOn");
        All.FastParticleType = param_get_int(ps, "FastParticleType");
        All.NoTreeType = param_get_int(ps, "NoTreeType");
        All.MeasureFFT = param_get_int(ps, "MeasureFFT");
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

        parameter_set_free(ps);

        if(All.TypeOfTimestepCriterion >= 3)
        {
            endrun(1, "The specified timestep criterion is not valid\n");
        }

    #ifdef SFR

        if(All.StarformationOn == 0)
        {
            message(1, "StarformationOn is disabled!\n");
        } else {
            if(All.CoolingOn == 0)
            {
                endrun(1, "You try to use the code with star formation enabled,\n"
                          "but you did not switch on cooling.\nThis mode is not supported.\n");
            }
        }
    #else
        if(All.StarformationOn == 1)
        {
            endrun(1, "Code was compiled with star formation switched off.\n"
                      "You must set `StarformationOn=0', or recompile the code.\n");
            All.StarformationOn = 0;
        }
    #endif

    #ifdef METALS
    #ifndef SFR
        endrun(1, "Code was compiled with METALS, but not with SFR.\n"
                  "This is not allowed.\n");
    #endif
    #endif

        DensityKernel kernel;
        density_kernel_init(&kernel, 1.0);
        All.DesNumNgb = density_kernel_desnumngb(&kernel, All.DensityResolutionEta);

        message(1, "The Density Kernel type is %s\n", kernel.name);
        message(1, "The Density resolution is %g * mean separation, or %d neighbours\n",
                    All.DensityResolutionEta, All.DesNumNgb);

        set_units();

        message(1, "Hubble (internal units) = %g\n", All.Hubble);
        message(1, "G (internal units) = %g\n", All.G);
        message(1, "UnitMass_in_g = %g \n", All.UnitMass_in_g);
        message(1, "UnitTime_in_s = %g \n", All.UnitTime_in_s);
        message(1, "UnitVelocity_in_cm_per_s = %g \n", All.UnitVelocity_in_cm_per_s);
        message(1, "UnitDensity_in_cgs = %g \n", All.UnitDensity_in_cgs);
        message(1, "UnitEnergy_in_cgs = %g \n", All.UnitEnergy_in_cgs);
        message(1, "Photon density OmegaG = %g\n",All.CP.OmegaG);
        message(1, "Massless Neutrino density OmegaNu0 = %g\n",All.CP.OmegaNu0);
        message(1, "Curvature density OmegaK = %g\n",All.CP.OmegaK);
        if(All.CP.RadiationOn) {
            /* note that this value is inaccurate if there is massive neutrino. */
            message(1, "Radiation is enabled in Hubble(a). "
                   "Following CAMB convention: Omega_Tot - 1 = %g\n",
                All.CP.OmegaG + All.CP.OmegaNu0 + All.CP.OmegaK + All.CP.Omega0 + All.CP.OmegaLambda - 1);
        }
        message(1, "\n");

    }

    MPI_Bcast(&All, sizeof(All), MPI_BYTE, 0, MPI_COMM_WORLD);
}

/*! Computes conversion factors between internal code units and the
 *  cgs-system.
 */
static void set_units(void)
{
    /*With slightly relativistic massive neutrinos, for consistency we need to include radiation.
     * A note on normalisation (as of 08/02/2012):
     * CAMB appears to set Omega_Lambda + Omega_Matter+Omega_K = 1,
     * calculating Omega_K in the code and specifying Omega_Lambda and Omega_Matter in the paramfile.
     * This means that Omega_tot = 1+ Omega_r + Omega_g, effectively
     * making h0 (very) slightly larger than specified, and the Universe is no longer flat!
     */

    All.CP.OmegaCDM = All.CP.Omega0 - All.CP.OmegaBaryon;
    All.CP.OmegaK = 1.0 - All.CP.Omega0 - All.CP.OmegaLambda;

    /* Omega_g = 4 \sigma_B T_{CMB}^4 8 \pi G / (3 c^3 H^2) */

    All.CP.OmegaG = 4 * STEFAN_BOLTZMANN
                  * pow(All.CP.CMBTemperature, 4)
                  * (8 * M_PI * GRAVITY)
                  / (3*C*C*C*HUBBLE*HUBBLE)
                  / (All.CP.HubbleParam*All.CP.HubbleParam);

    /* Neutrino + antineutrino background temperature as a ratio to T_CMB0
     * Note there is a slight correction from 4/11
     * due to the neutrinos being slightly coupled at e+- annihilation.
     * See Mangano et al 2005 (hep-ph/0506164)
     * The correction is (3.046/3)^(1/4), for N_eff = 3.046 */
    double TNu0_TCMB0 = pow(4/11., 1/3.) * 1.00328;

    /* For massless neutrinos,
     * rho_nu/rho_g = 7/8 (T_nu/T_cmb)^4 *N_eff,
     * but we absorbed N_eff into T_nu above. */
    All.CP.OmegaNu0 = All.CP.OmegaG * 7. / 8 * pow(TNu0_TCMB0, 4) * 3;

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

    meanweight = 4.0 / (1 + 3 * HYDROGEN_MASSFRAC);	/* note: assuming NEUTRAL GAS */

    All.MinEgySpec = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.MinGasTemp;
    All.MinEgySpec *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;

#ifdef SFR

    All.OverDensThresh =
        All.CritOverDensity * All.CP.OmegaBaryon * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.G);

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
}
