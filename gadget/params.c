#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <libgadget/allvars.h>
#include <libgadget/densitykernel.h>
#include <libgadget/timebinmgr.h>
#include <libgadget/utils.h>


/* Optional parameters are passed the flag 0 and required parameters 1.
 * These macros are just to document the semantic meaning of these flags. */
#define OPTIONAL 0
#define REQUIRED 1

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

#ifdef SFR
static int
StarformationCriterionAction(ParameterSet * ps, char * name, void * data)
{
    int v = param_get_enum(ps, name);
    if(!HAS(v, SFR_CRITERION_DENSITY)) {
        message(1, "error: At least use SFR_CRITERION_DENSITY\n");
        return 1;
    }
#if ! defined SPH_GRAD_RHO
    if(HAS(v, SFR_CRITERION_MOLECULAR_H2)) {
        message(1, "error: enable SPH_GRAD_RHO to use h2 criterion in sfr \n");
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

    /* Note TimeInit and TimeMax not yet initialised here*/

    /*First parse the string to get the number of outputs*/
    for(count=0, token=strtok(strtmp,","); token; count++, token=strtok(NULL, ","))
    {}
/*     message(1, "Found %d times in output list.\n", count); */

    /*Allocate enough memory*/
    All.OutputListLength = count;
    int maxcount = DMAX(sizeof(All.OutputListTimes) / sizeof(All.OutputListTimes[0]), MAXSNAPSHOTS);
    if(All.OutputListLength > maxcount) {
        message(1, "Too many entries (%d) in the OutputList, can take no more than %d.\n", All.OutputListLength, maxcount);
        return 1;
    }
    /*Now read in the values*/
    for(count=0,token=strtok(outputlist,","); count < All.OutputListLength && token; count++, token=strtok(NULL,","))
    {
        /* Skip a leading quote if one exists.
         * Extra characters are ignored by atof, so
         * no need to skip matching char.*/
        if(token[0] == '"')
            token+=1;

        double a = atof(token);

        if(a < 0.0) {
            endrun(1, "Requesting a negative output scaling factor a = %g\n", a);
        }
        All.OutputListTimes[count] = a;
/*         message(1, "Output at: %g\n", All.OutputListTimes[count]); */
    }
    free(strtmp);
    qsort(All.OutputListTimes, All.OutputListLength, sizeof(double), cmp_double);
    return 0;
}

static ParameterSet *
create_gadget_parameter_set()
{
    ParameterSet * ps = parameter_set_new();

    param_declare_string(ps, "InitCondFile", REQUIRED, NULL, "Path to the Initial Condition File");
    param_declare_string(ps, "OutputDir",    REQUIRED, NULL, "Prefix to the output files");
    param_declare_string(ps, "TreeCoolFile", OPTIONAL, "", "Path to the Cooling Table");
    param_declare_string(ps, "MetalCoolFile", OPTIONAL, "", "Path to the Metal Cooling Table. Empty string disables metal cooling. Refer to cooling.c");
    param_declare_string(ps, "UVFluctuationFile", OPTIONAL, "", "Path to the UVFluctation Table. Refer to cooling.c.");

    static ParameterEnum DensityKernelTypeEnum [] = {
        {"cubic", DENSITY_KERNEL_CUBIC_SPLINE},
        {"quintic", DENSITY_KERNEL_QUINTIC_SPLINE},
        {"quartic", DENSITY_KERNEL_QUARTIC_SPLINE},
        {NULL, DENSITY_KERNEL_QUARTIC_SPLINE},
    } ;
    param_declare_enum(ps,    "DensityKernelType", DensityKernelTypeEnum, OPTIONAL, "quintic", "");
    param_declare_string(ps, "SnapshotFileBase", OPTIONAL, "PART", "Base name of the snapshot files, _%03d will be appended to the name.");
    param_declare_string(ps, "FOFFileBase", OPTIONAL, "PIG", "Base name of the fof files, _%03d will be appended to the name.");
    param_declare_string(ps, "EnergyFile", OPTIONAL, "energy.txt", "");
    param_declare_int(ps,    "OutputEnergyDebug", OPTIONAL, 0,"Should we output energy statistics to energy.txt");
    param_declare_string(ps, "CpuFile", OPTIONAL, "cpu.txt", "");
    param_declare_string(ps, "OutputList", REQUIRED, NULL, "List of output times");

    param_declare_double(ps, "Omega0", REQUIRED, 0.2814, "");
    param_declare_double(ps, "CMBTemperature", OPTIONAL, 2.7255,
            "Present-day CMB temperature in Kelvin, default from Fixsen 2009; affects background if RadiationOn is set.");
    param_declare_double(ps, "OmegaBaryon", REQUIRED, 0.0464, "");
    param_declare_double(ps, "OmegaLambda", REQUIRED, 0.7186, "Dark energy density at z=0");
    param_declare_double(ps, "Omega_fld", OPTIONAL, 0, "Energy density of dark energy fluid.");
    param_declare_double(ps, "w0_fld", OPTIONAL, -1., "Dark energy equation of state.");
    param_declare_double(ps, "wa_fld", OPTIONAL, 0, "Dark energy evolution parameter.");
    param_declare_double(ps, "HubbleParam", REQUIRED, 0.697, "");

    param_declare_int(ps,    "OutputPotential", OPTIONAL, 1, "Save the potential in snapshots.");
    param_declare_int(ps,    "MaxMemSizePerNode", OPTIONAL, 0.6 * get_physmem_bytes() / (1024 * 1024), "Preallocate this much memory MB per computing node/ host. Default is 80\% of total physical mem per node. ");
    param_declare_double(ps, "AutoSnapshotTime", OPTIONAL, 0, "Seconds after which to automatically generate a snapshot if nothing is output.");

    param_declare_double(ps, "TimeMax", OPTIONAL, 1.0, "Scale factor to end run.");
    param_declare_double(ps, "TimeLimitCPU", REQUIRED, 0, "CPU time to run for in seconds.");

    param_declare_int   (ps, "DomainOverDecompositionFactor", OPTIONAL, 1, "Create on average this number of sub domains on a MPI rank. Load balancer will try to create this number of equal sized chunks on each rank. Higher numbers improve the load balancing but make domain more expensive.");
    param_declare_int   (ps, "DomainUseGlobalSorting", OPTIONAL, 1, "Determining the initial refinement of chunks globally. Enabling this produces better domains at costs of slowing down the domain decomposition.");
    param_declare_int   (ps, "TopNodeIncreaseFactor", OPTIONAL, 4, "Create on average this number of topNodes per MPI rank. Higher numbers improve the load balancing but make domain more expensive. Similar to DomainOverDecompositionFactor, but ignored by load balancer.");
    param_declare_double(ps, "ErrTolIntAccuracy", OPTIONAL, 0.02, "");
    param_declare_double(ps, "ErrTolForceAcc", OPTIONAL, 0.005, "Force accuracy required from tree. Controls tree opening criteria. Lower values are more accurate.");
    param_declare_double(ps, "Asmth", OPTIONAL, 1.25, "The scale of the short-range/long-range force split in units of FFT-mesh cells. Gadget-2 paper says larger values may be more accurate.");
    param_declare_int(ps,    "Nmesh", REQUIRED, 0, "");

    param_declare_double(ps, "MinGasHsmlFractional", OPTIONAL, 0, "Minimal gas Hsml as a fraction of gravity softening.");
    param_declare_double(ps, "MaxGasVel", OPTIONAL, 3e5, "");

    param_declare_int(ps,    "TypeOfTimestepCriterion", OPTIONAL, 0, "Compatibility only. Has no effect");
    param_declare_double(ps, "MaxSizeTimestep", OPTIONAL, 0.1, "");
    param_declare_double(ps, "MinSizeTimestep", OPTIONAL, 0, "");
    param_declare_int(ps, "ForceEqualTimesteps", OPTIONAL, 0, "Force all timesteps to be the same, the smallest required.");

    param_declare_double(ps, "MaxRMSDisplacementFac", OPTIONAL, 0.2, "");
    param_declare_double(ps, "ArtBulkViscConst", OPTIONAL, 0.75, "");
    param_declare_double(ps, "CourantFac", OPTIONAL, 0.15, "");
    param_declare_double(ps, "DensityResolutionEta", OPTIONAL, 1.0, "Resolution eta factor (See Price 2008) 1 = 33 for Cubic Spline");

    param_declare_double(ps, "DensityContrastLimit", OPTIONAL, 100, "Max contrast for hydro force calculation");
    param_declare_double(ps, "MaxNumNgbDeviation", OPTIONAL, 2, "");
    param_declare_double(ps, "HydroCostFactor", OPTIONAL, 1, "Cost factor of hydro calculation, default to 1.");

    param_declare_int(ps, "BytesPerFile", OPTIONAL, 1024 * 1024 * 1024, "number of bytes per file");
    param_declare_int(ps, "NumWriters", OPTIONAL, NTask, "Max number of concurrent writer processes. 0 implies Number of Tasks; ");
    param_declare_int(ps, "MinNumWriters", OPTIONAL, 1, "Min number of concurrent writer processes. We increase number of Files to avoid too few writers. ");
    param_declare_int(ps, "WritersPerFile", OPTIONAL, 8, "Number of Writer groups assigned to a file; total number of writers is capped by NumWriters.");

    param_declare_int(ps, "EnableAggregatedIO", OPTIONAL, 0, "Use the Aggregated IO policy for small data set (Experimental).");
    param_declare_int(ps, "AggregatedIOThreshold", OPTIONAL, 1024 * 1024 * 256, "Max number of bytes on a writer before reverting to throttled IO.");

    param_declare_int(ps, "MakeGlassFile", OPTIONAL, 0, "Enable to reverse the direction of gravity, only apply the PM force, and thus make a glass file.");
    param_declare_int(ps, "CoolingOn", REQUIRED, 0, "Enables cooling");
    param_declare_double(ps, "UVRedshiftThreshold", OPTIONAL, -1.0, "Earliest Redshift that UV background is enabled. This modulates UVFluctuation and TreeCool globally. Default -1.0 means no modulation.");

    param_declare_int(ps, "HydroOn", REQUIRED, 1, "Enables hydro force");
    param_declare_int(ps, "DensityOn", OPTIONAL, 1, "Enables SPH density computation.");
    param_declare_int(ps, "TreeGravOn", OPTIONAL, 1, "Enables tree gravity");
    param_declare_int(ps, "RadiationOn", OPTIONAL, 0, "Include radiation density in the background evolution.");
    param_declare_int(ps, "FastParticleType", OPTIONAL, 2, "Particles of this type will not decrease the timestep. Default neutrinos.");

    param_declare_double(ps, "GravitySoftening", OPTIONAL, 1./30., "Softening for collisionless particles; units of mean separation of DM. ForceSoftening is 2.8 times this.");
    param_declare_double(ps, "GravitySofteningGas", OPTIONAL, 1./30., "Softening for collisional particles (Gas); units of mean separation of DM; 0 to use Hsml of last step. ");

    param_declare_double(ps, "BufferSize", OPTIONAL, 100, "");
    param_declare_double(ps, "PartAllocFactor", REQUIRED, 0, "");
    param_declare_double(ps, "TopNodeAllocFactor", OPTIONAL, 0.5, "");

    param_declare_double(ps, "InitGasTemp", OPTIONAL, 300, "Initial gas temperature");
    param_declare_double(ps, "MinGasTemp", OPTIONAL, 5, "Minimum gas temperature");

    param_declare_int(ps, "SnapshotWithFOF", REQUIRED, 0, "Enable Friends-of-Friends halo finder.");
    param_declare_int(ps, "FOFSaveParticles", OPTIONAL, 1, "Save particles in the FOF catalog."); 
    param_declare_double(ps, "FOFHaloLinkingLength", OPTIONAL, 0.2, "Linking length for Friends of Friends halos.");
    param_declare_int(ps, "FOFHaloMinLength", OPTIONAL, 32, "");
    param_declare_double(ps, "MinFoFMassForNewSeed", OPTIONAL, 5e2, "Minimal Mass for seeding tracer particles ");
    param_declare_double(ps, "TimeBetweenSeedingSearch", OPTIONAL, 1e5, "Time Between Seeding Attempts: default to a a large value, meaning never.");

    /*Black holes*/
    param_declare_int(ps, "BlackHoleOn", REQUIRED, 1, "Enable Blackhole ");
    param_declare_double(ps, "BlackHoleAccretionFactor", OPTIONAL, 100, "");
    param_declare_double(ps, "BlackHoleEddingtonFactor", OPTIONAL, 3, "");
    param_declare_double(ps, "SeedBlackHoleMass", OPTIONAL, 5e-5, "");

    param_declare_double(ps, "BlackHoleNgbFactor", OPTIONAL, 2, "");

    param_declare_double(ps, "BlackHoleMaxAccretionRadius", OPTIONAL, 99999., "");
    param_declare_double(ps, "BlackHoleFeedbackFactor", OPTIONAL, 0.05, "");
    param_declare_double(ps, "BlackHoleFeedbackRadius", OPTIONAL, 0, "");

    param_declare_double(ps, "BlackHoleFeedbackRadiusMaxPhys", OPTIONAL, 0, "");

    static ParameterEnum BlackHoleFeedbackMethodEnum [] = {
        {"mass", BH_FEEDBACK_MASS},
        {"volume", BH_FEEDBACK_VOLUME},
        {"tophat", BH_FEEDBACK_TOPHAT},
        {"spline", BH_FEEDBACK_SPLINE},
        {NULL, BH_FEEDBACK_SPLINE | BH_FEEDBACK_MASS},
    };
    param_declare_enum(ps, "BlackHoleFeedbackMethod", BlackHoleFeedbackMethodEnum,
            OPTIONAL, "spline, mass", "");
    /*End black holes*/

    /*Star formation parameters*/
    static ParameterEnum StarformationCriterionEnum [] = {
        {"density", SFR_CRITERION_DENSITY},
        {"h2", SFR_CRITERION_MOLECULAR_H2},
        {"selfgravity", SFR_CRITERION_SELFGRAVITY},
        {"convergent", SFR_CRITERION_CONVERGENT_FLOW},
        {"continous", SFR_CRITERION_CONTINUOUS_CUTOFF},
        {NULL, SFR_CRITERION_DENSITY},
    };

    static ParameterEnum WindModelEnum [] = {
        {"subgrid", WIND_SUBGRID},
        {"decouple", WIND_DECOUPLE_SPH},
        {"halo", WIND_USE_HALO},
        {"fixedefficiency", WIND_FIXED_EFFICIENCY},
        {"sh03", WIND_SUBGRID | WIND_DECOUPLE_SPH | WIND_FIXED_EFFICIENCY} ,
        {"vs08", WIND_FIXED_EFFICIENCY},
        {"ofjt10", WIND_USE_HALO | WIND_DECOUPLE_SPH},
        {"isotropic", WIND_ISOTROPIC },
        {NULL, WIND_SUBGRID | WIND_DECOUPLE_SPH | WIND_FIXED_EFFICIENCY},
    };

    param_declare_int(ps, "StarformationOn", REQUIRED, 0, "Enables star formation");
    param_declare_int(ps, "WindOn", REQUIRED, 0, "Enables wind feedback");
    param_declare_enum(ps, "StarformationCriterion",
            StarformationCriterionEnum, OPTIONAL, "density", "");

    param_declare_double(ps, "CritOverDensity", OPTIONAL, 57.7, "");
    param_declare_double(ps, "CritPhysDensity", OPTIONAL, 0, "");

    param_declare_double(ps, "FactorSN", OPTIONAL, 0.1, "");
    param_declare_double(ps, "FactorEVP", OPTIONAL, 1000, "");
    param_declare_double(ps, "TempSupernova", OPTIONAL, 1e8, "");
    param_declare_double(ps, "TempClouds", OPTIONAL, 1000, "");
    param_declare_double(ps, "MaxSfrTimescale", OPTIONAL, 1.5, "");
    param_declare_enum(ps, "WindModel", WindModelEnum, OPTIONAL, "subgrid,decouple,fixedefficiency", "");

    /* The following two are for VS08 and SH03*/
    param_declare_double(ps, "WindEfficiency", OPTIONAL, 2.0, "");
    param_declare_double(ps, "WindEnergyFraction", OPTIONAL, 1.0, "");

    /* The following two are for OFJT10*/
    param_declare_double(ps, "WindSigma0", OPTIONAL, 353, "");
    param_declare_double(ps, "WindSpeedFactor", OPTIONAL, 3.7, "");

    param_declare_double(ps, "WindFreeTravelLength", OPTIONAL, 20, "");
    param_declare_double(ps, "WindFreeTravelDensFac", OPTIONAL, 0., "");

    param_declare_int(ps, "RandomSeed", OPTIONAL, 42, "Random number generator initial seed. Used to form stars.");

    /*These parameters are Lyman alpha forest specific*/
    param_declare_double(ps, "QuickLymanAlphaProbability", OPTIONAL, 0, "Probability gas is turned directly into stars, irrespective of pressure. One is equivalent to quick lyman alpha star formation.");
    /* Enable model for helium reionisation which adds extra photo-heating to under-dense gas.
     * Extra heating has the form: H = Amp * (rho / rho_c(z=0))^Exp
     * but is density-independent when rho / rho_c > Thresh. */
    param_declare_int(ps, "HeliumHeatOn", OPTIONAL, 0, "Change photo-heating rate to model helium reionisation on underdense gas.");
    param_declare_double(ps, "HeliumHeatThresh", OPTIONAL, 10, "Overdensity above which heating is density-independent.");
    param_declare_double(ps, "HeliumHeatAmp", OPTIONAL, 1, "Density-independent heat boost. Changes mean temperature.");
    param_declare_double(ps, "HeliumHeatExp", OPTIONAL, 0, "Density dependent heat boost (exponent). Changes gamma.");
    /*End of star formation parameters*/

    /*Parameters for the massive neutrino model*/
    param_declare_int(ps, "MassiveNuLinRespOn", REQUIRED, 0, "Enables linear response massive neutrinos of 1209.0461. Make sure you enable radiation too.");
    param_declare_int(ps, "HybridNeutrinosOn", OPTIONAL, 0, "Enables hybrid massive neutrinos, where some density is followed analytically, and some with particles. Requires MassivenuLinRespOn");
    param_declare_string(ps, "LinearTransferFunction", OPTIONAL, "camb_transfer_99.dat", "File containing linear transfer function in CAMB format. Used for massive neutrinos.");
    param_declare_double(ps, "InputSpectrum_UnitLength_in_cm", OPTIONAL, 3.085678e24, "Units of the CAMB transfer function in cm. By default Mpc.");
    param_declare_double(ps, "MNue", OPTIONAL, 0, "First neutrino mass in eV.");
    param_declare_double(ps, "MNum", OPTIONAL, 0, "Second neutrino mass in eV.");
    param_declare_double(ps, "MNut", OPTIONAL, 0, "Third neutrino mass in eV.");
    param_declare_double(ps, "Vcrit", OPTIONAL, 500., "For hybrid neutrinos: Critical velocity (in km/s) in the Fermi-Dirac distribution below which the neutrinos are particles in the ICs.");
    param_declare_double(ps, "NuPartTime", OPTIONAL, 0.3333333, "Scale factor at which to turn on hybrid neutrino particles.");
    /*End parameters for the massive neutrino model*/
  
    param_set_action(ps, "BlackHoleFeedbackMethod", BlackHoleFeedbackMethodAction, NULL);
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
        param_get_string2(ps, "FOFFileBase", All.FOFFileBase);
        param_get_string2(ps, "EnergyFile", All.EnergyFile);
        All.OutputEnergyDebug = param_get_int(ps, "EnergyFile");
        param_get_string2(ps, "CpuFile", All.CpuFile);

        All.DensityKernelType = param_get_enum(ps, "DensityKernelType");
        All.CP.CMBTemperature = param_get_double(ps, "CMBTemperature");
        All.CP.RadiationOn = param_get_int(ps, "RadiationOn");
        All.CP.Omega0 = param_get_double(ps, "Omega0");
        All.CP.OmegaBaryon = param_get_double(ps, "OmegaBaryon");
        All.CP.OmegaLambda = param_get_double(ps, "OmegaLambda");
        All.CP.Omega_fld = param_get_double(ps, "Omega_fld");
        if(All.CP.OmegaLambda > 0 && All.CP.Omega_fld > 0)
            endrun(0, "Cannot have OmegaLambda and Omega_fld (evolving dark energy) at the same time!\n");
        All.CP.w0_fld = param_get_double(ps,"w0_fld");
        All.CP.wa_fld = param_get_double(ps,"wa_fld");
        All.CP.HubbleParam = param_get_double(ps, "HubbleParam");

        All.DomainOverDecompositionFactor = param_get_int(ps, "DomainOverDecompositionFactor");
        All.DomainUseGlobalSorting = param_get_int(ps, "DomainUseGlobalSorting");
        All.TopNodeIncreaseFactor = param_get_int(ps, "TopNodeIncreaseFactor");
        All.OutputPotential = param_get_int(ps, "OutputPotential");
        All.MaxMemSizePerNode = param_get_int(ps, "MaxMemSizePerNode");

        All.TimeMax = param_get_double(ps, "TimeMax");
        All.ErrTolIntAccuracy = param_get_double(ps, "ErrTolIntAccuracy");
        All.ErrTolForceAcc = param_get_double(ps, "ErrTolForceAcc");
        All.Asmth = param_get_double(ps, "Asmth");
        All.Nmesh = param_get_int(ps, "Nmesh");

        All.MinGasHsmlFractional = param_get_double(ps, "MinGasHsmlFractional");
        All.MaxGasVel = param_get_double(ps, "MaxGasVel");
        All.MaxSizeTimestep = param_get_double(ps, "MaxSizeTimestep");

        All.MinSizeTimestep = param_get_double(ps, "MinSizeTimestep");
        All.ForceEqualTimesteps = param_get_int(ps, "ForceEqualTimesteps");
        All.MaxRMSDisplacementFac = param_get_double(ps, "MaxRMSDisplacementFac");
        All.ArtBulkViscConst = param_get_double(ps, "ArtBulkViscConst");
        All.CourantFac = param_get_double(ps, "CourantFac");
        All.DensityResolutionEta = param_get_double(ps, "DensityResolutionEta");
        All.HydroCostFactor = param_get_double(ps, "HydroCostFactor");
        All.DensityContrastLimit = param_get_double(ps, "DensityContrastLimit");
        All.MaxNumNgbDeviation = param_get_double(ps, "MaxNumNgbDeviation");

        All.IO.BytesPerFile = param_get_int(ps, "BytesPerFile");
        All.IO.UsePeculiarVelocity = 0; /* Will be set by the Initial Condition File */
        All.IO.NumWriters = param_get_int(ps, "NumWriters");
        All.IO.MinNumWriters = param_get_int(ps, "MinNumWriters");
        All.IO.WritersPerFile = param_get_int(ps, "WritersPerFile");
        All.IO.AggregatedIOThreshold = param_get_int(ps, "AggregatedIOThreshold");
        All.IO.EnableAggregatedIO = param_get_int(ps, "EnableAggregatedIO");

        All.MakeGlassFile = param_get_int(ps, "MakeGlassFile");
        All.CoolingOn = param_get_int(ps, "CoolingOn");
        All.UVRedshiftThreshold = param_get_double(ps, "UVRedshiftThreshold");
        All.HydroOn = param_get_int(ps, "HydroOn");
        All.DensityOn = param_get_int(ps, "DensityOn");
        All.TreeGravOn = param_get_int(ps, "TreeGravOn");
        All.FastParticleType = param_get_int(ps, "FastParticleType");
        All.TimeLimitCPU = param_get_double(ps, "TimeLimitCPU");
        All.AutoSnapshotTime = param_get_double(ps, "AutoSnapshotTime");
        All.GravitySoftening = param_get_double(ps, "GravitySoftening");
        All.GravitySofteningGas = param_get_double(ps, "GravitySofteningGas");

        All.BufferSize = param_get_double(ps, "BufferSize");
        All.PartAllocFactor = param_get_double(ps, "PartAllocFactor");
        All.TopNodeAllocFactor = param_get_double(ps, "TopNodeAllocFactor");

        All.InitGasTemp = param_get_double(ps, "InitGasTemp");
        All.MinGasTemp = param_get_double(ps, "MinGasTemp");

        All.SnapshotWithFOF = param_get_int(ps, "SnapshotWithFOF");
        All.FOFSaveParticles = param_get_int(ps, "FOFSaveParticles");
        All.FOFHaloLinkingLength = param_get_double(ps, "FOFHaloLinkingLength");
        All.FOFHaloMinLength = param_get_int(ps, "FOFHaloMinLength");
        All.MinFoFMassForNewSeed = param_get_double(ps, "MinFoFMassForNewSeed");
        All.TimeBetweenSeedingSearch = param_get_double(ps, "TimeBetweenSeedingSearch");

        All.RandomSeed = param_get_int(ps, "RandomSeed");

        All.BlackHoleOn = param_get_int(ps, "BlackHoleOn");
    #ifdef BLACK_HOLES
        All.BlackHoleSoundSpeedFromPressure = 0;

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

        All.StarformationOn = param_get_int(ps, "StarformationOn");
        All.WindOn = param_get_int(ps, "WindOn");
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
        All.HeliumHeatOn = param_get_int(ps, "HeliumHeatOn");
        All.HeliumHeatThresh = param_get_double(ps, "HeliumHeatThresh");
        All.HeliumHeatAmp = param_get_double(ps, "HeliumHeatAmp");
        All.HeliumHeatExp = param_get_double(ps, "HeliumHeatExp");

    #endif
        /*Massive neutrino parameters*/
        All.MassiveNuLinRespOn = param_get_int(ps, "MassiveNuLinRespOn");
        All.HybridNeutrinosOn = param_get_int(ps, "HybridNeutrinosOn");
        param_get_string2(ps, "LinearTransferFunction", All.CAMBTransferFunction);
        All.CAMBInputSpectrum_UnitLength_in_cm = param_get_double(ps, "InputSpectrum_UnitLength_in_cm");
        All.CP.MNu[0] = param_get_double(ps, "MNue");
        All.CP.MNu[1] = param_get_double(ps, "MNum");
        All.CP.MNu[2] = param_get_double(ps, "MNut");
        All.HybridVcrit = param_get_double(ps, "Vcrit");
        All.HybridNuPartTime = param_get_double(ps, "NuPartTime");
        if(All.MassiveNuLinRespOn && !All.CP.RadiationOn)
            endrun(2, "You have enabled (kspace) massive neutrinos without radiation, but this will give an inconsistent cosmology!\n");
        /*End massive neutrino parameters*/

        parameter_set_free(ps);

    #ifndef BLACK_HOLES
        if(All.BlackHoleOn)
        {
            endrun(1, "Code was compiled with black holes switched off but BlackHoleOn = 1. This does not work!\n");
        }
    #endif
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

        if(All.WindOn == 1) {
            if(All.StarformationOn == 0) {
                endrun(1, "You try to use the code with wind enabled,\n"
                          "but you did not switch on starformation.\nThis mode is not supported.\n");
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

        DensityKernel kernel;
        density_kernel_init(&kernel, 1.0);
        All.DesNumNgb = density_kernel_desnumngb(&kernel, All.DensityResolutionEta);

        message(1, "The Density Kernel type is %s\n", kernel.name);
        message(1, "The Density resolution is %g * mean separation, or %d neighbours\n",
                    All.DensityResolutionEta, All.DesNumNgb);

    }

    MPI_Bcast(&All, sizeof(All), MPI_BYTE, 0, MPI_COMM_WORLD);
}
