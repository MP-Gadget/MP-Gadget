#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <libgadget/gravity.h>
#include <libgadget/densitykernel.h>
#include <libgadget/timebinmgr.h>
#include <libgadget/timestep.h>
#include <libgadget/utils.h>
#include <libgadget/treewalk.h>
#include <libgadget/cooling_rates.h>
#include <libgadget/winds.h>
#include <libgadget/sfr_eff.h>
#include <libgadget/blackhole.h>
#include <libgadget/density.h>
#include <libgadget/hydra.h>
#include <libgadget/fof.h>
#include <libgadget/init.h>
#include <libgadget/timebinmgr.h>
#include <libgadget/petaio.h>
#include <libgadget/cooling_qso_lightup.h>

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

static int
StarformationCriterionAction(ParameterSet * ps, char * name, void * data)
{
    int v = param_get_enum(ps, name);
    if(!HAS(v, SFR_CRITERION_DENSITY)) {
        message(1, "error: At least use SFR_CRITERION_DENSITY\n");
        return 1;
    }
    return 0;
}

static ParameterSet *
create_gadget_parameter_set()
{
    ParameterSet * ps = parameter_set_new();

    param_declare_string(ps, "InitCondFile", REQUIRED, NULL, "Path to the Initial Condition File");
    param_declare_string(ps, "OutputDir",    REQUIRED, NULL, "Prefix to the output files");

    static ParameterEnum DensityKernelTypeEnum [] = {
        {"cubic", DENSITY_KERNEL_CUBIC_SPLINE},
        {"quintic", DENSITY_KERNEL_QUINTIC_SPLINE},
        {"quartic", DENSITY_KERNEL_QUARTIC_SPLINE},
        {NULL, DENSITY_KERNEL_QUARTIC_SPLINE},
    } ;
    param_declare_enum(ps,    "DensityKernelType", DensityKernelTypeEnum, OPTIONAL, "quintic", "SPH density kernel to use. Supported values are cubic, quartic and quintic.");
    param_declare_string(ps, "SnapshotFileBase", OPTIONAL, "PART", "Base name of the snapshot files, _%03d will be appended to the name.");
    param_declare_string(ps, "FOFFileBase", OPTIONAL, "PIG", "Base name of the fof files, _%03d will be appended to the name.");
    param_declare_string(ps, "EnergyFile", OPTIONAL, "energy.txt", "File to output energy statistics.");
    param_declare_int(ps,    "OutputEnergyDebug", OPTIONAL, 0, "Should we output energy statistics to energy.txt");
    param_declare_string(ps, "CpuFile", OPTIONAL, "cpu.txt", "File to output cpu usage information");
    param_declare_string(ps, "OutputList", REQUIRED, NULL, "List of output scale factors.");

    /*Cosmology parameters*/
    param_declare_double(ps, "Omega0", REQUIRED, 0.2814, "Total matter density at z=0");
    param_declare_double(ps, "CMBTemperature", OPTIONAL, 2.7255,
            "Present-day CMB temperature in Kelvin, default from Fixsen 2009; affects background if RadiationOn is set.");
    param_declare_double(ps, "OmegaBaryon", OPTIONAL, -1, "Baryon density at z=0");
    param_declare_double(ps, "OmegaLambda", OPTIONAL, -1, "Dark energy density at z=0");
    param_declare_double(ps, "Omega_fld", OPTIONAL, 0, "Energy density of dark energy fluid.");
    param_declare_double(ps, "w0_fld", OPTIONAL, -1., "Dark energy equation of state.");
    param_declare_double(ps, "wa_fld", OPTIONAL, 0, "Dark energy evolution parameter.");
    param_declare_double(ps, "Omega_ur", OPTIONAL, 0, "Extra radiation density, eg, a sterile neutrino");
    param_declare_double(ps, "HubbleParam", OPTIONAL, -1, "Hubble parameter. Does not affect gravity. Used only for cooling and star formation.");
    /*End cosmology parameters*/

    param_declare_int(ps,    "OutputPotential", OPTIONAL, 1, "Save the potential in snapshots.");
    param_declare_int(ps,    "OutputHeliumFractions", OPTIONAL, 0, "Save the helium ionic fractions in snapshots.");
    param_declare_int(ps,    "OutputDebugFields", OPTIONAL, 0, "Save a large number of debug fields in snapshots.");
    param_declare_int(ps,    "ShowBacktrace", OPTIONAL, 1, "Print a backtrace on crash. Hangs on stampede.");
    param_declare_double(ps,    "MaxMemSizePerNode", OPTIONAL, 0.6, "Pre-allocate this much memory per computing node/ host, in MB. Passing < 1 allocates a fraction of total available memory per node, defaults to 0.6 available memory.");
    param_declare_double(ps, "AutoSnapshotTime", OPTIONAL, 0, "Seconds after which to automatically generate a snapshot if nothing is output.");

    param_declare_double(ps, "TimeMax", OPTIONAL, 1.0, "Scale factor to end run.");
    param_declare_double(ps, "TimeLimitCPU", REQUIRED, 0, "CPU time to run for in seconds. Code will stop if it notices that the time to end of the next PM step is longer than the remaining time.");

    param_declare_int   (ps, "DomainOverDecompositionFactor", OPTIONAL, 4, "Create on average this number of sub domains on a MPI rank. Load balancer will then move these subdomains around to equalize the work per rank. Higher numbers improve the load balancing but make domain more expensive.");
    param_declare_double(ps, "RandomParticleOffset", OPTIONAL, 8., "Internally shift the particles within a periodic box by a random fraction of a PM grid cell each domain decomposition, ensuring that tree openings are decorrelated between timesteps. This shift is subtracted before particles are saved.");

    param_declare_int   (ps, "DomainUseGlobalSorting", OPTIONAL, 1, "Determining the initial refinement of chunks globally. Enabling this produces better domains at costs of slowing down the domain decomposition.");
    param_declare_double(ps, "ErrTolIntAccuracy", OPTIONAL, 0.02, "Controls the length of the short-range timestep. Smaller values are shorter timesteps.");
    param_declare_double(ps, "ErrTolForceAcc", OPTIONAL, 0.002, "Force accuracy required from tree. Controls tree opening criteria. Lower values are more accurate.");
    param_declare_double(ps, "BHOpeningAngle", OPTIONAL, 0.175, "Barnes-Hut opening angle. Alternative purely geometric tree opening angle. Lower values are more accurate.");
    param_declare_double(ps, "TreeRcut", OPTIONAL, 6, "Number of mesh cells at which we cease walking.");
    param_declare_int(ps, "TreeUseBH", OPTIONAL, 2, "If 1, use Barnes-Hut opening angle rather than the standard Gadget acceleration based opening angle. If 2, use BH criterion for the first timestep only, before we have relative accelerations.");
    param_declare_double(ps, "Asmth", OPTIONAL, 1.5, "The scale of the short-range/long-range force split in units of FFT-mesh cells."
                                                      "Larger values suppresses grid anisotropy. ShortRangeForceWindowType = erfc supports any value. 'exact' only supports 1.5. ");
    param_declare_int(ps,    "Nmesh", OPTIONAL, -1, "Size of the PM grid on which to compute the long-range force.");

    static ParameterEnum ShortRangeForceWindowTypeEnum [] = {
        {"exact", SHORTRANGE_FORCE_WINDOW_TYPE_EXACT},
        {"erfc", SHORTRANGE_FORCE_WINDOW_TYPE_ERFC },
        {NULL, SHORTRANGE_FORCE_WINDOW_TYPE_EXACT },
    };
    param_declare_enum(ps,    "ShortRangeForceWindowType", ShortRangeForceWindowTypeEnum, OPTIONAL, "exact", "type of shortrange window, exact or erfc (default is exact) ");

    param_declare_double(ps, "MinGasHsmlFractional", OPTIONAL, 0, "Minimal gas Hsml as a fraction of gravity softening.");
    param_declare_double(ps, "MaxGasVel", OPTIONAL, 3e5, "Maximal limit on the gas velocity in km/s. By default speed of light.");

    /*Setting MaxSizeTimestep = 0.05 increases the power on large scales by a constant factor of 1.002.*/
    param_declare_double(ps, "MaxSizeTimestep", OPTIONAL, 0.1, "Maximum size of the PM timestep (as delta-a).");
    param_declare_double(ps, "MinSizeTimestep", OPTIONAL, 0, "Minimum size of the PM timestep.");
    param_declare_int(ps, "ForceEqualTimesteps", OPTIONAL, 0, "Force all (tree) timesteps to be the same, and equal to the smallest required.");

    /* MaxRMSDisplacementFac = 0.1 increases the power on large scales by a small constant factor of 1.0005. */
    param_declare_double(ps, "MaxRMSDisplacementFac", OPTIONAL, 0.2, "Controls the length of the PM timestep. Max RMS displacement per timestep in units of the mean particle separation.");
    param_declare_double(ps, "ArtBulkViscConst", OPTIONAL, 0.75, "Artificial viscosity constant for SPH.");
    param_declare_double(ps, "CourantFac", OPTIONAL, 0.15, "Courant factor for the timestepping.");
    param_declare_double(ps, "DensityResolutionEta", OPTIONAL, 1.0, "Resolution eta factor (See Price 2008) 1 = 33 for Cubic Spline");

    param_declare_double(ps, "DensityContrastLimit", OPTIONAL, 100, "Has an effect only if DensityIndepndentSphOn=1. If = 0 enables the grad-h term in the SPH calculation. If > 0 also sets a maximum density contrast for hydro force calculation.");
    param_declare_double(ps, "MaxNumNgbDeviation", OPTIONAL, 2, "Maximal deviation from the desired number of neighbours for each SPH particle.");
    param_declare_double(ps, "HydroCostFactor", OPTIONAL, 1, "Cost factor of hydro calculation: this allows gas particles to be considered more expensive than gravity computations.");

    param_declare_int(ps, "BytesPerFile", OPTIONAL, 1024 * 1024 * 1024, "number of bytes per file");
    param_declare_int(ps, "NumWriters", OPTIONAL, 0, "Max number of concurrent writer processes. 0 implies Number of Tasks; ");
    param_declare_int(ps, "MinNumWriters", OPTIONAL, 1, "Min number of concurrent writer processes. We increase number of Files to avoid too few writers. ");
    param_declare_int(ps, "WritersPerFile", OPTIONAL, 8, "Number of Writer groups assigned to a file; total number of writers is capped by NumWriters.");

    param_declare_int(ps, "EnableAggregatedIO", OPTIONAL, 0, "Use the Aggregated IO policy for small data set (Experimental).");
    param_declare_int(ps, "AggregatedIOThreshold", OPTIONAL, 1024 * 1024 * 256, "Max number of bytes on a writer before reverting to throttled IO.");

    /*Parameters of the cooling module*/
    param_declare_int(ps, "CoolingOn", REQUIRED, 0, "Enables cooling");
    param_declare_string(ps, "TreeCoolFile", OPTIONAL, "", "Path to the Cooling Table");
    param_declare_string(ps, "MetalCoolFile", OPTIONAL, "", "Path to the Metal Cooling Table. Empty string disables metal cooling. Refer to cooling.c");
    param_declare_string(ps, "ReionHistFile", OPTIONAL, "", "Path to the file containing the helium III reionization table. Used if QSOLightupOn = 1.");
    param_declare_string(ps, "UVFluctuationFile", OPTIONAL, "", "Path to the UVFluctation Table. Refer to cooling.c.");

    param_declare_double(ps, "UVRedshiftThreshold", OPTIONAL, -1.0, "Earliest Redshift that UV background is enabled. This modulates UVFluctuation and TreeCool globally. Default -1.0 means no modulation.");
    static ParameterEnum CoolingTypeTable [] = {
        {"KWH92", KWH92 },
        {"Enzo2Nyx", Enzo2Nyx },
        {"Sherwood", Sherwood },
        {NULL, Cen92 },
    };
    static ParameterEnum RecombTypeTable [] = {
        {"Cen92", Cen92 },
        {"Verner96", Verner96 },
        {"Badnell06", Badnell06},
        {NULL, Cen92 },
    };
    param_declare_enum(ps, "CoolingRates", CoolingTypeTable, OPTIONAL, "Sherwood", "Which cooling rate table to use. Options are KWH92 (old gadget default), Enzo2Nyx and Sherwood (new default).");
    param_declare_enum(ps, "RecombRates", RecombTypeTable, OPTIONAL, "Verner96", "Which recombination rate table to use. Options are Cen92 (old gadget default), Verner96 (new default), Badnell06");
    param_declare_int(ps, "SelfShieldingOn", OPTIONAL, 1, "Enable a correction in the cooling table for self-shielding.");
    param_declare_double(ps, "PhotoIonizeFactor", OPTIONAL, 1, "Scale the TreeCool table by this factor.");
    param_declare_int(ps, "PhotoIonizationOn", OPTIONAL, 1, "Should PhotoIonization be enabled.");
    /* End cooling module parameters*/

    param_declare_int(ps, "HydroOn", OPTIONAL, 1, "Enables hydro force");
    param_declare_int(ps, "DensityOn", OPTIONAL, 1, "Enables SPH density computation.");
    param_declare_int(ps, "DensityIndependentSphOn", REQUIRED, 1, "Enables density-independent (pressure-entropy) SPH.");
    param_declare_int(ps, "TreeGravOn", OPTIONAL, 1, "Enables tree gravity");
    param_declare_int(ps, "RadiationOn", OPTIONAL, 1, "Include radiation density in the background evolution.");
    param_declare_int(ps, "FastParticleType", OPTIONAL, 2, "Particles of this type will not decrease the timestep. Default neutrinos.");
    param_declare_double(ps, "PairwiseActiveFraction", OPTIONAL, 5e-7, "Pairwise gravity instead of tree gravity is used if N(active particles) / N(particles) is less than this.");

    param_declare_double(ps, "GravitySoftening", OPTIONAL, 1./30., "Softening for collisionless particles; units of mean separation of DM. ForceSoftening is 2.8 times this.");
    param_declare_double(ps, "GravitySofteningGas", OPTIONAL, 1./30., "Softening for collisional particles (Gas); units of mean separation of DM; 0 to use Hsml of last step. ");

    param_declare_int(ps, "ImportBufferBoost", OPTIONAL, 2, "Memory factor to allow for there being more particles imported during treewlk than exported. Increase this if code crashes during treewalk with out of memory.");
    param_declare_double(ps, "PartAllocFactor", OPTIONAL, 1.5, "Over-allocation factor of particles. The load can be imbalanced to allow for the work to be more balanced.");
    param_declare_double(ps, "TopNodeAllocFactor", OPTIONAL, 0.5, "Initial TopNode allocation as a fraction of maximum particle number.");
    param_declare_double(ps, "SlotsIncreaseFactor", OPTIONAL, 0.01, "Percentage factor to increase slot allocation by when requested.");

    param_declare_double(ps, "InitGasTemp", OPTIONAL, -1, "Initial gas temperature. By default set to CMB temperature at starting redshift.");
    param_declare_double(ps, "MinGasTemp", OPTIONAL, 5, "Minimum gas temperature");

    param_declare_int(ps, "SnapshotWithFOF", REQUIRED, 0, "Enable Friends-of-Friends halo finder.");
    param_declare_int(ps, "FOFSaveParticles", OPTIONAL, 1, "Save particles in the FOF catalog.");
    param_declare_double(ps, "FOFHaloLinkingLength", OPTIONAL, 0.2, "Linking length for Friends of Friends halos.");
    param_declare_int(ps, "FOFHaloMinLength", OPTIONAL, 32, "Minimum number of particles per FOF Halo.");
    param_declare_double(ps, "MinFoFMassForNewSeed", OPTIONAL, 5, "Minimal halo mass for seeding tracer particles in internal mass units.");
    param_declare_double(ps, "TimeBetweenSeedingSearch", OPTIONAL, 1.04, "Scale factor fraction increase between Seeding Attempts.");

    /*Black holes*/
    param_declare_int(ps, "BlackHoleOn", REQUIRED, 1, "Master switch to enable black hole formation and feedback. If this is on, type 5 particles are treated as black holes.");
    param_declare_double(ps, "BlackHoleAccretionFactor", OPTIONAL, 100, "BH accretion boosting factor relative to the rate from the Bondi accretion model.");
    param_declare_double(ps, "BlackHoleEddingtonFactor", OPTIONAL, 3, "Maximum Black hole accretion as a function of Eddington.");
    param_declare_double(ps, "SeedBlackHoleMass", OPTIONAL, 5e-5, "Mass of initial black hole seed in internal mass units.");

    param_declare_double(ps, "BlackHoleNgbFactor", OPTIONAL, 2, "Factor by which to increase the number of neighbours for a black hole.");

    param_declare_double(ps, "BlackHoleMaxAccretionRadius", OPTIONAL, 99999., "Maximum neighbour search radius for black holes. Rarely needed.");
    param_declare_double(ps, "BlackHoleFeedbackFactor", OPTIONAL, 0.05, " Fraction of the black hole luminosity to turn into thermal energy");
    param_declare_double(ps, "BlackHoleFeedbackRadius", OPTIONAL, 0, "If set, the comoving radius at which the black hole feedback energy is deposited.");

    param_declare_double(ps, "BlackHoleFeedbackRadiusMaxPhys", OPTIONAL, 0, "If set, the physical radius at which the black hole feedback energy is deposited. When both this flag and BlackHoleFeedbackRadius are both set, the smaller radius is used.");
    param_declare_int(ps,"WriteBlackHoleDetails",OPTIONAL, 0, "If set, output BH details at every time step.");

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
        {"density", SFR_CRITERION_DENSITY}, /* SH03 density model for star formation*/
        {"h2", SFR_CRITERION_MOLECULAR_H2}, /* Form stars depending on the computed
                                               molecular gas fraction as a function of metallicity. */
        {"selfgravity", SFR_CRITERION_SELFGRAVITY}, /* Form stars only when the gas is self-gravitating. From Phil Hopkins.*/
        {"convergent", SFR_CRITERION_CONVERGENT_FLOW}, /* Modify self-gravitating star formation to form stars only when the gas flow is convergent. From Phil Hopkins.*/
        {"continuous", SFR_CRITERION_CONTINUOUS_CUTOFF}, /* Modify self-gravitating star formation to smooth the star formation threshold. From Phil Hopkins.*/
        {NULL, SFR_CRITERION_DENSITY},
    };

    static ParameterEnum WindModelEnum [] = {
        {"subgrid", WIND_SUBGRID}, /* the vanilla model of SH03, in which winds are included by returning energy to the star forming regions. This cools away and is ineffective.*/
        {"decouple", WIND_DECOUPLE_SPH}, /* Specifies that wind particles are created temporarily decoupled from the gas dynamics */
        {"halo", WIND_USE_HALO}, /* Wind speeds depend on the halo circular velocity*/
        {"fixedefficiency", WIND_FIXED_EFFICIENCY}, /* Winds have a fixed efficiency and thus fixed wind speed*/
        {"sh03", WIND_SUBGRID | WIND_DECOUPLE_SPH | WIND_FIXED_EFFICIENCY} , /*The canonical model of Spring & Hernquist 2003*/
        {"vs08", WIND_FIXED_EFFICIENCY},
        {"ofjt10", WIND_USE_HALO | WIND_DECOUPLE_SPH},
        {"isotropic", 0}, /*Wind direction is always random and isotropic.*/
        {NULL, WIND_SUBGRID | WIND_DECOUPLE_SPH | WIND_FIXED_EFFICIENCY},
    };

    param_declare_int(ps, "StarformationOn", REQUIRED, 0, "Enables star formation");
    param_declare_int(ps, "WindOn", REQUIRED, 0, "Enables wind feedback");
    param_declare_enum(ps, "StarformationCriterion",
            StarformationCriterionEnum, OPTIONAL, "density", "Extra star formation criteria to use. Default is density which corresponds to the SH03 model.");

    /*See Springel & Hernquist 2003 for the meaning of these parameters*/
    param_declare_double(ps, "CritOverDensity", OPTIONAL, 57.7, "Threshold over-density (in units of the critical density) for gas to be star forming.");
    param_declare_double(ps, "CritPhysDensity", OPTIONAL, 0, "Threshold physical density (in protons/cm^3) for gas to be star forming. If zero this is worked out from CritOverDensity.");

    param_declare_double(ps, "FactorSN", OPTIONAL, 0.1, "Fraction of the gas energy which is locally returned as supernovae on star formation.");
    param_declare_double(ps, "FactorEVP", OPTIONAL, 1000, "Parameter of the SH03 model, controlling the energy of the hot gas.");
    param_declare_double(ps, "TempSupernova", OPTIONAL, 1e8, "Temperature of the supernovae remnants in K.");
    param_declare_double(ps, "TempClouds", OPTIONAL, 1000, "Temperature of the cold star forming clouds in K.");
    param_declare_double(ps, "MaxSfrTimescale", OPTIONAL, 1.5, "Maximum star formation time in units of the density threshold.");
    param_declare_int(ps, "Generations", OPTIONAL, 2, "Number of stars to create per gas particle.");
    param_declare_enum(ps, "WindModel", WindModelEnum, OPTIONAL, "ofjt10", "Wind model to use. Default is the varying wind velocity model with isotropic winds.");

    /* The following two are for VS08 and SH03*/
    param_declare_double(ps, "WindEfficiency", OPTIONAL, 2.0, "Fraction of the stellar mass that goes into a wind. Needs sh03 or vs08 wind models.");
    param_declare_double(ps, "WindEnergyFraction", OPTIONAL, 1.0, "Fraction of the available energy that goes into winds.");

    /* The following two are for OFJT10*/
    param_declare_double(ps, "WindSigma0", OPTIONAL, 353, "Reference halo circular velocity at which to evaluate wind speed. Needs ofjt10 wind model.");
    param_declare_double(ps, "WindSpeedFactor", OPTIONAL, 3.7, "Factor connecting wind speed to halo circular velocity. ofjt10 wind model.");

    param_declare_double(ps, "WindFreeTravelLength", OPTIONAL, 20, "Expected decoupling distance for the wind in internal distance units.");
    param_declare_double(ps, "WindFreeTravelDensFac", OPTIONAL, 0.1, "If the density of the wind particle drops below this factor of the star formation density threshold, the gas will recouple.");

    param_declare_int(ps, "RandomSeed", OPTIONAL, 42, "Random number generator initial seed. Used to form stars.");

    /*These parameters are Lyman alpha forest specific*/
    param_declare_double(ps, "QuickLymanAlphaProbability", OPTIONAL, 0, "Probability gas is turned directly into stars, irrespective of pressure. One is equivalent to quick lyman alpha star formation.");
    param_declare_double(ps, "QuickLymanAlphaTempThresh", OPTIONAL, 1e5, "Temperature threshold for gas to be star forming in the quick lyman alpha model, in K. Gas above this temperature does not form stars.");
    param_declare_double(ps, "HydrogenHeatAmp", OPTIONAL, 1, "Density-independent heat boost to hydrogen.");
    /* Enable model for helium reionisation which adds extra photo-heating to under-dense gas.
     * Extra heating has the form: H = Amp * (rho / rho_c(z=0))^Exp
     * but is density-independent when rho / rho_c > Thresh. */
    param_declare_int(ps, "HeliumHeatOn", OPTIONAL, 0, "Change photo-heating rate to model helium reionisation on underdense gas.");
    param_declare_double(ps, "HeliumHeatThresh", OPTIONAL, 10, "Overdensity above which heating is density-independent.");
    param_declare_double(ps, "HeliumHeatAmp", OPTIONAL, 1, "Density-independent heat boost. Changes mean temperature.");
    param_declare_double(ps, "HeliumHeatExp", OPTIONAL, 0, "Density dependent heat boost (exponent). Changes gamma.");
    /*End of star formation parameters*/
    /* Parameters for the QSO lightup model for helium reionization*/
    param_declare_int(ps, "QSOLightupOn", OPTIONAL, 0, "Enable the quasar lighup model for helium reionization");
    /* Default QSO BH masses correspond to the Illustris BHs hosted in halos between 2x10^12 and 10^13 solar masses.
     * In small boxes this may be too small.*/
    param_declare_double(ps, "QSOMaxMass", OPTIONAL, 1000, "Maximum mass of a halo potentially hosting a quasar in internal mass units.");
    param_declare_double(ps, "QSOMinMass", OPTIONAL, 100, "Minimum mass of a halo potentially hosting a quasar in internal mass units.");
    param_declare_double(ps, "QSOMeanBubble", OPTIONAL, 30000, "Mean size of the ionizing bubble around a quasar. By default 30 Mpc.");
    param_declare_double(ps, "QSOVarBubble", OPTIONAL, 0, "Variance of the ionizing bubble around a quasar. By default zero so all bubbles are the same size");
    param_declare_double(ps, "QSOHeIIIReionFinishFrac", OPTIONAL, 0.95, "Reionization fraction at which all particles are flash-reionized instead of having quasar bubbles placed.");

    /*Parameters for the massive neutrino model*/
    param_declare_int(ps, "MassiveNuLinRespOn", REQUIRED, 0, "Enables linear response massive neutrinos of 1209.0461. Make sure you enable radiation too.");
    param_declare_int(ps, "HybridNeutrinosOn", OPTIONAL, 0, "Enables hybrid massive neutrinos, where some density is followed analytically, and some with particles. Requires MassivenuLinRespOn");
    param_declare_double(ps, "MNue", OPTIONAL, 0, "First neutrino mass in eV.");
    param_declare_double(ps, "MNum", OPTIONAL, 0, "Second neutrino mass in eV.");
    param_declare_double(ps, "MNut", OPTIONAL, 0, "Third neutrino mass in eV.");
    param_declare_double(ps, "Vcrit", OPTIONAL, 500., "For hybrid neutrinos: Critical velocity (in km/s) in the Fermi-Dirac distribution below which the neutrinos are particles in the ICs.");
    param_declare_double(ps, "NuPartTime", OPTIONAL, 0.3333333, "Scale factor at which to turn on hybrid neutrino particles.");
    /*End parameters for the massive neutrino model*/

    param_set_action(ps, "BlackHoleFeedbackMethod", BlackHoleFeedbackMethodAction, NULL);
    param_set_action(ps, "StarformationCriterion", StarformationCriterionAction, NULL);
    param_set_action(ps, "OutputList", OutputListAction, NULL);

    return ps;
}

/*! This function parses the parameterfile in a simple way.  Each paramater is
 *  defined by a keyword (`tag'), and can be either of type douple, int, or
 *  character string.  The routine makes sure that each parameter appears
 *  exactly once in the parameterfile, otherwise error messages are
 *  produced that complain about the missing parameters.
 */
void read_parameter_file(char *fname, int * ShowBacktrace, double * MaxMemSizePerNode)
{
    ParameterSet * ps = create_gadget_parameter_set();

    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    char * error;
    if(0 != param_parse_file(ps, fname, &error)) {
        endrun(1, "Parsing %s failed: %s\n", fname, error);
    }
    if(0 != param_validate(ps, &error)) {
        endrun(1, "Validation of %s failed: %s\n", fname, error);
    }

    message(0, "----------- Running with Parameters ----------\n");
    if(ThisTask == 0)
        param_dump(ps, stdout);
    message(0, "----------------------------------------------\n");

    *ShowBacktrace = param_get_int(ps, "ShowBacktrace");
    *MaxMemSizePerNode = param_get_double(ps, "MaxMemSizePerNode");
    if(*MaxMemSizePerNode <= 1) {
        *MaxMemSizePerNode *= get_physmem_bytes() / (1024. * 1024.);
    }

    /*Initialize per-module parameters.*/
    set_init_params(ps);
    set_petaio_params(ps);
    set_timestep_params(ps);
    set_cooling_params(ps);
    set_density_params(ps);
    set_hydro_params(ps);
    set_qso_lightup_params(ps);
    set_treewalk_params(ps);
    set_gravshort_tree_params(ps);
    set_domain_params(ps);
    set_sfr_params(ps);
    set_winds_params(ps);
    set_fof_params(ps);
    set_blackhole_params(ps);

    parameter_set_free(ps);
}
