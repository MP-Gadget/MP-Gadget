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

#ifndef ALLVARS_H
#define ALLVARS_H

#include <mpi.h>
#include <stdio.h>

#include <gsl/gsl_rng.h>

#include <signal.h>
#define BREAKPOINT raise(SIGTRAP)
#ifdef _OPENMP
#include <omp.h>
#include <pthread.h>
#else
#ifndef __clang_analyzer__
#error no OMP
#endif
#define omp_get_max_threads()  (1)
#define omp_get_thread_num()  (0)
#endif

#include "cosmology.h"
#include "walltime.h"

#include "assert.h"
#include "physconst.h"
#include "types.h"

#define NEAREST(x) (((x)>0.5*All.BoxSize)?((x)-All.BoxSize):(((x)<-0.5*All.BoxSize)?((x)+All.BoxSize):(x)))

#ifndef  GENERATIONS
#define  GENERATIONS     4	/*!< Number of star particles that may be created per gas particle */
#endif

#define MAXHSML 30000.0

#ifndef  GAMMA
#define  GAMMA         (5.0/3.0)	/*!< adiabatic index of simulated gas */
#endif

#define  GAMMA_MINUS1  (GAMMA-1)

#define  HYDROGEN_MASSFRAC 0.76	/*!< mass fraction of hydrogen, relevant only for radiative cooling */

#define  METAL_YIELD       0.02	/*!< effective metal yield for star formation */

#define  MAX_REAL_NUMBER  1e37
#define  MIN_REAL_NUMBER  1e-37

#ifndef RCUT
/*! RCUT gives the maximum distance (in units of the scale used for the force split) out to which short-range
 * forces are evaluated in the short-range tree walk.
 */
#define RCUT  4.5
#endif

#define MAXITER 400

enum BlackHoleFeedbackMethod {
     BH_FEEDBACK_TOPHAT   = 0x2,
     BH_FEEDBACK_SPLINE   = 0x4,
     BH_FEEDBACK_MASS     = 0x8,
     BH_FEEDBACK_VOLUME   = 0x10,
     BH_FEEDBACK_OPTTHIN  = 0x20,
};

/*
 * additional sfr criterion in addition to density threshold
 * All.StarformationCriterion */
enum StarformationCriterion {
    SFR_CRITERION_DENSITY = 1,
    SFR_CRITERION_MOLECULAR_H2 = 3, /* 2 + 1 */
    SFR_CRITERION_SELFGRAVITY = 5,  /* 4 + 1 */
    /* below are additional flags in SELFGRAVITY */
    SFR_CRITERION_CONVERGENT_FLOW = 13, /* 8 + 4 + 1 */
    SFR_CRITERION_CONTINUOUS_CUTOFF= 21, /* 16 + 4 + 1 */
};

/*
 * wind models SH03, VS08 and OFJT10
 * All.WindModel */
enum WindModel {
    WIND_SUBGRID = 1,
    WIND_DECOUPLE_SPH = 2,
    WIND_USE_HALO = 4,
    WIND_FIXED_EFFICIENCY = 8,
    WIND_ISOTROPIC = 16,
};

enum DensityKernelType {
    DENSITY_KERNEL_CUBIC_SPLINE = 1,
    DENSITY_KERNEL_QUINTIC_SPLINE = 2,
    DENSITY_KERNEL_QUARTIC_SPLINE = 4,
};


static inline double DMAX(double a, double b) {
    if(a > b) return a;
    return b;
}
static inline double DMIN(double a, double b) {
    if(a < b) return a;
    return b;
}
static inline int IMAX(int a, int b) {
    if(a > b) return a;
    return b;
}
static inline int IMIN(int a, int b) {
    if(a < b) return a;
    return b;
}

/*********************************************************/
/*  Global variables                                     */
/*********************************************************/

extern int ThisTask;		/*!< the number of the local processor  */
extern int NTask;		/*!< number of processors */

/* variables for input/output , usually only used on process 0 */
extern FILE *FdEnergy,			/*!< file handle for energy.txt log-file. */
       *FdCPU;			/*!< file handle for cpu.txt log-file. */

#ifdef SFR
extern FILE *FdSfr;		/*!< file handle for sfr.txt log-file. */
#endif

#ifdef BLACK_HOLES
extern FILE *FdBlackHoles;	/*!< file handle for blackholes.txt log-file. */
#endif

/*! This structure contains data which is the SAME for all tasks (mostly code parameters read from the
 * parameter file).  Holding this data in a structure is convenient for writing/reading the restart file, and
 * it allows the introduction of new global variables in a simple way. The only thing to do is to introduce
 * them into this structure.
 */
extern struct global_data_all_processes
{
/* THe following variables are set by petaio_read_header */
    int64_t TotNumPartInit; /* The initial total number of particles; we probably want to get rid of all references to this. */
    int64_t NTotalInit[6]; /* The initial number of total particles in the IC. */
    double TimeInit;		/* time of simulation start: if restarting from a snapshot this holds snapshot time.*/
    double TimeIC;       /* Time when the simulation ICs were generated*/
    double BoxSize;   /* Boxsize in case periodic boundary conditions are used */
    double MassTable[6]; /* Initial mass of particles */
    double UnitMass_in_g;		/*!< factor to convert internal mass unit to grams/h */
    double UnitVelocity_in_cm_per_s;	/*!< factor to convert intqernal velocity unit to cm/sec */
    double UnitLength_in_cm;		/*!< factor to convert internal length unit to cm/h */


/* end of read_header parameters */

    int NumThreads;     /* number of threads used to simulate OpenMP tls */

    struct {
        size_t BytesPerFile;   /* Number of bytes per physical file; this decides how many files bigfile creates each block */
        int WritersPerFile;    /* Number of concurrent writers per file; this decides number of writers */
        int NumWriters;        /* Number of concurrent writers, this caps number of writers */
        int MinNumWriters;        /* Min Number of concurrent writers, this caps number of writers */
        int EnableAggregatedIO;  /* Enable aggregated IO policy for small files.*/
        size_t AggregatedIOThreshold; /* bytes per writer above which to use non-aggregated IO (avoid OOM)*/
        int UsePeculiarVelocity;
    } IO;

    double BufferSize;		/*!< size of communication buffer in MB */

    double PartAllocFactor;	/*!< in order to maintain work-load balance, the particle load will usually
                              NOT be balanced.  Each processor allocates memory for PartAllocFactor times
                              the average number of particles to allow for that */

    double TreeAllocFactor;	/*!< Each processor allocates a number of nodes which is TreeAllocFactor times
                              the maximum(!) number of particles.  Note: A typical local tree for N
                              particles needs usually about ~0.65*N nodes. */

    double TopNodeAllocFactor;	/*!< Each processor allocates a number of nodes which is TreeAllocFactor times
                                  the maximum(!) number of particles.  Note: A typical local tree for N
                                  particles needs usually about ~0.65*N nodes. */

    int OutputPotential;        /*!< Flag whether to include the potential in snapshots*/

    /* some SPH parameters */

    int DesNumNgb;		/*!< Desired number of SPH neighbours */
    double DensityResolutionEta;		/*!< SPH resolution eta. See Price 2011. eq 12*/
    double MaxNumNgbDeviation;	/*!< Maximum allowed deviation neighbour number */
    double ArtBulkViscConst;	/*!< Sets the parameter \f$\alpha\f$ of the artificial viscosity */

    double InitGasTemp;		/*!< may be used to set the temperature in the IC's */
    double MinGasTemp;		/*!< may be used to set a floor for the gas temperature */
    double MinEgySpec;		/*!< the minimum allowed temperature expressed as energy per unit mass */

    /* system of units  */

    double UnitTime_in_s,		/*!< factor to convert internal time unit to seconds/h */
           UnitPressure_in_cgs,	/*!< factor to convert internal pressure unit to cgs units (little 'h' still
                                  around!) */
           UnitDensity_in_cgs,		/*!< factor to convert internal length unit to g/cm^3*h^2 */
           UnitCoolingRate_in_cgs,	/*!< factor to convert internal cooling rate to cgs units */
           UnitEnergy_in_cgs,		/*!< factor to convert internal energy to cgs units */
           UnitTime_in_Megayears,	/*!< factor to convert internal time to megayears/h */
           GravityConstantInternal,	/*!< If set to zero in the parameterfile, the internal value of the
                                      gravitational constant is set to the Newtonian value based on the system of
                                      units specified. Otherwise the value provided is taken as internal gravity
                                      constant G. */
           G;				/*!< Gravity-constant in internal units */
    double UnitDensity_in_Gev_per_cm3; /*!< factor to convert internal density unit to GeV/c^2 / cm^3 */
    /* Cosmology */
    Cosmology CP;

    /* Code options */
    /* Number of sub-domains per processor. TopNodes are refined so that no TopNode contains
     * no more than 1/(DODF * NTask) fraction of the work.
     * Then the load balancer will aim to produce DODF*NTask equal-sized chunks, distributed
     * evenly across MPI ranks.*/
    int DomainOverDecompositionFactor;
    int DomainUseGlobalSorting;
    /* Sets average TopNodes per MPI rank. Like DomainOverDecompositionFactor
     * but only changes refinement, not load balancing.*/
    int TopNodeIncreaseFactor;

    int CoolingOn;  /* if cooling is enabled */
    double UVRedshiftThreshold;  /* Initial redshift of UV background. */
    int HydroOn;  /*  if hydro force is enabled */
    int DensityOn;  /*  if SPH density computation is enabled */
    int TreeGravOn;     /* tree gravity force is enabled*/
    int BlackHoleOn;  /* if black holes are enabled */
    int StarformationOn;  /* if star formation is enabled */
    int WindOn; /* if Wind is enabled */
    int MassiveNuLinRespOn; /*!< flags that massive neutrinos using the linear
                                 response code of Ali-Haimoud & Bird 2013.*/
    int HybridNeutrinosOn; /*!< Flags that hybrid neutrinos are enabled */
    double HybridVcrit; /*!< Critical velocity switching between particle
                          and analytic solvers when hybrid neutrinos are on*/
    double HybridNuPartTime; /*!< Redshift at which hybrid neutrinos switch on*/
    char CAMBTransferFunction[512]; /*!< CAMB transfer function for initial neutrino power*/
    double CAMBInputSpectrum_UnitLength_in_cm; /*!< Units of CAMB transfer function*/

    enum StarformationCriterion StarformationCriterion;  /*!< flags that star formation is enabled */
    enum WindModel WindModel;  /*!< flags that star formation is enabled */

    int MakeGlassFile; /*!< flags that gravity is reversed and we are making a glass file*/
    int FastParticleType; /*!< flags a particle species to exclude timestep calculations.*/
    /* parameters determining output frequency */

    int SnapshotFileCount;	/*!< number of snapshot that is written next */
    int InitSnapshotCount;  /*!< Number of first snapshot written this run*/
    double AutoSnapshotTime;    /*!< cpu-time between regularly generated snapshots. */

    /* Current time of the simulation, global step, and end of simulation */

    double Time,			/*!< current time of the simulation */
           TimeStep,			/*!< difference between current times of previous and current timestep */
           TimeMax;			/*!< marks the point of time until the simulation is to be evolved */

    struct {
        double a;
        double a3inv;
        double a2inv;
        double fac_egy;
        double hubble;
        double hubble_a2;
    } cf;

    /* variables for organizing discrete timeline */

    inttime_t Ti_Current;		/*!< current time on integer timeline */

    int Nmesh;

    /* variables that keep track of cumulative CPU consumption */

    double TimeLimitCPU;
    struct ClockTable CT;

    /* tree code opening criterion */

    double ErrTolForceAcc;	/*!< parameter for relative opening criterion in tree walk */

    /*! The scale of the short-range/long-range force split in units of FFT-mesh cells */
    double Asmth;

    /* adjusts accuracy of time-integration */

    double ErrTolIntAccuracy;	/*!< accuracy tolerance parameter \f$ \eta \f$ for timestep criterion. The
                                  timesteps is \f$ \Delta t = \sqrt{\frac{2 \eta eps}{a}} \f$ */

    int ForceEqualTimesteps; /*If true, all timesteps have the same timestep, the smallest allowed.*/
    double MinSizeTimestep,	/*!< minimum allowed timestep. Normally, the simulation terminates if the
                              timestep determined by the timestep criteria falls below this limit. */
           MaxSizeTimestep;		/*!< maximum allowed timestep */

    double MaxRMSDisplacementFac;	/*!< this determines a global timestep criterion for cosmological simulations
                                      in comoving coordinates.  To this end, the code computes the rms velocity
                                      of all particles, and limits the timestep such that the rms displacement
                                      is a fraction of the mean particle separation (determined from the
                                      particle mass and the cosmological parameters). This parameter specifies
                                      this fraction. */

    double MaxGasVel; /* Limit on Gas velocity */
    int MaxMemSizePerNode;

    double CourantFac;		/*!< SPH-Courant factor */


    /* gravitational and hydrodynamical softening lengths (given in terms of an `equivalent' Plummer softening
     * length)
     *
     * five groups of particles are supported 0=gas,1=halo,2=disk,3=bulge,4=stars
     */
    double MinGasHsmlFractional,	/*!< minimum allowed SPH smoothing length in units of SPH gravitational
                                      softening length */
           MinGasHsml;			/*!< minimum allowed SPH smoothing length */


    enum DensityKernelType DensityKernelType;  /* 0 for Cubic Spline,  (recmd NumNgb = 33)
                               1 for Quintic spline (recmd  NumNgb = 97)
                             */
    double DensityContrastLimit; /* limit of density contrast ratio for hydro force calculation */
    double HydroCostFactor; /* cost factor for hydro in load balancing. */

    double GravitySoftening; /* Softening as a fraction of DM mean separation. */
    double GravitySofteningGas;  /* if 0, enable adaptive gravitational softening for gas particles, which uses the Hsml as ForceSoftening */

    double TreeNodeMinSize; /* The minimum size of a Force Tree Node in length units. */
    double MeanSeparation[6]; /* mean separation between particles. 0 if the species doesn't exist. */

    /* some filenames */
    char InitCondFile[100],
         TreeCoolFile[100],
         MetalCoolFile[100],
         OutputDir[100],
         SnapshotFileBase[100],
         FOFFileBase[100],
         EnergyFile[100],
         CpuFile[100];

    /*Should we store the energy to EnergyFile on PM timesteps.*/
    int OutputEnergyDebug;

    char UVFluctuationFile[100];

    double OutputListTimes[8192];
    int OutputListLength;

#ifdef SFR		/* star formation and feedback sector */
    double CritOverDensity;
    double CritPhysDensity;
    double OverDensThresh;
    double PhysDensThresh;
    double EgySpecSN;
    double FactorSN;
    double EgySpecCold;
    double FactorEVP;
    double FeedbackEnergy;
    double TempSupernova;
    double TempClouds;
    double MaxSfrTimescale;
    double WindFreeTravelLength;
    double WindFreeTravelDensFac;
    double FactorForSofterEQS;
    /* used in VS08 and SH03*/
    double WindEfficiency;
    double WindSpeed;
    double WindEnergyFraction;
    /* used in OFJT10*/
    double WindSigma0;
    double WindSpeedFactor;
#endif
    /*Lyman alpha forest specific parameters*/
    double QuickLymanAlphaProbability;
    int HeliumHeatOn;
    double HeliumHeatThresh;
    double HeliumHeatAmp;
    double HeliumHeatExp;

    double BlackHoleAccretionFactor;	/*!< Fraction of BH bondi accretion rate */
    double BlackHoleFeedbackFactor;	/*!< Fraction of the black luminosity feed into thermal feedback */
    enum BlackHoleFeedbackMethod BlackHoleFeedbackMethod;	/*!< method of the feedback*/
    double BlackHoleFeedbackRadius;	/*!< Radius the thermal feedback is fed comoving*/
    double BlackHoleFeedbackRadiusMaxPhys;	/*!< Radius the thermal cap */
    double SeedBlackHoleMass;	/*!< Seed black hole mass */
    double BlackHoleNgbFactor;	/*!< Factor by which the normal SPH neighbour should be increased/decreased */
    double BlackHoleMaxAccretionRadius;
    double BlackHoleEddingtonFactor;	/*! Factor above Eddington */
    int BlackHoleSoundSpeedFromPressure; /* 0 from Entropy, 1 from Pressure; */

    int SnapshotWithFOF; /*Flag that doing FOF for snapshot outputs is on*/
    int FOFSaveParticles ; /* saving particles in the fof group */
    double MinFoFMassForNewSeed;	/* Halo mass required before new seed is put in */
    double FOFHaloLinkingLength;
    double FOFHaloComovingLinkingLength; /* in code units */
    int FOFHaloMinLength;
    double TimeNextSeedingCheck;  /*Time for the next seed check.*/
    double TimeBetweenSeedingSearch; /*Factor to multiply TimeInit by to find the next seeding check.*/

    int RandomSeed; /*Initial seed for the random number table*/
}
All;

#endif
