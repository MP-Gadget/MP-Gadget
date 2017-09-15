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

#include "config-migrate.h"

#include <mpi.h>
#include <stdio.h>
#include <stdint.h>

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
#include "peano.h"
#include "physconst.h"


#define NEAREST(x) (((x)>0.5*All.BoxSize)?((x)-All.BoxSize):(((x)<-0.5*All.BoxSize)?((x)+All.BoxSize):(x)))

#ifndef  GENERATIONS
#define  GENERATIONS     4	/*!< Number of star particles that may be created per gas particle */
#endif

#define MAXHSML 30000.0

#ifdef ONEDIM
#define DIMS 1
#else
#ifdef TWODIMS    /* will only be compiled in 2D case */
#define DIMS 2
#else
#define DIMS 3
#endif
#endif

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

typedef uint32_t binmask_t;
typedef uint32_t inttime_t;

#define BINMASK_ALL ((uint32_t) (-1))
#define BINMASK(i) (1u << i)

typedef uint64_t MyIDType;

typedef LOW_PRECISION MyFloat;
typedef HIGH_PRECISION MyDouble;

#define HAS(val, flag) ((flag & (val)) == (flag))
#ifdef BLACK_HOLES
enum BlackHoleFeedbackMethod {
     BH_FEEDBACK_TOPHAT   = 0x2,
     BH_FEEDBACK_SPLINE   = 0x4,
     BH_FEEDBACK_MASS     = 0x8,
     BH_FEEDBACK_VOLUME   = 0x10,
     BH_FEEDBACK_OPTTHIN  = 0x20,
};
#endif
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
    WINDS_NONE = 0,
    WINDS_SUBGRID = 1,
    WINDS_DECOUPLE_SPH = 2,
    WINDS_USE_HALO = 4,
    WINDS_FIXED_EFFICIENCY = 8,
    WINDS_ISOTROPIC = 16,
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

#define FACT1 0.366025403785	/* FACT1 = 0.5 * (sqrt(3)-1) */
#define FACT2 0.86602540        /* FACT2 = 0.5 * sqrt(3) */



/*********************************************************/
/*  Global variables                                     */
/*********************************************************/

extern int ThisTask;		/*!< the number of the local processor  */
extern int NTask;		/*!< number of processors */

extern int NumPart;		/*!< number of particles on the LOCAL processor */

/* Local number of particles; this is accurate after a GC */
extern int64_t NLocal[6];
extern int64_t NTotal[6];

/* Number of used BHP slots */
extern int N_bh_slots;
extern int N_sph_slots;

extern gsl_rng *random_generator;	/*!< the random number generator used */

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
    double TimeInit;		/* time of initial conditions of the simulation */
    double BoxSize;   /* Boxsize in case periodic boundary conditions are used */
    double MassTable[6]; /* Initial mass of particles */
    double UnitMass_in_g;		/*!< factor to convert internal mass unit to grams/h */
    double UnitVelocity_in_cm_per_s;	/*!< factor to convert intqernal velocity unit to cm/sec */
    double UnitLength_in_cm;		/*!< factor to convert internal length unit to cm/h */


    int MaxPart;			/*!< This gives the maxmimum number of particles that can be stored on one
                              processor. */
    int MaxPartBh;		/*!< This gives the maxmimum number of BH particles that can be stored on one
                          processor. */

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
    /* Sets average TopNodes per MPI rank. Like DomainOverDecompositionFactor
     * but only changes refinement, not load balancing.*/
    int TopNodeIncreaseFactor;

    int CoolingOn;		/*!< flags that cooling is enabled */
    double UVRedshiftThreshold;		/* Initial redshift of UV background. */
    int HydroOn;		/*!< flags that hydro force is enabled */
    int DensityOn;		/*!< flags that SPH density computation is enabled */
    int TreeGravOn;          /*!< flags that tree gravity force is enabled*/
    int BlackHoleOn;		/*!< flags that black holes are enabled */
    int StarformationOn;		/*!< flags that star formation is enabled */
    enum StarformationCriterion StarformationCriterion;		/*!< flags that star formation is enabled */
    enum WindModel WindModel;		/*!< flags that star formation is enabled */

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
    double SofteningGas,		/*!< for type 0 */
           SofteningHalo,		/*!< for type 1 */
           SofteningDisk,		/*!< for type 2 */
           SofteningBulge,		/*!< for type 3 */
           SofteningStars,		/*!< for type 4 */
           SofteningBndry;		/*!< for type 5 */

    double SofteningGasMaxPhys,	/*!< for type 0 */
           SofteningHaloMaxPhys,	/*!< for type 1 */
           SofteningDiskMaxPhys,	/*!< for type 2 */
           SofteningBulgeMaxPhys,	/*!< for type 3 */
           SofteningStarsMaxPhys,	/*!< for type 4 */
           SofteningBndryMaxPhys;	/*!< for type 5 */

    double SofteningTable[6];	/*!< current (comoving) gravitational softening lengths for each particle type */
    double ForceSoftening[6];	/*!< the same, but multiplied by a factor 2.8 - at that scale the force is Newtonian */
    double MeanSeparation[6]; /* mean separation between particles. 0 if the species doesn't exist. */
    int AdaptiveGravsoftForGas; /*Flags that we have enabled adaptive gravitational softening for gas particles.
                                  This means that ForceSoftening[0] is unused. Instead pairwise interactions use 
                                  max(P[i].Hsml,ForceSoftening[P[j].Type]) for the particle is considered.*/

    /* some filenames */
    char InitCondFile[100],
         TreeCoolFile[100],
         MetalCoolFile[100],
         OutputDir[100],
         SnapshotFileBase[100],
         EnergyFile[100],
         CpuFile[100];

    /*Should we store the energy to EnergyFile on PM timesteps.*/
    int OutputEnergyDebug;

    char UVFluctuationFile[100];

    /*! table with desired output times, stored as log(a) */
    double OutputListTimes[8192];
    int OutputListLength;		/*!< number of times stored in table of desired output times */


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


#ifdef BLACK_HOLES
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
#endif

    int SnapshotWithFOF; /*Flag that doing FOF for snapshot outputs is on*/
    double MinFoFMassForNewSeed;	/* Halo mass required before new seed is put in */
    double FOFHaloLinkingLength;    
    double FOFHaloComovingLinkingLength; /* in code units */
    int FOFHaloMinLength;
    double TimeNextSeedingCheck;  /*Time for the next seed check.*/
    double TimeBetweenSeedingSearch; /*Factor to multiply TimeInit by to find the next seeding check.*/

}
All;
#ifdef _OPENMP
extern size_t BlockedParticleDrifts;
extern size_t TotalParticleDrifts;
#endif
/*! This structure holds all the information that is
 * stored for each particle of the simulation.
 */
extern struct particle_data
{
#ifdef OPENMP_USE_SPINLOCK
    pthread_spinlock_t SpinLock;
#endif

    float GravCost;     /*!< weight factor used for balancing the work-load */

    inttime_t Ti_drift;       /*!< current time of the particle position */
    inttime_t Ti_kick;        /*!< current time of the particle momentum */

    double Pos[3];   /*!< particle position at its current time */
    float Mass;     /*!< particle mass */
    struct {
        unsigned int Evaluated :1;
        unsigned int DensityIterationDone :1;
        unsigned int OnAnotherDomain     :1;
        unsigned int WillExport    :1; /* used in domain */
        unsigned int Type        :4;		/*!< flags particle type.  0=gas, 1=halo, 2=disk, 3=bulge, 4=stars, 5=bndry */
        /* first byte ends */
        signed char TimeBin;
        /* second byte ends */
        unsigned char Generation; /* How many particles it has spawned*/
#ifdef WINDS
        unsigned int IsNewParticle:1; /* whether it is created this step */
#endif
#ifdef BLACK_HOLES
        unsigned int Swallowed : 1; /* whether it is being swallowed */
#endif
        unsigned int SufferFromCoupling:1; /* whether it suffers from particle-coupling (nearest neighbour << gravity smoothing)*/
    };

    unsigned int PI; /* particle property index; used by BH. points to the BH property in BhP array.*/
    MyIDType ID;

    MyFloat Vel[3];   /* particle velocity at its current time */
    MyFloat GravAccel[3];  /* particle acceleration due to short-range gravity */

    MyFloat GravPM[3];		/* particle acceleration due to long-range PM gravity force */
    MyFloat OldAcc;			/* magnitude of old gravitational force. Used in relative opening
                              criterion, only used by gravtree cross time steps */

    MyFloat Potential;		/* gravitational potential. This is the total potential after gravtree is called. */
    MyFloat PM_Potential;  /* Only used by PM. useless after pm */

    MyFloat StarFormationTime;		/*!< formation time of star particle: needed to tell when wind is active. */

#ifdef METALS
    MyFloat Metallicity;		/*!< metallicity of gas or star particle */
#endif				/* closes METALS */

    MyFloat Hsml;

#ifdef BLACK_HOLES
    /* SwallowID is not reset in blackhole.c thus cannot be in a union */
    MyIDType SwallowID; /* who will swallow this particle, used only in blackhole.c */
#endif

    /* The peano key is a hash of the position used in the domain decomposition.
     * It is slow to generate so we store it here.*/
    peano_t Key; /* only by domain.c and forcetre.c */

    union {
        /* the following variables are transients.
         * FIXME: move them into the corresponding modules! Is it possible? */

        MyFloat NumNgb; /* Number of neighbours; only used in density.c */

        int RegionInd; /* which region the particle belongs to; only by petapm.c */

        struct {
            /* used by fof.c which calls domain_exchange that doesn't uses peano_t */
            int64_t GrNr; 
            int origintask;
            int targettask;
        };
    };

}
*P;				/*!< holds particle data on local processor */

struct particle_data_ext {
    int ReverseLink; /* used at GC for reverse link to P */
    MyIDType ID; /* for data consistency check, same as particle ID */
};
struct bh_particle_data {
    struct particle_data_ext base;

    int CountProgs;

    MyFloat Mass;
    MyFloat Mdot;
    MyFloat FeedbackWeightSum;
    MyFloat Density;
    MyFloat Entropy;
    MyFloat Pressure;
    MyFloat SurroundingGasVel[3];

    MyFloat accreted_Mass;
    MyFloat accreted_BHMass;
    MyFloat accreted_momentum[3];

    double  MinPotPos[3];
    MyFloat MinPotVel[3];
    MyFloat MinPot;

    short int TimeBinLimit;
} * BhP;


/* the following structure holds data that is stored for each SPH particle in addition to the collisionless
 * variables.
 */
extern struct sph_particle_data
{
    struct particle_data_ext base;

#ifdef DENSITY_INDEPENDENT_SPH
    MyFloat EgyWtDensity;           /*!< 'effective' rho to use in hydro equations */
    MyFloat DhsmlEgyDensityFactor;  /*!< correction factor for density-independent entropy formulation */
#define EOMDensity EgyWtDensity
#else
#define EOMDensity Density
#endif

    MyFloat Entropy;		/*!< current value of entropy (actually entropic function) of particle */
    MyFloat MaxSignalVel;           /*!< maximum signal velocity */
    MyFloat       Density;		/*!< current baryonic mass density of particle */
    MyFloat       DtEntropy;		/*!< rate of change of entropy */
    MyFloat       HydroAccel[3];	/*!< acceleration due to hydrodynamical force */
    MyFloat       DhsmlDensityFactor;	/*!< correction factor needed in entropy formulation of SPH */
    MyFloat       DivVel;		/*!< local velocity divergence */
    MyFloat       CurlVel;     	        /*!< local velocity curl */
    MyFloat       Rot[3];		/*!< local velocity curl */
    MyFloat Ne;  /*!< electron fraction, expressed as local electron number
                   density normalized to the hydrogen number density. Gives
                   indirectly ionization state and mean molecular weight. */

#ifdef BLACK_HOLES
    MyFloat       Injected_BH_Energy;
#endif

#ifdef SFR
    MyFloat Sfr;
#endif
#ifdef WINDS
    MyFloat DelayTime;		/*!< SH03: remaining maximum decoupling time of wind particle */
                            /*!< VS08: remaining waiting for wind particle to be eligible to form winds again */
#endif

#ifdef SPH_GRAD_RHO
    MyFloat GradRho[3];
#endif
} *SphP;				/*!< holds SPH particle data on local processor */

#define SPHP(i) SphP[P[i].PI]
#define BHP(i) BhP[P[i].PI]

#define MPI_UINT64 MPI_UNSIGNED_LONG
#define MPI_INT64 MPI_LONG

#endif
