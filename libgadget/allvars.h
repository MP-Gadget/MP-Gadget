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

enum ShortRangeForceWindowType {
    SHORTRANGE_FORCE_WINDOW_TYPE_EXACT = 0,
    SHORTRANGE_FORCE_WINDOW_TYPE_ERFC = 1,
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

/* To be removed at some point*/
extern int ThisTask;		/*!< the number of the local processor  */

/* variables for input/output , usually only used on process 0 */
extern FILE *FdEnergy,			/*!< file handle for energy.txt log-file. */
       *FdCPU;			/*!< file handle for cpu.txt log-file. */

extern FILE *FdSfr;		/*!< file handle for sfr.txt log-file. */

extern FILE *FdBlackHoles;	/*!< file handle for blackholes.txt log-file. */

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

    double PartAllocFactor;	/*!< in order to maintain work-load balance, the particle load will usually
                              NOT be balanced.  Each processor allocates memory for PartAllocFactor times
                              the average number of particles to allow for that */

    double SlotsIncreaseFactor; /* !< What percentage to increase the slot allocation by when requested*/
    int OutputPotential;        /*!< Flag whether to include the potential in snapshots*/

    /* some SPH parameters */

    int DesNumNgb;		/*!< Desired number of SPH neighbours */
    /* These are for black hole neighbour finding and so belong in the density module, not the black hole module.*/
    double BlackHoleNgbFactor;	/*!< Factor by which the normal SPH neighbour should be increased/decreased */
    double BlackHoleMaxAccretionRadius;


    double DensityResolutionEta;		/*!< SPH resolution eta. See Price 2011. eq 12*/
    double MaxNumNgbDeviation;	/*!< Maximum allowed deviation neighbour number */
    double ArtBulkViscConst;	/*!< Sets the parameter \f$\alpha\f$ of the artificial viscosity */

    double InitGasTemp;		/*!< may be used to set the temperature in the IC's */
    double MinGasTemp;		/*!< may be used to set a floor for the gas temperature */
    double MinEgySpec;		/*!< the minimum allowed temperature expressed as energy per unit mass */

    /* system of units  */

    double UnitTime_in_s,		/*!< factor to convert internal time unit to seconds/h */
           UnitDensity_in_cgs,		/*!< factor to convert internal length unit to g/cm^3*h^2 */
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
    int CoolingOn;  /* if cooling is enabled */
    int HydroOn;  /*  if hydro force is enabled */
    int DensityOn;  /*  if SPH density computation is enabled */
    int TreeGravOn;     /* tree gravity force is enabled*/
    int DensityIndependentSphOn; /* Enables density independent (Pressure-entropy) SPH */

    int BlackHoleOn;  /* if black holes are enabled */
    int StarformationOn;  /* if star formation is enabled */
    int WindOn; /* if Wind is enabled */

    int MassiveNuLinRespOn; /*!< flags that massive neutrinos using the linear
                                 response code of Ali-Haimoud & Bird 2013.*/
    int HybridNeutrinosOn; /*!< Flags that hybrid neutrinos are enabled */
    double HybridVcrit; /*!< Critical velocity switching between particle
                          and analytic solvers when hybrid neutrinos are on*/
    double HybridNuPartTime; /*!< Redshift at which hybrid neutrinos switch on*/

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
    double BHOpeningAngle;	/*!< Barnes-Hut parameter for opening criterion in tree walk */
    int TreeUseBH;              /*!< If true, use the BH opening angle. Otherwise use acceleration */


    /*! The scale of the short-range/long-range force split in units of FFT-mesh cells */
    double Asmth;
    enum ShortRangeForceWindowType ShortRangeForceWindowType;	/*!< method of the feedback*/

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

    double MeanSeparation[6]; /* mean separation between particles. 0 if the species doesn't exist. */

    /* some filenames */
    char InitCondFile[100],
         OutputDir[100],
         SnapshotFileBase[100],
         FOFFileBase[100],
         EnergyFile[100],
         CpuFile[100];
    char TreeCoolFile[100];
    char MetalCoolFile[100];
    char UVFluctuationFile[100];

    /*Should we store the energy to EnergyFile on PM timesteps.*/
    int OutputEnergyDebug;

    double OutputListTimes[8192];
    int OutputListLength;

    int SnapshotWithFOF; /*Flag that doing FOF for snapshot outputs is on*/

    int RandomSeed; /*Initial seed for the random number table*/
}
All;

#endif
