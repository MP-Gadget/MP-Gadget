/*! \file allvars.h
 *  \brief declares the All structure.
 *
 */

#ifndef ALLVARS_H
#define ALLVARS_H

#include <mpi.h>
#include <omp.h>

#include "cosmology.h"
#include "gravity.h"
#include "physconst.h"

/*! This structure contains data which is the SAME for all tasks (mostly code parameters read from the
 * parameter file).  Holding this data in a structure is convenient for writing/reading the restart file, and
 * it allows the introduction of new global variables in a simple way. The only thing to do is to introduce
 * them into this structure.
 */
extern struct global_data_all_processes
{
    /* The following variables are set by petaio_read_header */
    int64_t NTotalInit[6]; /* The initial number of total particles in the IC. */
    double TimeInit;		/* time of simulation start: if restarting from a snapshot this holds snapshot time.*/
    double TimeIC;       /* Time when the simulation ICs were generated*/
    double BoxSize;   /* Boxsize in case periodic boundary conditions are used */
    double MassTable[6]; /* Initial mass of particles */
    /* end of read_header parameters */

    double PartAllocFactor;	/*!< in order to maintain work-load balance, the particle load will usually
                              NOT be balanced.  Each processor allocates memory for PartAllocFactor times
                              the average number of particles to allow for that */

    double SlotsIncreaseFactor; /* !< What percentage to increase the slot allocation by when requested*/
    int OutputDebugFields;      /* Flag whether to include a lot of debug output in snapshots*/

    double RandomParticleOffset; /* If > 0, a random shift of max RandomParticleOffset * BoxSize is applied to every particle
                                  * every time a full domain decomposition is done. The box is periodic and the offset
                                  * is subtracted on output, so this only affects the internal gravity solver.
                                  * The purpose of this is to avoid correlated errors in the tree code, which occur when
                                  * the tree opening conditions are similar in every timestep and accumulate over a
                                  * long period of time. Upstream Arepo says this substantially improves momentum conservation,
                                  * and it has the side-effect of guarding against periodicity bugs.
                                  */
    /* some SPH parameters */
    double MinEgySpec; /* Minimum internal energy for timestepping, converted from MinGasTemp*/

    /* system of units  */
    struct UnitSystem units;

    /* Cosmology */
    Cosmology CP;

    /* Code options */
    int CoolingOn;  /* if cooling is enabled */
    int HydroOn;  /*  if hydro force is enabled */
    int DensityOn;  /*  if SPH density computation is enabled */
    int TreeGravOn;     /* tree gravity force is enabled*/

    int BlackHoleOn;  /* if black holes are enabled */
    int StarformationOn;  /* if star formation is enabled */
    int MetalReturnOn; /* If late return of metals from AGB stars is enabled*/
    int LightconeOn;    /* Enable the light cone module,
                           which writes a list of particles to a file as they cross a light cone*/

    int WriteBlackHoleDetails; /* write BH details every time step*/

    int MaxDomainTimeBinDepth; /* We should redo domain decompositions every timestep, after the timestep hierarchy gets deeper than this.
                                  Essentially forces a domain decompositon every 2^MaxDomainTimeBinDepth timesteps.*/
    int FastParticleType; /*!< flags a particle species to exclude timestep calculations.*/
    /* parameters determining output frequency */
    double PairwiseActiveFraction; /* Fraction of particles active for which we do a pairwise computation instead of a tree*/

    /* parameters determining output frequency */
    double AutoSnapshotTime;    /*!< cpu-time between regularly generated snapshots. */
    double TimeBetweenSeedingSearch; /*Factor to multiply TimeInit by to find the next seeding check.*/

    double TimeMax;			/*!< marks the point of time until the simulation is to be evolved */

    int Nmesh;

    /* variables that keep track of cumulative CPU consumption */

    double TimeLimitCPU;

    /*! The scale of the short-range/long-range force split in units of FFT-mesh cells */
    double Asmth;
    enum ShortRangeForceWindowType ShortRangeForceWindowType;	/*!< method of the feedback*/

    /* some filenames */
    char OutputDir[100],
         FOFFileBase[100],
         EnergyFile[100],
         CpuFile[100];

    /*Should we store the energy to EnergyFile on PM timesteps.*/
    int OutputEnergyDebug;

    int SnapshotWithFOF; /*Flag that doing FOF for snapshot outputs is on*/

    int RandomSeed; /*Initial seed for the random number table*/
}
All;

#endif
