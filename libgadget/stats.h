#ifndef STATS_H
#define STATS_H

/* Header for writing statistics*/

/* Structs and functions to open file descriptors for logging output*/
struct OutputFD
{
    FILE *FdEnergy;     /*!< file handle for energy.txt log-file. */
    FILE *FdCPU;    /*!< file handle for cpu.txt log-file. */
    FILE *FdSfr;     /*!< file handle for sfr.txt log-file. */
    FILE *FdBlackHoles;  /*!< file handle for blackholes.txt log-file. */
    FILE *FdBlackholeDetails;  /*!< file handle for BlackholeDetails binary file. */
    FILE *FdHelium; /* < file handle for the Helium reionization log file helium.txt */
};

void set_stats_params(ParameterSet * ps);

void open_outputfiles(int RestartSnapNum, struct OutputFD * fds, const char * OutputDir, int BlackHoleOn, int StarformationOn);
void close_outputfiles(struct OutputFD *fds);

/* Write out a CPU log file*/
void write_cpu_log(int NumCurrentTiStep, const double atime, FILE * FdCPU, double ElapsedTime);

/* Write out overall statistics of the energy of the simulation */
void energy_statistics(FILE * FdEnergy, const double Time,  struct part_manager_type * PartManager);
#endif
