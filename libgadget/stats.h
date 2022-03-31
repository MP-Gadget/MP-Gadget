#ifndef STATS_H
#define STATS_H

/* Header for writing statistics*/

/* Write out overall statistics of the energy of the simulation */
void energy_statistics(FILE * FdEnergy, const double Time,  struct part_manager_type * PartManager);

/* Write out a CPU log file*/
void write_cpu_log(int NumCurrentTiStep, const double atime, FILE * FdCPU, double ElapsedTime);

#endif
