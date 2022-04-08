#ifndef RUN_H
#define RUN_H
#include "types.h"
#include "petaio.h"

/* Initialise various structures, read snapshot header, but do not read snapshot data.*/
inttime_t begrun(const int RestartFlag, int RestartSnapNum, struct header_data * header);
/* Run the simulation main loop*/
void run(const int RestartSnapNum, const inttime_t Ti_init, const struct header_data * header);
/* Perform some gravity force accuracy tests and exit*/
void runtests(const int RestartSnapNum, const inttime_t Ti_init, const struct header_data * header);
/* Compute a FOF table and exit*/
void runfof(const int RestartSnapNum, const inttime_t Ti_init, const struct header_data * header);
/* Compute a power spectrum and exit*/
void runpower(const struct header_data * header);

void run_gravity_test(int RestartSnapNum, Cosmology * CP, const double Asmth, const int Nmesh, const int FastParticleType, const inttime_t Ti_Current, const char * OutputDir, const struct header_data * header);

/* Sets up the global_data_all_processes*/
void set_all_global_params(ParameterSet * ps);
#endif
