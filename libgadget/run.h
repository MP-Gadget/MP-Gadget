#ifndef RUN_H
#define RUN_H
#include "types.h"

/* Initialise various structures, read snapshot header, but do not read snapshot data.*/
inttime_t begrun(const int RestartFlag, int RestartSnapNum);
/* Run the simulation main loop*/
void run(const int RestartSnapNum, const inttime_t Ti_init);
/* Perform some gravity force accuracy tests and exit*/
void runtests(const int RestartSnapNum, const inttime_t Ti_init);
/* Compute a FOF table and exit*/
void runfof(const int RestartSnapNum, const inttime_t Ti_init);
/* Compute a power spectrum and exit*/
void runpower(void);

/* Sets up the global_data_all_processes*/
void set_all_global_params(ParameterSet * ps);
#endif
