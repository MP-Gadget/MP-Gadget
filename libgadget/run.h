#ifndef RUN_H
#define RUN_H
#include "petaio.h"

/* Initialise various structures, read snapshot header, but do not read snapshot data.*/
struct header_data begrun(int RestartFlag, int RestartSnapNum);
/* Run the simulation main loop*/
void run(int RestartSnapNum, struct header_data * header);
/* Perform some gravity force accuracy tests and exit*/
void runtests(int RestartSnapNum, struct header_data * header);
/* Compute a FOF table and exit*/
void runfof(int RestartSnapNum, struct header_data * header);
/* Compute a power spectrum and exit*/
void runpower(int RestartSnapNum, struct header_data * header);

/* Sets up the global_data_all_processes*/
void set_all_global_params(ParameterSet * ps);
#endif
