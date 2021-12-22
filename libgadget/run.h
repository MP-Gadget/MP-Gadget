#ifndef RUN_H
#define RUN_H

/* Initialise various structures, read snapshot header, but do not read snapshot data.*/
int begrun(int RestartFlag, int RestartSnapNum);
/* Run the simulation main loop*/
void run(int RestartSnapNum);
/* Perform some gravity force accuracy tests and exit*/
void runtests(int RestartSnapNum);
/* Compute a FOF table and exit*/
void runfof(int RestartSnapNum);
/* Compute a power spectrum and exit*/
void runpower(int RestartSnapNum);

/* Sets up the global_data_all_processes*/
void set_all_global_params(ParameterSet * ps);
#endif
