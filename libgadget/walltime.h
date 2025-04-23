#ifndef GADGET_WALLTIME_H
#define GADGET_WALLTIME_H

#include <stdio.h>
#include <mpi.h>

int walltime_clock(const char * name);
void walltime_reset(void);
#define WALLTIME_IGNORE "."
#define LINENO(a, b) a ":" # b
#define walltime_measure(name) walltime_measure_full(name, __FILE__ , __LINE__)
#define walltime_add(name, dt) walltime_add_full(name, dt,  __FILE__, __LINE__)
double walltime_measure_internal(const char * name);
double walltime_add_internal(const char * name, const double dt);
double walltime_measure_full(const char * name, const char * file, const int line);
double walltime_add_full(const char * name, const double dt, const char * file, const int line);

enum clocktype {
    CLOCK_STEP_MEAN ,
    CLOCK_STEP_MAX ,
    CLOCK_STEP_MIN ,
    CLOCK_ACCU_MEAN ,
    CLOCK_ACCU_MAX ,
    CLOCK_ACCU_MIN ,
};


char walltime_get_symbol(const char * name);

double walltime_get_time(const char * name);
double walltime_get(const char * name, const enum clocktype type);
#define walltime_step_min(id) walltime_get(id, CLOCK_STEP_MIN)
#define walltime_step_max(id) walltime_get(id, CLOCK_STEP_MAX)
#define walltime_step_mean(id) walltime_get(id, CLOCK_STEP_MEAN)
#define walltime_accu_min(id) walltime_get(id, CLOCK_ACCU_MIN)
#define walltime_accu_max(id) walltime_get(id, CLOCK_ACCU_MAX)
#define walltime_accu_mean(id) walltime_get(id, CLOCK_ACCU_MEAN)

void walltime_summary(const int root, MPI_Comm comm);
void walltime_report(FILE * fd, const int root, MPI_Comm comm);

struct Clock {
    char name[128];
    double time;
    double max;
    double min;
    double mean;
    char symbol;
};

struct ClockTable {
    int Nmax;
    int N;
    struct Clock C[512];
    struct Clock AC[512];
    int Nchildren[512];
    double ElapsedTime;
    double StepTime;
    /*These are used for estimating when to timeout*/
    double IOTime;
    double PMStepTime;
};
void walltime_init(struct ClockTable * table);
#endif
