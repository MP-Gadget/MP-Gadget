int walltime_clock(char * name);
void walltime_reset();
double walltime_measure(char * name);
double walltime_add(char * name, double dt);

enum clocktype {
    CLOCK_STEP_MEAN ,
    CLOCK_STEP_MAX ,
    CLOCK_STEP_MIN ,
    CLOCK_ACCU_MEAN ,
    CLOCK_ACCU_MAX ,
    CLOCK_ACCU_MIN ,
};


char walltime_get_symbol(char * name);

double walltime_get_time(char * name);
double walltime_get(char * name, enum clocktype type);
#define walltime_step_min(id) walltime_get(id, CLOCK_STEP_MIN)
#define walltime_step_max(id) walltime_get(id, CLOCK_STEP_MAX)
#define walltime_step_mean(id) walltime_get(id, CLOCK_STEP_MEAN)
#define walltime_accu_min(id) walltime_get(id, CLOCK_ACCU_MIN)
#define walltime_accu_max(id) walltime_get(id, CLOCK_ACCU_MAX)
#define walltime_accu_mean(id) walltime_get(id, CLOCK_ACCU_MEAN)

void walltime_summary(int root, MPI_Comm comm);
void walltime_report(FILE * fd, int root, MPI_Comm comm);

struct Clock {
    char name[40];
    double time;
    double max;
    double min;
    double mean;
    char symbol;
};

struct ClockTable {
    int Nmax;
    int N;
    struct Clock C[128];
    struct Clock AC[128];
    double ElapsedTime;
    double StepTime;
};
void walltime_init(struct ClockTable * table);
