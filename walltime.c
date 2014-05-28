#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include "walltime.h"

static struct ClockTable * CT = NULL;

static double WallTimeClock;

void walltime_init(struct ClockTable * ct) {
    CT = ct;
    CT->Nmax = 128;
    CT->N = 0;
    walltime_reset();
}

static void walltime_summary_clocks(struct Clock * C, int N) {
    double t[N];
    double min[N];
    double max[N];
    double sum[N];
    int i;
    for(i = 0; i < CT->N; i ++) {
        t[i] = C[i].time;
    }
    MPI_Reduce(t, min, N, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(t, max, N, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(t, sum, N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    int NTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    /* min, max and mean are good only on process 0 */
    for(i = 0; i < CT->N; i ++) {
        C[i].min = min[i];
        C[i].max = max[i];
        C[i].mean = sum[i] / NTask;
    }
}

/* put min max mean of MPI ranks to rank 0*/
/* AC will have the total timing, C will have the current step information */
void walltime_summary() {
    walltime_update_parents();
    int i;
    int N = 0;
    int rank = 0;
    /* add to the cumulative time */
    for(i = 0; i < CT->N; i ++) {
        CT->AC[i].time += CT->C[i].time;
    }
    walltime_summary_clocks(CT->C, CT->N);
    walltime_summary_clocks(CT->AC, CT->N);

    /* clear .time for next step */
    for(i = 0; i < CT->N; i ++) {
        CT->C[i].time = 0;
    }
}
static int clockcmp(struct Clock * p1, struct Clock * p2) {
    return strcmp(p1->name, p2->name);
}

static void walltime_clock_insert(char * name) {
    if(strlen(name) > 1) {
        char tmp[80];
        strcpy(tmp, name);
        char * p;
        int parent = walltime_clock("/");
        for(p = tmp + 1; *p; p ++) {
            if (*p == '/') {
                *p = 0;
                int parent = walltime_clock(tmp);
                *p = '/';
            }
        }
    }
    if(CT->N == CT->Nmax) {
        /* too many counters */
        abort();
    }
    strcpy(CT->C[CT->N].name, name);
    strcpy(CT->AC[CT->N].name, CT->C[CT->N].name);
    CT->N ++;
    qsort(CT->C, CT->N, sizeof(struct Clock), clockcmp);
    qsort(CT->AC, CT->N, sizeof(struct Clock), clockcmp);
}

int walltime_clock(char * name) {
    struct Clock dummy;
    strcpy(dummy.name, name);
    struct Clock * rt = bsearch(&dummy, CT->C, CT->N, sizeof(struct Clock), clockcmp);
    if(rt == NULL) {
        walltime_clock_insert(name);
        rt = bsearch(&dummy, CT->C, CT->N, sizeof(struct Clock), clockcmp);
    }
    return rt - CT->C;
};

char * walltime_get_name(int id) {
    return CT->C[id].name;
}

char walltime_get_symbol(int id) {
    return CT->C[id].symbol;
}

static double seconds();

double walltime_get(int id, enum clocktype type) {
    /* only make sense on root */
    switch(type) {
        case CLOCK_STEP_MEAN:
            return CT->C[id].mean;
        case CLOCK_STEP_MIN:
            return CT->C[id].min;
        case CLOCK_STEP_MAX:
            return CT->C[id].max;
        case CLOCK_ACCU_MEAN:
            return CT->AC[id].mean;
        case CLOCK_ACCU_MIN:
            return CT->AC[id].min;
        case CLOCK_ACCU_MAX:
            return CT->AC[id].max;
    }
    return 0;
}
double walltime_get_time(int id) {
    return CT->C[id].time;
}

static void walltime_update_parents() {
    /* returns the sum of every clock with the same prefix */
    int i = 0;
    for(i = 0; i < CT->N; i ++) {
        int j;
        char * prefix = CT->C[i].name;
        int l = strlen(prefix);
        double t = 0;
        for(j = i + 1; j < CT->N; j++) {
            if(0 == strncmp(prefix, CT->C[j].name, l)) {
                t += CT->C[j].time;
            } else {
                break;
            }
        }
        /* update only if there are children */
        if (t > 0) CT->C[i].time = t;
    }
}

void walltime_reset() {
    WallTimeClock = seconds();
}

double walltime_add(int id, double dt) {
    CT->C[id].time += dt;
    return dt;
}
double walltime_measure(int id) {
    double t = seconds();
    double dt = t - WallTimeClock;
    if(id >= 0) CT->C[id].time += dt;
    WallTimeClock = seconds();
    return dt;
}

/* returns the number of cpu-ticks in seconds that
 * have elapsed. (or the wall-clock time)
 */
static double seconds(void)
{
#ifdef WALLCLOCK
  return MPI_Wtime();
#else
  return ((double) clock()) / CLOCKS_PER_SEC;
#endif

  /* note: on AIX and presumably many other 32bit systems, 
   * clock() has only a resolution of 10ms=0.01sec 
   */
}
void walltime_report(FILE * fp) {
    int i; 
    double step_all = walltime_step_max(WALL_ALL);
    double accu_all = walltime_accu_max(WALL_ALL);
    for(i = 0; i < CT->N; i ++) {
        fprintf(fp, "%-26s  %10.2f %4.1f%%  %10.2f %4.1f%%  %10.2f %10.2f\n",
                walltime_get_name(i),
                walltime_accu_mean(i),
                walltime_accu_mean(i) / accu_all * 100.,
                walltime_step_mean(i),
                walltime_step_mean(i) / step_all * 100.,
                walltime_step_max(i),
                walltime_step_min(i)
                );
    }
}
#if 0
#define HELLO atom(&atomtable, "Hello")
#define WORLD atom(&atomtable, "WORLD")
int main() {
    walltime_init();
    printf("%d\n", HELLO, atom_name(&atomtable, HELLO));
    printf("%d\n", HELLO, atom_name(&atomtable, HELLO));
    printf("%d\n", WORLD, atom_name(&atomtable, WORLD));
    printf("%d\n", HELLO, atom_name(&atomtable, HELLO));
}
#endif
