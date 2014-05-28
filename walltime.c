#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include "walltime.h"

struct ClockTable {
    int Nmax;
    int N;
    struct Clock * C;
} CT = {0};

struct Clock {
    char * name;
    double accum;
    double time;
    double max;
    double min;
    double mean;
    char symbol;
    int parent;
};

static double WallTimeClock;

void walltime_init() {
    CT.C = (struct Clock *) malloc(sizeof(struct Clock) * 4096);
    CT.Nmax = 4096;
    CT.N = 0;
    walltime_reset();
}

/* put min max mean of MPI ranks to rank 0*/
void walltime_summary() {
    double t[CT.N];
    double min[CT.N];
    double max[CT.N];
    double sum[CT.N];
    int i;
    for(i = 0; i < CT.N; i ++) {
        t[i] = CT.C[i].time;
    }
    MPI_Reduce(t, min, CT.N, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(t, max, CT.N, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(t, sum, CT.N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    int N = 0;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &N);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    /* min, max and mean are good only on process 0 */
    for(i = 0; i < CT.N; i ++) {
        CT.C[i].min = min[i];
        CT.C[i].max = max[i];
        CT.C[i].mean = sum[i] / N;
        CT.C[i].accum += CT.C[i].mean;
        CT.C[i].time = 0;
    }
}
static int clockcmp(struct Clock * p1, struct Clock * p2) {
    return strcmp(p1->name, p2->name);
}

int walltime_clock(char * name) {
    if(CT.Nmax == 0) {
        walltime_init();
    }
    struct Clock dummy;
    dummy.name = name;
    struct Clock * rt = bsearch(&dummy, CT.C, CT.N, sizeof(struct Clock), clockcmp);
    if(rt == NULL) {
        if(CT.N == CT.Nmax) {
            CT.C = (struct Clock*) realloc(CT.C, sizeof(struct Clock) * CT.N * 2);
            CT.Nmax = CT.N * 2;
        }
        CT.C[CT.N].name = strdup(name);
        CT.N ++;
        qsort(CT.C, CT.N, sizeof(struct Clock), clockcmp);
        rt = bsearch(&dummy, CT.C, CT.N, sizeof(struct Clock), clockcmp);
    }
    return rt - CT.C;
};

char * walltime_get_name(int id) {
    return CT.C[id].name;
}

char walltime_get_symbol(int id) {
    return CT.C[id].symbol;
}

static double seconds();

double walltime_get_mean(int id) {
    /* returns the sum of every clock with the same prefix */
    char * prefix = CT.C[id].name;
    int i = 0;
    double t = 0;
    for(i = 0; i < CT.N; i ++) {
        if(!strncmp(prefix, CT.C[i].name, strlen(prefix))) {
            t += CT.C[i].mean;
        }
    }
    return t;
}

void walltime_reset() {
    WallTimeClock = seconds();
}

double walltime_add(int id, double dt) {
    CT.C[id].time += dt;
    return dt;
}
double walltime_measure(int id) {
    double t = seconds();
    double dt = t - WallTimeClock;
    if(id >= 0) CT.C[id].time += dt;
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
    double all = walltime_get_mean(WALL_ALL);
    for(i = 0; i < CT.N; i ++) {
        fprintf(fp, "%-18s    %10.2f   %4.1f%%\n",
                walltime_get_name(i),
                walltime_get_mean(i),
                walltime_get_mean(i) / all * 100.
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
