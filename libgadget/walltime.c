#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include "walltime.h"

#include "utils/mymalloc.h"
#include "utils/openmpsort.h"

static struct ClockTable * CT = NULL;

static double WallTimeClock;
static double LastReportTime;

static void walltime_clock_insert(const char * name);
static void walltime_summary_clocks(struct Clock * C, int N, int root, MPI_Comm comm);
static void walltime_update_parents(void);
static double seconds(void);

void walltime_init(struct ClockTable * ct) {
    CT = ct;
    CT->Nmax = 512;
    CT->N = 0;
    CT->ElapsedTime = 0;
    walltime_reset();
    walltime_clock_insert("/");
    LastReportTime = seconds();
}

static void walltime_summary_clocks(struct Clock * C, int N, int root, MPI_Comm comm) {
    double * t = ta_malloc("clocks", double, 4 * N);
    double * min = t + N;
    double * max = t + 2 * N;
    double * sum = t + 3 * N;
    int i;
    for(i = 0; i < CT->N; i ++) {
        t[i] = C[i].time;
    }
    MPI_Reduce(t, min, N, MPI_DOUBLE, MPI_MIN, root, comm);
    MPI_Reduce(t, max, N, MPI_DOUBLE, MPI_MAX, root, comm);
    MPI_Reduce(t, sum, N, MPI_DOUBLE, MPI_SUM, root, comm);

    int NTask;
    MPI_Comm_size(comm, &NTask);
    /* min, max and mean are good only on process 0 */
    for(i = 0; i < CT->N; i ++) {
        C[i].min = min[i];
        C[i].max = max[i];
        C[i].mean = sum[i] / NTask;
    }
    ta_free(t);
}

/* put min max mean of MPI ranks to rank 0*/
/* AC will have the total timing, C will have the current step information */
void walltime_summary(int root, MPI_Comm comm) {
    walltime_update_parents();
    int i;
    /* add to the cumulative time */
    for(i = 0; i < CT->N; i ++) {
        CT->AC[i].time += CT->C[i].time;
    }
    walltime_summary_clocks(CT->C, CT->N, root, comm);
    walltime_summary_clocks(CT->AC, CT->N, root, comm);

    /* clear .time for next step */
    for(i = 0; i < CT->N; i ++) {
        CT->C[i].time = 0;
    }
    MPI_Barrier(comm);
    /* wo do this here because all processes are sync after summary_clocks*/
    double step_all = seconds() - LastReportTime;
    LastReportTime = seconds();
    CT->ElapsedTime += step_all;
    CT->StepTime = step_all;
}

static int clockcmp(const void * c1, const void * c2) {
    const struct Clock * p1 = (const struct Clock *) c1;
    const struct Clock * p2 = (const struct Clock *) c2;
    return strcmp(p1->name, p2->name);
}

static void walltime_clock_insert(const char * name) {
    if(strlen(name) > 1) {
        char tmp[80] = {0};
        strncpy(tmp, name, 79);
        char * p;
        walltime_clock("/");
        for(p = tmp + 1; *p; p ++) {
            if (*p == '/') {
                *p = 0;
                walltime_clock(tmp);
                *p = '/';
            }
        }
    }
    if(CT->N == CT->Nmax) {
        /* too many counters */
        abort();
    }
    const int nmsz = sizeof(CT->C[CT->N].name);
    strncpy(CT->C[CT->N].name, name, nmsz);
    CT->C[CT->N].name[nmsz-1] = '\0';
    strncpy(CT->AC[CT->N].name, CT->C[CT->N].name, nmsz);
    CT->AC[CT->N].name[nmsz-1] = '\0';
    CT->N ++;
    qsort_openmp(CT->C, CT->N, sizeof(struct Clock), clockcmp);
    qsort_openmp(CT->AC, CT->N, sizeof(struct Clock), clockcmp);
}

int walltime_clock(const char * name) {
    struct Clock dummy;
    strncpy(dummy.name, name, sizeof(dummy.name));
    dummy.name[sizeof(dummy.name)-1]='\0';

    struct Clock * rt = (struct Clock *) bsearch(&dummy, CT->C, CT->N, sizeof(struct Clock), clockcmp);
    if(rt == NULL) {
        walltime_clock_insert(name);
        rt = (struct Clock *) bsearch(&dummy, CT->C, CT->N, sizeof(struct Clock), clockcmp);
    }
    return rt - CT->C;
};

char walltime_get_symbol(const char * name) {
    int id = walltime_clock(name);
    return CT->C[id].symbol;
}

double walltime_get(const char * name, enum clocktype type) {
    int id = walltime_clock(name);
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
double walltime_get_time(const char * name) {
    int id = walltime_clock(name);
    return CT->C[id].time;
}

static void walltime_update_parents() {
    /* returns the sum of every clock with the same prefix */
    int i = 0;
    for(i = 0; i < CT->N; i ++) {
        CT->Nchildren[i] = 0;
        int j;
        char * prefix = CT->C[i].name;
        int l = strlen(prefix);
        double t = 0;
        for(j = i + 1; j < CT->N; j++) {
            if(0 == strncmp(prefix, CT->C[j].name, l)) {
                t += CT->C[j].time;
                CT->Nchildren[i] ++;
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

double walltime_add_internal(const char * name, const double dt) {
    int id = walltime_clock(name);
    CT->C[id].time += dt;
    return dt;
}
double walltime_measure_internal(const char * name) {
    double t = seconds();
    double dt = t - WallTimeClock;
    WallTimeClock = seconds();
    if(name[0] != '.') {
        int id = walltime_clock(name);
        CT->C[id].time += dt;
    }
    return dt;
}
double walltime_measure_full(const char * name, const char * file, const int line) {
    char fullname[128] = {0};
    const char * basename = file + strlen(file);
    while(basename >= file && *basename != '/') basename --;
    basename ++;
    snprintf(fullname, 128, "%s@%s:%04d", name, basename, line);
    return walltime_measure_internal(fullname);
}
double walltime_add_full(const char * name, const double dt, const char * file, const int line) {
    char fullname[128] = {0};
    const char * basename = file + strlen(file);
    while(basename >= file && *basename != '/') basename --;
    basename ++;
    snprintf(fullname, 128, "%s@%s:%04d", name, basename, line);
    return walltime_add_internal(fullname, dt);

}

/* returns the number of cpu-ticks in seconds that
 * have elapsed. (or the wall-clock time)
 */
static double seconds(void)
{
  return MPI_Wtime();
}
void walltime_report(FILE * fp, int root, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    if(rank != root) return;
    int i;
    for(i = 0; i < CT->N; i ++) {
        char * name = CT->C[i].name;
        int level = 0;
        char * p = name;
        while(*p) {
            if(*p == '/') {
                level ++;
                name = p + 1;
            }
            p++;
        }
        /* if there is just one child, don't print it*/
        if(CT->Nchildren[i] == 1) continue;
        fprintf(fp, "%*s%-26s  %10.2f %4.1f%%  %10.2f %4.1f%%  %10.2f %10.2f\n",
                level, "",  /* indents */
                name,   /* just the last seg of name*/
                CT->AC[i].mean,
                CT->AC[i].mean / CT->ElapsedTime * 100.,
                CT->C[i].mean,
                CT->C[i].mean / CT->StepTime * 100.,
                CT->C[i].min,
                CT->C[i].max
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
