#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>

#include <omp.h>
#include "mymalloc.h"
#include "string.h"
#include "memory.h"
#include "system.h"
#include "endrun.h"

/* The main allocator is used to store large objects, e.g. tree, toptree */
Allocator A_MAIN[1];

/* The temp allocator is used to store objects that lives on the stack;
 * replacing alloca and similar cases to avoid stack induced memory fragmentation
 * */
Allocator A_TEMP[1];

#ifdef VALGRIND
#define allocator_init allocator_malloc_init
#endif

void
tamalloc_init(void)
{
    int Nt = omp_get_max_threads();
    int NTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);

    /* Reserve 4MB, 512 bytes per thread, 128 bytes per task and 128 bytes per thread per task (for export) for TEMP storage.*/
    size_t n = 4096 * 1024 + 128 * NTask + 128 * Nt * NTask + 512 * Nt;

    message(0, "Reserving %td bytes per rank for TEMP memory allocator. \n", n);

    if (MPIU_Any(ALLOC_ENOMEMORY == allocator_init(A_TEMP, "TEMP", n, 1, NULL), MPI_COMM_WORLD)) {
        endrun(0, "Insufficient memory for the TEMP allocator on at least one nodes."
                  "Requestion %td bytes. Try reducing MaxMemSizePerNode. Also check the node health status.\n", n);

    }
}

void
mymalloc_init(double MaxMemSizePerNode)
{
    /* Warning: this uses ta_malloc*/
    size_t Nhost = cluster_get_num_hosts();

    MPI_Comm comm = MPI_COMM_WORLD;

    int NTask;

    MPI_Comm_size(comm, &NTask);

    double nodespercpu = (1.0 * Nhost) / (1.0 * NTask);
    size_t n = 1.0 * MaxMemSizePerNode * nodespercpu * 1024. * 1024.;
    message(0, "Nhost = %ld\n", Nhost);
    message(0, "Reserving %td bytes per rank for MAIN memory allocator. \n", n);
    if(n < 1)
        endrun(2, "Mem too small! MB/node=%g, nodespercpu = %g NTask = %d\n", MaxMemSizePerNode, nodespercpu, NTask);


    if (MPIU_Any(ALLOC_ENOMEMORY == allocator_init(A_MAIN, "MAIN", n, 1, NULL), MPI_COMM_WORLD)) {
        endrun(0, "Insufficient memory for the MAIN allocator on at least one nodes."
                  "Requestion %td bytes. Try reducing MaxMemSizePerNode. Also check the node health status.\n", n);
    }
}

static size_t highest_memory_usage = 0;

void report_detailed_memory_usage(const char *label, const char * fmt, ...)
{
    if(allocator_get_used_size(A_MAIN, ALLOC_DIR_BOTH) < highest_memory_usage) {
        return;
    }

    MPI_Comm comm = MPI_COMM_WORLD;

    int NTask;
    int ThisTask;
    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &ThisTask);


    if (ThisTask != 0) {
        return;
    }

    highest_memory_usage = allocator_get_used_size(A_MAIN, ALLOC_DIR_BOTH);

    va_list va;
    va_start(va, fmt);
    char * buf = fastpm_strdup_vprintf(fmt, va);
    va_end(va);

    message(1, "Peak Memory usage induced by %s\n", buf);
    myfree(buf);
    allocator_print(A_MAIN);
}
