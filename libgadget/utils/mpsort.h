#ifndef _UTILS_MPSORT_H
#define _UTILS_MPSORT_H

/* MPI support */
#define MPSORT_DISABLE_GATHER_SORT (1 << 3)
#define MPSORT_REQUIRE_GATHER_SORT (1 << 4)

void mpsort_mpi_set_options(int options);
int mpsort_mpi_has_options(int options);
void mpsort_mpi_unset_options(int options);

void mpsort_mpi(void * base, size_t nmemb, size_t elsize,
        void (*radix)(const void * ptr, void * radix, void * arg), 
        size_t rsize, 
        void * arg, MPI_Comm comm);
void mpsort_mpi_newarray(void * base, size_t nmemb, 
        void * out, size_t outnmemb,
        size_t size,
        void (*radix)(const void * ptr, void * radix, void * arg), 
        size_t rsize, 
        void * arg, MPI_Comm comm);

void mpsort_mpi_report_last_run();

#endif
