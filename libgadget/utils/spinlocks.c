#include "spinlocks.h"
#include "mymalloc.h"

#ifndef NO_OPENMP_SPINLOCK
#include <pthread.h>
#else
#include <omp.h>
#endif

/* Temporary array for spinlocks*/
struct SpinLocks
{
#ifndef NO_OPENMP_SPINLOCK
    pthread_spinlock_t * SpinLocks;
#else
    omp_lock_t * SpinLocks;
#endif
    int NumSpinLock;
};

static struct SpinLocks spin;

void lock_spinlock(int i, struct SpinLocks * spin) {
#ifndef NO_OPENMP_SPINLOCK
    pthread_spin_lock(&spin->SpinLocks[i]);
#else
    omp_set_lock(&spin->SpinLocks[i]);
#endif
}
void unlock_spinlock(int i,  struct SpinLocks * spin) {
#ifndef NO_OPENMP_SPINLOCK
    pthread_spin_unlock(&spin->SpinLocks[i]);
#else
    omp_unset_lock(&spin->SpinLocks[i]);
#endif
}

int try_lock_spinlock(int i,  struct SpinLocks * spin)
{
#ifndef NO_OPENMP_SPINLOCK
    return pthread_spin_trylock(&spin->SpinLocks[i]);
#else
    /* omp returns true if successful, ie, lock taken.
     * pthread_spin_lock returns 0 on success*/
    return !omp_test_lock(&spin->SpinLocks[i]);
#endif
}

struct SpinLocks * init_spinlocks(int NumLock)
{
    int i;
    /* Initialize the spinlocks*/
#ifndef NO_OPENMP_SPINLOCK
    spin.SpinLocks = (pthread_spinlock_t *) mymalloc("SpinLocks", NumLock * sizeof(pthread_spinlock_t));
    #pragma omp parallel for
#else
    spin.SpinLocks = (omp_lock_t*)mymalloc("SpinLocks", NumLock * sizeof(omp_lock_t));
#endif
    for(i = 0; i < NumLock; i ++) {
#ifndef NO_OPENMP_SPINLOCK
        pthread_spin_init(&spin.SpinLocks[i], PTHREAD_PROCESS_PRIVATE);
#else
        omp_init_lock(&spin.SpinLocks[i]);
#endif
    }
    spin.NumSpinLock = NumLock;
    return &spin;
}

void free_spinlocks(struct SpinLocks * spin)
{
    int i;
    for(i = 0; i < spin->NumSpinLock; i ++) {
#ifndef NO_OPENMP_SPINLOCK
        pthread_spin_destroy(&(spin->SpinLocks[i]));
#else
        omp_destroy_lock(&(spin->SpinLocks[i]));
#endif
    }
    myfree((void *) spin->SpinLocks);
}

