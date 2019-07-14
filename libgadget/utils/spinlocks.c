#include "spinlocks.h"
#include "mymalloc.h"

#ifndef NO_OPENMP_SPINLOCK
#include <pthread.h>
#endif

/* Temporary array for spinlocks*/
struct SpinLocks
{
#ifndef NO_OPENMP_SPINLOCK
    pthread_spinlock_t * SpinLocks;
#endif
    int NumSpinLock;
};

static struct SpinLocks spin;

void lock_spinlock(int i, struct SpinLocks * spin) {
#ifndef NO_OPENMP_SPINLOCK
    pthread_spin_lock(&spin->SpinLocks[i]);
#endif
}
void unlock_spinlock(int i,  struct SpinLocks * spin) {
#ifndef NO_OPENMP_SPINLOCK
    pthread_spin_unlock(&spin->SpinLocks[i]);
#endif
}

int try_lock_spinlock(int i,  struct SpinLocks * spin)
{
#ifndef NO_OPENMP_SPINLOCK
    return pthread_spin_trylock(&spin->SpinLocks[i]);
#else
    return 0;
#endif
}

struct SpinLocks * init_spinlocks(int NumLock)
{
#ifndef NO_OPENMP_SPINLOCK
    int i;
    /* Initialize the spinlocks*/
    spin.SpinLocks = mymalloc("SpinLocks", NumLock * sizeof(pthread_spinlock_t));
    #pragma omp parallel for
    for(i = 0; i < NumLock; i ++) {
        pthread_spin_init(&spin.SpinLocks[i], PTHREAD_PROCESS_PRIVATE);
    }
#endif
    spin.NumSpinLock = NumLock;
    return &spin;
}

void free_spinlocks(struct SpinLocks * spin)
{
#ifndef NO_OPENMP_SPINLOCK
    int i;
    for(i = 0; i < spin->NumSpinLock; i ++) {
        pthread_spin_destroy(&(spin->SpinLocks[i]));
    }
    myfree((void *) spin->SpinLocks);
#endif
}

