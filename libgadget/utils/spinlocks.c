#include "spinlocks.h"
#include "mymalloc.h"
#include "omp.h"

/* Temporary array for spinlocks*/
struct SpinLocks
{
    omp_lock_t * SpinLocks;
    int NumSpinLock;
};

static struct SpinLocks spin;

void lock_spinlock(int i, struct SpinLocks * spin) {
    omp_set_lock(&spin->SpinLocks[i]);
}
void unlock_spinlock(int i,  struct SpinLocks * spin) {
    omp_unset_lock(&spin->SpinLocks[i]);
}

int try_lock_spinlock(int i,  struct SpinLocks * spin)
{
    /* omp_test_lock returns true if it got the lock, false otherwise.
     * pthread_spin_trylock returns 0 (false) if it got the lock, EBUSY (true) otherwise*/
    return !omp_test_lock(&spin->SpinLocks[i]);
}

struct SpinLocks * init_spinlocks(int NumLock)
{
    int i;
    /* Initialize the spinlocks*/
    spin.SpinLocks = mymalloc("SpinLocks", NumLock * sizeof(omp_lock_t));
    #pragma omp parallel for
    for(i = 0; i < NumLock; i ++) {
        omp_init_lock(&spin.SpinLocks[i]);
    }
    spin.NumSpinLock = NumLock;
    return &spin;
}

void free_spinlocks(struct SpinLocks * spin)
{
    int i;
    for(i = 0; i < spin->NumSpinLock; i ++) {
        omp_destroy_lock(&(spin->SpinLocks[i]));
    }
    myfree((void *) spin->SpinLocks);
}

