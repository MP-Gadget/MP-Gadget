#ifndef __SPINLOCKS_H
#define __SPINLOCKS_H
/* Manages particle locking:
 * init_spinlocks allocates and initialises an array of spinlocks.
 * free_spinlocks destroys and frees them.
 * (un)lock_particle locks and unlocks the i'th spinlock.
 * try_lock_particle trys to lock the particle, returning 0 on success.
 * Warning! If NO_OPENMP_SPINLOCK is defined, these functions do nothing!*/

struct SpinLocks;

struct SpinLocks * init_spinlocks(int NumLock);
void free_spinlocks(struct SpinLocks * spin);
void lock_particle(int i, struct SpinLocks * spin);
void unlock_particle(int i, struct SpinLocks * spin);
int try_lock_particle(int i, struct SpinLocks * spin);
#endif
