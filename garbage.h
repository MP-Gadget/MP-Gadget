#ifndef __GARBAGE_H
#define __GARBAGE_H

int domain_fork_particle(int parent, int ptype);
int domain_garbage_collection(void);
void domain_slots_init();
void domain_slots_grow(int newSlots[6]);

#endif
