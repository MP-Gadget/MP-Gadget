#ifndef __DRIFT_H
#define __DRIFT_H

void drift_particle(int i, unsigned int ti1);
void lock_particle(int i);
void unlock_particle(int i);

void drift_active_particles(unsigned int ti1);
void drift_all_particles(unsigned int ti1);

#endif
