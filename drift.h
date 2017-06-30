#ifndef __DRIFT_H
#define __DRIFT_H

int drift_particle_full(int i, int ti1, int blocking);
void drift_particle(int i, int ti1);
void lock_particle_if_not(int i, MyIDType id);
void unlock_particle_if_not(int i, MyIDType id);
void lock_particle(int i);
void unlock_particle(int i);

void move_particles(int ti1);

#endif
