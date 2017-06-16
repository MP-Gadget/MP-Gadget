#ifndef TIMESTEP_H
#define TIMESTEP_H

int find_active_timebins(int next_kick);
void reconstruct_timebins(void);
void set_global_time(double newtime);
void advance_and_find_timesteps(void);
double find_dt_displacement_constraint(void);

#endif
