#ifndef THERMALVEL_H
#define THERMALVEL_H

/*Single parameter is the amplitude of the random velocities. All the physics is in here.
 * max_fd and min_fd give the maximum and minimum velocities to integrate over.
 * Note these values are dimensionless*/
/*Returns total fraction of the Fermi-Dirac distribution between max_fd and min_fd*/
double
init_thermalvel(const double v_amp, double max_fd, const double min_fd);

/*Add a randomly generated thermal speed in v_amp*(min_fd, max_fd) to a 3-velocity*/
void
add_thermal_speeds(float Vel[]);

/*Amplitude of the random velocity for neutrinos*/
double
NU_V0(const double Time, const double kBTNubyMNu, const double UnitVelocity_in_cm_per_s);

#endif
