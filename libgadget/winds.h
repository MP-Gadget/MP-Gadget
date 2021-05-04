#ifndef WINDS_H
#define WINDS_H

#include "forcetree.h"
#include "utils/paramset.h"

/*
 * Enumeration of the supported wind models.
 * Wind models may be combined.
 * SH03, VS08 and OFJT10 are supported.
 * */
enum WindModel {
    WIND_SUBGRID = 1,
    WIND_DECOUPLE_SPH = 2,
    WIND_USE_HALO = 4,
    WIND_FIXED_EFFICIENCY = 8,
    /* Has no effect: only isotropic winds are implemented*/
    WIND_ISOTROPIC = 512,
};

/*Set the parameters of the wind model*/
void set_winds_params(ParameterSet * ps);

/*Initialize the wind model from the SFR module*/
void init_winds(double FactorSN, double EgySpecSN, double PhysDensThresh, double UnitTime_in_s);

/*Evolve a wind particle, reducing its DelayTime*/
void winds_evolve(int i, double a3inv, double hubble);

/*do the treewalk for the wind model*/
void winds_and_feedback(int * NewStars, int NumNewStars, const double Time, const double hubble, ForceTree * tree);

/*Make a wind particle at the site of recent star formation.*/
int winds_make_after_sf(int i, double sm, double atime);

/*Tests whether a given particle has been made a wind particle and is hydrodynamically decoupled*/
int winds_is_particle_decoupled(int i);

/* Set and evolve the hydro parameters for a decoupled wind particle.*/
void winds_decoupled_hydro(int i, double atime);

/* Returns 1 if the winds ever decouple, 0 otherwise*/
int winds_ever_decouple(void);
#endif
