#ifndef LONGRANGE_H
#define LONGRANGE_H

#include "forcetree.h"
#include "petapm.h"
#include "powerspectrum.h"

struct TreeAccParams
{
    double ErrTolForceAcc;      /*!< parameter for relative opening criterion in tree walk */
    double BHOpeningAngle;      /*!< Barnes-Hut parameter for opening criterion in tree walk */
    int TreeUseBH;              /*!< If true, use the BH opening angle. Otherwise use acceleration */
    /*! RCUT gives the maximum distance (in units of the scale used for the force split) out to which short-range
     * forces are evaluated in the short-range tree walk.*/
    double Rcut;
};

/* Fill the short-range gravity table*/
void gravshort_fill_ntab(void);

/*Defined in gravpm.c*/
PetaPM gravpm_init_periodic(double BoxSize, int Nmesh);

/* Apply the short-range window function, which includes the smoothing kernel.*/
int grav_apply_short_range_window(double r, double * fac, double * pot, const double cellsize);

/*Note: tree is rebuilt during this function*/
void gravpm_force(PetaPM * pm, ForceTree * tree);

void grav_short_pair(ForceTree * tree, double G, double BoxSize, double Nmesh, double Asmth, double rho0, int NeutrinoTracer, int FastParticleType, struct TreeAccParams treeacc);
void grav_short_tree(ForceTree * tree, double G, double BoxSize, double Nmesh, double Asmth, double rho0, int NeutrinoTracer, int FastParticleType, struct TreeAccParams treeacc);

/*Read the power spectrum, without changing the input value.*/
void measure_power_spectrum(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex *value);

/* Compute the power spectrum of the Fourier transformed grid in value.*/
void powerspectrum_add_mode(Power * PowerSpectrum, const int64_t k2, const int kpos[3], pfft_complex * const value, const double invwindow, double Nmesh);

#endif
