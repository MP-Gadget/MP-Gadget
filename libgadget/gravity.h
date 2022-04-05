#ifndef LONGRANGE_H
#define LONGRANGE_H

#include "forcetree.h"
#include "petapm.h"
#include "powerspectrum.h"
#include "timestep.h"

struct gravshort_tree_params
{
    double ErrTolForceAcc;      /*!< parameter for relative opening criterion in tree walk */
    double BHOpeningAngle;      /*!< Barnes-Hut parameter for opening criterion in tree walk */
    int TreeUseBH;              /*!< If true, use the BH opening angle. Otherwise use acceleration */
    /*! RCUT gives the maximum distance (in units of the scale used for the force split) out to which short-range
     * forces are evaluated in the short-range tree walk.*/
    double Rcut;
    /* Softening as a fraction of DM mean separation. */
    double FractionalGravitySoftening;
    /* if 1, enable adaptive gravitational softening for gas particles, which uses the Hsml as the ForceSoftening */
    int AdaptiveSoftening;
};

enum ShortRangeForceWindowType {
    SHORTRANGE_FORCE_WINDOW_TYPE_EXACT = 0,
    SHORTRANGE_FORCE_WINDOW_TYPE_ERFC = 1,
};

/* Fill the short-range gravity table*/
void gravshort_fill_ntab(const enum ShortRangeForceWindowType ShortRangeForceWindowType, const double Asmth);

/*! Sets the (comoving) softening length, converting from units of the mean DM separation to comoving internal units. */
void gravshort_set_softenings(double MeanDMSeparation);

/* gravitational softening length
 * (given in terms of an `equivalent' Plummer softening length) */
double FORCE_SOFTENING(int i, int type);

/*Defined in gravpm.c*/
void gravpm_init_periodic(PetaPM * pm, double BoxSize, double Asmth, int Nmesh, double G);

/* Apply the short-range window function, which includes the smoothing kernel.*/
int grav_apply_short_range_window(double r, double * fac, double * pot, const double cellsize);

/* Set up the module*/
void set_gravshort_tree_params(ParameterSet * ps);
/* Helpers for the tests*/
void set_gravshort_treepar(struct gravshort_tree_params tree_params);
struct gravshort_tree_params get_gravshort_treepar(void);

/* Computes the gravitational force on the PM grid
 * and saves the total matter power spectrum.
 * Parameters: Cosmology, Time, UnitLength_in_cm and PowerOutputDir are used by the power spectrum output code.
 * TimeIC and FastParticleType are used by the massive neutrino code. FastParticleType denotes possibly inactive particles.
 * SwallowedParticles is an optimisation to see if we have to check for swallowed BHs.
 * Note: tree is freed during this function*/
void gravpm_force(PetaPM * pm, ForceTree * tree, Cosmology * CP, double Time, double UnitLength_in_cm, const char * PowerOutputDir, double TimeIC, int FastParticleType, int SwallowedParticles);

void grav_short_pair(const ActiveParticles * act, PetaPM * pm, ForceTree * tree, double Rcut, double rho0, int NeutrinoTracer, int FastParticleType);
void grav_short_tree(const ActiveParticles * act, PetaPM * pm, ForceTree * tree, double rho0, int NeutrinoTracer, int FastParticleType);

/*Read the power spectrum, without changing the input value.*/
void measure_power_spectrum(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex *value);

/* Compute the power spectrum of the Fourier transformed grid in value.*/
void powerspectrum_add_mode(Power * PowerSpectrum, const int64_t k2, const int kpos[3], pfft_complex * const value, const double invwindow, double Nmesh);

#endif
