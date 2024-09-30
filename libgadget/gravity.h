#ifndef LONGRANGE_H
#define LONGRANGE_H

#include "forcetree.h"
#include "petapm.h"
#include "powerspectrum.h"
#include "timestep.h"

struct gravshort_tree_params
{
    double ErrTolForceAcc;      /*!< parameter for relative opening criterion in tree walk.
                                 * Desired accuracy of the tree force in units of the old acceleration.*/
    double BHOpeningAngle;      /*!< Barnes-Hut parameter for opening criterion in tree walk */
    double MaxBHOpeningAngle;    /* When using the relative acceleration criterion, we also enforce a maximum BH opening criterion to avoid pathological cases.*/
    int TreeUseBH;              /*!< If true, use the BH opening angle. Otherwise use acceleration. If > 0, use the Barnes-Hut opening angle.*
                                 *  If < 0, use the acceleration condition. */
    /*! RCUT gives the maximum distance (in units of the scale used for the force split) out to which short-range
     * forces are evaluated in the short-range tree walk.*/
    double Rcut;
    /* Softening as a fraction of DM mean separation. */
    double FractionalGravitySoftening;
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
double FORCE_SOFTENING(void);

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
 * TimeIC is used by the massive neutrino code. A tree is built and freed during this function*/
void gravpm_force(PetaPM * pm, DomainDecomp * ddecomp, Cosmology * CP, double Time, double UnitLength_in_cm, const char * PowerOutputDir, double TimeIC);

void grav_short_pair(const ActiveParticles * act, PetaPM * pm, ForceTree * tree, double Rcut, double rho0);
void grav_short_tree(const ActiveParticles * act, PetaPM * pm, ForceTree * tree, MyFloat (* AccelStore)[3], double rho0, inttime_t Ti_Current);

/*Read the power spectrum, without changing the input value.*/
void measure_power_spectrum(PetaPM * pm, int64_t k2, int kpos[3], cufftComplex *value);

/* Compute the power spectrum of the Fourier transformed grid in value.*/
void powerspectrum_add_mode(Power * PowerSpectrum, const int64_t k2, const int kpos[3], cufftComplex * const value, const double invwindow, double Nmesh);

#endif
