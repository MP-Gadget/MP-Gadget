#ifndef DENSITY_H
#define DENSITY_H

#include "forcetree.h"
#include "timestep.h"
#include "densitykernel.h"
#include "utils/paramset.h"

struct density_params
{
    double DensityResolutionEta;		/*!< SPH resolution eta. See Price 2011. eq 12*/
    double MaxNumNgbDeviation;	/*!< Maximum allowed deviation neighbour number */

    /* These are for black hole neighbour finding and so belong in the density module, not the black hole module.*/
    double BlackHoleNgbFactor;	/*!< Factor by which the normal SPH neighbour should be increased/decreased */
    double BlackHoleMaxAccretionRadius;

    enum DensityKernelType DensityKernelType;  /* 0 for Cubic Spline,  (recmd NumNgb = 33)
                               1 for Quintic spline (recmd  NumNgb = 97) */

    /*!< minimum allowed SPH smoothing length in units of SPH gravitational softening length */
    double MinGasHsmlFractional;
};

struct sph_pred_data
{
    /*!< Predicted entropy at current particle drift time for SPH computation*/
    MyFloat * EntVarPred;
    /* VelPred can always be derived from the current time and acceleration.
     * However, doing so makes the SPH and hydro code much (a factor of two)
     * slower. It requires computing get_gravkick_factor twice with different arguments,
     * which defeats the lookup cache in timefac.c. Because VelPred is used multiple times,
     * it is much quicker to compute it once and re-use this*/
    MyFloat * VelPred;            /*!< Predicted velocity at current particle drift time for SPH. 3x vector.*/
};

/*Set the parameters of the density module*/
void set_density_params(ParameterSet * ps);
/*Set cooling module parameters from a density_params struct for the tests*/
void set_densitypar(struct density_params dp);

/* This routine computes the particle densities. If update_hsml is true
 * it runs multiple times, changing the smoothing length until
 * there are enough neighbours. If update_hsml is false (when initializing the EgyWtDensity)
 * it just computes densities.
 * If DoEgyDensity is true it also computes the entropy-weighted density for
 * pressure-entropy SPH. */
void density(const ActiveParticles * act, int update_hsml, int DoEgyDensity, int BlackHoleOn, double HydroCostFactor, double MinEgySpec, const inttime_t Ti_Current, struct sph_pred_data * SPH_predicted, MyFloat * GradRho, const ForceTree * const tree);

/* Get the desired nuber of neighbours for the supplied kernel*/
double GetNumNgb(enum DensityKernelType KernelType);

/* Get the current density kernel type*/
enum DensityKernelType GetDensityKernelType(void);

struct sph_pred_data slots_allocate_sph_pred_data(int nsph);
void slots_free_sph_pred_data(struct sph_pred_data * sph_pred);

#endif
