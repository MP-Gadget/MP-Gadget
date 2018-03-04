#ifndef GENIC_POWER_H
#define GENIC_POWER_H

#include <libgadget/cosmology.h>

struct power_params
{
    int WhichSpectrum;
    int DifferentTransferFunctions;
    int ScaleDepVelocity;
    char * FileWithTransferFunction;
    char * FileWithFutureTransferFunction;
    char * FileWithInputSpectrum;
    double Sigma8;
    double InputPowerRedshift;
    double SpectrumLengthScale;
    double PrimordialIndex;
    double InputFutureRedshift;
};

/*A note on gauge choice:
 * We run Gadget in N-body gauge (Fidler et al 2015) arxiv:1505.04756.
 * See also (Valkenburg & Villaescusa-Navarro 2016) arxiv:1610.08501, which includes
 * an N-body IC code.
 *
 * In this gauge, as long as there is zero anisotropic stress (true if omega_r = 0)
 * the equations of motion are the Newtonian equations and the density is the
 * comoving synchronous gauge density. Roughly speaking one is choosing a spatial gauge
 * comoves with the large-scale radiation perturbation, making them zero.
 *
 * The velocities for this gauge can be obtained by differentiating the synchronous gauge
 * velocity perturbations. Note that on super-horizon scales this is not the same as the
 * longitudinal gauge velocity!
 *
 * There is a residual error from the gradient of the radiation perturbations, which peaks
 * at k = 0.002 h/Mpc at 2% for z_ic = 100. It is larger at higher redshift.
 */

/* delta (square root of power spectrum) at current redshift.
 * Type == 0 is Gas, Type == 1 is DM, Type == 2 is neutrinos, Type == 3 is CDM + baryons.
 * Other types are total power. */
double DeltaSpec(double kmag, int Type);
/* Scale-dependent derivative of the growth function,
 * computed by differentiating the provided transfer functions. */
double dlogGrowth(double kmag, int ptype);
/* Read power spectrum and transfer function tables from disk, set up growth factors, initialise cosmology. */
int initialize_powerspectrum(int ThisTask, double InitTime, double UnitLength_in_cm_in, Cosmology * CPin, struct power_params * ppar);

#endif
