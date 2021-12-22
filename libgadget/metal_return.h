#ifndef METAL_RETURN_H
#define METAL_RETURN_H

#include "forcetree.h"
#include "timestep.h"
#include "utils/paramset.h"
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_integration.h>
#include "slotsmanager.h"

struct interps
{
    gsl_interp2d * lifetime_interp;
    gsl_interp2d * agb_mass_interp;
    gsl_interp2d * agb_metallicity_interp;
    gsl_interp2d * agb_metals_interp[NMETALS];
    gsl_interp2d * snii_mass_interp;
    gsl_interp2d * snii_metallicity_interp;
    gsl_interp2d * snii_metals_interp[NMETALS];
};

/* Build the interpolators for each yield table. We use bilinear interpolation
 * so there is no extra memory allocation and we never free the tables*/
void setup_metal_table_interp(struct interps * interp);

struct MetalReturnPriv {
    gsl_integration_workspace ** gsl_work;
    MyFloat * StellarAges;
    MyFloat * MassReturn;
    MyFloat * LowDyingMass;
    MyFloat * HighDyingMass;
    double imf_norm;
    double hub;
    /* Maximum of the new gas mass*/
    double MaxGasMass;
    Cosmology *CP;
    MyFloat * StarVolumeSPH;
    struct interps interp;
    struct SpinLocks * spin;
};

void metal_return(const ActiveParticles * act, DomainDecomp * const ddecomp, Cosmology * CP, const double atime, const double AvgGasMass);

void set_metal_return_params(ParameterSet * ps);

/* Initialise the metal private structure, finding mass return.*/
int64_t metal_return_init(const ActiveParticles * act, Cosmology * CP, struct MetalReturnPriv * priv, const double atime);
/* Free memory allocated in metal_return_init*/
void metal_return_priv_free(struct MetalReturnPriv * priv);

/* Find stellar density, returning the total SPH Volume weights for each particle.*/
void stellar_density(const ActiveParticles * act, MyFloat * StarVolumeSPH, MyFloat * MassReturn, const ForceTree * const tree);

/* Determines whether metal return runs for this star this timestep*/
int metals_haswork(int i, MyFloat * MassReturn);
#endif
