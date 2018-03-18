#ifndef GENIC_POWER_H
#define GENIC_POWER_H

#include <libgadget/cosmology.h>

struct power_params
{
    int WhichSpectrum;
    char * FileWithTransferFunction;
    char * FileWithInputSpectrum;
    double Sigma8;
    double InputPowerRedshift;
    double SpectrumLengthScale;
    double PrimordialIndex;
};

/* delta (square root of power spectrum) at current redshift.
 * Type == 0 is Gas, Type == 1 is DM, Type == 2 is neutrinos, Type == 3 is CDM + baryons.
 * Other types are total power. */
double DeltaSpec(double kmag, int Type);
/* Read power spectrum and transfer function tables from disk, set up growth factors, initialise cosmology. */
int initialize_powerspectrum(int ThisTask, double InitTime, double UnitLength_in_cm_in, Cosmology * CPin, struct power_params * ppar);

#endif
