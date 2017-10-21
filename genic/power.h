#ifndef GENIC_POWER_H
#define GENIC_POWER_H

#include "cosmology.h"

struct power_params
{
    int WhichSpectrum;
    int DifferentTransferFunctions;
    char * FileWithTransferFunction;
    char * FileWithInputSpectrum;
    double Sigma8;
    double InputPowerRedshift;
    double SpectrumLengthScale;
    double PrimordialIndex;
};

double PowerSpec(double kmag, int Type);
int initialize_powerspectrum(int ThisTask, double InitTime, double UnitLength_in_cm_in, Cosmology * CPin, struct power_params * ppar);

#endif
