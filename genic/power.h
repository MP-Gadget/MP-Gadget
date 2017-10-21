#ifndef GENIC_POWER_H
#define GENIC_POWER_H
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
void   initialize_powerspectrum(struct power_params * ppar);

#endif
