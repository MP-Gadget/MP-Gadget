#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include "genic/allvars.h"
#include "genic/proto.h"
#include "endrun.h"
#include "paramset.h"
#include "physconst.h"

#define OPTIONAL 0
#define REQUIRED 1

static ParameterSet *
create_parameters()
{
    ParameterSet * ps = parameter_set_new();

    param_declare_string(ps, "FileWithInputSpectrum", REQUIRED, 0, "");
    param_declare_string(ps, "OutputDir", REQUIRED, 0, "");
    param_declare_string(ps, "FileBase", REQUIRED, 0, "");

    param_declare_double(ps, "Omega0", REQUIRED, 0, "");
    param_declare_double(ps, "OmegaLambda", REQUIRED, 0, "");
    param_declare_double(ps, "OmegaBaryon", REQUIRED, 0, "");
    param_declare_double(ps, "HubbleParam", REQUIRED, 0.7, "Hubble parameter");
    param_declare_int(ps,    "ProduceGas", REQUIRED, 0, "Should we create baryon particles?");
    param_declare_double(ps, "BoxSize", REQUIRED, 0, "");
    param_declare_double(ps, "Redshift", REQUIRED, 99, "Starting redshift");
    param_declare_int(ps, "Nmesh", OPTIONAL, 0, "Size of the FFT grid used to estimate displacements. Should be > Ngrid.");
    param_declare_int(ps, "Ngrid", REQUIRED, 0, "Size of regular grid on which the undisplaced particles are created.");
    param_declare_int(ps, "Seed", REQUIRED, 0, "");
    param_declare_int(ps, "WhichSpectrum", REQUIRED, 2, "Type of spectrum, 2 for file ");

    param_declare_double(ps, "MaxMemoryPerCore", OPTIONAL, 1300., "");
    param_declare_double(ps, "CMBTemperature", OPTIONAL, 2.7255, "CMB temperature in K");
    param_declare_double(ps, "RadiationOn", OPTIONAL, 1, "Include radiation in the background.");
    param_declare_int(ps, "UsePeculiarVelocity", OPTIONAL, 0, "Write a IC similiar to a FastPM output");
    param_declare_double(ps, "Sigma8", OPTIONAL, -1, "Renormalise Sigma8 to this number if positive");
    param_declare_int(ps, "InputPowerRedshift", OPTIONAL, 0, "Redshift at which the input power is. Power spectrum will be rescaled to the initial redshift. Negative disables rescaling.");
    param_declare_double(ps, "PrimordialIndex", OPTIONAL, 0.971, "Tilting power, ignored for tabulated input.");

    param_declare_double(ps, "UnitVelocity_in_cm_per_s", OPTIONAL, 1e5, "Velocity unit in cm/sec. Default is 1 km/s");
    param_declare_double(ps, "UnitLength_in_cm", OPTIONAL, 3.085678e21, "Length unit in cm. Default is 1 kpc"); 
    param_declare_double(ps, "UnitMass_in_g", OPTIONAL, 1.989e43, "Mass unit in g. Default is 10^10 M_sun.");
    param_declare_double(ps, "InputSpectrum_UnitLength_in_cm", REQUIRED, 3.085678e24, "Length unit of input power spectrum file in cm. Default is 1 Mpc");

    param_declare_int(ps, "NumPartPerFile", OPTIONAL, 1024 * 1024 * 128, "");
    param_declare_int(ps, "NumWriters", OPTIONAL, 0, "");


    param_declare_double(ps, "ShapeGamma", OPTIONAL, 0.201, "Ignored.");
    param_declare_double(ps, "OmegaDM_2ndSpecies", OPTIONAL, 0, "Ignored.");
    param_declare_int(ps, "SphereMode", OPTIONAL, 1, "Ignored.");
    param_declare_int(ps, "Nsample", OPTIONAL, 0, "Ignored.");
    param_declare_int(ps, "WDM_On", OPTIONAL, 0, "Ignored.");
    param_declare_int(ps, "WDM_Vtherm_On", OPTIONAL, 0, "Ignored.");
    param_declare_double(ps, "WDM_PartMass_in_kev", OPTIONAL, 0, "Ignored.");

    return ps;
}
void read_parameterfile(char *fname)
{

    /* read parameter file on all processes for simplicty */

    ParameterSet * ps = create_parameters();

    if(0 != param_parse_file(ps, fname)) {
        endrun(0, "Parsing %s failed.", fname);
    }
    if(0 != param_validate(ps)) {
        endrun(0, "Validation of %s failed.", fname);
    }

    message(0, "----------- Running with Parameters ----------\n");
    if(ThisTask == 0)
        param_dump(ps, stdout);

    message(0, "----------------------------------------------\n");
    
    CP.Omega0 = param_get_double(ps, "Omega0");
    CP.OmegaLambda = param_get_double(ps, "OmegaLambda");
    CP.OmegaBaryon = param_get_double(ps, "OmegaBaryon");
    CP.HubbleParam = param_get_double(ps, "HubbleParam");
    CP.CMBTemperature = param_get_double(ps, "CMBTemperature");
    CP.RadiationOn = param_get_double(ps, "RadiationOn");
    /* If massive neutrinos are implemented and enabled this
     * should be set to zero, so neutrinos are not included twice.*/
    CP.MasslessNeutrinosOn = 1;
    MaxMemoryPerCore = param_get_double(ps, "MaxMemoryPerCore");
    ProduceGas = param_get_int(ps, "ProduceGas");
    InputPowerRedshift = param_get_double(ps, "InputPowerRedshift");
    UsePeculiarVelocity = param_get_int(ps, "UsePeculiarVelocity");
    Sigma8 = param_get_double(ps, "Sigma8");
    PrimordialIndex = param_get_double(ps, "PrimordialIndex");
    Box = param_get_double(ps, "BoxSize");
    double Redshift = param_get_double(ps, "Redshift");
    Nmesh = param_get_int(ps, "Nmesh");
    Ngrid = param_get_int(ps, "Ngrid");
    FileWithInputSpectrum = param_get_string(ps, "FileWithInputSpectrum");
    Seed = param_get_int(ps, "Seed");
    OutputDir = param_get_string(ps, "OutputDir");
    FileBase = param_get_string(ps, "FileBase");
    WhichSpectrum = param_get_int(ps, "WhichSpectrum");
    UnitVelocity_in_cm_per_s = param_get_double(ps, "UnitVelocity_in_cm_per_s");
    UnitLength_in_cm = param_get_double(ps, "UnitLength_in_cm");
    UnitMass_in_g = param_get_double(ps, "UnitMass_in_g");
    InputSpectrum_UnitLength_in_cm = param_get_double(ps, "InputSpectrum_UnitLength_in_cm");
    NumPartPerFile = param_get_int(ps, "NumPartPerFile");
    NumWriters = param_get_int(ps, "NumWriters");

    if(Nmesh == 0) {
        Nmesh = 2*Ngrid;
    }
    /*Set some units*/
    InitTime = 1 / (1 + Redshift);
    UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s;

    G = GRAVITY / pow(UnitLength_in_cm, 3) * UnitMass_in_g * pow(UnitTime_in_s, 2);
    CP.Hubble = HUBBLE * UnitTime_in_s;
    init_cosmology(&CP);
}
