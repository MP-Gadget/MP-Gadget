#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include "allvars.h"
#include "proto.h"
#include "endrun.h"
#include "paramset.h"

void set_units(void)		/* ... set some units */
{
  UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s;

  G = GRAVITY / pow(UnitLength_in_cm, 3) * UnitMass_in_g * pow(UnitTime_in_s, 2);
  Hubble = HUBBLE * UnitTime_in_s;
}


static ParameterSet *
create_parameters()
{
    ParameterSet * ps = parameter_set_new();

    param_declare_string(ps, "FileWithInputSpectrum", 1, 0, "");
    param_declare_string(ps, "OutputDir", 1, 0, "");
    param_declare_string(ps, "FileBase", 1, 0, "");

    param_declare_double(ps, "MaxMemoryPerCore", 0, 1300., "");
    param_declare_double(ps, "Omega0", 1, 0, "");
    param_declare_double(ps, "OmegaLambda", 1, 0, "");
    param_declare_double(ps, "OmegaBaryon", 1, 0, "");
    param_declare_int(ps,    "ProduceGas", 1, 0, "");
    param_declare_double(ps, "OmegaDM_2ndSpecies", 1, 0, "");
    param_declare_double(ps, "HubbleParam", 1, 0, "");
    param_declare_double(ps, "ShapeGamma", 1, 0, "");
    param_declare_double(ps, "Sigma8", 1, 0, "");
    param_declare_double(ps, "PrimordialIndex", 1, 0, "Ignored for tabulated input.");
    param_declare_double(ps, "BoxSize", 1, 0, "");
    param_declare_double(ps, "Redshift", 1, 0, "");
    param_declare_int(ps, "Nmesh", 1, 0, "");
    param_declare_int(ps, "Nsample", 1, 0, "");
    param_declare_int(ps, "Ngrid", 1, 0, "");
    param_declare_int(ps, "Seed", 1, 0, "");
    param_declare_int(ps, "SphereMode", 1, 0, "");
    param_declare_int(ps, "WhichSpectrum", 1, 0, ""); 
    param_declare_double(ps, "UnitVelocity_in_cm_per_s", 1, 0, "");
    param_declare_double(ps, "UnitLength_in_cm", 1, 0, ""); 
    param_declare_double(ps, "UnitMass_in_g", 1, 0, "");
    param_declare_double(ps, "InputSpectrum_UnitLength_in_cm", 1, 0, "");
    param_declare_int(ps, "WDM_On", 1, 0, "");
    param_declare_int(ps, "WDM_Vtherm_On", 1, 0, "");
    param_declare_double(ps, "WDM_PartMass_in_kev", 1, 0, "");

    param_declare_int(ps, "NumPartPerFile", 0, 1024 * 1024 * 128, "");
    param_declare_int(ps, "NumWriters", 0, 0, "");

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
    param_dump(ps, stdout);
    message(0, "----------------------------------------------\n");
    
    Omega = param_get_double(ps, "Omega0");
    MaxMemoryPerCore = param_get_double(ps, "MaxMemoryPerCore");
    OmegaLambda = param_get_double(ps, "OmegaLambda");
    OmegaBaryon = param_get_double(ps, "OmegaBaryon");
    ProduceGas = param_get_int(ps, "ProduceGas");
    OmegaDM_2ndSpecies = param_get_double(ps, "OmegaDM_2ndSpecies");
    HubbleParam = param_get_double(ps, "HubbleParam");
    ShapeGamma = param_get_double(ps, "ShapeGamma");
    Sigma8 = param_get_double(ps, "Sigma8");
    PrimordialIndex = param_get_double(ps, "PrimordialIndex");
    Box = param_get_double(ps, "BoxSize");
    Redshift = param_get_double(ps, "Redshift");
    Nmesh = param_get_int(ps, "Nmesh");
    Nsample = param_get_int(ps, "Nsample");
    Ngrid = param_get_int(ps, "Ngrid");
    FileWithInputSpectrum = param_get_string(ps, "FileWithInputSpectrum");
    Seed = param_get_int(ps, "Seed");
    SphereMode = param_get_int(ps, "SphereMode");
    OutputDir = param_get_string(ps, "OutputDir");
    FileBase = param_get_string(ps, "FileBase");
    WhichSpectrum = param_get_int(ps, "WhichSpectrum");
    UnitVelocity_in_cm_per_s = param_get_double(ps, "UnitVelocity_in_cm_per_s");
    UnitLength_in_cm = param_get_double(ps, "UnitLength_in_cm");
    UnitMass_in_g = param_get_double(ps, "UnitMass_in_g");
    InputSpectrum_UnitLength_in_cm = param_get_double(ps, "InputSpectrum_UnitLength_in_cm");
    WDM_On = param_get_int(ps, "WDM_On");
    WDM_Vtherm_On = param_get_int(ps, "WDM_Vtherm_On");
    WDM_PartMass_in_kev = param_get_double(ps, "WDM_PartMass_in_kev");

    NumPartPerFile = param_get_int(ps, "NumPartPerFile");
    NumWriters = param_get_int(ps, "NumWriters");

    if(Ngrid == 0) {
        Ngrid = Nmesh;
    }
}
