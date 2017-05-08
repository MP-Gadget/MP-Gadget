#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include "genic-allvars.h"
#include "genic-proto.h"
#include "endrun.h"
#include "paramset.h"

#define OPTIONAL 0
#define REQUIRED 1

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

    param_declare_string(ps, "FileWithInputSpectrum", REQUIRED, 0, "");
    param_declare_string(ps, "OutputDir", REQUIRED, 0, "");
    param_declare_string(ps, "FileBase", REQUIRED, 0, "");

    param_declare_double(ps, "MaxMemoryPerCore", OPTIONAL, 1300., "");
    param_declare_double(ps, "Omega0", REQUIRED, 0, "");
    param_declare_double(ps, "OmegaLambda", REQUIRED, 0, "");
    param_declare_double(ps, "OmegaBaryon", REQUIRED, 0, "");
    param_declare_int(ps,    "ProduceGas", REQUIRED, 0, "");
    param_declare_double(ps, "OmegaDM_2ndSpecies", REQUIRED, 0, "");
    param_declare_int(ps, "UsePeculiarVelocity", OPTIONAL, 0, "Write a IC similiar to a FastPM output");
    param_declare_double(ps, "HubbleParam", REQUIRED, 0, "");
    param_declare_double(ps, "ShapeGamma", OPTIONAL, 0.201, "");
    param_declare_double(ps, "Sigma8", OPTIONAL, -1, "Renoramlize Sigma8 to this number if positive");
    param_declare_double(ps, "PrimordialIndex", OPTIONAL, 0.971, "Tilting power, ignored for tabulated input.");
    param_declare_double(ps, "BoxSize", REQUIRED, 0, "");
    param_declare_double(ps, "Redshift", REQUIRED, 0, "");
    param_declare_int(ps, "Nmesh", REQUIRED, 0, "");
    param_declare_int(ps, "Nsample", REQUIRED, 0, "");
    param_declare_int(ps, "Ngrid", REQUIRED, 0, "");
    param_declare_int(ps, "Seed", REQUIRED, 0, "");
    param_declare_int(ps, "SphereMode", OPTIONAL, 1, " if 1 only modes with |k| < k_Nyquist are used. otherwise all modes are filled. ");
    param_declare_int(ps, "WhichSpectrum", OPTIONAL, 2, "Type of spectrum, 2 for file "); 
    param_declare_double(ps, "UnitVelocity_in_cm_per_s", REQUIRED, 0, "");
    param_declare_double(ps, "UnitLength_in_cm", REQUIRED, 0, ""); 
    param_declare_double(ps, "UnitMass_in_g", REQUIRED, 0, "");
    param_declare_double(ps, "InputSpectrum_UnitLength_in_cm", REQUIRED, 0, "");
    param_declare_int(ps, "WDM_On", REQUIRED, 0, "");
    param_declare_int(ps, "WDM_Vtherm_On", REQUIRED, 0, "");
    param_declare_double(ps, "WDM_PartMass_in_kev", REQUIRED, 0, "");

    param_declare_int(ps, "NumPartPerFile", OPTIONAL, 1024 * 1024 * 128, "");
    param_declare_int(ps, "NumWriters", OPTIONAL, 0, "");

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
    
    Omega = param_get_double(ps, "Omega0");
    MaxMemoryPerCore = param_get_double(ps, "MaxMemoryPerCore");
    OmegaLambda = param_get_double(ps, "OmegaLambda");
    OmegaBaryon = param_get_double(ps, "OmegaBaryon");
    ProduceGas = param_get_int(ps, "ProduceGas");
    OmegaDM_2ndSpecies = param_get_double(ps, "OmegaDM_2ndSpecies");
    HubbleParam = param_get_double(ps, "HubbleParam");
    UsePeculiarVelocity = param_get_int(ps, "UsePeculiarVelocity");
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
