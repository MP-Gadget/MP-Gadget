#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <libgenic/allvars.h>
#include <libgadget/physconst.h>
#include <libgadget/utils.h>

#define OPTIONAL 0
#define REQUIRED 1

static ParameterSet *
create_parameters()
{
    ParameterSet * ps = parameter_set_new();

    param_declare_string(ps, "FileWithInputSpectrum", REQUIRED, 0, "");
    param_declare_string(ps, "OutputDir", REQUIRED, 0, "");
    param_declare_string(ps, "FileBase", REQUIRED, 0, "");

    param_declare_double(ps, "Omega0", REQUIRED, 0.2814, "");
    param_declare_double(ps, "OmegaBaryon", REQUIRED, 0.0464, "");
    param_declare_double(ps, "OmegaLambda", REQUIRED, 0.7186, "Dark energy density at z=0");
    param_declare_double(ps, "HubbleParam", REQUIRED, 0.697, "Hubble parameter");
    param_declare_int(ps,    "ProduceGas", REQUIRED, 0, "Should we create baryon particles?");
    param_declare_double(ps, "BoxSize", REQUIRED, 0, "");
    param_declare_double(ps, "Redshift", REQUIRED, 99, "Starting redshift");
    param_declare_int(ps, "Nmesh", OPTIONAL, 0, "Size of the FFT grid used to estimate displacements. Should be > Ngrid.");
    param_declare_int(ps, "Ngrid", REQUIRED, 0, "Size of regular grid on which the undisplaced particles are created.");
    param_declare_int(ps, "NgridNu", OPTIONAL, 0, "Number of neutrino particles created for hybrid neutrinos.");
    param_declare_int(ps, "Seed", REQUIRED, 0, "");
    param_declare_int(ps, "UnitaryAmplitude", OPTIONAL, 0, "If non-zero, generate unitary gaussians where |g| == 1.0.");
    param_declare_int(ps, "WhichSpectrum", OPTIONAL, 2, "Type of spectrum, 2 for file ");
    param_declare_double(ps, "Omega_fld", OPTIONAL, 0, "Energy density of dark energy fluid.");
    param_declare_double(ps, "w0_fld", OPTIONAL, -1., "Dark energy equation of state");
    param_declare_double(ps, "wa_fld", OPTIONAL, 0, "Dark energy evolution parameter");
    param_declare_double(ps, "MNue", OPTIONAL, 0, "First neutrino mass in eV.");
    param_declare_double(ps, "MNum", OPTIONAL, 0, "Second neutrino mass in eV.");
    param_declare_double(ps, "MNut", OPTIONAL, 0, "Third neutrino mass in eV.");
    param_declare_double(ps, "MWDM_therm", OPTIONAL, 0, "Assign a thermal velocity to the DM. Specifies WDM particle mass in keV.");
    param_declare_double(ps, "Max_nuvel", OPTIONAL, 5000, "Maximum neutrino velocity sampled from the F-D distribution.");

    param_declare_int(ps, "DifferentTransferFunctions", OPTIONAL, 0, "Use species specific transfer functions for baryon and CDM.");
    param_declare_string(ps, "FileWithTransferFunction", OPTIONAL, "", "File containing CAMB formatted transfer functions.");
    param_declare_double(ps, "MaxMemSizePerNode", OPTIONAL, 0.6 * get_physmem_bytes() / (1024 * 1024), "");
    param_declare_double(ps, "CMBTemperature", OPTIONAL, 2.7255, "CMB temperature in K");
    param_declare_double(ps, "RadiationOn", OPTIONAL, 1, "Include radiation in the background.");
    param_declare_int(ps, "UsePeculiarVelocity", OPTIONAL, 0, "Set up an run that uses Peculiar Velocity in IO");
    param_declare_int(ps, "InvertPhase", OPTIONAL, 0, "Flip phase for paired simulation");

    param_declare_double(ps, "Sigma8", OPTIONAL, -1, "Renormalise Sigma8 to this number if positive");
    param_declare_double(ps, "InputPowerRedshift", OPTIONAL, 0, "Redshift at which the input power is. Power spectrum will be rescaled to the initial redshift. Negative disables rescaling.");
    param_declare_double(ps, "PrimordialIndex", OPTIONAL, 0.971, "Tilting power, ignored for tabulated input.");

    param_declare_double(ps, "UnitVelocity_in_cm_per_s", OPTIONAL, 1e5, "Velocity unit in cm/sec. Default is 1 km/s");
    param_declare_double(ps, "UnitLength_in_cm", OPTIONAL, 3.085678e21, "Length unit in cm. Default is 1 kpc");
    param_declare_double(ps, "UnitMass_in_g", OPTIONAL, 1.989e43, "Mass unit in g. Default is 10^10 M_sun.");
    param_declare_double(ps, "InputSpectrum_UnitLength_in_cm", REQUIRED, 3.085678e24, "Length unit of input power spectrum file in cm. Default is 1 Mpc");

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

    /*Cosmology*/
    All.CP.Omega0 = param_get_double(ps, "Omega0");
    All.CP.OmegaLambda = param_get_double(ps, "OmegaLambda");
    All.CP.OmegaBaryon = param_get_double(ps, "OmegaBaryon");
    All.CP.HubbleParam = param_get_double(ps, "HubbleParam");
    All.CP.Omega_fld = param_get_double(ps, "Omega_fld");
    All.CP.w0_fld = param_get_double(ps,"w0_fld");
    All.CP.wa_fld = param_get_double(ps,"wa_fld");
    if(All.CP.OmegaLambda > 0 && All.CP.Omega_fld > 0)
        endrun(0, "Cannot have OmegaLambda and Omega_fld (evolving dark energy) at the same time!\n");
    All.CP.CMBTemperature = param_get_double(ps, "CMBTemperature");
    All.CP.RadiationOn = param_get_double(ps, "RadiationOn");
    All.CP.MNu[0] = param_get_double(ps, "MNue");
    All.CP.MNu[1] = param_get_double(ps, "MNum");
    All.CP.MNu[2] = param_get_double(ps, "MNut");
    All2.WDM_therm_mass = param_get_double(ps, "MWDM_therm");

    All.MaxMemSizePerNode = param_get_double(ps, "MaxMemSizePerNode");

    All2.ProduceGas = param_get_int(ps, "ProduceGas");
    All2.DifferentTransferFunctions = param_get_int(ps, "DifferentTransferFunctions");
    All2.InvertPhase = param_get_int(ps, "InvertPhase");
    /*Unit system*/
    All.UnitVelocity_in_cm_per_s = param_get_double(ps, "UnitVelocity_in_cm_per_s");
    All.UnitLength_in_cm = param_get_double(ps, "UnitLength_in_cm");
    All.UnitMass_in_g = param_get_double(ps, "UnitMass_in_g");

    /*Parameters of the power spectrum*/
    All2.PowerP.InputPowerRedshift = param_get_double(ps, "InputPowerRedshift");
    All2.PowerP.Sigma8 = param_get_double(ps, "Sigma8");
    /*Always specify Sigm8 at z=0*/
    if(All2.PowerP.Sigma8 > 0)
        All2.PowerP.InputPowerRedshift = 0;
    All2.PowerP.FileWithInputSpectrum = param_get_string(ps, "FileWithInputSpectrum");
    All2.PowerP.FileWithTransferFunction = param_get_string(ps, "FileWithTransferFunction");
    All2.PowerP.WhichSpectrum = param_get_int(ps, "WhichSpectrum");
    All2.PowerP.SpectrumLengthScale = param_get_double(ps, "InputSpectrum_UnitLength_in_cm") / All.UnitLength_in_cm;
    All2.PowerP.PrimordialIndex = param_get_double(ps, "PrimordialIndex");

    /*Simulation parameters*/
    All.IO.UsePeculiarVelocity = param_get_int(ps, "UsePeculiarVelocity");


    All.BoxSize = param_get_double(ps, "BoxSize");
    double Redshift = param_get_double(ps, "Redshift");
    All.Nmesh = param_get_int(ps, "Nmesh");

    All2.Ngrid = param_get_int(ps, "Ngrid");
    /*Enable 'hybrid' neutrinos*/
    All2.NGridNu = param_get_int(ps, "NgridNu");
    /* Convert physical km/s at z=0 in an unperturbed universe to
     * internal gadget (comoving) velocity units at starting redshift.*/
    All2.Max_nuvel = param_get_double(ps, "Max_nuvel") * pow(1+Redshift, 1.5) * (All.UnitVelocity_in_cm_per_s/1e5);
    All2.Seed = param_get_int(ps, "Seed");
    All2.UnitaryAmplitude = param_get_int(ps, "UnitaryAmplitude");
    param_get_string2(ps, "OutputDir", All.OutputDir);
    param_get_string2(ps, "FileBase", All.InitCondFile);

    int64_t NumPartPerFile = param_get_int(ps, "NumPartPerFile");

    int64_t Ngrid = All2.Ngrid;
    All2.NumFiles = ( Ngrid*Ngrid*Ngrid + NumPartPerFile - 1) / NumPartPerFile;
    All.IO.NumWriters = param_get_int(ps, "NumWriters");
    if(All2.DifferentTransferFunctions && All2.PowerP.InputPowerRedshift >= 0)
        message(0, "WARNING: Using different transfer functions but also rescaling power to account for linear growth. NOT what you want!\n");
    if((All.CP.MNu[0] + All.CP.MNu[1] + All.CP.MNu[2] > 0) || All2.DifferentTransferFunctions)
        if(0 == strlen(All2.PowerP.FileWithTransferFunction))
            endrun(0,"For massive neutrinos or different transfer functions you must specify a transfer function file\n");
    if(!All.CP.RadiationOn && (All.CP.MNu[0] + All.CP.MNu[1] + All.CP.MNu[2] > 0))
        endrun(0,"You want massive neutrinos but no background radiation: this will give an inconsistent cosmology.\n");

    if(All.Nmesh == 0) {
        All.Nmesh = 2*Ngrid;
    }
    /*Set some units*/
    All.TimeIC = 1 / (1 + Redshift);
    All.UnitTime_in_s = All.UnitLength_in_cm / All.UnitVelocity_in_cm_per_s;

    All.G = GRAVITY / pow(All.UnitLength_in_cm, 3) * All.UnitMass_in_g * pow(All.UnitTime_in_s, 2);
    All.CP.Hubble = HUBBLE * All.UnitTime_in_s;
}
