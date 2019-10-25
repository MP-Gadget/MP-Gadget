#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <libgenic/allvars.h>
#include <libgadget/physconst.h>
#include <libgadget/utils.h>

static ParameterSet *
create_parameters()
{
    ParameterSet * ps = parameter_set_new();

    param_declare_string(ps, "FileWithInputSpectrum", REQUIRED, 0, "File containing input power spectrum, from CLASS or CAMB.");
    param_declare_string(ps, "OutputDir", REQUIRED, 0, "Output directory in which to store the ICs");
    param_declare_string(ps, "FileBase", REQUIRED, 0, "File name of the ICs.");

    param_declare_double(ps, "Omega0", REQUIRED, 0.2814, "Total matter density, cdm + baryons + massive neutrinos at z=0.");
    param_declare_double(ps, "OmegaBaryon", REQUIRED, 0.0464, "Omega Baryon: note this may be used for transfer functions even if gas is not produced.");
    param_declare_double(ps, "OmegaLambda", REQUIRED, 0.7186, "Dark energy density at z=0");
    param_declare_double(ps, "HubbleParam", REQUIRED, 0.697, "Hubble parameter");
    param_declare_int(ps,    "ProduceGas", REQUIRED, 0, "Should we create baryon particles?");
    param_declare_double(ps, "BoxSize", REQUIRED, 0, "Size of box in internal units.");
    param_declare_double(ps, "Redshift", REQUIRED, 99, "Starting redshift");
    param_declare_int(ps, "Nmesh", OPTIONAL, 0, "Size of the FFT grid used to estimate displacements. Should be > Ngrid.");
    param_declare_int(ps, "Ngrid", REQUIRED, 0, "Size of regular grid on which the undisplaced CDM particles are created.");
    param_declare_int(ps, "NgridGas", OPTIONAL, -1, "Size of regular grid on which the undisplaced gas particles are created.");
    param_declare_int(ps, "NgridNu", OPTIONAL, 0, "Number of neutrino particles created for hybrid neutrinos.");
    param_declare_int(ps, "Seed", REQUIRED, 0, "Random number generator seed used for the phases of the Gaussian random field.");
    param_declare_int(ps, "MakeGlassGas", OPTIONAL, -1, "Generate Glass IC for gas instead of Grid IC.");
    param_declare_int(ps, "MakeGlassCDM", OPTIONAL, 0, "Generate Glass IC for CDM instead of Grid IC.");

    param_declare_int(ps, "UnitaryAmplitude", OPTIONAL, 0, "If non-zero, generate unitary gaussians where |g| == 1.0.");
    param_declare_int(ps, "WhichSpectrum", OPTIONAL, 2, "Type of spectrum, 2 for file ");
    param_declare_double(ps, "Omega_fld", OPTIONAL, 0, "Energy density of dark energy fluid.");
    param_declare_double(ps, "w0_fld", OPTIONAL, -1., "Dark energy equation of state");
    param_declare_double(ps, "wa_fld", OPTIONAL, 0, "Dark energy evolution parameter");
    param_declare_double(ps, "Omega_ur", OPTIONAL, 0, "Extra radiation density, eg, a sterile neutrino");
    param_declare_double(ps, "MNue", OPTIONAL, 0, "First neutrino mass in eV.");
    param_declare_double(ps, "MNum", OPTIONAL, 0, "Second neutrino mass in eV.");
    param_declare_double(ps, "MNut", OPTIONAL, 0, "Third neutrino mass in eV.");
    param_declare_double(ps, "MWDM_therm", OPTIONAL, 0, "Assign a thermal velocity to the DM. Specifies WDM particle mass in keV.");
    param_declare_double(ps, "Max_nuvel", OPTIONAL, 5000, "Maximum neutrino velocity sampled from the F-D distribution.");

    param_declare_int(ps, "DifferentTransferFunctions", OPTIONAL, 1, "Use species specific transfer functions for baryon and CDM.");
    param_declare_int(ps, "ScaleDepVelocity", OPTIONAL, -1, "Use scale dependent velocity transfer functions instead of the scale-independent Zel'dovich approximation. Enabled by default iff DifferentTransferFunctions = 1");
    param_declare_string(ps, "FileWithTransferFunction", OPTIONAL, "", "File containing CLASS formatted transfer functions with extra metric transfer functions=y.");
    param_declare_double(ps, "MaxMemSizePerNode", OPTIONAL, 0.6, "Maximum memory per node, in fraction of total memory, or MB if > 1.");
    param_declare_double(ps, "CMBTemperature", OPTIONAL, 2.7255, "CMB temperature in K");
    param_declare_double(ps, "RadiationOn", OPTIONAL, 1, "Include radiation in the background.");
    param_declare_int(ps, "UsePeculiarVelocity", OPTIONAL, 1, "Snapshots will save peculiar velocities to the Velocity field. If 0, then v/sqrt(a) will be used in the ICs to match Gadget-2, but snapshots will save v * a.");
    param_declare_int(ps, "InvertPhase", OPTIONAL, 0, "Flip phase for paired simulation");

    param_declare_double(ps, "PrimordialAmp", OPTIONAL, 2.215e-9, "Ignored, but used by external CLASS script to set powr spectrum amplitude.");
    param_declare_double(ps, "Sigma8", OPTIONAL, -1, "Renormalise Sigma8 to this number if positive");
    param_declare_double(ps, "InputPowerRedshift", OPTIONAL, -1, "Redshift at which the input power is. Power spectrum will be rescaled to the initial redshift. Negative disables rescaling.");
    param_declare_double(ps, "PrimordialIndex", OPTIONAL, 0.971, "Tilting power, ignored for tabulated input.");
    param_declare_double(ps, "PrimordialRunning", OPTIONAL, 0, "Running of the spectral index, ignored for tabulated input, only used to pass parameter to tools/make_class_power.py");

    param_declare_double(ps, "UnitVelocity_in_cm_per_s", OPTIONAL, 1e5, "Velocity unit in cm/sec. Default is 1 km/s");
    param_declare_double(ps, "UnitLength_in_cm", OPTIONAL, CM_PER_MPC/1000, "Length unit in cm. Default is 1 kpc");
    param_declare_double(ps, "UnitMass_in_g", OPTIONAL, 1.989e43, "Mass unit in g. Default is 10^10 M_sun.");

    param_declare_int(ps, "NumPartPerFile", OPTIONAL, 1024 * 1024 * 128, "Number of particles per striped bigfile. Internal implementation detail.");
    param_declare_int(ps, "NumWriters", OPTIONAL, 0, "Number of processors allowed to write at one time.");
    return ps;
}

void read_parameterfile(char *fname)
{

    /* read parameter file on all processes for simplicty */

    ParameterSet * ps = create_parameters();
    char * error;

    if(0 != param_parse_file(ps, fname, &error)) {
        endrun(0, "Parsing %s failed: %s\n", fname, *error);
    }
    if(0 != param_validate(ps, &error)) {
        endrun(0, "Validation of %s failed: %s\n", fname, *error);
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
    All.CP.Omega_ur = param_get_double(ps, "Omega_ur");
    if(All.CP.OmegaLambda > 0 && All.CP.Omega_fld > 0)
        endrun(0, "Cannot have OmegaLambda and Omega_fld (evolving dark energy) at the same time!\n");
    All.CP.CMBTemperature = param_get_double(ps, "CMBTemperature");
    All.CP.RadiationOn = param_get_double(ps, "RadiationOn");
    All.CP.MNu[0] = param_get_double(ps, "MNue");
    All.CP.MNu[1] = param_get_double(ps, "MNum");
    All.CP.MNu[2] = param_get_double(ps, "MNut");
    All2.WDM_therm_mass = param_get_double(ps, "MWDM_therm");
    double MaxMemSizePerNode = param_get_double(ps, "MaxMemSizePerNode");
    if(MaxMemSizePerNode <= 1) {
        MaxMemSizePerNode *= get_physmem_bytes() / (1024 * 1024);
    }
    All.MaxMemSizePerNode = MaxMemSizePerNode;

    All2.ProduceGas = param_get_int(ps, "ProduceGas");
    All2.InvertPhase = param_get_int(ps, "InvertPhase");
    /*Unit system*/
    All.UnitVelocity_in_cm_per_s = param_get_double(ps, "UnitVelocity_in_cm_per_s");
    All.UnitLength_in_cm = param_get_double(ps, "UnitLength_in_cm");
    All.UnitMass_in_g = param_get_double(ps, "UnitMass_in_g");

    double Redshift = param_get_double(ps, "Redshift");

    /*Parameters of the power spectrum*/
    All2.PowerP.InputPowerRedshift = param_get_double(ps, "InputPowerRedshift");
    if(All2.PowerP.InputPowerRedshift < 0)
        All2.PowerP.InputPowerRedshift = Redshift;
    All2.PowerP.Sigma8 = param_get_double(ps, "Sigma8");
    /*Always specify Sigm8 at z=0*/
    if(All2.PowerP.Sigma8 > 0)
        All2.PowerP.InputPowerRedshift = 0;
    All2.PowerP.FileWithInputSpectrum = param_get_string(ps, "FileWithInputSpectrum");
    All2.PowerP.FileWithTransferFunction = param_get_string(ps, "FileWithTransferFunction");
    All2.PowerP.DifferentTransferFunctions = param_get_int(ps, "DifferentTransferFunctions");
    All2.PowerP.ScaleDepVelocity = param_get_int(ps, "ScaleDepVelocity");
    /* By default ScaleDepVelocity follows DifferentTransferFunctions.*/
    if(All2.PowerP.ScaleDepVelocity < 0) {
        All2.PowerP.ScaleDepVelocity = All2.PowerP.DifferentTransferFunctions;
    }
    All2.PowerP.WhichSpectrum = param_get_int(ps, "WhichSpectrum");
    All2.PowerP.PrimordialIndex = param_get_double(ps, "PrimordialIndex");
    All2.PowerP.PrimordialRunning = param_get_double(ps, "PrimordialRunning");

    /*Simulation parameters*/
    All.IO.UsePeculiarVelocity = param_get_int(ps, "UsePeculiarVelocity");
    All.BoxSize = param_get_double(ps, "BoxSize");
    All.Nmesh = param_get_int(ps, "Nmesh");
    All2.Ngrid = param_get_int(ps, "Ngrid");
    All2.NgridGas = param_get_int(ps, "NgridGas");
    if(All2.NgridGas < 0)
        All2.NgridGas = All2.Ngrid;
    if(!All2.ProduceGas)
        All2.NgridGas = 0;
    /*Enable 'hybrid' neutrinos*/
    All2.NGridNu = param_get_int(ps, "NgridNu");
    /* Convert physical km/s at z=0 in an unperturbed universe to
     * internal gadget (comoving) velocity units at starting redshift.*/
    All2.Max_nuvel = param_get_double(ps, "Max_nuvel") * pow(1+Redshift, 1.5) * (All.UnitVelocity_in_cm_per_s/1e5);
    All2.Seed = param_get_int(ps, "Seed");
    All2.UnitaryAmplitude = param_get_int(ps, "UnitaryAmplitude");
    param_get_string2(ps, "OutputDir", All.OutputDir, sizeof(All.OutputDir));
    param_get_string2(ps, "FileBase", All.InitCondFile, sizeof(All.InitCondFile));
    All2.MakeGlassGas = param_get_int(ps, "MakeGlassGas");
    /* We want to use a baryon glass by default if we have different transfer functions,
     * since that is the way we reproduce the linear growth. Otherwise use a grid by default.*/
    if(All2.MakeGlassGas < 0) {
        if(All2.PowerP.DifferentTransferFunctions)
            All2.MakeGlassGas = 1;
        else
            All2.MakeGlassGas = 0;
    }
    All2.MakeGlassCDM = param_get_int(ps, "MakeGlassCDM");

    int64_t NumPartPerFile = param_get_int(ps, "NumPartPerFile");

    int64_t Ngrid = All2.Ngrid;
    if(Ngrid < All2.NgridGas)
        Ngrid = All2.NgridGas;
    All2.NumFiles = ( Ngrid*Ngrid*Ngrid + NumPartPerFile - 1) / NumPartPerFile;
    All.IO.NumWriters = param_get_int(ps, "NumWriters");
    if(All2.PowerP.DifferentTransferFunctions && All2.PowerP.InputPowerRedshift != Redshift
        && (All2.ProduceGas || All.CP.MNu[0] + All.CP.MNu[1] + All.CP.MNu[2]))
        message(0, "WARNING: Using different transfer functions but also rescaling power to account for linear growth. NOT what you want!\n");
    if((All.CP.MNu[0] + All.CP.MNu[1] + All.CP.MNu[2] > 0) || All2.PowerP.DifferentTransferFunctions || All2.PowerP.ScaleDepVelocity)
        if(0 == strlen(All2.PowerP.FileWithTransferFunction))
            endrun(0,"For massive neutrinos, different transfer functions, or scale dependent growth functions you must specify a transfer function file\n");
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
