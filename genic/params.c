#include <stdint.h>
#include <string.h>
#include <mpi.h>
#include <libgenic/allvars.h>
#include <libgenic/proto.h>
#include <libgadget/physconst.h>
#include <libgadget/utils.h>

static ParameterSet *
create_parameters(void)
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

    param_declare_int(ps, "UnitaryAmplitude", OPTIONAL, 1, "If 0, each Fourier mode in the initial power spectrum is scattered. If 1 each Fourier mode is not scattered and we generate unitary gaussians for the initial phases.");
    param_declare_int(ps, "WhichSpectrum", OPTIONAL, 2, "Type of spectrum, 2 for file ");
    param_declare_double(ps, "Omega_fld", OPTIONAL, 0, "Energy density of dark energy fluid.");
    param_declare_double(ps, "w0_fld", OPTIONAL, -1., "Dark energy equation of state");
    param_declare_double(ps, "wa_fld", OPTIONAL, 0, "Dark energy evolution parameter");
    param_declare_double(ps, "Omega_ur", OPTIONAL, 0, "Extra radiation density, eg, a sterile neutrino");
    param_declare_int(ps, "CLASS_Radiation", OPTIONAL, 0, "Boolean. If enabled, we enforce that sum(Omega_i) = 1. If disabled then Omega_m + Omega_L + Omega_fld + Omega_k = 1 and so sum(Omega_i) ~ 1+Omega_g");
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
    param_declare_int(ps, "SavePrePos", OPTIONAL, 1, "Save the pre-displacement positions in the snapshot.");
    param_declare_int(ps, "InvertPhase", OPTIONAL, 0, "Flip phase for paired simulation");
    param_declare_int(ps, "PrePosGridCenter", OPTIONAL, 0, "Set pre-displacement positions at the center of the grid");
    param_declare_int(ps, "ShowBacktrace", OPTIONAL, 1, "Print a backtrace on crash. Hangs on stampede.");

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

void read_parameterfile(char *fname, struct genic_config * GenicConfig, int * ShowBacktrace, double * MaxMemSizePerNode, Cosmology * CP)
{

    /* read parameter file on all processes for simplicty */
    ParameterSet * ps = create_parameters();
    int ThisTask;

    if(0 != param_parse_file(ps, fname)) {
        endrun(0, "Parsing %s failed. \n", fname);
    }
    if(0 != param_validate(ps)) {
        endrun(0, "Validation of %s failed.\n", fname);
    }

    message(0, "----------- Running with Parameters ----------\n");

    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0)
        param_dump(ps, stdout);

    message(0, "----------------------------------------------\n");

    /*Cosmology*/
    CP->Omega0 = param_get_double(ps, "Omega0");
    CP->OmegaLambda = param_get_double(ps, "OmegaLambda");
    CP->OmegaBaryon = param_get_double(ps, "OmegaBaryon");
    CP->HubbleParam = param_get_double(ps, "HubbleParam");
    CP->Omega_fld = param_get_double(ps, "Omega_fld");
    CP->w0_fld = param_get_double(ps,"w0_fld");
    CP->wa_fld = param_get_double(ps,"wa_fld");
    CP->Omega_ur = param_get_double(ps, "Omega_ur");
    CP->use_class_radiation_convention = param_get_int(ps, "CLASS_Radiation");
    if(CP->OmegaLambda > 0 && CP->Omega_fld > 0)
        endrun(0, "Cannot have OmegaLambda and Omega_fld (evolving dark energy) at the same time!\n");
    CP->CMBTemperature = param_get_double(ps, "CMBTemperature");
    CP->RadiationOn = param_get_double(ps, "RadiationOn");
    CP->MNu[0] = param_get_double(ps, "MNue");
    CP->MNu[1] = param_get_double(ps, "MNum");
    CP->MNu[2] = param_get_double(ps, "MNut");
    GenicConfig->WDM_therm_mass = param_get_double(ps, "MWDM_therm");
    *MaxMemSizePerNode = param_get_double(ps, "MaxMemSizePerNode");
    if(*MaxMemSizePerNode <= 1) {
        (*MaxMemSizePerNode) *= get_physmem_bytes() / (1024 * 1024);
    }

    GenicConfig->ProduceGas = param_get_int(ps, "ProduceGas");
    GenicConfig->InvertPhase = param_get_int(ps, "InvertPhase");
    /*Unit system*/
    GenicConfig->units.UnitVelocity_in_cm_per_s = param_get_double(ps, "UnitVelocity_in_cm_per_s");
    GenicConfig->units.UnitLength_in_cm = param_get_double(ps, "UnitLength_in_cm");
    GenicConfig->units.UnitMass_in_g = param_get_double(ps, "UnitMass_in_g");


    *ShowBacktrace = param_get_int(ps, "ShowBacktrace");

    double Redshift = param_get_double(ps, "Redshift");

    /*Parameters of the power spectrum*/
    GenicConfig->PowerP.InputPowerRedshift = param_get_double(ps, "InputPowerRedshift");
    if(GenicConfig->PowerP.InputPowerRedshift < 0)
        GenicConfig->PowerP.InputPowerRedshift = Redshift;
    GenicConfig->PowerP.Sigma8 = param_get_double(ps, "Sigma8");
    /*Always specify Sigm8 at z=0*/
    if(GenicConfig->PowerP.Sigma8 > 0)
        GenicConfig->PowerP.InputPowerRedshift = 0;
    GenicConfig->PowerP.FileWithInputSpectrum = param_get_string(ps, "FileWithInputSpectrum");
    GenicConfig->PowerP.FileWithTransferFunction = param_get_string(ps, "FileWithTransferFunction");
    GenicConfig->PowerP.DifferentTransferFunctions = param_get_int(ps, "DifferentTransferFunctions");
    GenicConfig->PowerP.ScaleDepVelocity = param_get_int(ps, "ScaleDepVelocity");
    /* By default ScaleDepVelocity follows DifferentTransferFunctions.*/
    if(GenicConfig->PowerP.ScaleDepVelocity < 0) {
        GenicConfig->PowerP.ScaleDepVelocity = GenicConfig->PowerP.DifferentTransferFunctions;
    }
    GenicConfig->PowerP.WhichSpectrum = param_get_int(ps, "WhichSpectrum");
    GenicConfig->PowerP.PrimordialIndex = param_get_double(ps, "PrimordialIndex");
    GenicConfig->PowerP.PrimordialRunning = param_get_double(ps, "PrimordialRunning");

    /*Simulation parameters*/
    GenicConfig->UsePeculiarVelocity = param_get_int(ps, "UsePeculiarVelocity");
    GenicConfig->SavePrePos = param_get_int(ps, "SavePrePos");
    GenicConfig->PrePosGridCenter = param_get_int(ps, "PrePosGridCenter");
    GenicConfig->BoxSize = param_get_double(ps, "BoxSize");
    GenicConfig->Nmesh = param_get_int(ps, "Nmesh");
    GenicConfig->Ngrid = param_get_int(ps, "Ngrid");
    GenicConfig->NgridGas = param_get_int(ps, "NgridGas");
    if(GenicConfig->NgridGas < 0)
        GenicConfig->NgridGas = GenicConfig->Ngrid;
    if(!GenicConfig->ProduceGas)
        GenicConfig->NgridGas = 0;
    /*Enable 'hybrid' neutrinos*/
    GenicConfig->NGridNu = param_get_int(ps, "NgridNu");
    /* Convert physical km/s at z=0 in an unperturbed universe to
     * internal gadget (comoving) velocity units at starting redshift.*/
    GenicConfig->Max_nuvel = param_get_double(ps, "Max_nuvel") * (1+Redshift) * (GenicConfig->units.UnitVelocity_in_cm_per_s/1e5);
    GenicConfig->Seed = param_get_int(ps, "Seed");
    GenicConfig->UnitaryAmplitude = param_get_int(ps, "UnitaryAmplitude");
    param_get_string2(ps, "OutputDir", GenicConfig->OutputDir, sizeof(GenicConfig->OutputDir));
    param_get_string2(ps, "FileBase", GenicConfig->InitCondFile, sizeof(GenicConfig->InitCondFile));
    GenicConfig->MakeGlassGas = param_get_int(ps, "MakeGlassGas");
    /* We want to use a baryon glass by default if we have different transfer functions,
     * since that is the way we reproduce the linear growth. Otherwise use a grid by default.*/
    if(GenicConfig->MakeGlassGas < 0) {
        if(GenicConfig->PowerP.DifferentTransferFunctions)
            GenicConfig->MakeGlassGas = 1;
        else
            GenicConfig->MakeGlassGas = 0;
    }
    GenicConfig->MakeGlassCDM = param_get_int(ps, "MakeGlassCDM");

    int64_t NumPartPerFile = param_get_int(ps, "NumPartPerFile");

    int64_t Ngrid = GenicConfig->Ngrid;
    if(Ngrid < GenicConfig->NgridGas)
        Ngrid = GenicConfig->NgridGas;
    GenicConfig->NumFiles = ( Ngrid*Ngrid*Ngrid + NumPartPerFile - 1) / NumPartPerFile;
    GenicConfig->NumWriters = param_get_int(ps, "NumWriters");
    if(GenicConfig->PowerP.DifferentTransferFunctions && GenicConfig->PowerP.InputPowerRedshift != Redshift
        && (GenicConfig->ProduceGas || CP->MNu[0] + CP->MNu[1] + CP->MNu[2]))
        message(0, "WARNING: Using different transfer functions but also rescaling power to account for linear growth. NOT what you want!\n");
    if((CP->MNu[0] + CP->MNu[1] + CP->MNu[2] > 0) || GenicConfig->PowerP.DifferentTransferFunctions || GenicConfig->PowerP.ScaleDepVelocity)
        if(0 == strlen(GenicConfig->PowerP.FileWithTransferFunction))
            endrun(0,"For massive neutrinos, different transfer functions, or scale dependent growth functions you must specify a transfer function file\n");
    if(!CP->RadiationOn && (CP->MNu[0] + CP->MNu[1] + CP->MNu[2] > 0))
        endrun(0,"You want massive neutrinos but no background radiation: this will give an inconsistent cosmology.\n");

    if(GenicConfig->Nmesh == 0) {
        GenicConfig->Nmesh = 2*Ngrid;
    }
    /*Set some units*/
    GenicConfig->TimeIC = 1 / (1 + Redshift);
    double UnitTime_in_s = GenicConfig->units.UnitLength_in_cm / GenicConfig->units.UnitVelocity_in_cm_per_s;

    CP->Hubble = HUBBLE * UnitTime_in_s;
}
