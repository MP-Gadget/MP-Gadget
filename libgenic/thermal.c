#include <gsl/gsl_integration.h>
#include <assert.h>
#include "thermal.h"
/*For speed of light*/
#include <libgadget/physconst.h>
#include <libgadget/utils.h>
#include <libgadget/timefac.h>

/*The Boltzmann constant in units of eV/K*/
#define BOLEVK 8.61734e-5

/* This function converts the dimensionless units used in the integral to dimensionful units.
 * Unit scaling velocity for neutrinos:
 * This is an arbitrary rescaling of the unit system in the Fermi-Dirac kernel so we can integrate dimensionless quantities.
 * The true thing to integrate is:
 * q^2 /(e^(q c / kT) + 1 ) dq between 0 and q.
 * So we choose x = (q c / kT_0) and integrate between 0 and x_0.
 * The units are restored by multiplying the resulting x by kT/c for q
 * To get a v we then use q = a m v/c^2
 * to get:   v/c =x kT/(m a)*/
/*NOTE: this m is the mass of a SINGLE neutrino species, not the sum of neutrinos!*/
double
NU_V0(const double Time, const double kBTNubyMNu, const double UnitVelocity_in_cm_per_s)
{
    return kBTNubyMNu / Time * (LIGHTCGS / UnitVelocity_in_cm_per_s);
}

//Amplitude of the random velocity for WDM
double WDM_V0(const double Time, const double WDM_therm_mass, const double Omega_CDM, const double HubbleParam, const double UnitVelocity_in_cm_per_s)
{
        //Not actually sure where this equation comes from: the fiducial values are from Bode, Ostriker & Turok 2001.
        double WDM_V0 = 0.012 / Time * pow(Omega_CDM / 0.3, 1.0 / 3) * pow(HubbleParam / 0.65, 2.0 / 3) * pow(1.0 /WDM_therm_mass,4.0 / 3);
        WDM_V0 *= 1.0e5 / UnitVelocity_in_cm_per_s;
        return WDM_V0;
}

/*Fermi-Dirac kernel for below*/
static double
fermi_dirac_kernel(double x, void * params)
{
  return x * x / (exp(x) + 1);
}

/*Initialise the probability tables*/
double
init_thermalvel(struct thermalvel* thermals, const double v_amp, double max_fd,const double min_fd)
{
    int i;
    if(max_fd <= min_fd)
        endrun(1,"Thermal vel called with negative interval: %g <= %g\n", max_fd, min_fd);

    if(max_fd > MAX_FERMI_DIRAC)
        max_fd = MAX_FERMI_DIRAC;
    thermals->m_vamp = v_amp;

    double abserr;

    // Lambda function wrapping the Fermi-Dirac kernel
    auto integrand = [](double x) {
        return fermi_dirac_kernel(x, nullptr);
    };

    for(i = 0; i < LENGTH_FERMI_DIRAC_TABLE; i++) {
        thermals->fermi_dirac_vel[i] = min_fd+(max_fd-min_fd)* i / (LENGTH_FERMI_DIRAC_TABLE - 1.0);
        thermals->fermi_dirac_cumprob[i] = tanh_sinh_integrate_adaptive(integrand, min_fd, thermals->fermi_dirac_vel[i], &abserr, 1e-6, 0.);
    //       printf("gsl_integration_qng in fermi_dirac_init_nu. Result %g, error: %g, intervals: %lu\n",fermi_dirac_cumprob[i], abserr,w->size);
    }
    /*Save the largest cum. probability, pre-normalisation,
     * divided by the total F-D probability (which is 3 Zeta(3)/2 ~ 1.8 if MAX_FERMI_DIRAC is large enough*/
    double total_fd;
    total_fd = tanh_sinh_integrate_adaptive(integrand, 0, MAX_FERMI_DIRAC, &abserr, 1e-6, 0.);
    assert(total_fd > 1.8);

    double total_frac = thermals->fermi_dirac_cumprob[LENGTH_FERMI_DIRAC_TABLE-1]/total_fd;
    //Normalise total integral to unity
    for(i = 0; i < LENGTH_FERMI_DIRAC_TABLE; i++)
        thermals->fermi_dirac_cumprob[i] /= thermals->fermi_dirac_cumprob[LENGTH_FERMI_DIRAC_TABLE - 1];

    /*Initialise the GSL table*/
    thermals->fd_intp = gsl_interp_alloc(gsl_interp_cspline,LENGTH_FERMI_DIRAC_TABLE);
    thermals->fd_intp_acc = gsl_interp_accel_alloc();
    gsl_interp_init(thermals->fd_intp,thermals->fermi_dirac_cumprob, thermals->fermi_dirac_vel,LENGTH_FERMI_DIRAC_TABLE);
    return total_frac;
}

/*Generate a table of random seeds, one for each pencil.*/
unsigned int *
init_rng(int Seed, int Nmesh)
{
    unsigned int * seedtable = (unsigned int *) mymalloc("randseeds", Nmesh*Nmesh*sizeof(unsigned int));
    gsl_rng * rng = gsl_rng_alloc(gsl_rng_ranlxd1);
    gsl_rng_set(rng, Seed);

    int i, j;
    for(i = 0; i < Nmesh; i++)
        for(j=0; j < Nmesh; j++)
        {
            seedtable[i+Nmesh*j] = gsl_rng_get(rng);
        }
    gsl_rng_free(rng);
    return seedtable;
}

/* Add a randomly generated thermal speed in v_amp*(min_fd, max_fd) to a 3-velocity.
 * The particle Id is used as a seed for the RNG.*/
void
add_thermal_speeds(struct thermalvel * thermals, gsl_rng *g_rng, float Vel[])
{
    const double p = gsl_rng_uniform (g_rng);
    /*m_vamp multiples by the dimensional factor to get a velocity again.*/
    const double v = thermals->m_vamp * gsl_interp_eval(thermals->fd_intp,thermals->fermi_dirac_cumprob, thermals->fermi_dirac_vel, p, thermals->fd_intp_acc);

    /*Random phase*/
    const double phi = 2 * M_PI * gsl_rng_uniform (g_rng);
    const double theta = acos(2 * gsl_rng_uniform (g_rng) - 1);

    Vel[0] += v * sin(theta) * cos(phi);
    Vel[1] += v * sin(theta) * sin(phi);
    Vel[2] += v * cos(theta);
}