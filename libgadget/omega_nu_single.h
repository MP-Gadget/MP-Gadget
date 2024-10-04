#ifndef OMEGA_NU_SINGLE_H
#define OMEGA_NU_SINGLE_H
/** \file
 * Routines for computing the matter density in a single neutrino species*/

// Undefine P before including Boost
#ifdef P
#undef P
#endif
#include <boost/math/interpolators/barycentric_rational.hpp>

/** Ratio between the massless neutrino temperature and the CMB temperature.
 * Note there is a slight correction from 4/11
 * due to the neutrinos being slightly coupled at e+- annihilation.
 * See Mangano et al 2005 (hep-ph/0506164)
 * We use the CLASS default value, chosen so that omega_nu = m_nu / 93.14 h^2
 * At time of writing this is T_nu / T_gamma = 0.71611.
 * See https://github.com/lesgourg/class_public/blob/master/explanatory.ini
 */
#define TNUCMB     (pow(4/11.,1/3.)*1.00328)
/** Number of massive neutrino species: 3
 * Could be made configurable at some point
 * Neutrino masses are in eV*/
#define NUSPECIES 3

/** Tables for rho_nu (neutrino density): stores precomputed values between
 * simulation start and a M_nu = 20 kT_nu for a single neutrino species.*/
struct _rho_nu_single {
    double * loga;
    double * rhonu;
    boost::math::interpolators::barycentric_rational<double>* interp;
    /*Neutrino mass for this structure*/
    double mnu;
};
typedef struct _rho_nu_single _rho_nu_single;

/** Initialise the tables for matter density in a single neutrino species, by doing numerical integration.
 * @param rho_nu_tab Structure to initialise
 * @param a0 First scale factor in rho_nu_tab. Do not try and evaluate rho_nu at higher redshift!
 * @param mnu neutrino mass to compute neutrino matter density for in eV
 * @param kBtnu Boltzmann constant times neutrino temperature. Dimensionful factor. */
void rho_nu_init(_rho_nu_single * rho_nu_tab, double a0, const double mnu, const double kBtnu);

/** Computes the neutrino density for a single neutrino species at a given redshift, either by looking up in a table,
 * or a simple calculation in the limits, or by direct integration.
 * @param rho_nu_tab Pre-computed table of values
 * @param a Redshift desired.
 * @param kT Boltzmann constant times neutrino temperature. Dimensionful factor.
 * @returns neutrino density in g cm^-3 */
double rho_nu(const _rho_nu_single * rho_nu_tab, const double a, const double kT);

/** \section Hybrid
 * Hybrid Neutrinos: The following functions and structures are used for hybrid neutrinos only.*/

/** Structure storing parameters related to the hybrid neutrino code.*/
struct _hybrid_nu {
    /*True if hybrid neutrinos are enabled*/
    int enabled;
    /* This is the fraction of neutrino mass not followed by the analytic integrator.
    The analytic method is cutoff at q < qcrit (specified using vcrit, below) and use
    particles for the slower neutrinos.*/
    double nufrac_low[NUSPECIES];
    /* Time at which to turn on the particle neutrinos.
     * Ultimately we want something better than this.*/
    double nu_crit_time;
    /*Critical velocity as a fraction of lightspeed*/
    double vcrit;
};
typedef struct _hybrid_nu _hybrid_nu;

/**Set up parameters for the hybrid neutrinos
 * @param hybnu initialised structure
 * @param mnu array of neutrino masses in eV
 * @param vcrit Critical velocity above which to treat neutrinos with particles.
 *   Note this is unperturbed velocity *TODAY*
 *   To get velocity at redshift z, multiply by (1+z)
 * @param light speed of light in internal units
 * @param nu_crit_time critical time to make neutrino particles live
 * @param kBtnu Boltzmann constant times neutrino temperature. Dimensionful factor.*/
void init_hybrid_nu(_hybrid_nu * const hybnu, const double mnu[], const double vcrit, const double light, const double nu_crit_time, const double kBtnu);

/** Get fraction of neutrinos currently followed by particles.
 * @param hybnu Structure with hybrid neutrino parameters.
 * @param i index of neutrino species to use.
 * @param a redshift of interest.
 * @returns the fraction of neutrinos currently traced by particles.
 * 0 when neutrinos are fully analytic at early times.
 */
double particle_nu_fraction(const _hybrid_nu * const hybnu, const double a, int i);

/** Integrate the fermi-dirac kernel between 0 and qc to find the fraction of neutrinos that are particles*/
double nufrac_low(const double qc);

/**End hybrid neutrino-only structures.*/

/**\section External
 * Externally callable functions.*/
/** Structure containing cosmological parameters related to the massive neutrinos*/
struct _omega_nu {
    /*Pointers to the array of structures we use to store rho_nu*/
    _rho_nu_single RhoNuTab[NUSPECIES];
    /* Which species have the same mass and can thus be counted together.*/
    int nu_degeneracies[NUSPECIES];
    /* Prefactor to turn density into matter density omega*/
    double rhocrit;
    /*neutrino temperature times Boltzmann constant*/
    double kBtnu;
    /*CMB temperature*/
    double tcmb0;
    /* Pointer to structure for hybrid neutrinos. */
    _hybrid_nu hybnu;
};
typedef struct _omega_nu _omega_nu;

/**Initialise the neutrino structure, do the time integration and allocate memory for the subclass rho_nu_single
 * @param omnu structure to initialise
 * @param MNu array of neutrino masses in eV. Three entries.
 * @param a0 initial scale factor.
 * @param HubbleParam Hubble parameter h0, eg, 0.7.
 * @param tcmb0 Redshift zero CMB temperature.*/
void init_omega_nu(_omega_nu * const omnu, const double MNu[], const double a0, const double HubbleParam, const double tcmb0);

/** Return the total matter density in neutrinos at scale factor a.*/
double get_omega_nu(const _omega_nu * const omnu, const double a);

/** Return the total matter density in neutrinos at scale factor a , excluding active particles.*/
double get_omega_nu_nopart(const _omega_nu * const omnu, const double a);

/** Return the photon matter density at scale factor a*/
double get_omegag(const _omega_nu * const omnu, const double a);

/** Return the matter density in a single neutrino species
 * @param rho_nu_tab structure containing pre-computed matter density values.
 * @param i index of neutrino species we want
 * @param a scale factor desired*/
double omega_nu_single(const _omega_nu * const rho_nu_tab, const double a, const int i);

#endif
