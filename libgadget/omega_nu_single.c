#include "omega_nu_single.h"

#include <math.h>
#include <string.h>
#include "physconst.h"
#include "utils/mymalloc.h"
#include "utils/endrun.h"
#include "timefac.h"

#define HBAR    6.582119e-16  /*hbar in units of eV s*/
#define STEFAN_BOLTZMANN 5.670373e-5
/*Size of matter density tables*/
#define NRHOTAB 200
/** Floating point accuracy*/
#define FLOAT_ACC   1e-6

void init_omega_nu(_omega_nu * omnu, const double MNu[], const double a0, const double HubbleParam, const double tcmb0)
{
    int mi;
    /*Explicitly disable hybrid neutrinos*/
    omnu->hybnu.enabled=0;
    /*CMB temperature*/
    omnu->tcmb0 = tcmb0;
    /*Neutrino temperature times k_B*/
    omnu->kBtnu = BOLEVK * TNUCMB * tcmb0;
    /*Store conversion between rho and omega*/
    omnu->rhocrit = (3 * HUBBLE * HubbleParam * HUBBLE * HubbleParam)/ (8 * M_PI * GRAVITY);
    /*First compute which neutrinos are degenerate with each other*/
    for(mi=0; mi<NUSPECIES; mi++){
        int mmi;
        omnu->nu_degeneracies[mi]=0;
        for(mmi=0; mmi<mi; mmi++){
            if(fabs(MNu[mi] -MNu[mmi]) < FLOAT_ACC){
                omnu->nu_degeneracies[mmi]+=1;
                break;
            }
        }
        if(mmi==mi) {
            omnu->nu_degeneracies[mi]=1;
        }
    }
    /*Now allocate a table for the species we want*/
    for(mi=0; mi<NUSPECIES; mi++){
        if(omnu->nu_degeneracies[mi]) {
            rho_nu_init(&omnu->RhoNuTab[mi], a0, MNu[mi], omnu->kBtnu);
        }
        else
            omnu->RhoNuTab[mi].loga = 0;
    }
}

/* Return the total matter density in neutrinos.
 * rho_nu and friends are not externally callable*/
double get_omega_nu(const _omega_nu * const omnu, const double a)
{
        double rhonu=0;
        int mi;
        for(mi=0; mi<NUSPECIES; mi++) {
            if(omnu->nu_degeneracies[mi] > 0){
                 rhonu += omnu->nu_degeneracies[mi] * rho_nu(&omnu->RhoNuTab[mi], a, omnu->kBtnu);
            }
        }
        return rhonu/omnu->rhocrit;
}


/* Return the total matter density in neutrinos, excluding that in active particles.*/
double get_omega_nu_nopart(const _omega_nu * const omnu, const double a)
{
    double omega_nu = get_omega_nu(omnu, a);
    double part_nu = get_omega_nu(omnu, 1) * particle_nu_fraction(&omnu->hybnu, a, 0) / (a*a*a);
    return omega_nu - part_nu;
}

/*Return the photon density*/
double get_omegag(const _omega_nu * const omnu, const double a)
{
    const double omegag = 4*STEFAN_BOLTZMANN/(LIGHTCGS*LIGHTCGS*LIGHTCGS)*pow(omnu->tcmb0,4)/omnu->rhocrit;
    return omegag/pow(a,4);
}

/** Value of kT/aM_nu on which to switch from the
 * analytic expansion to the numerical integration*/
#define NU_SW 100

/*Note q carries units of eV/c. kT/c has units of eV/c.
 * M_nu has units of eV  Here c=1. */
double rho_nu_int(double q, void * params)
{
        double amnu = *((double *)params);
        double kT = *((double *)params+1);
        double epsilon = sqrt(q*q+amnu*amnu);
        double f0 = 1./(exp(q/kT)+1);
        return q*q*epsilon*f0;
}

/*Get the conversion factor to go from (eV/c)^4 to g/cm^3
 * for a **single** neutrino species. */
double get_rho_nu_conversion()
{
        /*q has units of eV/c, so rho_nu now has units of (eV/c)^4*/
        double convert=4*M_PI*2; /* The factor of two is for antineutrinos*/
        /*rho_nu_val now has units of eV^4*/
        /*To get units of density, divide by (c*hbar)**3 in eV s and cm/s */
        const double chbar=1./(2*M_PI*LIGHTCGS*HBAR);
        convert*=(chbar*chbar*chbar);
        /*Now has units of (eV)/(cm^3)*/
        /* 1 eV = 1.60217646 × 10-12 g cm^2 s^(-2) */
        /* So 1eV/c^2 = 1.7826909604927859e-33 g*/
        /*So this is rho_nu_val in g /cm^3*/
        convert*=(1.60217646e-12/LIGHTCGS/LIGHTCGS);
        return convert;
}

/*Seed a pre-computed table of rho_nu values for speed*/
void rho_nu_init(_rho_nu_single * const rho_nu_tab, double a0, const double mnu, const double kBtnu)
{
     int i;
     double abserr;
     /* Tabulate to 1e-3, unless earlier requested.
      * Need early times for growth function.*/
     if(a0 > 1e-3)
         a0 = 1e-3;
     /* Do not need a table when relativistic*/
     if(a0 * mnu < 1e-6 * kBtnu)
        a0 = 1e-6 * kBtnu / mnu;
     /*Make the table over a slightly wider range than requested, in case there is roundoff error*/
     const double logA0=log(a0)-log(1.2);
     const double logaf=log(NU_SW*kBtnu/mnu)+log(1.2);

     /*Initialise constants*/
     rho_nu_tab->mnu = mnu;
     /*Shortcircuit if we don't need to do the integration*/
     if(mnu < 1e-6*kBtnu || logaf < logA0)
         return;

     /*Allocate memory for arrays*/
     rho_nu_tab->loga = (double *) mymalloc("rho_nu_table",2*NRHOTAB*sizeof(double));
     rho_nu_tab->rhonu = rho_nu_tab->loga+NRHOTAB;
     if(!rho_nu_tab->loga)
         endrun(2035,"Could not initialise tables for neutrino matter density\n");

     for(i=0; i< NRHOTAB; i++){
        double param[2];
        rho_nu_tab->loga[i]=logA0+i*(logaf-logA0)/(NRHOTAB-1);
        param[0]=mnu*exp(rho_nu_tab->loga[i]);
        param[1] = kBtnu;

        // Define the integrand for rho_nu_int
        auto integrand = [param](double q) {
            return rho_nu_int(q, (void *)param);
        };

        // Perform the Tanh-Sinh adaptive integration
        double result = tanh_sinh_integrate_adaptive(integrand, 0, 500 * kBtnu, &abserr, 1e-9);

        rho_nu_tab->rhonu[i] = result / pow(exp(rho_nu_tab->loga[i]), 4) * get_rho_nu_conversion();
     }

     rho_nu_tab->interp = new boost::math::interpolators::barycentric_rational<double>(rho_nu_tab->loga, rho_nu_tab->rhonu, NRHOTAB);
     return;
}

/*Heavily non-relativistic*/
static inline double non_rel_rho_nu(const double a, const double kT, const double amnu, const double kTamnu2)
{
    /*The constants are Riemann zetas: 3,5,7 respectively*/
    return amnu*(kT*kT*kT)/(a*a*a*a)*(1.5*1.202056903159594+kTamnu2*45./4.*1.0369277551433704+2835./32.*kTamnu2*kTamnu2*1.0083492773819229+80325/32.*kTamnu2*kTamnu2*kTamnu2*1.0020083928260826)*get_rho_nu_conversion();
}

/*Heavily relativistic: we could be more accurate here,
 * but in practice this will only be called for massless neutrinos, so don't bother.*/
static inline double rel_rho_nu(const double a, const double kT)
{
    return 7*pow(M_PI*kT/a,4)/120.*get_rho_nu_conversion();
}

/*Finds the physical density in neutrinos for a single neutrino species
  1.878 82(24) x 10-29 h02 g/cm3 = 1.053 94(13) x 104 h02 eV/cm3*/
double rho_nu(const _rho_nu_single * rho_nu_tab, const double a, const double kT)
{
        double rho_nu_val;
        double amnu=a*rho_nu_tab->mnu;
        const double kTamnu2=(kT*kT/amnu/amnu);
        /*Do it analytically if we are in a regime where we can
         * The next term is 141682 (kT/amnu)^8.
         * At kT/amnu = 8, higher terms are larger and the series stops converging.
         * Don't go lower than 50 here. */
        if(NU_SW*NU_SW*kTamnu2 < 1){
            /*Heavily non-relativistic*/
            rho_nu_val = non_rel_rho_nu(a, kT, amnu, kTamnu2);
        }
        else if(amnu < 1e-6*kT){
            /*Heavily relativistic*/
            rho_nu_val=rel_rho_nu(a, kT);
        }
        else{
            const double loga = log(a);
            /* Deal with early time case. In practice no need to be very accurate
             * so assume relativistic.*/
            if (!rho_nu_tab->loga || loga < rho_nu_tab->loga[0])
                rho_nu_val = rel_rho_nu(a,kT);
            else
                rho_nu_val=(*rho_nu_tab->interp)(loga);
        }
        return rho_nu_val;
}

/*The following function definitions are only used for hybrid neutrinos*/

/*Fermi-Dirac kernel for below*/
double fermi_dirac_kernel(double x, void * params)
{
  return x * x / (exp(x) + 1);
}

/* Fraction of neutrinos not followed analytically
 * This is integral f_0(q) q^2 dq between 0 and qc to compute the fraction of OmegaNu which is in particles.*/
double nufrac_low(const double qc)
{
    double abserr;
    // Define the integrand for Fermi-Dirac kernel
    auto integrand = [](double x) {
        return fermi_dirac_kernel(x, NULL);
    };

    // Use Tanh-Sinh adaptive integration for the Fermi-Dirac kernel
    double total_fd = tanh_sinh_integrate_adaptive(integrand, 0, qc, &abserr, 1e-6);
    /*divided by the total F-D probability (which is 3 Zeta(3)/2 ~ 1.8 if MAX_FERMI_DIRAC is large enough*/
    total_fd /= 1.5*1.202056903159594;

    return total_fd;
}

void init_hybrid_nu(_hybrid_nu * const hybnu, const double mnu[], const double vcrit, const double light, const double nu_crit_time, const double kBtnu)
{
    hybnu->enabled=1;
    int i;
    hybnu->nu_crit_time = nu_crit_time;
    hybnu->vcrit = vcrit / light;
    for(i=0; i< NUSPECIES; i++) {
        const double qc = mnu[i] * vcrit / light / kBtnu;
        hybnu->nufrac_low[i] = nufrac_low(qc);
    }
}

/* Returns the fraction of neutrinos currently traced by particles.
 * When neutrinos are fully analytic at early times, returns 0.
 * Last argument: neutrino species to use.
 */
double particle_nu_fraction(const _hybrid_nu * const hybnu, const double a, const int i)
{
    /*Return zero if hybrid neutrinos not enabled*/
    if(!hybnu->enabled)
        return 0;
    /*Just use a redshift cut for now. Really we want something more sophisticated,
     * based on the shot noise and average overdensity.*/
    if (a > hybnu->nu_crit_time){
        return hybnu->nufrac_low[i];
    }
    return 0;
}

/*End functions used only for hybrid neutrinos*/

/* Return the matter density in neutrino species i.*/
double omega_nu_single(const _omega_nu * const omnu, const double a, int i)
{
    /*Deal with case where we want a species degenerate with another one*/
    if(omnu->nu_degeneracies[i] == 0) {
        int j;
        for(j=i; j >=0; j--)
            if(omnu->nu_degeneracies[j]){
                i = j;
                break;
            }
    }
    double omega_nu = rho_nu(&omnu->RhoNuTab[i], a,omnu->kBtnu)/omnu->rhocrit;
    double omega_part = rho_nu(&omnu->RhoNuTab[i], 1,omnu->kBtnu)/omnu->rhocrit;
    omega_part *= particle_nu_fraction(&omnu->hybnu, a, i)/(a*a*a);
    omega_nu -= omega_part;
    return omega_nu;

}
