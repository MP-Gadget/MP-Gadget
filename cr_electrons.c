#ifdef JD_DPP

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_sf_expint.h>

#include "allvars.h"
#include "proto.h"

#define FOURPI 12.56637096405029296875

#define COMOV 1 
#define NOCOMOV All.Time
#define GADGET 0
#define PHYSICAL 1
#define h All.HubbleParam

#define YHELIUM (1-HYDROGEN_MASSFRAC)/(4*HYDROGEN_MASSFRAC)
#define MEAN_MOL_WEIGHT (1+4*YHELIUM)/(1+3*YHELIUM+1)
#define N2NE (HYDROGEN_MASSFRAC+0.5*(1-HYDROGEN_MASSFRAC))/(2*HYDROGEN_MASSFRAC+0.75*(1-HYDROGEN_MASSFRAC))
#define UMU 4.0/(5*HYDROGEN_MASSFRAC+3.)
#define FOURTHIRDSPI  4.188790204786
#define KPC2CM 3.08568025E+21

#define ETA_T 0.2
#define TURBULENCEINJECTIONSCALE 200    /* [kpc] */
#define TURBULENCEDAMPINGSCALE 0.1      /* [kpc] */

#define DPP_FACTOR  1.31145005476422225952E+11
/* = 4.45 PI^2/sqrt(32 PI^3)/c/sqrt(me*kb) */

static double I_x(size_t);
void compute_Dpp(size_t ipart);

/* Helper functions */
static double help_density(size_t,double,int);
static double help_pressure(size_t,double,int);
static double help_internal_energy(size_t,double,int);
static double help_temperature(size_t,double,int);
static double help_speed_of_sound(size_t,double,int);
static double help_electron_number_density(size_t,double,int);

/* Update Dpp/p^2 quantity: SphP.Dpp for particle  
 * according to Cassano & Brunetti 04 */
void compute_Dpp(size_t ipart)
{
    double n_th,T,dt,m_part,v_turb,hsml,
           scalefactor,Eturb,Vpart,d_mps,
           k_min,k_max,k_sml,k_mps;

	if(!ThisTask)
		printf("\nComputing Dpp/p^2 \neta_t = %g \n",ETA_T);

#ifdef JD_DPPONSNAPSHOTONLY /* Here we need a particle loop and discard the input parameter */
	for( ipart=0; ipart<N_gas; ipart++ ){ /* Physical units */
#endif        
        hsml = SphP[ipart].Hsml*All.UnitLength_in_cm
            *All.Time/All.HubbleParam;
        m_part = P[ipart].Mass * All.UnitMass_in_g
            /All.HubbleParam;
        v_turb = SphP[ipart].Vrms 
            * All.UnitVelocity_in_cm_per_s*sqrt(All.Time);
        dt = All.TimeStep * All.UnitTime_in_s
            * sqrt(All.Time)/All.HubbleParam;
        
        n_th = help_electron_number_density(ipart,NOCOMOV,PHYSICAL);
        T = help_temperature(ipart,NOCOMOV,PHYSICAL);
        
        Eturb = 0.5 * m_part * v_turb*v_turb;
        
        d_mps = hsml/pow(SphP[ipart].TrueNGB,1./3.);
        Vpart = FOURTHIRDSPI * d_mps*d_mps*d_mps;  

        k_min = 2*M_PI/(TURBULENCEINJECTIONSCALE*KPC2CM);
        k_max = 2*M_PI/(TURBULENCEDAMPINGSCALE*KPC2CM);
        
        k_mps = 2*M_PI/d_mps;
        k_sml = M_PI/hsml;   /* hsml is only half the scale */

        scalefactor = (pow(k_max,-2./3.)-pow(k_min,-2./3.))
                     /(pow(k_mps,-2./3.)-pow(k_sml,-2./3.));

		SphP[ipart].Dpp =  DPP_FACTOR /I_x(ipart) /n_th /sqrt(T)
            * ETA_T * Eturb *scalefactor /dt /Vpart;
#ifdef JD_DPPONSNAPSHOTONLY
	}
#endif
	
	return;
}

/* integral eqn.21 from Cassano & Brunetti 04 */
static double I_x(size_t ipart)
{
	double vms, vth, vsnd, valven; 
	double x, B, T, rho; 
	
	B = sqrt(SphP[ipart].B[0]*SphP[ipart].B[0]
			+SphP[ipart].B[1]*SphP[ipart].B[1]
			+SphP[ipart].B[2]*SphP[ipart].B[2]);

	T = help_temperature(ipart,NOCOMOV,PHYSICAL);

	rho = help_density(ipart,NOCOMOV,PHYSICAL);

	vsnd = help_speed_of_sound(ipart,NOCOMOV,PHYSICAL);
	valven = B/sqrt(FOURPI*help_density(ipart,NOCOMOV,PHYSICAL));
	vms = sqrt(4./3. * vsnd*vsnd + valven*valven); /* eqn.34 */
	vth = sqrt(2*BOLTZMANN/ELECTRONMASS*T);

	x = vms*vms/(vth*vth);
 
    if(x < 50 ){
        return((1+x) * gsl_sf_expint_E1(x) - exp(-x)); 
    } else {
        return(0);
    }
}

/* Unit System:
 * g ~ 1/h
 * cm ~ a/h
 * s ~ sqrt(a)/h */

static double help_density(size_t ipart, double a, int unit)
{
	switch(unit){ 
	case GADGET:
		return(SphP[ipart].d.Density/(a*a*a));
	case PHYSICAL:
		return(SphP[ipart].d.Density/(a*a*a)*h*h
			*All.UnitMass_in_g/pow(All.UnitLength_in_cm,3));
	default:
		endrun(10100);
        return(-1);
	}
}

/* P = S*rho^gamma */
static double help_pressure(size_t ipart, double a, int unit)
{
	switch(unit){ 
	case GADGET:
		return(SphP[ipart].Pressure*pow(a,3*GAMMA-6));
	case PHYSICAL:
		return(SphP[ipart].Pressure * GAMMA_MINUS1 * pow(a,3*GAMMA-6)
                * h*h * All.UnitPressure_in_cgs);
	default:
		endrun(10101);
        return(-1);
	}
}
/* U = S/(gamma-1)*rho^(gamma-1) */
static double help_internal_energy(size_t ipart, double a, int unit)
{

    switch(unit){  
	case GADGET:
        return(SphP[ipart].Entropy/GAMMA_MINUS1  
                * pow(SphP[ipart].d.Density/(a*a*a),GAMMA_MINUS1));
    case PHYSICAL:
        return(SphP[ipart].Entropy/GAMMA_MINUS1  
                * pow(SphP[ipart].d.Density/(a*a*a),GAMMA_MINUS1) 
                * All.UnitEnergy_in_cgs/All.UnitMass_in_g);
    default:
        endrun(10102);
        return(-1);
	}
	
}
/* [K] always */ 
static double help_temperature(size_t ipart, double a, int unit)
{   return(SphP[ipart].Entropy * pow(SphP[ipart].d.Density*a*a*a,GAMMA_MINUS1)
        * PROTONMASS * MEAN_MOL_WEIGHT / BOLTZMANN * All.UnitEnergy_in_cgs / All.UnitMass_in_g);
}

/* And birds go flying at the speed of sound - Coldplay*/
static double help_speed_of_sound(size_t ipart, double a, int unit)
{
    switch(unit){ 
	case GADGET:
	    return(sqrt(GAMMA * SphP[ipart].Entropy * sqrt(a)
		    * pow(SphP[ipart].d.Density,GAMMA_MINUS1)));
    case PHYSICAL:
        return(sqrt(GAMMA * SphP[ipart].Entropy * sqrt(a)
		    * pow(SphP[ipart].d.Density,GAMMA_MINUS1)) 
            * All.UnitVelocity_in_cm_per_s);
    default:
        endrun(10103);
        return(-1);
    }
}

static double help_electron_number_density(size_t ipart, double a, int unit)
{
    switch(unit){  
	case GADGET:
	    return(help_density(ipart,a,PHYSICAL)  /(a*a*a)
                * N2NE/(UMU*PROTONMASS) * pow(All.UnitLength_in_cm,3));
    case PHYSICAL:
        return(help_density(ipart,a,PHYSICAL)  /(a*a*a) * N2NE/(UMU*PROTONMASS));
    default:
        endrun(10104);
        return(-1);
    }

}

#undef ETA_T
#undef TURBULENCEINJECTIONSCALE
#undef TURBULENCEDAMPINGSCALE

#undef COMOV
#undef NOCOMOV
#undef GADGET
#undef PHYSICAL
#undef h

#undef FOURPI
#undef FOURTHIRDSPI
#undef KPC2CM
#undef YHELIUM
#undef MEAN_MOL_WEIGHT
#undef N2NE
#undef UMU


#endif 
