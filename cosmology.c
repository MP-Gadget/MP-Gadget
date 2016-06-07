#include <math.h>
#include "allvars.h"
#include "cosmology.h"



/*Hubble function at scale factor a, in dimensions of All.Hubble*/
double hubble_function(double a)
{

    double hubble_a;

    /* first do the terms in SQRT */
    hubble_a = All.OmegaLambda;

    hubble_a += All.OmegaK / (a * a);
    hubble_a += All.Omega0 / (a * a * a);

    if(All.RadiationOn) {
        hubble_a += All.OmegaG / (a * a * a * a);
        /* massless neutrinos are added only if there is no (massive) neutrino particle.*/
        if(!All.TotN_neutrinos)
            hubble_a += All.OmegaNu0 / (a * a * a * a);
    }

    /* Now finish it up. */
    hubble_a = All.Hubble * sqrt(hubble_a);
    return (hubble_a);
}

static double growth(double a);
static double growth_int(double a, void * params);

double GrowthFactor(double astart)
{
    return growth(astart) / growth(1.0);
}


static double growth(double a)
{
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (200);
    double hubble_a;
    double result,abserr;
    gsl_function F;
    F.function = &growth_int;

    hubble_a = hubble_function(a);

    gsl_integration_qag (&F, 0, a, 0, 1e-4,200,GSL_INTEG_GAUSS61, w,&result, &abserr);
    //   printf("gsl_integration_qng in growth. Result %g, error: %g, intervals: %lu\n",result, abserr,w->size);
    gsl_integration_workspace_free (w);
    return hubble_a * result;
}


static double growth_int(double a, void * params)
{
    if(a == 0) return 0;
    return pow(1 / (a * hubble_function(a)), 3);
}
