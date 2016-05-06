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
            hubble_a += All.OmegaNu / (a * a * a * a);
    }

    /* Now finish it up. */
    hubble_a = All.Hubble * sqrt(hubble_a);
    return (hubble_a);
}
