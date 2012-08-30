      subroutine compute_heating(Lstar, distance, temp, rho, 
     $                           heating_rate)
#ifdef CHEMCOOL
      implicit none
#include "cool.h"
      REAL Lstar, distance, temp, rho, heating_rate
      REAL kappa, flux, yn, specific_heating_rate
      REAL mp, pi
      parameter (mp = 1.6726d-24)
      parameter (pi = 3.14159265358979323846d0)
c
c Compute planck-mean kappa, given T, n   [cm^2 g^-1]
c
      yn = rho / ((1d0 + 4d0 * abhe) * mp)
crjs      call compute_md_kappa(temp, yn, kappa)
      call compute_md_kappa(temp, rho, kappa)
c
c Compute local bolometric flux, given source luminosity and distance
c NB. We assume for simplicity that the source radiates as a black-body
c
c [erg cm^-2 s^-1]
c
      flux = Lstar / (4d0 * pi * distance**2)
c
c Compute specific heating rate
c
      specific_heating_rate = flux * kappa
c
c Convert to volumetric heating rate
c
      heating_rate = rho * specific_heating_rate
c
#endif /* CHEMCOOL */
      return
      end
