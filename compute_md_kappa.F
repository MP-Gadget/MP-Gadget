      subroutine compute_md_kappa(temp, rho, kappa)
#ifdef CHEMCOOL
      implicit none
#include "cool.h"
#include "kappa-planck.h"
      REAL temp, yn, kappa
      REAL rho, logT, dT, logrho, rho_offset
      REAL drho, k1, k2, k3, k4, ka, kb
      integer iT, irho, idx, idx1, idx2, idx3, idx4
      REAL mp
      parameter (mp = 1.6726d-24)

crjs      rho = (1d0 + 4d0 * abhe) * mp * yn


      logT = log10(temp)
      iT = int((logT - 1.8) / 0.1) + 1
      if (iT .lt. 1) then
c At low T, use the opacities for log(T) = 1.8, in absence of anything
c better. NB 10**1.8 ~ 63K, so we probably won't have gas falling off the
c bottom of the table in the pop III case
        iT = 1
        dT = 0d0
      elseif (iT .ge. 29) then
c At high T, use upper limit in table
        iT = 29
        dT = 0d0
      else
        dT = (logT - (0.1*(iT-1) + 1.8)) / 0.1
      endif

      logrho = log10(rho)
      irho = int(logrho + 16.00) + 1
      rho_offset = 0d0

      if (irho .lt. 1) then
c At v. low density, assume constant opacity for log(T) < 3.1, 
c opacity ~ density at higher T. This isn't quite right, but should
c be accurate to within a factor of a few
        irho    = 1
        drho    = 0d0
        if (iT .gt. 14) then
          rho_offset = logrho + 16
        endif
      elseif (irho .ge. 15) then
c Keep constant if log(T) > 3.8, otherwise scale with density
        irho = 15
        drho = 0d0
        if (iT .le. 21) then
          rho_offset = logrho + 2
        endif
      else
crjs        drho = -(logrho - ((irho-1) - 16.))
         drho = (logrho - ((irho-1) - 16.))
      endif
c 
c Now do lookup
c
      if (dT .eq. 0d0) then
        if (drho .eq. 0d0) then
          idx = 15 * (iT - 1) + irho            
          kappa = kappa_p(idx) + rho_offset
        else
          idx1 = 15 * (iT - 1) + irho
          idx2 = idx1 + 1
          k1 = kappa_p(idx1)
          k2 = kappa_p(idx2)
          kappa = k1 + drho * (k2 - k1)
        endif
      else
        if (drho .eq. 0d0) then
          idx1 = 15 * (iT - 1) + irho            
          idx2 = 15 * iT + irho
          k1 = kappa_p(idx1)
          k2 = kappa_p(idx2)
          kappa = k1 + dT * (k2 - k1) + rho_offset
        else
          idx1 = 15 * (iT - 1) + irho
          idx2 = idx1 + 1
          k1 = kappa_p(idx1)
          k2 = kappa_p(idx2)
          ka = k1 + drho * (k2 - k1)
c
          idx3 = idx1 + 15
          idx4 = idx3 + 1
          k3 = kappa_p(idx3)
          k4 = kappa_p(idx4)
          kb = k3 + drho * (k4 - k3)
c
          kappa = ka + dT * (kb - ka)
        endif
      endif

c Finally, convert from log -> actual value
      kappa = 1d1**kappa

#endif /* CHEMCOOL */
      return
      end
