#if CHEMISTRYNETWORK == 2 || CHEMISTRYNETWORK == 3 || CHEMISTRYNETWORK == 7 || CHEMISTRYNETWORK == 11
#define NEEDS_CO_RATES
#endif

c=======================================================================
c
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////          C A L C _ P H O T O              \\\\\\\\\\
c
c=======================================================================
c
c    Written by S. Glover (2007)
c
c  REFERENCES:
c
c    L96    -- Lee et al, 1996, A&A, 311, 690
c    BHRB04 -- Bergin et al, 2004, ApJ, 612, 921
c
c iphoto == 0:  Gas assumed optically thin
c
c iphoto == 1:  Local approximation for shielding within the simulation 
c volume. We only account for material within a smoothing length. Optical 
c depth effects due to H2 line absorption (self-shielding) and dust 
c continuum absorption are tackled separately.
c
c Our treatment of self-shielding is rather approximate. We express its
c effect on the photodissociation rate in terms of a simple self-shielding
c function, fshield, the form of which is taken from Draine & Bertoldi
c (1996, ApJ, 468, 269; eq.37). This requires two input parameters: the
c Doppler parameter b and the H2 column density NH2.
c
c For iphoto .eq. 1, we estimate these under the assumption that only gas
c within the current grid cell contributes to the self-shielding. Larger
c values of iphoto are reserved for other methods.
c
c Treatment of dust attenuation is also based on Draine & Bertoldi (1996):
c we take the value of sigma_d at 1000 Angstroms as a reasonable approx.
c for the value throughout the Lyman-Werner bands, and use the value of
c sigma_d quoted by Draine & Bertoldi (1996) as appropriate for gas with
c R_V == A_V / E(B-V) = 3.1
c
c We also calculate the visual extinction, A_V, appropriate to this value
c of R_V, using the conversion from NH to E(B-V) given in Draine & Bertoldi
c (see also cool_func.F). This is not used to compute the H2 photodissociation
c rate, but is used to compute other dissociation rates below
c
c iphoto == 2: Six-ray approximation - column densities of dust, H2, CO
c are computed along six directions in box defined by coordinate axes.
c For each ray, partial rates are calculated (computing H2 shielding, dust
c etc. as above) and then summed appropriately to give the final rate.
c
c iphoto == 3: Crude approximation for spherically symmetric collapse - we
c assume that the column density of species X_i is given by n_i * r, where
c n_i is the local density and r is the distance to the center of the sphere
c
c iphoto == 4: Better approximation for the spherically symmetric collapse:
c Uses the potential of the gas particle to calculate the enclosed mass,  
c and then from mass conservation get the mass exterior to the particle. 
c The column density
c is then just taken to be the exterior mass divided by the area of the shell
c at the particle's radius (+ and external extinction).
c  	Sigma(r) ~ (4*pi*r^2)^-1 * [ phi(r)*r/G - Mcloud ] 
c
c iphoto == 5: Best spherical estimate using a radial density profile calculated
c in density_profile(.c). Each particle has its column density intergrated
c from its radial location. 
c

      subroutine calc_photo(temp, rpar, abh2, abhd, abco, phrates, 
     $                      G_dust, chi_isrf)
#ifdef CHEMCOOL
      implicit NONE
#include "cool.h"
#include "non_eq.h"
#include "shield_data.h"

      REAL rpar(nrpar), phrates(npr)
      REAL yn, dl, abh2, abhd, abco, temp

#ifdef RAYTRACE
      REAL col_tot(6), col_H2(6)
#if defined CO_SHIELDING && defined NEEDS_CO_RATES
      REAL col_CO(6)
#endif /* CO_SHIELDING && NEEDS_CO_RATES */
      REAL sigma_dust
#endif /* RAYTRACE */

      REAL fshield, f_dust, NH, fshield_CO,
     $     G_dust, b, b5, NH2, h2dis0, h2dis1,
     $     AV, f_simp, localapprox, chi_isrf, chi

      external chi

#if defined CO_SHIELDING && defined NEEDS_CO_RATES
      integer ipos
      REAL theta_H2, theta_CO, dth, DNact, NCO
#endif

      REAL fshield_HD, NHD, hddis0, hddis1

      REAL df05, df09, df12, df17, df18, df19, df21, df23,
     $     df24, df25, df28, df30, df31, df37, df39, dfsp1

      integer I

      REAL column_est

      REAL mh, GG
      parameter (mh  = PROTONMASS)
      parameter (GG  = 6.6726d-08)

#ifdef DEBUG_CALC_PHOTO
c Seed with very large init. values, to better pick up access
c to uninitialized rates
      do I = 1, npr
        phrates(I) = 1d20
      enddo
      G_dust = 1d20
#endif
c
      if (G0 .eq. 0) then
        do I = 1, npr
          phrates(I) = 0d0
        enddo
        G_dust = 0d0
      endif
c
c Unpack parameters
c
      yn    = rpar(1)
      dl    = rpar(2)
#ifdef RAYTRACE
      do I = 1, 6
        col_tot(I) = rpar(I+3)
        col_H2(I)  = rpar(I+9)
#if defined CO_SHIELDING && defined NEEDS_CO_RATES
        col_CO(I)   = rpar(I+15)
#endif /* CO_SHIELDING && NEEDS_CO_RATES */
      enddo
#else  /* RAYTRACE */
c
c If raytrace not on, then can use these for the method that uses the potential.
c
      column_est = rpar(4)
#ifdef DEBUGDUSTTEMP
      if (yn.gt.1e5) then
         print *, "yn, column_est", yn, column_est
         stop
      endif
#endif
#endif /* RAYTRACE */
c
c Total column density of HI nuclei within characteristic scale-length dl
c
      localapprox = 0.5d0 * dl * yn
      if (iphoto .eq. 3) then
        NH = dl * yn
      elseif (iphoto .eq. 4 .or. iphoto.eq.5 ) then
        NH = column_est/1.4/mh
        NH = NH + localapprox
      else
        NH = localapprox
      endif
#ifdef DEBUG_SIGMA_PCC
      write(88,8888) NH, localapprox
#endif      
8888  format(7(1X,E13.6))      
c
c H2 self-shielding
c
      if (iphoto .eq. 0 .or. G0 .eq. 0) then
        fshield = 1d0
      elseif (iphoto .eq. 1 .or. iphoto .eq. 3 
     &        .or. iphoto .eq. 4 .or. iphoto .eq. 5) then
        b = dsqrt(kboltz * temp / mh)
        b5 = b / 1d5
        NH2 = abh2 * NH

        h2dis0 = NH2 / 5d14
        h2dis1 = dsqrt(1d0 + h2dis0)

        fshield = 9.65d-1 / (1d0 + h2dis0 / b5)**2  +
     $            3.5d-2 * dexp(-8.5d-4 * h2dis1) / h2dis1
      elseif (iphoto .eq. 2) then
#ifdef RAYTRACE
c
c To compute H2 self-shielding, we assume that the local b param. is 
c representative of the whole flow. This isn't true in general, but 
c is still a minor approximation compared to the others we are making
c
        b  = dsqrt(kboltz * temp / mh)
        b5 = b / 1d5
        fshield = 0d0
        do I = 1, 6
          h2dis0  = col_H2(I) / 5d14
#ifdef DEBUG_CALC_PHOTO
          if (col_H2(I) .ne. col_H2(I)) then
            write(*,'(A)') 'CALC_PHOTO: invalid H2 column -- NH2 is NaN'
            print*, I, col_H2(I), id_current
            ABORT(124)
          elseif (col_H2(I) .lt. 0d0) then
            write(*,'(A)') 'CALC_PHOTO: invalid H2 column -- neg. NH2'
            print*, I, col_H2(I), id_current
            ABORT(124)
          endif
#endif
          h2dis1  = dsqrt(1d0 + h2dis0)
          fshield = fshield + (1d0 / 6d0) * (9.65d-1 /
     $            ((1d0 + h2dis0 / b5) * (1d0 + h2dis0 / b5))
     $            + 3.5d-2 * dexp(-8.5d-4 * h2dis1) / h2dis1)
c
        enddo
#else /* RAYTRACE */
        write(*,'(A)') 'CALC_PHOTO: invalid photodissociation option'
        write(*,'(A)') 'CALC_PHOTO: recompile with RAYTRACE enabled'
        ABORT(121)
#endif /* RAYTRACE */
      elseif (iphoto .gt. 5) then
         write(*,'(A)') 'CALC_PHOTO: invalid photodissociation option'
        ABORT(122)
      endif
c
c HD shielding
c
      if (iphoto .eq. 0 .or. abhd .eq. 0d0 .or. G0 .eq. 0) then
        fshield_hd = 1d0
      elseif (iphoto .eq. 1 .or. iphoto .eq. 3 
     &        .or. iphoto .eq. 4 .or. iphoto .eq. 5) then
c
c Approximate fit to the HD self-shielding curve. Good to within a factor
c of 2-3 (at worst) for all temperatures in the range 50 < T < 8000K; the
c typical rms deviation is actually more like 20-30%. Exact in the low NHD
c limit.
c
c NB We assume that overlap of the HD and H2 lines can be ignored.
c
        NHD = abhd * NH
        hddis0 = NHD / 1d14
        hddis1 = dsqrt(1d0 + hddis0)
        if (hddis0 .lt. 1d1) then
          fshield_hd = 9.5d-1 / (1d0 + hddis0 / b5)**2 +
     $                 5.0d-2 * dexp(-8.5d-4 * hddis1) / hddis1
        elseif (hddis0 .lt. 1d4) then
          fshield_hd = 9.4d-1 / (1d0 + hddis0 / b5)**2 +
     $                 6.0d-2 / hddis1**1.5d0
        else
          fshield_hd = 9.85d-1 / (1d0 + hddis0 / b5)**2 +
     $                 4.0d-3  / hddis1**0.9d0
        endif
      elseif (iphoto .eq. 2) then
#ifdef RAYTRACE
        fshield_hd = 0d0
        do I = 1, 6
c Approximate HD self-shielding
c To improve this, need to add HD to raytracing code
          if (abh2 .gt. 0d0) then
            NHD = (abhd / abh2) * NH2
          else
            NHD = 0d0
          endif
          hddis0 = NHD / 1d14
          hddis1 = dsqrt(1d0 + hddis0)
          if (hddis0 .lt. 1d1) then
            fshield_hd = 9.5d-1 / (1d0 + hddis0 / b5)**2 +
     $                   5.0d-2 * dexp(-8.5d-4 * hddis1) / hddis1
          elseif (hddis0 .lt. 1d4) then
            fshield_hd = 9.4d-1 / (1d0 + hddis0 / b5)**2 +
     $                   6.0d-2 / hddis1**1.5d0
          else
            fshield_hd = 9.85d-1 / (1d0 + hddis0 / b5)**2 +
     $                   4.0d-3  / hddis1**0.9d0
          endif
        enddo
#endif /* RAYTRACE */
      endif
c
c Dust
c
      if (iphoto .eq. 1 .or. iphoto .eq. 3 .or.
     $    iphoto .eq. 4 .or. iphoto .eq.5 .or. 
     $    dust_to_gas_ratio .eq. 0d0) then
c
c If the dust:gas ratio is zero, then there's no dust extinction from
c the matter in the box, and AV == AV_ext. In this case, the code below
c gives the right answer at much less cost than the raytracing code 
c usually enabled by iphoto .eq. 2
c
        if (iphoto .eq. 0) then
          AV = AV_ext
        else
          AV = AV_ext + AV_conversion_factor * dust_to_gas_ratio * NH 
        endif      
#ifdef DEBUG_CALC_PHOTO
        if (AV .ne. AV) then
          write(*,'(A)') 'CALC_PHOTO: invalid opacity -- AV is NaN'
          ABORT(123)
        endif
#endif
c
        if (AV .gt. 0d0) then
          f_dust = dexp(-AV * (2d-21 / AV_conversion_factor))
          chi_isrf = chi(AV)
#ifdef DEBUGDUSTTEMP
          if (yn .gt. 1e5 ) print *, "DUST T DEBUG (n, AV, NH, chi):", 
     &                      yn, AV, NH, chi_isrf
#endif
#if CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 6
          f_simp = dexp(-2.5d0 * AV)
#else
          f_simp = 1d0
#endif
          df25   = dexp(-2.5d0 * AV)
#if CHEMISTRYNETWORK != 4 && CHEMISTRYNETWORK != 5 && CHEMISTRYNETWORK != 6
          df05   = dexp(-0.5d0 * AV)
          df19   = dexp(-1.9d0 * AV)
          df30   = dexp(-3.0d0 * AV)
#endif
#if CHEMISTRYNETWORK == 3 || CHEMISTRYNETWORK == 7 || CHEMISTRYNETWORK == 11
          df09   = dexp(-0.9d0 * AV)
          df12   = dexp(-1.2d0 * AV)
          df17   = dexp(-1.7d0 * AV)
          df18   = dexp(-1.8d0 * AV)
          df21   = dexp(-2.1d0 * AV)
          df23   = dexp(-2.3d0 * AV)
          df24   = dexp(-2.4d0 * AV)
          df28   = dexp(-2.8d0 * AV)
          df31   = dexp(-3.1d0 * AV)
          df37   = dexp(-3.7d0 * AV)
          df39   = dexp(-3.9d0 * AV)
          if (AV .lt. 1.5d1) then
            dfsp1  = dexp(-2.55d0 * AV + 1.65d-2 * AV**2)
          else
            dfsp1  = dexp(-2.7975 * AV)
         endif
#endif
        else
          f_dust = 1d0
          f_simp = 1d0
          df05   = 1d0
          df09   = 1d0
          df12   = 1d0
          df17   = 1d0
          df18   = 1d0
          df19   = 1d0
          df21   = 1d0
          df23   = 1d0
          df24   = 1d0
          df25   = 1d0
          df28   = 1d0
          df30   = 1d0
          df31   = 1d0
          df37   = 1d0
          df39   = 1d0
          dfsp1  = 1d0
          chi_isrf = 1d0
        endif
      elseif (iphoto .eq. 2) then        
#ifdef RAYTRACE
        sigma_dust = 2d-21 * dust_to_gas_ratio
        f_dust     = 0d0
        f_simp     = 0d0
        df05       = 0d0
        df09       = 0d0
        df12       = 0d0
        df17       = 0d0
        df18       = 0d0
        df19       = 0d0
        df21       = 0d0
        df23       = 0d0
        df24       = 0d0
        df25       = 0d0
        df28       = 0d0
        df30       = 0d0
        df31       = 0d0
        df37       = 0d0
        df39       = 0d0
        dfsp1      = 0d0
        chi_isrf   = 0d0
        do I = 1, 6
          f_dust = f_dust + (1d0 / 6d0) * 
     $             dexp(-sigma_dust * col_tot(I))
          AV     = AV_ext + AV_conversion_factor 
     $           * dust_to_gas_ratio * col_tot(I)
          chi_isrf = chi_isrf + (1d0 / 6d0) * chi(AV)
#ifdef DEBUG_CALC_PHOTO
          if (AV .ne. AV) then
            write(*,'(A)') 'CALC_PHOTO: invalid opacity -- AV is NaN'
            print*, AV_ext, I, col_tot(I)
            ABORT(124)
          endif
#endif
#if CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 6
          f_simp = f_simp + (1d0 / 6d0) * dexp(-2.5d0 * AV)
#else
          f_simp = 1d0
#endif
          df25 = df25 + (1d0 / 6d0) * exp(-2.5d0 * AV)
#if CHEMISTRYNETWORK != 4 && CHEMISTRYNETWORK != 5 && CHEMISTRYNETWORK != 6
          df05 = df05 + (1d0 / 6d0) * exp(-0.5d0 * AV)
          df19 = df19 + (1d0 / 6d0) * exp(-1.9d0 * AV)
          df30 = df30 + (1d0 / 6d0) * exp(-3.0d0 * AV)
#endif
#if CHEMISTRYNETWORK == 3 || CHEMISTRYNETWORK == 7 || CHEMISTRYNETWORK == 11
          df09 = df09 + (1d0 / 6d0) * exp(-0.9d0 * AV)
          df12 = df12 + (1d0 / 6d0) * exp(-1.2d0 * AV)
          df17 = df17 + (1d0 / 6d0) * exp(-1.7d0 * AV)
          df18 = df18 + (1d0 / 6d0) * exp(-1.8d0 * AV)
          df21 = df21 + (1d0 / 6d0) * exp(-2.1d0 * AV)
          df23 = df23 + (1d0 / 6d0) * exp(-2.3d0 * AV)
          df24 = df24 + (1d0 / 6d0) * exp(-2.4d0 * AV)
          df28 = df28 + (1d0 / 6d0) * exp(-2.8d0 * AV)
          df31 = df31 + (1d0 / 6d0) * exp(-3.1d0 * AV)
          df37 = df37 + (1d0 / 6d0) * exp(-3.7d0 * AV)
          df39 = df39 + (1d0 / 6d0) * exp(-3.9d0 * AV)
          if (AV .lt. 1.5d1) then
            dfsp1 = dfsp1 + (1d0 / 6d0) * exp(-2.55d0 * AV
     $            + 1.65d-2 * AV**2)
          else
            dfsp1 = dfsp1 + (1d0 / 6d0) * exp(-2.7975d0 * AV)
         endif
#endif
        enddo
#endif /* RAYTRACE */
      endif
c
c CO self-shielding, shielding by H2.
c
c Our treatment here is based on L96.
c
      fshield_CO = 1d0
#ifdef NEEDS_CO_RATES
#ifdef CO_SHIELDING
      if (iphoto .eq. 1 .or. iphoto .eq. 3 
     &    .or. iphoto .eq. 4 .or. iphoto .eq. 5
     &    .or. G0 .eq. 0) then
        NCO = abco * NH
        if (NCO .le. NCO_shield_min) then
          theta_CO = 1d0
        elseif (NCO .ge. NCO_shield_max) then
          theta_CO = CO_self_shielding(imax_COss)
        else
          ipos  = int(dlog10(NCO) / dNshield) - ioff_co
          dth   = CO_self_shielding(ipos+1) - CO_self_shielding(ipos)
          dNact = dlog10(NCO) - (ipos + ioff_co) * dNshield
          theta_CO = CO_self_shielding(ipos) 
     $             + dth * (dNact / dNshield)
        endif
c
        NH2 = abh2 * NH
        if (NH2 .le. NH2_shield_min) then
          theta_H2 = 1d0
        elseif (NH2 .ge. NH2_shield_max) then
          theta_H2 = CO_H2_shielding(imax_COH2)
        else
          ipos  = int(dlog10(NH2) / dNshield) - ioff_h2
          if (ipos .eq. 0) then
            theta_H2 = 1d0
          endif
          dth   = CO_H2_shielding(ipos+1) - CO_H2_shielding(ipos)
          dNact = dlog10(NH2) - (ipos + ioff_h2) * dNshield
          theta_H2 = CO_H2_shielding(ipos) 
     $             + dth * (dNact / dNshield)
        endif
        fshield_CO = theta_H2 * theta_CO
      elseif (iphoto .eq. 2) then
#ifdef RAYTRACE
        theta_CO   = 0d0
        theta_H2   = 0d0
        do I = 1, 6
          NCO = col_CO(I)
#ifdef DEBUG_CALC_PHOTO
          if (NCO .ne. NCO) then
            write(*,'(A)') 'CALC_PHOTO: invalid CO column -- NCO is NaN'
            print*, I, col_CO(I)
            ABORT(124)
          endif
#endif
          if (NCO .le. NCO_shield_min) then
            theta_CO = theta_CO + 1d0 / 6d0
          elseif (NCO .ge. NCO_shield_max) then
            theta_CO = theta_CO + (1d0 / 6d0)
     $               * CO_self_shielding(imax_COss)
          else
            ipos  = int(dlog10(NCO) / dNshield) - ioff_co
            if (ipos .eq. 0) then
              theta_CO = theta_CO + 1d0 / 6d0
            endif
            dth = CO_self_shielding(ipos+1) - 
     $            CO_self_shielding(ipos)
            dNact = dlog10(NCO) - (ipos + ioff_co) * dNshield
            theta_CO = theta_CO + (1d0/6d0)
     $               * (CO_self_shielding(ipos) 
     $               + dth * (dNact / dNshield))
          endif
c
          NH2 = col_H2(I)
          if (NH2 .le. NH2_shield_min) then
            theta_H2 = theta_H2 + 1d0 / 6d0
          elseif (NH2 .ge. NH2_shield_max) then
            theta_H2 = theta_H2 + (1d0 / 6d0) 
     $               * CO_H2_shielding(imax_COH2)
          else
            ipos  = int(dlog10(NH2) / dNshield) - ioff_h2
            if (ipos .eq. 0) then
              theta_H2 = theta_H2 + 1d0 / 6d0
            endif
            dth   = CO_H2_shielding(ipos+1) - 
     $              CO_H2_shielding(ipos)
            dNact = dlog10(NH2) - (ipos + ioff_h2) * dNshield
            theta_H2 = theta_H2 + (1d0 / 6d0)
     $               * (CO_H2_shielding(ipos) 
     $               + dth * (dNact / dNshield))
          endif
        enddo
        fshield_CO = theta_H2 * theta_CO
#endif /* RAYTRACE */
      endif
#endif /* CO_SHIELDING */
#endif /* NEEDS_CO_RATES */

c H2:
      phrates(1) = phtab(1) * G0 * fshield * f_dust

c Network 4:
      if (nchem_network .eq. 4) then
        goto 212
      endif

c Networks 5 & 6:
      if (nchem_network .eq. 5 .or. nchem_network .eq. 6) then
#if CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 6
        phrates(10) = phtab(10) * G0 * f_simp
        UNUSED_PARAM(fshield_CO)
#endif
        goto 212
      endif

c Other networks -- shared rates:
      phrates(4) = phtab(4)  * G0 * df05
      phrates(5) = phtab(5)  * G0 * df19
c Dust scalings for (photoheating) rates 8,9 assumed same as for
c corresponding photoion. or photodissoc. rate
      phrates(8) = phtab(8) * G0 * df05
      phrates(9) = phtab(9) * G0 * df19
c
c HD:
      phrates(35) = phtab(35) * G0 * fshield_hd * f_dust

c Photoionization of HI, HeI, HeII, H2, OI and SiII. In this implementation, 
c we assume that absorption from gas outside of the simulation volume dominates,
c and that this has been taken into account in the calculation of the 
c rates given in photoinit(). However, for convenience we still keep the
c scaling with G0, to allow the strength of the field to be varied easily.
c
c Users desiring more complex behaviour should override these values using
c the user-defined subroutine SPECIAL_PHOTO called below.
c
c HI:
      phrates(38) = phtab(38) * G0
c HeI:
      phrates(39) = phtab(39) * G0
c HeII:
      phrates(40) = phtab(40) * G0
c H2:
      phrates(41) = phtab(41) * G0
c OI:
      phrates(46) = phtab(46) * G0
c SiII:
      phrates(47) = phtab(47) * G0
c
c Photoheating from HI, HeI, HeII and H2 photoionization 
c
      phrates(42) = phtab(42) * G0
      phrates(43) = phtab(43) * G0
      phrates(44) = phtab(44) * G0
      phrates(45) = phtab(45) * G0
c
c Network 1 is done, so skip to the end
c
      if (nchem_network .eq. 1) then
        goto 212
      endif
c
      phrates(2) = phtab(2) * G0 * df30
c Dust scalings for (photoheating) rate 6 assumed same as for
c corresponding photoion. or photodissoc. rate
      phrates(6)  = phtab(6)  * G0 * df30
      if (nchem_network .ne. 7 .and. nchem_network .ne. 11) then
        phrates(3)  = phtab(3)  * G0 * df19
        phrates(7)  = phtab(7)  * G0 * df19        
      endif
c
c Network 2 is done:
c
      if (nchem_network .eq. 2) then
        goto 212
      endif
c
      if (nchem_network .eq. 3) then
        phrates(31) = phtab(31) * G0 * df12
        phrates(32) = phtab(32) * G0 * df24
      endif

CO:
      phrates(10) = phtab(10) * G0 * fshield_CO * df25
      UNUSED_PARAM(f_simp)
      phrates(11) = phtab(11) * G0 * df17
      phrates(12) = phtab(12) * G0 * df17
      phrates(14) = phtab(14) * G0 * df25
      phrates(15) = phtab(15) * G0 * df17
      phrates(16) = phtab(16) * G0 * df17
      phrates(18) = phtab(18) * G0 * df12
      phrates(19) = phtab(19) * G0 * df28
      phrates(20) = phtab(20) * G0 * df17
      phrates(21) = phtab(21) * G0 * df23
      phrates(22) = phtab(22) * G0 * df18
      phrates(23) = phtab(23) * G0 * df31
      phrates(24) = phtab(24) * G0 * df39
      phrates(25) = phtab(25) * G0 * df21
      phrates(27) = phtab(27) * G0 * df18
      phrates(28) = phtab(28) * G0 * df37
      phrates(29) = phtab(29) * G0 * df09
      phrates(30) = phtab(30) * G0 * df05
      phrates(33) = phtab(33) * G0 * dfsp1
      phrates(34) = phtab(34) * G0 * dfsp1
      phrates(36) = 0.5d0 * phtab(36) * G0 * df18  ! H3+ -> H2  + H+
      phrates(37) = 0.5d0 * phtab(36) * G0 * df23  ! H3+ -> H2+ + H
c
212   continue
c
c The scaling factor f_rsc rescales the field strength to units of 
c the Habing field: for ISM simulations, G0 is already written in 
c terms of this, and so f_rsc = 1, but for high z simulations, this 
c is not the case. Note that the value of f_rsc is set in photoinit.
c
c This scaled value of G is used in our treatment of grain surface
c recombination and photoelectric heating, which is based on BHRB04.
c This treatment neglects any variation of the photoelectric 
c heating efficiency with increasing A_V.
c
      G_dust = f_rsc * G0 * df25
c
c Optionally, call a user-defined subroutine SPECIAL_PHOTO to override
c some or all of previous values
c
#ifdef SPECIAL_PHOTO
      call SPECIAL_PHOTO
#endif
c
#endif /* CHEMCOOL */
      return
      end
c=======================================================================
c
c    \\\\\\\\\\          E N D     F U N C T I O N        //////////
c    //////////             C A L C _ P H O T O           \\\\\\\\\\
c
c=======================================================================
#undef NEEDS_CO_RATES
c
c=======================================================================
c
c
c    \\\\\\\\\\      B E G I N   F U N C T I O N          //////////
c    //////////                C H I                      \\\\\\\\\\
c
c=======================================================================
c
      REAL function chi(AV)
#ifdef CHEMCOOL
      implicit NONE
#include "cool.h"
#include "isrf.h"
      REAL AV, dchi, dAV
      integer index

      if (AV .eq. 0) then
        chi = 1d0
        return
      elseif (AV .gt. 2d2) then
c Assume chi remains constant for AV > 200
        chi = chi_tab(200)
        return
      else
        index = int(AV)
        if (index .eq. 0) then
          dchi = chi_tab(1) - 1d0
          dAV  = AV - index
          chi  = 1d0 + dAV * dchi
        else
          dchi = chi_tab(index+1) - chi_tab(index)
          dAV  = AV - index
          chi  = chi_tab(index) + dAV * dchi
        endif
        return
      endif
#else
      chi = 1.0
#endif
      return
      end
c
c=======================================================================
c
c
c    \\\\\\\\\\          E N D   F U N C T I O N          //////////
c    //////////                  C H I                    \\\\\\\\\\
c
c=======================================================================
