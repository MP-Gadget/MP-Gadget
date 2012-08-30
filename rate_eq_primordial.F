#if CHEMISTRYNETWORK == 1
c=======================================================================
c
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////             R A T E _ E Q                 \\\\\\\\\\
c
c=======================================================================
c
c    Written by S. Glover (2004-2006)
c
      subroutine rate_eq(neq, time, y, ydot, rpar, ipar)
#ifdef CHEMCOOL
      implicit NONE
#include "cool.h"
#include "non_eq.h"
      integer neq
c
      REAL time, y(nspec), ydot(nspec), rpar(nrpar)
      integer ipar(nipar)
      REAL phrates(npr)
c
      integer niter, giter
      parameter (niter = 25)
      REAL eqtol
      parameter (eqtol = 1d-3)
c
      REAL    dtemp
      integer itemp, I, J, K, kfmin
c
      REAL abundances(nabn), rates(nrates),
     $     rates_chem(nrates_chem), ylam, ylam_chem,
     $     CMB_temp, ylam_cmb, rates_cmb(nrates)
      REAL temp, yn, dl, divv, energy, gamma, yntot,
     $     ekn
c
      REAL sigma, ssb, dust_cool, gas_dust, kappa, dlnkappadT,
     $     tdust_new, F_nr, dF_nr, gget, ggetn, f_K, fmin, dt
      parameter (ssb = 5.67d-5)

      REAL abhp, abh2, abhd, abdp, abhep, abhepp 
      REAL abhm, abhm_old, abh2p, abh2p_old
#ifdef FF_COLLAPSE_TEST
      REAL n0, tffc, ndot, compress, tff
#endif
      REAL num, denom
      REAL comp_equilib
      external comp_equilib

      REAL total_rad_heating, heating_rate, Lstar, distance

      REAL time_to_equilibrium(num_eqb_species)

      REAL abe, abHI, abDI, abheI

      REAL ph4,  ph5, ph8, ph9, ph35, ph38, ph39,
     $     ph40, ph41, ph42, ph43, ph44, ph45

      REAL cr1, cr1a, cr2, cr6

      REAL ykdh, ykdh2, ykdis, NH, ncrinv, h2var0, h2var1, 
     $     h2var2, G_dust, ynsq, rho, cs2, L_jeans, ykdhe,
     $     chi_isrf

      REAL  ch1,  ch2,  ch3,  ch4,  ch5,  ch6,  ch8,  ch9,
     $     ch10, ch11, ch13, ch14, ch15, ch17, ch18, ch20,
     $     ch21, ch22, ch23, ch24, ch25, ch26, ch27, ch28, 
     $     ch29, ch30, ch31, ch32, ch33, ch98, ch99, 
     $     ch100, ch101, ch102, ch103, ch108, ch109, ch110,
     $     ch126, ch127, ch137, ch138, ch139

      REAL  dtch1,  dtch2, dtch3,   dtch4,  dtch5,  dtch6,
     $      dtch8,  dtch9, dtch10, dtch11, dtch13, dtch14, 
     $     dtch15, dtch17, dtch18, dtch20, dtch21, dtch22, 
     $     dtch23, dtch24, dtch25, dtch26, dtch27, dtch28, 
     $     dtch29, dtch30, dtch31, dtch32, dtch33,
     $     dtch98, dtch99,  dtch100, dtch101, dtch102, 
     $     dtch103, dtch108, dtch109, dtch110,
     $     dtch126, dtch127, dtch137, dtch138, dtch139

      REAL cl14, dtcl14
c
#ifdef DMA
      REAL gas_column, dma_heat, dma_ion, dm_density_gevcc
#endif
      REAL mh
      parameter (mh  = PROTONMASS)
#ifdef FF_COLLAPSE_TEST
      REAL G_n
      parameter (G_n = GRAVITY)
#endif
c
      no_dchem = 0
c
c Unpack parameters
c
      yn    = rpar(1)
      dl    = rpar(2)
      divv  = rpar(3)
c
      ynsq  = yn**2
c
#ifdef FF_COLLAPSE_TEST
      n0  = rpar(4)
      tffc =  dsqrt(32d0 * GRAVITY * 1.4d0 * PROTONMASS / 
     $        (3d0 * PI))
      yn = (1d0 - 0.5d0 * tffc * dsqrt(n0) * time)**(-2d0) * n0 
#endif
c
      call validate_input(y, ipar)
      if (ipar(1) .eq. 1) then
        do I = 1, nspec
          ydot(I) = 0d0
        enddo
c We simply return here because we'll be redoing the step anyway
c (making any further work pointless) and because we may be doing 
c this on account of having negative chemical abundances, which 
c would otherwise cause serious problems later on.
        return
      endif
c
c From this point on, we know that our tracked abundances cannot be negative
c
      abhp   = y(ihp)
      abh2   = y(ih2)
      abhd   = y(ihd)
      abdp   = y(idp)
      abhep  = y(ihep)
      abhepp = y(ihepp)

      abe  = abhp + abdp + abhep + 2d0 * abhepp
c
      abHI = 1d0 - 2d0 * abH2 - abhp - abhd
      abHI = max(0d0, abHI)
c
      abDI  = max(0d0, abundD - abdp - abhd)
      if (yn .gt. 9d7 .and. no_dchem .eq. 0 .and. abundD .gt. 0d0) then
        abDI = max(abDI, atol(ihd))
      endif
c
      abheI = max(0d0, abhe - abhep - abhepp)
c
c Internal energy (erg cm^-3)
c
      energy = y(itmp)
c
c Compute current adiabatic index and total number density
c
      yntot = (1d0 + abhe + abe - abh2) * yn
      ekn = energy / (kboltz * yntot)
      temp  = (2d0 / 3d0) * ekn
      gamma = 5d0 / 3d0
c     call lookup_temp_gamma(abh2, ekn, temp, gamma)
c
c Compute local Jeans length (for use in cool_func)
c
      rho = (1d0 + 4d0 * abhe) * mh * yn
      cs2 = (gamma * (gamma - 1d0) * energy / rho)
      L_jeans  = dsqrt(PI * cs2 / 
     $           (GRAVITY * rho))
c
c Compute CMB temperature
c
      CMB_temp = 2.726d0 * (1d0 + redshift)
c
c Bounds checking
c
      if (temp .le. 0d0) then
        itemp   = 1
        dtemp   = 0d0
        temp    = tmin
        ipar(1) = 1
      elseif (temp .le. tmin) then
        itemp = 1
        dtemp = 0d0
      elseif (temp .ge. tmax) then
        itemp = nmd
        dtemp = temp - temptab(itemp)
      else
        itemp = int(dlog10(temp) / dtlog) + 1
        if (itemp .le. 0 .or. itemp .gt. nmd) then
          print*, 'Fatal error in rate_eq.F', itemp, temp
          ABORT(1)
        endif
        dtemp = temp - temptab(itemp)
      endif
c
c Rates:
c
       dtch1  = dtchtab(1, itemp)
       dtch2  = dtchtab(2, itemp)
       dtch3  = dtchtab(3, itemp)
       dtch4  = dtchtab(4, itemp)
       dtch5  = dtchtab(5, itemp)
       dtch6  = dtchtab(6, itemp)
c
       dtch8  = dtchtab(8, itemp)
       dtch9  = dtchtab(9, itemp)
c 
       dtch10 = dtchtab(10,itemp)
       dtch11 = dtchtab(11,itemp) 
c
       dtch13 = dtchtab(13,itemp)
       dtch14 = dtchtab(14,itemp)
       dtch15 = dtchtab(15,itemp)
c
       dtch17 = dtchtab(17,itemp)
       dtch18 = dtchtab(18,itemp)
c
       dtch20 = dtchtab(20,itemp)
       dtch21 = dtchtab(21,itemp)
       dtch22 = dtchtab(22,itemp)
       dtch23 = dtchtab(23,itemp)
       dtch24 = dtchtab(24,itemp)
       dtch25 = dtchtab(25,itemp)
       dtch26 = dtchtab(26,itemp)
       dtch27 = dtchtab(27,itemp)
       dtch28 = dtchtab(28,itemp)
       dtch29 = dtchtab(29,itemp)
       dtch30 = dtchtab(30,itemp)
       dtch31 = dtchtab(31,itemp)
       dtch32 = dtchtab(32,itemp)
       dtch33 = dtchtab(33,itemp)
c
       dtch98 = dtchtab(98,itemp)
       dtch99 = dtchtab(99,itemp)
       dtch100 = dtchtab(100,itemp)
       dtch101 = dtchtab(101,itemp)
       dtch102 = dtchtab(102,itemp)
       dtch103 = dtchtab(103,itemp)
c
       dtch108 = dtchtab(108,itemp)
       dtch109 = dtchtab(109,itemp)
       dtch110 = dtchtab(110,itemp)
c
       dtch126 = dtchtab(126,itemp)
       dtch127 = dtchtab(127,itemp)
c
       dtch137 = dtchtab(137,itemp)
       dtch138 = dtchtab(138,itemp)
       dtch139 = dtchtab(139,itemp)
c
       ch1  = chtab(1, itemp) + dtemp * dtch1
       ch2  = chtab(2, itemp) + dtemp * dtch2
       ch3  = chtab(3, itemp) + dtemp * dtch3
       ch4  = chtab(4, itemp) + dtemp * dtch4
       ch5  = chtab(5, itemp) + dtemp * dtch5
       ch6  = chtab(6, itemp) + dtemp * dtch6
c       
       ch8  = chtab(8, itemp) + dtemp * dtch8
       ch9  = chtab(9, itemp) + dtemp * dtch9
c
       ch10 = chtab(10,itemp) + dtemp * dtch10 
       ch11 = chtab(11,itemp) + dtemp * dtch11
c
       ch13 = chtab(13,itemp) + dtemp * dtch13
c
       ch14 = chtab(14,itemp) + dtemp * dtch14
       ch15 = chtab(15,itemp) + dtemp * dtch15
c
       ch17 = chtab(17,itemp) + dtemp * dtch17
       ch18 = chtab(18,itemp) + dtemp * dtch18
c
       ch20 = chtab(20,itemp) + dtemp * dtch20
       ch21 = chtab(21,itemp) + dtemp * dtch21
       ch22 = chtab(22,itemp) + dtemp * dtch22
       ch23 = chtab(23,itemp) + dtemp * dtch23
       ch24 = chtab(24,itemp) + dtemp * dtch24
       ch25 = chtab(25,itemp) + dtemp * dtch25
       ch26 = chtab(26,itemp) + dtemp * dtch26
       ch27 = chtab(27,itemp) + dtemp * dtch27
       ch28 = chtab(28,itemp) + dtemp * dtch28
       ch29 = chtab(29,itemp) + dtemp * dtch29
       ch30 = chtab(30,itemp) + dtemp * dtch30
       ch31 = chtab(31,itemp) + dtemp * dtch31
       ch32 = chtab(32,itemp) + dtemp * dtch32
       ch33 = chtab(33,itemp) + dtemp * dtch33
c
       ch98 = chtab(98,itemp) + dtemp * dtch98
       ch99 = chtab(99,itemp) + dtemp * dtch99
       ch100 = chtab(100,itemp) + dtemp * dtch100
       ch101 = chtab(101,itemp) + dtemp * dtch101
       ch102 = chtab(102,itemp) + dtemp * dtch102
       ch103 = chtab(103,itemp) + dtemp * dtch103
c
       ch108 = chtab(108,itemp) + dtemp * dtch108
       ch109 = chtab(109,itemp) + dtemp * dtch109
       ch110 = chtab(110,itemp) + dtemp * dtch110
c
       ch126 = chtab(126,itemp) + dtemp * dtch126
       ch127 = chtab(127,itemp) + dtemp * dtch127
c
       ch137 = chtab(137,itemp) + dtemp * dtch137
       ch138 = chtab(138,itemp) + dtemp * dtch138
       ch139 = chtab(139,itemp) + dtemp * dtch139
c
       dtcl14 = dtchtab(14,itemp)
       cl14   = cltab(14,itemp) + dtemp * dtcl14
c
c H2 collisional dissociation rates -- need special handling because of
c density dependence
c
      ncrinv   = (2d0 * abh2 * (ch6 - ch5) + ch5 + abheI * ch139)
      h2var0   = 1d0 / ( 1d0 + yn * ncrinv)
      h2var1   = ch3**h2var0
      h2var2   = ch4**h2var0
      ykdh     = ch1 * h2var1
      ykdh2    = ch2 * h2var2
      if (ch138 .lt. 1d-100) then
        ykdhe = 1d-100
      else
        ykdhe = ch138 * (ch137 / ch138)**h2var0
      endif
c
c Total column density of HI nuclei. Primarily used in photochemistry
c code.
c
      if (iphoto .eq. 3) then
        NH = dl * yn
      else
        NH = 0.5d0 * dl * yn
      endif
c
      call calc_photo(temp, rpar, abh2, abhd, 0d0, phrates, G_dust,
     $                chi_isrf)
c
      ykdis = phrates(1)
      ph4   = phrates(4)
      ph5   = phrates(5)
      ph8   = phrates(8)
      ph9   = phrates(9)
c
      ph35  = phrates(35)
c
      ph38  = phrates(38)
      ph39  = phrates(39)
      ph40  = phrates(40)
      ph41  = phrates(41)
      ph42  = phrates(42)
      ph43  = phrates(43)
      ph44  = phrates(44)
      ph45  = phrates(45)

      cr1  = crtab(1)  ! HI
      cr1a = crtab(1)  ! DI, assumed same as HI
      cr2  = crtab(2)  ! H2
      cr6  = crtab(6)  ! HeI
c
#ifdef DMA
c Calculate heating rate, ionization rate due to DMA
c
      gas_column = rpar(4)
c 
c Convert DM mass density to GeV/c^2/cm^3
c
c Note: 1 gramme = 1.04375e-12 GeV/c^2
c
      dm_density_gevcc = dm_density * 1.04375d-12
c
      call dma_calc(gas_column, dm_density_gevcc, dma_ion, dma_heat)
c
c XXX: need to do something with dma_ion, dma_heat now we have them
c
#endif
c
c Estimate whether we're optically thick to ionizing radiation;
c if we're not, then we need to set ch18 to zero.

      if (2d0 * NH * abhI .lt. 1.6d17) then
        ch18 = 0d0
      endif
c
      if (no_chem .gt. 0) then
        goto 100
      endif
c
c Equilibrium species
c
c Since the equilibrium abundances are coupled, we have to iterate to find 
c accpetable solutions. We iterate niter times (or until all abundances have
c converged).
c
      abhm_old  = 0d0
      abh2p_old = 0d0
c
      abhm  = 0d0
      abh2p = 0d0
#ifdef COLLECT_EQB_DATA
      do I = 1, num_eqb_species
        time_to_equilibrium(I) = 0d0
      enddo
#endif
c
      do I = 1, niter
c
c H- :
c
        num   = ch21 * abHI * abe + ch126 * abh2 * abe
        denom = ch22 * abHI + ch26 * abe + ch27 * abHI + ch28 * abhp
     $        + ch29 * abhp + (ph4 / yn) + ch8 * abhep
     $        + ch31 * abheI + 2d0 * ch127 * abh2p
c
        abhm = comp_equilib(num, denom, 1d0, 1, yn, time_to_equilibrium)
c
        abe  = max(0d0, abe  - (abhm - abhm_old))
        abHI = max(0d0, abHI - (abhm - abhm_old))
c
c H2+ :
c
        num   = ch23 * abHI * abhp + ch24 * abh2 * abhp
     $        + ch29 * abhp * abhm + (cr2 * abh2 / yn)
     $        + cst(75) * abhep * abh2 + (ph41 / yn) * abh2
        denom = cst(1) * abHI + ch30 * abe  + (ph5 / yn)
     $        + 2d0 * ch127 * abhm
c
        abh2p = comp_equilib(num, denom, 0.5d0, 2, yn, 
     $                       time_to_equilibrium)
c
        abe  = max(0d0, abe  + (abh2p - abh2p_old))
        abHI = max(0d0, abHI - 2d0*(abh2p - abh2p_old))
c
c Check for convergence:
c
        if (I .ge. 2) then
          if ((abs(abhm_old - abhm)   .lt. eqtol * abhm)  .and.
     $        (abs(abh2p_old - abh2p) .lt. eqtol * abh2p)) then
            goto 100
          endif
        endif
        abhm_old  = abhm
        abh2p_old = abh2p
      enddo
c
100   continue
c
c Protons
c
      if (no_chem .gt. 0) then
        ydot(ihp)   = 0d0
        ydot(ih2)   = 0d0
        ydot(idp)   = 0d0
        ydot(ihd)   = 0d0
        ydot(ihep)  = 0d0
        ydot(ihepp) = 0d0
      else
        ydot(ihp) = (ch11 * abe   * abHI          ! Collisional ion: H  + e-
     $         + cst(1) * abh2p * abHI          ! Charge transfer: H  + H2+
     $           - ch13 * abhp  * abe           ! Gas-phase H recomb.(case B)
     $           - ch23 * abHI  * abhp          ! H2+ formation:   H  + H+
     $           - ch24 * abh2  * abhp          ! Charge transfer: H2 + H+
     $           - ch28 * abhp  * abhm          ! Neutralization:  H- + H+
     $           - ch29 * abhp  * abhm          ! H2+ formation:   H- + H+
     $           + ch18 * abhep * abe           ! Ion. from He+ rec: He+ + e
     $           + ch32 * abhep * abHI          ! Charge transfer: He+ + H
     $           - ch33 * abheI * abhp          ! Charge transfer: He  + H+
     $          + ch9 * abhep * abh2) * yn    ! DiCT: He+ + H2 -> He + H + H+
     $          +  ph5  * abh2p                 ! Photodissoc:    H2+ + PH
     $          +  cr1  * abHI                  ! Cosmic ray ion.:  H + CR
     $          + ph38  * abhI                  ! Photoion: H + PH
c
      if (no_dchem .eq. 0 .and. abundD .gt. 0d0) then
        ydot(ihp) = ydot(ihp) 
     $            + (ch99 * abdp  * abHI          ! Charge transfer: H + D+
     $            -  ch98 * abhp  * abDI          ! Charge transfer: D + H+
     $            + ch100 * abdp  * abh2          ! HD form:         D+ + H2
     $            - ch101 * abhd  * abhp) * yn    ! HD dest:         HD + H+
      endif
c
c Molecular hydrogen
c
      ydot(ih2) = (ch108  * abHI**3           ! 3-body H2 form: 3H -> H2 + H 
     $           + ch109  * abHI**2 * abh2    ! "  : 2H + H2 -> H2 + H2
     $           + ch110  * abHI**2 * abheI) * yn**2 ! "  : 2H + He -> H2 + He 
     $          + (ch22   * abHI    * abhm    ! H2 formation:        H- + H
     $          + cst(1)  * abHI    * abh2p   ! Charge transfer:     H  + H2+
     $           -  ch24  * abhp    * abh2    ! Charge transfer:     H2 + H+
     $           -  ch25  * abe     * abh2    ! Collisional dissoc.: H2 + e- 
     $           -  ykdh  * abHI    * abh2    ! Collisional dissoc.: H2 + H 
     $           -  ykdh2 * abh2    * abh2    ! Collisional dissoc.: H2 + H2
     $           -  ykdhe * abheI   * abh2    ! Collisional dissoc.: H2 + He
     $           - ch9    * abhep   * abh2    ! DiCT: He+ + H2 -> He + H + H+
     $           - cst(75) * abhep  * abh2    ! CT: He+ + H2 -> He + H2+
     $           - ch126   * abh2   * abe     ! H2 + e -> H- + H
     $           + ch127   * abhm   * abh2p) * yn ! H2+ + H- -> H2 + H
     $           -  ykdis  * abh2             ! Photodissoc:         H2 + PH
     $           -  cr2    * abh2             ! Cosmic ray ion.:     H2 + CR
     $           - ph41    * abh2             ! Photoion:           H2  + PH
c
      if (no_dchem .eq. 0 .and. abundD .gt. 0d0) then
        ydot(ih2) = ydot(ih2)
     $            + (ch101  * abhd    * abhp       ! HD destruction:      HD + H+
     $            -  ch100  * abh2    * abdp       ! HD formation:        D+ + H2
     $            -  ch102  * abh2    * abDI       ! HD formation:         D + H2
     $            +  ch103  * abhd    * abHI) * yn ! HD destruction:      HD + H
      endif
c
c  Ionized deuterium (D+)
c
      if (abundD .eq. 0d0 .or. no_dchem .eq. 1) then
        ydot(idp) = 0d0
      else
        ydot(idp) = (ch98   * abhp  * abDI          ! Charge transfer: D + H+
     $             - ch99   * abdp  * abHI          ! Charge transfer: H + D+
     $             - ch100  * abdp  * abh2          ! HD form: D+ + H2
     $             + ch101  * abhd  * abhp          ! HD dest: HD + H+
     $             - ch13   * abdp  * abe           ! Case B recomb.
     $             + ch11   * abDI  * abe           ! Collisional ionization
     $             + ch32   * abhep * abDI          ! Charge transfer: He+ + D
     $             - ch33   * abheI * abdp) * yn    ! Charge transfer: He  + D+
     $             + cr1a   * abDI                  ! Cosmic ray ion: D + CR
     $             + ph38   * abDI                  ! Photoionization
      endif
c
c  Hydrogen deuteride (HD)
c
c  NB We follow Flower & Harris (2007) and assume that the three-body rates for
c  HD formation are the same as those for H2 formation
c
c  NB For simplicity, we assume that each photoionization leads to the destruction
c  of the molecule; this is incorrect when the fractional ionization is small, but
c  we do not expect HD photoionization to be important in those conditions
c
      if (abundD .eq. 0d0 .or. no_dchem .eq. 1) then
        ydot(ihd) = 0d0
      else
        ydot(ihd) = (ch108  * abHI
     $            +  ch109  * abh2
     $            +  ch110  * abheI) * abHI * abDI * ynsq
     $            + (ch100  * abh2  * abdp          ! HD form: D+ + H2
     $             - ch101  * abhd  * abhp          ! HD dest: HD + H+
     $             + ch102  * abh2  * abDI          ! HD form: D + H2
     $             - ch103  * abhd  * abHI) * yn    ! HD dest: HD + H
     $             - ph35   * abhd                  ! Photodissociation
     $             - ph41   * abhd                  ! Photoionization
      endif
c
c Singly-ionized helium (He+)
c
      if (abhe .eq. 0d0) then
        ydot(ihep) = 0d0
      else
        ydot(ihep) = (ch14    * abheI * abe
     $              - ch15    * abhep * abe
     $              - ch17    * abhep * abe
     $              + ch20    * abhepp * abe
     $              - ch32    * abhep * abHI
     $              + ch33    * abheI * abhp
     $              -  ch8    * abhep * abhm
     $              -  ch9    * abhep * abh2
     $              - cst(75) * abhep * abh2) * yn
     $              + cr6     * abheI
     $              + ph39    * abheI
     $              - ph40    * abhep
      endif
c
      if (no_dchem .eq. 0 .and. abundD .gt. 0d0) then
        ydot(ihep) = ydot(ihep)
     $             + (ch33    * abheI * abdp
     $             -  ch32    * abhep * abDI) * yn
      endif
c
c Doubly-ionized helium (He++)
c
      if (abhe .eq. 0d0) then
        ydot(ihepp)  = 0d0
      else
        ydot(ihepp) = (ch15 * abhep  * abe
     $              -  ch20 * abhepp * abe) * yn
     $              +  ph40 * abhep
      endif
      endif  ! no_chem
c
#if 0
      if (dust_to_gas_ratio .gt. 0d0) then
c
c Solve for equilibrium dust temperature.
c
c We assume currently that the only significant external heat source is
c the CMB, and include its effects in the calculation of the radiative
c cooling term, as in Stamatellos et al (2007, A&A, 475, 37)
c
c First, we need to compute our effective column density, sigma, which
c we estimate as the local density * local Jeans length
c
        sigma = rho * L_jeans
c
c Iterate using Newton's method
c
        if (tdust .lt. 0) then
          write(*,*) "T dust got negative", tdust
        endif
        tdust_new = tdust
        do J = 1, niter
          tdust = tdust_new
c Calculate kappa and dln (kappa)/d tdust
          call kbell_lin(tdust, rho, kappa, dlnkappadT)
          kappa = kappa * dust_to_gas_ratio
c
          dust_cool = 4d0 * ssb * rho * (tdust**4 - CMB_temp**4) /
     $                (sigma**2 * kappa + 1d0 / kappa)
c
c Gas-grain energy transfer -- Hollenbach & McKee (1989)
c
          gas_dust  = cl14 * (temp - tdust) * yn**2 * dust_to_gas_ratio
c
          F_nr  = gas_dust - dust_cool
c
          dF_nr = - cl14 * yn**2 * dust_to_gas_ratio 
     $          - ((16d0 * ssb * rho * tdust**3)
     $          / (sigma**2 * kappa + 1d0 / kappa))
     $          + dust_cool * (sigma**2 * kappa - 1d0 / kappa) 
     $          * dlnkappadT / (sigma**2 * kappa + 1d0 / kappa) 
c
c          write(*,900) F_nr," ", dF_nr," ", tdust, " ",  kappa," ",
c     $                 id_current
c900       format (E10.2,A, E10.2,A, E10.2,A, E10.2, A, I6)
c          write(*,*) tdust, id_current
          if (tdust - F_nr / dF_nr .gt. 0) then
            tdust_new = tdust - F_nr / dF_nr
          else
            tdust_new = (tdust + tdust_new)/2
          endif
            write(*,*) tdust, tdust_new, id_current, F_nr/dF_nr,
     $                 "Newton", rho, dust_to_gas_ratio, CMB_temp,
     $                 ssb, sigma, cl14, yn
c
c  If the iteration got 10^-3 precision, it finishes.
c
          if (abs(tdust - tdust_new) .lt. 
     $        (1d-3 * max(tdust_new, tdust)) 
     $         .and. tdust .gt. 0d0) then
            goto 200
c
c If the Newton method could not get to the value
c try the "Root in a Grid" methot.
c 
          elseif (tdust .gt. 0d0
     $         .and. J .gt. (niter - 1)) then
c            if (tdust > tdust_new) then
c              tdust = tdust * 2d0
c              tdust_new = tdust_new * 0.5d0
c            else
c              tdust = tdust * 0.5d0
c              tdust_new = tdust_new * 2d0
c            endif
c            giter = ceiling(abs(tdust - tdust_new)/
c     $                   (1d-3 * max(tdust, tdust_new)))
c            fmin =1
c            kfmin = 0
c            dt = (tdust_new - tdust)/giter
c            do K = 1, giter
c              tdust_new = tdust + K * dt
c              call kbell_lin(tdust_new, rho, kappa, dlnkappadT)
c              kappa = kappa * dust_to_gas_ratio
c              dust_cool = 4d0 * ssb * rho * (tdust_new**4 - CMB_temp**4)
c     $              / (sigma**2 * kappa + 1d0 / kappa)
c              gas_dust  = cl14 * (temp - tdust_new) * yn**2
c     $              * dust_to_gas_ratio
c              f_K  = gas_dust - dust_cool
c              if (abs(fmin) > abs(f_K)) then
c                 kfmin = K
c              endif
c              fmin = min(abs(fmin), abs(f_K))
c              write (*,*) K, fmin, f_K, tdust_new
c            enddo
c            tdust_new = tdust + kfmin * dt
c            write (*,*) "chosen tdust_new = ", tdust_new, K, kfmin
c            goto 200
c    Find f(tdust_new)
c          write(*,*) "Bisec Method used, J, tdust, tdust_new", J,
c     $               tdust, tdust_new
          tdust = temp
          tdust_new = CMB_temp
          call kbell_lin(tdust_new, rho, kappa, dlnkappadT)
          kappa = kappa * dust_to_gas_ratio
          dust_cool = 4d0 * ssb * rho * (tdust_new**4 - CMB_temp**4)
     $              / (sigma**2 * kappa + 1d0 / kappa)
          gas_dust  = cl14 * (temp - tdust_new) * yn**2
     $              * dust_to_gas_ratio
          ggetn  = gas_dust - dust_cool
c          call kbell_lin(tdust, rho, kappa, dlnkappadT)
c          kappa = kappa * dust_to_gas_ratio
c          dust_cool = 4d0 * ssb * rho * (tdust**4 - CMB_temp**4)
c     $              / (sigma**2 * kappa + 1d0 / kappa)
c          gas_dust  = cl14 * (temp - tdust) * yn**2
c     $              * dust_to_gas_ratio
c          ggetm  = gas_dust - dust_cool
c           write(*,*) "ggetn, ggetm = ", ggetn, ggetm
          if (ggetn <= 0) then
            lo = tdust_new
            hi = tdust
          else
            lo = tdust
            hi = tdust_new
          endif
c
          mid = lo + (hi-lo)/2
          do 888 while ((mid =/ lo) .and.
                        (mid =/ hi) .and.
                        (abs(hi  lo) > 1d-3 * mid) .and.
     $                  (K < 5*niter))
            K = K + 1
            call kbell_lin(mid, rho, kappa, dlnkappadT)
            kappa = kappa * dust_to_gas_ratio
            dust_cool = 4d0 * ssb * rho * (mid**4 - CMB_temp**4)
     $                  / (sigma**2 * kappa + 1d0 / kappa)
            gas_dust  = cl14 * (temp - mid) * yn**2
     $                  * dust_to_gas_ratio
            ggetn  = gas_dust - dust_cool
            if (ggetn <= 0) then
              lo = mid
            else
              hi = mid
            endif
            mid = lo + (hi - lo)/2
            if (abs(hi  lo) < 1d-3 * mid) then
              goto 200
            endif
888       return

c                call kbell_lin(tdustp, rho, kappa, dlnkappadT)
c                kappa = kappa * dust_to_gas_ratio
c                dust_cool = 4d0 * ssb * rho * (tdustp**4 - CMB_temp**4)
c     $              / (sigma**2 * kappa + 1d0 / kappa)
c                gas_dust  = cl14 * (temp - tdustp) * yn**2
c     $              * dust_to_gas_ratio
c                ggetp  = gas_dust - dust_cool
c                if (ggetp > ggetmi) then
c                  tdust = 1.1d0 * tdust
c                else
c                  tdust = 0.9d0 * tdust
c                endif
c            write(*,*) "ggetm and tdust ", ggetm, tdust
c            write(*,*) "gas_dust, dust_cool ", gas_dust, dust_cool
c            write(*,*) "id_current, K ", id_current, K
c888           continue
c            endif
c            lo = tdust_new
c            hi = tdust
c            write(*,*) "ggetn <= 0, hi, lo = ", hi, lo
c          else
c            lo = tdust
c            hi = tdust_new
c            write(*,*) "ggetn > 0, hi, lo = ", hi, lo
c          endif
c
c          mid = lo + (hi-lo)/2
c            K = 0
c Start loop
c            do 666 while ((abs(hi - lo) < 1d-3 * mid) .and.
c     $                    (K .lt. 5*niter))
c              K = K + 1
c   'Find f(midtemp)
c              call kbell_lin(mid, rho, kappa, dlnkappadT)
c              kappa = kappa * dust_to_gas_ratio
c              dust_cool = 4d0 * ssb * rho * (mid**4 - CMB_temp**4) /
c     $                (sigma**2 * kappa + 1d0 / kappa)
c              gas_dust  = cl14 * (temp - mid) * yn**2 
c     $        * dust_to_gas_ratio
c              ggetm  = gas_dust - dust_cool
c           write(*,*) "hi, lo, ggetm =", hi, lo, ggetm
c              if (ggetm <= 0) then
c                lo = mid
c              else
c                hi = mid
c              endif
c              mid = lo + (hi-lo)/2
c          write(*,*) "mid = ", mid
c666        continue 
c           tdust_new = mid
c          write(*,*) "tdust_new, K = ", tdust_new, K
c
c Debugging
c
c           write(*,*) "Bisec Method ", tdust, tdust_new,
c     $          tdust-tdust_new, K
c
c End of debugging
c
c           goto 200
          endif
        enddo
      
        write(*,*) "Dust temp iteration failed! ", tdust, tdust_new, 
     $          tdust-tdust_new
        write(*,*) "Dust_cool, gas_dust, temp, si", dust_cool, gas_dust,
     $          temp, sigma
        write(*,*) "rho, kappa, dlnkappadT, id",rho, kappa, dlnkappadT,
     $           id_current
        write(*,*) "tdust_new = tdust - F_nr / dF_nr", tdust_new, "=",
     $   tdust, "-", F_nr, "/", dF_nr
        stop
      else
        tdust = temp
          write(*,*) tdust, rho, kappa, dlnkappadT
      endif
c
200   continue
#endif
c
c Chemical heating & cooling
c
c
c H2 collisional dissociation cooling:
c
      if (no_chem .gt. 0) then
        ylam_chem = 0d0
      else
      rates_chem(1) = 4.48d0 * eV * (ykdh  * abHI * abh2 + 
     $                               ykdh2 * abh2 * abh2 +
     $                               ykdhe * abheI * abh2 +
     $                               ch25  * abe  * abh2 ) * ynsq
c
c H2 destruction by charge transfer with HII -- endothermic by 1.83eV 
c
      rates_chem(2) = 1.83d0 * eV * ch24 * abh2 * abhp * ynsq
c
c H2 photodissociation heating -- 0.4eV per photodissociation:
c
      rates_chem(3) = - 4d-1 * eV * abH2 * yn * ykdis
c
c Heating by UV pumping of H2 -- effective for n > n_crit
c
c We assume 2 eV per UV photon (Burton, Hollenbach & Tielens 1990),
c and adopt a pumping rate 8.5 times larger than the dissociation rate
c (Draine & Bertoldi 1996, table 2). 
c
      rates_chem(4) = - 2d0 * eV * abH2 * yn * ykdis * 8.5d0 * 
     $                  (1d0 - h2var0)
c
c H2 formation heating -- we assume a similar depedence on n / n_crit to 
c the collisional dissociation rates, although ideally one would like to
c do a more careful, level-dependent treatment.
c
c Note that energy input depends on formation mechanism:
c
c H-   -- 3.73eV
c H2+  -- 1.83eV
c 3-body -- 4.48eV
c
      rates_chem(5) = - 3.73d0 * eV * ch22 * abHI * abhm * ynsq
     $                * (1d0 - h2var0)
c
      rates_chem(6) = - 1.83d0 * eV * cst(1) * abHI * abh2p * 
     $                ynsq * (1d0 - h2var0)
c
      rates_chem(7) = - 4.48d0 * eV * (ch108 * abHI**3 + 
     $                  ch109 * abHI**2 * abh2 + 
     $                  ch110 * abHI**2 * abheI) * ynsq * yn
c
c HI collisional ionization cooling -- 13.6eV per ionization:
c
      rates_chem(8) = 13.6d0 * eV * ch11 * abe * abHI * ynsq
c
c HeI collisional ionization cooling -- 24.6 eV per ionization
c
      rates_chem(9) = 24.6d0 * eV * ch14 * abe * abheI * ynsq
c
c HeII collisional ionization cooling -- 54.4 eV per ionization
c
      rates_chem(10) = 54.4d0 * eV * ch15 * abe * abhep * ynsq
c
c HII recombination cooling -- we assume that kT ergs is radiated 
c per recombination and that this is independent of whether the
c recombination occurs in the gas phase or on a grain surface
c
      rates_chem(11) = kboltz * temp * abhp * ch13 * abe * 
     $                 ynsq
c
c HeII recombination cooling. Dielectronic recombination results
c in the loss of ~ 40.8eV per recombination, while for standard 
c radiative recombination, we make the same assumption as for HII
c
c Dielectronic:
      rates_chem(12) = 40.8d0 * eV * ch10 * abhep * abe * ynsq
c
c Radiative:
      rates_chem(13) = kboltz * temp * (ch17 - ch10) * abhep * abe * 
     $                 ynsq
c
c HeIII recombination cooling -- see note for HII
c
      rates_chem(14) = kboltz * temp * abhepp * ch20 * abe * 
     $                 ynsq
c
c H- photodissociation heating
c
      rates_chem(15) = - ph8 * abhm * yn
c
c H2+ photodissociation heating
c
      rates_chem(16) = - ph9 * abh2p * yn
c
c HI photoionization heating
c 
      rates_chem(17) = - ph42 * abHI * yn
c
c HeI photoionization heating
c 
      rates_chem(18) = - ph43 * abheI * yn
c
c HeII photoionization heating
c 
      rates_chem(19) = - ph44 * abhepp * yn
c
c H2 photoionization heating
c
      rates_chem(20) = - ph45 * abh2 * yn
c
      ylam_chem = 0d0
      do I = 1, nrates_chem
        ylam_chem = ylam_chem + rates_chem(I)
      enddo
      endif
c
c Initialize all abundances to zero, then just set the non-zero ones
c
      do i = 1, nabn
        abundances(i) = 0d0
      enddo
c
      if (no_dchem .eq. 1) then
        abhd = min(abundD, exp(4.88d2 / temp) * abh2)
      endif
c
      abundances(1)  = abh2
      abundances(10) = abe
      abundances(11) = abhp
      abundances(12) = abHI
      abundances(13) = abhd
      abundances(14) = abheI
      abundances(15) = abhep
      abundances(16) = abhepp
c
#ifdef ADIABATIC_DENSITY_THRESHOLD
      if (yn .ge. yn_adiabatic) then
        ydot(itmp) = 0d0
#ifdef DEBUG_RATE_EQ
        call validate_output(y,ydot,rpar)
#endif
        return
      endif
#endif
      call cool_func(temp, yn, L_jeans, divv, 0d0, abundances, ylam, 
     $               rates, 0)
c
c Now correct for effects of CMB:
c
      if (redshift .gt. 0d0) then
        call cool_func(CMB_temp, yn, L_jeans, divv, 0d0, abundances,
     $                 ylam_cmb, rates_cmb, 1)
      else
        ylam_cmb = 0d0
      endif

c
c RadHeat addition
c

      if (irad_heat .eq. 1) then
        total_rad_heating = 0d0
        if (nradsource .gt. 0) then
          do I = 1, nradsource
            distance = rad_source_distances(I)
            Lstar    = rad_source_luminosities(I)
            call compute_heating(Lstar, distance, temp, rho, 
     $                           heating_rate)
            total_rad_heating = total_rad_heating + heating_rate
          enddo
        endif
      else
        total_rad_heating = 0d0
      endif
c

crjs
c      if (id_current==1866845) then
c      if (total_rad_heating.eq.0.and.rho.gt.1.e-12) then
c         PRINT*,"ID= ",id_current," Heating zero"
c         PRINT*," distance=", distance, " Lstar" ,Lstar  
c         PRINT*,"tot_rad_heating = ", total_rad_heating,temp,rho
c      end if

      ydot(itmp) = -(ylam - ylam_cmb + ylam_chem) + pdv_term 
     $           + total_rad_heating




#ifdef THERMAL_INFO_DUMP
      if (abs(ydot(itmp) - pdv_term) .gt. 0d0) then
        dtcool_nopdv = y(itmp) / (ydot(itmp) - pdv_term)
      else
        dtcool_nopdv = 1d20
      endif
      cool_h2_line = rates(2) - rates_cmb(2)
      cool_h2_cie  = rates(20) - rates_cmb(20)
      cool_h2_diss = rates_chem(1)
      heat_3b      = rates_chem(5) + rates_chem(6) + rates_chem(7)
      pdv_heat     = pdv_term
crjs
      acc_heat    = total_rad_heating
#endif
c
#ifdef FF_COLLAPSE_TEST
      ndot = yn**1.5d0 * tffc
c
      compress = ((1.1d0 - abh2) * BOLTZMANN * temp) * ndot
      ydot(itmp) = ydot(itmp) + compress + 
     $             (ndot / yn) * y(itmp)
#endif

#ifdef DEBUG_RATE_EQ
      call validate_output(y,ydot,rpar)
#endif

#ifdef FF_COLLAPSE_TEST
      if (itest .gt. 0) then
        tff = 1d0 / (tffc * dsqrt(yn)) 
        do I = 1, num_eqb_species
          print*, I, time_to_equilibrium(I) / tff
        enddo
        itest = 0
      endif
#endif
c
#endif /* CHEMCOOL */
      return
      end
c=======================================================================
c
c    \\\\\\\\\\        E N D   S U B R O U T I N E        //////////
c    //////////              R A T E _ E Q                \\\\\\\\\\
c
c=======================================================================
c=======================================================================
c
c    \\\\\\\\\\        B E G I N   F U N C T I O N        //////////
c    //////////          C O M P _ E Q U I L I B          \\\\\\\\\\
c
c=======================================================================
c
#ifdef CHEMCOOL
      REAL function comp_equilib(num, denom, max_abundance, id, yn, 
     $                           time_to_equilibrium)
      implicit NONE
#include "cool.h"
#include "non_eq.h"
       REAL num, denom, max_abundance, yn
       REAL time_to_equilibrium(num_eqb_species)
       integer id
c
       if (num .eq. 0d0) then
         comp_equilib = 0d0
       elseif (denom .eq. 0d0) then
c If this happens, then our assumption that this species is in
c equilibrium has fatally broken down...
         print*, "Creation without destruction in comp_equilib"
         print*, "Species = ", id
         ABORT(9751)
      else
#ifdef COLLECT_EQB_DATA
         time_to_equilibrium(id) = 1d0 / (denom * yn)
#endif
        comp_equilib = num / denom
        if (comp_equilib .gt. max_abundance) then
          print*, "Error: impossible abundance!"
          print*, "Species ", id
          print*, num, denom, num/denom
          ABORT(9752)
        endif
      endif
c
      return
      end
#endif /* CHEMCOOL */
c=======================================================================
c
c    \\\\\\\\\\          E N D     F U N C T I O N        //////////
c    //////////          C O M P _ E Q U I L I B          \\\\\\\\\\
c
c=======================================================================
c=======================================================================
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////             K B E L L _ L I N             \\\\\\\\\\
c
c=======================================================================
      subroutine kbell_lin(T, rho, kr, dlnkdt)
c Opacity law parameters (from Bell & Lin 1994) "K0", "a", and "b".
      implicit none
      integer k, kri
      real*8 T, rho, kr, dlnkdt, Tij, k0(8), a(8), b(8)
c Opacity law parameters (from Bell & Lin 1994) "K0", "a", and "b".
      k0 = (/2d-4, 2d+16, 1d-1, 2d+81, 1d-8, 1d-36, 1.5d+20, 0.348d+0/)
      a  = (/0d0, 0d0, 0d0, 1d0, 0.666667d0, 0.333333d0, 1d0, 0d0/)
      b  = (/2d0, -7d0, 0.5d0, -24d0, 3d0, 10d0, -2.5d0, 0d0/)
c Find the Temperature for the transition between each opatity component or physical process
      k = 0
      Tij = 0d0
c
      do 555 while ((T > Tij).and.(k<8))
         k = k + 1
         kri = k
         Tij = ((k0(k+1)/k0(k))*(rho**(a(k+1) - a(k))))**
     $        (1/(b(k) - b(k+1)))
c         write(*,*) kri, Tij, T, rho
555      continue
c Calculates the opacity according to the opacity component or physical process "kri", using the formula "kr = k0 (rho**a) (T**b)". I used log10 because fortran did not like double precision numbers =P
      kr = k0(kri) * (rho**a(kri)) * (T**b(kri))
      dlnkdt = b(kri)/T
c      write(*,*) "Subroutine KBellLin", T, rho, kr, dlnkdt
c
      return
      end
c=======================================================================
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////             K B E L L _ L I N             \\\\\\\\\\
c
c=======================================================================
#endif /* CHEMISTRYNETWORK == 1 */

