c=======================================================================
c
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////              C O O L I N M O              \\\\\\\\\\
c
c=======================================================================
c
       subroutine coolinmo
#ifdef CHEMCOOL
c
c    Based on an original routine by G. Suttner (University Wuerzburg, 1995)
c    CO, H2O, OH, OI, CI cooling added by M. D. Smith 
c       (Armagh Observatory, 2000-2001)
c    Extensively rewritten by S. Glover (AMNH, 2003-2005)
c    -- fine structure cooling from CII, SiI, SiII added
c    -- fine structure cooling treatment is now exact & includes effects of CMB
c    -- new H2 cooling function used (taken from Le Bourlot et al 1999)
c    -- high T metal-free cooling from Sutherland & Dopita (1993)
c    -- many other minor changes
c
c  PURPOSE:  Tabulate cooling function for various ions, atoms and molecules
c            before starting run
c
c  REFERENCES:
c       FL77  -- Flower & Launay, 1977, J. Phys. B, 10, 3673
c       S78   -- Spitzer, 1978, 'Physical Processes in the Interstellar Medium'
c       HM79  -- Hollenbach and McKee, 1979, ApJS, 41, 555
c       B81   -- Black, 1981, MNRAS, 197, 553 
c       DRD83 -- Draine, Roberge & Dalgarno, 1983, ApJ, 264, 485
c       K86   -- Keenan et al, 1986, MNRAS, 220, 571
c       SK87  -- Shapiro & Kang, 1987, ApJ, 318, 32
c       JBK87 -- Johnson, Burke & Kingston, 1987, J. Phys B, 20, 2553
c       HM89  -- Hollenbach & McKee, 1989, ApJ, 342, 306
c       G90   -- Gerlich, 1990, J. Chem. Phys., 92, 2377
c       RLB90 -- Roueff & Le Bourlot, 1990, A&A, 236, 515
c       P90   -- Pequignot, 1990, A&A, 231, 499
c       R90   -- Roueff, 1990, A&A, 234, 567
c       DK91  -- Dufton & Kingston, 1991, MNRAS, 248, 827
c       S91   -- Schroeder et al, 1991, J. Phys. B, 24, 2487
c       C92   -- Cen, 1992, ApJS, 78, 341 
c       SD93  -- Sutherland & Dopita, 1993, ApJS, 88, 253
c       NK93  -- Neufeld & Kaufman, 1993, ApJ, 418, 263
c       NLM95 -- Neufeld, Lepp & Melnick, 1995, ApJS, 100, 132
c       WBV96 -- Warin, Benayoun & Viala, 1996, A&A, 308, 535
c       BBT98 -- Bell, Berrington & Thomas, 1998, MNRAS, 293, L83
c       FRZ98 -- Flower, Roueff & Zeippen, 1998, J. Phys. B, 31, 1105
c       BFD99 -- Balakrishnan, Forrey & Dalgarno, 1999, ApJ, 514, 520
c       LPF99 -- Le Bourlot, Pineau des Forets & Flower, 1999, MNRAS, 305, 802
c       BBFT00 -- Bray et al, 2000, A&AS, 146, 481
c       BYD02 -- Balakrishnan, Yan & Dalgarno, 2002, ApJ, 568, 443
c       CP02  -- Cecchi-Pestellini et al, 2002, ApJ, 571, 1015
c       K02   -- Krstic, 2002, Phys. Rev. A, 66, 042717
c       WB02  -- Wilson & Bell, 2002, MNRAS, 337, 1027
c       RA04  -- Ripamonti & Abel, 2004, MNRAS, 348, 1019
c       LNA05 -- Lipovka, Nunez-Lopez & Avila-Reese, MNRAS, 361, 850
c       YOHA06 -- Yoshida et al, 2006, ApJ, 652, 6
c       AKD07 -- Abrahamsson, Krems & Dalgarno, 2007, ApJ, 654, 1171
c       WF07  -- Wrathmall & Flower, 2007, J. Phys. B, 40, 3221
c
      implicit NONE
#include "cool.h"
#include "mol_data.h"
#include "fs_data.h"
c
       integer  i, j, natom, natom2, itemp, idx1, idx2
c
       parameter(natom = 81)
       REAL coolatom(natom), coolatom_temp(natom)
       parameter(natom2 = 76)
       REAL ca2(natom2), ca2_temp(natom2)
c
c Raw data -- extracted from DATA tables in mol_data.h
c
       REAL co_lte_raw(nco_temp), co_n05_raw(nco_temp), 
     $      co_alp_raw(nco_temp)
       REAL co_vib_LTE_raw(nco_vib_temp)
c
       REAL h2o_LTE_raw_ortho(nh2o_temp), 
     $      h2o_n05_raw_ortho(nh2o_temp), 
     $      h2o_alp_raw_ortho(nh2o_temp)
       REAL h2o_LTE_raw_para(nh2o_temp), 
     $      h2o_n05_raw_para(nh2o_temp),
     $      h2o_alp_raw_para(nh2o_temp)
      REAL  h2o_vib_LTE_raw(nh2o_vib_temp)
c
c Temporary variables used in fitting procedure
c
c CO rotational cooling:
c
       REAL co_lte_fit(nTco), co_alp_fit(nTco), co_n05_fit(nTco)
       REAL co_lte_fit2(ncdco), co_n05_fit2(ncdco), co_alp_fit2(ncdco)

       REAL co_lte_fxT(nco_column), co_n05_fxT(nco_column),
     $      co_alp_fxT(nco_column)

       REAL co_lte_smalltab(nco_column,nTco), 
     $      co_n05_smalltab(nco_column,nTco),
     $      co_alp_smalltab(nco_column,nTco)
c
c H2O rotational cooling:
c
       REAL h2o_LTE_fit_ortho(nTh2o), h2o_alp_fit_ortho(nTh2o), 
     $      h2o_n05_fit_ortho(nTh2o), h2o_LTE_fit_para(nTh2o), 
     $      h2o_alp_fit_para(nTh2o),  h2o_n05_fit_para(nTh2o)

       REAL h2o_LTE_fit2_ortho(ncdh2o), h2o_n05_fit2_ortho(ncdh2o), 
     $      h2o_alp_fit2_ortho(ncdh2o), h2o_LTE_fit2_para(ncdh2o), 
     $      h2o_n05_fit2_para(ncdh2o),  h2o_alp_fit2_para(ncdh2o)

       REAL h2o_LTE_fxT_ortho(nh2o_column), 
     $      h2o_n05_fxT_ortho(nh2o_column),
     $      h2o_alp_fxT_ortho(nh2o_column),
     $      h2o_LTE_fxT_para(nh2o_column), 
     $      h2o_n05_fxT_para(nh2o_column),
     $      h2o_alp_fxT_para(nh2o_column)

       REAL h2o_LTE_smalltab_ortho(nh2o_column,nTh2o), 
     $      h2o_n05_smalltab_ortho(nh2o_column,nTh2o),
     $      h2o_alp_smalltab_ortho(nh2o_column,nTh2o),
     $      h2o_LTE_smalltab_para(nh2o_column,nTh2o), 
     $      h2o_n05_smalltab_para(nh2o_column,nTh2o),
     $      h2o_alp_smalltab_para(nh2o_column,nTh2o)
c
c CO vibrational cooling:
c
       REAL co_vib_LTE_fit(nTco_vib), co_vib_LTE_fit2(ncdco_vib),
     $      co_vib_LTE_fxT(nco_vib_column)
       REAL co_vib_LTE_smalltab(nco_vib_column, nTco_vib)
c
c H2O vibrational cooling:
c
       REAL h2o_vib_LTE_fit(nTh2o_vib), h2o_vib_LTE_fit2(ncdh2o_vib),
     $      h2o_vib_LTE_fxT(nh2o_vib_column)
       REAL h2o_vib_LTE_smalltab(nh2o_vib_column, nTh2o_vib)
c
       REAL rate0(nmd), rate1(nmd), rate2(nmd), rate3(nmd)
c
       REAL   temp    , temp2   , f       , gg       , hh      
     $      , t4log   , dtemp   , tinv    , tinth    , tintq
     $      , tsqrt   , etinv   , blwatv  , watin    , tau
     $      , blwatr  , ctinv   , blcov   , t34      , ccr
     $      , sig     , vth     , coc     , blwatvh  , tinq
     $      , frac    , opratio , fortho  , tisqt
     $      , fpara   , atomic  , tloge   , temp3
     $      , h2e20   , h2e31   , h2n2    , h2n3     , h2q02
     $      , h2q13   , tlog    , thd     , gff      , tfinq
     $      , tfintq  , tfinth  , tfix
c
c atomic cooling:  table of SD93 for zero radiation field, non-equil., 
c Fe = -0.5 (table 10)
c
      DATA coolatom /-68.,-67.,-66.,-65.,-64.,-63.,-62.,-61.,-60.,-59.,
     *     -58.,-57.,-56.,-55.,-54.,-53.,-52.,-51.,-50.,-49.,-48.,-47.,
     *     -46.,-45.,-44.,-43.,-42.,-41.,-40.,-39.,-38.,-37.,-36.,-35.,
     *     -34.,-33.,-32.,-31.,-30.,-26.67,-23.16,-22.83,-22.53,-22.30,
     *     -22.20,-22.11,-22.00,-21.88,-21.77,-21.69,-21.60,-21.54,
     *     -21.53,-21.58,-21.73,-21.91,-22.00,-22.07,-22.23,-22.32,
     *     -22.35,-22.34,-22.43,-22.62,-22.75,-22.80,-22.83,-22.86,
     *     -22.85,-22.84,-22.85,-22.87,-22.87,-22.86,-22.85,-22.83,
     *     -22.79,-22.75,-22.70,-22.65,-22.60/
c
c atomic cooling #2: from SD93 -- zero radiation field, zero metals,
c collisional ionization equlibrium assumed. Does not include effects
c of HI collisional excitation. [Based on table 6 of SD93]
c
c NB. Data runs from log(T) = 4.3 - 8.0 in increments of 0.05.
c     First value corresponds to T = T_min, but is utterly arbitrary 
c     -- this is here purely to make the spline fitting somewhat 
c     easier, but as the actual entries for T < 10**4.3 K are handled
c     specially below, it should not significantly affect the computed
c     rates. 
c
      DATA ca2 /-60.00, -23.38, -23.17, -23.06, -22.95, -22.89, -22.89, 
     *          -22.93, -22.93, -22.86, -22.69, -22.47, -22.27, -22.18, 
     *          -22.19, -22.27, -22.36, -22.46, -22.56, -22.66, -22.74, 
     *          -22.82, -22.89, -22.95, -23.01, -23.05, -23.09, -23.13, 
     *          -23.16, -23.18, -23.20, -23.22, -23.23, -23.24, -23.25, 
     *          -23.25, -23.25, -23.25, -23.25, -23.25, -23.24, -23.23, 
     *          -23.23, -23.23, -23.21, -23.20, -23.19, -23.18, -23.16, 
     *          -23.15, -23.13, -23.11, -23.10, -23.08, -23.06, -23.05, 
     *          -23.03, -23.01, -22.99, -22.97, -22.95, -22.93, -22.91, 
     *          -22.89, -22.86, -22.84, -22.82, -22.80, -22.78, -22.76, 
     *          -22.74, -22.71, -22.69, -22.67, -22.65, -22.62/
c
c Sanity checks
c
      if (phi_pah .eq. 0d0) then
        write(*,'(A)') 'Warning: phi_pah must be non-zero'
        ABORT(12)
      endif
c
c H2 ortho-para ratio (N.B. Only used for OI fine structure cooling)
c
      opratio = 2.4d0
      fortho  = opratio / (1.d0 + opratio)
      fpara   = 1.d0 / (1.d0 + opratio)
c
c establish temperature table     
c
      tmin   = 1.d0
      tmax   = 1.d8
      dtlog  = dlog10(tmax) / (nmd - 1)
c
      do 100 itemp = 1, nmd
        temptab(itemp) = 1.D1**( (itemp - 1) * dtlog )
100   continue
c
c tabulate cooling functions for each temp
c
      do 200 itemp = 1, nmd
         temp  = temptab(itemp)
         temp2 = temp * 1d-2
         temp3 = temp * 1d-3
         tau   = temp * 1d-3 + 1d0
         tinv  = 1.d0 / temp
         tinth = tinv**(1d0/3d0)
         tinq  = tinv**0.25d0
         tintq = tinv**0.75d0
         tsqrt = dsqrt(temp)
         tisqt = 1.d0 / tsqrt
         tlog  = dlog10(temp)
         t4log = tlog - 4d0
         tloge = log(temp)
c
c cl[1-3]: H2 cooling (LPF99)
c
c   Values in range 100 -- 10000K are handled at end. 
c
c   Rate at T > 10000K is arbitrarily fixed to be the same as at 10000K
c   (this is incorrect, but simple, and in any case Lyman-alpha will 
c    dominate at these temperatures)
c
c   Below 100K, we assume that all cooling comes from the J=2-0, 3-1 
c   transitions in the vibrational ground state.
c
c   All rates assume an ortho:para ratio of 3:1 
c
c (cl1) -- LTE rate
c
         h2e20 = 508.95d0
         h2e31 = 852.5d0
c
         if (temp .lt. 5d0) then
           cltab(1,itemp) = 1d-60
         else if (temp .lt. 1d2) then
           h2n2 = 0.25d0 * (5d0 * dexp(-h2e20 / temp) /
     $            (1d0 + 5d0 * dexp(-h2e20 / temp)))
           h2n3 = 0.75d0 * ((7d0 / 3d0) * dexp(-h2e31 / temp) /
     $            (1d0 + (7d0 / 3d0) * dexp(-h2e31 / temp)))
           f    = 2.94d-11 * h2e20 * h2n2 * kboltz +
     $            4.76d-10 * h2e31 * h2n3 * kboltz
           cltab(1,itemp) = max(f, 1d-60)
         else if (temp .gt. 1d4) then
           cltab(1,itemp) = 1d1**(-18.253d0)
         endif
c
c (cl2) -- H rate -- 0->2 rate tweaked slightly to ensure that we match up
c                    properly with the tabulated rates at 100K
c
         h2q02 = 1d1**(-9.121d0 - (3.983d0 / tau) - (1.844d-1 / tau**2))
     $           * 5d0 * dexp(-h2e20 / temp)
c
         h2q13 = 1d1**(-9.493d0 - (4.435d0 / tau) + (2.707d-1 / tau**2))
     $           * (7d0 / 3d0) * dexp(-h2e31 / temp)
c
         if (temp .lt. 5d0) then
           cltab(2,itemp) = 1d-60
         else if (temp .lt. 1d2) then
           cltab(2,itemp) = (fpara  * h2q02 * h2e20 + 
     $                       fortho * h2q13 * h2e31) * kboltz
         else if (temp .gt. 1d4) then
           cltab(2,itemp) = 1d1**(-21.943d0)
         endif
c
c (cl3) -- H2 rate
c
         h2q02 = 1d1**(-9.946d0 - (2.688d0 / tau) + (2.020d-1 / tau**2))
     $           * 5d0 * dexp(-h2e20 / temp)
c
         h2q13 = 1d1**(-9.785d0 - (3.024d0 / tau) + (2.930d-1 / tau**2))
     $           * (7d0 / 3d0) * dexp(-h2e31 / temp)
c
         if (temp .lt. 5d0) then
           cltab(3,itemp) = 1d-60
         else if (temp .lt. 1d2) then
           cltab(3,itemp) = (fpara  * h2q02 * h2e20  + 
     $                       fortho * h2q13 * h2e31) * kboltz
         else if (temp .gt. 1d4) then
           cltab(3,itemp) = 1d1**(-22.758d0)
         endif
c 
c (cl4) -- H2-H, low density rate: based on WF07, assumes 3:1 o:p ratio
c     
c Data spans range 100 < T < 6000K, but is well-behaved up to 10000K;
c at higher T, we assume rate remains constant (but note that at these
c temperatures, atomic cooling dominates)
c
         if (temp .lt. 1d1) then
           cltab(4,itemp) = 0d0
         elseif (temp .lt. 1d2) then
           cltab(4,itemp) = 1d1**(-16.818342d0
     $                    + 37.383713d0 * dlog10(temp3)
     $                    + 58.145166d0 * dlog10(temp3)**2
     $                    + 48.656103d0 * dlog10(temp3)**3
     $                    + 20.159831d0 * dlog10(temp3)**4
     $                    + 3.8479610d0 * dlog10(temp3)**5)
         elseif (temp .lt. 1d3) then
           cltab(4,itemp) = 1d1**(-24.311209d0
     $                    + 3.5692468d0 * dlog10(temp3)
     $                    - 11.332860d0 * dlog10(temp3)**2
     $                    - 27.850082d0 * dlog10(temp3)**3
     $                    - 21.328264d0 * dlog10(temp3)**4
     $                    - 4.2519023d0 * dlog10(temp3)**5)
         elseif (temp .lt. 1d4) then
           cltab(4,itemp) = 1d1**(-24.311209d0
     $                    + 4.6450521d0 * dlog10(temp3)
     $                    - 3.7209846d0 * dlog10(temp3)**2
     $                    + 5.9369081d0 * dlog10(temp3)**3
     $                    - 5.5108047d0 * dlog10(temp3)**4
     $                    + 1.5538288d0 * dlog10(temp3)**5)
         else
           cltab(4,itemp) = 1d1**(-24.311209d0
     $                    + 4.6450521d0
     $                    - 3.7209846d0
     $                    + 5.9369081d0
     $                    - 5.5108047d0
     $                    + 1.5538288d0)
         endif
c
c (cl5) -- H2-H2, low density rate: based on LPF99, assumes 3:1 o:p ratio
c
c At T > 10000K, assume rate same as at T=10000K
c
        if (temp .lt. 1d1) then
          cltab(5,itemp) = 0d0
        elseif (temp .lt. 1d4) then 
          cltab(5,itemp) = 1d1**(-23.962112d0
     $                    + 2.09433740d0  * dlog10(temp3)
     $                    - 0.77151436d0  * dlog10(temp3)**2
     $                    + 0.43693353d0  * dlog10(temp3)**3
     $                    - 0.14913216d0  * dlog10(temp3)**4
     $                    - 0.033638326d0 * dlog10(temp3)**5)
        else
          cltab(5,itemp) = 1d1**(-23.962112d0
     $                    + 2.09433740d0
     $                    - 0.77151436d0
     $                    + 0.43693353d0
     $                    - 0.14913216d0
     $                    - 0.033638326d0)
        endif
c
c (cl64) -- H2-He, low density rate: based on FRZ98, BFD99; assumes 3:1 o:p ratio
c
        if (temp .lt. 1d1) then
          cltab(64,itemp) = 0d0
        elseif (temp .lt. 1d4) then
          cltab(64,itemp) = 1d1**(-23.689237d0
     $                    + 2.1892372d0  * dlog10(temp3)
     $                    - 0.81520438d0  * dlog10(temp3)**2
     $                    + 0.29036281d0 * dlog10(temp3)**3
     $                    - 0.16596184d0 * dlog10(temp3)**4
     $                    + 0.19191375d0 * dlog10(temp3)**5)
        else
          cltab(64,itemp) = 1d1**(-23.703432d0
     $                    + 2.2601756d0
     $                    - 1.0736440d0
     $                    + 0.66060709d0
     $                    - 0.26963286d0
     $                    + 0.12261185d0)
        endif
c
c (cl65) -- H2-H+, low density rate: based on G90, K02; assumes 3:1 o:p ratio
c
c Fit accurate to within 5%.
c
        if (temp .lt. 1d1) then
          cltab(65,itemp) = 0d0
        elseif (temp .lt. 1d4) then
          cltab(65,itemp) = 1d1**(-21.716699d0
     $                    + 1.3865783d0   * dlog10(temp3)
     $                    - 0.37915285d0  * dlog10(temp3)**2
     $                    + 0.11453688d0  * dlog10(temp3)**3
     $                    - 0.23214154d0  * dlog10(temp3)**4
     $                    + 0.058538864d0 * dlog10(temp3)**5)
        else
          cltab(65,itemp) = 1d1**(-21.716699d0
     $                    + 1.3865783d0   * dlog10(temp3)
     $                    - 0.37915285d0  * dlog10(temp3)**2
     $                    + 0.11453688d0  * dlog10(temp3)**3
     $                    - 0.23214154d0  * dlog10(temp3)**4
     $                    + 0.058538864d0 * dlog10(temp3)**5)
        endif
c
c (cl66) -- H2-e, low density rate: based on DRD83, assumes 3:1 o:p ratio
c
        if (temp .lt. 1d1) then
          cltab(66,itemp) = 0d0
        elseif (temp .lt. 2d2) then
          cltab(66,itemp) = 1d1**(-34.286155d0
     $                    - 48.537163d0  * dlog10(temp3)
     $                    - 77.121176d0  * dlog10(temp3)**2
     $                    - 51.352459d0  * dlog10(temp3)**3
     $                    - 15.169160d0  * dlog10(temp3)**4
     $                    - 0.98120322d0 * dlog10(temp3)**5)
        elseif (temp .lt. 1d4) then
          cltab(66,itemp) = 1d1**(-22.190316
     $                    + 1.5728955  * dlog10(temp3)
     $                    - 0.21335100 * dlog10(temp3)**2
     $                    + 0.96149759 * dlog10(temp3)**3
     $                    - 0.91023195 * dlog10(temp3)**4
     $                    + 0.13749749 * dlog10(temp3)**5)
        else
          cltab(66,itemp) = 1d1**(-22.190316
     $                    + 1.5728955  
     $                    - 0.21335100 
     $                    + 0.96149759 
     $                    - 0.91023195 
     $                    + 0.13749749)
        endif
c
c (cl67): Equilibrium J=1/J=0 H2 ratio (used to compute heating/cooling 
c from J = 0 <-> J=1 transitions in cool_func
c
        cltab(67,itemp) = 9d0 * exp(-170.5d0 / temp)
c
c (cl6) --  the atomic cooling function - this is computed by fitting
c           a cubic spline to the data specified in coolatom above, and
c           so is calculated after the main loop is done
c
c (cl[7-9]) --  H2O cooling
c
c MDS  4.2.2001: completely revised
c MDS 10.6.2001: rerevised. H20 and CO both in equilibrium
c
c For H2O vibrational cooling, we can use the low density, op. thin rate
c with little to no error for densities n < 10^10 cm^-3 -- to see this
c for H2, compare L_0, L_lte numbers in table 5 of NK93; for H, substitute
c L_0 as given by HM89
c
c (r) H2O: for H2 and H collisions
c
         watin          = 1.35d0 - 0.3d0 * dlog10(0.001 * temp)
         blwatr         = 1.32d-23 * (0.001 * temp) ** watin
c
         cltab(7,itemp) = blwatr
c
c (v) H2O: for H2 collisions only (NK93)
c
         etinv          = dexp( -2.325d3 * tinv )
         blwatv         = 1.03d-26 * temp * etinv * dexp(-47.5 * tinth)
c
         cltab(8,itemp) = blwatv
c
c (v) H2O: for H collisions only  (HM89)
c
         blwatvh        = 0.74d-26 * temp * etinv * dexp(-34.5 * tinth)
c
         cltab(9,itemp) = blwatvh
c
c (cl[10-13]) -- CO cooling (McKee et al 1982 (APJ 259, 647))
c
c (r) CO: for H2 and H collisions
c
         t34             = (0.001d0 * temp)**0.75
         ccr             = 3.3d+06 * t34
         sig             = 3.0d-13 * t34 * tinv
         vth             = tsqrt * 1.03d+04 
         coc             = 0.5d0 * kboltz * temp * sig * vth 
c
c modification by AR (after discussion with MDS) based on
c statement of errors in McKee et al. 1982 about the accuracy
c of these values (between equations 5.4 and 5.5 on p. 655)
c
         if(itemp.le.2800) then
c
c This addition for the full run of A3, which had the segmentation
c fault about half way through, still needed an even more drastic
c change
c               7/3/01
c
            if(itemp.le.1626) then
               frac = 2.014d0 * (t4log) + 5.056d0
               frac = 10.0d0**(frac)
c     
               cltab(10,itemp) = coc * frac
c     
            else
               frac = 0.4d0*(t4log) + 0.7d0
               frac = 10.0d0**(frac)
c     
               cltab(10,itemp) = coc * frac
c     
            endif
         else
c     
            cltab(10,itemp) = coc
c
         endif
c
         cltab(11,itemp) = ccr

c
c Similar to H2O, but smaller n_crit; the optically thin rate is a good 
c approx. only for n < 10^8 cm^-3.
c

c
c (v) CO: for H2 collisions only (NK93)
c
         ctinv           = dexp(-3.080d3 * tinv)
         blcov           = 1.83d-26 * temp * ctinv
     $                     * dexp(-68.0 * tinth)
c
         cltab(12,itemp) = blcov
c
c (v) CO: for H collisions only
c
c Analytical fit by SCOG to data of BYD02. Fit is accurate to within
c 15% for T > 700 K, to within ~50% at lower T. Note, however, that
c vibrational cooling is unlikely to be important compared to rotational
c at low T, so the larger error there is probably unimportant. If it 
c does prove to be significant, it would not be difficult to come up
c with a better (albeit more complicated) fit
c
         if (temp .lt. 7d2) then
           cltab(13,itemp) = 1.049d-35 * temp**3.98 * ctinv
         else
           cltab(13,itemp) = 1.062d-29 * temp**1.87 * ctinv
         endif
c
c NB Comparison of results of CP02 for CO-He collisions with those
c of BYD02 for CO-H collisions suggests that collisions with He are
c unimportant compared to those with H; we therefore don't need to
c tabulate a rate for CO-He collisions.
c
c (cl14) -- gas-grain cooling (HM89, eqn 2.15)
c
         gg              = 1d0 - 0.8 * dexp(-75d0 * tinv)
c
         cltab(14,itemp) = 3.8d-33 * tsqrt * gg
c
c (cl15) --  OH cooling - non-lte
c like water cooling: HM79 universal cooling law
c
         cltab(15,itemp) = 2.84d-28 * temp * tsqrt
c
c (cl[16-17]) -- Photoelectric heating
c
         cltab(16,itemp) = dsqrt(temp) / phi_pah
c
         cltab(17,itemp) = 3.7d-2 * (temp / 1d4)**0.7d0
c
c Fine structure cooling -- three-level atoms
c
c We use the convention that the ground state is level 0, the first
c excited state is level 1, and the second excited state is level 2.
c
c We tabulate here the collisional de-excitation rates for all relevent
c collision partners. Everything else is taken care of in the cooling
c function itself at run time.
c
c (cl[18-29]): OI fine-structure lines 
c
c Collisional de-excitation rates: 
c                    
c HI: taken from AKD07 below 1000K, extended to higher temperatures
c     with a simple power-law extrapolation
c
c 1 -> 0
         if (temp .lt. 5d0) then
           tfix = 5d0
           tfintq = 1d0 / tfix**0.75d0 
           cltab(18, itemp) = (5d-11 / 3d0) * exp(4.581
     $                      - 156.118       * tfintq
     $                      + 2679.979      * tfintq**2
     $                      - 78996.962     * tfintq**3
     $                      + 1308323.468   * tfintq**4
     $                      - 13011761.861  * tfintq**5
     $                      + 71010784.971  * tfintq**6
     $                      - 162826621.855 * tfintq**7)
     $                      * exp(oxe10 / (kboltz * tfix))
         elseif (temp .lt. 1d3) then
           cltab(18, itemp) = (5d-11 / 3d0) * exp(4.581
     $                      - 156.118       * tintq
     $                      + 2679.979      * tintq**2
     $                      - 78996.962     * tintq**3
     $                      + 1308323.468   * tintq**4
     $                      - 13011761.861  * tintq**5
     $                      + 71010784.971  * tintq**6
     $                      - 162826621.855 * tintq**7)
     $                      * exp(oxe10 / (kboltz * temp))
         else
           cltab(18, itemp) = 6.81d-11 * temp**0.376d0
         endif
c 2 -> 0
         if (temp .lt. 5d0) then
           tfix = 5d0
           tfintq = 1d0 / tfix**0.75d0
           cltab(19, itemp) = 5d-11 * exp(3.297
     $                      - 168.382       * tfintq
     $                      + 1844.099      * tfintq**2
     $                      - 68362.889     * tfintq**3
     $                      + 1376864.737   * tfintq**4
     $                      - 17964610.169  * tfintq**5
     $                      + 134374927.808 * tfintq**6
     $                      - 430107587.886 * tfintq**7)
     $                      * exp(oxe20 / (kboltz * tfix)) 
         elseif (temp .lt. 1d3) then
           cltab(19, itemp) = 5d-11 * exp(3.297
     $                      - 168.382       * tintq
     $                      + 1844.099      * tintq**2
     $                      - 68362.889     * tintq**3
     $                      + 1376864.737   * tintq**4
     $                      - 17964610.169  * tintq**5
     $                      + 134374927.808 * tintq**6
     $                      - 430107587.886 * tintq**7)
     $                      * exp(oxe20 / (kboltz * temp))
         else
           cltab(19, itemp) = 6.34d-11 * temp**0.36d0
         endif
c
c 2 -> 1:  Low T extrapolation here is necessary because the AKD07 
c          fitting function blows up
c
         if (temp .lt. 5d1) then
           cltab(20, itemp) = 2.62d-12 * temp**0.74d0
         elseif (temp .lt. 1d3) then
           cltab(20, itemp) = 3d-11 * exp(3.437
     $                      + 17.443    * tisqt
     $                      - 618.761   * tisqt**2
     $                      + 3757.156  * tisqt**3
     $                      - 12736.468 * tisqt**4
     $                      + 22785.266 * tisqt**5
     $                      - 22759.228 * tisqt**6
     $                      + 12668.261 * tisqt**7)
     $                      * exp(oxe21 / (kboltz * temp))
         else
           cltab(20,itemp) = 3.61d-10 * temp**0.158d0
         endif
c
c H2 rates supplied by Flower (priv. comm.)
c (NB. If the H2 rate is based on the data from J92, then strictly 
c speaking it is only applicable for T < 1500K; however, the rate 
c doesn't misbehave too badly at higher T).
c
c H2 - ortho and para states must be accounted for separately
c
c 1 -> 0
         f               = fortho * 2.70d-11 * (temp**0.362)
         hh              = fpara  * 3.46d-11 * (temp**0.316)
         cltab(21,itemp) = f + hh
c 2 -> 0
         f               = fortho * 5.49d-11 * (temp**0.317)
         hh              = fpara  * 7.07d-11 * (temp**0.268)
         cltab(22,itemp) = f + hh
c 2 -> 1
         f               = fortho * 2.74d-14 * (temp**1.060)
         hh              = fpara  * 3.33d-15 * (temp**1.360)
         cltab(23,itemp) = f + hh
c
c Electron rate -- from my fits to BBT98.
c
c 1 -> 0
         cltab(24,itemp) = 5.12d-10  * (temp**(-0.075))
c 2 -> 0
         cltab(25,itemp) = 4.863d-10 * (temp**(-0.026))
c 2 -> 1
         cltab(26,itemp) = 1.082d-14 * (temp**(0.926))
c
c Proton rate -- from P90
c
c 1 -> 0
         if (temp .lt. 194) then
           cltab(27,itemp) = 6.38d-11 * (temp**0.40)
         else if (temp .lt. 3686) then
           cltab(27,itemp) = 7.75d-12 * (temp**0.80)
         else
           cltab(27,itemp) = 2.65d-10 * (temp**0.37)
         endif
c 2 -> 0
         if (temp .lt. 511) then
           cltab(28,itemp) = 6.10d-13 * (temp**1.10)
         else if (temp .lt. 7510) then
           cltab(28,itemp) = 2.12d-12 * (temp**0.90)
         else
           cltab(28,itemp) = 4.49d-10 * (temp**0.30)
         endif
c 2 -> 1
         if (temp .lt. 2090) then
           cltab(29,itemp) = 2.029d-11 * (temp**0.56)
         else
           cltab(29,itemp) = 3.434d-10 * (temp**0.19)
         endif
c
c (cl[30-41]): CI fine-structure lines 
c
c Collisional de-excitation rates:  
c
c HI: taken from AKD07 below 1000K, extended to higher temperatures
c     with a simple power-law extrapolation
c
c 1 -> 0
         if (temp .lt. 5d0) then
           tfix = 5d0
           tfinq = 1d0 / tfix**0.25d0
           cltab(30, itemp) =  (1d-11 / 3d0) * dexp(3.6593
     $                      + 56.6023    * tfinq
     $                      - 802.9765   * tfinq**2
     $                      + 5025.1882  * tfinq**3 
     $                      - 17874.4255 * tfinq**4 
     $                      + 38343.6655 * tfinq**5 
     $                      - 49249.4895 * tfinq**6 
     $                      + 34789.3941 * tfinq**7 
     $                      - 10390.9809 * tfinq**8)
     $                      * dexp(cIe10 / (kboltz * tfix))
         elseif (temp .lt. 1d3) then
           cltab(30, itemp) = (1d-11 / 3d0) * dexp(3.6593
     $                      + 56.6023    * tinq
     $                      - 802.9765   * tinq**2
     $                      + 5025.1882  * tinq**3 
     $                      - 17874.4255 * tinq**4 
     $                      + 38343.6655 * tinq**5 
     $                      - 49249.4895 * tinq**6 
     $                      + 34789.3941 * tinq**7 
     $                      - 10390.9809 * tinq**8)
     $                      * dexp(cIe10 / (kboltz * temp))
         else
           cltab(30, itemp) = 2.57d-11 * temp**0.31d0
         endif
c 2 -> 0

         if (temp .lt. 5d0) then
           tfix = 5d0
           tfinth = 1d0 / tfix**(1d0/3d0)
           cltab(31, itemp) = 2.0d-12 * dexp(10.8377 
     $                      - 173.4153    * tfinth 
     $                      + 2024.0272   * tfinth**2 
     $                      - 13391.6549  * tfinth**3
     $                      + 52198.5522  * tfinth**4 
     $                      - 124518.3586 * tfinth**5 
     $                      + 178182.5823 * tfinth**6
     $                      - 140970.6106 * tfinth**7 
     $                      + 47504.5861  * tfinth**8)
     $                      * dexp(cIe20 / (kboltz * tfix))
         elseif (temp .lt. 1d3) then
           cltab(31, itemp) = 2.0d-12 * dexp(10.8377 
     $                      - 173.4153    * tinth 
     $                      + 2024.0272   * tinth**2 
     $                      - 13391.6549  * tinth**3
     $                      + 52198.5522  * tinth**4 
     $                      - 124518.3586 * tinth**5 
     $                      + 178182.5823 * tinth**6
     $                      - 140970.6106 * tinth**7 
     $                      + 47504.5861  * tinth**8)
     $                      * dexp(cIe20 / (kboltz * temp))
         else
           cltab(31, itemp) = 1.69d-11 * temp**0.35d0
         endif
c 2 -> 1
         if (temp .lt. 5d0) then
           tfix = 5d0
           tfinq = 1d0 / tfix**0.25d0
           cltab(32, itemp) = 6.0d-12 * dexp(15.8996 
     $                      - 201.3030   * tfinq
     $                      + 1533.6164  * tfinq**2 
     $                      - 6491.0083  * tfinq**3
     $                      + 15921.9239 * tfinq**4 
     $                      - 22691.1632 * tfinq**5 
     $                      + 17334.7529 * tfinq**6 
     $                      - 5517.9360  * tfinq**7) 
     $                      * dexp(cIe21 / (kboltz * tfix))
         elseif (temp .lt. 1d3) then
           cltab(32, itemp) = 6.0d-12 * dexp(15.8996 
     $                      - 201.3030   * tinq
     $                      + 1533.6164  * tinq**2 
     $                      - 6491.0083  * tinq**3
     $                      + 15921.9239 * tinq**4 
     $                      - 22691.1632 * tinq**5 
     $                      + 17334.7529 * tinq**6 
     $                      - 5517.9360  * tinq**7) 
     $                      * dexp(cIe21 / (kboltz * temp))
         elseif (temp .lt. 1d3) then
           cltab(32, itemp) = 6.0d-12 * dexp(15.8996 
     $                      - 201.3030   * tinq
     $                      + 1533.6164  * tinq**2 
     $                      - 6491.0083  * tinq**3
     $                      + 15921.9239 * tinq**4 
     $                      - 22691.1632 * tinq**5 
     $                      + 17334.7529 * tinq**6 
     $                      - 5517.9360  * tinq**7) 
     $                      * dexp(cIe21 / (kboltz * temp))
         else
           cltab(32,itemp) = 4.95d-11 * temp**0.35
         endif
c
c H2 -- ortho and para states must be accounted for separately
c    -- rates from S91 using the fit from WBV96
c  
c 1 -> 0 
         f  = (8.7d-11 - 6.6d-11 * dexp(-temp / 218.3d0) 
     $         + 6.6d-11 * exp(-2d0 * temp / 218.3d0) ) * fortho
c
         hh = (7.9d-11 - 8.7d-11 * dexp(-temp / 126.4d0) 
     $         + 1.3d-10 * exp(-2d0 * temp / 126.4d0) ) * fpara
c
         cltab(33,itemp) = f + hh
c 2 -> 0
         f  = (1.2d-10 - 6.1d-11 * dexp(-temp / 387.3d0)) * fortho
c
         hh = (1.1d-10 - 8.6d-11 * dexp(-temp / 223.0d0) 
     $         + 8.7d-11 * exp(-2d0 * temp / 223.0d0) ) * fpara
c
         cltab(34,itemp) = f + hh
c 2 -> 1
         f  = (2.9d-10 - 1.9d-10 * dexp(-temp / 348.9d0)) * fortho
c
         hh = (2.7d-10 - 2.6d-10 * dexp(-temp / 250.7d0) 
     $        + 1.8d-10 * exp(-2d0 * temp / 250.7d0)) * fpara
c
         cltab(35,itemp) = f + hh
c
c Electrons -- from JBK87. Note that the 'electron' rate given in HM89 is 
c              actually the proton rate, as footnote f of table 8 makes 
c              clear. Note that fits are only valid for T < 10^4 K --
c              at higher temperatures, we assume that the collision 
c              strength is constant (and so the rate scales as T^-1/2)
c 1 -> 0
         if (temp .lt. 1d3) then
             f = -9.25141d0 - 0.773782d0 * tloge + 0.361184d0 * tloge**2
     $           - 0.150892d-1 * tloge**3 - 0.656325d-3 * tloge**4
         elseif (temp .le. 1d4) then
             f = 4.44600d2 - 2.27913d2 * tloge + 42.5952 * tloge**2 
     $           - 3.47620d0 * tloge**3 + 0.105085d0 * tloge**4
         else
             f = -0.990634367510893d0
         endif
c
         cltab(36,itemp) = (1d0 / 3d0) * 8.629d-6 * dexp(f) / 
     $                     dsqrt(temp)
c
c 2 -> 0  
c
         if (temp .lt. 1d3) then
             f = -7.69735d0 - 1.30743d0 * tloge + 0.697638d0 * tloge**2 
     $           - 0.111338 * tloge**3 + 0.705277d-2 * tloge**4
         elseif (temp .le. 1d4) then
             f = 3.50609d2 - 1.87474d2 * tloge + 3.61803d1 * tloge**2 
     $           - 3.03283d0 * tloge**3 + 0.938138d-1 * tloge**4
         else
             f = -1.40040241654697d0
         endif
c
         cltab(37,itemp) = 0.2d0 * 8.629d-6 * dexp(f) / dsqrt(temp)
c
c 2 -> 1
         if (temp .lt. 1d3) then
             f = -7.4387d0 - 0.57443d0 * tloge + 0.358264d0 * 
     $           tloge**2  - 0.418166d-1 * tloge**3 + 0.235272d-2 * 
     $           tloge**4
         elseif (temp .le. 1d4) then
             f = 3.86186d2 - 2.02192d2 * tloge + 3.85049d1 * tloge**2 
     $           - 3.19268d0 * tloge**3 + 0.978573d-1 * tloge**4
         else
             f = 0.0198312027880547d0
         endif
c
         cltab(38,itemp) = 0.2d0 * 8.629d-6 * dexp(f) / dsqrt(temp)
c
c Protons -- from RLB90, fits by WBV96 (T < 5000K), SCOG (T > 5000K).
c Assume rate becomes constant above limit of tabulated data.
c
c 1 -> 0
         if (temp .lt. 5d3) then
             f = (9.6d-11 - 1.8d-14 * temp + 1.9d-18 * temp**2)
     $           * temp**0.45
         elseif (temp .lt. 2d4) then
             f = 8.873d-10 * temp**0.117
         else
             f = 2.8268d-9
         endif
c
         cltab(39,itemp) = f
c
c 2 -> 0
         if (temp .lt. 5d3) then
             f = (3.1d-12 - 6.0d-16 * temp + 3.9d-20 * temp**2)
     $           * temp
         elseif (temp .lt. 2d4) then
             f = 2.314d-9 * temp**0.0965
         else
             f = 6.0175e-09
         endif
c
         cltab(40,itemp) = f
c
c 2 -> 1
         if (temp .lt. 5d3) then
           f = (1.0d-10 - 2.2d-14 * temp + 1.7d-18 * temp**2) * 
     $         temp**0.70
         elseif (temp .lt. 2d4) then
           f = 9.198d-9 * temp**0.0535
         else
           f = 1.5624e-08
         endif
c
         cltab(41,itemp) = f
c
c (cl[42-43]) -- SiI fine structure lines
c
c Collisional de-excitation rates -- HI, protons from HM89
c
c HI:
c 1 -> 0
         cltab(42,itemp) = 3.5d-10 * temp2**(-0.03)
c 2 -> 0
         cltab(43,itemp) = 1.7d-11 * temp2**0.17
c 2 -> 1
         cltab(44,itemp) = 5.0d-10 * temp2**0.17
c
c Proton rates are independent of T, so we don't bother to tabulate
c them here -- instead, they're listed in cool_func.F
c
c No data for H2, electrons.
c
c Fine structure cooling -- two-level atoms
c
c We use the convention that the ground state is level 0 and 
c the excited state is level 1.
c
c The cooling rate is simply:
c
c Lambda = A(1->0) * E(1->0) * f(1,LTE) * C(1->0) / ( A(1->0) + 
c          C(1->0) * (1 + f(1,LTE)) )
c
c where f(1,LTE) is the level population of level 1 in LTE.
c We tabulate f(1,LTE) and the temperature-dependent bits of C(1->0),
c and compute everything else in the cooling function.
c
c (cl[45-48]) -- CII fine structure lines
c
c Collisional rates:
c
c (cl45) CII - HI (HM89 below 2000K; K86 above 2000K)
c
c (Note that the high T coefficient has been tweaked slightly to ensure that
c  the rate is continuous at 2000K -- the adjustment is well within the 
c  errors).
c
         if (temp .le. 2d3) then
            cltab(45,itemp) = 8d-10 * temp2**0.07d0
         else
            cltab(45,itemp) = 3.113619d-10 * temp2**0.385d0
         endif
c
c (cl46) CII - H2 (Below 250K, we use the fit from WBV96 to the data from
c                  FL77. Above 250K, we assume that the rate scales the
c                  same as the low-temp. HI rate)
c
         if (temp .lt. 250d0) then
             f  = (4.7d-10 + 4.6d-13 * temp) * fortho
             hh = (2.5d-10 * temp**0.12) * fpara
         else
             f  = (5.85d-10 * temp**0.07) * fortho
             hh = (4.85d-10 * temp**0.07) * fpara
         endif
c
         cltab(46,itemp) = f + hh
c
c (cl47) CII - electron (WB02). 
c
c (Note that the high T coefficient has been tweaked slightly to ensure that
c  the rate is continuous at 2000K -- the adjustment is well within the 
c  errors).
c
         if (temp .le. 2d3) then
            cltab(47,itemp) = 3.86d-7 / dsqrt(temp2)
         else
            cltab(47,itemp) = 2.426206d-7 / temp2**0.345d0
         endif
c
c Proton rate negligible below 10^4K, ignorable below 10^5K.
c
c Finally, cl48 holds f(1,LTE)
c
         cltab(48,itemp) = 2d0 * dexp(-91.25d0 / temp)
c
c (cl[49-51]) -- SiII fine structure lines
c
c Collisional rates:
c
c (cl49) SiII - HI (fit by SCOG to data from R90).
c
         cltab(49,itemp) = 4.95d-10 * temp2**0.24
c
c (cl50) SiII - electron (DK91 -- extrapolated to T < 4000K, 
c                         assuming constant collision strength)
c
         cltab(50,itemp) = 1.2d-6 / dsqrt(temp2)
c
c cl51 holds f(1,LTE)
c
         cltab(51, itemp) = 2d0 * dexp(-412.24d0 / temp)
c
c (cl52, cl59-63): atomic cooling processes in H-He plasmas
c Some or all of these may be included in our tabulated atomic
c cooling, in which case corresponding entries in cltab will
c be set to zero later in this routine.
c
c The temperature cutoffs in the various rates are arbitrary,
c and are purely to avoid floating-point underflow problems
c encountered on some machines. Note that the cutoffs are 
c chosen such that the cooling rate at the cutoff is certain 
c to be negligible.
c
c (cl52) HI excitation cooling (aka Lyman-alpha cooling)
c 
c Rate from C92, based on B81.
c
          if (temp .lt. 1d3) then
            cltab(52,itemp) = 0d0
          else
            cltab(52,itemp) = 7.5d-19 * dexp(-1.18348d5 / temp) /
     $                        (1d0 + dsqrt(temp / 1d5))
          endif        
c
c (cl59) HeI excitation cooling (from n=1)
c
c My own fit to data from BBFT00. Accurate to within 10% for
c 3.75 < log10(T) < 5.75
c
          if (temp .lt. 2d3) then
            cltab(59,itemp) = 0d0
          else
            cltab(59,itemp) = 1.1d-19 * temp**0.082d0 * 
     $                        exp(-2.3d5 / temp)
          endif
c
c (cl60) HeI excitation cooling (from metastable 2^3S state)
c
c From C92, based on B81. Assumes level population of 2^3S state is 
c set by balance between recombination, radiative de-excitation.
c
c Unimportant at low densities, but may become significant in high
c density gas (n >~ 1e4 cm^-3).
c
          if (temp .lt. 1d2) then
            cltab(60, itemp) = 0d0
          else
            cltab(60,itemp) = 9.1d-27 * exp(-1.3179d4 / temp) /
     $                      (temp**0.1687d0 * (1d0 + dsqrt(temp / 1d5)))
          endif
c
c (cl61) HeII excitation cooling
c
c From C92, based on B81.
c
          if (temp .lt. 4d3) then
            cltab(61,itemp) = 0d0
          else
            cltab(61,itemp) = 5.54d-17 * exp(-4.73638d5 / temp) /
     $                      (temp**0.397d0 * (1d0 + dsqrt(temp / 1d5)))
          endif
c
c (cl62-63) Thermal bremsstrahlung.
c
c Rates from SK87, based on S78.
c
c cl62: singly charged ions (H+, He+)
c
          if (temp .lt. 3.2d5) then
            gff = 0.79464 + 0.1243 * log10(temp)
          else
            gff = 2.13164 - 0.1240 * log10(temp)
          endif
c
          cltab(62,itemp) = 1.426d-27 * dsqrt(temp) * gff
c
c cl62: doubly charged ions (He++)
c
          if (temp .lt. 1.28d6) then
            gff = 0.79464 + 0.1243 * log10(temp / 4d0)
          else
            gff = 2.13164 - 0.1240 * log10(temp / 4d0)
          endif
c
          cltab(63,itemp) = 4d0 * 1.426d-27 * dsqrt(temp) * gff
c
c cl53 -- cl57: coeffs. for HD cooling function fit (LNA05)
c               Fit good for 1 < n < 1e8, 100 < T < 2d4
c               [NB Beware of typos in astro-ph version!!!]
c
c For T > 2d4, we use the same rate as at T = 2d4, to prevent the rate
c from becoming extremely large. Since we don't expect to find much HD at
c these temperatures, this should be OK.
c
         if (temp .gt. 2d4) then
           thd = dlog10(2d4)
         else
           thd = tlog
         endif
c
         cltab(53, itemp) = - 42.57688
     $                      + 21.93385 * thd
     $                      - 10.19097 * thd**2
     $                      + 2.19906  * thd**3
     $                      - 0.17334  * thd**4
         cltab(54, itemp) = + 0.92433
     $                      + 0.77952 * thd
     $                      - 0.54263 * thd**2
     $                      + 0.11711 * thd**3
     $                      - 8.35d-3 * thd**4
         cltab(55, itemp) = + 0.54962
     $                      - 1.06447 * thd
     $                      + 0.62343 * thd**2
     $                      - 0.13768 * thd**3
     $                      + 1.06d-2 * thd**4
         cltab(56, itemp) = - 7.676d-2
     $                      + 0.11864d0 * thd
     $                      - 7.366d-2  * thd**2
     $                      + 1.759d-2  * thd**3
     $                      - 1.482d-3  * thd**4
         cltab(57, itemp) = + 2.75d-3
     $                      - 3.66d-3   * thd
     $                      + 2.514d-3  * thd**2
     $                      - 6.6631d-4 * thd**3
     $                      + 6.1926d-5 * thd**4
c
c (cl58) H2 collision-induced emission cooling.
c
c From RA04. Assumes that H2-H2 collisions dominate
c
         cltab(58,itemp) = 2.289d-49 * temp**4
c
c (cl59 - cl63): See above
c
 200  continue
c
c  Rates requiring spline fits to tabulated data:
c
c H2 rovibrational cooling -- data in mol_data.h
c
      do I = 1, nh2data
        h2_temp(i) = 1d1**(2d0 + 5d-2 * (I - 1))
      enddo
c
      call spline_eval(nh2data, h2_temp, h2_lte, nmd, temptab, rate1)
      call spline_eval(nh2data, h2_temp, h2_h_rate, nmd, temptab, rate2)
      call spline_eval(nh2data, h2_temp, h2_h2_rate, nmd, temptab,rate3)
c
      do itemp = 1, nmd
        if (temptab(itemp) .ge. 1d2 .and. temptab(itemp) .le. 1d4) then 
          cltab(1,itemp) = 1d1**rate1(itemp)
          cltab(2,itemp) = 1d1**rate2(itemp)
          cltab(3,itemp) = 1d1**rate3(itemp)
        endif
      enddo
c
c Initialize tables for CO, H2O cooling rates. 
c
c CO data: 1K bins, 5K -> 2000K
c          0.1dex bins in column density          
c
      do I = 1, nTco
        co_temptab(I) = 4d0 + 1d0 * I
      enddo
c
      do I = 1, ncdco
        co_colntab(I) = 14.4d0 + 0.1d0 * I
      enddo
c
c (co1-4) -- CO rotational cooling
c For each quantity:
c   Do spline fits over T range for each N value
c
      call spline_eval(nco_temp, co_temp, co_data_L0, nTco, 
     $                 co_temptab, co_L0)
c
      do i = 1, nco_column
        do j = 1, nco_temp
          co_lte_raw(j) = co_data_LTE(nco_temp * (i-1) + j)
          co_n05_raw(j) = co_data_n05(nco_temp * (i-1) + j)
          co_alp_raw(j) = co_data_alp(nco_temp * (i-1) + j)
        enddo
c
        call spline_eval(nco_temp, co_temp, co_lte_raw, nTco, 
     $                   co_temptab, co_lte_fit)
c
        call spline_eval(nco_temp, co_temp, co_n05_raw, nTco, 
     $                   co_temptab, co_n05_fit)
c
        call spline_eval(nco_temp, co_temp, co_alp_raw, nTco, 
     $                   co_temptab, co_alp_fit)
c
        do j = 1, nTco
          co_lte_smalltab(i,j) = co_lte_fit(j)
          co_n05_smalltab(i,j) = co_n05_fit(j)
          co_alp_smalltab(i,j) = co_alp_fit(j)
        enddo
      enddo

c   do spline fits over N to fill in table

      do j = 1, nTco
        do i = 1, nco_column
          co_lte_fxT(i) = co_lte_smalltab(i,j)
          co_n05_fxT(i) = co_n05_smalltab(i,j)
          co_alp_fxT(i) = co_alp_smalltab(i,j)
        enddo
c
        call spline_eval(nco_column, co_column, co_lte_fxT, ncdco, 
     $                   co_colntab, co_lte_fit2)
c
        call spline_eval(nco_column, co_column, co_n05_fxT, ncdco, 
     $                   co_colntab, co_n05_fit2)
c
        call spline_eval(nco_column, co_column, co_alp_fxT, ncdco, 
     $                   co_colntab, co_alp_fit2)

        do i = 1, ncdco
          co_lte(i,j) = co_lte_fit2(i)
          co_n05(i,j) = co_n05_fit2(i)
          co_alp(i,j) = co_alp_fit2(i)
        enddo
      enddo
c
      do j = 1, nTco-1
        dTco_L0(j)  = co_L0(j+1)  - co_L0(j)
        do i = 1, ncdco
          dTco_lte(i,j) = co_lte(i,j+1) - co_lte(i,j)
          dTco_n05(i,j) = co_n05(i,j+1) - co_n05(i,j)
          dTco_alp(i,j) = co_alp(i,j+1) - co_alp(i,j)
        enddo
      enddo
c
      dTco_L0(nTco) = dTco_L0(nTco-1)
      do i = 1, ncdco
          dTco_lte(i,nTco) = dTco_lte(i,nTco-1)
          dTco_n05(i,nTco) = dTco_n05(i,nTco-1)
          dTco_alp(i,nTco) = dTco_alp(i,nTco-1)
      enddo  
c
c CO vibrational cooling (LTE rate)
c
c 1K bins, 100K -> 4000K [Below 100K, we assume non-exponential term
c                         is constant, but exponential ==> negligible]
c 0.1dex bins in column density
c
      do I = 1, nTco_vib
        co_vib_temptab(I)  = 99d0 + 1d0 * I
      enddo
c
      do I = 1, ncdco_vib
        co_vib_colntab(I) = 12.9d0 + 0.1d0 * I
      enddo
c
      do i = 1, nco_vib_column
        do j = 1, nco_vib_temp
          idx1 = nco_vib_temp * (i - 1) + j
          co_vib_LTE_raw(j) = co_vib_LTE(idx1)
        enddo
c
        call spline_eval(nco_vib_temp, co_vib_temp, 
     $                   co_vib_LTE_raw, nTco_vib, 
     $                   co_vib_temptab, co_vib_LTE_fit)
c
        do j = 1, nTco_vib
          co_vib_LTE_smalltab(i,j) = co_vib_LTE_fit(j)
        enddo
      enddo

      do i = 1, nTco_vib
        do j = 1, nco_vib_column
          co_vib_LTE_fxT(j) = co_vib_LTE_smalltab(j,i)
        enddo
c
        call spline_eval(nco_vib_column, co_vib_column, 
     $                   co_vib_LTE_fxT, ncdco_vib,
     $                   co_vib_colntab, co_vib_LTE_fit2)
c
        do j = 1, ncdco_vib
          co_vib_LTE_final(j,i) = co_vib_LTE_fit2(j)
        enddo
      enddo
c
      do i = 1, nTco_vib-1
        do j = 1, ncdco_vib
          dTco_vib_LTE(j,i) = co_vib_LTE_final(j,i+1) - 
     $                         co_vib_LTE_final(j,i)
        enddo
      enddo
      do i = 1, ncdco_vib
        dTco_vib_LTE(i,nTco_vib) = dTco_vib_LTE(i,nTco_vib-1)
      enddo
c
c H2O rotational cooling
c
c H2O data: 1K bins, 10K -> 4000K
c           0.1dex bins in column density          
c
      do I = 1, nTh2o
        h2o_temptab(I)  = 9d0 + 1d0 * I
      enddo
c
      do I = 1, ncdh2o
        h2o_colntab(I) = 9.9d0 + 0.1d0 * I
      enddo
c
c First, combine low temperature & high temperature data
c
      do I = 1, nh2o_temp_low
        h2o_data_L0_ortho(I) = h2o_L0_low_ortho(I)
        h2o_data_L0_para(I)  = h2o_L0_low_para(I)
        h2o_temp(I)          = h2o_temp_low(I)
      enddo
c
      do I = nh2o_temp_low+1, nh2o_temp
        h2o_data_L0_ortho(I) = h2o_L0_high(I - nh2o_temp_low)
        h2o_data_L0_para(I)  = h2o_L0_high(I - nh2o_temp_low)
        h2o_temp(I)          = h2o_temp_high(I - nh2o_temp_low)
      enddo
c
      do J = 1, nh2o_column
        do I = 1, nh2o_temp
          idx1 = I + (J - 1) * nh2o_temp
          if (I .le. nh2o_temp_low) then
            idx2 = I + (J - 1) * nh2o_temp_low
            h2o_data_LTE_ortho(idx1) = h2o_LTE_low_ortho(idx2)
            h2o_data_LTE_para(idx1)  = h2o_LTE_low_para(idx2)
            h2o_data_alp_ortho(idx1) = h2o_alp_low_ortho(idx2)
            h2o_data_alp_para(idx1)  = h2o_alp_low_para(idx2)
            h2o_data_n05_ortho(idx1) = h2o_n05_low_ortho(idx2)
            h2o_data_n05_para(idx1)  = h2o_n05_low_para(idx2)
          else
            idx2 = (I - nh2o_temp_low) + (J - 1) * nh2o_temp_high
            h2o_data_LTE_ortho(idx1) = h2o_LTE_high(idx2)
            h2o_data_LTE_para(idx1)  = h2o_LTE_high(idx2)
            h2o_data_alp_ortho(idx1) = h2o_alp_high(idx2)
            h2o_data_alp_para(idx1)  = h2o_alp_high(idx2)
            h2o_data_n05_ortho(idx1) = h2o_n05_high(idx2)
            h2o_data_n05_para(idx1)  = h2o_n05_high(idx2)
          endif
        enddo
      enddo
c
      call spline_eval(nh2o_temp, h2o_temp, h2o_data_L0_ortho, 
     $                 nTh2o, h2o_temptab, h2o_L0_ortho)

      call spline_eval(nh2o_temp, h2o_temp, h2o_data_L0_para, 
     $                 nTh2o, h2o_temptab, h2o_L0_para)
c
      do i = 1, nh2o_column
        do j = 1, nh2o_temp
          idx1 = nh2o_temp * (i - 1) + j
          h2o_LTE_raw_ortho(j) = h2o_data_LTE_ortho(idx1)
          h2o_n05_raw_ortho(j) = h2o_data_n05_ortho(idx1)
          h2o_alp_raw_ortho(j) = h2o_data_alp_ortho(idx1)
          h2o_LTE_raw_para(j)  = h2o_data_LTE_para(idx1)
          h2o_n05_raw_para(j)  = h2o_data_n05_para(idx1)
          h2o_alp_raw_para(j)  = h2o_data_alp_para(idx1)
        enddo
c
        call spline_eval(nh2o_temp, h2o_temp, h2o_LTE_raw_ortho, 
     $                   nTh2o, h2o_temptab, h2o_LTE_fit_ortho)

        call spline_eval(nh2o_temp, h2o_temp, h2o_LTE_raw_para, 
     $                   nTh2o, h2o_temptab, h2o_LTE_fit_para)
c
        call spline_eval(nh2o_temp, h2o_temp, h2o_n05_raw_ortho, 
     $                   nTh2o, h2o_temptab, h2o_n05_fit_ortho)

        call spline_eval(nh2o_temp, h2o_temp, h2o_n05_raw_para, 
     $                   nTh2o, h2o_temptab, h2o_n05_fit_para)
c
        call spline_eval(nh2o_temp, h2o_temp, h2o_alp_raw_ortho, 
     $                   nTh2o, h2o_temptab, h2o_alp_fit_ortho)

        call spline_eval(nh2o_temp, h2o_temp, h2o_alp_raw_para, 
     $                   nTh2o, h2o_temptab, h2o_alp_fit_para)
c
        do j = 1, nTh2o
          h2o_LTE_smalltab_ortho(i,j) = h2o_LTE_fit_ortho(j)
          h2o_n05_smalltab_ortho(i,j) = h2o_n05_fit_ortho(j)
          h2o_alp_smalltab_ortho(i,j) = h2o_alp_fit_ortho(j)
          h2o_LTE_smalltab_para(i,j)  = h2o_LTE_fit_para(j)
          h2o_n05_smalltab_para(i,j)  = h2o_n05_fit_para(j)
          h2o_alp_smalltab_para(i,j)  = h2o_alp_fit_para(j)
        enddo
      enddo
c
c   do spline fits over N to fill in table
c
      do j = 1, nTh2o
        do i = 1, nh2o_column
          h2o_LTE_fxT_ortho(i) = h2o_LTE_smalltab_ortho(i,j)
          h2o_n05_fxT_ortho(i) = h2o_n05_smalltab_ortho(i,j)
          h2o_alp_fxT_ortho(i) = h2o_alp_smalltab_ortho(i,j)
          h2o_LTE_fxT_para(i)  = h2o_LTE_smalltab_para(i,j)
          h2o_n05_fxT_para(i)  = h2o_n05_smalltab_para(i,j)
          h2o_alp_fxT_para(i)  = h2o_alp_smalltab_para(i,j)
        enddo
c
        call spline_eval(nh2o_column, h2o_column, h2o_LTE_fxT_ortho, 
     $                   ncdh2o, h2o_colntab, h2o_LTE_fit2_ortho)
        call spline_eval(nh2o_column, h2o_column, h2o_LTE_fxT_para, 
     $                   ncdh2o, h2o_colntab, h2o_LTE_fit2_para)
c
        call spline_eval(nh2o_column, h2o_column, h2o_n05_fxT_ortho, 
     $                   ncdh2o, h2o_colntab, h2o_n05_fit2_ortho)
        call spline_eval(nh2o_column, h2o_column, h2o_n05_fxT_para, 
     $                   ncdh2o, h2o_colntab, h2o_n05_fit2_para)
c
        call spline_eval(nh2o_column, h2o_column, h2o_alp_fxT_ortho,
     $                   ncdh2o, h2o_colntab, h2o_alp_fit2_ortho)
        call spline_eval(nh2o_column, h2o_column, h2o_alp_fxT_para,
     $                   ncdh2o, h2o_colntab, h2o_alp_fit2_para)
c
        do i = 1, ncdh2o
          h2o_LTE_ortho(i,j) = h2o_LTE_fit2_ortho(i)
          h2o_n05_ortho(i,j) = h2o_n05_fit2_ortho(i)
          h2o_alp_ortho(i,j) = h2o_alp_fit2_ortho(i)
          h2o_LTE_para(i,j)  = h2o_LTE_fit2_para(i)
          h2o_n05_para(i,j)  = h2o_n05_fit2_para(i)
          h2o_alp_para(i,j)  = h2o_alp_fit2_para(i)
        enddo
      enddo
c
      do j = 1, nTh2o-1
        dTh2o_L0_ortho(j) = h2o_L0_ortho(j+1) - h2o_L0_ortho(j)
        dTh2o_L0_para(j)  = h2o_L0_para(j+1)  - h2o_L0_para(j)
        do i = 1, ncdh2o
          dTh2o_LTE_ortho(i,j) = h2o_LTE_ortho(i,j+1) - 
     $                           h2o_LTE_ortho(i,j)
          dTh2o_n05_ortho(i,j) = h2o_n05_ortho(i,j+1) - 
     $                           h2o_n05_ortho(i,j)
          dTh2o_alp_ortho(i,j) = h2o_alp_ortho(i,j+1) - 
     $                           h2o_alp_ortho(i,j)
          dTh2o_LTE_para(i,j)  = h2o_LTE_para(i,j+1) - 
     $                            h2o_LTE_para(i,j)
          dTh2o_n05_para(i,j)  = h2o_n05_para(i,j+1) - 
     $                            h2o_n05_para(i,j)
          dTh2o_alp_para(i,j)  = h2o_alp_para(i,j+1) - 
     $                            h2o_alp_para(i,j)
        enddo
      enddo
c
      dTh2o_L0_ortho(nTh2o) = dTh2o_L0_ortho(nTh2o-1)
      dTh2o_L0_para(nTh2o)  = dTh2o_L0_para(nTh2o-1)
      do i = 1, ncdh2o
        dTh2o_LTE_ortho(i,nTh2o) = dTh2o_LTE_ortho(i,nTh2o-1)
        dTh2o_n05_ortho(i,nTh2o) = dTh2o_n05_ortho(i,nTh2o-1)
        dTh2o_alp_ortho(i,nTh2o) = dTh2o_alp_ortho(i,nTh2o-1)
        dTh2o_LTE_para(i,nTh2o)  = dTh2o_LTE_para(i,nTh2o-1)
        dTh2o_n05_para(i,nTh2o)  = dTh2o_n05_para(i,nTh2o-1)
        dTh2o_alp_para(i,nTh2o)  = dTh2o_alp_para(i,nTh2o-1)
      enddo  
c
c H2O vibrational cooling (LTE rate)
c
c 1K bins, 100K -> 4000K [Below 100K, we assume non-exponential term
c                         is constant, but exponential ==> negligible]
c 0.1dex bins in column density
c
      do I = 1, nTh2o_vib
        h2o_vib_temptab(I)  = 99d0 + 1d0 * I
      enddo
c
      do I = 1, ncdh2o_vib
        h2o_vib_colntab(I) = 12.9d0 + 0.1d0 * I
      enddo
c
      do i = 1, nh2o_vib_column
        do j = 1, nh2o_vib_temp
          idx1 = nh2o_vib_temp * (i - 1) + j
          h2o_vib_LTE_raw(j) = h2o_vib_LTE(idx1)
        enddo
c
        call spline_eval(nh2o_vib_temp, h2o_vib_temp, 
     $                   h2o_vib_LTE_raw, nTh2o_vib, 
     $                   h2o_vib_temptab, h2o_vib_LTE_fit)
c
        do j = 1, nTh2o_vib
          h2o_vib_LTE_smalltab(i,j) = h2o_vib_LTE_fit(j)
        enddo
      enddo

      do i = 1, nTh2o_vib
        do j = 1, nh2o_vib_column
          h2o_vib_LTE_fxT(j) = h2o_vib_LTE_smalltab(j,i)
        enddo
c
        call spline_eval(nh2o_vib_column, h2o_vib_column, 
     $                   h2o_vib_LTE_fxT, ncdh2o_vib,
     $                   h2o_vib_colntab, h2o_vib_LTE_fit2)
c
        do j = 1, ncdh2o_vib
          h2o_vib_LTE_final(j,i) = h2o_vib_LTE_fit2(j)
        enddo
      enddo
c
      do i = 1, nTh2o_vib-1
        do j = 1, ncdh2o_vib
          dTh2o_vib_LTE(j,i) = h2o_vib_LTE_final(j,i+1) - 
     $                         h2o_vib_LTE_final(j,i)
        enddo
      enddo
      do i = 1, ncdh2o_vib
        dTh2o_vib_LTE(i,nTh2o_vib) = dTh2o_vib_LTE(i,nTh2o_vib-1)
      enddo
c
c (cl6) --  the atomic cooling function
c
c Calculate the temperatures corresponding to the data in coolatom
c
      if (iflag_atom .eq. 1) then
        do 210 i = 1, natom
          coolatom_temp(i) = 10**(8d0 * (i-1) / 79d0)
210     continue
c
c Compute and evaluate a spline fit to the data in coolatom
c
        call spline_eval(natom, coolatom_temp, coolatom, nmd, temptab,
     $                   rate0)

        do 220 itemp = 1, nmd
          temp   = temptab(itemp)
          atomic = 1d1**(rate0(itemp))
          cltab(6,itemp) = atomic
c
c This rate includes cooling from HI, HeI, HeII excitation, so we set 
c the corresponding individual rates to zero to avoid double counting
c
c HI excitation cooling ('Lyman-alpha' cooling):
c
          cltab(52,itemp) = 0d0
c
c HeI excitation cooling
c
          cltab(59,itemp) = 0d0
          cltab(60,itemp) = 0d0
c
c HeII excitation cooling
c
          cltab(61,itemp) = 0d0
c
220     continue
c
      elseif (iflag_atom .eq. 2) then
c
        ca2_temp(1) = tmin
        do 230 i = 2, natom2
          ca2_temp(i) = 10**(4.25d0 + 0.05d0 * i)
230     continue
c
        call spline_eval(natom2, ca2_temp, ca2, nmd, temptab, rate0)
c
        do 240 itemp = 1, nmd
          temp   = temptab(itemp)
          if (temp .lt. 10**4.3d0) then
            cltab(6,itemp) = 0d0
          else
            cltab(6,itemp) = 1d1**(rate0(itemp)) 
          endif
c
c HeI, HeII excitation cooling & bremsstrahlung are included
c in the rate given in the DATA statement above, and so we 
c set the individual rates to zero to avoid double counting
c
         cltab(59,itemp) = 0d0
         cltab(60,itemp) = 0d0
         cltab(61,itemp) = 0d0
         cltab(62,itemp) = 0d0
         cltab(63,itemp) = 0d0
c
240     continue
c
      elseif (iflag_atom .eq. 3) then
c All rates are tabulated separately, so cl6=0.
        do itemp = 1, nmd
          cltab(6,itemp) = 0d0
        enddo
      endif
c
c Approximate temperature derivatives
c
      do 300 itemp = 1,nmd-1
         dtemp           = 1d0 / ( temptab(itemp + 1) - temptab(itemp) )
c
         do 250 j = 1, ncltab 
            dtcltab(j,itemp) = ( cltab(j, itemp+1)
     $                         - cltab(j, itemp  ) ) * dtemp
c
 250     continue
 300  continue
c
c Manually add on the final tabulated value at itemp = ntabtemp
c
      do 310 j = 1, ncltab
         dtcltab(j,nmd) = dtcltab(j,nmd-1)
 310  continue
c
#endif /* CHEMCOOL */
      return
      end
c=======================================================================
c
c    \\\\\\\\\\        E N D   S U B R O U T I N E        //////////
c    //////////             C O O L I N M O               \\\\\\\\\\
c
c=======================================================================
