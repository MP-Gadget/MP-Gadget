c=======================================================================
c
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////             P H O T O I N I T             \\\\\\\\\\
c
c=======================================================================
c
       subroutine photoinit
#ifdef CHEMCOOL
c
c    written by: S. Glover, AMNH, June 2004
c
c    PURPOSE: Precompute and tabulate photochemical rate coefficients.
c
c             This version assumes an optical/UV spectrum corresponding
c             to a 10^5 K black body, cut off at 13.6eV. Scaling is 
c             controlled by G0: the rates are normalized so that G0 = 1
c             corresponds to a flux density of 10^21 erg/s/cm^2/Hz/sr at
c             the Lyman limit.
c
c    REFERENCES:
c
c      D68   -- Dunn, 1968, Phys. Rev., 172, 1
c      DJ72  -- de Jong, 1972, A&A, 20, 263
c      D78   -- Draine, 1978, ApJS, 36, 595
c      MMP83 -- Mathis, Mezger, & Panagia, 1983, A&A, 128, 212
c      L84   -- Lee, 1984, ApJ, 282, 172
c      NL85  -- Nee & Lee, 1985, ApJ, 291, 202 
c      vD87  -- van Dishoeck, 1987, in 'Astrochemistry', IAU Symp. 120, 51
c      vD88  -- van Dishoeck, 1988,in 'Rate Coefficients in Astrochemistry', 49
c      vDB88 -- van Dishoeck & Black, 1988, ApJ, 334, 771
c      RJLD91 -- Roberge, Jones, Lepp & Dalgarno, 1991, ApJS, 77, 287 
c      SD95  -- Sternberg & Dalgarno, 1995, ApJS, 99, 565
c      DB96  -- Draine & Bertoldi, 1996, ApJ, 468, 269
c      V96  -- Verner et al, 1996, ApJ, 465, 487 
c      S97   -- Stancil et al, 1997, ApJ, 486, 574
c      G03   -- Glover, 2003, ApJ, 584, 331
c
      implicit NONE
#include "cool.h"
c
c (ph1) -- H2 photodissociation
c
      phtab(1) = 1.30d-12
c
c (ph2) -- CI ionization -- cross-section from V96
c 
      phtab(2) = 5.56d-12
c
c (ph3) -- SiI ionization -- cross-section from V96
c
      phtab(3) = 2.44d-11
c
c (ph4) -- H- photodissociation:
c
      phtab(4) = 1.36d-11
c
c (ph5) -- H2+ photodissociation
c
      phtab(5) = 4.11d-12
c
c Heating rates:  (cross-sections as above)
c
c (ph6) -- CI ionization
c
      phtab(6) = 1.02d-23
c
c (ph7) -- SiI ionization
c
      phtab(7) = 7.85d-23
c
c (ph8) -- H- photodissociation: 
c
      phtab(8) = 1.18d-22
c
c (ph9) -- H2+ photodissociation: 
c
      phtab(9) = 5.50d-23
c
c (ph10) -- CO photodissociation: CO + gamma -> C + O
c 
c From vD88, rescaled
c
      phtab(10) = 1.59d-13
c
c (ph11) -- OH photodissociation: OH + gamma -> O + H
c
c From NL85
c
      phtab(11) = 3.49d-13
c
c (ph12) -- H2O photodissociation: H2O + gamma -> OH + H
c
c From L84
c
      phtab(12) = 2.75d-13
c
c (ph13) -- unused
c
c (ph14) -- CH+ photodissociation
c  
c UMIST99 gives the products as C+ + H and cites RJLD91 as the source.
c However, RJLD91 actually list products as C + H+ (as does Kirby 1980)
c Moreover, UMIST05 now also lists products as C + H+, implying that
c UMIST99 is simply in error. 
c
      phtab(14) = 1.17d-13
c
c (ph15) -- CH2+ photodissociation: CH+ + H
c
c UMIST06, quoting van Dishoeck et al, 2006, in prep.
c
      phtab(15) = 2.16d-14
c
c (ph16) -- CH3+ photodissociation
c
c Rate and branching ratio from UMIST99:
c 50% of photodissociations -> CH2+ + H (16a), 50% -> CH+ + H2 (16b)
c
      phtab(16) = 4.69d-13
c
c ph17 -- unused
c
c (ph18) -- CH photodissociation: CH + gamma -> C + H
c
c From RJLD91
c
      phtab(18) = 4.03d-13
c
c (ph19) -- CH photoionization: CH + gamma -> CH+ + e-
c
c From vD88
c
      phtab(19) = 3.56d-13
c
c (ph20) -- CH2 photodissociation: CH2 + gamma -> CH + H
c
c From vD88
c
      phtab(20) = 3.38d-13
c
c (ph21) -- CH2 photoionization: CH2 + gamma -> CH2+ + e-
c
c From UMIST99
c
      phtab(21) = 4.69d-13
c
c (ph22) -- OH+ photodissociation:  OH+ + gamma -> O + H+
c
c From vD87
c
      phtab(22) = 4.69d-16
c
c (ph23) -- OH photoionization: OH + gamma -> OH+ + e-
c
c From UMIST99
c
      phtab(23) = 7.5d-16
c
c (ph24) -- H2O photoionization: H2O + gamma -> H2O+ + e-
c
c From vD88
c
      phtab(24) = 1.55d-14
c
c (ph25) -- C2 photodissociation: C2 + gamma -> C + C
c
c From RJLD91
c
      phtab(25) = 7.03d-14
c
c (ph26) -- unused
c
c (ph27) -- O2 photodissociation: O2 + gamma -> O + O
c
c From RJLD91
c
      phtab(27) = 3.2d-13
c
c (ph28) -- O2 photoionization: O2 + gamma -> O2+ + e-
c
c From RJLD91
c
      phtab(28) = 2.63d-14
c
c (ph29) -- C- photodetachment:  C- + gamma -> C + e-
c
c Rate from UMIST99
c
      phtab(29) = 1.13d-10
c
c (ph30) -- O- photodetachment:  O- + gamma -> O + e-
c
c Rate from UMIST99
c
      phtab(30) = 1.13d-10
c
c (ph31-32) -- SiH+ photodissociation, rates from S97.
c
      phtab(31) = 9.67d-13    ! SiH+ + gamma -> Si+ + H
      phtab(32) = 1.02d-13    ! SiH+ + gamma -> Si  + H+
c
c (ph33) -- H2O+ photodissociation; branching ratios in rate_eq.F
c           Estimate from SD95.
c
      phtab(33) = 1.41d-13
c
c (ph34) -- H3O+ photodissociation; branching ratios in rate_eq.F
c           Estimate from SD95.
c
      phtab(34) = 3.87d-14
c
c (ph35) -- HD photodissociation
c
c Computed using procedure of Draine & Bertoldi (1996), based on data
c from Abgrall & Roueff (2006).
c
      phtab(35) = 1.45d-12 
c
c (ph36) -- H3+ photodissociation, total rate
c 
c vD88 quotes an _upper limit_ of 1e-12 s^-1 for the H3+ 
c photodissociation rate in the local ISM. UMIST99 assumes
c that two outcomes are equally likely: H2 + H+ or H2+ + H 
c We adopt the same assumption here.
c
      phtab(36) = 4.69d-16
c
c (ph37): unused
c
c (ph38 - ph45): Photoionization of HI, HeI, HeII and H2,
c                together with associated heating. 
c
c These are not currently set in the chemistry code, but can be
c set by external code
c
      phtab(38) = 0d0  ! HI   ion
      phtab(39) = 0d0  ! HeI  ion
      phtab(40) = 0d0  ! HeII ion
      phtab(41) = 0d0  ! H2   ion
      phtab(42) = 0d0  ! HI   heat
      phtab(43) = 0d0  ! HeI  heat
      phtab(44) = 0d0  ! HeII heat
      phtab(45) = 0d0  ! H2   heat
c
c (ph46): Photoionization of OI - currently set to zero
c
      phtab(46) = 0d0
c
c (ph47): Photoionization of SiI - currently set to zero
c
      phtab(47) = 0d0
c
c Scaling factor used to express radiation field strength in units
c of the Habing field. For the radiation field used here, we obtain it
c by dividing the energy density, integrated from 6eV to 13.6eV, for
c our diluted black-body spectrum by the corresponding energy density
c for the Habing field
c
      f_rsc = 7.4d-3
c
#endif /* CHEMCOOL */
      return
      end
c=======================================================================
c
c    \\\\\\\\\\        E N D   S U B R O U T I N E        //////////
c    //////////             P H O T O I N I T             \\\\\\\\\\
c
c=======================================================================
