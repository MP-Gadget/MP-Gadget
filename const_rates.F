c=======================================================================
c
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////            C O N S T _ R A T E S          \\\\\\\\\\
c
c=======================================================================
c
       subroutine const_rates
#ifdef CHEMCOOL
c
c    written by: S. Glover, AMNH, March 2005
c
c    PURPOSE: Tabulate constant rate coefficients.
c
c    REFERENCES: 
c
c      UMIST99 is the UMIST database for astrochemistry 1999
c      [described in Le Teuff, Millar & Markwick, 2000, A&AS, 146, 157]
c
c      UMIST06 is the UMIST database for astrochemistry 2006
c      [available on the web at www.udfa.net, described in 
c      Woodall, Agundez, Markwick-Kemper, Millar, 2006, A&A, submitted.]
c
c      Rates listed as being taken from UMIST99 (my original source) are
c      unchanged in UMIST06.
c
c      KTH74  -- Kim, Theard & Huntress 1974, 
c                Int. J. Mass Spectrom. Ion. Phys., 15, 223
c      AFHK75 -- Anicich et al, 1975, Int. J. Mass Spectrom. Ion. Phys., 18, 63
c      KTH75  -- Kim, Theard & Huntress, 1975, Chem. Phys. Lett., 32, 610
c      AHF76  -- Anicich, Huntress & Futrell, 1976, Chem. Phys. Lett., 40, 233
c      AS76a  -- Adams & Smith, 1976a, Int. J. Mass Spectrom. Ion. Phys., 21, 349
c      AS76b  --  Adams & Smith, 1976b, J. Phys. B., 9, 1439
c      FEH76  -- Fehsenfeld, 1976, ApJ, 209, 638
c      WAH76  -- Watson, Anicich & Huntress, 1976, ApJ, 205, L165
c      SA77a  -- Smith & Adams, 1977a, Int. J. Mass Spectrom. Ion. Phys., 23, 123
c      SA77b  -- Smith & Adams, 1977b, Chem. Phys. Lett., 47,  383
c      ASG78  -- Adams, Smith & Grief, 1978, Int. J. Mass Spectrom. Ion. Phys., 26, 405
c      MDM78a -- Mauclaire et al, 1978a, Int. J. Mass Spectrom. Ion. Phys., 26, 284
c      MDM78b -- Mauclaire et al, 1978b, Dyn. Mass Spectrom., 5, 139
c      SAM78  -- Smith, Adams & Miller, 1978, J. Chem. Phys, 69, 308
c      KAH79  -- Karpas, Anicich & Huntress, 1979, J. Chem. Phys, 70, 2877
c      ASP80  -- Adams, Smith & Paulson, 1980, J. Chem. Phys., 72, 288
c      RW80   -- Raksit & Warneck, 1980, J. Chem. Soc. Farad. Trans., 76, 1084
c      PH80   -- Prasad & Huntress, 1980, ApJS, 43, 1
c      VIG80  -- Viggiano, 1980, ApJ, 236, 492
c      JBT81  -- Jones, Birkinshaw & Twiddy, 1981, Chem. Phys. Lett., 77, 484
c      B84    -- Barlow, 1984, PhD thesis, Univ. Colorado
c      FED84a -- Federer et al, 1984a, Phys.\ Rev.\ Lett., 52, 2084
c      FED84b -- Federer et al, 1984b, Contrib.-Symp. At. Surf. Phys., 141      
c      W84    -- Warnatz, 1984, in `Combustion Chemistry', (Springer-Verlag:NY), 197 
c      WR85   -- Wagner-Redeker, Kemper, Jarrold & Bowers, 1985, J. Chem. Phys., 83, 1121
c      HMWB89 -- Herbst et al, 1989, A&A, 222, 205
c      GH92   -- Gerlich & Horning, 1992, Chem. Rev., 92, 1509
c      SSM92  -- Smith, Spanel & Mayhew 1992, Int. J. Mass Spectrom. Ion. Proc., 117, 457
c      SD98   -- Stancil & Dalgarno, 1998, Faraday Discussions, 109, 61
c      SLD98  -- Stancil, Lepp & Dalgarno, 1998, ApJ, 509, 1
c      MSA99  -- McEwan et al, 1999, ApJ, 513, 287 
c      MM00   -- Milligan & McEwan, 2000, Chem. Phys. Lett., 319, 482 
c      SSRG02 -- Smith et al, 2002, ApJ, 578, L87
c      SHC04  -- Smith, Herbst & Chang, 2004, MNRAS, 350, 323
c
      implicit NONE
#include "cool.h"
c
c H2 formation from H2+  (H2+ + H -> H2 + H+)
c
c Rate from KAH79
c
      cst(1) = 6.4d-10
c
c Charge transfer: C+ + Si -> C + Si+
c
c Rate from UMIST99
c
      cst(2) = 2.1d-9
c
c CO+ formation from C+ and OH:  C+ + OH -> CO+ + H
c
c Rate from UMIST99
c
      cst(3) = 7.7d-10
c
c H2O destruction by C+:  H2O + C+ -> HCO+ + H
c
c Rate from UMIST99, based on AHF76, WAH76
c
      cst(4) = 9d-10
c
c H2O destruction by C+:  H2O + C+ -> HOC+ + H
c
c Rate from UMIST99, based on AHF76, WAH76
c
      cst(5) = 1.8d-9
c
c CH+ formation by radiative association: C+ + H -> CH+ + gamma
c
c UMIST99,06 gives this as a constant rate, 1.7e-17, but new work by
c Barinovs & van Hemert (2006) gives a T dependent fit, which I've 
c implemented as rate ch104. Consequently, this rate is no longer
c used
c
      cst(6) = 1.7d-17
c
c CH+ destruction by atomic hydrogen: CH+ + H -> C+ + H2
c
c Rate from MSA99.
c
      cst(7) = 7.5d-10
c
c CH2+ formation from CH+:  CH+ + H2 -> CH2+ + H
c
c Rate from MSA99
c
      cst(8) = 1.2d-9
c
c CH3+ formation from CH2+:  CH2+ + H2 -> CH3+ + H
c
c Rate from UMIST99, based on SA77a, SA77b
c
      cst(9) = 1.6d-9
c
c C2 formation from CH and C:  C + CH -> C2 + H
c
c Rate from SHC04.
c
      cst(10) = 6.59d-11
c
c CH2 destruction by atomic H: CH2 + H -> CH + H2
c
c Rate from W84.
c
      cst(11) = 6.64d-11
c
c CH formation by radiative association: C + H -> CH + gamma
c
c Rate from PH80
c
      cst(12) = 1d-17
c
c OH+ formation from O & H2+:  O + H2+ -> OH+ + H
c
c Rate from UMIST99
c
      cst(13) = 1.5d-9
c
c OH+ formation from O & H3+:  O + H3+ -> OH+ + H2
c
c Rate from MM00
c
      cst(14) = 8.4d-10
c
c OH+ formation from O+ & H2: O+ + H2 -> OH+ + H
c
c Rate from UMIST99, based on SAM78, ASP80.
c
      cst(15) = 1.7d-9
c
c H2O+ formation from OH+: OH+ + H2 -> H2O+ + H
c
c Rate from JBT81
c
      cst(16) = 1.01d-9
c
c H3O+ formation from H2O+: H2O+ + H2 -> H3O+ + H
c
c Rate from RW80
c
      cst(17) = 6.4d-10
c
c SiH+ formation by radiative association: Si+ + H -> SiH+ + gamma
c
c XXX: no longer used; replaced with more recent temperature dependent rate
c
      cst(18) = 1d-17
c
c SiH2+ formation by radiative association: Si+ + H2 -> SiH2+ + gamma
c
c XXX: No longer used.
c
      cst(19) = 3d-18
c
c SiH+ destruction by atomic hydrogen: SiH+ + H -> Si+ + H2
c
c Rate from HMWB89
c
      cst(20) = 1.9d-9
c
c H3O+ destruction by Si:  H3O+ + Si -> H2O + SiH+
c
c Rate from UMIST99
c
      cst(21) = 1.8d-9
c
c OH formation by AD of H- with O:  H- + O -> OH + e-
c
c Rate from UMIST99
c
      cst(22) = 1.0d-9
c
c O- formation by RA of O and e-:  O + e- -> O- + gamma
c
c Rate from UMIST99
c
      cst(23) = 1.5d-15
c
c OH formation from AD of O- with H:  O- + H -> OH + e-
c
c Rate from UMIST99
c
      cst(24) = 5d-10
c
c OH charge transfer with H+:  OH + H+ -> OH+ + H
c
c Rate from UMIST99
c
      cst(25) = 2.1d-9
c
c OH dissoc. charge transfer with He+:  OH + He+ -> O+ + He + H
c
c Rate from UMIST99
c
      cst(26) = 1.1d-9
c
c H2O charge transfer with H+:   H2O + H+ -> H2O+ + H
c
c Rate from SSM92
c
      cst(27) = 6.9d-9
c
c H2O charge transfer with He+:  H2O + He+ -> OH + He + H+
c
c Rate from UMIST99, based on MDM78a, MDM78b
c
      cst(28) = 2.04d-10
c
c H2O charge transfer with He+:  H2O + He+ -> OH+ + He + H
c
c Rate from UMIST99, based on MDM78a, MDM78b
c
      cst(29) = 2.86d-10
c
c H2O charge transfer with He+:  H2O + He+ -> H2O+ + He
c
c Rate from UMIST99, based on MDM78a, MDM78b
c
      cst(30) = 6.05d-11
c
c CO+  formation from O+ and C2:  C2 + O+ -> CO+ + C
c
c Rate from UMIST99
c
      cst(31) = 4.8d-10
c
c Dissoc. charge transfer -- C2 and He+: C2 + He+ -> C+ + C + He
c
c Rate from UMIST99
c
      cst(32) = 1.6d-9
c
c O2 destruction by C+:  O2 + C+ -> CO+ + O
c
c Rate from UMIST99, based on SA77a, SA77b
c 
      cst(33) = 3.8d-10
c
c O2 destruction by C+:  O2 + C+ -> CO + O+
c
c Rate from UMIST99, based on SA77a, SA77b
c 
      cst(34) = 6.2d-10
c
c O2 CT with H+:  O2 + H+ -> O2+ + H
c
c Rate from SSM92
c
      cst(35) = 2d-9
c
c O2 non-dissoc. CT with He+:  O2 + He+ -> O2+ + He
c
c Rate from UMIST06, based on AS76a, AS76b
c
      cst(36) = 3.3d-11
c
c O2 dissoc. CT with He+:  O2 + He+ -> O+ + O + He
c
c Rate from UMIST06, based on AS76a, AS76b
c
      cst(37) = 1.1d-9
c
c CO+ formation from O2+ and C:  O2+ + C -> CO+ + O
c
c Rate from UMIST99
c
      cst(38) = 5.2d-11
c
c O2+ charge transfer with C:  O2+ + C -> O2 + C+
c
c Rate from UMIST99
c
      cst(39) = 5.2d-11
c
c CO formation from CO+ CT with H:  CO+ + H -> CO + H+
c
c Rate from UMIST99, based on FED84a, FED84b
c
      cst(40) = 7.5d-10
c
c CO+ formation from CH+ and O:  CH+ + O -> CO+ + H
c
c Rate from VIG80
c
      cst(41) = 3.5d-10
c
c CO formation from HCO+ and C:  HCO+ + C -> CO + CH+
c
c Rate from UMIST99
c
      cst(42) = 1.1d-9
c
c CO formation from HCO+ and Si:  HCO+ + Si -> CO + SiH+
c
c Rate from UMIST99
c
      cst(43) = 1.6d-9
c
c CO formation from HCO+ and H2O:  HCO+ + H2O -> CO + H3O+
c
c Rate from ASG78
c
      cst(44) = 2.5d-9
c
c Conversion of HOC+ to HCO+ by CO: HOC+ + CO -> HCO+ + CO
c
c Rate from WR85
c
      cst(45) = 4d-10
c
c Conversion of HOC+ to HCO+ by H2: HOC+ + H2 -> HCO+ + H2
c
c Rate from SSRG02
c
      cst(46) = 3.8d-10
c
c C- formation by radiative association: C + e- -> C- + gamma
c
c From UMIST06, who cite SD98
c
      cst(47) = 2.25d-15
c
c CO formation from AD of C- with O:  C- + O -> CO + e-
c
c Rate from UMIST99
c
      cst(48) = 5d-10
c
c CH formation from AD of C- with H:  C- + H -> CH + e-
c
c Rate from UMIST99
c
      cst(49) = 5d-10
c
c CO formation from AD of O- with C:  O- + C -> CO + e-
c
c Rate from UMIST99
c
      cst(50) = 5d-10
c
c H3+ formation by radiative association of H2 and H+. 
c
c iflag_h3pra .eq. 1 -- GH92  rate
c iflag_h3pra .eq. 2 -- SLD98 rate
c
c Following SLD98, we use same rate for RA of H2 and D+ and
c HD and H+ 
c
      if (iflag_h3pra .eq. 1) then
        cst(51) = 1d-16
      elseif (iflag_h3pra .eq. 2) then
        cst(51) = 1d-20
      else
        print*, 'Unknown H3+ RA flag', iflag_h3pra
        ABORT(0)
      endif
c
c Charge transfer: Si + He+ -> Si+ + He
c
c Rate taken from UMIST99.
c
      cst(52) = 3.3d-9
c
c HCO+ formation from CH2+:  CH2+ + O -> HCO+ + H
c
c Rate from UMIST99.
c
      cst(53) = 7.5d-10
c
c CH2 formation by radiative association: C + H2 -> CH2 + gamma
c
c Rate from PH80
c
      cst(54) = 1d-17
c
c CH+ formation from H3+:  C + H3+ -> CH+ + H2
c
c Rate from UMIST99.
c 
      cst(55) = 2.0d-9
c
c H2O+ formation from H3+:  OH + H3+ -> H2O+ + H2
c
c Rate from UMIST99
c
      cst(56) = 1.3d-9
c
c Conversion of H2O to H3O+  by H3+: H2O + H3+ -> H3O+ + H2
c
c Rate from UMIST99, based on KTH74, AFHK75
c
      cst(57) = 5.9d-9
c
c Conversion of CO to HOC+  by H3+: CO + H3+ -> HOC+ + H2
c
c Rate from KTH75
c
      cst(58) = 2.7d-11
c
c Conversion of CO to HCO+  by H3+: CO + H3+ -> HCO+ + H2
c
c Rate from KTH75
c
      cst(59) = 1.7d-9
c
c Conversion of Si to SiH+  by H3+: Si + H3+ -> SiH+ + H2
c
c Rate from UMIST99
c
      cst(60) = 3.7d-9
c
c Formation of CH2 from C-:  C- + H2 -> CH2 + e-
c
c Rate from UMIST99
c
      cst(61) = 1.0d-13
c
c Destruction of CH2 by H+:  CH2 + H+ -> CH+ + H2
c
c Rate from UMIST99
c
      cst(62) = 1.4d-9
c
c Formation of H2O from O-:  O- + H2 -> H2O + e
c
c Rate from UMIST99
c
      cst(63) = 7d-10
c
c Formation of CO from CH2: CH2 + O -> CO + H + H
c                                   -> CO + H2
c
c Rate from UMIST06.
c Branching ratios are in rate_eq_mol.F
c
      cst(64) = 2.13d-10
c
c Charge transfer from H+ to CH:  CH + H+ -> CH+ + H
c
c Rate from UMIST99
c
      cst(65) = 1.9d-9
c
c Charge transfer from H+ to CH2: CH2 + H+ -> CH2+ + H
c
c Rate from UMIST99
c
      cst(66) = 1.4d-9
c
c Destruction of CH2+ by O2:  CH2+ + O2 -> HCO+ + OH
c
c Rate from UMIST99, based on SA77a, SA77b
c
      cst(67) = 9.1d-10
c
c Dissociative CT -- CH2 and He+:  CH2 + He+ -> C+ + He + H2 
c
c Rate from UMIST99
c
      cst(68) = 7.5d-10
c
c Destruction of CH3+ by O:  CH3+ + O -> HCO+ + H2
c
c Rate from FEH76
c
      cst(69) = 4.0d-10
c
c Destruction of water by SiH+:  SiH+ + H2O -> Si + H3O+
c
c Rate from HMWB89
c
      cst(70) = 8.0d-10
c
c CO formation from OH: C + OH -> CO + H
c
c Rate from SHC04
c
      cst(71) = 1.0d-10
c
c Mg charge transfer with H: H+ + Mg -> H + Mg+
c
c Rate from UMIST99
c 
      cst(72) = 1.1d-9
c
c Mg charge transfer with C+: C+ + Mg -> C + Mg+
c
c Rate from UMIST99
c 
      cst(73) = 1.1d-9
c
c Associative detachment of H- by OH: H- + OH -> H2O + e-
c
c Rate from UMIST99
c
      cst(74) = 1.0d-10
c
c H2 destruction by He+ CT  (He+ + H2 -> H2+ + He + PHOTON)
c
c Rate from B84
c
      cst(75) = 7.2d-15
c
c CH formation by AD of H-: C + H- -> CH + e-
c
c Rate from UMIST99
c
      cst(76) = 1.0d-9
c
c CH+ form from H2+: H2+ + C -> CH+ + H
c
c Rate from UMIST99
c
      cst(77) = 2.4d-9
c
c HCO+ form: H3O+ + C -> HCO+ + H2
c
c Rate from UMIST99
c
      cst(78) = 1.0d-11
c
c CO2 formation: OH + CO -> CO2 + H
c
      cst(79) = 1.0d-13
c
c HCO+ formation from CO+:  CO+ + H2 -> HCO+ + H
c
      cst(80) = 1.8d-9
c
#endif /* CHEMCOOL */
      return
      end
c=======================================================================
c
c    \\\\\\\\\\        E N D   S U B R O U T I N E        //////////
c    //////////           C O N S T _ R A T E S           \\\\\\\\\\
c
c=======================================================================
