c=======================================================================
c
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////            C O O L _ F U N C              \\\\\\\\\\
c
c=======================================================================
c
      subroutine cool_func(temp, yn, dl, divv, G_dust, abundances, ylam,
     $                     rates, iflag_cmb)
#ifdef CHEMCOOL
c                                                                       
c    Based on cool_h, written for ZEUS-3D by Michael D. Smith and
c    Georgi Pavlovski (Armagh Observatory, 2003) and substantially
c    modified by S. Glover (AMNH, 2003-2005).
c
c    Rewritten to remove unneeded T derivative code by S. Glover,
c    September 2005; to avoid confusion with other versions, changed
c    routine name to cool_func
c                                                                       
c    PURPOSE:  Compute the cooling function for the gas. Note that 
c    the convention used here is that Lambda (the net _cooling_ rate)
c    is positive and has units of erg s^-1 cm^-3, so that for gas at
c    rest the equation for the internal energy density, e, can be 
c    written as:   
c                 de/dt = - Lambda
c    This means that any heating terms which we include in Lambda
c    (e.g. photoelectric heating) must be NEGATIVE; i.e. heating is
c    treated as negative cooling.
c
c    N.B. When used for cosmological simulations, it is necessary to
c         account for the effects of the CMB on the various heating & 
c         cooling rates. In the case of the fine structure coolants,
c         this is done explicitly (and exactly) by including the 
c         appropriate stimulated emission & absorption terms in the 
c         level population equations. For other species (HD, CO, H2O),
c         this effect is not treated explicitly -- instead, we 
c         approximate it by first calling cool_func with the gas 
c         temperature and then with the CMB temperature, and then 
c         subtracting the cooling rate given by the second call from 
c         that given by the first. This approximation is of questionable
c         accuracy near T_cmb -- for instance, it can lead to ~50% errors
c         in the HD cooling rate -- but is probably sufficient for our 
c         present purposes, given the other uncertainties
c                  
c  INPUT VARIABLES: temp       -- temperature
c                   yn         -- density
c                   divv       -- velocity divergence
c                   abundances -- chemical abundances
c
c  OUTPUT VARIABLES: Computed values of ylam & rates
c
c  LOCAL VARIABLES:  rates  -- parts of the cooling function
c                       cl[1-*] -- cooling table parameters
c                     dtcl[1-*] -- their change with T = T + dT
c
c  NB: the dt... terms are TEMPERATURE derivatives, not time derivatives
c
c-----------------------------------------------------------------------
c
      implicit none
#include "cool.h"
#include "fs_data.h"

      integer    itemp     , iflag_cmb
      REAL       temp      , dtemp   , yn      , divv       , dl

      REAL       abundances(nabn)
      REAL       ynh2      , ynh     , yne     , ynhp       , ynhd
      REAL       ynhe
      REAL       rates(nrates)
c
      REAL       ylam      , lnh       , r19scale
c
      REAL       abh2      , abo       , aboh    , abco     , abh2o,
     $           abcI      , abcII     , absiI   , absiII   , abe  ,
     $           abhp      , abhI      , abhd    , abheI    , abheII,
     $           abheIII
c
      REAL       h2var0
c
      REAL       oxc01, oxc02, oxlam1, oxlam2,
     $           oxb10, oxb01, oxc10 , oxb02 , oxb20 , oxb21 ,
     $           oxb12, oxc20, oxn1  , oxn2  , oxn0  , oxr01 ,
     $           oxr02, oxr12, oxr10 , oxr20 , oxr21 , oxa   ,
     $           oxb  , oxc  , oxc21 , oxc12
c
      REAL       cIc01, cIc02, cIlam1, cIlam2,
     $           cIb10, cIb01, cIc10 , cIb02 , cIb20 , cIb21 ,
     $           cIb12, cIc20, cIn1  , cIn2  , cIn0  , cIr01 ,
     $           cIr02, cIr12, cIr10 , cIr20 , cIr21 , cIa   ,
     $           cIb  , cIc  , cIc21 , cIc12
c
      REAL       siIc01, siIc02, siIlam1, siIlam2,
     $           siIb10, siIb01, siIc10 , siIb02 , siIb20 , siIb21 ,
     $           siIb12, siIc20, siIn1  , siIn2  , siIn0  , siIr01 ,
     $           siIr02, siIr12, siIr10 , siIr20 , siIr21 , siIa   ,
     $           siIb  , siIc  , siIc21 , siIc12
c
      REAL       cIIc10 , cIIc01 , cIIb10, cIIb01, cIIvar0, cIIvar1,
     $           cIIvar2, cIIvar3, cIIn0 , cIIn1
c
      REAL       siIIc10 , siIIc01 , siIIb10 , siIIb01 ,
     $           siIIvar0, siIIvar1, siIIvar2, siIIvar3,
     $           siIIn0  , siIIn1
c
      REAL       eps       , PEvar0   , PEvar1   , PEvar2,
     $           G_dust
c
      REAL       compvar1  , CMB_temp
c
      REAL       h2_opacity_correction, N_H2_eff, N_H2_jeans, 
     $           N_H2_LVG, vth, tau_CIE
c
      REAL       neff
c
c Set ortho:para ratio for water -- default value is 3:1
c
      REAL       f_ortho_h2o, log_fo_h2o
      parameter (f_ortho_h2o = 0.75d0)
      parameter (log_fo_h2o  = -0.1249387d0)

      REAL       f_para_h2o, log_fp_h2o
      parameter (f_para_h2o = 0.25d0)
      parameter (log_fp_h2o  = -0.602060)
      
      REAL       dv, N_co_eff, N_c13o_eff, N_co18_eff
      REAL       N_co_eff_vib, N_c13o_eff_vib, N_co18_eff_vib
      REAL       N_h2o_eff_ortho, N_h2o18_eff_ortho,
     $           N_h2o_eff_para,  N_h2o18_eff_para,
     $           N_h2o_eff_vib,   N_h2o18_eff_vib
      REAL       maximum_CO_column_rot, maximum_CO_column_vib
c
      REAL       co_rot_inv, h2o_rot_inv
      REAL       co_rot_L0, co_rot_lte, co_rot_n05,
     $           co_rot_alpha, co_vib_lte_rate
      REAL       c13o_rot_L0, c13o_rot_lte, c13o_rot_n05, 
     $           c13o_rot_alpha, c13o_vib_lte_rate
      REAL       co18_rot_L0, co18_rot_lte, co18_rot_n05, 
     $           co18_rot_alpha, co18_vib_lte_rate,
     $           co_vib_L0_rate, co_vib_inv
c
      REAL       h2o_rot_L0_ortho,  h2o_rot_lte_ortho, 
     $           h2o_rot_n05_ortho, h2o_rot_alpha_ortho
      REAL       h2o_rot_L0_para,   h2o_rot_lte_para, 
     $           h2o_rot_n05_para,  h2o_rot_alpha_para
      REAL       h2o18_rot_L0_ortho,  h2o18_rot_lte_ortho, 
     $           h2o18_rot_n05_ortho, h2o18_rot_alpha_ortho
      REAL       h2o18_rot_L0_para,   h2o18_rot_lte_para, 
     $           h2o18_rot_n05_para,  h2o18_rot_alpha_para
      REAL       h2o_vib_lte_rate,    h2o18_vib_lte_rate,
     $           h2o_vib_L0_rate,     h2o_vib_inv
c
      REAL       cl1       , cl2     , cl3     , cl4     , cl5
     $         , cl6       , cl7     , cl8     , cl9     , cl10
     $         , cl11      , cl12    , cl13    , cl14    , cl15
     $         , cl16      , cl17    , cl18    , cl19    , cl20
     $         , cl21      , cl22    , cl23    , cl24    , cl25
     $         , cl26      , cl27    , cl28    , cl29    , cl30
     $         , cl31      , cl32    , cl33    , cl34    , cl35
     $         , cl36      , cl37    , cl38    , cl39    , cl40
     $         , cl41      , cl42    , cl43    , cl44    , cl45
     $         , cl46      , cl47    , cl48    , cl49    , cl50
     $         , cl51      , cl52    , cl53    , cl54    , cl55
     $         , cl56      , cl57    , cl58    , cl59    , cl60
     $         , cl61      , cl62    , cl63    , cl64    , cl65
     $         , cl66      , cl67
c
      REAL       dtcl1     , dtcl2   , dtcl3   , dtcl4   , dtcl5
     $         , dtcl6     , dtcl7   , dtcl8   , dtcl9   , dtcl10
     $         , dtcl11    , dtcl12  , dtcl13  , dtcl14  , dtcl15
     $         , dtcl16    , dtcl17  , dtcl18  , dtcl19  , dtcl20
     $         , dtcl21    , dtcl22  , dtcl23  , dtcl24  , dtcl25
     $         , dtcl26    , dtcl27  , dtcl28  , dtcl29  , dtcl30
     $         , dtcl31    , dtcl32  , dtcl33  , dtcl34  , dtcl35
     $         , dtcl36    , dtcl37  , dtcl38  , dtcl39  , dtcl40
     $         , dtcl41    , dtcl42  , dtcl43  , dtcl44  , dtcl45
     $         , dtcl46    , dtcl47  , dtcl48  , dtcl49  , dtcl50
     $         , dtcl51    , dtcl52  , dtcl53  , dtcl54  , dtcl55
     $         , dtcl56    , dtcl57  , dtcl58  , dtcl59  , dtcl60
     $         , dtcl61    , dtcl62  , dtcl63  , dtcl64  , dtcl65
     $         , dtcl66    , dtcl67
c
c ---------------------------------------------------------------------
c
c  ylam and rates will still contain their previous
c  values on entry to this routine, but they do not need to be 
c  explicitly zeroed here as the old values will be overwritten 
c  below
c
c Verify that temperature is valid
      if (temp .lt. 0d0) then
        write(*,'(A)') 'COOL_FUNC: initial temperature is unphysical,' 
        write(*,'(A, e12.5)') ' temp = ', temp
        return
      endif
c
c Read out tables.
c
      if (temp .le. tmin) then
        itemp = 1
        dtemp = 0d0
      elseif (temp .gt. tmax) then
        itemp = nmd
        dtemp = temp - temptab(itemp)
      else
        itemp = dlog10(temp) / dtlog + 1
        if (itemp .le. 0 .or. itemp .gt. nmd) then
          print*, 'Fatal error in cool_func.F', itemp, temp
          ABORT(0)
        endif
        dtemp = temp - temptab(itemp)
      endif
c
      dtcl1  = dtcltab(1,itemp)
      dtcl2  = dtcltab(2,itemp)
      dtcl3  = dtcltab(3,itemp)
      dtcl4  = dtcltab(4,itemp)
      dtcl5  = dtcltab(5,itemp)
      dtcl6  = dtcltab(6,itemp)
      dtcl7  = dtcltab(7,itemp)
      dtcl8  = dtcltab(8,itemp)
      dtcl9  = dtcltab(9,itemp)
      dtcl10 = dtcltab(10,itemp)
      dtcl11 = dtcltab(11,itemp)
      dtcl12 = dtcltab(12,itemp)
      dtcl13 = dtcltab(13,itemp)
      dtcl14 = dtcltab(14,itemp)             
      dtcl15 = dtcltab(15,itemp)
      dtcl16 = dtcltab(16,itemp)
      dtcl17 = dtcltab(17,itemp)
      dtcl18 = dtcltab(18,itemp)
      dtcl19 = dtcltab(19,itemp)
      dtcl20 = dtcltab(20,itemp)
      dtcl21 = dtcltab(21,itemp)
      dtcl22 = dtcltab(22,itemp)
      dtcl23 = dtcltab(23,itemp)
      dtcl24 = dtcltab(24,itemp)
      dtcl25 = dtcltab(25,itemp)
      dtcl26 = dtcltab(26,itemp)
      dtcl27 = dtcltab(27,itemp)
      dtcl28 = dtcltab(28,itemp)
      dtcl29 = dtcltab(29,itemp)
      dtcl30 = dtcltab(30,itemp)
      dtcl31 = dtcltab(31,itemp)
      dtcl32 = dtcltab(32,itemp)
      dtcl33 = dtcltab(33,itemp)
      dtcl34 = dtcltab(34,itemp)
      dtcl35 = dtcltab(35,itemp)
      dtcl36 = dtcltab(36,itemp)
      dtcl37 = dtcltab(37,itemp)
      dtcl38 = dtcltab(38,itemp)
      dtcl39 = dtcltab(39,itemp)
      dtcl40 = dtcltab(40,itemp)
      dtcl41 = dtcltab(41,itemp)
      dtcl42 = dtcltab(42,itemp)
      dtcl43 = dtcltab(43,itemp)
      dtcl44 = dtcltab(44,itemp)
      dtcl45 = dtcltab(45,itemp)
      dtcl46 = dtcltab(46,itemp)
      dtcl47 = dtcltab(47,itemp)
      dtcl48 = dtcltab(48,itemp)
      dtcl49 = dtcltab(49,itemp)
      dtcl50 = dtcltab(50,itemp)
      dtcl51 = dtcltab(51,itemp)
      dtcl52 = dtcltab(52,itemp)
      dtcl53 = dtcltab(53,itemp)
      dtcl54 = dtcltab(54,itemp)
      dtcl55 = dtcltab(55,itemp)
      dtcl56 = dtcltab(56,itemp)
      dtcl57 = dtcltab(57,itemp)
      dtcl58 = dtcltab(58,itemp)
      dtcl59 = dtcltab(59,itemp)
      dtcl60 = dtcltab(60,itemp)
      dtcl61 = dtcltab(61,itemp)
      dtcl62 = dtcltab(62,itemp)
      dtcl63 = dtcltab(63,itemp)
      dtcl64 = dtcltab(64,itemp)
      dtcl65 = dtcltab(65,itemp)
      dtcl66 = dtcltab(66,itemp)
      dtcl67 = dtcltab(67,itemp)

      cl1  = cltab(1,itemp) + dtemp * dtcl1
      cl2  = cltab(2,itemp) + dtemp * dtcl2
      cl3  = cltab(3,itemp) + dtemp * dtcl3
      cl4  = cltab(4,itemp) + dtemp * dtcl4
      cl5  = cltab(5,itemp) + dtemp * dtcl5
      cl6  = cltab(6,itemp) + dtemp * dtcl6
      cl7  = cltab(7,itemp) + dtemp * dtcl7
      cl8  = cltab(8,itemp) + dtemp * dtcl8
      cl9  = cltab(9,itemp) + dtemp * dtcl9
      cl10 = cltab(10,itemp) + dtemp * dtcl10
      cl11 = cltab(11,itemp) + dtemp * dtcl11
      cl12 = cltab(12,itemp) + dtemp * dtcl12
      cl13 = cltab(13,itemp) + dtemp * dtcl13
      cl14 = cltab(14,itemp) + dtemp * dtcl14
      cl15 = cltab(15,itemp) + dtemp * dtcl15
      cl16 = cltab(16,itemp) + dtemp * dtcl16
      cl17 = cltab(17,itemp) + dtemp * dtcl17
      cl18 = cltab(18,itemp) + dtemp * dtcl18
      cl19 = cltab(19,itemp) + dtemp * dtcl19
      cl20 = cltab(20,itemp) + dtemp * dtcl20
      cl21 = cltab(21,itemp) + dtemp * dtcl21
      cl22 = cltab(22,itemp) + dtemp * dtcl22
      cl23 = cltab(23,itemp) + dtemp * dtcl23
      cl24 = cltab(24,itemp) + dtemp * dtcl24
      cl25 = cltab(25,itemp) + dtemp * dtcl25
      cl26 = cltab(26,itemp) + dtemp * dtcl26
      cl27 = cltab(27,itemp) + dtemp * dtcl27
      cl28 = cltab(28,itemp) + dtemp * dtcl28
      cl29 = cltab(29,itemp) + dtemp * dtcl29
      cl30 = cltab(30,itemp) + dtemp * dtcl30
      cl31 = cltab(31,itemp) + dtemp * dtcl31
      cl32 = cltab(32,itemp) + dtemp * dtcl32
      cl33 = cltab(33,itemp) + dtemp * dtcl33
      cl34 = cltab(34,itemp) + dtemp * dtcl34
      cl35 = cltab(35,itemp) + dtemp * dtcl35
      cl36 = cltab(36,itemp) + dtemp * dtcl36
      cl37 = cltab(37,itemp) + dtemp * dtcl37
      cl38 = cltab(38,itemp) + dtemp * dtcl38
      cl39 = cltab(39,itemp) + dtemp * dtcl39
      cl40 = cltab(40,itemp) + dtemp * dtcl40
      cl41 = cltab(41,itemp) + dtemp * dtcl41
      cl42 = cltab(42,itemp) + dtemp * dtcl42
      cl43 = cltab(43,itemp) + dtemp * dtcl43
      cl44 = cltab(44,itemp) + dtemp * dtcl44
      cl45 = cltab(45,itemp) + dtemp * dtcl45
      cl46 = cltab(46,itemp) + dtemp * dtcl46
      cl47 = cltab(47,itemp) + dtemp * dtcl47
      cl48 = cltab(48,itemp) + dtemp * dtcl48
      cl49 = cltab(49,itemp) + dtemp * dtcl49
      cl50 = cltab(50,itemp) + dtemp * dtcl50
      cl51 = cltab(51,itemp) + dtemp * dtcl51
      cl52 = cltab(52,itemp) + dtemp * dtcl52
      cl53 = cltab(53,itemp) + dtemp * dtcl53
      cl54 = cltab(54,itemp) + dtemp * dtcl54
      cl55 = cltab(55,itemp) + dtemp * dtcl55
      cl56 = cltab(56,itemp) + dtemp * dtcl56
      cl57 = cltab(57,itemp) + dtemp * dtcl57
      cl58 = cltab(58,itemp) + dtemp * dtcl58
      cl59 = cltab(59,itemp) + dtemp * dtcl59
      cl60 = cltab(60,itemp) + dtemp * dtcl60
      cl61 = cltab(61,itemp) + dtemp * dtcl61
      cl62 = cltab(62,itemp) + dtemp * dtcl62
      cl63 = cltab(63,itemp) + dtemp * dtcl63
      cl64 = cltab(64,itemp) + dtemp * dtcl64
      cl65 = cltab(65,itemp) + dtemp * dtcl65
      cl66 = cltab(66,itemp) + dtemp * dtcl66
      cl67 = cltab(67,itemp) + dtemp * dtcl67
c
c Compute CMB temperature (used for Compton cooling and for adjusting
c the fine structure cooling rates to take account of radiative pumping 
c by the CMB)
c
      CMB_temp  = 2.726d0 * (1d0 + redshift)
c
c Set abundances
c
      abh2   = abundances(1)
      abo    = abundances(2)
      aboh   = abundances(3)
      abh2o  = abundances(4)
      abco   = abundances(5)
      abcI   = abundances(6)
      abcII  = abundances(7)
      absiI  = abundances(8)
      absiII = abundances(9)
      abe    = abundances(10) 
      abhp   = abundances(11)
      abhI   = abundances(12)
      abhd   = abundances(13)
      abheI  = abundances(14)
      abheII = abundances(15)
      abheIII = abundances(16)
c
c Verify that H2 abundance is valid
      if (abh2 .gt. (0.5d0 + cool_eps) .or. abh2 .lt. 0d0) then
        write(*,'(A)') 'COOL_FUNC: initial  H2 fraction is unphysical,' 
        write(*,'(A, e12.5)') ' fraction = ', abh2
        ABORT(2)
        return
      endif

c Compute useful auxiliary variables

      ynh2 = abh2 * yn
      ynh  = abHI * yn
      yne  = abe  * yn
      ynhp = abhp * yn
      ynhd = abhd * yn
      ynhe = abheI * yn 
c
c Quiet some compiler warnings in g95
      UNUSED_PARAM(cl7)
      UNUSED_PARAM(cl10)
      UNUSED_PARAM(cl11)
c
c Computing CO, H2O cooling rates is costly. However, if they have small 
c abundances then they will not contribute significantly to the total 
c cooling and so we can ignore them. 
c
c Compute effective column density for CO, H2O rotational cooling 
c rates. (See eq. 4 of Neufeld & Kaufman, 1993, ApJ, 418, 263).
c Tabulated values for the effective column density are in terms of 
c cm^-2 / (km s^-1), but divv is in units of cm s^-1, so we need to do
c a unit conversion here.
c
c If divv is very small (or zero), then we set the effective column
c density to the largest tabulated value
c
c CO -- standard isotope, plus C13O and CO18
c
      if (abco .gt. 1d-5 * abundo .and. temp .ge. co_temptab(1)) then
        maximum_CO_column_rot = co_colntab(ncdco)
        maximum_CO_column_vib = co_vib_colntab(ncdco_vib)
        if (divv .eq. 0d0) then
          N_co_eff   = maximum_CO_column_rot
          N_c13o_eff = maximum_CO_column_rot
          N_co18_eff = maximum_CO_column_rot
          N_co_eff_vib   = maximum_CO_column_vib
          N_c13o_eff_vib = maximum_CO_column_vib
          N_co18_eff_vib = maximum_CO_column_vib
        else
          dv = 1d-5 * dabs(divv) 
          N_co_eff   = dlog10(abco * yn / dv) 
c Isotopic abundance ratios are from NLM95
          N_c13o_eff = N_co_eff - 2d0
          N_co18_eff = N_co_eff - 2.69897d0  ! Log10(2d-3)
          N_co_eff_vib   = N_co_eff
          N_c13o_eff_vib = N_c13o_eff
          N_co18_eff_vib = N_co18_eff
c
          if (N_co18_eff .gt. maximum_CO_column_rot) then
            N_co_eff   = maximum_CO_column_rot
            N_c13o_eff = maximum_CO_column_rot
            N_co18_eff = maximum_CO_column_rot
          else if (N_c13o_eff .gt. maximum_CO_column_rot) then
            N_co_eff   = maximum_CO_column_rot
            N_c13o_eff = maximum_CO_column_rot
          else if (N_co_eff .gt. maximum_CO_column_rot) then
            N_co_eff   = maximum_CO_column_rot
          endif
c
          if (N_co18_eff_vib .gt. maximum_CO_column_vib) then
            N_co_eff_vib   = maximum_CO_column_vib
            N_c13o_eff_vib = maximum_CO_column_vib
            N_co18_eff_vib = maximum_CO_column_vib
          else if (N_c13o_eff_vib .gt. maximum_CO_column_vib) then
            N_co_eff_vib   = maximum_CO_column_vib
            N_c13o_eff_vib = maximum_CO_column_vib
          else if (N_co_eff_vib .gt. maximum_CO_column_vib) then
            N_co_eff_vib   = maximum_CO_column_vib
          endif
        endif
c
        call co_cool(temp, N_co_eff, N_co_eff_vib, co_rot_L0, 
     $               co_rot_lte, co_rot_alpha, co_rot_n05, 
     $               co_vib_lte_rate)
c
c If the column density of our main isotope so small that the gas is
c optically thin, then it must also be optically thin to the other 
c isotopes
        if (N_co_eff .le. co_colntab(1)) then
          c13o_rot_L0  = co_rot_L0
          co18_rot_L0  = co_rot_L0
          c13o_rot_lte = co_rot_lte
          co18_rot_lte = co_rot_lte
          c13o_rot_alpha = co_rot_alpha
          co18_rot_alpha = co_rot_alpha
          c13o_rot_n05 = co_rot_n05
          co18_rot_n05 = co_rot_n05
          c13o_vib_lte_rate = co_vib_lte_rate
          co18_vib_lte_rate = co_vib_lte_rate
        else
          call co_cool(temp, N_c13o_eff, N_c13o_eff_vib, c13o_rot_L0, 
     $                 c13o_rot_lte, c13o_rot_alpha, c13o_rot_n05, 
     $                 c13o_vib_lte_rate)
c
          call co_cool(temp, N_co18_eff, N_co18_eff_vib, co18_rot_L0, 
     $                 co18_rot_lte, co18_rot_alpha, co18_rot_n05, 
     $                 co18_vib_lte_rate)
        endif
      endif
c
c
c H2O -- standard isotope, plus H2O18
c
      if (abh2o .gt. 1d-5 * abundo) then
        if (abs(divv) .lt. 1d0) then
          N_h2o_eff_para  = h2o_colntab(ncdh2o)
          N_h2o_eff_ortho = h2o_colntab(ncdh2o)
          N_h2o18_eff_para  = h2o_colntab(ncdh2o)
          N_h2o18_eff_ortho = h2o_colntab(ncdh2o)
          N_h2o_eff_vib   = h2o_vib_colntab(ncdh2o_vib)
          N_h2o18_eff_vib = h2o_vib_colntab(ncdh2o_vib)
        else
          dv = 1d-5 * dabs(divv)
          N_h2o_eff_para  = N_h2o_eff_vib + log_fp_h2o
          N_h2o_eff_ortho = N_h2o_eff_vib + log_fo_h2o
          N_h2o_eff_vib = dlog10(abh2o * yn / dv)
c Isotopic abundance ratio is from NLM95
          N_h2o18_eff_para  = N_h2o_eff_para  - 2.69897d0  ! Log10(2d-3)
          N_h2o18_eff_ortho = N_h2o_eff_ortho - 2.69897d0  ! Log10(2d-3)
          N_h2o18_eff_vib   = N_h2o_eff_vib   - 2.69897d0  ! Log10(2d-3)
c
          if (N_h2o_eff_para .gt. h2o_colntab(ncdh2o)) then
            N_h2o_eff_para = h2o_colntab(ncdh2o)
          endif
          if (N_h2o_eff_ortho .gt. h2o_colntab(ncdh2o)) then
            N_h2o_eff_ortho = h2o_colntab(ncdh2o)
          endif
          if (N_h2o_eff_vib .gt. h2o_vib_colntab(ncdh2o_vib)) then
            N_h2o_eff_vib = h2o_vib_colntab(ncdh2o_vib)
          endif
c
          if (N_h2o18_eff_para .gt. h2o_colntab(ncdh2o)) then
            N_h2o18_eff_para = h2o_colntab(ncdh2o)
          endif
          if (N_h2o18_eff_ortho .gt. h2o_colntab(ncdh2o)) then
            N_h2o18_eff_ortho = h2o_colntab(ncdh2o)
          endif
          if (N_h2o18_eff_vib .gt. h2o_vib_colntab(ncdh2o_vib)) then
            N_h2o18_eff_vib = h2o_vib_colntab(ncdh2o_vib)
          endif
        endif
c
        call h2o_rot_cool(temp, N_h2o_eff_para, h2o_rot_L0_para, 
     $                h2o_rot_LTE_para, h2o_rot_alpha_para, 
     $                h2o_rot_n05_para, 0)
c
        call h2o_rot_cool(temp, N_h2o_eff_ortho, h2o_rot_L0_ortho, 
     $                h2o_rot_LTE_ortho, h2o_rot_alpha_ortho, 
     $                h2o_rot_n05_ortho, 1)

        call h2o_rot_cool(temp, N_h2o18_eff_para, h2o18_rot_L0_para, 
     $                h2o18_rot_LTE_para, h2o18_rot_alpha_para, 
     $                h2o18_rot_n05_para, 0)
c
        call h2o_rot_cool(temp, N_h2o18_eff_ortho, h2o18_rot_L0_ortho, 
     $                h2o18_rot_LTE_ortho, h2o18_rot_alpha_ortho, 
     $                h2o18_rot_n05_ortho, 1)
c
        call h2o_vib_cool(temp, N_h2o_eff_vib, h2o_vib_lte_rate)
c
        call h2o_vib_cool(temp, N_h2o18_eff_vib, h2o18_vib_lte_rate)
c
      endif
c
c Compute heating/cooling rates

c
c (R1) -- gas-grain cooling-heating
c
      if (iflag_cmb .eq. 0) then
        rates(1) = cl14 * (temp - tdust) * yn**2 * dust_to_gas_ratio
#if CHEMISTRYNETWORK == 10
        gamma_gd = cl14 * yn**2
#endif
      else
        rates(1) = 0d0
      endif
c
c
c (R2) -- H2 (vr) cooling
c
      if (ynh2 .eq. 0d0) then
        rates(2) = 0d0
      else
        h2var0   = ynh * cl4 + ynh2 * cl5 + ynhe * cl64
     $           + ynhp * cl65 + yne * cl66
        if (h2var0 .lt. 1d-4 * cl1) then
          rates(2) = h2var0 * ynh2
        else
          rates(2) = ynh2 * cl1 / (1d0 + cl1 / h2var0)
        endif
c Ortho-para conversion heating / cooling
        rates(2) = rates(2) + 4.76d-24 * ynhp * ynh2 *
     $             (cl67 * 0.25d0 - 0.75d0)

c Optical depth correction from RA04
c        rates(2) = rates(2) * min(1d0, (yn / 8d9)**(-0.45d0))
c
c Optically thick H2 cooling - we assume that this is unimportant at
c low gas densities (i.e. below n = 10^8 cm^-3). 
c
        if (yn .lt. 1d8) then
          H2_opacity_correction = 1d0
        else
c Estimate #1 -- H2 column within local Jeans length
          vth = dsqrt(kboltz * temp / PROTONMASS)
          N_H2_jeans = ynh2 / (vth / dl)

c Estimate #2 -- LVG estimate
          N_H2_LVG = ynh2 / dabs(divv)

c Take the smallest value -- our local Jeans length estimate ensures that
c don't use an artifically large value in the case that divv is very small
          N_H2_eff = min(N_H2_LVG, N_H2_jeans)

          call compute_h2_opacity(temp, N_H2_eff, H2_opacity_correction)
          rates(2) = rates(2) * H2_opacity_correction
        endif
      endif
c
c (R3)  -- atomic cooling, table-based; depending on the value of iflag_atom,
c          may include some or all of R25 - R28 (in which case the coefficients
c          for these rates will have been set to zero, to avoid double counting)
c (R25) -- HI electronic excitation cooling [aka Lyman-alpha cooling]
c (R26) -- HeI electronic excitation cooling
c          (N.B. Note the dependence of the HeII metastable rate (cl60) on
c           the HeII number density and the _square_ of the electron density)
c (R27) -- HeII electronic excitation cooling
c (R28) -- thermal bremsstrahlung
c
      rates(3)  = cl6 * yn**2
      rates(25) = cl52 * ynh  * yne
      rates(26) = cl59 * ynhe * yne + cl60 * abheII * yn * yne**2
      rates(27) = cl61 * abheII * yn * yne
      rates(28) = cl62 * (abhp + abheII) * yn * yne
     $          + cl63 * abheIII * yn * yne
c
c (R4)  -- cooling through coll. H20 (r) w. H, H2 and He
c (R23) -- cooling through coll. H2018 (r) w. H, H2 and He
c
      neff = ynh2 + dsqrt(2d0) * ynh + 0.5d0 * ynhe
      if (abh2o .le. 1d-5 * abundo .or. neff .eq. 0d0) then
        rates(4)  = 0d0
        rates(23) = 0d0
      else
c
c Standard H2O:
c
        h2o_rot_inv = (1d0 / h2o_rot_L0_ortho) + 
     $      (neff / h2o_rot_LTE_ortho) + (1d0 / 
     $      h2o_rot_L0_ortho) * (1d0 - h2o_rot_n05_ortho * 
     $      h2o_rot_L0_ortho / h2o_rot_LTE_ortho) * 
     $      (neff / h2o_rot_n05_ortho)**h2o_rot_alpha_ortho
c
        rates(4) = f_ortho_h2o * abh2o * neff * yn / h2o_rot_inv
c
        h2o_rot_inv = (1d0 / h2o_rot_L0_para) + 
     $      (neff / h2o_rot_lte_para) + (1d0 / 
     $      h2o_rot_L0_para) * (1d0 - h2o_rot_n05_para * 
     $      h2o_rot_L0_para / h2o_rot_LTE_para) * 
     $      (neff / h2o_rot_n05_para)**h2o_rot_alpha_para
c
        rates(4) = rates(4) + f_para_h2o * abh2o * neff * yn / 
     $             h2o_rot_inv
c
c Isotopes:
c
        h2o_rot_inv = (1d0 / h2o18_rot_L0_ortho) + 
     $      (neff / h2o18_rot_lte_ortho) + (1d0 / 
     $      h2o18_rot_L0_ortho) * (1d0 - h2o18_rot_n05_ortho * 
     $      h2o18_rot_L0_ortho / h2o18_rot_LTE_ortho) * 
     $      (neff / h2o18_rot_n05_ortho)**h2o18_rot_alpha_ortho
c
        rates(23) = 2d-3 * f_ortho_h2o * abh2o * neff * yn / 
     $              h2o_rot_inv
c
        h2o_rot_inv = (1d0 / h2o18_rot_L0_para) + 
     $      (neff / h2o18_rot_lte_para) + (1d0 / 
     $      h2o18_rot_L0_para) * (1d0 - h2o18_rot_n05_para * 
     $      h2o18_rot_L0_para / h2o18_rot_LTE_para) * 
     $      (neff / h2o18_rot_n05_para)**h2o18_rot_alpha_para
c
        rates(23) = rates(23) + 2d-3 * f_para_h2o * abh2o * neff * 
     $              yn / h2o_rot_inv
      endif
c
c (R5) -- H2O vibrational cooling   (coll. with H, H2)
c (R6) -- H2O18 vibrational cooling (coll. with H, H2)
c
      neff = ynh2 + ynh
      if (abh2o .le. 1d-5 * abundo .or. neff .eq. 0d0
     $    .or. temp .lt. 1d2) then
        rates(5) = 0d0
        rates(6) = 0d0
      else
        h2o_vib_L0_rate = cl8 * abh2 + cl9 * abHI
        h2o_vib_inv = (1d0 / h2o_vib_L0_rate) + (yn / h2o_vib_lte_rate)
        rates(5) = abh2o * yn * neff * (1d0 / h2o_vib_inv)
c
        h2o_vib_inv = (1d0 / h2o_vib_L0_rate) + 
     $                (yn / h2o18_vib_lte_rate)
        rates(6) = 2d-3 * abh2o * yn * neff * (1d0 / h2o_vib_inv)
      endif
c
c (R7)  -- cooling through coll. CO (r) H, H2 and He
c (R21) -- cooling from C13O (r) with H, H2 and He
c (R22) -- cooling from CO18 (r) with H, H2 and He
c
      neff = ynh2 + dsqrt(2d0) * ynh + 0.5d0 * ynhe
      if (abco .le. 1d-5 * abundo .or. neff .eq. 0d0 .or. 
     $    temp .lt. co_temptab(1)) then
        rates(7)  = 0d0
        rates(21) = 0d0
        rates(22) = 0d0
      else
c
c Standard CO:
c
        co_rot_inv = (1d0 / co_rot_L0) + (neff / co_rot_lte) +
     $               (1d0 / co_rot_L0) * (1d0 - co_rot_n05 * 
     $               co_rot_L0  / co_rot_lte) *
     $               (neff / co_rot_n05)**co_rot_alpha
        rates(7) = abco * neff * yn / co_rot_inv
c
c Isotopes:
c
        co_rot_inv = (1d0 / c13o_rot_L0) + (neff / c13o_rot_lte) +
     $               (1d0 / c13o_rot_L0) * (1d0 - c13o_rot_n05 * 
     $               c13o_rot_L0  / c13o_rot_lte) *
     $               (neff / c13o_rot_n05)**c13o_rot_alpha
        rates(21) = 1d-2 * abco * neff * yn / co_rot_inv
c
        co_rot_inv = (1d0 / co18_rot_L0) + (neff / co18_rot_lte) +
     $               (1d0 / co18_rot_L0) * (1d0 - co18_rot_n05 * 
     $               co18_rot_L0  / co18_rot_lte) *
     $               (neff / co18_rot_n05)**co18_rot_alpha
c
        rates(22) = 2d-3 * abco * neff * yn / co_rot_inv
      endif
c
c (R8) -- CO vibrational cooling   (coll. with H, H2)
c (R9) -- C13O vibrational cooling (coll. with H, H2)
c (R24) -- CO18 vibrational cooling (coll. with H, H2)
c
      neff = ynh2 + ynh
      if (abco .le. 1d-5 * abundo .or. neff .eq. 0d0
     $    .or. temp .lt. 1d2) then
        rates(8)  = 0d0
        rates(9)  = 0d0
        rates(24) = 0d0
      else
        co_vib_L0_rate = cl12 * abh2 + cl13 * abHI
        co_vib_inv = (1d0 / co_vib_L0_rate) + (yn / co_vib_lte_rate)
        rates(8) = abco * yn * neff * (1d0 / co_vib_inv)
c
        co_vib_inv = (1d0 / co_vib_L0_rate) + (yn / c13o_vib_lte_rate)
        rates(9) = 1d-2 * abco * yn * neff * (1d0 / co_vib_inv)
c
        co_vib_inv = (1d0 / co_vib_L0_rate) + (yn / co18_vib_lte_rate)
        rates(24) = 2d-3 * abco * yn * neff * (1d0 / co_vib_inv)
      endif
c
c (R10) -- OH cooling
c
      rates(10) = cl15 * aboh * neff
c
c (R11) -- cosmic ray heating; independent of gas temperature
c       -- following Goldsmith & Langer (1978), we assume that 
c          each ionization deposits 20eV as heat
c
      if (iflag_cmb .eq. 0) then
        rates(11) =  -3.2d-11 * (ynh2 + ynh) * cosmic_ray_ion_rate
      else
        rates(11) = 0d0
      endif
c
c (R12) -- photoelectric heating
c
c If there's no UV field, or if the electron density is very low (in which
c case the photoheating efficiency will also be very low), then we set the
c rates to zero. Otherwise, we compute the heating rate using the 
c Bakes & Tielens (1994) formula (as modified by Wolfire et al, 2003).
c
      if (G_dust .eq. 0d0 .or. dust_to_gas_ratio .eq. 0d0 
     $                .or. iflag_cmb .eq. 1) then
        rates(12) = 0d0
      elseif (yne .lt. 1d-9 * G_dust * cl16) then
        rates(12) = 0d0
      else
        PEvar0 = G_dust * cl16 / yne
        PEvar1 = (1d0 + 4d-3 * PEvar0**0.73)
        PEvar2 = (1d0 + 2d-4 * PEvar0)

        eps = (4.9d-2 / PEvar1) + (cl17 / PEvar2)

        rates(12) = -1.3d-24 * eps * G_dust * yn * dust_to_gas_ratio
      endif
c
c (R13) -- OI fine-structure cooling
c
c Total collisional rates:
c
      oxc10  = cl18 * ynh + cl21 * ynh2 + cl24 * yne + cl27 * ynhp
      oxc20  = cl19 * ynh + cl22 * ynh2 + cl25 * yne + cl28 * ynhp
      oxc21  = cl20 * ynh + cl23 * ynh2 + cl26 * yne + cl29 * ynhp
c
      if (abo .le. 1d-5 * abundo) then
        rates(13) = 0d0
      elseif (oxc10 .eq. 0d0 .and. oxc20 .eq. 0d0 .and.
     $    oxc21 .eq. 0d0) then
        rates(13) = 0d0
      else
        oxa =  dexp(-oxe10 / (kboltz * temp))
        oxb =  dexp(-oxe20 / (kboltz * temp))
        oxc =  dexp(-oxe21 / (kboltz * temp))

        oxc01  = 0.6d0 * oxc10 * oxa
        oxc02  = 0.2d0 * oxc20 * oxb
        oxc12  = (1d0 / 3d0) * oxc21 * oxc
c
c Stimulated emission and absorption:
c
        call compute_stim(oxa10, oxe10, CMB_temp, oxb10)
        oxb01 = 0.6d0 * oxb10

        call compute_stim(oxa20, oxe20, CMB_temp, oxb20)
        oxb02 = 0.2d0 * oxb20

        call compute_stim(oxa21, oxe21, CMB_temp, oxb21)
        oxb12 = (1d0 / 3d0) * oxb21
c
c Total transition rates:
c
        oxR01  = oxc01 + oxb01
        oxR02  = oxc02 + oxb02
        oxR12  = oxc12 + oxb12
        oxR10  = oxc10 + oxa10 + oxb10
        oxR20  = oxc20 + oxa20 + oxb20
        oxR21  = oxc21 + oxa21 + oxb21
c
c If the excitation rate from 0 -> 1 is negligible, then the same
c must be true for the 0 -> 2 rate, and the populations of the
c excited states will be essentially zero. Therefore, we skip the
c level population calculation (which can blow up), and just set
c the populations
c
        if (oxR01 .lt. 1d-40) then
          oxn0 = 1d0
          oxn1 = 0d0
          oxn2 = 0d0
        else
          call three_level_pops(oxR01, oxR02, oxR12, oxR10, oxR20,
     $                          oxR21, oxn0, oxn1, oxn2)
        endif
c
c Total emitted energy:
c
        oxlam1 = (oxa10 + oxb10) * oxe10 * oxn1 + ((oxa20 +
     $            oxb20) * oxe20 + (oxa21 + oxb21) * oxe21) *
     $            oxn2
c
c Total absorbed energy:
c
        oxlam2 = (oxb01 * oxe10 + oxb02 * oxe20) * oxn0 +
     $            oxb12 * oxe21 * oxn1
c
c Net cooling rate is emission - absorption; note that if the latter
c term is larger, we have heating
c
        rates(13) = (oxlam1 - oxlam2) * abo * yn
      endif
c
c (R14) --  CI fine-structure cooling 
c
c Collisional rates:
c
      cIc10  = cl30 * ynh + cl33 * ynh2 + cl36 * yne + cl39 * ynhp
      cIc20  = cl31 * ynh + cl34 * ynh2 + cl37 * yne + cl40 * ynhp
      cIc21  = cl32 * ynh + cl35 * ynh2 + cl38 * yne + cl41 * ynhp

      if (abcI .le. 1d-5 * abundc) then
        rates(14) = 0d0
      elseif (cIc10 .eq. 0d0 .and. cIc20 .eq. 0d0 .and.
     $    cIc21 .eq. 0d0) then
        rates(14) = 0d0
      else
        cIa =  dexp(-cIe10 / (kboltz * temp))
        cIb =  dexp(-cIe20 / (kboltz * temp))
        cIc =  dexp(-cIe21 / (kboltz * temp))

        cIc01  = 3d0 * cIc10 * cIa
        cIc02  = 5d0 * cIc20 * cIb
        cIc12  = (5d0 / 3d0) * cIc21 * cIc
c
c Stimulated emission and absorption:
c
        call compute_stim(cIa10, cIe10, CMB_temp, cIb10)
        cIb01 = 3d0 * cIb10

        call compute_stim(cIa20, cIe20, CMB_temp, cIb20)
        cIb02 = 5d0 * cIb20

        call compute_stim(cIa21, cIe21, CMB_temp, cIb21)
        cIb12 = (5d0 / 3d0) * cIb21

c Total transition rates:
c
        cIR01  = cIc01 + cIb01
        cIR02  = cIc02 + cIb02
        cIR12  = cIc12 + cIb12
        cIR10  = cIc10 + cIa10 + cIb10
        cIR20  = cIc20 + cIa20 + cIb20
        cIR21  = cIc21 + cIa21 + cIb21
c
        call three_level_pops(cIR01, cIR02, cIR12, cIR10, cIR20,
     $                        cIR21, cIn0, cIn1, cIn2)
c
c Total emitted energy:
c
        cIlam1 = (cIa10 + cIb10) * cIe10 * cIn1 + ((cIa20 +
     $            cIb20) * cIe20 + (cIa21 + cIb21) * cIe21) *
     $            cIn2
c
c Total absorbed energy:
c
        cIlam2 = (cIb01 * cIe10 + cIb02 * cIe20) * cIn0 +
     $            cIb12 * cIe21 * cIn1
c
c Net cooling rate is emission - absorption; note that if the latter
c term is larger, we have heating
c
        rates(14) = (cIlam1 - cIlam2) * abcI * yn
      endif
c
c (R15) --  SiI fine-structure cooling 
c
c Proton rates (from HM89) are constant and so there's no point
c tabulating them in coolinmo.
c
      siIc10 = cl42 * ynh + 7.2d-9 * ynhp
      siIc20 = cl43 * ynh + 7.2d-9 * ynhp
      siIc21 = cl44 * ynh + 2.2d-8 * ynhp

      if (absiI .le. 1d-5 * abundsi) then
        rates(15) = 0d0
      elseif (siIc10 .eq. 0d0 .and. siIc20 .eq. 0d0 .and.
     $    siIc21 .eq. 0d0) then
        rates(15) = 0d0
      else
        siIa =  dexp(-siIe10 / (kboltz * temp))
        siIb =  dexp(-siIe20 / (kboltz * temp))
        siIc =  dexp(-siIe21 / (kboltz * temp))
c
        siIc01  = 3d0 * siIc10 * siIa
        siIc02  = 5d0 * siIc20 * siIb
        siIc12  = (5d0 / 3d0) * siIc21 * siIc
c
c Stimulated emission and absorption:
c
        call compute_stim(siIa10, siIe10, CMB_temp, siIb10)
        siIb01 = 3d0 * siIb10

        call compute_stim(siIa20, siIe20, CMB_temp, siIb20)
        siIb02 = 5d0 * siIb20

        call compute_stim(siIa21, siIe21, CMB_temp, siIb21)
        siIb12 = (5d0 / 3d0) * siIb21
c
c Total transition rates:
c
        siIR01  = siIc01 + siIb01
        siIR02  = siIc02 + siIb02
        siIR12  = siIc12 + siIb12
        siIR10  = siIc10 + siIa10 + siIb10
        siIR20  = siIc20 + siIa20 + siIb20
        siIR21  = siIc21 + siIa21 + siIb21
c
        call three_level_pops(siIR01, siIR02, siIR12, siIR10,
     $                        siIR20, siIR21, siIn0, siIn1, siIn2)
c
c Total emitted energy:
c
        siIlam1 = (siIa10 + siIb10) * siIe10 * siIn1 +
     $            ((siIa20 + siIb20) * siIe20 + (siIa21 + siIb21) *
     $            siIe21) * siIn2
c
c Total absorbed energy:
c
        siIlam2 = (siIb01 * siIe10 + siIb02 * siIe20) * siIn0 +
     $             siIb12 * siIe21 * siIn1
c
c Net cooling rate is emission - absorption; note that if the latter
c term is larger, we have heating
c
        rates(15) = (siIlam1 - siIlam2) * absiI * yn
      endif
c
c (R16) -- CII fine-structure cooling
c
      cIIc10 = cl45 * ynh + cl46 * ynh2 + cl47 * yne
      cIIc01 = cl48 * cIIc10

      if (cIIc10 .eq. 0d0 .or. abcII .le. 1d-5 * abundc) then
        rates(16) = 0d0
      else
c
c cIIb10 is the Einstein B coefficient (B10) for the downwards 
c transition multiplied by the strength of the CMB at the 
c appropriate energy:
c
        call compute_stim(cIIa10, cIIe10, CMB_temp, cIIb10)
        cIIb01 = 2d0 * cIIb10
c
        cIIvar0 = (cIIc10 + cIIa10 + cIIb10)
        cIIvar1 = (cIIc01 + cIIb01)
c
        cIIn0 = cIIvar0 / (cIIvar0 + cIIvar1)
        cIIn1 = cIIvar1 / (cIIvar0 + cIIvar1)
c
        cIIvar2 = (cIIa10 + cIIb10) * cIIe10 * cIIn1
        cIIvar3 = cIIb01 * cIIe10 * cIIn0
c
        rates(16) = (cIIvar2 - cIIvar3) * abcII * yn
      endif
c
c (R17) -- SiII fine-structure cooling
c
      siIIc10 = cl49 * ynh + cl50 * yne
      siIIc01 = cl51 * siIIc10

      if (siIIc10 .eq. 0d0 .or. absiII .le. 1d-5 * abundsi) then
        rates(17) = 0d0
      else
        call compute_stim(siIIa10, siIIe10, CMB_temp, siIIb10)
        siIIb01 = 2d0 * siIIb10
c
        siIIvar0 = (siIIc10 + siIIa10 + siIIb10)
        siIIvar1 = (siIIc01 + siIIb01)
c
        siIIn0 = siIIvar0 / (siIIvar0 + siIIvar1)
        siIIn1 = siIIvar1 / (siIIvar0 + siIIvar1)
c
        siIIvar2 = (siIIa10 + siIIb10) * siIIn1
        siIIvar3 = siIIb01 * siIIn0
c
        rates(17) = (siIIvar2 - siIIvar3) * siIIe10 * absiII * yn
      endif
c
c (R18) -- Compton cooling
c
      compvar1  = 1.017d-37 * CMB_temp**4
      rates(18) = compvar1 * (temp - CMB_temp) * yne
c
c (R19) -- HD cooling
c
c This assumes that collisions with H dominate unless n > 1e8 (i.e. that
c the gas only becomes highly molecular at very high density)
c
      if (abhd .gt. 0d0) then
        r19scale = 1d0
        if (yn .ge. 1d8) then
          lnh = 8d0
        elseif (ynh .lt. 1d0) then
          lnh = 0d0
          r19scale = ynh
        else
          lnh = log10(ynh)
        endif
        rates(19) = 1d1**(cl53 + lnh * (cl54 + lnh * (cl55 + 
     $              lnh * (cl56 + lnh * cl57)))) * ynhd
        rates(19) = rates(19) * r19scale
      else
        rates(19) = 0d0
      endif
c
c CMB effects: see note above for molecular coolants
c
c (R20) -- H2 CIE cooling: assumes molecular fraction ~ 1, although this
c          should always be true at the densities at which CIE cooling is
c          significant
c 
      rates(20) = cl58 * yn * (2d0 * ynh2)
c
c CIE opacity cutoff; from M. Turk, private communication 
c
      tau_CIE = (yn / 1.4d16)**2.8d0
      tau_CIE = max(tau_CIE, 1d-5)
      rates(20) = rates(20) * min((1d0 - dexp(-tau_CIE)) / tau_CIE, 1d0)
c
c Benchmarking suggests that writing this out explicitly is more efficient
c than using a loop (although this is probably only true if the compiler
c optimization is poor).
c
      ylam = rates(1)  + rates(2)  + rates(3)  + rates(4)  + rates(5)  +
     $       rates(6)  + rates(7)  + rates(8)  + rates(9)  + rates(10) +
     $       rates(11) + rates(12) + rates(13) + rates(14) + rates(15) +
     $       rates(16) + rates(17) + rates(18) + rates(19) + rates(20) +
     $       rates(21) + rates(22) + rates(23) + rates(24) + rates(25) +
     $       rates(26) + rates(27) + rates(28)
#endif /* CHEMCOOL */
c
      return
      end
c=======================================================================
c
c    \\\\\\\\\\        E N D   S U B R O U T I N E        //////////
c    //////////             C O O L _ F U N C             \\\\\\\\\\
c
c=======================================================================
