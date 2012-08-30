c
c  Originally based on work by G. Suttner (Univ. Wuerzburg, 1995) and  
c  M. D. Smith (Armagh Observatory, 2000-2001).
c  Substantially modified and rewritten by S. Glover (AMNH, 2002-2005, AIP 2006-2007)
c
c /*from param.h in ZEUSMP for FORTRAN files*/
#if defined SINGLE_PRECISON || defined UNICOS 
#define REAL real
#else
#define REAL real*8
#endif

#define UNUSED_PARAM(x)  x = x
#define ABORT(x) stop

#include "chemcool_consts.h"
#ifdef CHEMCOOL
c
c He:H ratio by number (=> ratio by mass is 4*abhe)
c
      REAL abhe
      parameter(abhe = ABHE)
c
c Symbolic constants representing the slot in the abundance vector passed
c to DVODE occupied by each species. The same ordering is used in 
c TracAbund, but the numbers all shift down by one (since C arrays are 
c zero based). Note that we include the internal energy in this list,
c since we pass it to DVODE, but that it is handled separately elsewhere
c in the code.
c
      integer ih2
      parameter(ih2 = IH2+1)

      integer ihp
      parameter(ihp = IHP+1)

      integer ic
      parameter(ic  = IC+1)

      integer isi
      parameter(isi = ISi+1)

      integer io
      parameter(io  = IO+1)

      integer ihd
      parameter(ihd = IHD+1)

      integer idp
      parameter(idp = IDP+1)

      integer ihep
      parameter(ihep = IHEP+1)

      integer ihepp
      parameter(ihepp = IHEPP+1)

      integer ico
      parameter(ico  = ICO+1)

      integer icos
      parameter(icos = ICOS+1)

      integer ic2
      parameter(ic2  = IC2+1)

      integer ioh
      parameter(ioh  = IOH+1)

      integer ih2o
      parameter(ih2o  = IH2O+1)

      integer io2
      parameter(io2  = IO2+1)

      integer ihcop
      parameter(ihcop = IHCOP+1)

      integer ich
      parameter(ich = ICH+1)

      integer ich2
      parameter(ich2 = ICH2+1)

      integer isipp
      parameter(isipp = ISIPP+1)

      integer ich3p
      parameter(ich3p = ICH3P+1)

      integer imgp
      parameter(imgp = IMGP+1)

      integer ich3
      parameter(ich3 = ICH3+1)

      integer ich4
      parameter(ich4 = ICH4+1)

      integer ico2
      parameter(ico2 = ICO2+1)

      integer ih2os
      parameter(ih2os = IH2OS+1)

      integer io2s
      parameter(io2s = IO2S+1)

      integer itmp
      parameter(itmp = ITMP+1)

      integer itd
      parameter(itd = ITD+1)

c Number of entries in cooling table
      integer nmd
      parameter(nmd = NMD)

c Number of cooling / heating rates computed in cooling fn.
      integer nrates
      parameter(nrates = NRATES)

c Number of cooling / heating rates computed in chemistry routines
      integer nrates_chem
      parameter(nrates_chem = NRATES_CHEM)

c Total number of cooling / heating rates
      integer nrates_tot
      parameter(nrates_tot = nrates + nrates_chem)

c Number of abundances passed to cooling function
c (Note that this is not necessarily the same as the number
c that we actually track as field variables)
      integer nabn
      parameter(nabn = ABUND_NUM)

c Boltzmann constant
      REAL kboltz
      parameter (kboltz = BOLTZMANN)

c One electron volt, in ergs
      REAL eV
      parameter (eV = ELECTRON_VOLT)

c FP tolerance for H2 abundance test
      REAL cool_eps
      parameter (cool_eps = COOL_EPS)

c Number of different quantities stored in cooling look-up table
      integer ncltab
      parameter (ncltab = NCLTAB) 

c Number of different quantities stored in chemistry look-up table
      integer nchtab
      parameter (nchtab = NCHTAB) 

c Number of cosmic ray ionization rates tabulated
      integer ncrtab
      parameter(ncrtab = NCRTAB)

c Number of cosmic ray induced photoionizations/photodissociations tabulated
      integer ncrphot
      parameter(ncrphot = NCRPHOT)

c Number of photochemical rates tabulated
      integer nphtab
      parameter(nphtab = NPHTAB)

c Size of rate array returned by calc_photo - NB not all of the entries are filled
      integer npr
      parameter (npr = NPR)

c Number of constant rate coefficient initialized in const_rates
      integer nconst
      parameter(nconst = NCONST)

c These variables are initialized in cheminmo
      REAL chtab(nchtab,nmd), dtchtab(nchtab,nmd),
     $     crtab(ncrtab), crphot(ncrphot)

c These variables are initialized in photoinit
      REAL phtab(nphtab), f_rsc

c This is initialized in const_rates
      REAL cst(nconst)

c These variables are initialized in coolinmo
      REAL temptab(nmd)
      REAL cltab(ncltab,nmd), dtcltab(ncltab,nmd)
      REAL dtlog,tmax,tmin
c
c CO rotational cooling
       integer nTco
       parameter (nTco = 1996)

       integer ncdco
       parameter (ncdco = 46)

       REAL co_temptab(nTco), co_colntab(ncdco)

       REAL co_L0(nTco), dTco_L0(nTco)
       REAL co_lte(ncdco,nTco), co_n05(ncdco,nTco), co_alp(ncdco,nTco)
       REAL dTco_lte(ncdco,nTco), dTco_n05(ncdco,nTco), 
     $      dTco_alp(ncdco,nTco)

c CO vibrational cooling
       integer nTco_vib
       parameter (nTco_vib = 3901)

       integer ncdco_vib
       parameter (ncdco_vib = 61)

       REAL co_vib_temptab(nTco_vib), co_vib_colntab(ncdco_vib)
       REAL co_vib_LTE_final(ncdco_vib, nTco_vib)
       REAL dTco_vib_LTE(ncdco_vib, nTco_vib)

       common /co_data/ co_temptab, co_colntab, co_L0, dTco_L0,
     $                  co_lte, co_n05, co_alp, dTco_lte, dTco_n05, 
     $                  dTco_alp, co_vib_temptab, co_vib_colntab,
     $                  co_vib_LTE_final, dTco_vib_LTE

c H2O rotational cooling
       integer nTh2o
       parameter (nTh2o = 3991)

       integer ncdh2o
       parameter (ncdh2o = 91)

       REAL h2o_temptab(nTh2o), h2o_colntab(ncdh2o)

       REAL h2o_L0_ortho(nTh2o), dTh2o_L0_ortho(nTh2o),
     $      h2o_L0_para(nTh2o),  dTh2o_L0_para(nTh2o)

       REAL h2o_LTE_ortho(ncdh2o,nTh2o), 
     $      h2o_n05_ortho(ncdh2o,nTh2o), 
     $      h2o_alp_ortho(ncdh2o,nTh2o),
     $      h2o_LTE_para(ncdh2o,nTh2o), 
     $      h2o_n05_para(ncdh2o,nTh2o), 
     $      h2o_alp_para(ncdh2o,nTh2o)

       REAL dTh2o_LTE_ortho(ncdh2o,nTh2o), 
     $      dTh2o_n05_ortho(ncdh2o,nTh2o), 
     $      dTh2o_alp_ortho(ncdh2o,nTh2o),
     $      dTh2o_LTE_para(ncdh2o,nTh2o), 
     $      dTh2o_n05_para(ncdh2o,nTh2o), 
     $      dTh2o_alp_para(ncdh2o,nTh2o)

c H2O vibrational cooling
       integer nTh2o_vib
       parameter (nTh2o_vib = 3901)

       integer ncdh2o_vib
       parameter (ncdh2o_vib = 61)

       REAL h2o_vib_temptab(nTh2o_vib), h2o_vib_colntab(ncdh2o_vib)
       REAL h2o_vib_LTE_final(ncdh2o_vib, nTh2o_vib)
       REAL dTh2o_vib_LTE(ncdh2o_vib, nTh2o_vib)

       common /h2o_data/ h2o_temptab, h2o_colntab, h2o_L0_ortho,
     $                   dTh2o_L0_ortho, h2o_L0_para, dTh2o_L0_para,
     $                   h2o_LTE_ortho, h2o_n05_ortho, h2o_alp_ortho,
     $                   h2o_LTE_para, h2o_n05_para, h2o_alp_para,
     $                   dTh2o_LTE_ortho, dTh2o_n05_ortho,
     $                   dTh2o_alp_ortho, dTh2o_LTE_para,
     $                   dTh2o_n05_para, dTh2o_alp_para,
     $                   h2o_vib_temptab, h2o_vib_colntab,
     $                   h2o_vib_LTE_final, dTh2o_vib_LTE
c
c These variables are initialized during problem setup
c 
      REAL deff, abundc, abundo, abundsi, abundD, abundmg, G0, 
     $     phi_pah, tdust, dust_to_gas_ratio, 
     $     AV_conversion_factor, cosmic_ray_ion_rate, redshift, 
     $     AV_ext, pdv_term, h2_form_ex, h2_form_kin, dm_density
      integer iphoto, iflag_mn, iflag_ad, iflag_atom, 
     $        iflag_3bh2a, iflag_3bh2b, iflag_h3pra,
     $        iflag_h2opc, id_current, idma_mass_option,
     $        no_chem,irad_heat,index_current
#ifdef TEST
      integer itest
#endif /* TEST */

      common /coolr/ temptab, cltab, chtab, dtcltab, dtchtab, 
     $               crtab, crphot, 
     $               phtab, cst, dtlog, tdust, tmax, tmin, 
     $               deff, abundc, abundo, abundsi, abundD, 
     $               abundmg, G0, f_rsc, phi_pah, 
     $               dust_to_gas_ratio, AV_conversion_factor,
     $               cosmic_ray_ion_rate, redshift, AV_ext,
     $               pdv_term, h2_form_ex, h2_form_kin,
     $               dm_density

      common /cooli/ iphoto, iflag_mn, iflag_ad, iflag_atom
     $,              iflag_3bh2a, iflag_3bh2b, iflag_h3pra
     $,              iflag_h2opc,id_current, index_current
     $,              idma_mass_option, no_chem, irad_heat
#ifdef TEST
     $,              itest
#endif

#ifdef THERMAL_INFO_DUMP
      REAL cool_h2_line, cool_h2_cie, cool_h2_diss
      REAL heat_3b, pdv_heat
c  /* crjs */
      REAL dtcool_nopdv,acc_heat
c  /* crjs */
      common /thermal_info/ cool_h2_line, cool_h2_cie, 
     $       cool_h2_diss, heat_3b, pdv_heat, dtcool_nopdv,
     $       acc_heat
#endif

#ifdef DEBUG_COOLING_RATES
      REAL radiative_rates(nrates), chemical_rates(nrates_chem)
      REAL newdt
      common /rate_block/ radiative_rates, chemical_rates, newdt
#endif

#ifdef TREE_RAD
#define NPIX  12*NSIDE*NSIDE
      REAL column_density_projection(NPIX)
      common /project/ column_density_projection
#endif

#if CHEMISTRYNETWORK == 1 || CHEMISTRYNETWORK == 8
      integer no_dchem
      common /dchem_block/ no_dchem
#endif

#if CHEMISTRYNETWORK == 10
      REAL gamma_gd
      common /g2d/ gamma_gd
#endif

#ifdef ADIABATIC_DENSITY_THRESHOLD
      REAL yn_adiabatic
      parameter (yn_adiabatic = ADIABATIC_DENSITY_THRESHOLD)
#endif

      integer nradsource
      integer max_nrad
      parameter (max_nrad = 10000)
      REAL rad_source_distances(max_nrad)
      REAL rad_source_luminosities(max_nrad)
      character*4 dummy
      common /rad_source_data/ dummy, nradsource, rad_source_distances, 
     $                         rad_source_luminosities


#endif /* CHEMCOOL */
