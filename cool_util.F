c=======================================================================
c
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////                C O _ C O O L              \\\\\\\\\\
c
c=======================================================================
c
      subroutine co_cool(temp, N_co_eff, N_co_eff_vib, co_rot_L0, 
     $                   co_rot_lte, co_rot_alpha, co_rot_n05, 
     $                   co_vib_lte_out)
#ifdef CHEMCOOL
c
c     written by: Simon Glover, AMNH, 2004-2005, AIP, 2006
c
c  PURPOSE: Compute CO cooling rates based on tabulated Neufeld & 
c           Kaufman (1993) and Neufeld, Lepp & Melnick (1995) 
c           cooling functions [see coolinmo.F]
c
c  INPUT VARIABLES: temp, N_co_eff
c
c  OUTPUT VARIABLES: co_rot_L0, co_rot_lte, co_rot_alpha, co_rot_n05,
c                    co_vib_lte
c
c--------------------------------------------------------------------
c
      implicit none
#include "cool.h"
c
      REAL       temp, dtemp_co, dN_co, N_co_eff, N_co_eff_vib
      integer    itemp_co, iN_co
c
      REAL       co_rot_L0, co_rot_lte, co_rot_n05,
     $           co_rot_alpha, co_rot_lte_1, co_rot_lte_2,
     $           co_rot_alp_1, co_rot_alp_2, co_rot_n05_1,
     $           co_rot_n05_2, co_vib_lte_1, co_vib_lte_2,
     $           co_vib_lte_out
c
c CO rotational cooling
c
c No CO cooling below 5K, as no good data
c
      if (temp .lt. co_temptab(1)) then
        co_rot_L0    = 0.0
        co_rot_lte   = 0.0
        co_rot_n05   = 1.0
        co_rot_alpha = 1.0
        co_vib_lte_out = 0.0
        return
      elseif (temp .eq. co_temptab(1)) then
        itemp_co = 1
        dtemp_co = 0d0
      elseif (temp .ge. co_temptab(nTco)) then
        itemp_co = nTco
        dtemp_co = 0d0
      else
        itemp_co = int(temp) - 4    ! Table currently starts at 5K
        dtemp_co = temp - int(temp)
      endif
c
c For column densities that do not lie within the region covered by the
c NK93 or NLM95 data, we use the smallest or largest of the tabulated 
c values, as appropriate.
c
      if (N_co_eff .le. co_colntab(1)) then
        iN_co = 1
        dN_co = 0d0
      elseif (N_co_eff .ge. co_colntab(ncdco)) then
        iN_co = ncdco
        dN_co = 0d0
      else
        iN_co = int((10 * N_co_eff) - 144)
        dN_co = (N_co_eff - co_colntab(iN_co)) / 0.1d0
      endif
c
      co_rot_L0 = co_L0(itemp_co) + dtemp_co * dTco_L0(itemp_co)
c
      co_rot_lte_1 =   co_lte(iN_co,itemp_co) + dtemp_co *
     $               dTco_lte(iN_co,itemp_co)
      co_rot_alp_1 =   co_alp(iN_co,itemp_co) + dtemp_co *
     $               dTco_alp(iN_co,itemp_co)
      co_rot_n05_1 =   co_n05(iN_co,itemp_co) + dtemp_co *
     $               dTco_n05(iN_co,itemp_co)
      if (iN_co .eq. ncdco) then
        co_rot_lte   = co_rot_lte_1
        co_rot_alpha = co_rot_alp_1
        co_rot_n05   = co_rot_n05_1
      else
        co_rot_lte_2 =   co_lte(iN_co+1,itemp_co) + dtemp_co *
     $                 dTco_lte(iN_co+1,itemp_co)
        co_rot_alp_2 =   co_alp(iN_co+1,itemp_co) + dtemp_co *
     $                 dTco_alp(iN_co+1,itemp_co)
        co_rot_n05_2 =   co_n05(iN_co+1,itemp_co) + dtemp_co *
     $                 dTco_n05(iN_co+1,itemp_co)
c
        co_rot_lte = co_rot_lte_1 + (co_rot_lte_2 - co_rot_lte_1) *
     $               dN_co
        co_rot_alpha = co_rot_alp_1 + (co_rot_alp_2 - co_rot_alp_1) *
     $                 dN_co
        co_rot_n05 = co_rot_n05_1 + (co_rot_n05_2 - co_rot_n05_1) *
     $               dN_co
      endif
c
c Do final conversion to correct units:
c
      co_rot_L0  = 10d0**(-co_rot_L0)
      co_rot_lte = 10d0**(-co_rot_lte)
      co_rot_n05 = 10d0**(co_rot_n05)
c
c CO vibrational cooling
c
      if (temp .le. co_vib_temptab(1)) then
        itemp_co = 1
        dtemp_co = 0d0
      elseif (temp .ge. co_vib_temptab(nTco_vib)) then
        itemp_co = nTco_vib
        dtemp_co = 0d0
      else
        itemp_co = int(temp) - 99    ! Table starts at 100K
        dtemp_co = temp - int(temp)
      endif
c
      if (N_co_eff_vib .le. co_vib_colntab(1)) then
        iN_co = 1
        dN_co = 0d0
      elseif (N_co_eff_vib .ge. co_vib_colntab(ncdco_vib)) then
        iN_co = ncdco_vib
        dN_co = 0d0
      else
        iN_co = int((10 * N_co_eff) - 129)
        dN_co = (N_co_eff_vib - co_vib_colntab(iN_co)) / 0.1d0
      endif

      co_vib_lte_1 =   co_vib_lte_final(iN_co,itemp_co) + dtemp_co *
     $               dTco_vib_lte(iN_co,itemp_co)

      if (iN_co .eq. ncdco_vib) then
        co_vib_lte_out = co_vib_lte_1
      else
        co_vib_lte_2 =   co_vib_lte_final(iN_co+1,itemp_co) + dtemp_co *
     $                 dTco_vib_lte(iN_co+1,itemp_co)
        co_vib_lte_out = co_vib_lte_1 + (co_vib_lte_2 - co_vib_lte_1) *
     $               dN_co
      endif
c
      co_vib_lte_out = 10d0**(-co_vib_lte_out) * exp(-3.08d3 / temp)
c
#endif /* CHEMCOOL */
c
      return
      end
c=======================================================================
c
c    \\\\\\\\\\        E N D   S U B R O U T I N E        //////////
c    //////////               C O _ C O O L               \\\\\\\\\\
c
c=======================================================================
c
c=======================================================================
c
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////          H 2 O _ R O T _ C O O L          \\\\\\\\\\
c
c=======================================================================
c
      subroutine h2o_rot_cool(temp, N_h2o_eff, h2o_rot_L0, h2o_rot_lte, 
     $                        h2o_rot_alpha, h2o_rot_n05, iop_flag)
#ifdef CHEMCOOL
c
c     written by: Simon Glover, AMNH, 2004-2005, AIP 2006
c
c  PURPOSE: Compute H2O rotational cooling rates based on tabulated 
c           Neufeld & Kaufman (1993) and Neufeld, Lepp & Melnick (1995) 
c           cooling functions [see coolinmo.F]
c
c  INPUT VARIABLES: temp, N_h2o_eff, iop_flag (0 == para, 1 == ortho)
c
c  OUTPUT VARIABLES: h2o_rot_L0, h2o_rot_lte, h2o_rot_alpha, h2o_rot_n05
c
c--------------------------------------------------------------------
c
      implicit none
#include "cool.h"
c
      REAL     temp, dtemp_h2o, dN_h2o, N_h2o_eff
      integer  itemp_h2o, iN_h2o, iop_flag
c
      REAL     h2o_rot_L0, h2o_rot_lte, h2o_rot_n05,
     $         h2o_rot_alpha, h2o_rot_lte_1, h2o_rot_lte_2,
     $         h2o_rot_alp_1, h2o_rot_alp_2, h2o_rot_n05_1,
     $         h2o_rot_n05_2
c
      if (temp .le. h2o_temptab(1)) then
        itemp_h2o = 1
        dtemp_h2o = 0d0
      elseif (temp .ge. h2o_temptab(nTh2o)) then
        itemp_h2o = nTh2o
        dtemp_h2o = 0d0
      else
        itemp_h2o = int(temp) - 9    ! XXX: Table currently starts at 10K
        dtemp_h2o = temp - int(temp)
      endif
c
c For column densities that do not lie within the region covered by the
c NK93 or NLM95 data, we use the smallest or largest of the tabulated 
c values, as appropriate.
c
      if (N_h2o_eff .le. h2o_colntab(1)) then
        iN_h2o = 1
        dN_h2o = 0d0
      elseif (N_h2o_eff .ge. h2o_colntab(ncdh2o)) then
        iN_h2o = ncdh2o
        dN_h2o = 0d0
      else
        iN_h2o = int((10 * N_h2o_eff) - 144)
        dN_h2o = (N_h2o_eff - h2o_colntab(iN_h2o)) / 0.1d0
      endif
c
      if (iop_flag .eq. 0) then
        h2o_rot_L0 = h2o_L0_para(itemp_h2o) + dtemp_h2o * 
     $             dTh2o_L0_para(itemp_h2o)
        h2o_rot_lte_1 = h2o_lte_para(iN_h2o,itemp_h2o) + 
     $    dtemp_h2o * dTh2o_lte_para(iN_h2o,itemp_h2o)
        h2o_rot_alp_1 = h2o_alp_para(iN_h2o,itemp_h2o) + 
     $    dtemp_h2o * dTh2o_alp_para(iN_h2o,itemp_h2o)
        h2o_rot_n05_1 = h2o_n05_para(iN_h2o,itemp_h2o) + 
     $    dtemp_h2o * dTh2o_n05_para(iN_h2o,itemp_h2o)
        if (iN_h2o .eq. ncdh2o) then
          h2o_rot_lte   = h2o_rot_lte_1
          h2o_rot_alpha = h2o_rot_alp_1
          h2o_rot_n05   = h2o_rot_n05_1
        else
          h2o_rot_lte_2 = h2o_lte_para(iN_h2o+1,itemp_h2o) + 
     $      dtemp_h2o * dTh2o_lte_para(iN_h2o+1,itemp_h2o)
          h2o_rot_alp_2 = h2o_alp_para(iN_h2o+1,itemp_h2o) + 
     $      dtemp_h2o * dTh2o_alp_para(iN_h2o+1,itemp_h2o)
          h2o_rot_n05_2 = h2o_n05_para(iN_h2o+1,itemp_h2o) + 
     $      dtemp_h2o * dTh2o_n05_para(iN_h2o+1,itemp_h2o)
c
          h2o_rot_lte = h2o_rot_lte_1 + (h2o_rot_lte_2 - 
     $                  h2o_rot_lte_1) * dN_h2o
          h2o_rot_alpha = h2o_rot_alp_1 + (h2o_rot_alp_2 - 
     $                    h2o_rot_alp_1) * dN_h2o
          h2o_rot_n05 = h2o_rot_n05_1 + (h2o_rot_n05_2 - 
     $                  h2o_rot_n05_1) * dN_h2o
        endif
      else
        h2o_rot_L0 = h2o_L0_ortho(itemp_h2o) + dtemp_h2o * 
     $             dTh2o_L0_ortho(itemp_h2o)
        h2o_rot_lte_1 = h2o_lte_ortho(iN_h2o,itemp_h2o) + 
     $    dtemp_h2o * dTh2o_lte_ortho(iN_h2o,itemp_h2o)
        h2o_rot_alp_1 = h2o_alp_ortho(iN_h2o,itemp_h2o) + 
     $    dtemp_h2o * dTh2o_alp_ortho(iN_h2o,itemp_h2o)
        h2o_rot_n05_1 = h2o_n05_ortho(iN_h2o,itemp_h2o) + 
     $    dtemp_h2o * dTh2o_n05_ortho(iN_h2o,itemp_h2o)
        if (iN_h2o .eq. ncdh2o) then
          h2o_rot_lte   = h2o_rot_lte_1
          h2o_rot_alpha = h2o_rot_alp_1
          h2o_rot_n05   = h2o_rot_n05_1
        else 
          h2o_rot_lte_2 = h2o_lte_ortho(iN_h2o+1,itemp_h2o) + 
     $        dtemp_h2o * dTh2o_lte_ortho(iN_h2o+1,itemp_h2o)
          h2o_rot_alp_2 = h2o_alp_ortho(iN_h2o+1,itemp_h2o) + 
     $        dtemp_h2o * dTh2o_alp_ortho(iN_h2o+1,itemp_h2o)
          h2o_rot_n05_2 = h2o_n05_ortho(iN_h2o+1,itemp_h2o) + 
     $        dtemp_h2o * dTh2o_n05_ortho(iN_h2o+1,itemp_h2o)
c
          h2o_rot_lte = h2o_rot_lte_1 + (h2o_rot_lte_2 - 
     $                  h2o_rot_lte_1) * dN_h2o
          h2o_rot_alpha = h2o_rot_alp_1 + (h2o_rot_alp_2 - 
     $                    h2o_rot_alp_1) * dN_h2o
          h2o_rot_n05 = h2o_rot_n05_1 + (h2o_rot_n05_2 - 
     $                  h2o_rot_n05_1) * dN_h2o
        endif
      endif
c
c Do final conversion to correct units:
c
      h2o_rot_L0  = 10d0**(-h2o_rot_L0)
      h2o_rot_lte = 10d0**(-h2o_rot_lte)
      h2o_rot_n05 = 10d0**(h2o_rot_n05)
c
#endif /* CHEMCOOL */
c
      return
      end
c=======================================================================
c
c    \\\\\\\\\\        E N D   S U B R O U T I N E        //////////
c    //////////          H 2 O _ R O T _ C O O L          \\\\\\\\\\
c
c=======================================================================
c
c=======================================================================
c
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////          H 2 O _ V I B _ C O O L          \\\\\\\\\\
c
c=======================================================================
c
      subroutine h2o_vib_cool(temp, N_h2o_eff, h2o_vib_lte_out)
#ifdef CHEMCOOL
c
c     written by: Simon Glover, AMNH, 2004-2005, AIP 2006
c
c  PURPOSE: Compute H2O vibrational cooling rates based on tabulated 
c           Neufeld & Kaufman (1993) and Neufeld, Lepp & Melnick (1995) 
c           cooling functions [see coolinmo.F]
c
c  INPUT VARIABLES: temp, N_h2o_eff
c
c  OUTPUT VARIABLES: h2o_vib_lte_out
c
c--------------------------------------------------------------------
c
      implicit none
#include "cool.h"
c
      REAL     temp, dtemp_h2o, dN_h2o, N_h2o_eff
      integer  itemp_h2o, iN_h2o
c
      REAL     h2o_vib_lte_1, h2o_vib_lte_2,
     $         h2o_vib_lte_out
c
c H2O vibrational cooling
c
      if (temp .le. h2o_vib_temptab(1)) then
        itemp_h2o = 1
        dtemp_h2o = 0d0
      elseif (temp .ge. h2o_vib_temptab(nTh2o_vib)) then
        itemp_h2o = nTh2o_vib
        dtemp_h2o = 0d0
      else
        itemp_h2o = int(temp) - 99    ! Table starts at 100K
        dtemp_h2o = temp - int(temp)
      endif
c
      if (N_h2o_eff .le. h2o_vib_colntab(1)) then
        iN_h2o = 1
        dN_h2o = 0d0
      elseif (N_h2o_eff .ge. h2o_vib_colntab(ncdh2o_vib)) then
        iN_h2o = ncdh2o_vib
        dN_h2o = 0d0
      else
        iN_h2o = int((10 * N_h2o_eff) - 129)
        dN_h2o = (N_h2o_eff - h2o_vib_colntab(iN_h2o)) / 0.1d0
      endif

      h2o_vib_lte_1 =   h2o_vib_lte_final(iN_h2o,itemp_h2o) + 
     $               dtemp_h2o * dTh2o_vib_lte(iN_h2o,itemp_h2o)

      if (iN_h2o .eq. ncdh2o_vib) then
        h2o_vib_lte_out = h2o_vib_lte_1
      else
        h2o_vib_lte_2 =   h2o_vib_lte_final(iN_h2o+1,itemp_h2o) + 
     $                 dtemp_h2o * dTh2o_vib_lte(iN_h2o+1,itemp_h2o)
        h2o_vib_lte_out = h2o_vib_lte_1 + (h2o_vib_lte_2 - 
     $                    h2o_vib_lte_1) * dN_h2o
      endif
c
      h2o_vib_lte_out = 10d0**(-h2o_vib_lte_out) * exp(-2.325d3 / temp)
c
#endif /* CHEMCOOL */
c
      return
      end
c=======================================================================
c
c    \\\\\\\\\\        E N D   S U B R O U T I N E        //////////
c    //////////          H 2 O _ V I B _ C O O L          \\\\\\\\\\
c
c=======================================================================
c
c=======================================================================
c
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////      T H R E E _ L E V E L _ P O P S      \\\\\\\\\\
c
c=======================================================================
c
      subroutine three_level_pops(r01, r02, r12, r10, r20, r21, 
     $                            n0, n1, n2)
#ifdef CHEMCOOL
      implicit none
#include "cool.h"
      REAL r01, r02, r12, r10, r20, r21
      REAL n0 , n1 , n2
      REAL a1 , a2 , a3 , b1 , b2 , b3
c
c If excitation rates are negligibly small, then we assume that all
c of the atoms are in level 0:
c
      if (r01 .eq. 0d0 .and. r02 .eq. 0d0) then
        n0 = 1d0
        n1 = 0d0
        n2 = 0d0
        return
      endif

      a1 = r01 + r02
      a2 = -r10
      a3 = -r20
      b1 = r01
      b2 = -(r10 + r12)
      b3 = r21
c
      n2 = -a1 * (a1 * b2 - b1 * a2) / ((a1 - a2) * 
     $     (a1 * b3 - b1 * a3) - (a1 - a3) * 
     $     (a1 * b2 - b1 * a2))
c
      n1 = (a1 / (a1 - a2)) - ((a1 - a3) / (a1 - a2)) * n2
c
      n0 = 1d0 - n1 - n2
c
#endif /* CHEMCOOL */
c
      return
      end
c=======================================================================
c
c
c    \\\\\\\\\\        E N D   S U B R O U T I N E      //////////
c    //////////      T H R E E _ L E V E L _ P O P S      \\\\\\\\\\
c
c=======================================================================
c
c=======================================================================
c
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////          C O M P U T E _ S T I M          \\\\\\\\\\
c
c=======================================================================
c
      subroutine compute_stim(a10, e10, rad_temp, b10)
#ifdef CHEMCOOL
      implicit none
#include "cool.h"
      REAL a10, e10, rad_temp, b10
      REAL x

      x = e10 / (kboltz * rad_temp)
      if (x .lt. 5d0) then
        b10 = a10 / (dexp(x) - 1d0)
      else
        b10 = 0d0
      endif

#endif /* CHEMCOOL */
c
      return
      end
c=======================================================================
c
c
c    \\\\\\\\\\        E N D   S U B R O U T I N E      //////////
c    //////////          C O M P U T E _ S T I M          \\\\\\\\\\
c
c=======================================================================
c
c=======================================================================
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////        L O A D _ H 2 _ T A B L E          \\\\\\\\\\
c
c=======================================================================
c
      subroutine load_H2_table
#ifdef CHEMCOOL
      implicit none
#include "cool.h"
#include "mol_data.h"
      integer I, J

      open(12, file='H2-cooling-ratios.dat', status='old')
      do I = 1, nh2op
        do J = 1, nh2op
          read(12,*) h2_opac_temp(I), h2_opac_column(J), h2_opac(I,J)
        enddo
      enddo
      close (12, status='keep')

#endif /* CHEMCOOL */
c
      return
      end
c=======================================================================
c
c    \\\\\\\\\\        E N D   S U B R O U T I N E        //////////
c    //////////         L O A D _ H 2 _ T A B L E         \\\\\\\\\\
c
c=======================================================================

c=======================================================================
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////    C O M P U T E _ H 2 _ O P A C I T Y    \\\\\\\\\\
c
c=======================================================================
c
      subroutine compute_h2_opacity(temp, N_H2_eff, opac)
#ifdef CHEMCOOL
      implicit none
#include "cool.h"
#include "mol_data.h"
      REAL temp, N_H2_eff, opac
      REAL column_min, column_max, logN, logT, diff, dN, dT
      REAL opac_tmp(2)
      integer ii, jj

      column_min = 10**(h2_opac_column(1))
      column_max = 10**(h2_opac_column(nh2op))

      if (N_H2_eff .le. column_min) then
        opac = 1d0
        return
      elseif (N_H2_eff .ge. column_max) then
        opac = 0d0
        return
      else
        logN = log10(N_H2_eff)
        jj   = 1 + int(10 * (logN - 17.0))
        diff = h2_opac_column(jj+1) - h2_opac_column(jj)
        dN   = (logN - h2_opac_column(jj)) / diff
c
        logT = log10(temp)
        if (logT .le. h2_opac_temp(1)) then
          ii = 1
          dT = 0d0
        elseif (logT .ge. h2_opac_temp(nh2op)) then  
          ii = nh2op
          dT = 0d0
        else
          ii   = 1 + int((logT - 1.5) / 0.03)
          diff = h2_opac_temp(ii+1) - h2_opac_temp(ii)
          dT   = (logT - h2_opac_temp(ii)) / diff
        endif
c
        if (dT .gt. 0d0) then
          opac_tmp(1) = h2_opac(ii,jj) + dT * (h2_opac(ii+1,jj) 
     $                - h2_opac(ii,jj))
          opac_tmp(2) = h2_opac(ii,jj+1) + dT * (h2_opac(ii+1,jj+1) 
     $                - h2_opac(ii,jj+1))
        else
          opac_tmp(1) = h2_opac(ii,jj)
          opac_tmp(2) = h2_opac(ii,jj+1)
        endif
c
        opac = opac_tmp(1) + dN * (opac_tmp(2) - opac_tmp(1))
        opac = 1d1**opac
      endif
#endif /* CHEMCOOL */
c
      return
      end
c=======================================================================
c
c    \\\\\\\\\\        E N D   S U B R O U T I N E        //////////
c    //////////    C O M P U T E _ H 2 _ O P A C I T Y    \\\\\\\\\\
c
c=======================================================================
c=======================================================================
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////        COMPUTE H2 HEATING FRACTION        \\\\\\\\\\
c
c=======================================================================
c
      subroutine compute_h2_heating_fraction(T, n, xh2, frac)
#ifdef CHEMCOOL
      implicit none
#include "cool.h"
#include "h2heat.h"
      REAL T, n, xh2, frac
      REAL logT, logn, dT, dn, nscale, dh2
      REAL f1, f2, f3, f4, fr1, fr2, frac1, frac2
      integer idx1, idx2, idx3, offset

      logT = log10(T)
      logn = log10(n)

      if (logT .lt. 1d0) then
        idx1 = 1
        dT = 0d0
      elseif (logT .gt. 3.5d0) then
        idx1 = 11
        dT = 0d0
      else
        idx1 = 1 + int(4d0 * (logT - 1d0))
        dT = 4d0 * (logT - 1d0) - (idx1 - 1)
      endif

      if (logn .lt. 1d0) then
        idx2   = 1
        dn     = 0d0
        nscale = (n / 1d1)
      elseif (logn .gt. 8d0) then
        idx2   = 29
        dn     = 0d0
        nscale = 1d0
      else
        idx2 = 1 + int(4d0 * (logn - 1d0))
        dn = 4d0 * (logn - 1d0) - (idx2 - 1)
        nscale = 1d0
      endif

      if (xh2 .eq. 0.5d0) then
        idx3 = 6
        dh2  = 0d0
      else
        idx3 = 1 + int(xh2 / 1d-1)
        dh2  = xh2 / 1d-1 - (idx3 - 1)
      endif
c
c Compute frac for idx3
c 
      offset = 319 * (idx3 - 1) + 29 * (idx1 - 1) + idx2

      if (dT .eq. 0d0) then
        if (dn .eq. 0d0) then
          frac1 = nscale * h2_rv_heat(offset)
        else
          f1 = h2_rv_heat(offset)
          f2 = h2_rv_heat(offset + 1)
          frac1 = f1 + dn * (f2 - f1)
        endif
      else
        if (dn .eq. 0d0) then
          f1 = nscale * h2_rv_heat(offset)
          f2 = nscale * h2_rv_heat(offset + 29)
          frac1 = f1 + dT * (f2 - f1)
        else
          f1 = h2_rv_heat(offset)
          f2 = h2_rv_heat(offset + 1)
          f3 = h2_rv_heat(offset + 29)
          f4 = h2_rv_heat(offset + 30)
          fr1 = f1 + dT * (f2 - f1)
          fr2 = f3 + dT * (f4 - f3)
          frac1 = fr1 + dn * (fr2 - fr1)
        endif
      endif
c
c Compute frac for idx3+1 (if required)
c 
      if (dh2 .eq. 0d0) then
        frac = frac1
      else
        offset = 319 * idx3 + 29 * (idx1 - 1) + idx2

        if (dT .eq. 0d0) then
          if (dn .eq. 0d0) then
            frac2 = nscale * h2_rv_heat(offset)
          else
            f1 = h2_rv_heat(offset)
            f2 = h2_rv_heat(offset + 1)
            frac2 = f1 + dn * (f2 - f1)
          endif
        else
          if (dn .eq. 0d0) then
            f1 = nscale * h2_rv_heat(offset)
            f2 = nscale * h2_rv_heat(offset + 29)
            frac2 = f1 + dT * (f2 - f1)
          else
            f1 = h2_rv_heat(offset)
            f2 = h2_rv_heat(offset + 1)
            f3 = h2_rv_heat(offset + 29)
            f4 = h2_rv_heat(offset + 30)
            fr1 = f1 + dT * (f2 - f1)
            fr2 = f3 + dT * (f4 - f3)
            frac2 = fr1 + dn * (fr2 - fr1)
          endif
        endif

        frac = frac1 + dh2 * (frac2 - frac1)
      endif
c
#endif /* CHEMCOOL */
c
      return
      end
c=======================================================================
c
c    \\\\\\\\\\        E N D   S U B R O U T I N E        //////////
c    //////////        COMPUTE H2 HEATING FRACTION        \\\\\\\\\\
c
c=======================================================================
c
c=======================================================================
c
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////             D M A _ C A L C               \\\\\\\\\\
c
c=======================================================================
c
#if defined DMA && defined CHEMCOOL
      subroutine dma_calc(gas_column, dm_density_gevcc, ion_rate, 
     $                    heat_rate)
      implicit none
#include "cool.h"
#include "dma_data.h"
c gas_column has units g/cm^2
c dm_density has units GeV/c^2/cm^3
      REAL gas_column, dm_density_gevcc, ion_rate, heat_rate, eps_ann

      integer I, idx
      REAL diff, mdm, fQ

c k_ann is the thermally averaged annilihation rate coefficient,
c <sigma v>, with units 3d-26 cm^3 s^-1
      REAL k_ann
      parameter (k_ann = 3d-26)

c Compute index for table lookup
      if (gas_column .le. fQ_en(1)) then
c For now, we asssume that fQ is zero if we're below the first of the
c tabulated values
        idx  = 1
        diff = 0
      elseif (gas_column .ge. fQ_en(ifqsz)) then
        idx  = ifqsz
        diff = 0d0
      else
        do I = 1, ifqsz
          if (gas_column .ge. fQ_en(I) .and. 
     $        gas_column .le. fQ_en(I+1)) then
            idx = I
            diff = (gas_column - fQ_en(idx)) / 
     $             (fQ_en(idx+1) - fQ_en(idx))
            goto 100
          endif
        enddo
      endif

100   continue

c mdm is dark matter particle mass, in units of GeV
c fQ is the fraction of energy absorbed by the gas
      if (idma_mass_option .eq. 1) then
        mdm = 1d0
        if (diff .eq. 0d0) then
          fQ = fQ_1(idx)
        else
          fQ = fQ_1(idx) + diff * (fQ_1(idx+1) - fQ_1(idx))
        endif
      elseif (idma_mass_option .eq. 2) then
        mdm = 1d1
        if (diff .eq. 0d0) then
          fQ = fQ_10(idx)
        else
          fQ = fQ_10(idx) + diff * (fQ_10(idx+1) - fQ_10(idx))
        endif
      elseif (idma_mass_option .eq. 3) then
        mdm = 1d2
        if (diff .eq. 0d0) then
          fQ = fQ_100(idx)
        else
          fQ = fQ_100(idx) + diff * (fQ_100(idx+1) - fQ_100(idx))
        endif
      elseif (idma_mass_option .eq. 4) then
        mdm = 1d3
        if (diff .eq. 0d0) then
          fQ = fQ_1000(idx)
        else
          fQ = fQ_1000(idx) + diff * (fQ_1000(idx+1) - fQ_1000(idx))
        endif
      else
        print*, "Error: unknown DM mass option ", 
     $          idma_mass_option
        stop
      endif
      
      eps_ann = 1.602d-29 * fQ * k_ann * dm_density_gevcc**2 / mdm

c XXX: Paolo gives us an expression for the ionization rate, but
c      how much of the rest of the energy goes into heat?
      ion_rate  = 0d0
      heat_rate = 0d0

      return
      end
#endif /* DMA && CHEMCOOL */
c
c=======================================================================
c
c
c    \\\\\\\\\\        E N D     S U B R O U T I N E      //////////
c    //////////             D M A _ C A L C               \\\\\\\\\\
c
c=======================================================================

