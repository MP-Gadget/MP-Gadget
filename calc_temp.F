c=======================================================================
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////          INIT_TEMPERATURE_LOOKUP          \\\\\\\\\\
c
c=======================================================================
c
      subroutine init_temperature_lookup
#ifdef CHEMCOOL
      implicit none
#include "cool.h"
#include "gamma.h"
      REAL foo
      integer I

      open(1, file='energy-table.txt', status='old')
      do I = 1, isize
        read(1, *) foo, temp_table(I), gamma_table(I)
      enddo
      close(1, status='keep')
      open(1, file='eh2-table.txt', status='old')
      do I = 1, isize2
        read(1, *) foo, eh2_table(I)
      enddo
      close(1, status='keep')

#endif /* CHEMCOOL */
      end
c=======================================================================
c
c    \\\\\\\\\\        E N D   S U B R O U T I N E        //////////
c    //////////          INIT_TEMPERATURE_LOOKUP          \\\\\\\\\\
c
c=======================================================================
c
c
c=======================================================================
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////             C A L C _ S P E C             \\\\\\\\\\
c
c=======================================================================
c
c Compute contribution of H2 to specific energy of gas
c
      subroutine calc_spec(temp, en)
#ifdef CHEMCOOL
      implicit none
#include "cool.h"
#include "gamma.h"
      REAL en, temp, dtemp
      integer idx
c
      if (temp .lt. 1d0 .or. temp .gt. 5d4) then
        en = 1.5d0 * temp
      else
        idx   = int(temp)
        dtemp = temp - idx
        en = eh2_table(idx) + dtemp * (eh2_table(idx+1)
     $     - eh2_table(idx))
      endif
      return
#endif
      end
c=======================================================================
c
c    \\\\\\\\\\         E N D    S U B R O U T I N E      //////////
c    //////////             C A L C _ S P E C             \\\\\\\\\\
c
c=======================================================================
c
c=======================================================================
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////     C A L C _  G A M M A _ F R O M _ T    \\\\\\\\\\
c
c=======================================================================
c
c Compute contribution of H2 to specific energy of gas
c
      subroutine calc_gamma_temp(abh2, temp, gamma)
#ifdef CHEMCOOL
      implicit none
#include "cool.h"
#include "gamma.h"
      REAL gamma, temp, en, abh2, ekn

      call calc_spec(temp, en)
      ekn = (abh2 / (1d0 + abhe - abh2)) * en + 
     $      ((1d0 + abhe) / (1d0 + abhe - abh2)) * 1.5d0 * kboltz * temp
      call calc_gamma(abh2, ekn, gamma)

      return
#endif
      end
c=======================================================================
c
c    \\\\\\\\\\         E N D    S U B R O U T I N E      //////////
c    //////////     C A L C _  G A M M A _ F R O M _ T    \\\\\\\\\\
c
c=======================================================================
c
c=======================================================================
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////             C A L C _ T E M P             \\\\\\\\\\
c
c=======================================================================
c
      subroutine calc_temp(abh2, ekn, temp)
#ifdef CHEMCOOL
      implicit none
#include "cool.h"
      REAL abh2, ekn, temp, gamma
c
      call lookup_temp_gamma(abh2, ekn, temp, gamma)
#endif
      end
c=======================================================================
c
c    \\\\\\\\\\         E N D    S U B R O U T I N E      //////////
c    //////////             C A L C _ T E M P             \\\\\\\\\\
c
c=======================================================================
c
c=======================================================================
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////           C A L C _ G A M M A             \\\\\\\\\\
c
c=======================================================================
c
      subroutine calc_gamma(abh2, ekn, gamma)
#ifdef CHEMCOOL
      implicit none
#include "cool.h"
      REAL abh2, ekn, temp, gamma
c
      call lookup_temp_gamma(abh2, ekn, temp, gamma)
#endif
      end
c=======================================================================
c
c    \\\\\\\\\\         E N D  S U B R O U T I N E      //////////
c    //////////           C A L C _ G A M M A           \\\\\\\\\\
c
c=======================================================================
c
c=======================================================================
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////     L O O K U P _ T E M P _ G A M M A     \\\\\\\\\\
c
c=======================================================================
      subroutine lookup_temp_gamma(abh2, ekn, temp, gamma)
#ifdef CHEMCOOL
      implicit none
#include "cool.h"
#include "gamma.h"
      REAL abh2, ekn, temp, gamma
      REAL foo, diff_en, diff_h2
      REAL t1, t2, g1, g2
      REAL temp_1, temp_2, gamma_1, gamma_2
      integer I, idx_en, idx_h2
c
      if (ekn .lt. 1d0 .or. ekn .gt. 5d4 .or. abh2 .eq. 0d0) then
c We assume that for ekn > 50000, H2 content of the gas is negligible
        temp  = (2d0 / 3d0) * ekn
        gamma = 5d0 / 3d0
      else
        idx_en  = int(ekn)
        diff_en = ekn - int(ekn)
        if (abh2 .eq. 0.5d0) then
          t1 = temp_table(200000 + idx_en)
          t2 = temp_table(200000 + idx_en + 1)
          temp = t1 + (t2 - t1) * diff_en
          g1 = gamma_table(200000 + idx_en)
          g2 = gamma_table(200000 + idx_en + 1)
          gamma = g1 + (g2 - g1) * diff_en
        else
          idx_h2  = int(abh2 / 0.1)
          diff_h2 = (abh2 - 0.1 * idx_h2) / 0.1
          if (idx_h2 .eq. 0) then
            temp_1  = (2d0 / 3d0) * ekn
            gamma_1 = (5d0 / 3d0)
          else
            t1 = temp_table(50000*(idx_h2 - 1)  + idx_en)
            g1 = gamma_table(50000*(idx_h2 - 1) + idx_en)
            t2 = temp_table(50000*(idx_h2 - 1)  + idx_en + 1)
            g2 = gamma_table(50000*(idx_h2 - 1) + idx_en + 1)
            temp_1  = t1 + (t2 - t1) * diff_en
            gamma_1 = g1 + (g2 - g1) * diff_en
          endif
          t1 = temp_table(50000*idx_h2  + idx_en)
          g1 = gamma_table(50000*idx_h2 + idx_en)
          t2 = temp_table(50000*idx_h2  + idx_en + 1)
          g2 = gamma_table(50000*idx_h2 + idx_en + 1)
          temp_2  = t1 + (t2 - t1) * diff_en
          gamma_2 = g1 + (g2 - g1) * diff_en

          temp  = temp_1  + (temp_2  - temp_1)  * diff_h2
          gamma = gamma_1 + (gamma_2 - gamma_1) * diff_h2
        endif
      endif
#endif
      return
      end
c=======================================================================
c
c    \\\\\\\\\\        E N D    S U B R O U T I N E       //////////
c    //////////     L O O K U P _ T E M P _ G A M M A     \\\\\\\\\\
c
c=======================================================================
