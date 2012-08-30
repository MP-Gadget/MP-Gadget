c=======================================================================
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////        V A L I D A T E _ I N P U T        \\\\\\\\\\
c
c=======================================================================
c
      subroutine validate_input(y, ipar)
#ifdef CHEMCOOL
      implicit NONE
#include "cool.h"
#include "non_eq.h"
      REAL y(nspec)
c
      integer ipar(nipar)
c
      integer I
c
c Bounds checking -- chemical species:
c
      do I = 1, nspec
c
c This test will fail only if y(I) is NaN (not a number); if this happens, 
c we abort...
c
        if (y(I) .ne. y(I)) then
          print*, 'Species', I, ' is NaN. Aborting...'
          print*, y
          print*, 'Particle ID', id_current
          ABORT(1)
        endif
c
        if (y(I) .lt. 0d0) then
#if 0
          if (y(I) .lt. -atol(I)) then
            if (I .eq. idp .and. abs(y(I)) .lt. 1d-12) then
              y(I) = 0d0
            else
              ipar(1) = 1
#ifdef DEBUG_EVOLVE
              print*, "Negative abundance", I, y(I)
              print*, y
              print*, 'ID', id_current
#endif
            endif
          endif
#endif
          y(I) = 0d0
        endif
c
        if (I .ne. itmp .and. I .ne. ih2) then
          if (y(I) .gt. 1d0) then
            if (y(I) .gt. 1d0 + eps_max) then
              ipar(1) = 1
#ifdef DEBUG_EVOLVE
            print*, "Excess abundance", I, y(I)
#endif
            endif
            y(I) = 1d0
          endif
        elseif (I .eq. ih2) then
          if (y(I) .gt. 0.5d0) then
            if (y(I) .gt. 0.5d0 + eps_max) then
              ipar(1) = 1
#ifdef DEBUG_EVOLVE
            print*, "Excess abundance", I, y(I)
#endif
            endif
            y(I) = 0.5d0
          endif
        endif
      enddo
c
#endif /* CHEMCOOL */
      return
      end
c=======================================================================
c
c    \\\\\\\\\\        E N D     S U B R O U T I N E      //////////
c    //////////         V A L I D A T E _ I N P U T       \\\\\\\\\\
c
c=======================================================================
c
c=======================================================================
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////       V A L I D A T E _ O U T P U T       \\\\\\\\\\
c
c=======================================================================
c
      subroutine validate_output(y, ydot, rpar)
#ifdef CHEMCOOL
      implicit NONE
#include "cool.h"
#include "non_eq.h"
      REAL y(nspec), ydot(nspec), rpar(nrpar)
      integer I

      do I = 1, nspec
        if (ydot(I) .ne. ydot(I)) then
          print*, 'ydot for species ', I, ' is NaN...'
          print*, 'ID ', id_current
          print*, ydot
          print*, y
          print*, rpar
          ABORT(1)
        endif
      enddo

#endif /* CHEMCOOL */
      return
      end
c=======================================================================
c
c    \\\\\\\\\\        E N D     S U B R O U T I N E      //////////
c    //////////        V A L I D A T E _ O U TP U T       \\\\\\\\\\
c
c=======================================================================
