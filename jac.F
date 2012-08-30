#ifdef ROSENBROCK
      SUBROUTINE JAC (NEQ, T, Y, DFY, LDFY, RPAR, IPAR)
#else
      SUBROUTINE JAC (NEQ, T, Y, ML, MU, PD, NROWPD, RPAR, IPAR)
#endif
      IMPLICIT NONE
#include "cool.h"
      integer NEQ, IPAR
      REAL T, Y(NEQ), RPAR
#ifdef ROSENBROCK
      integer LDFY
      REAL DFY(LDFY,NEQ)
#else
      integer ML, MU, NROWPD
      REAL PD(NROWPD,NEQ)
#endif
      return
      end
c Other dummy subroutines
#ifdef ROSENBROCK
      SUBROUTINE DFX(N,X,Y,FX,RPAR,IPAR)
      IMPLICIT NONE
#include "cool.h"
      integer IPAR, N
      REAL X,Y(N),FX(N), RPAR
      return
      end

      SUBROUTINE MAS(N,AM,LMAS,RPAR,IPAR)
      IMPLICIT NONE
#include "cool.h"
      integer IPAR, LMAS, N
      REAL AM(LMAS,N), RPAR
      return
      end

      SUBROUTINE SOLOUT(NR,XOLD,X,Y,CONT,LRC,N,
     $                  RPAR,IPAR,IRTRN)
      IMPLICIT NONE
#include "cool.h"
      integer NR, LRC, N, IPAR, IRTRN
      REAL X,XOLD,Y(N),CONT(LRC), RPAR
      return
      end
#endif
