c=======================================================================
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////           S P L I N E _ E V A L           \\\\\\\\\\
c
c=======================================================================
c
      subroutine spline_eval(nval, positions, values, nnew, 
     $                       new_positions, new_values)
#ifdef CHEMCOOL
c
c    written by: Simon Glover, January 2004
c
c  PURPOSE: Compute a spline fit to a set of nval data points.
c  Evaluate this fit at nnew specified locations and return the
c  nnew computed values.
c
      implicit NONE
#include "cool.h"
      integer nval, nnew
      REAL positions(nval), values(nval)
      REAL new_positions(nnew), new_values(nnew)
      REAL coefficients(4,nval-1)
      REAL dpt, pt, a, b, c, d
      integer I, J, index

      call spline_coefficients(nval, values, coefficients)

      do I = 1, nnew
        pt = new_positions(I)

c We're using -1 to indicate that we've been asked for a value outside of
c the range of our data, and setting the value for that point to zero;
c anything fancier should probably be handled in the caller.
        if (pt .lt. positions(1) .or. pt .gt. positions(nval)) then
          index         = -1
          new_values(I) = 0d0
        elseif (pt .eq. positions(nval)) then
c Another special case, but this one is easy to deal with
          index         = -1
          new_values(I) = values(nval)
        else
          index = 1
          do J = 1, nval-1
            if (pt .ge. positions(J) .and. pt .lt. positions(J+1)) then
              index = J
              dpt = (pt - positions(index)) / (positions(index+1) - 
     $               positions(index))
            endif
          enddo
        endif

        if (index .ne. -1) then
          a = coefficients(1,index)
          b = coefficients(2,index)
          c = coefficients(3,index)
          d = coefficients(4,index)
          new_values(I) = a + b * dpt + c * dpt**2 + d * dpt**3 
        endif

      enddo

#endif /* CHEMCOOL */      
      end
c
c=======================================================================
c
c    \\\\\\\\\\        E  N  D   S U B R O U T I N E      //////////
c    //////////           S P L I N E _ E V A L           \\\\\\\\\\
c
c=======================================================================


c=======================================================================
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////   S P L I N E _ C O E F F I C I E N T S   \\\\\\\\\\
c
c=======================================================================
c
      subroutine spline_coefficients(nval, values, coefficients)
#ifdef CHEMCOOL
c
c    written by: Simon Glover, January 2004
c
c  PURPOSE: Compute coefficients for a cubic spline fit to the set
c  of nval datapoints specified in values.
c
c  The solution follows the procedure of Bartels et al., 1998, "An 
c  Introduction to Splines for Use in Computer Graphics and Geometric 
c  Modelling", Morgan Kaufmann, ch. 3, pp 9-17, as outlined at 
c  http://mathworld.wolfram.com/CubicSpline.html
c
c  NB This routine does little or no error-checking, since it is only
c  intended to be used with a few small sets of known good data.
c
       implicit NONE
#include "cool.h"
      integer I, nval
      REAL values(nval), derivatives(nval), coefficients(4,nval-1)

c We first calculate the first derivative of the curve at each of
c our data points
      call spline_derivatives(nval, values, derivatives)

c The spline coefficients can then be expressed in terms of the derivatives
c and the values at the data points
      do I = 1, nval-1
        coefficients(1,I) = values(I)
        coefficients(2,I) = derivatives(I)
        coefficients(3,I) = 3d0 * (values(I+1) - values(I)) - 
     $                      2d0 * derivatives(I) - derivatives(I+1)
        coefficients(4,I) = 2d0 * (values(I) - values(I+1)) + 
     $                      derivatives(I) + derivatives(I+1)
      enddo

#endif /* CHEMCOOL */      
      end
c=======================================================================
c
c    \\\\\\\\\\        E N D     S U B R O U T I N E      //////////
c    //////////   S P L I N E _ C O E F F I C I E N T S   \\\\\\\\\\
c
c=======================================================================


c=======================================================================
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////    S P L I N E _ D E R I V A T I V E S    \\\\\\\\\\
c
c=======================================================================
      subroutine spline_derivatives(nval, values, derivatives)
#ifdef CHEMCOOL
c
c    written by: Simon Glover, January 2004
c
c  PURPOSE: Compute the derivates of the spline curve required by 
c  the spline_coefficient subroutine. For more details, see the
c  preamble to that subroutine.
c
       implicit NONE
#include "cool.h"
      integer I, J, nval
      REAL values(nval), derivatives(nval)
      REAL Cc(nval,nval+1)
      REAL f,sum

c The derivatives we require are given by the solution of the matrix equation
c Ax = B, where:
c
c      |2 1        |       |D_1  |           |3*(y_1 - y_0)  |
c      |1 4 1      |       |D_2  |           |3*(y_2 - y_0)  |
c  A = |  .....    |,  x = |...  |  and  B = |3*(y_3 - y_1)  |
c      |    .....  |       |...  |           | ............. |
c      |      1 4 1|       |D_n-1|           |3*(y_n - y_n-2)|
c      |        1 2|       |D_n  |           |3*(y_n - y_n-1)|
c
c To solve this equation, we first construct the 'augmented', n by n+1 matrix
c Cc = AB:

      do I = 1, nval
        do J = 1, nval+1
          Cc(I,J) = 0d0
        enddo
        if (I .eq. 1) then
          Cc(I,I)      = 2d0
          Cc(I,I+1)    = 1d0
          Cc(I,nval+1) = 3d0 * (values(2) - values(1))
        else if (I .eq. nval) then
          Cc(I,I-1)    = 1d0
          Cc(I,I)      = 2d0
          Cc(I,nval+1) = 3d0 * (values(nval) - values(nval-1))
        else
          Cc(I,I-1)    = 1d0
          Cc(I,I)      = 4d0
          Cc(I,I+1)    = 1d0
          Cc(I,nval+1) = 3d0 * (values(I+1) - values(I-1))
        endif
      enddo

c We then reduce this matrix to upper triangular form by Gaussian 
c elimination...
      do I = 2, nval
        f = Cc(I,I-1) / Cc(I-1,I-1)
        do J = 1, nval+1
          Cc(I,J) = Cc(I,J) - f * Cc(I-1,J)
        enddo
      enddo

c ...and finally solve for the derivatives by back substitution
      derivatives(nval) = Cc(nval,nval+1) / Cc(nval,nval)
 
      do I = nval-1, 1, -1
        sum = 0d0
        do J = I+1, nval
          sum = sum + Cc(I,J) * derivatives(J)
        enddo
	derivatives(I) = (Cc(I,nval+1) - sum) / Cc(I,I)
      enddo

#endif /* CHEMCOOL */      
      end
c=======================================================================
c
c    \\\\\\\\\\        E N D     S U B R O U T I N E      //////////
c    //////////    S P L I N E _ D E R I V A T I V E S    \\\\\\\\\\
c
c=======================================================================

