c=======================================================================
c
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////     E V O L V E _ A B U N D A N C E S     \\\\\\\\\\
c
c=======================================================================
c
c     Written by S. Glover (AMNH, 2004-2005, AIP 2006-2007)
c
#ifdef TEST
      subroutine evolve_abundances(t_start, t_end, dl, yn, divv, 
     $                             energy, non_eq_abundances, rpar)
#else
      subroutine evolve_abundances(time, dl, yn, divv, energy, 
     $                             non_eq_abundances
#ifdef RAYTRACE
     $                            , col_tot, col_H2
#ifdef CO_SHIELDING
     $                            , col_CO
#endif
     $                              ) 
#else /* RAYTRACE */
     $                            , column_est)
#endif /* RAYTRACE */
#endif /* TEST */
#ifdef CHEMCOOL
      implicit NONE
#include "cool.h"
#include "non_eq.h"
c
      REAL    time, dl, yn, divv, energy
      REAL    non_eq_abundances(num_non_eq_species)
      REAL    radius, column_est
      REAL    abe, abh2, yntot, ekn, temp, gamma
#ifdef RAYTRACE
      REAL    col_tot(6), col_H2(6), col_CO(6)
#endif
      REAL    t_start, t_end, dt_est, dt, t_in, t_out
      integer new_step
c
      integer nstep
      parameter (nstep = NSTEP)
c
      REAL safety(nspec)
      integer used_for_stepsize_control(nspec)
#if CHEMISTRYNETWORK == 1
      DATA safety /1d-1, 0d0, 0d0, 0d0, 0d0, 0d0,
     $             DTCOOL_SCALE_FACTOR/
      DATA used_for_stepsize_control /1, 0, 0, 0, 0, 0, 1/
#endif
#if CHEMISTRYNETWORK == 2
      DATA safety /1d-1, 1d-1, 0d0, 0d0, 0d0, 0d0, 1d-1, 1d-1, 0d0,
     $             0d0, DTCOOL_SCALE_FACTOR/
      DATA used_for_stepsize_control /1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1/
#endif
#if CHEMISTRYNETWORK == 3
      DATA safety /1d-1, 1d-1, 0d0, 0d0, 0d0, 0d0, 0d0, 0d0, 0d0,
     $             0d0, 0d0, 0d0, 0d0, 0d0, 0d0, 0d0, 0d0, 0d0,
     $             0d0, DTCOOL_SCALE_FACTOR/
      DATA used_for_stepsize_control /1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
     $                                0, 0, 0, 0, 0, 0, 0, 0, 0, 1/
#endif
#if CHEMISTRYNETWORK == 4
      DATA safety /1d-1, 1d-1, DTCOOL_SCALE_FACTOR/
      DATA used_for_stepsize_control /1, 1, 1/
#endif
#if CHEMISTRYNETWORK == 5
      DATA safety /1d-1, 1d-1, 1d-1, DTCOOL_SCALE_FACTOR/
      DATA used_for_stepsize_control /1, 1, 1, 1/
#endif
#if CHEMISTRYNETWORK == 6
      DATA safety /1d-1, 1d-1, 1d-1, 1d-1, DTCOOL_SCALE_FACTOR/
      DATA used_for_stepsize_control /1, 1, 1, 1, 1/
#endif
#if CHEMISTRYNETWORK == 7
      DATA safety /1d-1, 1d-1, 0d0, 0d0, 0d0, 0d0, 0d0, 0d0, 0d0,
     $             0d0, 0d0, 0d0, 0d0, 0d0,
     $             DTCOOL_SCALE_FACTOR/
      DATA used_for_stepsize_control /1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
     $                                0, 0, 0, 0, 1/
#endif
#if CHEMISTRYNETWORK == 8
      DATA safety /1d-1, 1d-1, 1d-1, DTCOOL_SCALE_FACTOR/
      DATA used_for_stepsize_control /1, 0, 1, 1/
#endif
#if CHEMISTRYNETWORK == 11
      DATA safety /1d-1, 1d-1, 0d0, 0d0, 0d0, 0d0, 0d0, 0d0, 0d0,
     $             0d0, 0d0, 0d0, 0d0, 0d0, 0d0, 0d0, 0d0,
     $             DTCOOL_SCALE_FACTOR/
      DATA used_for_stepsize_control /1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
     $                                0, 0, 0, 0, 0, 0, 0, 1/
#endif
c
      integer I, J
c
c DVODE control variables -- NB atol declared in non_eq.h
c
#ifdef ROSENBROCK
      REAL    rtol(nspec)
#else
      REAL    rtol
#endif
      integer itask, iopt, itol, mf, istate, iflag_dvi
c
c Other DVODE variables
c
      integer lrw, liw
      parameter (lrw   = 1000)
      parameter (liw   = 1000)
      REAL    rpar(nrpar), rwork(lrw)
      integer ipar(nipar), iwork(liw)
      REAL    y(nspec), ydot(nspec), y_old(nspec)
c
#ifdef ROSENBROCK
      integer ifcn, ijac, mljac, mujac, idfx, imas, mlmas, mumas,
     $        iout, idid
      external dfx, solout, mas
#endif

      external jac, rate_eq
c
c Sanity check for our input time
c
#ifdef TEST
      time = t_start
#endif
      if (time .le. 0d0) then
#ifdef DEBUG_EVOLVE
        if (time .lt. -1d0) then
          print*, "Warning: negative timestep in evolve_abundances",
     $             time
        endif
#endif
        return
      endif        
c
c Initialize abundance vector, treating the internal energy as 
c an additional abundance. 
c
c N.B. We assume that abundances are stored consecutively, with
c no gaps, and that itmp == nspec (i.e. that the energy is the last
c value stored)
c
      do I = 1, nspec-1
        y(I) = non_eq_abundances(I)
      enddo
      y(itmp)  = energy
c
c Setup initial parameters for DVODE integrator. These don't vary over 
c the course of the timestep.
c
c Optional input supplied -- see iwork, rwork init. below:
c
      iopt = 1
c
c Evolve ODEs from t -> tout, allowing overshooting & interpolation
c [See DVODE documentation for more details].
c
      itask = 1
c
#ifdef ROSENBROCK
      itol = 1  ! rtol, atol are vectors
#else
c
c Use same relative tolerance for each species, different absolute tolerances
c (since magnitudes may vary by many orders of magnitude). 
c
      itol = 2
#endif
c
c Set relative tolerance -- for simplicity we keep this fixed at 0.1% for now  
c
#ifdef ROSENBROCK
      do I = 1, nspec
       rtol(I) = RTOL
      enddo
#else
      rtol = RTOL
#endif
c
c ODE integration method: 22 => BDF with an internally generated Jacobian
c
      mf = 22
c
c Start DVODE in start-up mode (which forces it to initialize all of its
c internal arrays). Subsequent calls within the loop below use istate = 2
c
      istate = 1
c
c Optional input -- increase maximum number of steps that DVODE can take 
c internally without returning an to 20000, as 500 doesn't appear to be 
c enough for portions of our parameter space
c
      do I = 5, 10      
        rwork(I) = 0d0
        iwork(I) = 0
      enddo
c      
      iwork(6) = 200000
c
c At high n, we might want to switch off the deuterium chemistry
c
#if CHEMISTRYNETWORK == 1 || CHEMISTRYNETWORK == 8
      abe = y(ihp)
#if CHEMISTRYNETWORK == 1
      abe  = abe + y(idp) + y(ihep) + 2d0 * y(ihepp)
#endif
      abh2 = y(ih2)
      yntot = (1d0 + abhe + abe - abh2) * yn
      ekn = energy / (kboltz * yntot)
      temp  = (2d0 / 3d0) * ekn
      gamma = 5d0 / 3d0
c     call lookup_temp_gamma(abh2, ekn, temp, gamma)
c      if (yn .gt. 1d9 .or. (yn .gt. 1d8 .and. temp .gt. 5d2)) then
c        no_dchem = 1
c#if CHEMISTRYNETWORK == 1
c        y(idp) = 0d0
c        y(ihd) = min(abundD, abundD * exp(4.88d2 / temp) * abh2)
c#endif
c      else
c        no_dchem = 0
c      endif
#endif
c
c Parameters to pass to rate_eq
c
      rpar(1) = yn
      rpar(2) = dl
      rpar(3) = divv
#ifdef RAYTRACE
      do I = 1, 6
        rpar(I+3) = col_tot(I)
        rpar(I+9) = col_H2(I)
#ifdef CO_SHIELDING
        rpar(I+15) = col_CO(I)
#endif
      enddo
#else  
c
c Not using the Raytracing, so can put the values needed for
c iphoto = 4 option into this array... note that do_chemcool_step()
c puts everything into cgs units, before it gets passed through to here.
c It then gets passed on from rate_eq* to calc_photo, where it finally gets
c unpacked/used. Note that dl contains either the radius or the smoothing
c by this point...
c
      rpar(4) = column_est
#endif
c
      ipar(1) = 0
c
#ifndef TEST
      t_start  = 0d0
      t_end    = time
#endif
      new_step = 1
c
      do I = 1, nstep
c
c If we're starting a new step, then figure out a reasonable timestep to
c pass to DVODE -- we need to do this here to avoid DVODE producing -ve
c abundances or -ve temperatures, which can occur if it starts with too
c large a timestep.
c
c If we're not starting a new step, then dt has already been set. 
c
        if (new_step .eq. 1) then
c
c First, save a copy of the current abundances
c
          do J = 1, nspec
            y_old(J) = y(J)
          enddo
c
c Now compute ydot. On the first step, we must use rate_eq directly,
c but on subsequent steps, we can just call dvindy [if we're using
c the dvode integrator]
c
#ifndef ROSENBROCK
          if (I .eq. 1) then
            call rate_eq(nspec, t_start, y, ydot, rpar, ipar)
          else
            call dvindy(t_start, 1, rwork(21), nspec, ydot, iflag_dvi)
          endif
#else
          call rate_eq(nspec, t_start, y, ydot, rpar, ipar)
#endif
c
c Init. dt_est to an unrealistic value
c
          dt_est = 1d18

          do J = 1, nspec
            if (used_for_stepsize_control(J) .eq. 1) then
              if (ydot(J)   .ne. 0d0 .and.
     $            safety(J) .ne. 0d0 .and.
     $            y(J)      .gt. atol(J)) then
                dt_est = min(dt_est, abs(safety(J) * y(J) / ydot(J)))
              endif
            endif
          enddo
c
          ipar(2) = 0 
c
c Don't evolve past the end of the global timestep
c
          dt = min(dt_est, t_end - t_start)
        endif
c
        t_in  = t_start
        t_out = t_start + dt
        if (dt .lt. 0d0) then
          print*, 'Negative timestep!'
          ABORT(1)
        endif
        if (dt .eq. 0d0) then 
          if (t_end .eq. t_start) then
            return
          else
            print*, 'Zero timestep!'
            ABORT(2)
          endif
        endif
c
#ifdef ROSENBROCK
        ifcn = 1  ! differential terms may depend explicity on t
        ijac = 1  ! Jacobian is computed internally
        mljac = nspec
        mujac = 0
        idfx = 0  ! Partial derivatives computed internally 
        imas = 0
        mlmas = nspec
        mumas = 0
        iout  = 0  ! solout routine never called
        istate = 0

        call RODAS(nspec, rate_eq, ifcn, t_in, y, t_out, 1d-2*dt, rtol, 
     $             atol, itol, jac, ijac, mljac, mujac, dfx, idfx, mas,
     $             imas, mlmas, mumas, solout, iout, rwork, lrw, iwork,
     $             liw, rpar, ipar, idid)
        if (idid .eq. 1) then
          istate = 2
        endif
#else
        call dvode(rate_eq, nspec, y, t_in, t_out, itol, rtol, atol, 
     $             itask, istate, iopt, rwork, lrw, iwork, liw, jac, 
     $             mf, rpar, ipar)
#endif
c
c Ensure that dvode has produced sane output values -- if not, we
c need to shrink the timestep. Note that if ipar(1) is non-zero, 
c or if istate != 2 (indicating an integrator error), we're going
c to shrink the timestep anyway, so we can skip the check.
c
        if (ipar(1) .eq. 0 .and. istate .eq. 2) then
          call check_abundances(y, ipar)
        endif
c
        if ((ipar(1) .ne. 0) .or. (istate .ne. 2)) then 
#ifdef DEBUG_EVOLVE
         if (istate .ne. 2) then
           print*, "Error: istate = ", istate
         endif
#endif
          do J = 1, nspec
            y(J) = y_old(J)
          enddo
          dt       = dt / 2d0
          if (dt .lt. 1d-8) then
            print*, "Excessively small timestep! ", dt, dt_est, 
     $              t_start, t_end
            print*, "ID: ", id_current
            print*, yn, dl, divv
            print*, y
	    print*, ydot
#ifndef NOSTOP_WHEN_BELOW_MINTIMESTEP
            ABORT(100)
#endif
          endif
          new_step = 0
          ipar(1)  = 0
          istate   = 1
        else
          t_start = t_start + dt
          new_step = 1
        endif
c
        if (t_start .ge. t_end) then
          goto 100
        endif
      enddo
c
      write(*,'(A)') 'EVOLVE_ABUNDANCES: nstep exceeded,' 
      print*, id_current
      print*, dt
      print*, rpar
      print*, y
      print*, ydot
      ABORT(0)
      return

100   continue

      do I = 1, nspec-1
        non_eq_abundances(I) = y(I)
      enddo
      energy = y(itmp)
c
#endif /* CHEMCOOL */
      return      
      end
c=======================================================================
c
c
c    \\\\\\\\\\        E N D     S U B R O U T I N E      //////////
c    //////////     E V O L V E _ A B U N D A N C E S     \\\\\\\\\\
c
c=======================================================================
c
c=======================================================================
c
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////      C H E C K _ A B U N D A N C E S    \\\\\\\\\\
c
c=======================================================================
c
      subroutine check_abundances(y, ipar)
#ifdef CHEMCOOL
      implicit NONE
#include "cool.h"
#include "non_eq.h"
      REAL y(nspec), max_abundance(nspec)
      REAL diff
      integer ipar(nipar)
      integer J
c
      if (nchem_network .le. 3) then
        max_abundance(ih2)   = 0.5d0
        max_abundance(ihp)   = 1.0d0
        max_abundance(idp)   = abundD
        max_abundance(ihd)   = abundD
        max_abundance(ihep)  = abhe
        max_abundance(ihepp) = abhe
      endif
c
      if (nchem_network .eq. 2 .or. nchem_network .eq. 3) then
        max_abundance(io)    = abundo
        max_abundance(ic)    = abundc
        max_abundance(isi)   = abundsi
        max_abundance(isipp) = abundsi
      endif
c
      if (nchem_network .eq. 3) then
        max_abundance(ico)    = min(abundo, abundc)
        max_abundance(ic2)    = 0.5d0 * abundc
        max_abundance(ioh)    = abundo
        max_abundance(ih2o)   = abundo
        max_abundance(io2)    = 0.5d0 * abundo 
        max_abundance(ihcop)  = min(abundo, abundc)
        max_abundance(ich)    = abundc
        max_abundance(ich2)   = abundc
        max_abundance(ich3p)  = abundc
      endif
c
      if (nchem_network .eq. 4) then
        max_abundance(ih2)   = 0.5d0
        max_abundance(ihp)   = 1.0d0
      endif
c
      if (nchem_network .eq. 5) then
        max_abundance(ih2)   = 0.5d0
        max_abundance(ihp)   = 1.0d0
        max_abundance(ico)   = min(abundc, abundo)
      endif
c
      if (nchem_network .eq. 6) then
        max_abundance(ih2)   = 0.5d0
        max_abundance(ihp)   = 1.0d0
        max_abundance(ico)   = min(abundc, abundo)
        max_abundance(icos)  = min(abundc, abundo)
      endif
c
      if (nchem_network .eq. 7 .or. nchem_network .eq. 11) then
        max_abundance(ih2)    = 0.5d0
        max_abundance(ihp)    = 1.0d0
        max_abundance(ihep)   = abhe
        max_abundance(io)     = abundo
        max_abundance(ic)     = abundc
        max_abundance(ico)    = min(abundo, abundc)
        max_abundance(ic2)    = 0.5d0 * abundc
        max_abundance(ioh)    = abundo
        max_abundance(ih2o)   = abundo
        max_abundance(io2)    = 0.5d0 * abundo 
        max_abundance(ihcop)  = min(abundo, abundc)
        max_abundance(ich)    = abundc
        max_abundance(ich2)   = abundc
        max_abundance(ich3p)  = abundc
        if (nchem_network .eq. 11) then
          max_abundance(icos) = min(abundc, abundo)
          max_abundance(ih2os) = abundo
          max_abundance(io2s) = 0.5d0 * abundo
        endif
      endif
c
      if (nchem_network .eq. 8) then
        max_abundance(ih2) = 0.5d0
        max_abundance(ihd) = abundD  
        max_abundance(ihp) = 1d0
      endif
c
c Check for NaNs
c
      do J = 1, nspec
        if (y(J) .ne. y(J)) then
#ifdef DEBUG_EVOLVE
          print*, "NaN for ID ", id_current, y
#endif
          ipar(1) = 1
          return
        endif
      enddo
c
c Check for negative abundances. Small negative values (which have
c absolute values less than the corresponding absolute tolerance)
c are accepted, but are clipped to zero. For large negative values,
c we repeat the step with a smaller dt.
c
      do J = 1, nspec
        if (y(J) .lt. 0d0) then
          if (y(J) .lt. -atol(J)) then
#ifdef DEBUG_EVOLVE
            print*, "Negative abundance for ID ", id_current, y
#endif
            ipar(1) = 1
            return
          else
            y(J) = 0d0
          endif
        endif
      enddo
c
c Don't allow ion abundances to exceed corresponding total metal
c abundances. Again, small errors are silently corrected, large
c errors lead to us repeating with a smaller timestep
c
      do J = 1, nspec
        if (J .ne. itmp .and. y(J) .gt. max_abundance(J)) then
          diff = y(J) - max_abundance(J)
          if (diff .lt. RTOL * y(J)) then
            y(J) = max_abundance(J)
          else
#ifdef DEBUG_EVOLVE
            print*, "Excess abundance for ID ", id_current, y
#endif
            ipar(1) = 1
            return
          endif
        endif
      enddo
c
#endif /* CHEMCOOL */
      return
      end
c=======================================================================
c
c
c    \\\\\\\\\\        E N D   S U B R O U T I N E      //////////
c    //////////      C H E C K _ A B U N D A N C E S    \\\\\\\\\\
c
c=======================================================================
c=======================================================================
c
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////       I N I T _ T O L E R A N C E S       \\\\\\\\\\
c
c=======================================================================
c
      subroutine init_tolerances
#ifdef CHEMCOOL
      implicit NONE
#include "non_eq.h"
#include "cool.h"
c
c Set-up absolute tolerances
c
      if (nchem_network .le. 3) then
        atol(ih2)   = ATOL_H2
        atol(ihp)   = ATOL_HP
        atol(idp)   = ATOL_DP
        atol(ihd)   = ATOL_HD
        atol(ihep)  = ATOL_HEP       
        atol(ihepp) = ATOL_HEPP
      endif
c
      if (nchem_network .eq. 2 .or. nchem_network .eq. 3) then
        atol(ic)    = ATOL_C
        atol(isi)   = ATOL_SI
        atol(io)    = ATOL_O
        atol(isipp) = ATOL_SIPP
      endif
c
      if (nchem_network .eq. 3) then
        atol(ico)   = ATOL_CO
        atol(ic2)   = ATOL_C2
        atol(ioh)   = ATOL_OH
        atol(ih2o)  = ATOL_H2O
        atol(io2)   = ATOL_O2
        atol(ihcop) = ATOL_HCOP
        atol(ich)   = ATOL_CH
        atol(ich2)  = ATOL_CH2
        atol(ich3p) = ATOL_CH3P
      endif
c
      if (nchem_network .eq. 4) then
        atol(ihp) = ATOL_HP
        atol(ih2) = ATOL_H2
      endif
c
      if (nchem_network .eq. 5) then
        atol(ihp) = ATOL_HP
        atol(ih2) = ATOL_H2
        atol(ico) = ATOL_CO
      endif
c
      if (nchem_network .eq. 6) then
        atol(ihp)  = ATOL_HP
        atol(ih2)  = ATOL_H2
        atol(ico)  = ATOL_CO
        atol(icos) = ATOL_COS
      endif
c
      if (nchem_network .eq. 7 .or. nchem_network .eq. 11) then
        atol(ih2)   = ATOL_H2
        atol(ihp)   = ATOL_HP
        atol(ihep)  = ATOL_HEP       
        atol(ic)    = ATOL_C
        atol(io)    = ATOL_O
        atol(ico)   = ATOL_CO
        atol(ic2)   = ATOL_C2
        atol(ioh)   = ATOL_OH
        atol(ih2o)  = ATOL_H2O
        atol(io2)   = ATOL_O2
        atol(ihcop) = ATOL_HCOP
        atol(ich)   = ATOL_CH
        atol(ich2)  = ATOL_CH2
        atol(ich3p) = ATOL_CH3P
        if (nchem_network .eq. 11) then
          atol(icos)  = ATOL_COS
          atol(ih2os) = ATOL_H2OS
          atol(io2s)  = ATOL_O2S
        endif
      endif
c
      if (nchem_network .eq. 8) then
        atol(ih2) = ATOL_H2
        atol(ihd) = ATOL_HD
        atol(ihp) = ATOL_HP
      endif
c
      atol(itmp)  = ATOL_TMP
c
#endif /* CHEMCOOL */
      return
      end
c=======================================================================
c
c
c    \\\\\\\\\\        E N D     S U B R O U T I N E      //////////
c    //////////       I N I T _ T O L E R A N C E S       \\\\\\\\\\
c
c=======================================================================
