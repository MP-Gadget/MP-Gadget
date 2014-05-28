#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "proto.h"

/*! \file accel.c
 *  \brief driver routines to carry out force computation
 */


/*! This routine computes the accelerations for all active particles.  First, the gravitational forces are
 * computed. This also reconstructs the tree, if needed, otherwise the drift/kick operations have updated the
 * tree to make it fullu usable at the current time.
 *
 * If gas particles are presented, the `interior' of the local domain is determined. This region is guaranteed
 * to contain only particles local to the processor. This information will be used to reduce communication in
 * the hydro part.  The density for active SPH particles is computed next. If the number of neighbours should
 * be outside the allowed bounds, it will be readjusted by the function ensure_neighbours(), and for those
 * particle, the densities are recomputed accordingly. Finally, the hydrodynamical forces are added.
 */
void compute_accelerations(int mode)
{
#ifdef RADTRANSFER
    double timeeach = 0, timeall = 0, tstart = 0, tend = 0;
#endif
#if defined(BUBBLES) || defined(MULTI_BUBBLES)
    double hubble_a;
#endif
    int TreeReconstructFlag = 0;

    if(ThisTask == 0)
    {
        printf("Start force computation...\n");
        fflush(stdout);
    }

#ifdef REIONIZATION
    heating();
#endif

    walltime_measure("/Misc");

#ifdef PMGRID
    if(All.PM_Ti_endstep == All.Ti_Current)
    {
#ifdef PETA_PM
        petapm_prepare();
#endif
        force_treefree();
        TreeReconstructFlag = 1;
        long_range_force();

#ifdef PETA_PM
        petapm_finish();
#endif

        walltime_measure("/LongRange");

    }
#endif

    /* Check whether it is really time for a new domain decomposition */
    if(All.NumForcesSinceLastDomainDecomp >= All.TotNumPart * All.TreeDomainUpdateFrequency
            || All.DoDynamicUpdate == 0)
    {

        domain_Decomposition();	/* do domain decomposition */
        TreeReconstructFlag = 1;
    }

    if(TreeReconstructFlag) {
        force_treebuild_simple();
    }

#ifndef ONLY_PM
    gravity_tree();		/* computes gravity accel. */

    if(All.TypeOfOpeningCriterion == 1 && All.Ti_Current == 0)
        gravity_tree();		/* For the first timestep, we redo it
                             * to allow usage of relative opening
                             * criterion for consistent accuracy.
                             */
#endif


    if(All.Ti_Current == 0 && RestartFlag == 0 && header.flag_ic_info == FLAG_SECOND_ORDER_ICS)
        second_order_ics();		/* produces the actual ICs from the special second order IC file */


#ifdef FORCETEST
    gravity_forcetest();
#endif


    if(All.TotN_sph > 0)
    {
        /***** density *****/
        if(ThisTask == 0)
        {
            printf("Start density computation...\n");
            fflush(stdout);
        }

        density();		/* computes density, and pressure */

#if (defined(DIVBCLEANING_DEDNER) || defined(SMOOTH_ROTB) || defined(BSMOOTH) || defined(VECT_POTENTIAL))
        smoothed_values();
#endif


#if defined(SNIA_HEATING)
        snIa_heating();
#endif

        /***** update smoothing lengths in tree *****/
        force_update_hmax();

        /***** hydro forces *****/
        if(ThisTask == 0)
        {
            printf("Start hydro-force computation...\n");
            fflush(stdout);
        }

        hydro_force();		/* adds hydrodynamical accelerations  and computes du/dt  */

#ifdef CONDUCTION
        if(All.Conduction_Ti_endstep == All.Ti_Current)
            conduction();
#endif

#ifdef CR_DIFFUSION
        if(All.CR_Diffusion_Ti_endstep == All.Ti_Current)
            cosmic_ray_diffusion();
#endif

#ifdef RADTRANSFER
        if(Flag_FullStep)		/* only do it for full timesteps */
        {
            All.Radiation_Ti_endstep = All.Ti_Current;


            if(ThisTask == 0)
            {
                printf("Start Eddington tensor computation...\n");
                fflush(stdout);
            }

            eddington();

#ifdef RT_RAD_PRESSURE
            n();
#endif

            if(ThisTask == 0)
            {
                printf("done Eddington tensor! \n");
                fflush(stdout);
            }

#ifdef EDDINGTON_TENSOR_SFR
            density_sfr();
            sfr_lum();
#endif

#ifdef EDDINGTON_TENSOR_STARS
            rt_get_lum_stars();
            star_lum();
#endif

#ifdef EDDINGTON_TENSOR_GAS
            gas_lum();
#endif

#ifdef EDDINGTON_TENSOR_BH
            bh_lum();
#endif

            /***** evolve the transport of radiation *****/
            if(ThisTask == 0)
            {
                printf("start radtransfer...\n");
                fflush(stdout);
            }

            tstart = second();

            radtransfer();

            radtransfer_update_chemistry();

            tend = second();
            timeeach = timediff(tstart, tend);
            MPI_Allreduce(&timeeach, &timeall, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            if(ThisTask == 0)
            {
                printf("time consumed is %g \n", timeall);
                printf("done with radtransfer! \n");
                fflush(stdout);
            }

            All.Radiation_Ti_begstep = All.Radiation_Ti_endstep;
        }
#endif

#ifdef MHM
        /***** kinetic feedback *****/
        kinetic_feedback_mhm();
#endif


#ifdef BLACK_HOLES
        /***** black hole accretion and feedback *****/
        blackhole_accretion();
#ifdef FOF
        /* this will find new black hole seed halos */
        if(All.Time >= All.TimeNextBlackHoleCheck)
        {
            fof_fof(-1);

            if(All.ComovingIntegrationOn)
                All.TimeNextBlackHoleCheck *= All.TimeBetBlackHoleSearch;
            else
                All.TimeNextBlackHoleCheck += All.TimeBetBlackHoleSearch;
        }
#endif
#endif


#ifdef COOLING	/**** radiative cooling and star formation *****/

#ifdef SFR
        cooling_and_starformation();
#else
        cooling_only();
#endif

#endif /*ends COOLING */

#ifdef CHEMCOOL
        do_chemcool(-1, 0);
#endif

#ifndef BH_BUBBLES
#ifdef BUBBLES
        /**** bubble feedback *****/
        if(All.Time >= All.TimeOfNextBubble)
        {
#ifdef FOFs
            fof_fof(-1);
            bubble();
#else
            bubble();
#endif
            if(All.ComovingIntegrationOn)
            {
                hubble_a = hubble_function(All.Time);
                All.TimeOfNextBubble *= (1.0 + All.BubbleTimeInterval * hubble_a);
            }
            else
                All.TimeOfNextBubble += All.BubbleTimeInterval / All.UnitTime_in_Megayears;

            if(ThisTask == 0)
                printf("Time of the bubble generation: %g\n", 1. / All.TimeOfNextBubble - 1.);
        }
#endif
#endif

#if defined(MULTI_BUBBLES) && defined(FOF)
        if(All.Time >= All.TimeOfNextBubble)
        {
            fof_fof(-1);

            if(All.ComovingIntegrationOn)
            {
                hubble_a = hubble_func(All.Time);
                All.TimeOfNextBubble *= (1.0 + All.BubbleTimeInterval * hubble_a);
            }
            else
                All.TimeOfNextBubble += All.BubbleTimeInterval / All.UnitTime_in_Megayears;

            if(ThisTask == 0)
                printf("Time of the bubble generation: %g\n", 1. / All.TimeOfNextBubble - 1.);
        }
#endif

    }

    if(ThisTask == 0)
    {
        printf("force computation done.\n");
        fflush(stdout);
    }
}
