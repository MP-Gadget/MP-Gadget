#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "proto.h"
#include "forcetree.h"
#include "fof.h"
#include "endrun.h"

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
    int TreeReconstructFlag = 0;

    message(0, "Start force computation...\n");

    walltime_measure("/Misc");

    if(All.PM_Ti_endstep == All.Ti_Current)
    {
        long_range_force();
        TreeReconstructFlag = 1;
        walltime_measure("/LongRange");
    }

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


    if(All.TotN_sph > 0)
    {
        /***** density *****/
        message(0, "Start density computation...\n");

        density();		/* computes density, and pressure */

        /***** update smoothing lengths in tree *****/
        force_update_hmax();

        /***** hydro forces *****/
        message(0, "Start hydro-force computation...\n");

        hydro_force();		/* adds hydrodynamical accelerations  and computes du/dt  */

#ifdef BLACK_HOLES
        /***** black hole accretion and feedback *****/
        blackhole_accretion();
#endif
#ifdef FOF
        /* this will find new black hole seed halos */
        if(All.Time >= All.TimeNextSeedingCheck)
        {
            fof_fof(-1);

            All.TimeNextSeedingCheck *= All.TimeBetweenSeedingSearch;
        }
#endif


#ifdef COOLING	/**** radiative cooling and star formation *****/

#ifdef SFR
        cooling_and_starformation();
#else
        cooling_only();
#endif

#endif /*ends COOLING */

    }
    message(0, "force computation done.\n");
}
