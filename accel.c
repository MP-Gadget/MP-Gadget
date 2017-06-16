#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "proto.h"
#include "forcetree.h"
#include "domain.h"
#include "fof.h"
#include "endrun.h"
#include "timestep.h"

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
    message(0, "Start force computation...\n");

    walltime_measure("/Misc");

    /* Before computing forces, do a full domain decomposition
     * and rebuild the tree, so we have synced all particles
     * to current positions, and the tree nodes have the same positions.
     * Future drifts are now null-ops.*/
    domain_Decomposition();
    force_tree_rebuild();
    if(All.PM_Ti_endstep == All.Ti_Current)
    {
        long_range_force();
        walltime_measure("/LongRange");
        /* This is needed because of the call to force_tree_free inside gravpm_force.
         * That in turn is needed because the PM code may move particles to different regions.*/;
        force_tree_rebuild();
        /* compute and output energy statistics if desired. */
        if(All.OutputEnergyDebug)
            energy_statistics();
        /*Update the displacement timestep*/
        All.MaxTimeStepDisplacement = find_dt_displacement_constraint();
    }

    grav_short_tree();		/* computes gravity accel. */

    if(All.TypeOfOpeningCriterion == 1 && All.Ti_Current == 0)
        grav_short_tree();		/* For the first timestep, we redo it
                             * to allow usage of relative opening
                             * criterion for consistent accuracy.
                             */


    if(NTotal[0] > 0)
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
        blackhole();
#endif

/**** radiative cooling and star formation *****/
#ifdef SFR
        cooling_and_starformation();
#else
        cooling_only();
#endif

    }
    message(0, "force computation done.\n");
}
