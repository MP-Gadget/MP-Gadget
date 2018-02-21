#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "utils.h"

#include "allvars.h"
#include "cooling.h"
#include "densitykernel.h"
#include "treewalk.h"
#include "timestep.h"

#include "gravshort.h"

static void
grav_short_pair_ngbiter(
        TreeWalkQueryGravShort * I,
        TreeWalkResultGravShort * O,
        TreeWalkNgbIterGravShort * iter,
        LocalTreeWalk * lv);

void grav_short_pair(void)
{
    TreeWalk tw[1] = {0};

    message(0, "Starting pair-wise short range gravity...\n");

    tw->ev_label = "GRAV_SHORT";
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterGravShort);
    tw->ngbiter = (TreeWalkNgbIterFunction) grav_short_pair_ngbiter;

    tw->haswork = grav_short_haswork;
    tw->fill = (TreeWalkFillQueryFunction) grav_short_copy;
    tw->reduce = (TreeWalkReduceResultFunction) grav_short_reduce;
    tw->postprocess = (TreeWalkProcessFunction) grav_short_postprocess;
    tw->UseNodeList = 1;
    tw->query_type_elsize = sizeof(TreeWalkQueryGravShort);
    tw->result_type_elsize = sizeof(TreeWalkResultGravShort);

    walltime_measure("/Misc");

    treewalk_run(tw, ActiveParticle, NumActiveParticle);

    walltime_measure("/Grav/Short");
}


static void
grav_short_pair_ngbiter(
        TreeWalkQueryGravShort * I,
        TreeWalkResultGravShort * O,
        TreeWalkNgbIterGravShort * iter,
        LocalTreeWalk * lv)
{
    if(iter->base.other == -1) {
        iter->base.Hsml = RCUT * All.Asmth * All.BoxSize / All.Nmesh;
        iter->base.mask = 0xff; /* all particles */
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        return;
    }

    int other = iter->base.other;
    double r = iter->base.r;
    double r2 = iter->base.r2;
    double * dist = iter->base.dist;

    if(P[other].Mass == 0) {
        endrun(-1, "Encoutered zero mass particle during density;"
                  " We haven't implemented tracer particles and this shall not happen\n");
    }

    double mass = P[other].Mass;

    double h = I->Soft;
    double otherh = FORCE_SOFTENING(other);
    if (otherh > h) h = otherh;

    double fac, pot;

    if(r >= h) {
        fac = mass / (r2 * r);
        pot = -mass / r;
    } else {
        double h_inv = 1.0 / h;
        double h3_inv = h_inv * h_inv * h_inv;
        double u = r * h_inv;
        double wp;
        if(u < 0.5)
            fac = mass * h3_inv * (10.666666666667 + u * u * (32.0 * u - 38.4));
        else
            fac =
                mass * h3_inv * (21.333333333333 - 48.0 * u +
                        38.4 * u * u - 10.666666666667 * u * u * u - 0.066666666667 / (u * u * u));
        if(u < 0.5)
            wp = -2.8 + u * u * (5.333333333333 + u * u * (6.4 * u - 9.6));
        else
            wp =
                -3.2 + 0.066666666667 / u + u * u * (10.666666666667 +
                        u * (-16.0 + u * (9.6 - 2.133333333333 * u)));

        pot = mass * h_inv * wp;
    }

    if (grav_apply_short_range_window(r, &fac, &pot) == 0) {
        int d;
        for(d = 0; d < 3; d ++)
            O->Acc[d] += - dist[d] * fac;

        O->Potential += pot;
        O->Ninteractions ++;
    }
}
