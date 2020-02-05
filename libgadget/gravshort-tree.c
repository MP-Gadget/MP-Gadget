#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ipc.h>
#include <sys/sem.h>

#include "utils.h"

#include "forcetree.h"
#include "treewalk.h"
#include "timestep.h"
#include "gravshort.h"
#include "walltime.h"

/*! \file gravtree.c
 *  \brief main driver routines for gravitational (short-range) force computation
 *
 *  This file contains the code for the gravitational force computation by
 *  means of the tree algorithm. To this end, a tree force is computed for all
 *  active local particles, and particles are exported to other processors if
 *  needed, where they can receive additional force contributions. If the
 *  TreePM algorithm is enabled, the force computed will only be the
 *  short-range part.
 */

static struct gravshort_tree_params TreeParams;

/*This is a helper for the tests*/
void set_gravshort_treepar(struct gravshort_tree_params tree_params)
{
    TreeParams = tree_params;
}

struct gravshort_tree_params get_gravshort_treepar(void)
{
    return TreeParams;
}

/* Sets up the module*/
void
set_gravshort_tree_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        TreeParams.BHOpeningAngle = param_get_double(ps, "BHOpeningAngle");
        TreeParams.ErrTolForceAcc = param_get_double(ps, "ErrTolForceAcc");
        TreeParams.BHOpeningAngle = param_get_double(ps, "BHOpeningAngle");
        TreeParams.TreeUseBH= param_get_int(ps, "TreeUseBH");
        TreeParams.Rcut = param_get_double(ps, "TreeRcut");
    }
    MPI_Bcast(&TreeParams, sizeof(struct gravshort_tree_params), MPI_BYTE, 0, MPI_COMM_WORLD);
}

/* According to upstream P-GADGET3
 * correct workcount slows it down and yields little benefits in load balancing
 *
 * YF: anything we shall do about this?
 * */

int
force_treeev_shortrange(TreeWalkQueryGravShort * input,
        TreeWalkResultGravShort * output,
        LocalTreeWalk * lv);


/*! This function computes the gravitational forces for all active particles.
 *  If needed, a new tree is constructed, otherwise the dynamically updated
 *  tree is used.  Particles are only exported to other processors when really
 *  needed, thereby allowing a good use of the communication buffer.
 *  NeutrinoTracer = All.HybridNeutrinosOn && (All.Time <= All.HybridNuPartTime);
 *  rho0 = All.CP.Omega0 * 3 * All.CP.Hubble * All.CP.Hubble / (8 * M_PI * All.G)
 */
void
grav_short_tree(const ActiveParticles * act, PetaPM * pm, ForceTree * tree, double rho0, int NeutrinoTracer, int FastParticleType)
{
    double timeall = 0;
    double timetree, timewait, timecomm;

    TreeWalk tw[1] = {{0}};
    struct GravShortPriv priv;
    priv.cellsize = tree->BoxSize / pm->Nmesh;
    priv.Rcut = TreeParams.Rcut * pm->Asmth * priv.cellsize;;
    priv.ErrTolForceAcc = TreeParams.ErrTolForceAcc;
    priv.TreeUseBH = TreeParams.TreeUseBH;
    priv.BHOpeningAngle = TreeParams.BHOpeningAngle;
    priv.FastParticleType = FastParticleType;
    priv.NeutrinoTracer = NeutrinoTracer;
    priv.G = pm->G;
    priv.cbrtrho0 = pow(rho0, 1.0 / 3);

    tw->ev_label = "FORCETREE_SHORTRANGE";
    tw->visit = (TreeWalkVisitFunction) force_treeev_shortrange;
    /* gravity applies to all particles. Including Tracer particles to enhance numerical stability. */
    tw->haswork = NULL;
    tw->reduce = (TreeWalkReduceResultFunction) grav_short_reduce;
    tw->postprocess = (TreeWalkProcessFunction) grav_short_postprocess;

    tw->query_type_elsize = sizeof(TreeWalkQueryGravShort);
    tw->result_type_elsize = sizeof(TreeWalkResultGravShort);
    tw->fill = (TreeWalkFillQueryFunction) grav_short_copy;
    tw->tree = tree;
    tw->priv = &priv;

    walltime_measure("/Misc");

    /* allocate buffers to arrange communication */
    MPIU_Barrier(MPI_COMM_WORLD);
    message(0, "Begin tree force.  (presently allocated=%g MB)\n", mymalloc_usedbytes() / (1024.0 * 1024.0));

    walltime_measure("/Misc");

    treewalk_run(tw, act->ActiveParticle, act->NumActiveParticle);

    /* now add things for comoving integration */

    MPIU_Barrier(MPI_COMM_WORLD);
    message(0, "Short-range gravitational tree force is done.\n");

    /* Now the force computation is finished */

    /*  gather some diagnostic information */

    timetree = tw->timecomp1 + tw->timecomp2 + tw->timecomp3;
    timewait = tw->timewait1 + tw->timewait2;
    timecomm= tw->timecommsumm1 + tw->timecommsumm2;

    walltime_add("/Tree/Walk1", tw->timecomp1);
    walltime_add("/Tree/Walk2", tw->timecomp2);
    walltime_add("/Tree/PostProcess", tw->timecomp3);
    walltime_add("/Tree/Send", tw->timecommsumm1);
    walltime_add("/Tree/Recv", tw->timecommsumm2);
    walltime_add("/Tree/Wait1", tw->timewait1);
    walltime_add("/Tree/Wait2", tw->timewait2);

    timeall = walltime_measure(WALLTIME_IGNORE);

    walltime_add("/Tree/Misc", timeall - (timetree + timewait + timecomm));

    /* TreeUseBH > 1 means use the BH criterion on the initial timestep only,
     * avoiding the fully open O(N^2) case.*/
    if(TreeParams.TreeUseBH > 1)
        TreeParams.TreeUseBH = 0;
}

/* Add the acceleration from a node or particle to the output structure,
 * computing the short-range kernel and softening.*/
static void
apply_accn_to_output(TreeWalkResultGravShort * output, const double dx[3], const double r2, const double h, const double mass, const double cellsize)
{
    double facpot, fac;

    const double r = sqrt(r2);

    if(r >= h)
    {
        fac = mass / (r2 * r);
        facpot = -mass / r;
    }
    else
    {
        double wp;
        const double h_inv = 1.0 / h;
        const double h3_inv = h_inv * h_inv * h_inv;
        const double u = r * h_inv;
        if(u < 0.5) {
            fac = mass * h3_inv * (10.666666666667 + u * u * (32.0 * u - 38.4));
            wp = -2.8 + u * u * (5.333333333333 + u * u * (6.4 * u - 9.6));
        }
        else {
            fac =
                mass * h3_inv * (21.333333333333 - 48.0 * u +
                        38.4 * u * u - 10.666666666667 * u * u * u - 0.066666666667 / (u * u * u));
            wp =
                -3.2 + 0.066666666667 / u + u * u * (10.666666666667 +
                        u * (-16.0 + u * (9.6 - 2.133333333333 * u)));
        }

        facpot = mass * h_inv * wp;
    }

    if(0 == grav_apply_short_range_window(r, &fac, &facpot, cellsize)) {
        int i;
        for(i = 0; i < 3; i++)
            output->Acc[i] += dx[i] * fac;
        output->Ninteractions++;
        output->Potential += facpot;
    }
}

/* Check whether a node should be discarded completely, its contents not contributing
 * to the acceleration. This happens if the node is further away than the short-range force cutoff.
 * Return 1 if the node should be discarded, 0 otherwise. */
static int
shall_we_discard_node(const double len, const double r2, const double center[3], const double inpos[3], const double BoxSize, const double rcut, const double rcut2)
{
    /* This checks the distance from the node center of mass
     * is greater than the cutoff. */
    if(r2 > rcut2)
    {
        /* check whether we can stop walking along this branch */
        const double eff_dist = rcut + 0.5 * len;
        int i;
        /*This checks whether we are also outside this region of the oct-tree*/
        /* As long as one dimension is outside, we are fine*/
        for(i=0; i < 3; i++)
            if(fabs(NEAREST(center[i] - inpos[i], BoxSize)) > eff_dist)
                return 1;
    }
    return 0;
}

/* This function tests whether a node shall be opened (ie, should the next node be .
 * If it should be discarded, 0 is returned.
 * If it should be used, 1 is returned, otherwise zero is returned. */
static int
shall_we_open_node(const double len, const double mass, const double r2, const double center[3], const double inpos[3], const double BoxSize, const double aold, const int TreeUseBH, const double BHOpeningAngle2)
{
    /* Check the relative acceleration opening condition*/
    if((TreeUseBH == 0) && (mass * len * len > r2 * r2 * aold))
         return 1;
     /*Check Barnes-Hut opening angle*/
    if((TreeUseBH > 0) && (len * len > r2 * BHOpeningAngle2))
         return 1;

    const double inside = 0.6 * len;
    /* Open the cell if we are inside it, even if the opening criterion is not satisfied.*/
    if(fabs(NEAREST(center[0] - inpos[0], BoxSize)) < inside &&
        fabs(NEAREST(center[1] - inpos[1], BoxSize)) < inside &&
        fabs(NEAREST(center[2] - inpos[2], BoxSize)) < inside)
        return 1;

    /* ok, node can be used */
    return 0;
}

/*! In the TreePM algorithm, the tree is walked only locally around the
 *  target coordinate.  Tree nodes that fall outside a box of half
 *  side-length Rcut= RCUT*ASMTH*MeshSize can be discarded. The short-range
 *  potential is modified by a complementary error function, multiplied
 *  with the Newtonian form. The resulting short-range suppression compared
 *  to the Newtonian force is tabulated, because looking up from this table
 *  is faster than recomputing the corresponding factor, despite the
 *  memory-access penalty (which reduces cache performance) incurred by the
 *  table.
 */
int force_treeev_shortrange(TreeWalkQueryGravShort * input,
        TreeWalkResultGravShort * output,
        LocalTreeWalk * lv)
{
    const ForceTree * tree = lv->tw->tree;
    const double BoxSize = tree->BoxSize;

    /*Tree-opening constants*/
    const double cellsize = GRAV_GET_PRIV(lv->tw)->cellsize;
    const double rcut = GRAV_GET_PRIV(lv->tw)->Rcut;
    const double rcut2 = rcut * rcut;
    const double aold = GRAV_GET_PRIV(lv->tw)->ErrTolForceAcc * input->OldAcc;
    const int TreeUseBH = GRAV_GET_PRIV(lv->tw)->TreeUseBH;
    const double BHOpeningAngle2 = GRAV_GET_PRIV(lv->tw)->BHOpeningAngle * GRAV_GET_PRIV(lv->tw)->BHOpeningAngle;
    const int NeutrinoTracer = GRAV_GET_PRIV(lv->tw)->NeutrinoTracer;
    const int FastParticleType = GRAV_GET_PRIV(lv->tw)->FastParticleType;

    /*Input particle data*/
    const double * inpos = input->base.Pos;

    /*Start the tree walk*/
    int listindex;

    /* Primary treewalk only ever has one nodelist entry*/
    for(listindex = 0; listindex < NODELISTLENGTH && (lv->mode == 1 || listindex < 1); listindex++)
    {
        int numcand = 0;
        /* Use the next node in the node list if we are doing a secondary walk.
         * For a primary walk the node list only ever contains one node. */
        int no = input->base.NodeList[listindex];
        int startno = no;
        if(no < 0)
            break;

        while(no >= 0)
        {
            /* The tree always walks internal nodes*/
            struct NODE *nop = &tree->Nodes[no];

            if(lv->mode == 1)
            {
                if(nop->f.TopLevel && no != startno)	/* we reached a top-level node again, which means that we are done with the branch */
                {
                    no = -1;
                    continue;
                }
            }

            int i;
            double dx[3];
            for(i = 0; i < 3; i++)
                dx[i] = NEAREST(nop->mom.cofm[i] - inpos[i], BoxSize);
            const double r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];

            /* Discard this node, move to sibling*/
            if(shall_we_discard_node(nop->len, r2, nop->center, inpos, BoxSize, rcut, rcut2))
            {
                no = nop->sibling;
                /* Don't add this node*/
                continue;
            }

            /* This node accelerates the particle directly, and is not opened.*/
            if(!shall_we_open_node(nop->len, nop->mom.mass, r2, nop->center, inpos, BoxSize, aold, TreeUseBH, BHOpeningAngle2))
            {
                double h = DMAX(input->Soft, nop->mom.MaxSoftening);
                /* Always open the node if it has a larger softening than the particle,
                 * and the particle is inside its softening radius.
                 * This condition essentially never happens, and it is not clear how much sense it makes. */
                if(input->Soft < nop->mom.MaxSoftening)
                {
                    if(r2 < h * h)
                    {
                        no = nop->nextnode;
                        continue;
                    }
                }

                /* ok, node can be used */
                no = nop->sibling;
                /* Compute the acceleration and apply it to the output structure*/
                apply_accn_to_output(output, dx, r2, h, nop->mom.mass, cellsize);
                continue;
            }

            /* Now we have a cell that needs to be opened.
             * If it contains particles we can add them directly here */
            if(nop->f.ChildType == PARTICLE_NODE_TYPE)
            {
                /* Loop over child particles*/
                for(i = 0; i < nop->s.noccupied; i++) {
                    int pp = nop->s.suns[i];
                    lv->ngblist[numcand++] = pp;
                }
                no = nop->sibling;
            }
            else if (nop->f.ChildType == PSEUDO_NODE_TYPE)
            {
                if(lv->mode == 0)
                {
                    if(-1 == treewalk_export_particle(lv, nop->nextnode))
                        return -1;
                }

                /* Move to the sibling (likely also a pseudo node)*/
                no = nop->sibling;
            }
            else if(nop->f.ChildType == NODE_NODE_TYPE)
            {
                /* This node contains other nodes and we need to open it.*/
                no = nop->nextnode;
            }
        }
        int i;
        for(i = 0; i < numcand; i++)
        {
            int pp = lv->ngblist[i];
            /* Fast particle neutrinos don't cause short-range acceleration before activation.*/
            if(NeutrinoTracer && P[pp].Type == FastParticleType)
                continue;

            double dx[3];
            int j;
            for(j = 0; j < 3; j++)
                dx[j] = NEAREST(P[pp].Pos[j] - inpos[j], BoxSize);
            const double r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];

            double h = DMAX(input->Soft, FORCE_SOFTENING(pp));
            /* Compute the acceleration and apply it to the output structure*/
            apply_accn_to_output(output, dx, r2, h, P[pp].Mass, cellsize);
        }
    }

    lv->Ninteractions += output->Ninteractions;
    if(lv->mode == 1) {
        lv->Nnodesinlist += listindex;
        lv->Nlist += 1;
    }
    return output->Ninteractions;
}


