#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ipc.h>
#include <sys/sem.h>

#include "utils/endrun.h"
#include "utils/mymalloc.h"

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
/*Softening length*/
static double GravitySoftening;
static double MaxSoftening;

/* gravitational softening length
 * (given in terms of an `equivalent' Plummer softening length)
 */
double FORCE_SOFTENING(int type)
{
    if (TreeParams.MultiSpeciesSoftening == 1) {
        if (type == 0)
            return 2.8 * TreeParams.SofteningType0;
        if (type == 1)
            return 2.8 * TreeParams.SofteningType1;
        if (type == 4)
            return 2.8 * TreeParams.SofteningType4;
        if (type == 5)
            return 2.8 * TreeParams.SofteningType5;
    }
    /* Force is Newtonian beyond this.*/
    return 2.8 * GravitySoftening;
}


/*! Sets the (comoving) softening length, converting from units of the mean separation to comoving internal units. */
void
gravshort_set_softenings(double MeanSeparation)
{
    GravitySoftening = TreeParams.FractionalGravitySoftening * MeanSeparation;
    /* 0: Gas is collisional */
    message(0, "GravitySoftening = %g\n", GravitySoftening);
}

void gravshort_set_max_softening(void)
{
    if (TreeParams.MultiSpeciesSoftening == 0) {
        MaxSoftening = GravitySoftening;
        message(0, "Maximum Softening = %g\n", MaxSoftening);
        return;
    }

    double maxsoft = 0;

    if (TreeParams.SofteningType0 > maxsoft)
        maxsoft = TreeParams.SofteningType0;
    if (TreeParams.SofteningType1 > maxsoft)
        maxsoft = TreeParams.SofteningType1;
    if (TreeParams.SofteningType4 > maxsoft)
        maxsoft = TreeParams.SofteningType4;
    if (TreeParams.SofteningType5 > maxsoft)
        maxsoft = TreeParams.SofteningType5;
    MaxSoftening = 2.8 * maxsoft;
    message(0, "Maximum Softening = %g\n", MaxSoftening);
}


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
        TreeParams.TreeUseBH= param_get_int(ps, "TreeUseBH");
        TreeParams.Rcut = param_get_double(ps, "TreeRcut");
        TreeParams.FractionalGravitySoftening = param_get_double(ps, "GravitySoftening");
        TreeParams.MaxBHOpeningAngle = param_get_double(ps, "MaxBHOpeningAngle");
        TreeParams.MultiSpeciesSoftening = param_get_int(ps, "MultiSpeciesSoftening");
        TreeParams.SofteningType0 = param_get_double(ps, "SofteningType0");
        TreeParams.SofteningType1 = param_get_double(ps, "SofteningType1");
        TreeParams.SofteningType4 = param_get_double(ps, "SofteningType4");
        TreeParams.SofteningType5 = param_get_double(ps, "SofteningType5");
    }
    MPI_Bcast(&TreeParams, sizeof(struct gravshort_tree_params), MPI_BYTE, 0, MPI_COMM_WORLD);
}

int
force_treeev_shortrange(TreeWalkQueryGravShort * input,
        TreeWalkResultGravShort * output,
        LocalTreeWalk * lv);

/*! This function computes the gravitational forces for all active particles from all particles in the tree.
 * Particles are only exported to other processors when really
 *  needed, thereby allowing a good use of the communication buffer.
 *  NeutrinoTracer = All.HybridNeutrinosOn && (atime <= All.HybridNuPartTime);
 *  rho0 = CP.Omega0 * 3 * CP.Hubble * CP.Hubble / (8 * M_PI * G)
 *  ActiveParticle should contain only gravitationally active particles.
 *  If this tree contains all particles, as specified by the full_particle_tree_flag, we calculate the short-
 * range gravitational potential and update the fulltreegravaccel. Note that in practice
 * for hierarchical gravity only active particles are in the tree and so this is
 * only true on PM steps where all particles are active.
 */
void
grav_short_tree(const ActiveParticles * act, PetaPM * pm, ForceTree * tree, MyFloat (* AccelStore)[3], double rho0, inttime_t Ti_Current)
{
    TreeWalk tw[1] = {{0}};
    struct GravShortPriv priv;
    priv.cellsize = tree->BoxSize / pm->Nmesh;
    priv.Rcut = TreeParams.Rcut * pm->Asmth * priv.cellsize;;
    priv.G = pm->G;
    priv.cbrtrho0 = pow(rho0, 1.0 / 3);
    priv.Ti_Current = Ti_Current;
    priv.NonPeriodic = pm->NonPeriodic;
    priv.Accel = AccelStore;
    int accelstorealloc = 0;
    if(!AccelStore) {
        priv.Accel = (MyFloat (*) [3]) mymalloc2("GravAccel", PartManager->NumPart * sizeof(priv.Accel[0]));
        accelstorealloc = 1;
    }

    if(!tree->moments_computed_flag)
        endrun(2, "Gravtree called before tree moments computed!\n");

    tw->ev_label = "GRAVTREE";
    tw->visit = (TreeWalkVisitFunction) force_treeev_shortrange;
    /* gravity applies to all gravitationally active particles.*/
    tw->haswork = NULL;
    tw->reduce = (TreeWalkReduceResultFunction) grav_short_reduce;
    tw->postprocess = (TreeWalkProcessFunction) grav_short_postprocess;

    tw->query_type_elsize = sizeof(TreeWalkQueryGravShort);
    tw->result_type_elsize = sizeof(TreeWalkResultGravShort);
    tw->fill = (TreeWalkFillQueryFunction) grav_short_copy;
    tw->tree = tree;
    tw->priv = &priv;

    treewalk_run(tw, act->ActiveParticle, act->NumActiveParticle);

    /* Now the force computation is finished */
    /*  gather some diagnostic information */

    double timetree = tw->timecomp0 + tw->timecomp1 + tw->timecomp2 + tw->timecomp3;
    walltime_add("/Tree/WalkTop", tw->timecomp0);
    walltime_add("/Tree/WalkPrim", tw->timecomp1);
    walltime_add("/Tree/WalkSec", tw->timecomp2);
    walltime_add("/Tree/Reduce", tw->timecommsumm);
    walltime_add("/Tree/PostPre", tw->timecomp3);
    walltime_add("/Tree/Wait", tw->timewait1);

    double timeall = walltime_measure(WALLTIME_IGNORE);

    walltime_add("/Tree/Misc", timeall - (timetree + tw->timewait1 + tw->timecommsumm));

    treewalk_print_stats(tw);

    /* TreeUseBH > 1 means use the BH criterion on the initial timestep only,
     * avoiding the fully open O(N^2) case.*/
    if(TreeParams.TreeUseBH > 1)
        TreeParams.TreeUseBH = 0;
    if(accelstorealloc)
        myfree(priv.Accel);
}

/* Add the acceleration from a node or particle to the output structure,
 * computing the short-range kernel and softening.*/
static void
apply_accn_to_output(TreeWalkResultGravShort * output, const double dx[3], const double r2, const double mass, const double cellsize, const double h)
{
    const double r = sqrt(r2);
    double fac = mass / (r2 * r);
    double facpot = -mass / r;

    if(r2 < h*h)
    {
        double wp;
        const double h3_inv = 1.0 / h / h / h;
        const double u = r / h;
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
        facpot = mass / h * wp;
    }

    if(0 == grav_apply_short_range_window(r, &fac, &facpot, cellsize)) {
        int i;
        for(i = 0; i < 3; i++)
            output->Acc[i] += dx[i] * fac;
        output->Potential += facpot;
    }
}

/* Check whether a node should be discarded completely, its contents not contributing
 * to the acceleration. This happens if the node is further away than the short-range force cutoff.
 * Return 1 if the node should be discarded, 0 otherwise. */
static int
shall_we_discard_node(const double len, const double r2, const double center[3], const double inpos[3], const double BoxSize, const double rcut, const double rcut2, const int NonPeriodic)
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
        for(i=0; i < 3; i++) {
            if ((NonPeriodic) && (fabs(center[i] - inpos[i]) > eff_dist)) {
                return 1;
            }
            else if (fabs(NEAREST(center[i] - inpos[i], BoxSize)) > eff_dist) {
                return 1;
            }
        }
    }
    return 0;
}

/* This function tests whether a node shall be opened (ie, should the next node be .
 * If it should be discarded, 0 is returned.
 * If it should be used, 1 is returned, otherwise zero is returned. */
static int
shall_we_open_node(const double len, const double mass, const double r2, const double center[3], const double inpos[3], const double BoxSize, const double aold, const int TreeUseBH, const double BHOpeningAngle2, const int NonPeriodic)
{
    /* Check the relative acceleration opening condition*/
    if((TreeUseBH == 0) && (mass * len * len > r2 * r2 * aold))
         return 1;

    double bhangle = len * len  / r2;
     /*Check Barnes-Hut opening angle*/
    if(bhangle > BHOpeningAngle2)
         return 1;

    const double inside = 0.6 * len;
    /* Open the cell if we are inside it, even if the opening criterion is not satisfied.*/
    if (NonPeriodic) {
        if (fabs(center[0] - inpos[0]) < inside &&
            fabs(center[1] - inpos[1]) < inside &&
            fabs(center[2] - inpos[2]) < inside)
            return 1;
    }
    else if (fabs(NEAREST(center[0] - inpos[0], BoxSize)) < inside &&
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
    const double aold = TreeParams.ErrTolForceAcc * input->OldAcc;
    const int TreeUseBH = TreeParams.TreeUseBH;
    double BHOpeningAngle2 = TreeParams.BHOpeningAngle * TreeParams.BHOpeningAngle;
    /* Enforce a maximum opening angle even for relative acceleration criterion, to avoid
     * pathological cases. Default value is 0.9, from Volker Springel.*/
    if(TreeUseBH == 0)
        BHOpeningAngle2 = TreeParams.MaxBHOpeningAngle * TreeParams.MaxBHOpeningAngle;
    const int NonPeriodic = GRAV_GET_PRIV(lv->tw)->NonPeriodic;

    /*Input particle data*/
    const double * inpos = input->base.Pos;

    /*Start the tree walk*/
    int listindex, ninteractions=0;

    /* Primary treewalk only ever has one nodelist entry*/
    for(listindex = 0; listindex < NODELISTLENGTH; listindex++)
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

            if(lv->mode == TREEWALK_GHOSTS && nop->f.TopLevel && no != startno)  /* we reached a top-level node again, which means that we are done with the branch */
                break;

            int i;
            double dx[3];
            for(i = 0; i < 3; i++) {
                if (NonPeriodic) {
                        dx[i] = nop->mom.cofm[i] - inpos[i];
                    }
                else {
                    dx[i] = NEAREST(nop->mom.cofm[i] - inpos[i], BoxSize);
                }
            }
            const double r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];

            /* Discard this node, move to sibling*/
            if(shall_we_discard_node(nop->len, r2, nop->center, inpos, BoxSize, rcut, rcut2, NonPeriodic))
            {
                no = nop->sibling;
                /* Don't add this node*/
                continue;
            }

            /* This node accelerates the particle directly, and is not opened.*/
            int open_node = shall_we_open_node(nop->len, nop->mom.mass, r2, nop->center, inpos, BoxSize, aold, TreeUseBH, BHOpeningAngle2, NonPeriodic);

            if ((TreeParams.MultiSpeciesSoftening == 1) && (input->Soft < MaxSoftening)) {
                if (r2 < MaxSoftening * MaxSoftening)
                        open_node = 1;
            }

            if(!open_node)
            {
                double h = input->Soft;
                if (TreeParams.MultiSpeciesSoftening)
                    h = DMAX(input->Soft, MaxSoftening);
                /* ok, node can be used */
                no = nop->sibling;
                if(lv->mode != TREEWALK_TOPTREE) {
                    /* Compute the acceleration and apply it to the output structure*/
                    apply_accn_to_output(output, dx, r2, nop->mom.mass, cellsize, h);
                }
                continue;
            }

            if(lv->mode == TREEWALK_TOPTREE) {
                if(nop->f.ChildType == PSEUDO_NODE_TYPE) {
                    /* Export the pseudo particle*/
                    if(-1 == treewalk_export_particle(lv, nop->s.suns[0]))
                        return -1;
                    /* Move sideways*/
                    no = nop->sibling;
                    continue;
                }
                /* Only walk toptree nodes here*/
                if(nop->f.TopLevel && !nop->f.InternalTopLevel) {
                    no = nop->sibling;
                    continue;
                }
                no = nop->s.suns[0];
            }
            else {
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
                    /* Move to the sibling (likely also a pseudo node)*/
                    no = nop->sibling;
                }
                else //NODE_NODE_TYPE
                    /* This node contains other nodes and we need to open it.*/
                    no = nop->s.suns[0];
            }
        }
        int i;
        for(i = 0; i < numcand; i++)
        {
            int pp = lv->ngblist[i];
            double dx[3];
            int j;
            for(j = 0; j < 3; j++) {
                if (NonPeriodic) {
                    dx[j] = P[pp].Pos[j] - inpos[j];
                }
                else {
                    dx[j] = NEAREST(P[pp].Pos[j] - inpos[j], BoxSize);
                }
            }
            const double r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
            /* This is always the Newtonian softening,
             * match the default from FORCE_SOFTENING. */
            double h = 2.8 * GravitySoftening;
            if( TreeParams.MultiSpeciesSoftening == 1)  {
                h = DMAX(input->Soft, FORCE_SOFTENING(P[pp].Type));
            }
            /* Compute the acceleration and apply it to the output structure*/
            apply_accn_to_output(output, dx, r2, P[pp].Mass, cellsize, h);
        }
        ninteractions = numcand;
    }
    treewalk_add_counters(lv, ninteractions);
    return 1;
}
