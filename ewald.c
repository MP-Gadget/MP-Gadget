#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "allvars.h"
#include "proto.h"

#ifndef PETAPM
#ifdef PERIODIC

/*! This function initializes tables with the correction force and the
 *  correction potential due to the periodic images of a point mass located
 *  at the origin. These corrections are obtained by Ewald summation. (See
 *  Hernquist, Bouchet, Suto, ApJS, 1991, 75, 231) The correction fields
 *  are used to obtain the full periodic force if periodic boundaries
 *  combined with the pure tree algorithm are used. For the TreePM
 *  algorithm, the Ewald correction is not used.
 *
 *  The correction fields are stored on disk once they are computed. If a
 *  corresponding file is found, they are loaded from disk to speed up the
 *  initialization.  The Ewald summation is done in parallel, i.e. the
 *  processors share the work to compute the tables if needed.
 */
void ewald_init(void)
{
    int i, j, k, beg, len, size, n, task, count;
    double x[3], force[3];
    char buf[200];
    FILE *fd;

    if(ThisTask == 0)
    {
        printf("initialize Ewald correction...\n");
        fflush(stdout);
    }

    sprintf(buf, "ewald_spc_table_%d_dbl.dat", EN);

    if((fd = fopen(buf, "r")))
    {
        if(ThisTask == 0)
        {
            printf("\nreading Ewald tables from file `%s'\n", buf);
            fflush(stdout);
        }

        my_fread(&fcorrx[0][0][0], sizeof(MyFloat), (EN + 1) * (EN + 1) * (EN + 1), fd);
        my_fread(&fcorry[0][0][0], sizeof(MyFloat), (EN + 1) * (EN + 1) * (EN + 1), fd);
        my_fread(&fcorrz[0][0][0], sizeof(MyFloat), (EN + 1) * (EN + 1) * (EN + 1), fd);
        my_fread(&potcorr[0][0][0], sizeof(MyFloat), (EN + 1) * (EN + 1) * (EN + 1), fd);
        fclose(fd);
    }
    else
    {
        if(ThisTask == 0)
        {
            printf("\nNo Ewald tables in file `%s' found.\nRecomputing them...\n", buf);
            fflush(stdout);
        }

        /* ok, let's recompute things. Actually, we do that in parallel. */

        size = (EN + 1) * (EN + 1) * (EN + 1) / NTask;
        beg = ((int64_t) ThisTask) * size / NTask;
        len = ((int64_t) ThisTask + 1) * size / NTask;
        len -= beg;
        for(i = 0, count = 0; i <= EN; i++)
            for(j = 0; j <= EN; j++)
                for(k = 0; k <= EN; k++)
                {
                    n = (i * (EN + 1) + j) * (EN + 1) + k;
                    if(n >= beg && n < (beg + len))
                    {
                        if(ThisTask == 0)
                        {
                            if((len / 20) > 0 && (count % (len / 20)) == 0)
                            {
                                printf("%4.1f percent done\n", count / (len / 100.0));
                                fflush(stdout);
                            }
                        }

                        x[0] = 0.5 * ((double) i) / EN;
                        x[1] = 0.5 * ((double) j) / EN;
                        x[2] = 0.5 * ((double) k) / EN;
                        ewald_force(i, j, k, x, force);
                        fcorrx[i][j][k] = force[0];
                        fcorry[i][j][k] = force[1];
                        fcorrz[i][j][k] = force[2];
                        if(i + j + k == 0)
                            potcorr[i][j][k] = 2.8372975;
                        else
                            potcorr[i][j][k] = ewald_psi(x);
                        count++;
                    }
                }

        for(task = 0; task < NTask; task++)
        {
            beg = task * size;
            len = size;
            if(task == (NTask - 1))
                len = (EN + 1) * (EN + 1) * (EN + 1) - beg;
            MPI_Bcast(&fcorrx[0][0][beg], len * sizeof(MyFloat), MPI_BYTE, task, MPI_COMM_WORLD);
            MPI_Bcast(&fcorry[0][0][beg], len * sizeof(MyFloat), MPI_BYTE, task, MPI_COMM_WORLD);
            MPI_Bcast(&fcorrz[0][0][beg], len * sizeof(MyFloat), MPI_BYTE, task, MPI_COMM_WORLD);
            MPI_Bcast(&potcorr[0][0][beg], len * sizeof(MyFloat), MPI_BYTE, task, MPI_COMM_WORLD);
        }

        if(ThisTask == 0)
        {
            printf("\nwriting Ewald tables to file `%s'\n", buf);
            fflush(stdout);
            if((fd = fopen(buf, "w")))
            {
                my_fwrite(&fcorrx[0][0][0], sizeof(MyFloat), (EN + 1) * (EN + 1) * (EN + 1), fd);
                my_fwrite(&fcorry[0][0][0], sizeof(MyFloat), (EN + 1) * (EN + 1) * (EN + 1), fd);
                my_fwrite(&fcorrz[0][0][0], sizeof(MyFloat), (EN + 1) * (EN + 1) * (EN + 1), fd);
                my_fwrite(&potcorr[0][0][0], sizeof(MyFloat), (EN + 1) * (EN + 1) * (EN + 1), fd);
                fclose(fd);
            }
        }
    }

    fac_intp = 2 * EN / All.BoxSize;
    for(i = 0; i <= EN; i++)
        for(j = 0; j <= EN; j++)
            for(k = 0; k <= EN; k++)
            {
                potcorr[i][j][k] /= All.BoxSize;
                fcorrx[i][j][k] /= All.BoxSize * All.BoxSize;
                fcorry[i][j][k] /= All.BoxSize * All.BoxSize;
                fcorrz[i][j][k] /= All.BoxSize * All.BoxSize;
            }

    if(ThisTask == 0)
    {
        printf("initialization of periodic boundaries finished.\n");
        fflush(stdout);
    }
}


/*! This function looks up the correction force due to the infinite number
 *  of periodic particle/node images. We here use trilinear interpolation
 *  to get it from the precomputed tables, which contain one octant
 *  around the target particle at the origin. The other octants are
 *  obtained from it by exploiting the symmetry properties.
 */
#ifdef FORCETEST
void ewald_corr(double dx, double dy, double dz, double *fper)
{
    int signx, signy, signz;
    int i, j, k;
    double u, v, w;
    double f1, f2, f3, f4, f5, f6, f7, f8;

    if(dx < 0)
    {
        dx = -dx;
        signx = +1;
    }
    else
        signx = -1;
    if(dy < 0)
    {
        dy = -dy;
        signy = +1;
    }
    else
        signy = -1;
    if(dz < 0)
    {
        dz = -dz;
        signz = +1;
    }
    else
        signz = -1;
    u = dx * fac_intp;
    i = (int) u;
    if(i >= EN)
        i = EN - 1;
    u -= i;
    v = dy * fac_intp;
    j = (int) v;
    if(j >= EN)
        j = EN - 1;
    v -= j;
    w = dz * fac_intp;
    k = (int) w;
    if(k >= EN)
        k = EN - 1;
    w -= k;
    f1 = (1 - u) * (1 - v) * (1 - w);
    f2 = (1 - u) * (1 - v) * (w);
    f3 = (1 - u) * (v) * (1 - w);
    f4 = (1 - u) * (v) * (w);
    f5 = (u) * (1 - v) * (1 - w);
    f6 = (u) * (1 - v) * (w);
    f7 = (u) * (v) * (1 - w);
    f8 = (u) * (v) * (w);
    fper[0] = signx * (fcorrx[i][j][k] * f1 +
            fcorrx[i][j][k + 1] * f2 +
            fcorrx[i][j + 1][k] * f3 +
            fcorrx[i][j + 1][k + 1] * f4 +
            fcorrx[i + 1][j][k] * f5 +
            fcorrx[i + 1][j][k + 1] * f6 +
            fcorrx[i + 1][j + 1][k] * f7 + fcorrx[i + 1][j + 1][k + 1] * f8);
    fper[1] =
        signy * (fcorry[i][j][k] * f1 + fcorry[i][j][k + 1] * f2 +
                fcorry[i][j + 1][k] * f3 + fcorry[i][j + 1][k + 1] * f4 +
                fcorry[i + 1][j][k] * f5 + fcorry[i + 1][j][k + 1] * f6 +
                fcorry[i + 1][j + 1][k] * f7 + fcorry[i + 1][j + 1][k + 1] * f8);
    fper[2] =
        signz * (fcorrz[i][j][k] * f1 + fcorrz[i][j][k + 1] * f2 +
                fcorrz[i][j + 1][k] * f3 + fcorrz[i][j + 1][k + 1] * f4 +
                fcorrz[i + 1][j][k] * f5 + fcorrz[i + 1][j][k + 1] * f6 +
                fcorrz[i + 1][j + 1][k] * f7 + fcorrz[i + 1][j + 1][k + 1] * f8);
}
#endif


/*! This function looks up the correction potential due to the infinite
 *  number of periodic particle/node images. We here use tri-linear
 *  interpolation to get it from the precomputed table, which contains
 *  one octant around the target particle at the origin. The other
 *  octants are obtained from it by exploiting symmetry properties.
 */
double ewald_pot_corr(double dx, double dy, double dz)
{
    int i, j, k;
    double u, v, w;
    double f1, f2, f3, f4, f5, f6, f7, f8;

    if(dx < 0)
        dx = -dx;
    if(dy < 0)
        dy = -dy;
    if(dz < 0)
        dz = -dz;
    u = dx * fac_intp;
    i = (int) u;
    if(i >= EN)
        i = EN - 1;
    u -= i;
    v = dy * fac_intp;
    j = (int) v;
    if(j >= EN)
        j = EN - 1;
    v -= j;
    w = dz * fac_intp;
    k = (int) w;
    if(k >= EN)
        k = EN - 1;
    w -= k;
    f1 = (1 - u) * (1 - v) * (1 - w);
    f2 = (1 - u) * (1 - v) * (w);
    f3 = (1 - u) * (v) * (1 - w);
    f4 = (1 - u) * (v) * (w);
    f5 = (u) * (1 - v) * (1 - w);
    f6 = (u) * (1 - v) * (w);
    f7 = (u) * (v) * (1 - w);
    f8 = (u) * (v) * (w);
    return potcorr[i][j][k] * f1 +
        potcorr[i][j][k + 1] * f2 +
        potcorr[i][j + 1][k] * f3 +
        potcorr[i][j + 1][k + 1] * f4 +
        potcorr[i + 1][j][k] * f5 +
        potcorr[i + 1][j][k + 1] * f6 + potcorr[i + 1][j + 1][k] * f7 + potcorr[i + 1][j + 1][k + 1] * f8;
}



/*! This function computes the potential correction term by means of Ewald
 *  summation.
 */
double ewald_psi(double x[3])
{
    double alpha, psi;
    double r, sum1, sum2, hdotx;
    double dx[3];
    int i, n[3], h[3], h2;

    alpha = 2.0;
    for(n[0] = -4, sum1 = 0; n[0] <= 4; n[0]++)
        for(n[1] = -4; n[1] <= 4; n[1]++)
            for(n[2] = -4; n[2] <= 4; n[2]++)
            {
                for(i = 0; i < 3; i++)
                    dx[i] = x[i] - n[i];
                r = sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
                sum1 += erfc(alpha * r) / r;
            }

    for(h[0] = -4, sum2 = 0; h[0] <= 4; h[0]++)
        for(h[1] = -4; h[1] <= 4; h[1]++)
            for(h[2] = -4; h[2] <= 4; h[2]++)
            {
                hdotx = x[0] * h[0] + x[1] * h[1] + x[2] * h[2];
                h2 = h[0] * h[0] + h[1] * h[1] + h[2] * h[2];
                if(h2 > 0)
                    sum2 += 1 / (M_PI * h2) * exp(-M_PI * M_PI * h2 / (alpha * alpha)) * cos(2 * M_PI * hdotx);
            }

    r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    psi = M_PI / (alpha * alpha) - sum1 - sum2 + 1 / r;
    return psi;
}


/*! This function computes the force correction term (difference between full
 *  force of infinite lattice and nearest image) by Ewald summation.
 */
void ewald_force(int iii, int jjj, int kkk, double x[3], double force[3])
{
    double alpha, r2;
    double r, val, hdotx, dx[3];
    int i, h[3], n[3], h2;

    alpha = 2.0;
    for(i = 0; i < 3; i++)
        force[i] = 0;
    if(iii == 0 && jjj == 0 && kkk == 0)
        return;
    r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
    for(i = 0; i < 3; i++)
        force[i] += x[i] / (r2 * sqrt(r2));
    for(n[0] = -4; n[0] <= 4; n[0]++)
        for(n[1] = -4; n[1] <= 4; n[1]++)
            for(n[2] = -4; n[2] <= 4; n[2]++)
            {
                for(i = 0; i < 3; i++)
                    dx[i] = x[i] - n[i];
                r = sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
                val = erfc(alpha * r) + 2 * alpha * r / sqrt(M_PI) * exp(-alpha * alpha * r * r);
                for(i = 0; i < 3; i++)
                    force[i] -= dx[i] / (r * r * r) * val;
            }

    for(h[0] = -4; h[0] <= 4; h[0]++)
        for(h[1] = -4; h[1] <= 4; h[1]++)
            for(h[2] = -4; h[2] <= 4; h[2]++)
            {
                hdotx = x[0] * h[0] + x[1] * h[1] + x[2] * h[2];
                h2 = h[0] * h[0] + h[1] * h[1] + h[2] * h[2];
                if(h2 > 0)
                {
                    val = 2.0 / ((double) h2) * exp(-M_PI * M_PI * h2 / (alpha * alpha)) * sin(2 * M_PI * hdotx);
                    for(i = 0; i < 3; i++)
                        force[i] -= h[i] * val;
                }
            }
}

/*! This function computes the Ewald correction, and is needed if periodic
 *  boundary conditions together with a pure tree algorithm are used. Note
 *  that the ordinary tree walk does not carry out this correction directly
 *  as it was done in Gadget-1.1. Instead, the tree is walked a second
 *  time. This is actually faster because the "Ewald-Treewalk" can use a
 *  different opening criterion than the normal tree walk. In particular,
 *  the Ewald correction is negligible for particles that are very close,
 *  but it is large for particles that are far away (this is quite
 *  different for the normal direct force). So we can here use a different
 *  opening criterion. Sufficient accuracy is usually obtained if the node
 *  length has dropped to a certain fraction ~< 0.25 of the
 *  BoxLength. However, we may only short-cut the interaction list of the
 *  normal full Ewald tree walk if we are sure that the whole node and all
 *  daughter nodes "lie on the same side" of the periodic boundary,
 *  i.e. that the real tree walk would not find a daughter node or particle
 *  that was mapped to a different nearest neighbour position when the tree
 *  walk would be further refined.
 */
int force_treeev_ewald_correction(int target, int mode, 
        struct gravitydata_in  * input,
        struct gravitydata_out  * output,
        LocalEvaluator * lv, void * unused)
{
    struct NODE *nop = 0;
    int no, listindex = 0;
    double dx, dy, dz, mass, r2;
    int signx, signy, signz, nexp;
    int i, j, k, openflag, task;
    double u, v, w;
    double f1, f2, f3, f4, f5, f6, f7, f8;
    MyLongDouble acc_x, acc_y, acc_z;
    double boxsize, boxhalf;
    double pos_x, pos_y, pos_z, aold;

    int ninteractions = 0, nnodesinlist = 0;
    boxsize = All.BoxSize;
    boxhalf = 0.5 * All.BoxSize;

    acc_x = 0;
    acc_y = 0;
    acc_z = 0;

    no = input->NodeList[0];
    listindex ++;
    no = Nodes[no].u.d.nextnode;	/* open it */

    pos_x = input->Pos[0];
    pos_y = input->Pos[1];
    pos_z = input->Pos[2];
    aold = All.ErrTolForceAcc * input->OldAcc;

    while(no >= 0)
    {
        while(no >= 0)
        {
            if(no < All.MaxPart)	/* single particle */
            {
                /* the index of the node is the index of the particle */
                /* observe the sign */
                drift_particle(no, All.Ti_Current);

                dx = P[no].Pos[0] - pos_x;
                dy = P[no].Pos[1] - pos_y;
                dz = P[no].Pos[2] - pos_z;
                mass = P[no].Mass;
            }
            else			/* we have an  internal node */
            {
                if(no >= All.MaxPart + MaxNodes)	/* pseudo particle */
                {
                    if(mode == 0)
                    {
                        if(-1 == ev_export_particle(lv, target, no)) 
                            return -1;
                    }
                    no = Nextnode[no - MaxNodes];
                    continue;
                }

                nop = &Nodes[no];

                if(mode == 1)
                {
                    if(nop->u.d.bitflags & (1 << BITFLAG_TOPLEVEL))	/* we reached a top-level node again, which means that we are done with the branch */
                    {
                        no = -1;
                        continue;
                    }
                }

                if(!(nop->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
                {
                    /* open cell */
                    no = nop->u.d.nextnode;
                    continue;
                }

                force_drift_node(no, All.Ti_Current);

                mass = nop->u.d.mass;
                dx = nop->u.d.s[0] - pos_x;
                dy = nop->u.d.s[1] - pos_y;
                dz = nop->u.d.s[2] - pos_z;
            }

            dx = NEAREST(dx);
            dy = NEAREST(dy);
            dz = NEAREST(dz);

            if(no < All.MaxPart)
                no = Nextnode[no];
            else			/* we have an  internal node. Need to check opening criterion */
            {
                openflag = 0;
                r2 = dx * dx + dy * dy + dz * dz;
                if(All.ErrTolTheta)	/* check Barnes-Hut opening criterion */
                {
                    if(nop->len * nop->len > r2 * All.ErrTolTheta * All.ErrTolTheta)
                    {
                        openflag = 1;
                    }
                }
                else		/* check relative opening criterion */
                {
                    if(mass * nop->len * nop->len > r2 * r2 * aold)
                    {
                        openflag = 1;
                    }
                    else
                    {
                        if(fabs(nop->center[0] - pos_x) < 0.60 * nop->len)
                        {
                            if(fabs(nop->center[1] - pos_y) < 0.60 * nop->len)
                            {
                                if(fabs(nop->center[2] - pos_z) < 0.60 * nop->len)
                                {
                                    openflag = 1;
                                }
                            }
                        }
                    }
                }

                if(openflag)
                {
                    /* now we check if we can avoid opening the cell */

                    u = nop->center[0] - pos_x;
                    if(u > boxhalf)
                        u -= boxsize;
                    if(u < -boxhalf)
                        u += boxsize;
                    if(fabs(u) > 0.5 * (boxsize - nop->len))
                    {
                        no = nop->u.d.nextnode;
                        continue;
                    }

                    u = nop->center[1] - pos_y;
                    if(u > boxhalf)
                        u -= boxsize;
                    if(u < -boxhalf)
                        u += boxsize;
                    if(fabs(u) > 0.5 * (boxsize - nop->len))
                    {
                        no = nop->u.d.nextnode;
                        continue;
                    }

                    u = nop->center[2] - pos_z;
                    if(u > boxhalf)
                        u -= boxsize;
                    if(u < -boxhalf)
                        u += boxsize;
                    if(fabs(u) > 0.5 * (boxsize - nop->len))
                    {
                        no = nop->u.d.nextnode;
                        continue;
                    }

                    /* if the cell is too large, we need to refine
                     * it further 
                     */
                    if(nop->len > 0.20 * boxsize)
                    {
                        /* cell is too large */
                        no = nop->u.d.nextnode;
                        continue;
                    }
                }

                no = nop->u.d.sibling;	/* ok, node can be used */
            }

            /* compute the Ewald correction force */

            if(dx < 0)
            {
                dx = -dx;
                signx = +1;
            }
            else
                signx = -1;
            if(dy < 0)
            {
                dy = -dy;
                signy = +1;
            }
            else
                signy = -1;
            if(dz < 0)
            {
                dz = -dz;
                signz = +1;
            }
            else
                signz = -1;
            u = dx * fac_intp;
            i = (int) u;
            if(i >= EN)
                i = EN - 1;
            u -= i;
            v = dy * fac_intp;
            j = (int) v;
            if(j >= EN)
                j = EN - 1;
            v -= j;
            w = dz * fac_intp;
            k = (int) w;
            if(k >= EN)
                k = EN - 1;
            w -= k;
            /* compute factors for trilinear interpolation */
            f1 = (1 - u) * (1 - v) * (1 - w);
            f2 = (1 - u) * (1 - v) * (w);
            f3 = (1 - u) * (v) * (1 - w);
            f4 = (1 - u) * (v) * (w);
            f5 = (u) * (1 - v) * (1 - w);
            f6 = (u) * (1 - v) * (w);
            f7 = (u) * (v) * (1 - w);
            f8 = (u) * (v) * (w);
            acc_x += FLT(mass * signx * (fcorrx[i][j][k] * f1 +
                        fcorrx[i][j][k + 1] * f2 +
                        fcorrx[i][j + 1][k] * f3 +
                        fcorrx[i][j + 1][k + 1] * f4 +
                        fcorrx[i + 1][j][k] * f5 +
                        fcorrx[i + 1][j][k + 1] * f6 +
                        fcorrx[i + 1][j + 1][k] * f7 + fcorrx[i + 1][j + 1][k + 1] * f8));
            acc_y +=
                FLT(mass * signy *
                        (fcorry[i][j][k] * f1 + fcorry[i][j][k + 1] * f2 +
                         fcorry[i][j + 1][k] * f3 + fcorry[i][j + 1][k + 1] * f4 + fcorry[i +
                         1]
                         [j][k] * f5 + fcorry[i + 1][j][k + 1] * f6 + fcorry[i + 1][j +
                         1][k] *
                         f7 + fcorry[i + 1][j + 1][k + 1] * f8));
            acc_z +=
                FLT(mass * signz *
                        (fcorrz[i][j][k] * f1 + fcorrz[i][j][k + 1] * f2 +
                         fcorrz[i][j + 1][k] * f3 + fcorrz[i][j + 1][k + 1] * f4 + fcorrz[i +
                         1]
                         [j][k] * f5 + fcorrz[i + 1][j][k + 1] * f6 + fcorrz[i + 1][j +
                         1][k] *
                         f7 + fcorrz[i + 1][j + 1][k + 1] * f8));
#if defined(PERIODIC) && !defined(GRAVITY_NOT_PERIODIC)
            pot += FLT(mass * ewald_pot_corr(dx, dy, dz));
#endif

            ninteractions ++;
        }

        if(listindex < NODELISTLENGTH)
        {
            no = input->NodeList[listindex];
            if(no >= 0) {
                no = Nodes[no].u.d.nextnode;	/* open it */
                nnodesinlist ++;
                listindex ++;
            }
        }
    }

    output->Acc[0] = acc_x;
    output->Acc[1] = acc_y;
    output->Acc[2] = acc_z;
    output->Ninteractions = ninteractions;

    lv->Ninteractions = ninteractions;
    lv->Nnodesinlist = nnodesinlist;
    return 0;
}

#endif

#endif /* if PETAPM*/
