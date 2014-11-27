#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#include "allvars.h"
#include "proto.h"


#ifdef SHELL_CODE
static struct radius_data
{
    MyDouble radius;
    MyDouble enclosed_mass;
    MyDouble dMdr;
    int GrNr;
    int SubNr;
}
*rad_data;

int compare_radius(const void *a, const void *b)
{
    if(((struct radius_data *) a)->radius < ((struct radius_data *) b)->radius)
        return -1;

    if(((struct radius_data *) a)->radius > ((struct radius_data *) b)->radius)
        return +1;

    return 0;
}

int compare_GrNr_SubNr(const void *a, const void *b)
{
    if(((struct radius_data *) a)->GrNr < (((struct radius_data *) b)->GrNr))
        return -1;

    if(((struct radius_data *) a)->GrNr > (((struct radius_data *) b)->GrNr))
        return +1;

    if(((struct radius_data *) a)->SubNr < (((struct radius_data *) b)->SubNr))
        return -1;

    if(((struct radius_data *) a)->SubNr > (((struct radius_data *) b)->SubNr))
        return +1;

    return 0;
}
#endif


void gravity_static_potential() {
#ifdef SCFPOTENTIAL
    MyDouble xs, ys, zs;
    MyDouble pots, axs, ays, azs;

    if(ThisTask == 0)
    {
        printf("Starting SCF calculation...\n");
        fflush(stdout);
    }

    /* reset the expansion coefficients to zero */
    SCF_reset();
#ifdef SCF_HYBRID
    /* 
       calculate SCF coefficients for local DM particles.
       sum them up from all processors, so every processor
       sees the same expansion coefficients 
       */
    SCF_calc_from_particles();

    /* sum up local coefficients */
    MPI_Allreduce(sinsum, sinsum_all, (SCF_NMAX+1)*(SCF_LMAX+1)*(SCF_LMAX+1), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(cossum, cossum_all, (SCF_NMAX+1)*(SCF_LMAX+1)*(SCF_LMAX+1), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    /* update local coefficients to global coefficients -> every processor has now complete SCF expansion */
    SCF_collect_update();  
    if(ThisTask == 0)
    {
        printf("calculated and collected coefficients.\n");
        fflush(stdout);
    }

#else  
    long old_seed, global_seed_min, global_seed_max;

    /* 
       resample coefficients for expansion 
       make sure that every processors sees the SAME potential, 
       i.e. has the same seed to generate coefficients  
       */
    old_seed=scf_seed;
    SCF_calc_from_random(&scf_seed);
    /* check that all cpus have the same random seed (min max must be the same) */
    MPI_Allreduce(&scf_seed, &global_seed_max, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&scf_seed, &global_seed_min, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);  
    if(ThisTask == 0)
    {
        printf("sampled coefficients with old/new seed = %ld/%ld         min/max=%ld/%ld\n", old_seed, scf_seed, global_seed_min, global_seed_max);
        fflush(stdout);
    }
#endif


    /* get accelerations for all active particles based on current expansion */
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        /* convert to unit sphere */
        to_unit(P[i].Pos[0], P[i].Pos[1], P[i].Pos[2], &xs, &ys, &zs) ;
        /* OR: not */
        //xs = P[i].Pos[0]; ys = P[i].Pos[1]; zs = P[i].Pos[2];

        /* evaluate potential and acceleration */
        SCF_evaluate(xs, ys, zs, &pots, &axs, &ays, &azs);      

        /* scale to system size and add to acceleration*/
#ifdef SCF_HYBRID
        /* 
           add missing STAR<-DM force from SCF (was excluded in tree above)
           */
        if (P[i].Type==2)  
        {
#endif
            /* scale */
            P[i].GravAccel[0] += All.G * SCF_HQ_MASS/(SCF_HQ_A*SCF_HQ_A) * axs;
            P[i].GravAccel[1] += All.G * SCF_HQ_MASS/(SCF_HQ_A*SCF_HQ_A) * ays;      
            P[i].GravAccel[2] += All.G * SCF_HQ_MASS/(SCF_HQ_A*SCF_HQ_A) * azs;            
            /* OR: not */
            //P[i].GravAccel[0] += All.G * axs;
            //P[i].GravAccel[1] += All.G * ays;      
            //P[i].GravAccel[2] += All.G * azs;            

#ifdef DEBUG
            if (P[i].ID==150000)
            {
                printf("SCF-ACCEL (scf)   %d  (%g|%g|%g)\n", All.NumCurrentTiStep, All.G *  SCF_HQ_MASS/(SCF_HQ_A*SCF_HQ_A)*axs, All.G *  SCF_HQ_MASS/(SCF_HQ_A*SCF_HQ_A)*ays, All.G *  SCF_HQ_MASS/(SCF_HQ_A*SCF_HQ_A)*azs);
                /* analyic potential of zeroth order of expansion */
                sphere_acc(xs, ys, zs, &axs, &ays, &azs);
                printf("SCF-ACCEL (exact) %d  (%g|%g|%g)\n", All.NumCurrentTiStep, All.G * axs, All.G * ays, All.G * azs);	  
            } 
#endif

#ifdef SCF_HYBRID
        }
#endif	
    }

    if(ThisTask == 0)
    {
        printf("done.\n");
        fflush(stdout);
    }
#endif





#ifdef SHELL_CODE
    /* core softening */
    MyDouble hsoft, hsoft_tidal;

    /* cumul. masses from other CPUs */
    double *masslist;

    /* number of particles used to smooth out mass profile to get dM/dr */
    int ndiff = SHELL_CODE;

    if(ThisTask == 0)
    {
        printf("Starting shell code calculation...\n");
        fflush(stdout);
    }
#ifdef SIM_ADAPTIVE_SOFT
    double turnaround_radius_local = 0.0, turnaround_radius_global, v;
#endif

    /* set up data for sorting */
    rad_data = (struct radius_data *) mymalloc("rad_data", sizeof(struct radius_data) * NumPart);


    /* set up particle data */
    for(i = 0; i < NumPart; i++)
    {
        P[i].radius = sqrt(P[i].Pos[0] * P[i].Pos[0] + P[i].Pos[1] * P[i].Pos[1] + P[i].Pos[2] * P[i].Pos[2]);

        rad_data[i].radius = P[i].radius;
        rad_data[i].enclosed_mass = P[i].Mass;
        rad_data[i].GrNr = ThisTask;
        rad_data[i].SubNr = i;

#ifdef SIM_ADAPTIVE_SOFT
        v = (P[i].Pos[0] * P[i].Vel[0] + P[i].Pos[1] * P[i].Vel[1] + P[i].Pos[2] * P[i].Vel[2]);
        if((v < 0.0) && (P[i].radius > turnaround_radius_local))
            turnaround_radius_local = P[i].radius;
#endif
    }

#ifdef SIM_ADAPTIVE_SOFT
    /* find global turnaround radius by taking maximum of all CPUs */
    MPI_Allreduce(&turnaround_radius_local, &turnaround_radius_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

#ifdef ANALYTIC_TURNAROUND
#ifdef COMOVING_DISTORTION
    /* comoving turnaround radius */
    All.CurrentTurnaroundRadius =
        All.InitialTurnaroundRadius * pow(All.Time / All.TimeBegin, 1.0 / (3.0 * All.SIM_epsilon));
#else
    All.CurrentTurnaroundRadius =
        All.InitialTurnaroundRadius * pow(All.Time / All.TimeBegin,
                2.0 / 3.0 + 2.0 / (3.0 * 3 * All.SIM_epsilon));
#endif
#else
    All.CurrentTurnaroundRadius = turnaround_radius_global;
#endif /* ANALYTIC_TURNAROUND */

    if(ThisTask == 0)
    {
#ifdef ANALYTIC_TURNAROUND
#ifdef COMOVING_DISTORTION
        printf("COMOVING_DISTORTION: comoving turnaround radius = %g\n", All.CurrentTurnaroundRadius);
#else
        printf("SIM/SHEL_CODE adaptive core softening: simulation turnaround radius = %g\n",
                turnaround_radius_global);
        printf("SIM/SHEL_CODE adaptive core softening: analytic turnaround radius   = %g\n",
                All.CurrentTurnaroundRadius);
#endif
#else
        printf("SIM/SHEL_CODE adaptive core softening: current turnaround radius  = %g\n",
                All.CurrentTurnaroundRadius);
#endif /* ANALYTIC TURNAROUND */
        fflush(stdout);
    }

#endif /* SIM_ADAPTIVE_SOFT */

    parallel_sort(rad_data, NumPart, sizeof(struct radius_data), compare_radius);

    /* add up masses to get enclosed mass M(<r) */
    for(i = 1; i < NumPart; i++)
        rad_data[i].enclosed_mass = rad_data[i - 1].enclosed_mass + rad_data[i].enclosed_mass;

    /* get masses from other CPUs */
    masslist = (double *) mymalloc("masslist", NTask * sizeof(double));
    MPI_Allgather(&rad_data[NumPart - 1].enclosed_mass, 1, MPI_DOUBLE, masslist, 1, MPI_DOUBLE, MPI_COMM_WORLD);

    /* add results from other cpus */
    if(ThisTask > 0)
    {
        for(i = 0; i < NumPart; i++)
        {
            for(k = 0; k < ThisTask; k++)
                rad_data[i].enclosed_mass += masslist[k];
        }
    }

#ifdef COMOVING_DISTORTION
    /* subtract background mass */
    for(i = 0; i < NumPart; i++)
        rad_data[i].enclosed_mass -=
            All.Omega0 * 3.0 * All.Hubble * All.Hubble / (8.0 * M_PI * All.G) * (4.0 * M_PI / 3.0) *
            pow(rad_data[i].radius, 3.0);
#endif

    for(i = ndiff; i < NumPart - ndiff; i++)
    {
        /* simple finite difference estimate for derivative */
        rad_data[i].dMdr = (rad_data[i + ndiff].enclosed_mass - rad_data[i - ndiff].enclosed_mass) /
            (rad_data[i + ndiff].radius - rad_data[i - ndiff].radius);
    }

    /* set the remaining derivatives (quick&dirty solution that avoids CPU communication) */
    for(i = 0; i < ndiff; i++)
        rad_data[i].dMdr = rad_data[ndiff].dMdr;

    for(i = NumPart - ndiff; i < NumPart; i++)
        rad_data[i].dMdr = rad_data[NumPart - ndiff - 1].dMdr;


    /* sort back -> associate with particle data structure */
    parallel_sort(rad_data, NumPart, sizeof(struct radius_data), compare_GrNr_SubNr);

    /* write data into particle data */
    for(i = 0; i < NumPart; i++)
    {
        P[i].enclosed_mass = rad_data[i].enclosed_mass - P[i].Mass;
        P[i].dMdr = rad_data[i].dMdr;
    }

    /* get the core softening length */
#ifdef SIM_ADAPTIVE_SOFT
    /* adaptive softening */
    hsoft = All.SofteningHalo * All.CurrentTurnaroundRadius;
    hsoft_tidal = All.SofteningHalo * All.CurrentTurnaroundRadius;
#else
    /* fixed softening */
    hsoft = All.SofteningHalo;
    hsoft_tidal = All.SofteningHalo;
#endif

    /* set the table values, because it is used for the time stepping, softening table contains Plummer equivalent softening length */
    for(i = 0; i < 6; i++)
        All.SofteningTable[i] = hsoft;

    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        if(P[i].radius != 0.0)
        {
            /* radial forces on shell */
            P[i].GravAccel[0] +=
                -All.G * P[i].enclosed_mass / pow(P[i].radius * P[i].radius + hsoft * hsoft, 1.5) * P[i].Pos[0];
            P[i].GravAccel[1] +=
                -All.G * P[i].enclosed_mass / pow(P[i].radius * P[i].radius + hsoft * hsoft, 1.5) * P[i].Pos[1];
            P[i].GravAccel[2] +=
                -All.G * P[i].enclosed_mass / pow(P[i].radius * P[i].radius + hsoft * hsoft, 1.5) * P[i].Pos[2];

#ifdef DISTORTIONTENSORPS
            /* tidal tensor */
            P[i].tidal_tensorps[0][0] +=
                All.G * (-P[i].enclosed_mass / pow(P[i].radius * P[i].radius + hsoft_tidal * hsoft_tidal, 1.5) -
                        P[i].dMdr / pow(P[i].radius * P[i].radius + hsoft_tidal * hsoft_tidal,
                            1.5) * P[i].Pos[0] * P[i].Pos[0] / pow(P[i].radius * P[i].radius +
                                0.0 * hsoft_tidal * hsoft_tidal,
                                0.5) +
                        3 * P[i].enclosed_mass / pow(P[i].radius * P[i].radius + hsoft_tidal * hsoft_tidal,
                            2.5) * P[i].Pos[0] * P[i].Pos[0]);
            P[i].tidal_tensorps[0][1] +=
                All.G * (-0.0 * P[i].enclosed_mass /
                        pow(P[i].radius * P[i].radius + hsoft_tidal * hsoft_tidal,
                            1.5) + -P[i].dMdr / pow(P[i].radius * P[i].radius + hsoft_tidal * hsoft_tidal,
                                1.5) * P[i].Pos[0] * P[i].Pos[1] / pow(P[i].radius *
                                    P[i].radius +
                                    0.0 * hsoft_tidal *
                                    hsoft_tidal,
                                    0.5) +
                        3 * P[i].enclosed_mass / pow(P[i].radius * P[i].radius + hsoft_tidal * hsoft_tidal,
                            2.5) * P[i].Pos[0] * P[i].Pos[1]);
            P[i].tidal_tensorps[0][2] +=
                All.G * (-0.0 * P[i].enclosed_mass /
                        pow(P[i].radius * P[i].radius + hsoft_tidal * hsoft_tidal,
                            1.5) + -P[i].dMdr / pow(P[i].radius * P[i].radius + hsoft_tidal * hsoft_tidal,
                                1.5) * P[i].Pos[0] * P[i].Pos[2] / pow(P[i].radius *
                                    P[i].radius +
                                    0.0 * hsoft_tidal *
                                    hsoft_tidal,
                                    0.5) +
                        3 * P[i].enclosed_mass / pow(P[i].radius * P[i].radius + hsoft_tidal * hsoft_tidal,
                            2.5) * P[i].Pos[0] * P[i].Pos[2]);
            P[i].tidal_tensorps[1][1] +=
                All.G * (-P[i].enclosed_mass / pow(P[i].radius * P[i].radius + hsoft_tidal * hsoft_tidal, 1.5) +
                        -P[i].dMdr / pow(P[i].radius * P[i].radius + hsoft_tidal * hsoft_tidal,
                            1.5) * P[i].Pos[1] * P[i].Pos[1] / pow(P[i].radius * P[i].radius +
                                0.0 * hsoft_tidal * hsoft_tidal,
                                0.5) +
                        3 * P[i].enclosed_mass / pow(P[i].radius * P[i].radius + hsoft_tidal * hsoft_tidal,
                            2.5) * P[i].Pos[1] * P[i].Pos[1]);
            P[i].tidal_tensorps[1][2] +=
                All.G * (-0.0 * P[i].enclosed_mass /
                        pow(P[i].radius * P[i].radius + hsoft_tidal * hsoft_tidal,
                            1.5) + -P[i].dMdr / pow(P[i].radius * P[i].radius + hsoft_tidal * hsoft_tidal,
                                1.5) * P[i].Pos[1] * P[i].Pos[2] / pow(P[i].radius *
                                    P[i].radius +
                                    0.0 * hsoft_tidal *
                                    hsoft_tidal,
                                    0.5) +
                        3 * P[i].enclosed_mass / pow(P[i].radius * P[i].radius + hsoft_tidal * hsoft_tidal,
                            2.5) * P[i].Pos[1] * P[i].Pos[2]);
            P[i].tidal_tensorps[2][2] +=
                All.G * (-P[i].enclosed_mass / pow(P[i].radius * P[i].radius + hsoft_tidal * hsoft_tidal, 1.5) +
                        -P[i].dMdr / pow(P[i].radius * P[i].radius + hsoft_tidal * hsoft_tidal,
                            1.5) * P[i].Pos[2] * P[i].Pos[2] / pow(P[i].radius * P[i].radius +
                                0.0 * hsoft_tidal * hsoft_tidal,
                                0.5) +
                        3 * P[i].enclosed_mass / pow(P[i].radius * P[i].radius + hsoft_tidal * hsoft_tidal,
                            2.5) * P[i].Pos[2] * P[i].Pos[2]);
            P[i].tidal_tensorps[1][0] = P[i].tidal_tensorps[0][1];
            P[i].tidal_tensorps[2][0] = P[i].tidal_tensorps[0][2];
            P[i].tidal_tensorps[2][1] = P[i].tidal_tensorps[1][2];
#endif
        }
    }

    /* free data */
    myfree(masslist);
    myfree(rad_data);

    if(ThisTask == 0)
    {
        printf("done with shell code calculation.\n");
        fflush(stdout);
    }

#endif /* SHELL_CODE */

#ifdef STATICNFW
    double r, m;
    int l;

    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        r = sqrt(P[i].Pos[0] * P[i].Pos[0] + P[i].Pos[1] * P[i].Pos[1] + P[i].Pos[2] * P[i].Pos[2]);
        m = enclosed_mass(r);
#ifdef NFW_DARKFRACTION
        m *= NFW_DARKFRACTION;
#endif
        if(r > 0)
        {
            for(l = 0; l < 3; l++)
                P[i].GravAccel[l] += -All.G * m * P[i].Pos[l] / (r * r * r);

#ifdef DISTORTIONTENSORPS
            double R200 = pow(NFW_M200 * All.G / (100 * All.Hubble * All.Hubble), 1.0 / 3);
            double Rs = R200 / NFW_C;
            double K = All.G * NFW_M200 / (Rs * (log(1 + NFW_C) - NFW_C / (1 + NFW_C)));
            double r_red = r / Rs;
            double x, y, z;

            x = P[i].Pos[0];
            y = P[i].Pos[1];
            z = P[i].Pos[2];

            P[i].tidal_tensorps[0][0] +=
                -(-K * (1.0 / (r * (1 + r_red)) - log(1 + r_red) / (r * r_red)) * (1 / r - x * x / (r * r * r)) -
                        K * (-2.0 / (r * r * (1 + r_red)) - 1.0 / (r * (1 + r_red) * (1 + r_red) * Rs) +
                            2.0 * Rs * log(1 + r_red) / (r * r * r)) * x * x / (r * r));
            P[i].tidal_tensorps[0][1] +=
                -(-K * (1.0 / (r * (1 + r_red)) - log(1 + r_red) / (r * r_red)) * (0 - x * y / (r * r * r)) -
                        K * (-2.0 / (r * r * (1 + r_red)) - 1.0 / (r * (1 + r_red) * (1 + r_red) * Rs) +
                            2.0 * Rs * log(1 + r_red) / (r * r * r)) * x * y / (r * r));
            P[i].tidal_tensorps[0][2] +=
                -(-K * (1.0 / (r * (1 + r_red)) - log(1 + r_red) / (r * r_red)) * (0 - x * z / (r * r * r)) -
                        K * (-2.0 / (r * r * (1 + r_red)) - 1.0 / (r * (1 + r_red) * (1 + r_red) * Rs) +
                            2.0 * Rs * log(1 + r_red) / (r * r * r)) * x * z / (r * r));
            P[i].tidal_tensorps[1][1] +=
                -(-K * (1.0 / (r * (1 + r_red)) - log(1 + r_red) / (r * r_red)) * (1 / r - y * y / (r * r * r)) -
                        K * (-2.0 / (r * r * (1 + r_red)) - 1.0 / (r * (1 + r_red) * (1 + r_red) * Rs) +
                            2.0 * Rs * log(1 + r_red) / (r * r * r)) * y * y / (r * r));
            P[i].tidal_tensorps[1][2] +=
                -(-K * (1.0 / (r * (1 + r_red)) - log(1 + r_red) / (r * r_red)) * (0 - y * z / (r * r * r)) -
                        K * (-2.0 / (r * r * (1 + r_red)) - 1.0 / (r * (1 + r_red) * (1 + r_red) * Rs) +
                            2.0 * Rs * log(1 + r_red) / (r * r * r)) * y * z / (r * r));
            P[i].tidal_tensorps[2][2] +=
                -(-K * (1.0 / (r * (1 + r_red)) - log(1 + r_red) / (r * r_red)) * (1 / r - z * z / (r * r * r)) -
                        K * (-2.0 / (r * r * (1 + r_red)) - 1.0 / (r * (1 + r_red) * (1 + r_red) * Rs) +
                            2.0 * Rs * log(1 + r_red) / (r * r * r)) * z * z / (r * r));

            P[i].tidal_tensorps[1][0] += P[i].tidal_tensorps[0][1];
            P[i].tidal_tensorps[2][0] += P[i].tidal_tensorps[0][2];
            P[i].tidal_tensorps[2][1] += P[i].tidal_tensorps[1][2];
#endif

        }
    }
#endif



#ifdef STATICPLUMMER
    int l;
    double r;


    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        r = sqrt(P[i].Pos[0] * P[i].Pos[0] + P[i].Pos[1] * P[i].Pos[1] + P[i].Pos[2] * P[i].Pos[2]);

        for(l = 0; l < 3; l++)
            P[i].GravAccel[l] += -P[i].Pos[l] / pow(r * r + 1, 1.5);

#ifdef DISTORTIONTENSORPS
        double x, y, z, r2, f, f2;

        x = P[i].Pos[0];
        y = P[i].Pos[1];
        z = P[i].Pos[2];

        r2 = r * r;;
        f = pow(r2 + 1, 1.5);
        f2 = pow(r2 + 1, 2.5);


        P[i].tidal_tensorps[0][0] += -1.0 / f + 3.0 * x * x / f2;
        P[i].tidal_tensorps[0][1] += -0.0 / f + 3.0 * x * y / f2;
        P[i].tidal_tensorps[0][2] += -0.0 / f + 3.0 * x * z / f2;
        P[i].tidal_tensorps[1][1] += -1.0 / f + 3.0 * y * y / f2;
        P[i].tidal_tensorps[1][2] += -0.0 / f + 3.0 * y * z / f2;
        P[i].tidal_tensorps[2][2] += -1.0 / f + 3.0 * z * z / f2;
        P[i].tidal_tensorps[1][0] += P[i].tidal_tensorps[0][1];
        P[i].tidal_tensorps[2][0] += P[i].tidal_tensorps[0][2];
        P[i].tidal_tensorps[2][1] += P[i].tidal_tensorps[1][2];
#endif
    }
#endif



#ifdef STATICHQ
    double r, m, a;
    int l;


    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        r = sqrt(P[i].Pos[0] * P[i].Pos[0] + P[i].Pos[1] * P[i].Pos[1] + P[i].Pos[2] * P[i].Pos[2]);

        a = pow(All.G * HQ_M200 / (100 * All.Hubble * All.Hubble), 1.0 / 3) / HQ_C *
            sqrt(2 * (log(1 + HQ_C) - HQ_C / (1 + HQ_C)));

        m = HQ_M200 * pow(r / (r + a), 2);
#ifdef HQ_DARKFRACTION
        m *= HQ_DARKFRACTION;
#endif
        if(r > 0)
        {
            for(l = 0; l < 3; l++)
                P[i].GravAccel[l] += -All.G * m * P[i].Pos[l] / (r * r * r);

#ifdef DISTORTIONTENSORPS
            double x, y, z, r2, r3, f, f2, f3;

            x = P[i].Pos[0];
            y = P[i].Pos[1];
            z = P[i].Pos[2];

            r2 = r * r;
            r3 = r * r2;
            f = r + a;
            f2 = f * f;
            f3 = f2 * f;


            P[i].tidal_tensorps[0][0] +=
                All.G * (2.0 * HQ_M200 / (r2 * f3) * x * x + HQ_M200 / (r3 * f2) * x * x - HQ_M200 / (r * f2));
            P[i].tidal_tensorps[0][1] +=
                All.G * (2.0 * HQ_M200 / (r2 * f3) * x * y + HQ_M200 / (r3 * f2) * x * y);
            P[i].tidal_tensorps[0][2] +=
                All.G * (2.0 * HQ_M200 / (r2 * f3) * x * z + HQ_M200 / (r3 * f2) * x * z);
            P[i].tidal_tensorps[1][1] +=
                All.G * (2.0 * HQ_M200 / (r2 * f3) * y * y + HQ_M200 / (r3 * f2) * y * y - HQ_M200 / (r * f2));
            P[i].tidal_tensorps[1][2] +=
                All.G * (2.0 * HQ_M200 / (r2 * f3) * y * z + HQ_M200 / (r3 * f2) * y * z);
            P[i].tidal_tensorps[2][2] +=
                All.G * (2.0 * HQ_M200 / (r2 * f3) * z * z + HQ_M200 / (r3 * f2) * z * z - HQ_M200 / (r * f2));
            P[i].tidal_tensorps[1][0] += P[i].tidal_tensorps[0][1];
            P[i].tidal_tensorps[2][0] += P[i].tidal_tensorps[0][2];
            P[i].tidal_tensorps[2][1] += P[i].tidal_tensorps[1][2];
#endif
        }
    }
#endif

#ifdef STATICLP
    double x, y, z, f;

    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        x = P[i].Pos[0];
        y = P[i].Pos[1];
        z = P[i].Pos[2];
        f = LP_RC2 + x * x + y * y / LP_Q2 + z * z / LP_P2;
        if(f > 0)
        {
            P[i].GravAccel[0] += -LP_V02 * x / f;
            P[i].GravAccel[1] += -LP_V02 * y / (LP_Q2 * f);
            P[i].GravAccel[2] += -LP_V02 * z / (LP_P2 * f);

#ifdef DISTORTIONTENSORPS
            double f2;


            f2 = f * f;

            P[i].tidal_tensorps[0][0] += 2.0 * LP_V02 * x * x / f2 - LP_V02 / f;
            P[i].tidal_tensorps[0][1] += 2.0 * LP_V02 * x * y / (LP_Q2 * f2);
            P[i].tidal_tensorps[0][2] += 2.0 * LP_V02 * x * z / (LP_P2 * f2);
            P[i].tidal_tensorps[1][1] += 2.0 * LP_V02 * y * y / (LP_Q2 * LP_Q2 * f2) - LP_V02 / (LP_Q2 * f);
            P[i].tidal_tensorps[1][2] += 2.0 * LP_V02 * y * z / (LP_Q2 * LP_P2 * f2);
            P[i].tidal_tensorps[2][2] += 2.0 * LP_V02 * z * z / (LP_P2 * LP_P2 * f2) - LP_V02 / (LP_P2 * f);
            P[i].tidal_tensorps[1][0] += P[i].tidal_tensorps[0][1];
            P[i].tidal_tensorps[2][0] += P[i].tidal_tensorps[0][2];
            P[i].tidal_tensorps[2][1] += P[i].tidal_tensorps[1][2];

#endif
        }
    }
#endif

#ifdef STATICSM
    double x, y, z, r, r2;

    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        x = P[i].Pos[0];
        y = P[i].Pos[1];
        z = P[i].Pos[2];
        r = sqrt(x * x + y * y + z * z);
        r2 = r * r;
        if(r > 0)
        {
            P[i].GravAccel[0] += -SM_V02 / r2 * (1 - SM_a / r * atan(r / SM_a)) * x;
            P[i].GravAccel[1] += -SM_V02 / r2 * (1 - SM_a / r * atan(r / SM_a)) * y;
            P[i].GravAccel[2] += -SM_V02 / r2 * (1 - SM_a / r * atan(r / SM_a)) * z;


#ifdef DISTORTIONTENSORPS
            double SM_a2 = SM_a * SM_a;


            P[i].tidal_tensorps[0][0] += -SM_V02 / r2 * (1 - SM_a / r * atan(r / SM_a)) +
                1.0 / (SM_a2 + r2) * SM_V02 / (r2 * r2) * (3 * SM_a2 + 2 * r2 -
                        3 * (SM_a2 + r2) * SM_a / r * atan(r / SM_a)) * x * x;
            P[i].tidal_tensorps[0][1] +=
                -0 + 1.0 / (SM_a2 + r2) * SM_V02 / (r2 * r2) * (3 * SM_a2 + 2 * r2 -
                        3 * (SM_a2 +
                            r2) * SM_a / r * atan(r / SM_a)) * x * y;
            P[i].tidal_tensorps[0][2] +=
                -0 + 1.0 / (SM_a2 + r2) * SM_V02 / (r2 * r2) * (3 * SM_a2 + 2 * r2 -
                        3 * (SM_a2 +
                            r2) * SM_a / r * atan(r / SM_a)) * x * z;
            P[i].tidal_tensorps[1][1] +=
                -SM_V02 / r2 * (1 - SM_a / r * atan(r / SM_a)) + 1.0 / (SM_a2 +
                        r2) * SM_V02 / (r2 * r2) * (3 * SM_a2 +
                            2 * r2 -
                            3 * (SM_a2 +
                                r2) *
                            SM_a / r *
                            atan(r /
                                SM_a)) *
                        y * y;
            P[i].tidal_tensorps[1][2] +=
                -0 + 1.0 / (SM_a2 + r2) * SM_V02 / (r2 * r2) * (3 * SM_a2 + 2 * r2 -
                        3 * (SM_a2 +
                            r2) * SM_a / r * atan(r / SM_a)) * y * z;
            P[i].tidal_tensorps[2][2] +=
                -SM_V02 / r2 * (1 - SM_a / r * atan(r / SM_a)) + 1.0 / (SM_a2 +
                        r2) * SM_V02 / (r2 * r2) * (3 * SM_a2 +
                            2 * r2 -
                            3 * (SM_a2 +
                                r2) *
                            SM_a / r *
                            atan(r /
                                SM_a)) *
                        z * z;
            P[i].tidal_tensorps[1][0] += P[i].tidal_tensorps[0][1];
            P[i].tidal_tensorps[2][0] += P[i].tidal_tensorps[0][2];
            P[i].tidal_tensorps[2][1] += P[i].tidal_tensorps[1][2];

#endif
        }
    }
#endif

#ifdef STATICBRANDT
    double r, m;

    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {

        r = sqrt((P[i].Pos[0] - 10.0) * (P[i].Pos[0] - 10.0) + (P[i].Pos[1] - 10.0) * (P[i].Pos[1] - 10.0));

        m = (r * r * r * BRANDT_OmegaBr * BRANDT_OmegaBr) / (1 + (r / BRANDT_R0) * (r / BRANDT_R0));

        /* note there is no acceleration in z */

        if(r > 0)
        {
            for(k = 0; k < 2; k++)
                P[i].GravAccel[k] += -All.G * m * (P[i].Pos[k] - 10.0) / (r * r * r);
        }
    }

#endif



}
