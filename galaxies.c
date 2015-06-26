#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "densitykernel.h"
#include "proto.h"
#include "evaluator.h"

/*! \file blackhole.c
 *  \brief routines for gas accretion onto black holes, and black hole mergers
 */

#ifdef GAL_PART

struct gal_event {
    enum {
        BHEVENT_ACCRETION = 0,
        BHEVENT_SCATTER = 1,
        BHEVENT_SWALLOW = 2, 
    } type;
    double time;
    MyIDType ID;
    union {
        struct blackhole_accretion_event {
            /* for ACCRETION*/
            double pos[3];
            float bhmass;
            float bhmdot;
            float rho_proper;
            float soundspeed;
            float bhvel;
            float gasvel[3];
            float hsml;
        } a;
        struct blackhole_swallow_event { /* for SCATTER and SWALLOW */
            MyIDType ID_swallow;
            float bhmass_before;
            float bhmass_swallow;
            float vrel;   /* unused for SWALLOW */
            float soundspeed; /* unused for SWALLOW */
        } s;
    };
};

struct feedbackdata_in
{
    int NodeList[NODELISTLENGTH];
    MyDouble Pos[3];
    MyFloat Density;
    MyFloat FeedbackWeightSum;
    MyFloat Mdot;
    MyFloat Sfr;
    MyFloat Dt;
    MyFloat Hsml;
    MyFloat Mass;
    MyFloat BH_Mass;
    MyFloat Vel[3];
    MyIDType ID;
    MyFloat AngularMomentum[3];
    MyFloat DiskMassGas;
    MyFloat DiskMassStar;
    MyFloat Mhalo;
};

struct feedbackdata_out
{
    MyFloat BH_MinPotPos[3];
    MyFloat BH_MinPotVel[3];
    MyFloat BH_MinPot;
    short int BH_TimeBinLimit;
    double V1sum[3];
    double V2sum;
    int Ngb;
};

struct swallowdata_in
{
    int NodeList[NODELISTLENGTH];
    MyDouble Pos[3];
    MyFloat Hsml;
    MyFloat BH_Mass;
    MyIDType ID;
};

struct swallowdata_out
{
    MyDouble Mass;
    MyDouble BH_Mass;
    int BH_CountProgs;
};

static void * blackhole_alloc_ngblist();
static void galaxy_starformation_evaluate(int n); /* growth of disk mass */
static void blackhole_postprocess(int n);

static int blackhole_feedback_isactive(int n);
static void blackhole_feedback_reduce(int place, struct feedbackdata_out * remote, int mode);
static void blackhole_feedback_copy(int place, struct feedbackdata_in * I);

static int blackhole_feedback_evaluate(int target, int mode, 
        struct feedbackdata_in * I, 
        struct feedbackdata_out * O, 
        LocalEvaluator * lv, int * ngblist); /* thermal and wind feedback */

static int blackhole_swallow_isactive(int n);
static void blackhole_swallow_reduce(int place, struct swallowdata_out * remote, int mode);
static void blackhole_swallow_copy(int place, struct swallowdata_in * I);

static int blackhole_swallow_evaluate(int target, int mode, 
        struct swallowdata_in * I, 
        struct swallowdata_out * O, 
        LocalEvaluator * lv, int * ngblist); /* eliminating gas particles and merger */

#define BHPOTVALUEINIT 1.0e30

static int N_sph_swallowed, N_BH_swallowed;

static int make_particle_wind(int i, double v);

// NOTES on what we need to include
// galaxy growth & star formation
// inputs:
// Galaxy particle index, Halo Radius, Mass, and Angular momentum, Mass gas in the vacinity (similar to BH), a time step, OLD gas, Mass in stars. 
//
/* double exp_disc(double M, double r) {
   }
   */ 
// empirical wind proportional 
// M_dot_out = M_dot_star
// v_out = v_esc
// AGN feedback
// if M_dot_star > 5 Msol/yr
// E_out = epsilon * M_dot_star * delta_t

// empirical wind proportional
// 

void galaxy_growth(void)
{
    int i, j, k, n, bin;
    int ndone_flag, ndone;
    int ngrp, sendTask, recvTask, place, nexport, nimport, dummy;
    int Ntot_gas_swallowed, Ntot_BH_swallowed;

    walltime_measure("/Misc");
    Evaluator fbev = {0};

    fbev.ev_label = "BH_FEEDBACK";
    fbev.ev_evaluate = (ev_ev_func) blackhole_feedback_evaluate;
    fbev.ev_isactive = blackhole_feedback_isactive;
    fbev.ev_alloc = blackhole_alloc_ngblist;
    fbev.ev_copy = (ev_copy_func) blackhole_feedback_copy;
    fbev.ev_reduce = (ev_reduce_func) blackhole_feedback_reduce;
    fbev.UseNodeList = 1;
    fbev.ev_datain_elsize = sizeof(struct feedbackdata_in);
    fbev.ev_dataout_elsize = sizeof(struct feedbackdata_out);

    Evaluator swev = {0};
    swev.ev_label = "BH_SWALLOW";
    swev.ev_evaluate = (ev_ev_func) blackhole_swallow_evaluate;
    swev.ev_isactive = blackhole_swallow_isactive;
    swev.ev_alloc = blackhole_alloc_ngblist;
    swev.ev_copy = (ev_copy_func) blackhole_swallow_copy;
    swev.ev_reduce = (ev_reduce_func) blackhole_swallow_reduce;
    swev.UseNodeList = 1;
    swev.ev_datain_elsize = sizeof(struct swallowdata_in);
    swev.ev_dataout_elsize = sizeof(struct swallowdata_out);

    if(ThisTask == 0)
    {
        printf("Beginning black-hole accretion\n");
        fflush(stdout);
    }


    /* Let's first compute the Mdot values */
    int Nactive;
    int * queue = ev_get_queue(&fbev, &Nactive);

    for(i = 0; i < Nactive; i ++) {
        int n = queue[i];
        if(!BHP(n).IsCentral) continue;
        galaxy_starformation_evaluate(n);
    }

    /* Now let's invoke the functions that stochasticall swallow gas
     * and deal with black hole mergers.
     */

    if(ThisTask == 0)
    {
        printf("Start swallowing of gas particles and black holes\n");
        fflush(stdout);
    }


    N_sph_swallowed = N_BH_swallowed = 0;


    /* allocate buffers to arrange communication */

    Ngblist = (int *) mymalloc("Ngblist", All.NumThreads * NumPart * sizeof(int));

    /* Let's first spread the feedback energy, 
     * and determine which particles may be swalled by whom */

    ev_run(&fbev);

    /* Now do the swallowing of particles */
    ev_run(&swev);

    myfree(Ngblist);

    MPI_Reduce(&N_sph_swallowed, &Ntot_gas_swallowed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&N_BH_swallowed, &Ntot_BH_swallowed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(ThisTask == 0)
    {
        printf("Accretion done: %d gas particles swallowed, %d BH particles swallowed\n",
                Ntot_gas_swallowed, Ntot_BH_swallowed);
        fflush(stdout);
    }



    for(n = 0; n < TIMEBINS; n++)
    {
        if(TimeBinActive[n])
        {
            TimeBin_BH_mass[n] = 0;
            TimeBin_BH_dynamicalmass[n] = 0;
            TimeBin_BH_Mdot[n] = 0;
            TimeBin_BH_Medd[n] = 0;
        }
    }

    for(i = 0; i < Nactive; i++) {
        int n = queue[i];
        blackhole_postprocess(n);
    }

    myfree(queue);

    double total_mass_real, total_mdoteddington;
    double total_mass_holes, total_mdot;
    double mdot = 0;
    double mass_holes = 0;
    double mass_real = 0;
    double medd = 0;
    for(bin = 0; bin < TIMEBINS; bin++)
        if(TimeBinCount[bin])
        {
            mass_holes += TimeBin_BH_mass[bin];
            mass_real += TimeBin_BH_dynamicalmass[bin];
            mdot += TimeBin_BH_Mdot[bin];
            medd += TimeBin_BH_Medd[bin];
        }

    MPI_Reduce(&mass_holes, &total_mass_holes, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mass_real, &total_mass_real, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mdot, &total_mdot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&medd, &total_mdoteddington, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(ThisTask == 0)
    {
        /* convert to solar masses per yr */
        double mdot_in_msun_per_year =
            total_mdot * (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);

        total_mdoteddington *= 1.0 / ((4 * M_PI * GRAVITY * C * PROTONMASS /
                    (0.1 * C * C * THOMPSON)) * All.UnitTime_in_s);

        fprintf(FdGals, "%g %td %g %g %g %g %g\n",
                All.Time, All.TotN_bh, total_mass_holes, total_mdot, mdot_in_msun_per_year,
                total_mass_real, total_mdoteddington);
        fflush(FdGals);
    }


    fflush(FdGalsDetails);

    walltime_measure("/BH");
}
static double get_Rhalo(double Mhalo) {
    /* proper R halo*/
    double Rref0 = 100.0;
    double Mref = All.Omega0 * 200 * 27.75 * pow(Rref0 / 1000., 3.0)  * (4 * M_PI / 3.);

    double Rhalo = Rref0 * pow(Mhalo / Mref, 0.33333) * All.cf.a; //physical radius
    return Mhalo;
}
static void galaxy_starformation_evaluate(int n) {
    double dt = (P[n].TimeBin ? (1 << P[n].TimeBin) : 0) * All.Timebase_interval / All.cf.hubble;
    //double rho = BHP(n).Density*All.cf.a3inv; //Physical density

    double Rref0 = 100.0;
    double Mref = All.Omega0 * 200 * 27.75 * pow(Rref0 / 1000., 3.0)  * (4 * M_PI / 3.);

    double Mhalo = BHP(n).HostProperty.Mass;
    double Rhalo = get_Rhalo(Mhalo);
    //double Mgas_v = 4 * M_PI / 3. * pow(P[n].Hsml, 3.0) * BHP(n).Density; 

    double FDIFF = 1.0; 
    double ETA = 0.0008333; 
    double A_ACCRETE = 2.0; 
    double RADSCALE = 0.5; 
    double sig_z = 10.0;
    if(dt > 0) {
        double Lambda_denom_fac = 2.0 * sqrt (All.G * Mhalo * Mhalo * Mhalo * Rhalo); 
        double Jhalo = 0;
        int k;
        for(k = 0; k < 3; k ++) {
            double j = BHP(n).HostProperty.AngularMomentum[k];
            Jhalo += j * j;
        }
        Jhalo = sqrt(Jhalo); /* r * a * v_proper = r_proper * v_proper, thus Jhalo is in physical*/

        double Lambda = Jhalo / Lambda_denom_fac;
        Lambda = 0.05;
        double Rgas_c = Lambda * Rhalo ;
        double Mgas_v = 4 * M_PI / 3. * pow(A_ACCRETE*Rgas_c, 3.0) * BHP(n).Density / pow(All.cf.a,3);
        double Rstar = RADSCALE * Rgas_c;
        double Rhalo3 = Rhalo * Rhalo * Rhalo;
        double Sig_gas_c = BHP(n).DiskMassGas / (2.0 * M_PI * Rgas_c * Rgas_c);
        double Sig_star = BHP(n).DiskMassStar / (2.0 * M_PI * Rstar * Rstar);
        double rho_star_disc = Sig_star / (0.54 * Rstar); //Leroy+ 2008 

        double  ms_denom_fac = M_PI * M_PI * Sig_gas_c * Sig_gas_c * All.G; 

        if (BHP(n).Mass_v_old <= 0) {
            BHP(n).GasDiskAccretionRate = 0;
        } else {
            if ((Mgas_v - BHP(n).Mass_v_old) > 0)  {
                BHP(n).GasDiskAccretionRate = (Mgas_v  - BHP(n).Mass_v_old) * sqrt(32.0 * All.G * (Mhalo/Rhalo3) / (3.0 * M_PI)); 
            } else {
                BHP(n).GasDiskAccretionRate = 0;
            }
        }

        if (BHP(n).DiskMassGas <= 0) {
            BHP(n).Sfr = 0;
        } else {
            BHP(n).Sfr = ETA * FDIFF * 0.125 * M_PI 
                * BHP(n).DiskMassGas 
                * Sig_gas_c * All.G 
                * ( 2 - FDIFF + sqrt((2 - FDIFF)*(2 - FDIFF) + 
                            32.0 * sig_z * rho_star_disc / ms_denom_fac));
        }
            //A_ACCRETE * Mgas_v * 
            //sqrt(All.G * BHP(n).Mass / pow(Rgas_c, 3));

        printf("ID = %td Mgas_v = %g Mgas_v_old = %g Sfr = %g "
                "Density = %g, "
                "Rgas_c = %g, "
                "Jhalo = %g, "
                "Lambda = %g, "
                "DiskMassGas = %g, "
                "GasDiskAccretionRate = %g\n",
                P[n].ID, Mgas_v, BHP(n).Mass_v_old, BHP(n).Sfr, BHP(n).Density, Rgas_c, 
                Jhalo, Lambda_denom_fac,
                BHP(n).DiskMassGas,
                BHP(n).GasDiskAccretionRate);

        BHP(n).DiskMassStar += BHP(n).Sfr * dt;
        BHP(n).DiskMassGas += 
            (BHP(n).GasDiskAccretionRate - BHP(n).Sfr) * dt;

        BHP(n).Mass = BHP(n).DiskMassStar + BHP(n).DiskMassGas;
        BHP(n).Mass_v_old = Mgas_v;
    }
}

static void blackhole_postprocess(int n) {
    int k;
    if(BHP(n).accreted_Mass > 0)
    {
        P[n].Mass += BHP(n).accreted_Mass;
        BHP(n).Mass += BHP(n).accreted_BHMass;
        BHP(n).accreted_Mass = 0;
    }
    int bin = P[n].TimeBin;
#pragma omp atomic
    TimeBin_BH_mass[bin] += BHP(n).Mass;
#pragma omp atomic
    TimeBin_BH_dynamicalmass[bin] += P[n].Mass;
#pragma omp atomic
    TimeBin_BH_Mdot[bin] += BHP(n).Sfr;
    if(BHP(n).Mass > 0) {
#pragma omp atomic
        TimeBin_BH_Medd[bin] += BHP(n).Sfr / BHP(n).Mass;
    }
}

static int blackhole_feedback_evaluate(int target, int mode, 
        struct feedbackdata_in * I, 
        struct feedbackdata_out * O, 
        LocalEvaluator * lv, int * ngblist)
{

    int startnode, numngb, k, n, listindex = 0;
    double hsearch;

    int ptypemask = 0;

    ptypemask = 1 + 2 + 4 + 8 + 16 + 32;

    O->BH_TimeBinLimit = -1;
    O->BH_MinPot = BHPOTVALUEINIT;

    startnode = I->NodeList[0];
    listindex ++;
    startnode = Nodes[startnode].u.d.nextnode;	/* open it */

    density_kernel_t kernel;
    density_kernel_t bh_feedback_kernel;
    hsearch = density_decide_hsearch(5, I->Hsml);

    density_kernel_init(&kernel, I->Hsml);
    density_kernel_init(&bh_feedback_kernel, hsearch);

    /* initialize variables before SPH loop is started */

    /* Now start the actual SPH computation for this particle */

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb = ngb_treefind_threads(I->Pos, hsearch, target, &startnode, mode, lv,
                    ngblist, NGB_TREEFIND_ASYMMETRIC, ptypemask);

            if(numngb < 0)
                return numngb;

            for(n = 0; 
                    n < numngb; 
                    (unlock_particle_if_not(ngblist[n], I->ID), n++)
               )
            {
                lock_particle_if_not(ngblist[n], I->ID);
                int j = ngblist[n];

                if(P[j].Mass < 0) continue;

                if(P[j].Type != 5) {
                    if (O->BH_TimeBinLimit <= 0 || O->BH_TimeBinLimit >= P[j].TimeBin) 
                        O->BH_TimeBinLimit = P[j].TimeBin;
                }
                double dx = I->Pos[0] - P[j].Pos[0];
                double dy = I->Pos[1] - P[j].Pos[1];
                double dz = I->Pos[2] - P[j].Pos[2];
#if defined(PERIODIC) 
                dx = NEAREST(dx);
                dy = NEAREST(dy);
                dz = NEAREST(dz);
#endif
                double r2 = dx * dx + dy * dy + dz * dz;

                /* if this option is switched on, we may also encounter dark matter particles or stars */
                if(r2 < kernel.HH)
                {
                    if(P[j].Potential < O->BH_MinPot)
                    {
                        if(P[j].Type == 0 || P[j].Type == 1 || P[j].Type == 4 || P[j].Type == 5)
                        {
                            O->BH_MinPot = P[j].Potential;
                            for(k = 0; k < 3; k++) {
                                O->BH_MinPotPos[k] = P[j].Pos[k];
                                O->BH_MinPotVel[k] = P[j].Vel[k];
                            }
                        }
                    }
                }

                if(P[j].Type == 5 && r2 < kernel.HH)	/* we have a black hole merger */
                {
                    if(I->ID != P[j].ID)
                    {
                        if(P[j].SwallowID < I->ID && P[j].ID < I->ID)
                            P[j].SwallowID = I->ID;
                    }
                }
                if(P[j].Type == 0) {
                    /* BH does not accrete wind */
                    if(SPHP(j).DelayTime > 0) continue;

                    if(r2 < kernel.HH) {
                        /* here we have a gas particle */

                        double r = sqrt(r2);
                        double u = r * kernel.Hinv;
                        double wk = density_kernel_wk(&kernel, u);
                        /* compute accretion probability */
                        double p, w;

                        /*FIXME: prefer low entropy particles */
                        if((I->BH_Mass - I->Mass) > 0 && I->Density > 0)
                            p = (I->BH_Mass - I->Mass) * wk / I->Density;
                        else
                            p = 0;

                        /* compute random number, uniform in [0,1] */
                        w = get_random_number(P[j].ID);
                        if(w < p)
                        {
                            if(P[j].SwallowID < I->ID)
                                P[j].SwallowID = I->ID;
                        }
                    }
                    if(r2 < bh_feedback_kernel.HH && P[j].Mass > 0)
                    {  // THIS IS JUST THE WIND
                        double windeff;
                        double v;
                        if(HAS(All.WindModel, WINDS_SUBGRID)) {
                            /* Here comes the Springel Hernquist 03 wind model */
                            double sm = I->Sfr * I->Dt;
                            double pw = All.WindEfficiency * sm / P[j].Mass;
                            double prob = 1 - exp(-pw);
                            double zero[3] = {0, 0, 0};
                            double Rhalo = get_Rhalo(I->Mhalo);
                            double vesc = sqrt(2 * All.G * I->Mhalo / Rhalo);
                            if(get_random_number(P[j].ID + 2) < prob)
                                make_particle_wind(j, vesc * All.cf.a);
                        } else {
                            abort(); /* only subgrid wind is supported */
                        }
                    }
                    if(r2 < bh_feedback_kernel.HH && P[j].Mass > 0
                            && I->Sfr > 5 / 10.2 /* ~5 Msun/year in code units FIXME*/
                      ) {
                        double r = sqrt(r2);
                        double u = r * bh_feedback_kernel.Hinv;
                        double wk;
                        double mass_j;
                        if(HAS(All.GalaxyFeedbackMethod, GAL_FEEDBACK_MASS)) {
                            mass_j = P[j].Mass;
                        } else {
                            mass_j = P[j].Hsml * P[j].Hsml * P[j].Hsml;
                        }
                        if(HAS(All.GalaxyFeedbackMethod, GAL_FEEDBACK_SPLINE))
                            wk = density_kernel_wk(&bh_feedback_kernel, u);
                        else
                            wk = 1.0;
                        double energy = All.BlackHoleFeedbackFactor * 0.1 * I->Sfr * I->Dt *
                            pow(C / All.UnitVelocity_in_cm_per_s, 2);

                        if(I->FeedbackWeightSum > 0)
                        {
                            SPHP(j).Injected_BH_Energy += (energy * mass_j * wk / I->FeedbackWeightSum);
                        }

                    }
                }
            }
        }

        if(listindex < NODELISTLENGTH)
        {
            startnode = I->NodeList[listindex];
            if(startnode >= 0) {
                startnode = Nodes[startnode].u.d.nextnode;	/* open it */
                listindex ++;
            }
        }
    }

    return 0;
}

static int make_particle_wind(int i, double v) {
    /* v and vmean are in internal units (km/s *a ), not km/s !*/
    /* returns 0 if particle i is converteed to wind. */
    int j;
    /* ok, make the particle go into the wind */
    double dir[3];

    double theta = acos(2 * get_random_number(P[i].ID + 3) - 1);
    double phi = 2 * M_PI * get_random_number(P[i].ID + 4);

    dir[0] = sin(theta) * cos(phi);
    dir[1] = sin(theta) * sin(phi);
    dir[2] = cos(theta);

    double norm = 0;
    for(j = 0; j < 3; j++)
        norm += dir[j] * dir[j];

    norm = sqrt(norm);
    if(get_random_number(P[i].ID + 5) < 0.5)
        norm = -norm;

    if(norm != 0)
    {
        for(j = 0; j < 3; j++)
            dir[j] /= norm;

        fprintf(stdout,//FdSfrDetails,
                "Wind T %g %lu M %g P %g %g %g V %g D %g %g %g\n",
                All.Time, P[i].ID, P[i].Mass, P[i].Pos[0], P[i].Pos[1], P[i].Pos[2],
                v / All.cf.a, dir[0], dir[1], dir[2]);

        for(j = 0; j < 3; j++)
        {
            P[i].Vel[j] += v * dir[j];
            SPHP(i).VelPred[j] += v * dir[j];
        }
        SPHP(i).DelayTime = All.WindFreeTravelLength / (v / All.cf.a);
    }
    return 0;   
}






/** 
 * perform blackhole swallow / merger; 
 * ran only if BH_MERGER or BH_SWALLOWGAS
 * is defined
 */
int blackhole_swallow_evaluate(int target, int mode, 
        struct swallowdata_in * I, 
        struct swallowdata_out * O, 
        LocalEvaluator * lv, int * ngblist)
{
    int startnode, numngb, k, n, listindex = 0;

    int ptypemask = 0;
    ptypemask = 1 + 2 + 4 + 8 + 16 + 32;

    startnode = I->NodeList[0];
    listindex ++;
    startnode = Nodes[startnode].u.d.nextnode;	/* open it */

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb = ngb_treefind_threads(I->Pos, I->Hsml, target, &startnode, 
                    mode, lv, ngblist, NGB_TREEFIND_SYMMETRIC, ptypemask);

            if(numngb < 0)
                return numngb;

            for(n = 0; n < numngb; 
                    (unlock_particle_if_not(ngblist[n], I->ID), n++)
               )
            {
                lock_particle_if_not(ngblist[n], I->ID);
                int j = ngblist[n];
                if(P[j].SwallowID != I->ID) continue;

                if(P[j].Type == 5)	/* we have a black hole merger */
                {
                    struct gal_event event;
                    event.type = BHEVENT_SWALLOW;
                    event.time = All.Time;
                    event.ID = I->ID;
                    event.s.ID_swallow = P[j].ID;
                    event.s.bhmass_before = I->BH_Mass;
                    event.s.bhmass_swallow = BHP(j).Mass;
                    fwrite(&event, sizeof(event), 1, FdGalsDetails);

                    O->Mass += (P[j].Mass);
                    O->BH_Mass += (BHP(j).Mass);

                    O->BH_CountProgs += BHP(j).CountProgs;

                    int bin = P[j].TimeBin;
#pragma omp atomic
                    TimeBin_BH_mass[bin] -= BHP(j).Mass;
#pragma omp atomic
                    TimeBin_BH_dynamicalmass[bin] -= P[j].Mass;
#pragma omp atomic
                    TimeBin_BH_Mdot[bin] -= BHP(j).Sfr;
                    if(BHP(j).Mass > 0) {
#pragma omp atomic
                        TimeBin_BH_Medd[bin] -= BHP(j).Sfr / BHP(j).Mass;
                    }

                    P[j].Mass = 0;
                    BHP(j).Mass = 0;
                    BHP(j).Sfr = 0;

#pragma omp atomic
                    N_BH_swallowed++;
                }

                if(P[j].Type == 0)
                {
                    O->Mass += (P[j].Mass);

                    P[j].Mass = 0;
                    int bin = P[j].TimeBin;
#pragma omp atomic
                    N_sph_swallowed++;
                }
            }
        }
        if(listindex < NODELISTLENGTH)
        {
            startnode = I->NodeList[listindex];
            if(startnode >= 0) {
                startnode = Nodes[startnode].u.d.nextnode;	/* open it */
                listindex++;
            }
        }
    }

    return 0;
}

static int blackhole_feedback_isactive(int n) {
    return (P[n].Type == 5) && (P[n].Mass > 0) && (BHP(n).IsCentral);
}
static void * blackhole_alloc_ngblist() {
    int threadid = omp_get_thread_num();
    return Ngblist + threadid * NumPart;
}
static void blackhole_feedback_reduce(int place, struct feedbackdata_out * remote, int mode) {
    int k;
    if(mode == 0 || BHP(place).MinPot > remote->BH_MinPot)
    {
        BHP(place).MinPot = remote->BH_MinPot;
        for(k = 0; k < 3; k++) {
            BHP(place).MinPotPos[k] = remote->BH_MinPotPos[k];
            BHP(place).MinPotVel[k] = remote->BH_MinPotVel[k];
        }
    }
    if (mode == 0 || 
            BHP(place).TimeBinLimit < 0 || 
            BHP(place).TimeBinLimit > remote->BH_TimeBinLimit) {
        BHP(place).TimeBinLimit = remote->BH_TimeBinLimit;
    }
}

static void blackhole_feedback_copy(int place, struct feedbackdata_in * I) {
    int k;
    for(k = 0; k < 3; k++)
    {
        I->Pos[k] = P[place].Pos[k];
        I->Vel[k] = P[place].Vel[k];
        I->AngularMomentum[k] = BHP(place).HostProperty.AngularMomentum[k];
    }

    I->Hsml = P[place].Hsml;
    I->Mass = P[place].Mass;
    I->BH_Mass = BHP(place).Mass;
    I->Density = BHP(place).Density;
    I->FeedbackWeightSum = BHP(place).FeedbackWeightSum;
    I->DiskMassGas = BHP(place).DiskMassGas;
    I->DiskMassStar = BHP(place).DiskMassStar;
    I->Sfr = BHP(place).Sfr;
    I->Dt =
        (P[place].TimeBin ? (1 << P[place].TimeBin) : 0) * All.Timebase_interval / All.cf.hubble;
    I->ID = P[place].ID;
    I->Mhalo = BHP(place).HostProperty.Mass;
}
static int blackhole_swallow_isactive(int n) {
    return (P[n].Type == 5) && (P[n].SwallowID == 0) && (BHP(n).IsCentral);
}
static void blackhole_swallow_copy(int place, struct swallowdata_in * I) {
    int k;
    for(k = 0; k < 3; k++)
    {
        I->Pos[k] = P[place].Pos[k];
    }
    I->Hsml = P[place].Hsml;
    I->BH_Mass = BHP(place).Mass;
    I->ID = P[place].ID;
}

static void blackhole_swallow_reduce(int place, struct swallowdata_out * remote, int mode) {
    int k;

#define EV_REDUCE(A, B) (A) = (mode==0)?(B):((A) + (B))
    EV_REDUCE(BHP(place).accreted_Mass, remote->Mass);
    EV_REDUCE(BHP(place).accreted_BHMass, remote->BH_Mass);
    EV_REDUCE(BHP(place).CountProgs, remote->BH_CountProgs);
}

void gal_make_one(int index) {
    if(P[index].Type != 0) endrun(7772);

    int child = domain_fork_particle(index);

    P[child].PI = atomic_fetch_and_add(&N_bh, 1);
    P[child].Type = 5;	/* make it a black hole particle */
#ifdef STELLARAGE
    P[child].StellarAge = All.Time;
#endif
    P[child].Mass = All.SeedBlackHoleMass;
    P[index].Mass -= All.SeedBlackHoleMass;
    BHP(child).ID = P[child].ID;
    BHP(child).Mass = All.SeedBlackHoleMass;
    BHP(child).Sfr = 0;
    BHP(child).DiskMassGas = 0;
    BHP(child).DiskMassStar = 0;
    BHP(child).Mass_v_old = -1;
    BHP(child).IsCentral = 1;
    BHP(child).MinPot = BHPOTVALUEINIT;

    BHP(child).CountProgs = 1;
}
#endif
