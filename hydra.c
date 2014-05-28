#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>


#include "allvars.h"
#include "evaluator.h"
#include "proto.h"
#include "densitykernel.h"
#ifdef COSMIC_RAYS
#include "cosmic_rays.h"
#endif
#ifdef MACHNUM
#include "machfinder.h"
#endif

#ifdef JD_DPP
#include "cr_electrons.h"
#endif	

#ifndef DEBUG
#define NDEBUG
#endif
#include <assert.h>

#if defined(MAGNETIC) && defined(SFR)
#define POW_CC 1./3.
#endif

/*! \file hydra.c
 *  \brief Computation of SPH forces and rate of entropy generation
 *
 *  This file contains the "second SPH loop", where the SPH forces are
 *  computed, and where the rate of change of entropy due to the shock heating
 *  (via artificial viscosity) is computed.
 */
struct hydrodata_in
{
#ifndef DONOTUSENODELIST
    int NodeList[NODELISTLENGTH];
#else
    int NodeList[2]; /* At least 2 elements are needed to drive the evaluator */
#endif
#ifdef DENSITY_INDEPENDENT_SPH
    MyFloat EgyRho;
    MyFloat EntVarPred;
#endif

    MyDouble Pos[3];
    MyFloat Vel[3];
    MyFloat Hsml;
    MyFloat Mass;
    MyFloat Density;
    MyFloat Pressure;
    MyFloat F1;
    MyFloat DhsmlDensityFactor;
    int Timestep;

#ifdef PARTICLE_DEBUG
    MyIDType I->ID;			/*!< particle identifier */
#endif

#ifdef JD_VTURB
    MyFloat Vbulk[3];
#endif

#ifdef MAGNETIC
    MyFloat BPred[3];
#ifdef VECT_POTENTIAL
    MyFloat I->Apred[3];
#endif
#ifdef ALFA_OMEGA_DYN
    MyFloat I->alfaomega;
#endif
#ifdef EULER_DISSIPATION
    MyFloat EulerA, EulerB;
#endif
#ifdef TIME_DEP_MAGN_DISP
    MyFloat I->Balpha;
#endif
#ifdef DIVBCLEANING_DEDNER
    MyFloat I->PhiPred;
#endif
#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS)
    MyFloat RotB[3];
#endif
#endif
#ifdef TIME_DEP_ART_VISC
    MyFloat I->alpha;
#endif

#if defined(NAVIERSTOKES)
    MyFloat I->Entropy;
#endif



#ifdef NAVIERSTOKES
    MyFloat I->stressoffdiag[3];
    MyFloat I->stressdiag[3];
    MyFloat I->shear_viscosity;
#endif

#ifdef NAVIERSTOKES_BULK
    MyFloat I->divvel;
#endif

#ifdef EOS_DEGENERATE
    MyFloat dpdr;
#endif

};

struct hydrodata_out
{
    MyLongDouble Acc[3];
    MyLongDouble DtEntropy;
#ifdef ALTERNATIVE_VISCOUS_TIMESTEP
    MyFloat MinViscousDt;
#else
    MyFloat MaxSignalVel;
#endif
#ifdef JD_VTURB
    MyFloat Vrms;
#endif
#if defined(MAGNETIC) && (!defined(EULERPOTENTIALS) || !defined(VECT_POTENTIAL))
    MyFloat DtB[3];
#ifdef DIVBFORCE3
    MyFloat magacc[3];
    MyFloat magcorr[3];
#endif
#ifdef DIVBCLEANING_DEDNER
    MyFloat GradPhi[3];
#endif
#endif
#if defined(EULERPOTENTIALS) && defined(EULER_DISSIPATION)
    MyFloat DtEulerA, DtEulerB;
#endif
#ifdef VECT_POTENTIAL
    MyFloat O->dta[3];
#endif
#if  defined(CR_SHOCK)
    MyFloat CR_EnergyChange[NUMCRPOP];
    MyFloat CR_BaryonFractionChange[NUMCRPOP];
#endif

#ifdef HYDRO_COST_FACTOR
    int Ninteractions;
#endif
};


static int hydro_evaluate(int target, int mode, 
        struct hydrodata_in * I, 
        struct hydrodata_out * O,
        LocalEvaluator * lv, int * ngblist);
static int hydro_isactive(int n);
static void * hydro_alloc_ngblist();
static void hydro_post_process(int i);


static void hydro_copy(int place, struct hydrodata_in * input);
static void hydro_reduce(int place, struct hydrodata_out * result, int mode);

#ifdef MACHNUM
double hubble_a, atime, hubble_a2, fac_mu, fac_vsic_fix, a3inv, fac_egy;
#else
static double hubble_a, atime, hubble_a2, fac_mu, fac_vsic_fix, a3inv, fac_egy;
#endif

/*! This function is the driver routine for the calculation of hydrodynamical
 *  force and rate of change of entropy due to shock heating for all active
 *  particles .
 */
void hydro_force(void)
{
    int _clockid = WALL_HYD;
    Evaluator ev = {0};

    ev.ev_evaluate = (ev_evaluate_func) hydro_evaluate;
    ev.ev_isactive = hydro_isactive;
    ev.ev_alloc = hydro_alloc_ngblist;
    ev.ev_copy = (ev_copy_func) hydro_copy;
    ev.ev_reduce = (ev_reduce_func) hydro_reduce;
#ifdef DONOTUSENODELIST
    ev.UseNodeList = 0;
#else
    ev.UseNodeList = 1;
#endif
    ev.ev_datain_elsize = sizeof(struct hydrodata_in);
    ev.ev_dataout_elsize = sizeof(struct hydrodata_out);

    int i, j, k, ngrp, ndone, ndone_flag;
    int sendTask, recvTask, place;
    double timeall = 0, timenetwork = 0;
    double timecomp, timecomm, timewait, tstart, tend;

    int64_t n_exported = 0;

#ifdef NAVIERSTOKES
    double fac;
#endif

#if (!defined(COOLING) && !defined(CR_SHOCK) && (defined(CR_DISSIPATION) || defined(CR_THERMALIZATION)))
    double utherm;
    double dt;
    int CRpop;
#endif

#if defined(CR_SHOCK)
    double rShockEnergy;
    double rNonRethermalizedEnergy;

#ifndef COOLING
    double utherm, CRpop;
#endif
#endif

#if defined(MAGNETIC) && defined(MAGFORCE)
#if defined(TIME_DEP_MAGN_DISP) || defined(DIVBCLEANING_DEDNER)
    double mu0 = 1;
#endif
#endif

#if defined(HEALPIX)
    double r_new, t[3];
    long ipix;
    int count = 0;
    int count2 = 0;
    int total_count = 0;
    double ded_heal_fac = 0;
#endif

#ifdef NUCLEAR_NETWORK
    double dedt_nuc;
    int nuc_particles = 0;
    int nuc_particles_sum;
#endif

#ifdef WAKEUP
#pragma omp parallel for
    for(i = 0; i < NumPart; i++)
    {
        if(P[i].Type == 0)
            SPHP(i).wakeup = 0;
    }
#endif

    if(All.ComovingIntegrationOn)
    {
        /* Factors for comoving integration of hydro */
        hubble_a = hubble_function(All.Time);
        hubble_a2 = All.Time * All.Time * hubble_a;

        fac_mu = pow(All.Time, 3 * (GAMMA - 1) / 2) / All.Time;

        fac_egy = pow(All.Time, 3 * (GAMMA - 1));

        fac_vsic_fix = hubble_a * pow(All.Time, 3 * GAMMA_MINUS1);

        a3inv = 1 / (All.Time * All.Time * All.Time);
        atime = All.Time;
    }
    else
        hubble_a = hubble_a2 = atime = fac_mu = fac_vsic_fix = a3inv = fac_egy = 1.0;

#if defined(MAGFORCE) && defined(TIME_DEP_MAGN_DISP) || defined(DIVBCLEANING_DEDNER)
#ifndef MU0_UNITY
    mu0 *= (4 * M_PI);
    mu0 /= All.UnitTime_in_s * All.UnitTime_in_s * All.UnitLength_in_cm / All.UnitMass_in_g;
    if(All.ComovingIntegrationOn)
        mu0 /= (All.HubbleParam * All.HubbleParam);
#endif
#endif


    /* allocate buffers to arrange communication */

    Ngblist = (int *) mymalloc("Ngblist", All.NumThreads * NumPart * sizeof(int));

    CPU_Step[CPU_HYDMISC] += walltime_measure(WALL_HYDMISC);

    evaluate_begin(&ev);
    do
    {
        /* do local particles and prepare export list */
        evaluate_primary(&ev);

        n_exported += ev.Nexport;

        evaluate_get_remote(&ev, TAG_HYDRO_A);

        report_memory_usage(&HighMark_sphhydro, "SPH_HYDRO");

        /* now do the particles that were sent to us */

        evaluate_secondary(&ev);

        /* get the result */
        evaluate_reduce_result(&ev, TAG_HYDRO_B);
    }
    while(evaluate_ndone(&ev) < NTask);
    
    evaluate_finish(&ev);

    myfree(Ngblist);


    /* do final operations on results */


#ifdef FLTROUNDOFFREDUCTION
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
        if(P[i].Type == 0)
        {
            SPHP(i).e.DtEntropy = FLT(SPHP(i).e.dDtEntropy);

            for(j = 0; j < 3; j++)
                SPHP(i).a.HydroAccel[j] = FLT(SPHP(i).a.dHydroAccel[j]);
        }
#endif



    int Nactive; 
    int * queue = evaluate_get_queue(&ev, &Nactive);
#pragma omp parallel for if(Nactive > 64) 
    for(i = 0; i < Nactive; i++)
        hydro_post_process(queue[i]);

    myfree(queue);

#if defined(HEALPIX)
    MPI_Allreduce(&count, &total_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    count = 0;
    MPI_Allreduce(&count2, &count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(total_count > 0)
    {
        if(ThisTask == 0)
            printf(" hey %i (%i) particles where freeezed and limit is %f \n", total_count, count,
                    (float) All.TotN_sph / 1000.0);
        if(total_count * 1000.0 > All.TotN_sph)	/*//for normal resolution ~100 */
        {
            if(ThisTask == 0)
                printf(" Next calculation of Healpix\n");
            healpix_halo_cond(All.healpixmap);

        }
        total_count = 0;
        count2 = 0;
        fflush(stdout);
    }
#endif

#ifdef NUCLEAR_NETWORK
    if(ThisTask == 0)
    {
        printf("Doing nuclear network.\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    tstart = second();

    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
        if(P[i].Type == 0)
        {
            /* evaluate network here, but do it only for temperatures > 10^7 K */
            if(SPHP(i).temp > 1e7)
            {
                nuc_particles++;
                network_integrate(SPHP(i).temp, SPHP(i).d.Density * All.UnitDensity_in_cgs, SPHP(i).xnuc,
                        SPHP(i).dxnuc,
                        (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval *
                        All.UnitTime_in_s, &dedt_nuc);
                SPHP(i).e.DtEntropy += dedt_nuc * All.UnitEnergy_in_cgs / All.UnitTime_in_s;
            }
            else
            {
                for(k = 0; k < EOS_NSPECIES; k++)
                {
                    SPHP(i).dxnuc[k] = 0;
                }
            }
        }

    tend = second();
    timenetwork += timediff(tstart, tend);

    MPI_Allreduce(&nuc_particles, &nuc_particles_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(ThisTask == 0)
    {
        printf("Nuclear network done for %d particles.\n", nuc_particles_sum);
    }

    timewait1 += timediff(tend, second());
#endif

#ifdef RT_RAD_PRESSURE
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
        if(P[i].Type == 0)
        {
            if(All.Time != All.TimeBegin)
                for(j = 0; j < N_BINS; j++)
                {
                    for(k = 0; k < 3; k++)
                        SPHP(i).a.HydroAccel[k] += SPHP(i).dn_gamma[j] *
                            (HYDROGEN_MASSFRAC * SPHP(i).d.Density) / (PROTONMASS / All.UnitMass_in_g *
                                    All.HubbleParam) * SPHP(i).n[k][j] * 13.6 *
                            1.60184e-12 / All.UnitEnergy_in_cgs * All.HubbleParam / (C / All.UnitVelocity_in_cm_per_s) /
                            ((P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval) / SPHP(i).d.Density;
                }
        }
#endif

    /* collect some timing information */

    timeall += walltime_measure(-1);

    timecomp = ev.timecomp1 + ev.timecomp2;
    timewait = ev.timewait1 + ev.timewait2;
    timecomm = ev.timecommsumm1 + ev.timecommsumm2;

    CPU_Step[CPU_HYDCOMPUTE] += walltime_add(WALL_HYDCOMPUTE, timecomp);
    CPU_Step[CPU_HYDWAIT] += walltime_add(WALL_HYDWAIT, timewait);
    CPU_Step[CPU_HYDCOMM] += walltime_add(WALL_HYDCOMM, timecomm);
#ifdef NUCLEAR_NETWORK
    CPU_Step[CPU_HYDNETWORK] += walltime_add(WALL_HYDNETWORK, timenetwork);
#endif
    CPU_Step[CPU_HYDMISC] += walltime_add(WALL_HYDMISC, timeall - (timecomp + timewait + timecomm + timenetwork));
}

static void hydro_copy(int place, struct hydrodata_in * input) {
    int k;
    double soundspeed_i;
    for(k = 0; k < 3; k++)
    {
        input->Pos[k] = P[place].Pos[k];
        input->Vel[k] = SPHP(place).VelPred[k];
    }
    input->Hsml = P[place].Hsml;
    input->Mass = P[place].Mass;
    input->Density = SPHP(place).d.Density;
#ifdef DENSITY_INDEPENDENT_SPH
    input->EgyRho = SPHP(place).EgyWtDensity;
    input->EntVarPred = SPHP(place).EntVarPred;
    input->DhsmlDensityFactor = SPHP(place).DhsmlEgyDensityFactor;
#else
    input->DhsmlDensityFactor = SPHP(place).h.DhsmlDensityFactor;
#endif

    input->Pressure = SPHP(place).Pressure;
    input->Timestep = (P[place].TimeBin ? (1 << P[place].TimeBin) : 0);
#ifdef EOS_DEGENERATE
    input->dpdr = SPHP(place).dpdr;
#endif

    /* calculation of F1 */
#ifndef ALTVISCOSITY
#ifndef EOS_DEGENERATE
    soundspeed_i = sqrt(GAMMA * SPHP(place).Pressure / SPHP(place).EOMDensity);
#else
    soundspeed_i = sqrt(SPHP(place).dpdr);
#endif
#ifndef NAVIERSTOKES
    input->F1 = fabs(SPHP(place).v.DivVel) /
        (fabs(SPHP(place).v.DivVel) + SPHP(place).r.CurlVel +
         0.0001 * soundspeed_i / P[place].Hsml / fac_mu);
#else
    input->F1 = fabs(SPHP(place).v.DivVel) /
        (fabs(SPHP(place).v.DivVel) + SPHP(place).u.s.CurlVel +
         0.0001 * soundspeed_i / P[place].Hsml / fac_mu);
#endif

#else
    input->F1 = SPHP(place).v.DivVel;
#endif

#ifdef JD_VTURB
    input->Vbulk[0] = SPHP(place).Vbulk[0];
    input->Vbulk[1] = SPHP(place).Vbulk[1];
    input->Vbulk[2] = SPHP(place).Vbulk[2];
#endif

#ifdef MAGNETIC
    for(k = 0; k < 3; k++)
    {
#ifndef SFR
        input->BPred[k] = SPHP(place).BPred[k];
#else
        input->BPred[k] = SPHP(place).BPred[k] * pow(1.-SPHP(place).XColdCloud,2.*POW_CC);
#endif
#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS)
#ifdef SMOOTH_ROTB
        input->RotB[k] = SPHP(place).SmoothedRotB[k];
#else
        input->RotB[k] = SPHP(place).RotB[k];
#endif
#ifdef SFR
        input->RotB[k] *= pow(1.-SPHP(place).XColdCloud,3.*POW_CC);
#endif
#endif
    }
#ifdef ALFA_OMEGA_DYN
    input->I->alfaomega =
        Sph[place].r.Rot[0] * Sph[place].VelPred[0] + Sph[place].r.Rot[1] * Sph[place].VelPred[1] +
        Sph[place].r.Rot[2] * Sph[place].VelPred[2];
#endif
#if defined(EULERPOTENTIALS) && defined(EULER_DISSIPATION)
    input->EulerA = SPHP(place).EulerA;
    input->EulerB = SPHP(place).EulerB;
#endif
#ifdef VECT_POTENTIAL
    input->I->Apred[0] = SPHP(place).APred[0];
    input->I->Apred[1] = SPHP(place).APred[1];
    input->I->Apred[2] = SPHP(place).APred[2];
#endif
#ifdef DIVBCLEANING_DEDNER
#ifdef SMOOTH_PHI
    input->I->PhiPred = SPHP(place).SmoothPhi;
#else
    input->I->PhiPred = SPHP(place).PhiPred;
#endif
#endif
#endif


#if defined(NAVIERSTOKES)
    input->I->Entropy = SPHP(place).Entropy;
#endif


#ifdef TIME_DEP_ART_VISC
    input->I->alpha = SPHP(place).alpha;
#endif


#ifdef PARTICLE_DEBUG
    input->I->ID = P[place].ID;
#endif

#ifdef NAVIERSTOKES
    for(k = 0; k < 3; k++)
    {
        input->I->stressdiag[k] = SPHP(i).u.s.StressDiag[k];
        input->I->stressoffdiag[k] = SPHP(i).u.s.StressOffDiag[k];
    }
    input->I->shear_viscosity = get_shear_viscosity(i);

#ifdef NAVIERSTOKES_BULK
    input->I->divvel = SPHP(i).u.s.DivVel;
#endif
#endif

#ifdef TIME_DEP_MAGN_DISP
    input->I->Balpha = SPHP(place).Balpha;
#endif
}

static void hydro_reduce(int place, struct hydrodata_out * result, int mode) {
#define REDUCE(A, B) (A) = (mode==0)?(B):((A) + (B))
    int k;

    for(k = 0; k < 3; k++)
    {
        REDUCE(SPHP(place).a.dHydroAccel[k], result->Acc[k]);
    }

    REDUCE(SPHP(place).e.dDtEntropy, result->DtEntropy);

#ifdef HYDRO_COST_FACTOR
    if(All.ComovingIntegrationOn)
        P[place].GravCost += HYDRO_COST_FACTOR * All.Time * result->Ninteractions;
    else
        P[place].GravCost += HYDRO_COST_FACTOR * result->Ninteractions;
#endif

#ifdef ALTERNATIVE_VISCOUS_TIMESTEP
    if(mode == 0 || SPHP(place).MinViscousDt > result->MinViscousDt)
        SPHP(place).MinViscousDt = result->MinViscousDt;
#else
    if(mode == 0 || SPHP(place).MaxSignalVel < result->MaxSignalVel)
        SPHP(place).MaxSignalVel = result->MaxSignalVel;
#endif

#ifdef JD_VTURB
    REDUCE(SPHP(place).Vrms, result->Vrms); 
#endif

#if defined(MAGNETIC) && ( !defined(EULERPOTENTIALS) || !defined(VECT_POTENTIAL) )
    for(k = 0; k < 3; k++)
        REDUCE(SPHP(place).DtB[k], result->DtB[k]);
#endif
#ifdef DIVBFORCE3
    for(k = 0; k < 3; k++)
        REDUCE(SPHP(place).magacc[k], result->magacc[k]);
    for(k = 0; k < 3; k++)
        REDUCE(SPHP(place).magcorr[k], result->magcorr[k]);
#endif
#ifdef DIVBCLEANING_DEDNER
    for(k = 0; k < 3; k++)
        REDUCE(SPHP(place).GradPhi[k], result->GradPhi[k]);
#endif
#if VECT_POTENTIAL
    for(k = 0; k < 3; k++)
        REDUCE(SPHP(place).DtA[k], result->dta[k]);
#endif
#if defined(EULERPOTENTIALS) && defined(EULER_DISSIPATION)
    REDUCE(SPHP(place).DtEulerA, result->DtEulerA);
    REDUCE(SPHP(place).DtEulerB, result->DtEulerB);
#endif
}


/*! This function is the 'core' of the SPH force computation. A target
 *  particle is specified which may either be local, or reside in the
 *  communication buffer.
 */
static int hydro_evaluate(int target, int mode, 
        struct hydrodata_in * I, 
        struct hydrodata_out * O,
        LocalEvaluator * lv, int * ngblist)
{
    int startnode, numngb, listindex = 0;
    int j, k, n; 

    int ninteractions = 0;
    int nnodesinlist = 0;

    double p_over_rho2_i, p_over_rho2_j, soundspeed_i, soundspeed_j;

    density_kernel_t kernel_i;
    density_kernel_t kernel_j;

    startnode = I->NodeList[0];
    listindex ++;
    startnode = Nodes[startnode].u.d.nextnode;	/* open it */

#ifndef EOS_DEGENERATE
#ifdef DENSITY_INDEPENDENT_SPH
    soundspeed_i = sqrt(GAMMA * I->Pressure / I->EgyRho);
#else
    soundspeed_i = sqrt(GAMMA * I->Pressure / I->Density);
#endif
#else
    soundspeed_i = sqrt(I->dpdr);
#endif

    /* initialize variables before SPH loop is started */

    O->Acc[0] = O->Acc[1] = O->Acc[2] = O->DtEntropy = 0;
    density_kernel_init(&kernel_i, I->Hsml);


#ifdef MAGNETIC
    double mu0_1 = 1;
#ifndef MU0_UNITY
    mu0_1 /= (4 * M_PI);
    mu0_1 *= All.UnitTime_in_s * All.UnitTime_in_s * All.UnitLength_in_cm / (All.UnitMass_in_g);
    if(All.ComovingIntegrationOn)
        mu0_1 /= (All.HubbleParam * All.HubbleParam);

#endif
#ifdef MAGFORCE
    double magfac_i_base = 1 / (I->Density * rho);
#ifndef MU0_UNITY
    magfac_i_base /= (4 * M_PI);
#endif
#ifdef CORRECTBFRC
    magfac_i_base *= I->DhsmlDensityFactor;
#endif
    double mm_i[3][3];
    for(k = 0, b2_i = 0; k < 3; k++)
    {
        b2_i += I->Bpred[k] * I->Bpred[k];
        for(l = 0; l < 3; l++)
            mm_i[k][l] = I->Bpred[k] * I->Bpred[l];
    }
    for(k = 0; k < 3; k++)
        mm_i[k][k] -= 0.5 * b2_i;
#ifdef MAGNETIC_SIGNALVEL
#ifdef ALFVEN_VEL_LIMITER
    double vcsa2_i = soundspeed_i * soundspeed_i +
        DMIN(mu0_1 * b2_i / I->Density, ALFVEN_VEL_LIMITER * soundspeed_i * soundspeed_i);
#else
    double vcsa2_i = soundspeed_i * soundspeed_i + mu0_1 * b2_i / I->Density;
#endif
#endif
#endif /* end of MAGFORCE */
#endif /* end of MAGNETIC */

#ifndef TRADITIONAL_SPH_FORMULATION
#ifdef DENSITY_INDEPENDENT_SPH
    p_over_rho2_i = I->Pressure / (I->EgyRho * I->EgyRho);
#else
    p_over_rho2_i = I->Pressure / (I->Density * I->Density);
#endif
#else
    p_over_rho2_i = I->Pressure / (I->Density * I->Density);
#endif

#ifdef ALTERNATIVE_VISCOUS_TIMESTEP
    O->MinViscousDt = 1.0e32;
#else
    O->MaxSignalVel = soundspeed_i;
#endif


    /* Now start the actual SPH computation for this particle */

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb =
                ngb_treefind_threads(I->Pos, I->Hsml, target, &startnode, 
                        mode, lv, ngblist, NGB_TREEFIND_SYMMETRIC, 1); /* gas only 1 << 0 */

            if(numngb < 0)
                return numngb;

            for(n = 0; n < numngb; n++)
            {
                j = ngblist[n];

                ninteractions++;

#ifdef BLACK_HOLES
                if(P[j].Mass == 0)
                    continue;
#endif

#ifdef WINDS
#ifdef NOWINDTIMESTEPPING
                if(HAS(All.WindModel, WINDS_DECOUPLE_SPH)) {
                    if(P[j].Type == 0)
                        if(SPHP(j).DelayTime > 0)	/* ignore the wind particles */
                            continue;
                }
#endif
#endif
                double dx = I->Pos[0] - P[j].Pos[0];
                double dy = I->Pos[1] - P[j].Pos[1];
                double dz = I->Pos[2] - P[j].Pos[2];
#ifdef PERIODIC			/*  now find the closest image in the given box size  */
                dx = NEAREST_X(dx);
                dy = NEAREST_Y(dy);
                dz = NEAREST_Z(dz);
#endif
                double r2 = dx * dx + dy * dy + dz * dz;
                density_kernel_init(&kernel_j, P[j].Hsml);
                if(r2 > 0 && (r2 < kernel_i.HH || r2 < kernel_j.HH))
                {
                    double r = sqrt(r2);
                    p_over_rho2_j = SPHP(j).Pressure / (SPHP(j).EOMDensity * SPHP(j).EOMDensity);

#ifndef EOS_DEGENERATE
#ifdef DENSITY_INDEPENDENT_SPH
                    soundspeed_j = sqrt(GAMMA * SPHP(j).Pressure / SPHP(j).EOMDensity);
#else
                    soundspeed_j = sqrt(GAMMA * p_over_rho2_j * SPHP(j).d.Density);
#endif
#else
                    soundspeed_j = sqrt(SPHP(j).dpdr);
#endif

                    double dvx = I->Vel[0] - SPHP(j).VelPred[0];
                    double dvy = I->Vel[1] - SPHP(j).VelPred[1];
                    double dvz = I->Vel[2] - SPHP(j).VelPred[2];
                    double vdotr = dx * dvx + dy * dvy + dz * dvz;
                    double rho_ij = 0.5 * (I->Density + SPHP(j).d.Density);
                    double vdotr2;
                    if(All.ComovingIntegrationOn)
                        vdotr2 = vdotr + hubble_a2 * r2;
                    else
                        vdotr2 = vdotr;

                    double dwk_i = density_kernel_dwk(&kernel_i, r * kernel_i.Hinv);
                    double dwk_j = density_kernel_dwk(&kernel_j, r * kernel_j.Hinv);

#ifdef JD_VTURB
                    if ( I->Hsml >= P[j].Hsml)  /* Make sure j is inside targets hsml */
                        O->Vrms += (SPHP(j).VelPred[0]-I->Vbulk[0])*(SPHP(j).VelPred[0]-vBulk[0]) 
                            + (SPHP(j).VelPred[1]-I->Vbulk[1])*(SPHP(j).VelPred[1]-vBulk[1]) 
                            + (SPHP(j).VelPred[2]-I->Vbulk[2])*(SPHP(j).VelPred[2]-vBulk[2]);
#endif

#ifdef MAGNETIC
#ifndef SFR
                    double dBx = I->Bpred[0] - SPHP(j).BPred[0];
                    double dBy = I->Bpred[1] - SPHP(j).BPred[1];
                    double dBz = I->Bpred[2] - SPHP(j).BPred[2];
#else
                    double dBx = I->Bpred[0] - SPHP(j).BPred[0] * pow(1.-SPHP(j).XColdCloud,2.*POW_CC);
                    double dBy = I->Bpred[1] - SPHP(j).BPred[1] * pow(1.-SPHP(j).XColdCloud,2.*POW_CC);
                    double dBz = I->Bpred[2] - SPHP(j).BPred[2] * pow(1.-SPHP(j).XColdCloud,2.*POW_CC);
#endif

                    double magfac = P[j].Mass / r;	/* we moved 'dwk_i / I->Density' down ! */
                    magfac *= 1.0 / All.cf.hubbla_a2;
                    /* last factor takes care of all cosmological prefactor */
#ifdef CORRECTDB
                    magfac *= I->DhsmlDensityFactor;
#endif

#if defined(MAGNETIC_DISSIPATION) || defined(DIVBCLEANING_DEDNER) || defined(EULER_DISSIPATION) || defined(MAGNETIC_DIFFUSION)
                    double magfac_sym = magfac * (dwk_i + dwk_j) * 0.5;
#endif
#ifdef MAGNETIC_DISSIPATION
#ifdef TIME_DEP_MAGN_DISP
                    double Balpha_ij = 0.5 * (I->Balpha + SPHP(j).Balpha);
#else
                    double Balpha_ij = All.ArtMagDispConst;
#endif
#endif

                    magfac *= dwk_i / I->Density;
#if VECT_POTENTIAL
                    O->dta[0] +=
                        P[j].Mass * dwk_i / r * (I->Apred[0] -
                                SPHP(j).APred[0]) * dx * I->Vel[0] / (I->Density * All.cf.hubble_a2);
                    O->dta[1] +=
                        P[j].Mass * dwk_i / r * (I->Apred[1] -
                                SPHP(j).APred[1]) * dy * I->Vel[1] / (I->Density * All.cf.hubble_a2);
                    O->dta[2] +=
                        P[j].Mass * dwk_i / r * (I->Apred[2] -
                                SPHP(j).APred[2]) * dz * I->Vel[2] / (I->Density * All.cf.hubble_a2);
                    O->dta[0] +=
                        P[j].Mass * dwk_i / r * ((I->Apred[0] - SPHP(j).APred[0]) * dx * I->Vel[0] +
                                (I->Apred[0] - SPHP(j).APred[0]) * dy * I->Vel[1] + (Apred[0] -
                                    SPHP(j).
                                    APred[0]) *
                                dz * I->Vel[2]) / (I->Density * All.cf.hubble_a2);

#endif
#if ( !defined(EULERPOTENTIALS) || !defined(VECT_POTENTIAL) )
                    O->DtB[0] +=
                        magfac * ((I->Bpred[0] * dvy - I->Bpred[1] * dvx) * dy +
                                (I->Bpred[0] * dvz - I->Bpred[2] * dvx) * dz);
                    O->DtB[1] +=
                        magfac * ((I->Bpred[1] * dvz - I->Bpred[2] * dvy) * dz +
                                (I->Bpred[1] * dvx - I->Bpred[0] * dvy) * dx);
                    O->DtB[2] +=
                        magfac * ((I->Bpred[2] * dvx - I->Bpred[0] * dvz) * dx +
                                (I->Bpred[2] * dvy - I->Bpred[1] * dvz) * dy);
#endif
#ifdef MAGNETIC_DIFFUSION  
                    double magfac_diff = (All.MagneticEta + All.MagneticEta) * magfac_sym / (rho_ij * rho_ij);
                    O->DtB[0] += magfac_diff * I->Density * dBx;
                    O->DtB[1] += magfac_diff * I->Density * dBy;
                    O->DtB[2] += magfac_diff * I->Density * dBz;
#ifdef MAGNETIC_DIFFUSION_HEAT
                    magfac_diff *= All.cf.hubble_a2 * All.cf.a * All.cf.a * All.cf.a;
                    O->DtEntropy -= 0.5 * magfac_diff * mu0_1 * (dBx * dBx + dBy * dBy + dBz * dBz);
#endif
#endif
#ifdef MAGFORCE
                    double magfac_j = 1 / (SPHP(j).d.Density * SPHP(j).d.Density);
#ifndef MU0_UNITY
                    magfac_j /= (4 * M_PI);
#endif
#ifdef CORRECTBFRC
                    magfac_j *= dwk_j * SPHP(j).h.DhsmlDensityFactor;
                    double magfac_i = dwk_i * magfac_i_base;
#else
                    double magfac_i = magfac_i_base;
#endif
                    double b2_j = 0;
                    double mm_j[3][3];
                    for(k = 0; k < 3; k++)
                    {
#ifndef SFR
                        b2_j += SPHP(j).BPred[k] * SPHP(j).BPred[k];
                        for(l = 0; l < 3; l++)
                            mm_j[k][l] = SPHP(j).BPred[k] * SPHP(j).BPred[l];
#else
                        b2_j += SPHP(j).BPred[k] * SPHP(j).BPred[k] * pow(1.-SPHP(j).XColdCloud,4.*POW_CC);
                        for(l = 0; l < 3; l++)
                            mm_j[k][l] = SPHP(j).BPred[k] * SPHP(j).BPred[l] * pow(1.-SPHP(j).XColdCloud,4.*POW_CC);
#endif
                    }
                    for(k = 0; k < 3; k++)
                        mm_j[k][k] -= 0.5 * b2_j;

#ifdef DIVBCLEANING_DEDNER
                    double phifac = magfac_sym * I->Density / rho_ij;
#ifndef SFR
#ifdef SMOOTH_PHI
                    phifac *= (I->PhiPred - SPHP(j).SmoothPhi) / (rho_ij);
#else
                    phifac *= (I->PhiPred - SPHP(j).PhiPred) / (rho_ij);
#endif
#else /* SFR */ 
#ifdef SMOOTH_PHI
                    phifac *= (I->PhiPred - SPHP(j).SmoothPhi * pow(1.-SPHP(j).XColdCloud,POW_CC)) / (rho_ij);
#else
                    phifac *= (I->PhiPred - SPHP(j).PhiPred   * pow(1.-SPHP(j).XColdCloud,POW_CC)) / (rho_ij);
#endif 
#endif /* SFR */

                    O->GradPhi[0]+=phifac *dx;
                    O->GradPhi[1]+=phifac *dy;
                    O->GradPhi[2]+=phifac *dz;
#endif
#ifdef MAGNETIC_SIGNALVEL
#ifdef ALFVEN_VEL_LIMITER
                    double vcsa2_j = soundspeed_j * soundspeed_j +
                        DMIN(mu0_1 * b2_j / SPHP(j).d.Density,
                                ALFVEN_VEL_LIMITER * soundspeed_j * soundspeed_j);
#else
                    double vcsa2_j = soundspeed_j * soundspeed_j + mu0_1 * b2_j / SPHP(j).d.Density;
#endif
#ifndef SFR
                    double Bpro2_j = (SPHP(j).BPred[0] * dx + SPHP(j).BPred[1] * dy + SPHP(j).BPred[2] * dz) / r;
#else
                    double Bpro2_j = (SPHP(j).BPred[0] * dx + SPHP(j).BPred[1] * dy + SPHP(j).BPred[2] * dz) * pow(1.-SPHP(j).XColdCloud,2.*POW_CC) / r;
#endif
                    Bpro2_j *= Bpro2_j;

                    double magneticspeed_j = sqrt(vcsa2_j +
                            sqrt(DMAX((vcsa2_j * vcsa2_j -
                                        4 * soundspeed_j * soundspeed_j * Bpro2_j
                                        * mu0_1 / SPHP(j).d.Density), 0))) / 1.4142136;
                    double Bpro2_i = (I->Bpred[0] * dx + I->Bpred[1] * dy + I->Bpred[2] * dz) / r;
                    double Bpro2_i *= Bpro2_i;
                    double magneticspeed_i = sqrt(vcsa2_i +
                            sqrt(DMAX((vcsa2_i * vcsa2_i -
                                        4 * soundspeed_i * soundspeed_i * Bpro2_i
                                        * mu0_1 / I->Density), 0))) / 1.4142136;
#endif
#ifdef MAGNETIC_DISSIPATION
                    double dTu_diss_b = -magfac_sym * Balpha_ij * (dBx * dBx + dBy * dBy + dBz * dBz);
#endif
#ifdef CORRECTBFRC
                    magfac = P[j].Mass / r;
#else
                    magfac = P[j].Mass * 0.5 * (dwk_i + dwk_j) / r;
#endif
                    if(All.ComovingIntegrationOn)
                        magfac *= pow(All.Time, 3 * GAMMA);
                    /* last factor takes care of all cosmological prefactor */
#ifndef MU0_UNITY
                    magfac *= All.UnitTime_in_s * All.UnitTime_in_s *
                        All.UnitLength_in_cm / All.UnitMass_in_g;
                    if(All.ComovingIntegrationOn)
                        magfac /= (All.HubbleParam * All.HubbleParam);
                    /* take care of B unit conversion into GADGET units ! */
#endif
                    for(k = 0; k < 3; k++)
#ifndef DIVBFORCE3
                        O->Acc[k] +=
#else
                            O->magacc[k]+=
#endif
                            magfac * ((mm_i[k][0] * magfac_i + mm_j[k][0] * magfac_j) * dx +
                                    (mm_i[k][1] * magfac_i + mm_j[k][1] * magfac_j) * dy +
                                    (mm_i[k][2] * magfac_i + mm_j[k][2] * magfac_j) * dz);
#if defined(DIVBFORCE) && !defined(DIVBFORCE3)
                    for(k = 0; k < 3; k++)
                        O->Acc[k] -=
#ifndef SFR
                            magfac * I->Bpred[k] *(((I->Bpred[0]) * magfac_i + (SPHP(j).BPred[0]) * magfac_j) * dx
                                    + ((I->Bpred[1]) * magfac_i + (SPHP(j).BPred[1]) * magfac_j) * dy
                                    + ((I->Bpred[2]) * magfac_i + (SPHP(j).BPred[2]) * magfac_j) * dz);
#else
                    magfac * (	((I->Bpred[k] * I->Bpred[0]) * magfac_i + (I->Bpred[k] * SPHP(j).BPred[0] * pow(1.-SPHP(j).XColdCloud,2.*POW_CC)) * magfac_j) * dx
                            +   ((I->Bpred[k] * I->Bpred[1]) * magfac_i + (I->Bpred[k] * SPHP(j).BPred[1] * pow(1.-SPHP(j).XColdCloud,2.*POW_CC)) * magfac_j) * dy
                            +   ((I->Bpred[k] * I->Bpred[2]) * magfac_i + (I->Bpred[k] * SPHP(j).BPred[2] * pow(1.-SPHP(j).XColdCloud,2.*POW_CC)) * magfac_j) * dz);
#endif
#endif
#if defined(DIVBFORCE3) && !defined(DIVBFORCE)
                    for(k = 0; k < 3; k++)
                        O->magcorr[k] +=
#ifndef SFR
                            magfac * I->Bpred[k] *(((I->Bpred[0]) * magfac_i + (SPHP(j).BPred[0]) * magfac_j) * dx
                                    + ((I->Bpred[1]) * magfac_i + (SPHP(j).BPred[1]) * magfac_j) * dy
                                    + ((I->Bpred[2]) * magfac_i + (SPHP(j).BPred[2]) * magfac_j) * dz);
#else
                    magfac * I->Bpred[k] *(((I->Bpred[0]) * magfac_i + (SPHP(j).BPred[0] * pow(1.-SPHP(j).XColdCloud,2.*POW_CC) ) * magfac_j) * dx
                            + ((I->Bpred[1]) * magfac_i + (SPHP(j).BPred[1] * pow(1.-SPHP(j).XColdCloud,2.*POW_CC) ) * magfac_j) * dy
                            + ((I->Bpred[2]) * magfac_i + (SPHP(j).BPred[2] * pow(1.-SPHP(j).XColdCloud,2.*POW_CC) ) * magfac_j) * dz);
#endif
#endif
#endif /* end MAG FORCE   */
#ifdef ALFA_OMEGA_DYN // Known Bug
                    O->DtB[0] += magfac * I->alfaomega * All.Tau_A0 / 3.0 * (dBy * dz - dBy * dy);
                    O->DtB[1] += magfac * I->alfaomega * All.Tau_A0 / 3.0 * (dBz * dx - dBx * dz);
                    O->DtB[2] += magfac * I->alfaomega * All.Tau_A0 / 3.0 * (dBx * dy - dBy * dx);
#endif
#endif /* end of MAGNETIC */

#ifndef MAGNETIC_SIGNALVEL
                    double vsig = soundspeed_i + soundspeed_j;
#else
                    double vsig = magneticspeed_i + magneticspeed_j;
#endif


#ifndef ALTERNATIVE_VISCOUS_TIMESTEP
                    if(vsig > O->MaxSignalVel)
                        O->MaxSignalVel = vsig;
#endif

                    double visc = 0;

                    if(vdotr2 < 0)	/* ... artificial viscosity visc is 0 by default*/
                    {
#ifndef ALTVISCOSITY
#ifndef CONVENTIONAL_VISCOSITY
                        double mu_ij = fac_mu * vdotr2 / r;	/* note: this is negative! */
#else
                        double c_ij = 0.5 * (soundspeed_i + soundspeed_j);
                        double h_ij = 0.5 * (I->Hsml + P[j].Hsml);
                        double mu_ij = fac_mu * h_ij * vdotr2 / (r2 + 0.0001 * h_ij * h_ij);
#endif
#ifdef MAGNETIC
                        vsig -= 1.5 * mu_ij;
#else
                        vsig -= 3 * mu_ij;
#endif


#ifndef ALTERNATIVE_VISCOUS_TIMESTEP
                        if(vsig > O->MaxSignalVel)
                            O->MaxSignalVel = vsig;
#endif

#ifndef NAVIERSTOKES
                        double f2 =
                            fabs(SPHP(j).v.DivVel) / (fabs(SPHP(j).v.DivVel) + SPHP(j).r.CurlVel +
                                    0.0001 * soundspeed_j / fac_mu / P[j].Hsml);
#else
                        double f2 =
                            fabs(SPHP(j).v.DivVel) / (fabs(SPHP(j).v.DivVel) + SPHP(j).u.s.CurlVel +
                                    0.0001 * soundspeed_j / fac_mu / P[j].Hsml);
#endif

#ifdef NO_SHEAR_VISCOSITY_LIMITER
                        I->F1 = f2 = 1;
#endif
#ifdef TIME_DEP_ART_VISC
                        double BulkVisc_ij = 0.5 * (I->alpha + SPHP(j).alpha);
#else
                        double BulkVisc_ij = All.ArtBulkViscConst;
#endif

#ifndef CONVENTIONAL_VISCOSITY
                        visc = 0.25 * BulkVisc_ij * vsig * (-mu_ij) / rho_ij * (I->F1 + f2);
#else
                        visc =
                            (-BulkVisc_ij * mu_ij * c_ij + 2 * BulkVisc_ij * mu_ij * mu_ij) /
                            rho_ij * (I->F1 + f2) * 0.5;
#endif

#else /* start of ALTVISCOSITY block */
                        double mu_i;
                        if(I->F1 < 0)
                            mu_i = I->Hsml * fabs(I->F1);	/* f1 hold here the velocity divergence of particle i */
                        else
                            mu_i = 0;
                        if(SPHP(j).v.DivVel < 0)
                            mu_j = P[j].Hsml * fabs(SPHP(j).v.DivVel);
                        else
                            mu_j = 0;
                        visc = All.ArtBulkViscConst * ((soundspeed_i + mu_i) * mu_i / I->Density +
                                (soundspeed_j + mu_j) * mu_j / SPHP(j).d.Density);
#endif /* end of ALTVISCOSITY block */


                        /* .... end artificial viscosity evaluation */
                        /* now make sure that viscous acceleration is not too large */
#ifdef ALTERNATIVE_VISCOUS_TIMESTEP
                        if(visc > 0)
                        {
                            dt = fac_vsic_fix * vdotr2 /
                                (0.5 * (I->Mass + P[j].Mass) * (dwk_i + dwk_j) * r * visc);

                            dt /= All.cf.hubble;

                            if(dt < O->MinViscousDt)
                                O->MinViscousDt = dt;
                        }
#endif

#ifndef NOVISCOSITYLIMITER
                        double dt =
                            2 * IMAX(I->Timestep,
                                    (P[j].TimeBin ? (1 << P[j].TimeBin) : 0)) * All.Timebase_interval;
                        if(dt > 0 && (dwk_i + dwk_j) < 0)
                        {
#ifdef BLACK_HOLES
                            if((I->Mass + P[j].Mass) > 0)
#endif
                                visc = DMIN(visc, 0.5 * fac_vsic_fix * vdotr2 /
                                        (0.5 * (I->Mass + P[j].Mass) * (dwk_i + dwk_j) * r * dt));
                        }
#endif
                    }
                    double hfc_visc = 0.5 * P[j].Mass * visc * (dwk_i + dwk_j) / r;
#ifndef TRADITIONAL_SPH_FORMULATION

#ifdef DENSITY_INDEPENDENT_SPH
                    double hfc = hfc_visc;
                    /* leading-order term */
                    hfc += P[j].Mass *
                        (dwk_i*p_over_rho2_i*SPHP(j).EntVarPred/I->EntVarPred +
                         dwk_j*p_over_rho2_j*I->EntVarPred/SPHP(j).EntVarPred) / r;

                    /* enable grad-h corrections only if contrastlimit is non negative */
                    if(All.DensityContrastLimit >= 0) {
                        double r1 = I->EgyRho / I->Density;
                        double r2 = SPHP(j).EgyWtDensity / SPHP(j).d.Density;
                        if(All.DensityContrastLimit > 0) {
                            /* apply the limit if it is enabled > 0*/
                            if(r1 > All.DensityContrastLimit) {
                                r1 = All.DensityContrastLimit;
                            }
                            if(r2 > All.DensityContrastLimit) {
                                r2 = All.DensityContrastLimit;
                            }
                        }
                        /* grad-h corrections */
                        /* I->DhsmlDensityFactor is actually EgyDensityFactor */
                        hfc += P[j].Mass *
                            (dwk_i*p_over_rho2_i*r1*I->DhsmlDensityFactor +
                             dwk_j*p_over_rho2_j*r2*SPHP(j).DhsmlEgyDensityFactor) / r;
                    }
#else
                    /* Formulation derived from the Lagrangian */
                    double hfc = hfc_visc + P[j].Mass * (p_over_rho2_i *I->DhsmlDensityFactor * dwk_i 
                            + p_over_rho2_j * SPHP(j).h.DhsmlDensityFactor * dwk_j) / r;
#endif
#else
                    double hfc = hfc_visc +
                        0.5 * P[j].Mass * (dwk_i + dwk_j) / r * (p_over_rho2_i + p_over_rho2_j);

                    /* hfc_egy = 0.5 * P[j].Mass * (dwk_i + dwk_j) / r * (p_over_rho2_i + p_over_rho2_j); */
                    double hfc_egy = P[j].Mass * (dwk_i + dwk_j) / r * (p_over_rho2_i);
#endif

#ifdef WINDS
                    if(HAS(All.WindModel, WINDS_DECOUPLE_SPH)) {
                        if(P[j].Type == 0)
                            if(SPHP(j).DelayTime > 0)	/* No force by wind particles */
                            {
                                hfc = hfc_visc = 0;
                            }
                    }
#endif

#ifndef NOACCEL
                    O->Acc[0] += FLT(-hfc * dx);
                    O->Acc[1] += FLT(-hfc * dy);
                    O->Acc[2] += FLT(-hfc * dz);
#endif

#if !defined(EOS_DEGENERATE) && !defined(TRADITIONAL_SPH_FORMULATION)
                    O->DtEntropy += FLT(0.5 * hfc_visc * vdotr2);
#else

#ifdef TRADITIONAL_SPH_FORMULATION
                    O->DtEntropy += FLT(0.5 * (hfc_visc + hfc_egy) * vdotr2);
#else
                    O->DtEntropy += FLT(0.5 * hfc * vdotr2);
#endif
#endif


#ifdef NAVIERSTOKES
                    double faci = I->Mass * I->shear_viscosity / (I->Density * rho) * dwk_i / r;
                    double facj = P[j].Mass * get_shear_viscosity(j) /
                        (SPHP(j).d.Density * SPHP(j).d.Density) * dwk_j / r;

#ifndef NAVIERSTOKES_CONSTANT
                    faci *= pow((I->Entropy * pow(I->Density * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1), 2.5);	/*multiplied by E^5/2 */
                    facj *= pow((SPHP(j).I->Entropy * pow(SPHP(j).d.Density * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1), 2.5);	/*multiplied by E^5/2 */
#endif

#ifdef NAVIERSTOKES_BULK
                    double facbi = I->Mass * All.NavierStokes_BulkViscosity / (I->Density * rho) * dwk_i / r;
                    double facbj = P[j].Mass * All.NavierStokes_BulkViscosity /
                        (SPHP(j).d.Density * SPHP(j).d.Density) * dwk_j / r;
#endif

#ifdef WINDS
                    if(HAS(All.WindModel, WINDS_DECOUPLE_SPH)) {
                        if(P[j].Type == 0)
                            if(SPHP(j).DelayTime > 0)	/* No visc for wind particles */
                            {
                                faci = facj = 0;
#ifdef NAVIERSTOKES_BULK
                                facbi = facbj = 0;
#endif
                            }
                    }
#endif

#ifdef VISCOSITY_SATURATION
                    double IonMeanFreePath_i = All.IonMeanFreePath * pow((I->Entropy * pow(I->Density * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1), 2.0) / rho;	/* u^2/rho */

                    double IonMeanFreePath_j = All.IonMeanFreePath * pow((SPHP(j).I->Entropy * pow(SPHP(j).d.Density * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1), 2.0) / SPHP(j).d.Density;	/* u^2/I->Density */

                    double VelLengthScale_i = 0;
                    double VelLengthScale_j = 0;
                    for(k = 0; k < 3; k++)
                    {
                        if(fabs(I->stressdiag[k]) > 0)
                        {
                            VelLengthScale_i = 2 * soundspeed_i / fabs(I->stressdiag[k]);

                            if(VelLengthScale_i < IonMeanFreePath_i && VelLengthScale_i > 0)
                            {
                                I->stressdiag[k] = stressdiag[k] * (VelLengthScale_i / IonMeanFreePath_i);

                            }
                        }
                        if(fabs(SPHP(j).u.s.StressDiag[k]) > 0)
                        {
                            VelLengthScale_j = 2 * soundspeed_j / fabs(SPHP(j).u.s.StressDiag[k]);

                            if(VelLengthScale_j < IonMeanFreePath_j && VelLengthScale_j > 0)
                            {
                                SPHP(j).u.s.StressDiag[k] = SPHP(j).u.s.StressDiag[k] *
                                    (VelLengthScale_j / IonMeanFreePath_j);

                            }
                        }
                        if(fabs(I->stressoffdiag[k]) > 0)
                        {
                            VelLengthScale_i = 2 * soundspeed_i / fabs(I->stressoffdiag[k]);

                            if(VelLengthScale_i < IonMeanFreePath_i && VelLengthScale_i > 0)
                            {
                                I->stressoffdiag[k] =
                                    I->stressoffdiag[k] * (VelLengthScale_i / IonMeanFreePath_i);
                            }
                        }
                        if(fabs(SPHP(j).u.s.StressOffDiag[k]) > 0)
                        {
                            VelLengthScale_j = 2 * soundspeed_j / fabs(SPHP(j).u.s.StressOffDiag[k]);

                            if(VelLengthScale_j < IonMeanFreePath_j && VelLengthScale_j > 0)
                            {
                                SPHP(j).u.s.StressOffDiag[k] = SPHP(j).u.s.StressOffDiag[k] *
                                    (VelLengthScale_j / IonMeanFreePath_j);
                            }
                        }
                    }
#endif

                    /* Acceleration due to the shear viscosity */
                    O->Acc[0] += faci * (I->stressdiag[0] * dx + I->stressoffdiag[0] * dy + stressoffdiag[1] * dz)
                        + facj * (SPHP(j).u.s.StressDiag[0] * dx + SPHP(j).u.s.StressOffDiag[0] * dy +
                                SPHP(j).u.s.StressOffDiag[1] * dz);

                    O->Acc[1] += faci * (I->stressoffdiag[0] * dx + I->stressdiag[1] * dy + stressoffdiag[2] * dz)
                        + facj * (SPHP(j).u.s.StressOffDiag[0] * dx + SPHP(j).u.s.StressDiag[1] * dy +
                                SPHP(j).u.s.StressOffDiag[2] * dz);

                    O->Acc[2] += faci * (I->stressoffdiag[1] * dx + stressoffdiag[2] * dy + I->stressdiag[2] * dz)
                        + facj * (SPHP(j).u.s.StressOffDiag[1] * dx + SPHP(j).u.s.StressOffDiag[2] * dy +
                                SPHP(j).u.s.StressDiag[2] * dz);

                    /*Acceleration due to the bulk viscosity */
#ifdef NAVIERSTOKES_BULK
#ifdef VISCOSITY_SATURATION
                    VelLengthScale_i = 0;
                    VelLengthScale_j = 0;

                    if(fabs(I->divvel) > 0)
                    {
                        VelLengthScale_i = 3 * soundspeed_i / fabs(I->divvel);

                        if(VelLengthScale_i < IonMeanFreePath_i && VelLengthScale_i > 0)
                        {
                            I->divvel = divvel * (VelLengthScale_i / IonMeanFreePath_i);
                        }
                    }

                    if(fabs(SPHP(j).u.s.a4.DivVel) > 0)
                    {
                        VelLengthScale_j = 3 * soundspeed_j / fabs(SPHP(j).u.s.a4.DivVel);

                        if(VelLengthScale_j < IonMeanFreePath_j && VelLengthScale_j > 0)
                        {
                            SPHP(j).u.s.a4.DivVel = SPHP(j).u.s.a4.DivVel *
                                (VelLengthScale_j / IonMeanFreePath_j);

                        }
                    }
#endif


                    O->Acc[0] += facbi * I->divvel * dx + facbj * SPHP(j).u.s.a4.DivVel * dx;
                    O->Acc[1] += facbi * I->divvel * dy + facbj * SPHP(j).u.s.a4.DivVel * dy;
                    O->Acc[2] += facbi * I->divvel * dz + facbj * SPHP(j).u.s.a4.DivVel * dz;
#endif
#endif /* end NAVIERSTOKES */


#ifdef MAGNETIC
#ifdef EULER_DISSIPATION
                    double alpha_ij_eul = All.ArtMagDispConst;

                    O->DtEulerA +=
                        alpha_ij_eul * 0.5 * vsig * (I->EulerA -
                                SPHP(j).EulerA) * magfac_sym * r * I->Density / (rho_ij *
                                rho_ij);
                    O->DtEulerB +=
                        alpha_ij_eul * 0.5 * vsig * (I->EulerB -
                                SPHP(j).EulerB) * magfac_sym * r * I->Density / (rho_ij *
                                rho_ij);

                    double dTu_diss_eul = -magfac_sym * alpha_ij_eul * (dBx * dBx + dBy * dBy + dBz * dBz);
                    O->DtEntropy += dTu_diss_eul * 0.25 * vsig * mu0_1 * r / (rho_ij * rho_ij);
#endif
#ifdef MAGNETIC_DISSIPATION
                    magfac_sym *= vsig * 0.5 * Balpha_ij * r * I->Density / (rho_ij * rho_ij);
                    O->DtEntropy += dTu_diss_b * 0.25 * vsig * mu0_1 * r / (rho_ij * rho_ij);
                    O->DtB[0] += magfac_sym * dBx;
                    O->DtB[1] += magfac_sym * dBy;
                    O->DtB[2] += magfac_sym * dBz;
#endif
#endif

#ifdef WAKEUP
#error This needs to be prtected by a lock
                    if(vsig > WAKEUP * SPHP(j).MaxSignalVel)
                    {
                        SPHP(j).wakeup = 1;
                    }
#endif
                }
            }
        }

        if(listindex < NODELISTLENGTH)
        {
            startnode = I->NodeList[listindex];
            if(startnode >= 0) {
                startnode = Nodes[startnode].u.d.nextnode;	/* open it */
                listindex++;
                nnodesinlist ++;
            }
        }
    }

    /* Now collect the result at the right place */
#ifdef HYDRO_COST_FACTOR
        O->Ninteractions = ninteractions;
#endif

    /* some performance measures not currently used */
    lv->Ninteractions += ninteractions;
    lv->Nnodesinlist += nnodesinlist;

    return 0;
}

static void * hydro_alloc_ngblist() {
    int threadid = omp_get_thread_num();
    return Ngblist + threadid * NumPart;
}
static int hydro_isactive(int i) {
    return P[i].Type == 0;
}

static void hydro_post_process(int i) {
    int k;
    if(P[i].Type == 0)
    {

#ifndef EOS_DEGENERATE

#ifndef TRADITIONAL_SPH_FORMULATION
        /* Translate energy change rate into entropy change rate */
        SPHP(i).e.DtEntropy *= GAMMA_MINUS1 / (hubble_a2 * pow(SPHP(i).EOMDensity, GAMMA_MINUS1));
#endif

#else
        /* DtEntropy stores the energy change rate in internal units */
        SPHP(i).e.DtEntropy *= All.UnitEnergy_in_cgs / All.UnitTime_in_s;
#endif

#ifdef MACHNUM

        /* Estimates the Mach number of particle i for non-radiative runs,
         * or the Mach number, density jump and specific energy jump
         * in case of cosmic rays!
         */
        GetMachNumber(i);
#endif /* MACHNUM */
#ifdef MACHSTATISTIC
        GetShock_DtEnergy(&SPHP(i));
#endif

#ifdef NAVIERSTOKES
        /* sigma_ab * sigma_ab */
        double fac = 0;
        for(k = 0; k < 3; k++)
        {
            fac += SPHP(i).u.s.StressDiag[k] * SPHP(i).u.s.StressDiag[k] +
                2 * SPHP(i).u.s.StressOffDiag[k] * SPHP(i).u.s.StressOffDiag[k];
        }

#ifndef NAVIERSTOKES_CONSTANT	/*entropy increase due to the shear viscosity */
#ifdef NS_TIMESTEP
        SPHP(i).ViscEntropyChange = 0.5 * GAMMA_MINUS1 /
            (hubble_a2 * pow(SPHP(i).d.Density, GAMMA_MINUS1)) *
            get_shear_viscosity(i) / SPHP(i).d.Density * fac *
            pow((SPHP(i).I->Entropy * pow(SPHP(i).d.Density * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1), 2.5);

        SPHP(i).e.DtEntropy += SPHP(i).ViscEntropyChange;
#else
        SPHP(i).e.DtEntropy += 0.5 * GAMMA_MINUS1 /
            (hubble_a2 * pow(SPHP(i).d.Density, GAMMA_MINUS1)) *
            get_shear_viscosity(i) / SPHP(i).d.Density * fac *
            pow((SPHP(i).I->Entropy * pow(SPHP(i).d.Density * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1), 2.5);
#endif

#else
        SPHP(i).e.DtEntropy += 0.5 * GAMMA_MINUS1 /
            (hubble_a2 * pow(SPHP(i).d.Density, GAMMA_MINUS1)) *
            get_shear_viscosity(i) / SPHP(i).d.Density * fac;

#ifdef NS_TIMESTEP
        SPHP(i).ViscEntropyChange = 0.5 * GAMMA_MINUS1 /
            (hubble_a2 * pow(SPHP(i).d.Density, GAMMA_MINUS1)) *
            get_shear_viscosity(i) / SPHP(i).d.Density * fac;
#endif

#endif

#ifdef NAVIERSTOKES_BULK	/*entropy increase due to the bulk viscosity */
        SPHP(i).e.DtEntropy += GAMMA_MINUS1 /
            (hubble_a2 * pow(SPHP(i).d.Density, GAMMA_MINUS1)) *
            All.NavierStokes_BulkViscosity / SPHP(i).d.Density * pow(SPHP(i).u.s.a4.DivVel, 2);

#ifdef NS_TIMESTEP
        SPHP(i).ViscEntropyChange = GAMMA_MINUS1 /
            (hubble_a2 * pow(SPHP(i).d.Density, GAMMA_MINUS1)) *
            All.NavierStokes_BulkViscosity / SPHP(i).d.Density * pow(SPHP(i).u.s.a4.DivVel, 2);
#endif

#endif

#endif /* these entropy increases directly follow from the general heat transfer equation */


#ifdef JD_VTURB
        SPHP(i).Vrms += (SPHP(i).VelPred[0]-SPHP(i).Vbulk[0])*(SPHP(i).VelPred[0]-SPHP(i).Vbulk[0]) 
            + (SPHP(i).VelPred[1]-SPHP(i).Vbulk[1])*(SPHP(i).VelPred[1]-SPHP(i).Vbulk[1]) 
            + (SPHP(i).VelPred[2]-SPHP(i).Vbulk[2])*(SPHP(i).VelPred[2]-SPHP(i).Vbulk[2]);
        SPHP(i).Vrms = sqrt(SPHP(i).Vrms/SPHP(i).TrueNGB);
#endif

#if defined(JD_DPP) && !defined(JD_DPPONSNAPSHOTONLY)
        compute_Dpp(i);
#endif

#if defined(MAGNETIC) && !defined(EULERPOTENTIALS) && !defined(VECT_POTENTIAL)
        /* take care of cosmological dilution */
        if(All.ComovingIntegrationOn)
            for(k = 0; k < 3; k++)
#ifndef SFR
                SPHP(i).DtB[k] -= 2.0 * SPHP(i).BPred[k];
#else
        SPHP(i).DtB[k] -= 2.0 * SPHP(i).BPred[k] * pow(1.-SPHP(i).XColdCloud,2.*POW_CC);
#endif
#endif

#ifdef WINDS
        /* if we have winds, we decouple particles briefly if delaytime>0 */

        if(HAS(All.WindModel, WINDS_DECOUPLE_SPH)) {
            if(SPHP(i).DelayTime > 0)
            {
                for(k = 0; k < 3; k++)
                    SPHP(i).a.HydroAccel[k] = 0;

                SPHP(i).e.DtEntropy = 0;

#ifdef NOWINDTIMESTEPPING
                SPHP(i).MaxSignalVel = 2 * sqrt(GAMMA * SPHP(i).Pressure / SPHP(i).d.Density);
#else
                double windspeed = All.WindSpeed * All.cf.a;
                windspeed *= fac_mu;
                double hsml_c = pow(All.WindFreeTravelDensFac * All.PhysDensThresh /
                        (SPHP(i).d.Density * a3inv), (1. / 3.));
                SPHP(i).MaxSignalVel = hsml_c * DMAX((2 * windspeed), SPHP(i).MaxSignalVel);
#endif
            }
        }
#endif

#if VECT_POTENTIAL
        /*check if SFR cahnge is needed */
        SPHP(i).DtA[0] +=
            (SPHP(i).VelPred[1] * SPHP(i).BPred[2] -
             SPHP(i).VelPred[2] * SPHP(i).BPred[1]) / (atime * atime * hubble_a);
        SPHP(i).DtA[1] +=
            (SPHP(i).VelPred[2] * SPHP(i).BPred[0] -
             SPHP(i).VelPred[0] * SPHP(i).BPred[2]) / (atime * atime * hubble_a);
        SPHP(i).DtA[2] +=
            (SPHP(i).VelPred[0] * SPHP(i).BPred[1] -
             SPHP(i).VelPred[1] * SPHP(i).BPred[0]) / (atime * atime * hubble_a);
        if(All.ComovingIntegrationOn)
            for(k = 0; k < 3; k++)
                SPHP(i).DtA[k] -= SPHP(i).APred[k];

#endif


#if defined(HEALPIX)
        r_new = 0;
        ded_heal_fac = 1.;
        for(k = 0; k < 3; k++)
        {
            t[k] = P[i].Pos[k] - SysState.CenterOfMassComp[0][k];
            r_new = r_new + t[k] * t[k];
        }
        r_new = sqrt(r_new);
        vec2pix_nest((long) All.Nside, t, &ipix);
        if(r_new > All.healpixmap[ipix] * HEALPIX)
        {
            SPHP(i).e.DtEntropy = 0;
            for(k = 0; k < 3; k++)
            {
                SPHP(i).a.HydroAccel[k] = 0;
                SPHP(i).VelPred[k] = 0.0;
                P[i].Vel[k] = 0.0;
            }
            ded_heal_fac = 2.;
            SPHP(i).v.DivVel = 0.0;
            count++;
            if(r_new > All.healpixmap[ipix] * HEALPIX * 1.5)
            {
                count2++;
            }
        }
#endif

#ifdef TIME_DEP_ART_VISC
#if !defined(EOS_DEGENERATE)
        double cs_h = sqrt(GAMMA * SPHP(i).Pressure / SPHP(i).d.Density) / P[i].Hsml;
#else
        double cs_h = sqrt(SPHP(i).dpdr) / P[i].Hsml;
#endif
        double f = fabs(SPHP(i).v.DivVel) / (fabs(SPHP(i).v.DivVel) + SPHP(i).r.CurlVel + 0.0001 * cs_h / fac_mu);
        SPHP(i).Dtalpha = -(SPHP(i).I->alpha - All.AlphaMin) * All.DecayTime *
            0.5 * SPHP(i).MaxSignalVel / (P[i].Hsml * fac_mu)
            + f * All.ViscSource * DMAX(0.0, -SPHP(i).v.DivVel);
        if(All.ComovingIntegrationOn)
            SPHP(i).Dtalpha /= (hubble_a * All.Time * All.Time);
#endif
#ifdef MAGNETIC
#ifdef TIME_DEP_MAGN_DISP
        SPHP(i).DtBalpha = -(SPHP(i).I->Balpha - All.ArtMagDispMin) * All.ArtMagDispTime *
            0.5 * SPHP(i).MaxSignalVel / (P[i].Hsml * fac_mu)
#ifndef ROT_IN_MAG_DIS
            + All.ArtMagDispSource * fabs(SPHP(i).divB) / sqrt(mu0 * SPHP(i).d.Density);
#else
#ifdef SMOOTH_ROTB
        + All.ArtMagDispSource / sqrt(mu0 * SPHP(i).d.Density) *
            DMAX(fabs(SPHP(i).divB), fabs(sqrt(SPHP(i).SmoothedRotB[0] * SPHP(i).SmoothedRotB[0] +
                            SPHP(i).SmoothedRotB[1] * SPHP(i).SmoothedRotB[1] +
                            SPHP(i).SmoothedRotB[2] * SPHP(i).SmoothedRotB[2])));
#else
        + All.ArtMagDispSource / sqrt(mu0 * SPHP(i).d.Density) *
            DMAX(fabs(SPHP(i).divB), fabs(sqrt(SPHP(i).RotB[0] * SPHP(i).RotB[0] +
                            SPHP(i).RotB[1] * SPHP(i).RotB[1] +
                            SPHP(i).RotB[2] * SPHP(i).RotB[2])));
#endif /* End SMOOTH_ROTB        */
#endif /* End ROT_IN_MAG_DIS     */
#endif /* End TIME_DEP_MAGN_DISP */

#ifdef DIVBFORCE3
        double phiphi = sqrt(pow( SPHP(i).magcorr[0] 	   , 2.)+ pow( SPHP(i).magcorr[1]      ,2.) +pow( SPHP(i).magcorr[2] 	  ,2.));
        double tmpb =   sqrt(pow( SPHP(i).magacc[0] 	   , 2.)+ pow( SPHP(i).magacc[1]       ,2.) +pow( SPHP(i).magacc[2] 	  ,2.));

        if(phiphi > DIVBFORCE3 * tmpb)
            for(k = 0; k < 3; k++)
                SPHP(i).magcorr[k]*= DIVBFORCE3 * tmpb / phiphi;

        for(k = 0; k < 3; k++)
            SPHP(i).a.HydroAccel[k]+=(SPHP(i).magacc[k]-SPHP(i).magcorr[k]);

#endif

#ifdef DIVBCLEANING_DEDNER
        double tmpb = 0.5 * SPHP(i).MaxSignalVel;
        double phiphi = tmpb * All.DivBcleanHyperbolicSigma * atime
#ifdef HEALPIX
            / ded_heal_fac 
#endif
#ifdef SFR 
            * pow(1.-SPHP(i).XColdCloud,3.*POW_CC)
#endif
            * SPHP(i).SmoothDivB;
#ifdef SMOOTH_PHI
        phiphi += SPHP(i).SmoothPhi *
#else
            phiphi += SPHP(i).I->PhiPred *
#endif
#ifdef HEALPIX
            ded_heal_fac * 
#endif
#ifdef SFR 
            pow(1.-SPHP(i).XColdCloud,POW_CC) *
#endif
            All.DivBcleanParabolicSigma / P[i].Hsml;

        if(All.ComovingIntegrationOn)
            SPHP(i).DtPhi =
#ifdef SMOOTH_PHI
                - SPHP(i).SmoothPhi
#else
                - SPHP(i).I->PhiPred
#endif
#ifdef SFR 
                * pow(1.-SPHP(i).XColdCloud,POW_CC)
#endif
                - ( phiphi * tmpb) / (hubble_a * atime);	///carefull with the + or not +
        else
            SPHP(i).DtPhi = (-phiphi * tmpb);

        if(All.ComovingIntegrationOn){
            SPHP(i).GradPhi[0]*=1/(hubble_a * atime);
            SPHP(i).GradPhi[1]*=1/(hubble_a * atime);
            SPHP(i).GradPhi[2]*=1/(hubble_a * atime);
        }
        double phiphi = sqrt(pow( SPHP(i).GradPhi[0] , 2.)+pow( SPHP(i).GradPhi[1]  ,2.)+pow( SPHP(i).GradPhi[2] ,2.));
        double tmpb   = sqrt(pow( SPHP(i).DtB[0]      ,2.)+pow( SPHP(i).DtB[1]      ,2.)+pow( SPHP(i).DtB[2]     ,2.));

        if(phiphi > All.DivBcleanQ * tmpb){
            SPHP(i).GradPhi[0]*= All.DivBcleanQ * tmpb / phiphi;
            SPHP(i).GradPhi[1]*= All.DivBcleanQ * tmpb / phiphi;
            SPHP(i).GradPhi[2]*= All.DivBcleanQ * tmpb / phiphi;
        }	

        SPHP(i).e.DtEntropy += mu0 * (SPHP(i).BPred[0] * SPHP(i).GradPhi[0] + SPHP(i).BPred[1] * SPHP(i).GradPhi[1] + SPHP(i).BPred[2] * SPHP(i).GradPhi[2]) 
#ifdef SFR
            * pow(1.-SPHP(i).XColdCloud,3.*POW_CC)
#endif
            * GAMMA_MINUS1 / (hubble_a2 * pow(SPHP(i).d.Density, GAMMA_MINUS1));

        SPHP(i).DtB[0]+=SPHP(i).GradPhi[0];
        SPHP(i).DtB[1]+=SPHP(i).GradPhi[1];
        SPHP(i).DtB[2]+=SPHP(i).GradPhi[2];


#endif /* End DEDNER */
#endif /* End Magnetic */

#ifdef SPH_BND_PARTICLES
        if(P[i].I->ID == 0)
        {
            SPHP(i).e.DtEntropy = 0;
#ifdef NS_TIMESTEP
            SPHP(i).ViscEntropyChange = 0;
#endif

#ifdef DIVBCLEANING_DEDNER
            SPHP(i).DtPhi = 0;
#endif
#if defined(MAGNETIC) && !defined(EULERPOTENTIALS) && !defined(VECT_POTENTIAL)
            for(k = 0; k < 3; k++)
                SPHP(i).DtB[k] = 0;
#endif

            for(k = 0; k < 3; k++)
                SPHP(i).a.HydroAccel[k] = 0;
        }
#endif
    }
}
