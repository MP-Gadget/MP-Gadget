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

extern int Nexport, Nimport;

/*! \file hydra.c
 *  \brief Computation of SPH forces and rate of entropy generation
 *
 *  This file contains the "second SPH loop", where the SPH forces are
 *  computed, and where the rate of change of entropy due to the shock heating
 *  (via artificial viscosity) is computed.
 */

static int hydro_evaluate(int target, int mode, Exporter * exporter, int * ngblist);
static int hydro_isactive(int n);
static void * hydro_alloc_ngblist();

struct hydrodata_in
{
#ifndef DONOTUSENODELIST
    int NodeList[NODELISTLENGTH];
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
    MyIDType ID;			/*!< particle identifier */
#endif

#ifdef JD_VTURB
    MyFloat Vbulk[3];
#endif

#ifdef MAGNETIC
    MyFloat BPred[3];
#ifdef VECT_POTENTIAL
    MyFloat Apred[3];
#endif
#ifdef ALFA_OMEGA_DYN
    MyFloat alfaomega;
#endif
#ifdef EULER_DISSIPATION
    MyFloat EulerA, EulerB;
#endif
#ifdef TIME_DEP_MAGN_DISP
    MyFloat Balpha;
#endif
#ifdef DIVBCLEANING_DEDNER
    MyFloat PhiPred;
#endif
#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS)
    MyFloat RotB[3];
#endif
#endif
#ifdef TIME_DEP_ART_VISC
    MyFloat alpha;
#endif

#if defined(NAVIERSTOKES)
    MyFloat Entropy;
#endif



#ifdef NAVIERSTOKES
    MyFloat stressoffdiag[3];
    MyFloat stressdiag[3];
    MyFloat shear_viscosity;
#endif

#ifdef NAVIERSTOKES_BULK
    MyFloat divvel;
#endif

#ifdef EOS_DEGENERATE
    MyFloat dpdr;
#endif

} *HydroDataGet;


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
    MyFloat dta[3];
#endif
#if  defined(CR_SHOCK)
    MyFloat CR_EnergyChange[NUMCRPOP];
    MyFloat CR_BaryonFractionChange[NUMCRPOP];
#endif

#ifdef HYDRO_COST_FACTOR
    int Ninteractions;
#endif
}
*HydroDataResult;


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
    double timecomp, timecomm, timewait, tstart, tend, t0, t1;

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

#ifdef WINDS
    double windspeed, hsml_c;

#endif


#ifdef TIME_DEP_ART_VISC
    double f, cs_h;
#endif
#if defined(MAGNETIC) && defined(MAGFORCE)
#if defined(TIME_DEP_MAGN_DISP) || defined(DIVBCLEANING_DEDNER)
    double mu0 = 1;
#endif
#endif

#if defined(DIVBCLEANING_DEDNER) || defined(DIVBFORCE3)
    double phiphi, tmpb;
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

    CPU_Step[CPU_HYDMISC] += measure_time();
    t0 = second();

    evaluate_begin(&ev);
    do
    {
        /* do local particles and prepare export list */
        evaluate_primary(&ev);

        n_exported += ev.Nexport;

        HydroDataGet = (struct hydrodata_in *) evaluate_get_remote(&ev, TAG_HYDRO_A);

        HydroDataResult =
            (struct hydrodata_out *) mymalloc("HydroDataResult", ev.Nimport * sizeof(struct hydrodata_out));

        report_memory_usage(&HighMark_sphhydro, "SPH_HYDRO");

        /* now do the particles that were sent to us */

        evaluate_secondary(&ev);

        /* get the result */
        evaluate_reduce_result(&ev, HydroDataResult, TAG_HYDRO_B);

        myfree(HydroDataResult);
        myfree(HydroDataGet);
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



    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
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
            for(k = 0, fac = 0; k < 3; k++)
            {
                fac += SPHP(i).u.s.StressDiag[k] * SPHP(i).u.s.StressDiag[k] +
                    2 * SPHP(i).u.s.StressOffDiag[k] * SPHP(i).u.s.StressOffDiag[k];
            }

#ifndef NAVIERSTOKES_CONSTANT	/*entropy increase due to the shear viscosity */
#ifdef NS_TIMESTEP
            SPHP(i).ViscEntropyChange = 0.5 * GAMMA_MINUS1 /
                (hubble_a2 * pow(SPHP(i).d.Density, GAMMA_MINUS1)) *
                get_shear_viscosity(i) / SPHP(i).d.Density * fac *
                pow((SPHP(i).Entropy * pow(SPHP(i).d.Density * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1), 2.5);

            SPHP(i).e.DtEntropy += SPHP(i).ViscEntropyChange;
#else
            SPHP(i).e.DtEntropy += 0.5 * GAMMA_MINUS1 /
                (hubble_a2 * pow(SPHP(i).d.Density, GAMMA_MINUS1)) *
                get_shear_viscosity(i) / SPHP(i).d.Density * fac *
                pow((SPHP(i).Entropy * pow(SPHP(i).d.Density * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1), 2.5);
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

            if(SPHP(i).DelayTime > 0)
            {
                for(k = 0; k < 3; k++)
                    SPHP(i).a.HydroAccel[k] = 0;

                SPHP(i).e.DtEntropy = 0;

#ifdef NOWINDTIMESTEPPING
                SPHP(i).MaxSignalVel = 2 * sqrt(GAMMA * SPHP(i).Pressure / SPHP(i).d.Density);
#else
                windspeed = sqrt(2 * All.WindEnergyFraction * All.FactorSN *
                        All.EgySpecSN / (1 - All.FactorSN) / All.WindEfficiency) * All.Time;
                windspeed *= fac_mu;
                hsml_c = pow(All.WindFreeTravelDensFac * All.PhysDensThresh /
                        (SPHP(i).d.Density * a3inv), (1. / 3.));
                SPHP(i).MaxSignalVel = hsml_c * DMAX((2 * windspeed), SPHP(i).MaxSignalVel);
#endif
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
            cs_h = sqrt(GAMMA * SPHP(i).Pressure / SPHP(i).d.Density) / P[i].Hsml;
#else
            cs_h = sqrt(SPHP(i).dpdr) / P[i].Hsml;
#endif
            f = fabs(SPHP(i).v.DivVel) / (fabs(SPHP(i).v.DivVel) + SPHP(i).r.CurlVel + 0.0001 * cs_h / fac_mu);
            SPHP(i).Dtalpha = -(SPHP(i).alpha - All.AlphaMin) * All.DecayTime *
                0.5 * SPHP(i).MaxSignalVel / (P[i].Hsml * fac_mu)
                + f * All.ViscSource * DMAX(0.0, -SPHP(i).v.DivVel);
            if(All.ComovingIntegrationOn)
                SPHP(i).Dtalpha /= (hubble_a * All.Time * All.Time);
#endif
#ifdef MAGNETIC
#ifdef TIME_DEP_MAGN_DISP
            SPHP(i).DtBalpha = -(SPHP(i).Balpha - All.ArtMagDispMin) * All.ArtMagDispTime *
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
            phiphi = sqrt(pow( SPHP(i).magcorr[0] 	   , 2.)+ pow( SPHP(i).magcorr[1]      ,2.) +pow( SPHP(i).magcorr[2] 	  ,2.));
            tmpb =   sqrt(pow( SPHP(i).magacc[0] 	   , 2.)+ pow( SPHP(i).magacc[1]       ,2.) +pow( SPHP(i).magacc[2] 	  ,2.));

            if(phiphi > DIVBFORCE3 * tmpb)
                for(k = 0; k < 3; k++)
                    SPHP(i).magcorr[k]*= DIVBFORCE3 * tmpb / phiphi;

            for(k = 0; k < 3; k++)
                SPHP(i).a.HydroAccel[k]+=(SPHP(i).magacc[k]-SPHP(i).magcorr[k]);

#endif

#ifdef DIVBCLEANING_DEDNER
            tmpb = 0.5 * SPHP(i).MaxSignalVel;
            phiphi = tmpb * All.DivBcleanHyperbolicSigma * atime
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
                phiphi += SPHP(i).PhiPred *
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
                    - SPHP(i).PhiPred
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
            phiphi = sqrt(pow( SPHP(i).GradPhi[0] , 2.)+pow( SPHP(i).GradPhi[1]  ,2.)+pow( SPHP(i).GradPhi[2] ,2.));
            tmpb   = sqrt(pow( SPHP(i).DtB[0]      ,2.)+pow( SPHP(i).DtB[1]      ,2.)+pow( SPHP(i).DtB[2]     ,2.));

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
            if(P[i].ID == 0)
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

    t1 = WallclockTime = second();
    timeall += timediff(t0, t1);

    timecomp = ev.timecomp1 + ev.timecomp2;
    timewait = ev.timewait1 + ev.timewait2;
    timecomm = ev.timecommsumm1 + ev.timecommsumm2;

    CPU_Step[CPU_HYDCOMPUTE] += timecomp;
    CPU_Step[CPU_HYDWAIT] += timewait;
    CPU_Step[CPU_HYDCOMM] += timecomm;
    CPU_Step[CPU_HYDNETWORK] += timenetwork;
    CPU_Step[CPU_HYDMISC] += timeall - (timecomp + timewait + timecomm + timenetwork);
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
    input->alfaomega =
        Sph[place].r.Rot[0] * Sph[place].VelPred[0] + Sph[place].r.Rot[1] * Sph[place].VelPred[1] +
        Sph[place].r.Rot[2] * Sph[place].VelPred[2];
#endif
#if defined(EULERPOTENTIALS) && defined(EULER_DISSIPATION)
    input->EulerA = SPHP(place).EulerA;
    input->EulerB = SPHP(place).EulerB;
#endif
#ifdef VECT_POTENTIAL
    input->Apred[0] = SPHP(place).APred[0];
    input->Apred[1] = SPHP(place).APred[1];
    input->Apred[2] = SPHP(place).APred[2];
#endif
#ifdef DIVBCLEANING_DEDNER
#ifdef SMOOTH_PHI
    input->PhiPred = SPHP(place).SmoothPhi;
#else
    input->PhiPred = SPHP(place).PhiPred;
#endif
#endif
#endif


#if defined(NAVIERSTOKES)
    input->Entropy = SPHP(place).Entropy;
#endif


#ifdef TIME_DEP_ART_VISC
    input->alpha = SPHP(place).alpha;
#endif


#ifdef PARTICLE_DEBUG
    input->ID = P[place].ID;
#endif

#ifdef NAVIERSTOKES
    for(k = 0; k < 3; k++)
    {
        input->stressdiag[k] = SPHP(i).u.s.StressDiag[k];
        input->stressoffdiag[k] = SPHP(i).u.s.StressOffDiag[k];
    }
    input->shear_viscosity = get_shear_viscosity(i);

#ifdef NAVIERSTOKES_BULK
    input->divvel = SPHP(i).u.s.DivVel;
#endif
#endif

#ifdef TIME_DEP_MAGN_DISP
    input->Balpha = SPHP(place).Balpha;
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
static int hydro_evaluate(int target, int mode, Exporter * exporter, int * ngblist) 
{
    int startnode, numngb, listindex = 0;
    int j, k, n, timestep;
    MyDouble *pos;
    MyFloat *vel;
    MyFloat mass, dhsmlDensityFactor, rho, pressure, f1, f2;
    MyLongDouble acc[3], dtEntropy;

#ifdef DENSITY_INDEPENDENT_SPH
    double egyrho = 0, entvarpred = 0;
#endif

#ifdef HYDRO_COST_FACTOR
    int ninteractions = 0;
#endif


#ifdef ALTERNATIVE_VISCOUS_TIMESTEP
    MyFloat minViscousDt;
#else
    MyFloat maxSignalVel;
#endif
    double dx, dy, dz, dvx, dvy, dvz;
    double p_over_rho2_i, p_over_rho2_j, soundspeed_i, soundspeed_j;
    double hfc, vdotr, vdotr2, visc, mu_ij, rho_ij, vsig;
    double h_i;
    density_kernel_t kernel_i;
    double h_j;
    density_kernel_t kernel_j;
    double dwk_i, dwk_j;
    double r, r2, u;
    double hfc_visc;

#ifdef TRADITIONAL_SPH_FORMULATION
    double hfc_egy;
#endif

    double BulkVisc_ij;

#ifdef NAVIERSTOKES
    double faci, facj;
    MyFloat *stressdiag;
    MyFloat *stressoffdiag;
    MyFloat shear_viscosity;

#ifdef VISCOSITY_SATURATION
    double VelLengthScale_i, VelLengthScale_j;
    double IonMeanFreePath_i, IonMeanFreePath_j;
#endif
#ifdef NAVIERSTOKES_BULK
    double facbi, facbj;
    MyFloat divvel;
#endif
#endif

#if defined(NAVIERSTOKES)
    double Entropy;
#endif

#ifdef TIME_DEP_ART_VISC
    MyFloat alpha;
#endif

#ifdef ALTVISCOSITY
    double mu_i, mu_j;
#endif

#ifndef NOVISCOSITYLIMITER
    double dt;
#endif

#ifdef JD_VTURB
    MyFloat vRms=0;
    MyFloat vBulk[3]={0};
#endif

#ifdef MAGNETIC
    MyFloat bpred[3];

#ifdef ALFA_OMEGA_DYN
    double alfaomega;
#endif
#if ( !defined(EULERPOTENTIALS) || !defined(VECT_POTENTIAL) )
    double dtB[3];
#endif

    double dBx, dBy, dBz;
    double magfac, magfac_i, magfac_j, magfac_i_base;
    double mu0_1;

#if defined(MAGNETIC_DIFFUSION) || defined(VECT_POTENTIAL)
    double magfac_diff;
#endif

#ifdef MAGFORCE
    double mm_i[3][3], mm_j[3][3];
    double b2_i, b2_j;
    int l;
#endif

#if defined(MAGNETIC_DISSIPATION) || defined(DIVBCLEANING_DEDNER) || defined(EULER_DISSIPATION) || defined(MAGNETIC_DIFFUSION)
    double magfac_sym;
#endif

#ifdef MAGNETIC_DISSIPATION
    double dTu_diss_b, Balpha_ij;

#ifdef MAGDISSIPATION_PERPEN
    double mft, mvt[3];
#endif
#ifdef TIME_DEP_MAGN_DISP
    double Balpha;
#endif
#endif

#ifdef EULER_DISSIPATION
    double eulA, eulB, dTu_diss_eul, alpha_ij_eul;
    double dteulA, dteulB;
#endif
#ifdef DIVBFORCE3
    double magacc[3];
    double magcorr[3];
#endif

#ifdef DIVBCLEANING_DEDNER
    double PhiPred, phifac;
    double gradphi[3];
#endif
#ifdef MAGNETIC_SIGNALVEL
    double magneticspeed_i, magneticspeed_j, vcsa2_i, vcsa2_j, Bpro2_i, Bpro2_j;
#endif
#ifdef VECT_POTENTIAL
    double dta[3];
    double Apred[3];

    dta[0] = dta[1] = dta[2] = 0.0;
#endif
#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS)
    MyFloat rotb[3];
#endif

#endif /* end magnetic */

#ifdef PARTICLE_DEBUG
    MyIDType ID;			/*!< particle identifier */
#endif

#ifdef CONVENTIONAL_VISCOSITY
    double c_ij, h_ij;
#endif

    if(mode == 0)
    {
        pos = P[target].Pos;
        vel = SPHP(target).VelPred;
        h_i = P[target].Hsml;
        mass = P[target].Mass;
        rho = SPHP(target).d.Density;
        pressure = SPHP(target).Pressure;
        timestep = (P[target].TimeBin ? (1 << P[target].TimeBin) : 0);

#ifdef DENSITY_INDEPENDENT_SPH
        egyrho = SPHP(target).EgyWtDensity;
        entvarpred = SPHP(target).EntVarPred;
        dhsmlDensityFactor = SPHP(target).DhsmlEgyDensityFactor;
#else
        dhsmlDensityFactor = SPHP(target).h.DhsmlDensityFactor;
#endif

#ifndef EOS_DEGENERATE
#ifdef DENSITY_INDEPENDENT_SPH
        soundspeed_i = sqrt(GAMMA * pressure / egyrho);
#else
        soundspeed_i = sqrt(GAMMA * pressure / rho);
#endif
#else
        soundspeed_i = sqrt(SPHP(target).dpdr);
#endif

#ifndef ALTVISCOSITY
#ifndef NAVIERSTOKES
        f1 = fabs(SPHP(target).v.DivVel) /
            (fabs(SPHP(target).v.DivVel) + SPHP(target).r.CurlVel +
             0.0001 * soundspeed_i / P[target].Hsml / fac_mu);
#else
        f1 = fabs(SPHP(target).v.DivVel) /
            (fabs(SPHP(target).v.DivVel) + SPHP(target).u.s.CurlVel +
             0.0001 * soundspeed_i / P[target].Hsml / fac_mu);
#endif
#else
        f1 = SPHP(target).v.DivVel;
#endif

#ifdef JD_VTURB
        vBulk[0] = SPHP(target).Vbulk[0];
        vBulk[1] = SPHP(target).Vbulk[1];
        vBulk[2] = SPHP(target).Vbulk[2];
#endif

#ifdef MAGNETIC
#ifndef SFR
        bpred[0] = SPHP(target).BPred[0];
        bpred[1] = SPHP(target).BPred[1];
        bpred[2] = SPHP(target).BPred[2];
#else
        bpred[0] = SPHP(target).BPred[0] * pow(1.-SPHP(target).XColdCloud,2.*POW_CC);
        bpred[1] = SPHP(target).BPred[1] * pow(1.-SPHP(target).XColdCloud,2.*POW_CC);
        bpred[2] = SPHP(target).BPred[2] * pow(1.-SPHP(target).XColdCloud,2.*POW_CC);
#endif
#ifdef ALFA_OMEGA_DYN
        alfaomega =
            Sph[target].r.Rot[0] * Sph[target].VelPred[0] + Sph[target].r.Rot[1] * Sph[target].VelPred[1] +
            Sph[target].r.Rot[2] * Sph[target].VelPred[2];
#endif
#ifdef VECT_POTENTIAL
        Apred[0] = SPHP(target).APred[0];
        Apred[1] = SPHP(target).APred[1];
        Apred[2] = SPHP(target).APred[2];
#endif
#ifdef DIVBCLEANING_DEDNER
#ifdef SMOOTH_PHI
        PhiPred = SPHP(target).SmoothPhi;
#else
        PhiPred = SPHP(target).PhiPred;
#endif
#ifdef SFR
        PhiPred *= pow(1.-SPHP(target).XColdCloud,POW_CC);
#endif
#endif
#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS)
#ifdef SMOOTH_ROTB
        rotb[0] = SPHP(target).SmoothedRotB[0];
        rotb[1] = SPHP(target).SmoothedRotB[1];
        rotb[2] = SPHP(target).SmoothedRotB[2];
#else
        rotb[0] = SPHP(target).RotB[0];
        rotb[1] = SPHP(target).RotB[1];
        rotb[2] = SPHP(target).RotB[2];
#endif
#ifdef SFR
        rotb[0] *= pow(1.-SPHP(target).XColdCloud,3.*POW_CC);
        rotb[1] *= pow(1.-SPHP(target).XColdCloud,3.*POW_CC);
        rotb[2] *= pow(1.-SPHP(target).XColdCloud,3.*POW_CC);
#endif
#endif
#ifdef TIME_DEP_MAGN_DISP
        Balpha = SPHP(target).Balpha;
#endif
#ifdef EULER_DISSIPATION
        eulA = SPHP(target).EulerA;
        eulB = SPHP(target).EulerB;
#endif
#endif /*  MAGNETIC  */

#ifdef TIME_DEP_ART_VISC
        alpha = SPHP(target).alpha;
#endif

#if defined(NAVIERSTOKES)
        Entropy = SPHP(target).Entropy;
#endif



#ifdef PARTICLE_DEBUG
        ID = P[target].ID;
#endif

#ifdef NAVIERSTOKES
        stressdiag = SPHP(target).u.s.StressDiag;
        stressoffdiag = SPHP(target).u.s.StressOffDiag;
        shear_viscosity = get_shear_viscosity(target);
#ifdef NAVIERSTOKES_BULK
        divvel = SPHP(target).u.s.a4.DivVel;
#endif
#endif

    }
    else
    {
        pos = HydroDataGet[target].Pos;
        vel = HydroDataGet[target].Vel;
        h_i = HydroDataGet[target].Hsml;
        mass = HydroDataGet[target].Mass;
        dhsmlDensityFactor = HydroDataGet[target].DhsmlDensityFactor;
#ifdef DENSITY_INDEPENDENT_SPH
        egyrho = HydroDataGet[target].EgyRho;
        entvarpred = HydroDataGet[target].EntVarPred;
#endif
        rho = HydroDataGet[target].Density;
        pressure = HydroDataGet[target].Pressure;
        timestep = HydroDataGet[target].Timestep;
#ifndef EOS_DEGENERATE
#ifdef DENSITY_INDEPENDENT_SPH
        soundspeed_i = sqrt(GAMMA * pressure / egyrho);
#else
        soundspeed_i = sqrt(GAMMA * pressure / rho);
#endif
#else
        soundspeed_i = sqrt(HydroDataGet[target].dpdr);
#endif
        f1 = HydroDataGet[target].F1;

#ifdef JD_VTURB
        vBulk[0] = HydroDataGet[target].Vbulk[0];
        vBulk[1] = HydroDataGet[target].Vbulk[1];
        vBulk[2] = HydroDataGet[target].Vbulk[2];
#endif
#ifdef MAGNETIC
        bpred[0] = HydroDataGet[target].BPred[0];
        bpred[1] = HydroDataGet[target].BPred[1];
        bpred[2] = HydroDataGet[target].BPred[2];
#ifdef ALFA_OMEGA_DYN
        alfaomega = HydroDataGet[target].alfaomega;
#endif
#ifdef VECT_POTENTIAL
        Apred[0] = HydroDataGet[target].Apred[0];
        Apred[1] = HydroDataGet[target].Apred[1];
        Apred[2] = HydroDataGet[target].Apred[2];
#endif
#ifdef DIVBCLEANING_DEDNER
        PhiPred = HydroDataGet[target].PhiPred;
#endif
#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS)
        rotb[0] = HydroDataGet[target].RotB[0];
        rotb[1] = HydroDataGet[target].RotB[1];
        rotb[2] = HydroDataGet[target].RotB[2];
#endif
#ifdef TIME_DEP_MAGN_DISP
        Balpha = HydroDataGet[target].Balpha;
#endif
#ifdef EULER_DISSIPATION
        eulA = HydroDataGet[target].EulerA;
        eulB = HydroDataGet[target].EulerB;
#endif
#endif /* MAGNETIC */

#ifdef TIME_DEP_ART_VISC
        alpha = HydroDataGet[target].alpha;
#endif

#if defined(NAVIERSTOKES)
        Entropy = HydroDataGet[target].Entropy;
#endif


#ifdef PARTICLE_DEBUG
        ID = HydroDataGet[target].ID;
#endif


#ifdef NAVIERSTOKES
        stressdiag = HydroDataGet[target].stressdiag;
        stressoffdiag = HydroDataGet[target].stressoffdiag;
        shear_viscosity = HydroDataGet[target].shear_viscosity;
#endif
#ifdef NAVIERSTOKES
        stressdiag = HydroDataGet[target].stressdiag;
        stressoffdiag = HydroDataGet[target].stressoffdiag;
        shear_viscosity = HydroDataGet[target].shear_viscosity;
#ifdef NAVIERSTOKES_BULK
        divvel = HydroDataGet[target].divvel;
#endif
#endif
    }


    /* initialize variables before SPH loop is started */

    acc[0] = acc[1] = acc[2] = dtEntropy = 0;
    density_kernel_init(&kernel_i, h_i);

#ifdef DIVBFORCE3
    magacc[0]=magacc[1]=magacc[2]=0.0;
    magcorr[0]=magcorr[1]=magcorr[2]=0.0;
#endif


#ifdef MAGNETIC
#if ( !defined(EULERPOTENTIALS) || !defined(VECT_POTENTIAL) )
    for(k = 0; k < 3; k++)
        dtB[k] = 0;
#endif
#ifdef EULER_DISSIPATION
    dteulA = 0;
    dteulB = 0;
#endif
    mu0_1 = 1;
#ifndef MU0_UNITY
    mu0_1 /= (4 * M_PI);
    mu0_1 *= All.UnitTime_in_s * All.UnitTime_in_s * All.UnitLength_in_cm / (All.UnitMass_in_g);
    if(All.ComovingIntegrationOn)
        mu0_1 /= (All.HubbleParam * All.HubbleParam);

#endif
#ifdef DIVBCLEANING_DEDNER
    gradphi[2]= gradphi[1]=  gradphi[0]=0.0;
#endif
#ifdef MAGFORCE
    magfac_i_base = 1 / (rho * rho);
#ifndef MU0_UNITY
    magfac_i_base /= (4 * M_PI);
#endif
#ifdef CORRECTBFRC
    magfac_i_base *= dhsmlDensityFactor;
#endif
    for(k = 0, b2_i = 0; k < 3; k++)
    {
        b2_i += bpred[k] * bpred[k];
        for(l = 0; l < 3; l++)
            mm_i[k][l] = bpred[k] * bpred[l];
    }
    for(k = 0; k < 3; k++)
        mm_i[k][k] -= 0.5 * b2_i;
#ifdef MAGNETIC_SIGNALVEL
#ifdef ALFVEN_VEL_LIMITER
    vcsa2_i = soundspeed_i * soundspeed_i +
        DMIN(mu0_1 * b2_i / rho, ALFVEN_VEL_LIMITER * soundspeed_i * soundspeed_i);
#else
    vcsa2_i = soundspeed_i * soundspeed_i + mu0_1 * b2_i / rho;
#endif
#endif
#endif /* end of MAGFORCE */
#endif /* end of MAGNETIC */

#ifndef TRADITIONAL_SPH_FORMULATION
#ifdef DENSITY_INDEPENDENT_SPH
    p_over_rho2_i = pressure / (egyrho * egyrho);
#else
    p_over_rho2_i = pressure / (rho * rho);
#endif
#else
    p_over_rho2_i = pressure / (rho * rho);
#endif

#ifdef ALTERNATIVE_VISCOUS_TIMESTEP
    minViscousDt = 1.0e32;
#else
    maxSignalVel = soundspeed_i;
#endif


    /* Now start the actual SPH computation for this particle */

    if(mode == 0)
    {
        startnode = All.MaxPart;	/* root node */
    }
    else
    {
#ifndef DONOTUSENODELIST
        startnode = HydroDataGet[target].NodeList[0];
        startnode = Nodes[startnode].u.d.nextnode;	/* open it */
#else
        startnode = All.MaxPart;	/* root node */
#endif
    }

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb =
                ngb_treefind_pairs_threads(pos, h_i, target, &startnode, 
                        mode, exporter, ngblist);

            if(numngb < 0)
                return -1;

            for(n = 0; n < numngb; n++)
            {
                j = ngblist[n];

#ifdef HYDRO_COST_FACTOR
                ninteractions++;
#endif

#ifdef BLACK_HOLES
                if(P[j].Mass == 0)
                    continue;
#endif

#ifdef NOWINDTIMESTEPPING
#ifdef WINDS
                if(P[j].Type == 0)
                    if(SPHP(j).DelayTime > 0)	/* ignore the wind particles */
                        continue;
#endif
#endif
                dx = pos[0] - P[j].Pos[0];
                dy = pos[1] - P[j].Pos[1];
                dz = pos[2] - P[j].Pos[2];
#ifdef PERIODIC			/*  now find the closest image in the given box size  */
                dx = NEAREST_X(dx);
                dy = NEAREST_Y(dy);
                dz = NEAREST_Z(dz);
#endif
                r2 = dx * dx + dy * dy + dz * dz;
                h_j = P[j].Hsml;
                density_kernel_init(&kernel_j, h_j);
                if(r2 < kernel_i.HH || r2 < kernel_j.HH)
                {
                    r = sqrt(r2);
                    if(r > 0)
                    {
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

                        dvx = vel[0] - SPHP(j).VelPred[0];
                        dvy = vel[1] - SPHP(j).VelPred[1];
                        dvz = vel[2] - SPHP(j).VelPred[2];
                        vdotr = dx * dvx + dy * dvy + dz * dvz;
                        rho_ij = 0.5 * (rho + SPHP(j).d.Density);

                        if(All.ComovingIntegrationOn)
                            vdotr2 = vdotr + hubble_a2 * r2;
                        else
                            vdotr2 = vdotr;

                        dwk_i = density_kernel_dwk(&kernel_i, r * kernel_i.Hinv);
                        dwk_j = density_kernel_dwk(&kernel_j, r * kernel_j.Hinv);

#ifdef JD_VTURB
                        if ( h_i >= P[j].Hsml)  /* Make sure j is inside targets hsml */
                            vRms += (SPHP(j).VelPred[0]-vBulk[0])*(SPHP(j).VelPred[0]-vBulk[0]) 
                                + (SPHP(j).VelPred[1]-vBulk[1])*(SPHP(j).VelPred[1]-vBulk[1]) 
                                + (SPHP(j).VelPred[2]-vBulk[2])*(SPHP(j).VelPred[2]-vBulk[2]);
#endif

#ifdef MAGNETIC
#ifndef SFR
                        dBx = bpred[0] - SPHP(j).BPred[0];
                        dBy = bpred[1] - SPHP(j).BPred[1];
                        dBz = bpred[2] - SPHP(j).BPred[2];
#else
                        dBx = bpred[0] - SPHP(j).BPred[0] * pow(1.-SPHP(j).XColdCloud,2.*POW_CC);
                        dBy = bpred[1] - SPHP(j).BPred[1] * pow(1.-SPHP(j).XColdCloud,2.*POW_CC);
                        dBz = bpred[2] - SPHP(j).BPred[2] * pow(1.-SPHP(j).XColdCloud,2.*POW_CC);
#endif

                        magfac = P[j].Mass / r;	/* we moved 'dwk_i / rho' down ! */
                        if(All.ComovingIntegrationOn)
                            magfac *= 1. / (hubble_a * All.Time * All.Time);
                        /* last factor takes care of all cosmological prefactor */
#ifdef CORRECTDB
                        magfac *= dhsmlDensityFactor;
#endif
#if defined(MAGNETIC_DISSIPATION) || defined(DIVBCLEANING_DEDNER) || defined(EULER_DISSIPATION) || defined(MAGNETIC_DIFFUSION)
                        magfac_sym = magfac * (dwk_i + dwk_j) * 0.5;
#endif
#ifdef MAGNETIC_DISSIPATION
#ifdef TIME_DEP_MAGN_DISP
                        Balpha_ij = 0.5 * (Balpha + SPHP(j).Balpha);
#else
                        Balpha_ij = All.ArtMagDispConst;
#endif
#endif
                        magfac *= dwk_i / rho;
#if VECT_POTENTIAL
                        dta[0] +=
                            P[j].Mass * dwk_i / r * (Apred[0] -
                                    SPHP(j).APred[0]) * dx * vel[0] / (rho * atime * atime *
                                    hubble_a);
                        dta[1] +=
                            P[j].Mass * dwk_i / r * (Apred[1] -
                                    SPHP(j).APred[1]) * dy * vel[1] / (rho * atime * atime *
                                    hubble_a);
                        dta[2] +=
                            P[j].Mass * dwk_i / r * (Apred[2] -
                                    SPHP(j).APred[2]) * dz * vel[2] / (rho * atime * atime *
                                    hubble_a);
                        dta[0] +=
                            P[j].Mass * dwk_i / r * ((Apred[0] - SPHP(j).APred[0]) * dx * vel[0] +
                                    (Apred[0] - SPHP(j).APred[0]) * dy * vel[1] + (Apred[0] -
                                        SPHP(j).
                                        APred[0]) *
                                    dz * vel[2]) / (rho * atime * atime * hubble_a);

#endif
#if ( !defined(EULERPOTENTIALS) || !defined(VECT_POTENTIAL) )
                        dtB[0] +=
                            magfac * ((bpred[0] * dvy - bpred[1] * dvx) * dy +
                                    (bpred[0] * dvz - bpred[2] * dvx) * dz);
                        dtB[1] +=
                            magfac * ((bpred[1] * dvz - bpred[2] * dvy) * dz +
                                    (bpred[1] * dvx - bpred[0] * dvy) * dx);
                        dtB[2] +=
                            magfac * ((bpred[2] * dvx - bpred[0] * dvz) * dx +
                                    (bpred[2] * dvy - bpred[1] * dvz) * dy);
#endif
#ifdef MAGNETIC_DIFFUSION  
                        magfac_diff = (All.MagneticEta + All.MagneticEta) * magfac_sym / (rho_ij * rho_ij);
                        dtB[0] += magfac_diff * rho * dBx;
                        dtB[1] += magfac_diff * rho * dBy;
                        dtB[2] += magfac_diff * rho * dBz;
#ifdef MAGNETIC_DIFFUSION_HEAT
                        if(All.ComovingIntegrationOn)
                            magfac_diff *= (hubble_a * All.Time * All.Time * All.Time * All.Time * All.Time);
                        dtEntropy -= 0.5 * magfac_diff * mu0_1 * (dBx * dBx + dBy * dBy + dBz * dBz);
#endif
#endif
#ifdef MAGFORCE
                        magfac_j = 1 / (SPHP(j).d.Density * SPHP(j).d.Density);
#ifndef MU0_UNITY
                        magfac_j /= (4 * M_PI);
#endif
#ifdef CORRECTBFRC
                        magfac_j *= dwk_j * SPHP(j).h.DhsmlDensityFactor;
                        magfac_i = dwk_i * magfac_i_base;
#else
                        magfac_i = magfac_i_base;
#endif
                        for(k = 0, b2_j = 0; k < 3; k++)
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
                        phifac = magfac_sym * rho / rho_ij;
#ifndef SFR
#ifdef SMOOTH_PHI
                        phifac *= (PhiPred - SPHP(j).SmoothPhi) / (rho_ij);
#else
                        phifac *= (PhiPred - SPHP(j).PhiPred) / (rho_ij);
#endif
#else /* SFR */ 
#ifdef SMOOTH_PHI
                        phifac *= (PhiPred - SPHP(j).SmoothPhi * pow(1.-SPHP(j).XColdCloud,POW_CC)) / (rho_ij);
#else
                        phifac *= (PhiPred - SPHP(j).PhiPred   * pow(1.-SPHP(j).XColdCloud,POW_CC)) / (rho_ij);
#endif 
#endif /* SFR */

                        gradphi[0]+=phifac *dx;
                        gradphi[1]+=phifac *dy;
                        gradphi[2]+=phifac *dz;
#endif
#ifdef MAGNETIC_SIGNALVEL
#ifdef ALFVEN_VEL_LIMITER
                        vcsa2_j = soundspeed_j * soundspeed_j +
                            DMIN(mu0_1 * b2_j / SPHP(j).d.Density,
                                    ALFVEN_VEL_LIMITER * soundspeed_j * soundspeed_j);
#else
                        vcsa2_j = soundspeed_j * soundspeed_j + mu0_1 * b2_j / SPHP(j).d.Density;
#endif
#ifndef SFR
                        Bpro2_j = (SPHP(j).BPred[0] * dx + SPHP(j).BPred[1] * dy + SPHP(j).BPred[2] * dz) / r;
#else
                        Bpro2_j = (SPHP(j).BPred[0] * dx + SPHP(j).BPred[1] * dy + SPHP(j).BPred[2] * dz) * pow(1.-SPHP(j).XColdCloud,2.*POW_CC) / r;
#endif
                        Bpro2_j *= Bpro2_j;
                        magneticspeed_j = sqrt(vcsa2_j +
                                sqrt(DMAX((vcsa2_j * vcsa2_j -
                                            4 * soundspeed_j * soundspeed_j * Bpro2_j
                                            * mu0_1 / SPHP(j).d.Density), 0))) / 1.4142136;
                        Bpro2_i = (bpred[0] * dx + bpred[1] * dy + bpred[2] * dz) / r;
                        Bpro2_i *= Bpro2_i;
                        magneticspeed_i = sqrt(vcsa2_i +
                                sqrt(DMAX((vcsa2_i * vcsa2_i -
                                            4 * soundspeed_i * soundspeed_i * Bpro2_i
                                            * mu0_1 / rho), 0))) / 1.4142136;
#endif
#ifdef MAGNETIC_DISSIPATION
                        dTu_diss_b = -magfac_sym * Balpha_ij * (dBx * dBx + dBy * dBy + dBz * dBz);
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
                            acc[k] +=
#else
                                magacc[k]+=
#endif
                                magfac * ((mm_i[k][0] * magfac_i + mm_j[k][0] * magfac_j) * dx +
                                        (mm_i[k][1] * magfac_i + mm_j[k][1] * magfac_j) * dy +
                                        (mm_i[k][2] * magfac_i + mm_j[k][2] * magfac_j) * dz);
#if defined(DIVBFORCE) && !defined(DIVBFORCE3)
                        for(k = 0; k < 3; k++)
                            acc[k] -=
#ifndef SFR
                                magfac * bpred[k] *(((bpred[0]) * magfac_i + (SPHP(j).BPred[0]) * magfac_j) * dx
                                        + ((bpred[1]) * magfac_i + (SPHP(j).BPred[1]) * magfac_j) * dy
                                        + ((bpred[2]) * magfac_i + (SPHP(j).BPred[2]) * magfac_j) * dz);
#else
                        magfac * (	((bpred[k] * bpred[0]) * magfac_i + (bpred[k] * SPHP(j).BPred[0] * pow(1.-SPHP(j).XColdCloud,2.*POW_CC)) * magfac_j) * dx
                                +   ((bpred[k] * bpred[1]) * magfac_i + (bpred[k] * SPHP(j).BPred[1] * pow(1.-SPHP(j).XColdCloud,2.*POW_CC)) * magfac_j) * dy
                                +   ((bpred[k] * bpred[2]) * magfac_i + (bpred[k] * SPHP(j).BPred[2] * pow(1.-SPHP(j).XColdCloud,2.*POW_CC)) * magfac_j) * dz);
#endif
#endif
#if defined(DIVBFORCE3) && !defined(DIVBFORCE)
                        for(k = 0; k < 3; k++)
                            magcorr[k] +=
#ifndef SFR
                                magfac * bpred[k] *(((bpred[0]) * magfac_i + (SPHP(j).BPred[0]) * magfac_j) * dx
                                        + ((bpred[1]) * magfac_i + (SPHP(j).BPred[1]) * magfac_j) * dy
                                        + ((bpred[2]) * magfac_i + (SPHP(j).BPred[2]) * magfac_j) * dz);
#else
                        magfac * bpred[k] *(((bpred[0]) * magfac_i + (SPHP(j).BPred[0] * pow(1.-SPHP(j).XColdCloud,2.*POW_CC) ) * magfac_j) * dx
                                + ((bpred[1]) * magfac_i + (SPHP(j).BPred[1] * pow(1.-SPHP(j).XColdCloud,2.*POW_CC) ) * magfac_j) * dy
                                + ((bpred[2]) * magfac_i + (SPHP(j).BPred[2] * pow(1.-SPHP(j).XColdCloud,2.*POW_CC) ) * magfac_j) * dz);
#endif
#endif
#endif /* end MAG FORCE   */
#ifdef ALFA_OMEGA_DYN // Known Bug
                        dtB[0] += magfac * alfaomega * All.Tau_A0 / 3.0 * (dBy * dz - dBy * dy);
                        dtB[1] += magfac * alfaomega * All.Tau_A0 / 3.0 * (dBz * dx - dBx * dz);
                        dtB[2] += magfac * alfaomega * All.Tau_A0 / 3.0 * (dBx * dy - dBy * dx);
#endif
#endif /* end of MAGNETIC */


#ifndef MAGNETIC_SIGNALVEL
                        vsig = soundspeed_i + soundspeed_j;
#else
                        vsig = magneticspeed_i + magneticspeed_j;
#endif


#ifndef ALTERNATIVE_VISCOUS_TIMESTEP
                        if(vsig > maxSignalVel)
                            maxSignalVel = vsig;
#endif
                        if(vdotr2 < 0)	/* ... artificial viscosity */
                        {
#ifndef ALTVISCOSITY
#ifndef CONVENTIONAL_VISCOSITY
                            mu_ij = fac_mu * vdotr2 / r;	/* note: this is negative! */
#else
                            c_ij = 0.5 * (soundspeed_i + soundspeed_j);
                            h_ij = 0.5 * (h_i + h_j);
                            mu_ij = fac_mu * h_ij * vdotr2 / (r2 + 0.0001 * h_ij * h_ij);
#endif
#ifdef MAGNETIC
                            vsig -= 1.5 * mu_ij;
#else
                            vsig -= 3 * mu_ij;
#endif


#ifndef ALTERNATIVE_VISCOUS_TIMESTEP
                            if(vsig > maxSignalVel)
                                maxSignalVel = vsig;
#endif

#ifndef NAVIERSTOKES
                            f2 =
                                fabs(SPHP(j).v.DivVel) / (fabs(SPHP(j).v.DivVel) + SPHP(j).r.CurlVel +
                                        0.0001 * soundspeed_j / fac_mu / P[j].Hsml);
#else
                            f2 =
                                fabs(SPHP(j).v.DivVel) / (fabs(SPHP(j).v.DivVel) + SPHP(j).u.s.CurlVel +
                                        0.0001 * soundspeed_j / fac_mu / P[j].Hsml);
#endif

#ifdef NO_SHEAR_VISCOSITY_LIMITER
                            f1 = f2 = 1;
#endif
#ifdef TIME_DEP_ART_VISC
                            BulkVisc_ij = 0.5 * (alpha + SPHP(j).alpha);
#else
                            BulkVisc_ij = All.ArtBulkViscConst;
#endif
#ifndef CONVENTIONAL_VISCOSITY
                            visc = 0.25 * BulkVisc_ij * vsig * (-mu_ij) / rho_ij * (f1 + f2);
#else
                            visc =
                                (-BulkVisc_ij * mu_ij * c_ij + 2 * BulkVisc_ij * mu_ij * mu_ij) /
                                rho_ij * (f1 + f2) * 0.5;
#endif

#else /* start of ALTVISCOSITY block */
                            if(f1 < 0)
                                mu_i = h_i * fabs(f1);	/* f1 hold here the velocity divergence of particle i */
                            else
                                mu_i = 0;
                            if(SPHP(j).v.DivVel < 0)
                                mu_j = h_j * fabs(SPHP(j).v.DivVel);
                            else
                                mu_j = 0;
                            visc = All.ArtBulkViscConst * ((soundspeed_i + mu_i) * mu_i / rho +
                                    (soundspeed_j + mu_j) * mu_j / SPHP(j).d.Density);
#endif /* end of ALTVISCOSITY block */


                            /* .... end artificial viscosity evaluation */
                            /* now make sure that viscous acceleration is not too large */
#ifdef ALTERNATIVE_VISCOUS_TIMESTEP
                            if(visc > 0)
                            {
                                dt = fac_vsic_fix * vdotr2 /
                                    (0.5 * (mass + P[j].Mass) * (dwk_i + dwk_j) * r * visc);

                                dt /= hubble_a;

                                if(dt < minViscousDt)
                                    minViscousDt = dt;
                            }
#endif

#ifndef NOVISCOSITYLIMITER
                            dt =
                                2 * IMAX(timestep,
                                        (P[j].TimeBin ? (1 << P[j].TimeBin) : 0)) * All.Timebase_interval;
                            if(dt > 0 && (dwk_i + dwk_j) < 0)
                            {
#ifdef BLACK_HOLES
                                if((mass + P[j].Mass) > 0)
#endif
                                    visc = DMIN(visc, 0.5 * fac_vsic_fix * vdotr2 /
                                            (0.5 * (mass + P[j].Mass) * (dwk_i + dwk_j) * r * dt));
                            }
#endif
                        }
                        else
                        {
                            visc = 0;
                        }
                        hfc_visc = 0.5 * P[j].Mass * visc * (dwk_i + dwk_j) / r;
#ifndef TRADITIONAL_SPH_FORMULATION

#ifdef DENSITY_INDEPENDENT_SPH
                        hfc = hfc_visc;
                        /* leading-order term */
                        hfc += P[j].Mass *
                            (dwk_i*p_over_rho2_i*SPHP(j).EntVarPred/entvarpred +
                             dwk_j*p_over_rho2_j*entvarpred/SPHP(j).EntVarPred) / r;

                        /* enable grad-h corrections only if contrastlimit is non negative */
                        if(All.DensityContrastLimit >= 0) {
                            double r1 = egyrho / rho;
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
                            /* dhsmlDensityFactor is actually EgyDensityFactor */
                            hfc += P[j].Mass *
                                (dwk_i*p_over_rho2_i*r1*dhsmlDensityFactor +
                                 dwk_j*p_over_rho2_j*r2*SPHP(j).DhsmlEgyDensityFactor) / r;
                        }
#else
                        /* Formulation derived from the Lagrangian */
                        hfc = hfc_visc + P[j].Mass * (p_over_rho2_i *dhsmlDensityFactor * dwk_i 
                                + p_over_rho2_j * SPHP(j).h.DhsmlDensityFactor * dwk_j) / r;
#endif
#else
                        hfc = hfc_visc +
                            0.5 * P[j].Mass * (dwk_i + dwk_j) / r * (p_over_rho2_i + p_over_rho2_j);

                        /* hfc_egy = 0.5 * P[j].Mass * (dwk_i + dwk_j) / r * (p_over_rho2_i + p_over_rho2_j); */
                        hfc_egy = P[j].Mass * (dwk_i + dwk_j) / r * (p_over_rho2_i);
#endif

#ifdef WINDS
                        if(P[j].Type == 0)
                            if(SPHP(j).DelayTime > 0)	/* No force by wind particles */
                            {
                                hfc = hfc_visc = 0;
                            }
#endif

#ifndef NOACCEL
                        acc[0] += FLT(-hfc * dx);
                        acc[1] += FLT(-hfc * dy);
                        acc[2] += FLT(-hfc * dz);
#endif

#if !defined(EOS_DEGENERATE) && !defined(TRADITIONAL_SPH_FORMULATION)
                        dtEntropy += FLT(0.5 * hfc_visc * vdotr2);
#else

#ifdef TRADITIONAL_SPH_FORMULATION
                        dtEntropy += FLT(0.5 * (hfc_visc + hfc_egy) * vdotr2);
#else
                        dtEntropy += FLT(0.5 * hfc * vdotr2);
#endif
#endif


#ifdef NAVIERSTOKES
                        faci = mass * shear_viscosity / (rho * rho) * dwk_i / r;

#ifndef NAVIERSTOKES_CONSTANT
                        faci *= pow((Entropy * pow(rho * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1), 2.5);	/*multiplied by E^5/2 */
#endif
                        facj = P[j].Mass * get_shear_viscosity(j) /
                            (SPHP(j).d.Density * SPHP(j).d.Density) * dwk_j / r;

#ifndef NAVIERSTOKES_CONSTANT
                        facj *= pow((SPHP(j).Entropy * pow(SPHP(j).d.Density * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1), 2.5);	/*multiplied by E^5/2 */
#endif

#ifdef NAVIERSTOKES_BULK
                        facbi = mass * All.NavierStokes_BulkViscosity / (rho * rho) * dwk_i / r;
                        facbj = P[j].Mass * All.NavierStokes_BulkViscosity /
                            (SPHP(j).d.Density * SPHP(j).d.Density) * dwk_j / r;
#endif

#ifdef WINDS
                        if(P[j].Type == 0)
                            if(SPHP(j).DelayTime > 0)	/* No visc for wind particles */
                            {
                                faci = facj = 0;
#ifdef NAVIERSTOKES_BULK
                                facbi = facbj = 0;
#endif
                            }
#endif

#ifdef VISCOSITY_SATURATION
                        IonMeanFreePath_i = All.IonMeanFreePath * pow((Entropy * pow(rho * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1), 2.0) / rho;	/* u^2/rho */

                        IonMeanFreePath_j = All.IonMeanFreePath * pow((SPHP(j).Entropy * pow(SPHP(j).d.Density * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1), 2.0) / SPHP(j).d.Density;	/* u^2/rho */

                        for(k = 0, VelLengthScale_i = 0, VelLengthScale_j = 0; k < 3; k++)
                        {
                            if(fabs(stressdiag[k]) > 0)
                            {
                                VelLengthScale_i = 2 * soundspeed_i / fabs(stressdiag[k]);

                                if(VelLengthScale_i < IonMeanFreePath_i && VelLengthScale_i > 0)
                                {
                                    stressdiag[k] = stressdiag[k] * (VelLengthScale_i / IonMeanFreePath_i);

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
                            if(fabs(stressoffdiag[k]) > 0)
                            {
                                VelLengthScale_i = 2 * soundspeed_i / fabs(stressoffdiag[k]);

                                if(VelLengthScale_i < IonMeanFreePath_i && VelLengthScale_i > 0)
                                {
                                    stressoffdiag[k] =
                                        stressoffdiag[k] * (VelLengthScale_i / IonMeanFreePath_i);
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
                        acc[0] += faci * (stressdiag[0] * dx + stressoffdiag[0] * dy + stressoffdiag[1] * dz)
                            + facj * (SPHP(j).u.s.StressDiag[0] * dx + SPHP(j).u.s.StressOffDiag[0] * dy +
                                    SPHP(j).u.s.StressOffDiag[1] * dz);

                        acc[1] += faci * (stressoffdiag[0] * dx + stressdiag[1] * dy + stressoffdiag[2] * dz)
                            + facj * (SPHP(j).u.s.StressOffDiag[0] * dx + SPHP(j).u.s.StressDiag[1] * dy +
                                    SPHP(j).u.s.StressOffDiag[2] * dz);

                        acc[2] += faci * (stressoffdiag[1] * dx + stressoffdiag[2] * dy + stressdiag[2] * dz)
                            + facj * (SPHP(j).u.s.StressOffDiag[1] * dx + SPHP(j).u.s.StressOffDiag[2] * dy +
                                    SPHP(j).u.s.StressDiag[2] * dz);

                        /*Acceleration due to the bulk viscosity */
#ifdef NAVIERSTOKES_BULK
#ifdef VISCOSITY_SATURATION
                        VelLengthScale_i = 0;
                        VelLengthScale_j = 0;

                        if(fabs(divvel) > 0)
                        {
                            VelLengthScale_i = 3 * soundspeed_i / fabs(divvel);

                            if(VelLengthScale_i < IonMeanFreePath_i && VelLengthScale_i > 0)
                            {
                                divvel = divvel * (VelLengthScale_i / IonMeanFreePath_i);
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


                        acc[0] += facbi * divvel * dx + facbj * SPHP(j).u.s.a4.DivVel * dx;
                        acc[1] += facbi * divvel * dy + facbj * SPHP(j).u.s.a4.DivVel * dy;
                        acc[2] += facbi * divvel * dz + facbj * SPHP(j).u.s.a4.DivVel * dz;
#endif
#endif /* end NAVIERSTOKES */


#ifdef MAGNETIC
#ifdef EULER_DISSIPATION
                        alpha_ij_eul = All.ArtMagDispConst;

                        dteulA +=
                            alpha_ij_eul * 0.5 * vsig * (eulA -
                                    SPHP(j).EulerA) * magfac_sym * r * rho / (rho_ij *
                                    rho_ij);
                        dteulB +=
                            alpha_ij_eul * 0.5 * vsig * (eulB -
                                    SPHP(j).EulerB) * magfac_sym * r * rho / (rho_ij *
                                    rho_ij);

                        dTu_diss_eul = -magfac_sym * alpha_ij_eul * (dBx * dBx + dBy * dBy + dBz * dBz);
                        dtEntropy += dTu_diss_eul * 0.25 * vsig * mu0_1 * r / (rho_ij * rho_ij);
#endif
#ifdef MAGNETIC_DISSIPATION
                        magfac_sym *= vsig * 0.5 * Balpha_ij * r * rho / (rho_ij * rho_ij);
                        dtEntropy += dTu_diss_b * 0.25 * vsig * mu0_1 * r / (rho_ij * rho_ij);
                        dtB[0] += magfac_sym * dBx;
                        dtB[1] += magfac_sym * dBy;
                        dtB[2] += magfac_sym * dBz;
#endif
#endif

#ifdef WAKEUP
                        if(vsig > WAKEUP * SPHP(j).MaxSignalVel)
                        {
                            SPHP(j).wakeup = 1;
                        }
#endif
                    }
                }
            }
        }

#ifndef DONOTUSENODELIST
        if(mode == 1)
        {
            listindex++;
            if(listindex < NODELISTLENGTH)
            {
                startnode = HydroDataGet[target].NodeList[listindex];
                if(startnode >= 0)
                    startnode = Nodes[startnode].u.d.nextnode;	/* open it */
            }
        }
#endif
    }

    /* Now collect the result at the right place */
    if(mode == 0)
    {
        for(k = 0; k < 3; k++)
            SPHP(target).a.dHydroAccel[k] = acc[k];
        SPHP(target).e.dDtEntropy = dtEntropy;
#ifdef DIVBFORCE3
        for(k = 0; k < 3; k++)
            SPHP(target).magacc[k] = magacc[k];
        for(k = 0; k < 3; k++)
            SPHP(target).magcorr[k] = magcorr[k];
#endif
#ifdef HYDRO_COST_FACTOR
        if(All.ComovingIntegrationOn)
            P[target].GravCost += HYDRO_COST_FACTOR * All.Time * ninteractions;
        else
            P[target].GravCost += HYDRO_COST_FACTOR * ninteractions;
#endif

#ifdef ALTERNATIVE_VISCOUS_TIMESTEP
        SPHP(target).MinViscousDt = minViscousDt;
#else
        SPHP(target).MaxSignalVel = maxSignalVel;
#endif

#ifdef JD_VTURB
        SPHP(target).Vrms = vRms; 
#endif

#if defined(MAGNETIC) && ( !defined(EULERPOTENTIALS) || !defined(VECT_POTENTIAL) )
        for(k = 0; k < 3; k++)
            SPHP(target).DtB[k] = dtB[k];
#endif
#ifdef DIVBCLEANING_DEDNER
        for(k = 0; k < 3; k++)
            SPHP(target).GradPhi[k] = gradphi[k];
#endif
#ifdef VECT_POTENTIAL
        SPHP(target).DtA[0] = dta[0];
        SPHP(target).DtA[1] = dta[1];
        SPHP(target).DtA[2] = dta[2];
#endif
#ifdef EULER_DISSIPATION
        SPHP(target).DtEulerA = dteulA;
        SPHP(target).DtEulerB = dteulB;
#endif
    }
    else
    {
        for(k = 0; k < 3; k++)
            HydroDataResult[target].Acc[k] = acc[k];
        HydroDataResult[target].DtEntropy = dtEntropy;
#ifdef DIVBFORCE3
        for(k = 0; k < 3; k++)
            HydroDataResult[target].magacc[k] = magacc[k];
        for(k = 0; k < 3; k++)
            HydroDataResult[target].magcorr[k] = magcorr[k];
#endif
#ifdef HYDRO_COST_FACTOR
        HydroDataResult[target].Ninteractions = ninteractions;
#endif

#ifdef ALTERNATIVE_VISCOUS_TIMESTEP
        HydroDataResult[target].MinViscousDt = minViscousDt;
#else
        HydroDataResult[target].MaxSignalVel = maxSignalVel;
#endif
#ifdef JD_VTURB
        HydroDataResult[target].Vrms = vRms; 
#endif
#if defined(MAGNETIC) && ( !defined(EULERPOTENTIALS) || !defined(VECT_POTENTIAL) )
        for(k = 0; k < 3; k++)
            HydroDataResult[target].DtB[k] = dtB[k];
#ifdef DIVBCLEANING_DEDNER
        for(k = 0; k < 3; k++)
            HydroDataResult[target].GradPhi[k] = gradphi[k];
#endif
#endif
#ifdef VECT_POTENTIAL
        HydroDataResult[target].dta[0] = dta[0];
        HydroDataResult[target].dta[1] = dta[1];
        HydroDataResult[target].dta[2] = dta[2];
#endif
#ifdef EULER_DISSIPATION
        HydroDataResult[target].DtEulerA = dteulA;
        HydroDataResult[target].DtEulerB = dteulB;
#endif
    }

    return 0;
}

static void * hydro_alloc_ngblist() {
    int threadid = omp_get_thread_num();
    return Ngblist + threadid * NumPart;
}
static int hydro_isactive(int i) {
    return P[i].Type == 0;
}

