#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <gsl/gsl_sf_gamma.h>

#include "allvars.h"
#include "proto.h"

/*! \file init.c
 *  \brief code for initialisation of a simulation from initial conditions
 */


/*! This function reads the initial conditions, and allocates storage for the
 *  tree(s). Various variables of the particle data are initialised and An
 *  intial domain decomposition is performed. If SPH particles are present,
 *  the inial SPH smoothing lengths are determined.
 */
void init(void)
{
    int i, j;
    double a3, atime;

#if defined (CHEMISTRY) || defined (UM_CHEMISTRY)
    int ifunc;
    double min_t_cool, max_t_cool;
    double min_t_elec, max_t_elec;
    double a_start, a_end;
#endif  

#ifdef EULERPOTENTIALS
    double a0, a1, a2;
    double b0, b1, b2;
#endif

#ifdef START_WITH_EXTRA_NGBDEV
    double MaxNumNgbDeviationMerk;
#endif

#ifdef DISTORTIONTENSORPS
    int i1, i2;
#endif

    set_global_time(All.TimeBegin);

    if(RestartFlag == 3 && RestartSnapNum < 0)
    {
        if(ThisTask == 0)
            printf("Need to give the snapshot number if FOF is selected for output\n");
        endrun(0);
    }

    if(RestartFlag == 4 && RestartSnapNum < 0)
    {
        if(ThisTask == 0)
            printf("Need to give the snapshot number if snapshot should be converted\n");
        endrun(0);
    }

    if(RestartFlag >= 2 && RestartSnapNum >= 0)  {
        petaio_read_snapshot(RestartSnapNum);
    } else 
    if(RestartFlag == 0) {
        petaio_read_ic();
    } else {
        if(ThisTask == 0) {
            fprintf(stderr, "RestartFlag and SnapNum comination is unknown");
        }
        abort();
    }

    /* this ensures the initial BhP array is consistent */
    domain_garbage_collection_bh();

    set_global_time(All.TimeBegin);
#ifdef COOLING
    IonizeParams();
#endif

#if defined (CHEMISTRY) || defined (UM_CHEMISTRY)
    InitChem();
#endif

    if(All.ComovingIntegrationOn)
    {
        All.Timebase_interval = (log(All.TimeMax) - log(All.TimeBegin)) / TIMEBASE;
        All.Ti_Current = 0;
        a3 = All.Time * All.Time * All.Time;
        atime = All.Time;
    }
    else
    {
        All.Timebase_interval = (All.TimeMax - All.TimeBegin) / TIMEBASE;
        All.Ti_Current = 0;
        a3 = 1;
        atime = 1;
    }

#ifdef RADTRANSFER
    All.Radiation_Ti_begstep = 0;
#endif


    set_softenings();

    All.NumCurrentTiStep = 0;	/* setup some counters */
    All.SnapshotFileCount = 0;
    if(RestartFlag == 2)
    {
        if(RestartSnapNum < 0)
            All.SnapshotFileCount = atoi(All.InitCondFile + strlen(All.InitCondFile) - 3) + 1;
        else
            All.SnapshotFileCount = RestartSnapNum + 1;
    }

#ifdef OUTPUTLINEOFSIGHT
    All.Ti_nextlineofsight = (int) (log(All.TimeFirstLineOfSight / All.TimeBegin) / All.Timebase_interval);
    if(RestartFlag == 2)
        endrun(78787);
#endif

    All.TotNumOfForces = 0;
    All.NumForcesSinceLastDomainDecomp = 0;
#if defined(MAGNETIC) && defined(BSMOOTH)
#ifdef SETMAINTIMESTEPCOUNT
    All.MainTimestepCounts = All.MainTimestepCountIni;
#else
    All.MainTimestepCounts = 0;
#endif
#endif

    All.TreeAllocFactor = 0.7;


    All.Cadj_Cost = 1.0e-30;
    All.Cadj_Cpu = 1.0e-3;

    if(All.ComovingIntegrationOn)
        if(All.PeriodicBoundariesOn == 1)
            check_omega();

    All.TimeLastStatistics = All.TimeBegin - All.TimeBetStatistics;
#ifdef BLACK_HOLES
    All.TimeNextBlackHoleCheck = All.TimeBegin;
#endif

    /* DISTORTION GENERAL SETUP */
#ifdef DISTORTIONTENSORPS
#if defined(SIM_ADAPTIVE_SOFT) || defined(REINIT_AT_TURNAROUND)
#ifdef REINIT_AT_TURNAROUND
    All.SIM_epsilon = REINIT_AT_TURNAROUND;
#endif
#ifdef SIM_ADAPTIVE_SOFT
    All.SIM_epsilon = SIM_ADAPTIVE_SOFT;
#endif
    All.cms_x = 0.0;
    All.cms_y = 0.0;
    All.cms_z = 0.0;
#ifdef COMOVING_DISTORTION
    All.InitialTurnaroundRadius = COMOVING_DISTORTION * pow(All.TimeBegin, 1.0 / (3.0 * All.SIM_epsilon));
    if(ThisTask == 0)
    {
        printf("REINIT_AT_TURNAROUND: comoving initial/final turnaround radius = %g/%g\n",
                All.InitialTurnaroundRadius, COMOVING_DISTORTION);
        fflush(stdout);
    }
#else
    double turnaround_radius_local = 0.0, turnaround_radius_global = 0.0, v_part, r_part;

    /* get local turnaroundradius */
    for(i = 0; i < NumPart; i++)
    {
        r_part = sqrt(P[i].Pos[0] * P[i].Pos[0] + P[i].Pos[1] * P[i].Pos[1] + P[i].Pos[2] * P[i].Pos[2]);
        v_part = (P[i].Pos[0] * P[i].Vel[0] + P[i].Pos[1] * P[i].Vel[1] + P[i].Pos[2] * P[i].Vel[2]);
        if((v_part < 0.0) && (r_part > turnaround_radius_local))
            turnaround_radius_local = r_part;
    }

    /* find global turnaround radius by taking maximum of all CPUs */
    MPI_Allreduce(&turnaround_radius_local, &turnaround_radius_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    All.InitialTurnaroundRadius = turnaround_radius_global;

#endif
    All.CurrentTurnaroundRadius = All.InitialTurnaroundRadius;
    if(ThisTask == 0)
    {
        printf("REINIT_AT_TURNAROUND: current (initial) turnaround radius = %g\n", All.CurrentTurnaroundRadius);
        fflush(stdout);
    }
#endif
#endif


    for(i = 0; i < NumPart; i++)	/*  start-up initialization */
    {
        for(j = 0; j < 3; j++) {
            P[i].GravAccel[j] = 0;
        }
        /* DISTORTION PARTICLE SETUP */
#ifdef DISTORTIONTENSORPS
        /*init tidal tensor for first output (not used for calculation) */
        for(i1 = 0; i1 < 3; i1++)
            for(i2 = 0; i2 < 3; i2++)
                P[i].tidal_tensorps[i1][i2] = 0.0;

        /* find caustics by sign analysis of configuration space distortion */
        P[i].last_stream_determinant = 1.0;

        /* distortion tensor alrady read from ICs -> get correct last_stream_determinant */
#ifdef DISTORTION_READALL
        MyDouble product_matrix[3][3];

        product_matrix[0][0] = P[i].distortion_tensorps[0][0] +
            P[i].distortion_tensorps[0][3] * P[i].V_matrix[0][0] +
            P[i].distortion_tensorps[0][4] * P[i].V_matrix[1][0] +
            P[i].distortion_tensorps[0][5] * P[i].V_matrix[2][0];
        product_matrix[0][1] = P[i].distortion_tensorps[0][1] +
            P[i].distortion_tensorps[0][3] * P[i].V_matrix[0][1] +
            P[i].distortion_tensorps[0][4] * P[i].V_matrix[1][1] +
            P[i].distortion_tensorps[0][5] * P[i].V_matrix[2][1];
        product_matrix[0][2] = P[i].distortion_tensorps[0][2] +
            P[i].distortion_tensorps[0][3] * P[i].V_matrix[0][2] +
            P[i].distortion_tensorps[0][4] * P[i].V_matrix[1][2] +
            P[i].distortion_tensorps[0][5] * P[i].V_matrix[2][2];
        product_matrix[1][0] = P[i].distortion_tensorps[1][0] +
            P[i].distortion_tensorps[1][3] * P[i].V_matrix[0][0] +
            P[i].distortion_tensorps[1][4] * P[i].V_matrix[1][0] +
            P[i].distortion_tensorps[1][5] * P[i].V_matrix[2][0];
        product_matrix[1][1] = P[i].distortion_tensorps[1][1] +
            P[i].distortion_tensorps[1][3] * P[i].V_matrix[0][1] +
            P[i].distortion_tensorps[1][4] * P[i].V_matrix[1][1] +
            P[i].distortion_tensorps[1][5] * P[i].V_matrix[2][1];
        product_matrix[1][2] = P[i].distortion_tensorps[1][2] +
            P[i].distortion_tensorps[1][3] * P[i].V_matrix[0][2] +
            P[i].distortion_tensorps[1][4] * P[i].V_matrix[1][2] +
            P[i].distortion_tensorps[1][5] * P[i].V_matrix[2][2];
        product_matrix[2][0] = P[i].distortion_tensorps[2][0] +
            P[i].distortion_tensorps[2][3] * P[i].V_matrix[0][0] +
            P[i].distortion_tensorps[2][4] * P[i].V_matrix[1][0] +
            P[i].distortion_tensorps[2][5] * P[i].V_matrix[2][0];
        product_matrix[2][1] = P[i].distortion_tensorps[2][1] +
            P[i].distortion_tensorps[2][3] * P[i].V_matrix[0][1] +
            P[i].distortion_tensorps[2][4] * P[i].V_matrix[1][1] +
            P[i].distortion_tensorps[2][5] * P[i].V_matrix[2][1];
        product_matrix[2][2] = P[i].distortion_tensorps[2][2] +
            P[i].distortion_tensorps[2][3] * P[i].V_matrix[0][2] +
            P[i].distortion_tensorps[2][4] * P[i].V_matrix[1][2] +
            P[i].distortion_tensorps[2][5] * P[i].V_matrix[2][2];

        /* this determinant will change sign when we pass through a caustic -> criterion for caustics */
        P[i].last_stream_determinant =
            ((product_matrix[0][0]) * (product_matrix[1][1]) * (product_matrix[2][2]) +
             (product_matrix[0][1]) * (product_matrix[1][2]) * (product_matrix[2][0]) +
             (product_matrix[0][2]) * (product_matrix[1][0]) * (product_matrix[2][1]) -
             (product_matrix[0][2]) * (product_matrix[1][1]) * (product_matrix[2][0]) -
             (product_matrix[0][0]) * (product_matrix[1][2]) * (product_matrix[2][1]) -
             (product_matrix[0][1]) * (product_matrix[1][0]) * (product_matrix[2][2]));


#endif

#ifdef REINIT_AT_TURNAROUND
        /* no phase-space analysis for particles that are initially within the turnaround radius */
        if(sqrt(P[i].Pos[0] * P[i].Pos[0] + P[i].Pos[1] * P[i].Pos[1] + P[i].Pos[2] * P[i].Pos[2]) <
                All.CurrentTurnaroundRadius)
#ifdef DISTORTION_READALL
            P[i].turnaround_flag = 1;
#else
        P[i].turnaround_flag = -1;
#endif
        else
            P[i].turnaround_flag = 0;
#endif

#ifdef OUTPUT_LAST_CAUSTIC
        /* all entries zero -> no caustic yet */
        P[i].lc_Time = 0.0;
        P[i].lc_Pos[0] = 0.0;
        P[i].lc_Pos[1] = 0.0;
        P[i].lc_Pos[2] = 0.0;
        P[i].lc_Vel[0] = 0.0;
        P[i].lc_Vel[1] = 0.0;
        P[i].lc_Vel[2] = 0.0;
        P[i].lc_rho_normed_cutoff = 0.0;

        P[i].lc_Dir_x[0] = 0.0;
        P[i].lc_Dir_x[1] = 0.0;
        P[i].lc_Dir_x[2] = 0.0;
        P[i].lc_Dir_y[0] = 0.0;
        P[i].lc_Dir_y[1] = 0.0;
        P[i].lc_Dir_y[2] = 0.0;
        P[i].lc_Dir_z[0] = 0.0;
        P[i].lc_Dir_z[1] = 0.0;
        P[i].lc_Dir_z[2] = 0.0;

        P[i].lc_smear_x = 0.0;
        P[i].lc_smear_y = 0.0;
        P[i].lc_smear_z = 0.0;
#endif


#ifdef PETAPM
        /* long range tidal field init */
        P[i].tidal_tensorpsPM[0][0] = 0;
        P[i].tidal_tensorpsPM[0][1] = 0;
        P[i].tidal_tensorpsPM[0][2] = 0;
        P[i].tidal_tensorpsPM[1][0] = 0;
        P[i].tidal_tensorpsPM[1][1] = 0;
        P[i].tidal_tensorpsPM[1][2] = 0;
        P[i].tidal_tensorpsPM[2][0] = 0;
        P[i].tidal_tensorpsPM[2][1] = 0;
        P[i].tidal_tensorpsPM[2][2] = 0;
#endif

#ifndef DISTORTION_READALL
        for(i1 = 0; i1 < 6; i1++)
            for(i2 = 0; i2 < 6; i2++)
            {
                if((i1 == i2))
                    P[i].distortion_tensorps[i1][i2] = 1.0;
                else
                    P[i].distortion_tensorps[i1][i2] = 0.0;
            }
#endif

#ifdef COMOVING_DISTORTION
        /* for cosmological simulations we do init here, not read from ICs */

        /* Lagrange time of particle */
        P[i].a0 = All.TimeBegin;

#ifndef COMOVING_READIC
        /* no caustic passages in the beginning */
        P[i].caustic_counter = 0.0;

        /* approximation: perfect Hubble Flow -> peculiar sheet orientation is exactly zero */
        for(i1 = 0; i1 < 3; i1++)
            for(i2 = 0; i2 < 3; i2++)
                P[i].V_matrix[i1][i2] = 0.0;

        /* approximation: initial sream density equals background density */
        P[i].init_density = All.Omega0 * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.G);
#endif
#endif

        /* annihilation stuff */
        P[i].s_1_last = 1.0;
        P[i].s_2_last = 1.0;
        P[i].s_3_last = 1.0;
        P[i].second_deriv_last = 0.0;
        P[i].rho_normed_cutoff_last = 1.0;

        P[i].s_1_current = 1.0;
        P[i].s_2_current = 1.0;
        P[i].s_3_current = 1.0;
        P[i].second_deriv_current = 0.0;
        P[i].rho_normed_cutoff_current = 1.0;

        P[i].annihilation = 0.0;
        P[i].analytic_caustics = 0.0;
        P[i].analytic_annihilation = 0.0;

#ifdef COMOVING_DISTORTION
        P[i].stream_density = P[i].init_density / (All.TimeBegin * All.TimeBegin * All.TimeBegin);
#else
        P[i].stream_density = P[i].init_density;
#endif

#endif /* DISTORTIONTENSORPS */

#ifdef KEEP_DM_HSML_AS_GUESS
        if(RestartFlag != 1)
            P[i].DM_Hsml = -1;
#endif

#ifdef PETAPM
        for(j = 0; j < 3; j++)
            P[i].GravPM[j] = 0;
#endif
        P[i].Ti_begstep = 0;
        P[i].Ti_current = 0;
        P[i].DensityIterationDone = 0;
        P[i].OnAnotherDomain = 0;
        P[i].WillExport = 0;
        P[i].Generation = 0;
        P[i].TimeBin = 0;

        if(header.flag_ic_info != FLAG_SECOND_ORDER_ICS)
            P[i].OldAcc = 0;	/* Do not zero in 2lpt case as masses are stored here */

        P[i].GravCost = 1;

        if(RestartFlag < 3)
            P[i].Potential = 0;

#ifdef STELLARAGE
        if(RestartFlag == 0)
            P[i].StellarAge = 0;
#endif

#ifdef METALS
        if(RestartFlag == 0)
            P[i].Metallicity = 0;
#endif

#ifdef BLACK_HOLES
        if(P[i].Type == 5)
        {
            if(RestartFlag == 0)
                BHP(i).Mass = All.SeedBlackHoleMass;
        }
#endif
    }

    for(i = 0; i < TIMEBINS; i++)
        TimeBinActive[i] = 1;

    reconstruct_timebins();

#ifdef PETAPM
    All.PM_Ti_endstep = All.PM_Ti_begstep = 0;
#endif

#ifdef CONDUCTION
    All.Conduction_Ti_endstep = All.Conduction_Ti_begstep = 0;
#endif
#ifdef CR_DIFFUSION
    All.CR_Diffusion_Ti_endstep = All.CR_Diffusion_Ti_begstep = 0;
#endif

    for(i = 0; i < N_sph; i++)	/* initialize sph_properties */
    {
        for(j = 0; j < 3; j++)
        {
            SPHP(i).VelPred[j] = P[i].Vel[j];
            SPHP(i).HydroAccel[j] = 0;
        }

        SPHP(i).DtEntropy = 0;

#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
        SPHP(i).Gamma = GAMMA;	/* set universal value */
        SPHP(i).t_cool = 0;
        SPHP(i).t_elec = 0;
#endif

        if(RestartFlag == 0)
        {
#ifndef READ_HSML
            P[i].Hsml = 0;
#endif
            SPHP(i).Density = -1;
#ifdef DENSITY_INDEPENDENT_SPH
            SPHP(i).EgyWtDensity = -1;
            SPHP(i).EntVarPred = -1;
#endif
#ifdef VOLUME_CORRECTION
            SPHP(i).DensityOld = 1;
#endif
#ifdef COOLING
#ifndef UM_CHEMISTRY          
            SPHP(i).Ne = 1.0;
#endif
#endif
#ifdef CHEMCOOL
            SPHP(i).TracAbund[IH2] = All.InitMolHydroAbund;
            SPHP(i).TracAbund[IHP] = All.InitHPlusAbund;
            SPHP(i).TracAbund[IDP] = All.InitDIIAbund;
            SPHP(i).TracAbund[IHD] = All.InitHDAbund;
            SPHP(i).TracAbund[IHEP] = All.InitHeIIAbund;
            SPHP(i).TracAbund[IHEPP] = All.InitHeIIIAbund;
#endif
            SPHP(i).DivVel = 0;
        }
#ifdef WINDS
        SPHP(i).DelayTime = 0;
#endif
#ifdef SFR
        SPHP(i).Sfr = 0;
#endif
#if defined (UM_CHEMISTRY) && defined (UM_CHEMISTRY_INISET)
        SPHP(i).elec = All.Startelec;

        SPHP(i).HI = All.StartHI;
        SPHP(i).HII = All.StartHII;
        SPHP(i).HM = All.StartHM;

        SPHP(i).HeI = All.StartHeI;
        SPHP(i).HeII = All.StartHeII;
        SPHP(i).HeIII = All.StartHeIII;

        SPHP(i).H2I = All.StartH2I;
        SPHP(i).H2II = All.StartH2II;

        SPHP(i).HD = All.StartHD;
        SPHP(i).DI = All.StartDI;
        SPHP(i).DII = All.StartDII;

        SPHP(i).HeHII = All.StartHeHII;
#endif      
#ifdef MAGNETIC
#ifdef BINISET
        if (RestartFlag == 0){ /* Set only when starting from ICs */
            SPHP(i).BPred[0] = All.BiniX;
            SPHP(i).BPred[1] = All.BiniY;
            SPHP(i).BPred[2] = All.BiniZ;
        }
#ifdef VECT_POTENTIAL
        if(All.BiniY == 0 && All.BiniZ == 0)
        {
            SPHP(i).APred[0] = 0.;
            SPHP(i).APred[1] = 0.;
            SPHP(i).APred[2] = (All.BiniX * P[i].Pos[1]) * atime * All.HubbleParam;
            SPHP(i).A[0] = SPHP(i).APred[0];
            SPHP(i).A[1] = SPHP(i).APred[1];
            SPHP(i).A[2] = SPHP(i).APred[2];
            SPHP(i).DtA[0] = SPHP(i).DtA[1] = SPHP(i).DtA[2] = 0.;
        }
        else
        {
            if(ThisTask == 0)
            {
                printf("Can not reconstruct VECTOR potentials from Bini values !\n");
                endrun(6724);
            }
        }
#endif
#ifdef EULERPOTENTIALS
        if(All.BiniY == 0 && All.BiniZ == 0)
        {
            a0 = 0;
            a1 = All.BiniX;
            a2 = 0;
            b0 = 0;
            b1 = 1;
            b2 = 1;
        }
        else
        {
            if(All.BiniX != 0 && (All.BiniY != 0 || All.BiniZ != 0))
            {
                b0 = -(All.BiniZ + All.BiniY) / All.BiniX;
                b1 = 1;
                b2 = 1;
                a0 = 0;
                a1 = -All.BiniZ / b0;
                a2 = All.BiniY / b0;
            }
            else
            {
                a0 = a1 = a2 = b0 = b1 = b2 = 0;
                if(ThisTask == 0)
                {
                    printf("Can not reconstruct Euler potentials from Bini values !\n");
                    endrun(6723);
                }
            }
        }
        SPHP(i).EulerA = (a0 * P[i].Pos[0] + a1 * P[i].Pos[1] + a2 * P[i].Pos[2]) * atime * All.HubbleParam;
        SPHP(i).EulerB = (b0 * P[i].Pos[0] + b1 * P[i].Pos[1] + b2 * P[i].Pos[2]) * atime * All.HubbleParam;
#endif
#endif /*BINISET*/
#ifndef EULERPOTENTIALS
        for(j = 0; j < 3; j++)
        {
#ifndef VECT_POTENTIAL
            SPHP(i).DtB[j] = 0;
#endif
            SPHP(i).B[j] = SPHP(i).BPred[j];
        }
#endif
#ifdef TIME_DEP_MAGN_DISP
#ifdef HIGH_MAGN_DISP_START
        SPHP(i).Balpha = All.ArtMagDispConst;
#else
        SPHP(i).Balpha = All.ArtMagDispMin;
#endif
        SPHP(i).DtBalpha = 0.0;
#endif
#ifdef DIVBCLEANING_DEDNER
        SPHP(i).Phi = SPHP(i).PhiPred = SPHP(i).DtPhi = 0;
#endif
#endif

#ifdef TIME_DEP_ART_VISC
#ifdef HIGH_ART_VISC_START
        if(HIGH_ART_VISC_START == 0)
            SPHP(i).alpha = All.ArtBulkViscConst;
        if(HIGH_ART_VISC_START > 0)
            if(P[i].Pos[0] > HIGH_ART_VISC_START)
                SPHP(i).alpha = All.ArtBulkViscConst;
            else
                SPHP(i).alpha = All.AlphaMin;
        if(HIGH_ART_VISC_START < 0)
            if(P[i].Pos[0] < -HIGH_ART_VISC_START)
                SPHP(i).alpha = All.ArtBulkViscConst;
            else
                SPHP(i).alpha = All.AlphaMin;
#else
        SPHP(i).alpha = All.AlphaMin;
#endif
        SPHP(i).Dtalpha = 0.0;
#endif

#if defined(BH_THERMALFEEDBACK) || defined(BH_KINETICFEEDBACK)
        SPHP(i).Injected_BH_Energy = 0;
#endif
    }

#ifdef TWODIMS
    for(i = 0; i < NumPart; i++)
    {
        P[i].Pos[2] = 0;
        P[i].Vel[2] = 0;

        P[i].GravAccel[2] = 0;

        if(P[i].Type == 0)
        {
            SPHP(i).VelPred[2] = 0;
            SPHP(i).HydroAccel[2] = 0;
        }
    }
#endif

#ifdef ONEDIM
    for(i = 0; i < NumPart; i++)
    {
        P[i].Pos[1] = P[i].Pos[2] = 0;
        P[i].Vel[1] = P[i].Vel[2] = 0;

        P[i].GravAccel[1] = P[i].GravAccel[2] = 0;

        if(P[i].Type == 0)
        {
            SPHP(i).VelPred[1] = SPHP(i).VelPred[2] = 0;
            SPHP(i).HydroAccel[1] =SPHP(i).HydroAccel[2] = 0;
        }
    }
#endif

#ifdef ASSIGN_NEW_IDS
    assign_unique_ids();
#endif

#ifndef NOTEST_FOR_IDUNIQUENESS
    test_id_uniqueness();
#endif


    Flag_FullStep = 1;		/* to ensure that Peano-Hilber order is done */

    domain_Decomposition();	/* do initial domain decomposition (gives equal numbers of particles) */

    set_softenings();

    /* will build tree */
    ngb_treebuild();

    All.Ti_Current = 0;

#ifdef START_WITH_EXTRA_NGBDEV
    MaxNumNgbDeviationMerk = All.MaxNumNgbDeviation;
    All.MaxNumNgbDeviation = All.MaxNumNgbDeviationStart;
#endif

    if(RestartFlag != 3)
        setup_smoothinglengths();

#ifdef START_WITH_EXTRA_NGBDEV
    All.MaxNumNgbDeviation = MaxNumNgbDeviationMerk;
#endif

#if defined(HEALPIX)

    compute_global_quantities_of_system();

    //this should be readed in the parameterfile
    All.Nside = 32;
    //
    if(ThisTask == 0)
        printf(" First calculation of Healpix %i with %i \n", All.Nside, NSIDE2NPIX(All.Nside));
    // initialize the healpix array (just in case)
    All.healpixmap = (float *) malloc(NSIDE2NPIX(All.Nside) * sizeof(float));
    for(i = 0; i < NSIDE2NPIX(All.Nside); i++)
        All.healpixmap[i] = 0;

    double Minmass, Maxmass;

    All.Maxmass = 0.0;
    Minmass = 1E10;
    for(i = 0; i < NumPart; i++)
    {
        if(P[i].Type == 1)
        {
            if(P[i].Mass != 0.0)
                Minmass = P[i].Mass < Minmass ? P[i].Mass : Minmass;
            Maxmass = P[i].Mass > Maxmass ? P[i].Mass : Maxmass;
        }
    }
    All.Minmass = Maxmass;

    if(ThisTask == 0)
        printf("Pasamos masas: %f // %f \n", Maxmass, Minmass);

    MPI_Allreduce(&Maxmass, &All.Maxmass, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&Minmass, &All.Minmass, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    if(ThisTask == 0)
        printf("Pasamos masas x 2: %f // %f \n", All.Maxmass, All.Minmass);

    healpix_halo_cond(All.healpixmap);

#endif /* final of general HEALPIX */

    /* at this point, the entropy variable actually contains the
     * internal energy, read in from the initial conditions file.
     * Once the density has been computed, we can convert to entropy.
     */

    for(i = 0; i < N_sph; i++)	/* initialize sph_properties */
    {
        /* PETAIO:
         * NON IC, this flag is 1
         * IC it is 0. */
        if(header.flag_entropy_instead_u == 0)
        {
#ifndef EOS_DEGENERATE

#if !defined(TRADITIONAL_SPH_FORMULATION) && !defined(DENSITY_INDEPENDENT_SPH)
/* for DENSITY_INDEPENDENT_SPH, this is done already. */

            if(ThisTask == 0 && i == 0)
                printf("Converting u -> entropy !\n");

            SPHP(i).Entropy = GAMMA_MINUS1 * SPHP(i).Entropy / pow(SPHP(i).Density / a3, GAMMA_MINUS1);
#endif

#else
            for(j = 0; j < EOS_NSPECIES; j++)
            {
                SPHP(i).dxnuc[j] = 0;
            }

            SPHP(i).Entropy *= All.UnitEnergy_in_cgs;
            /* call eos with physical units, energy and entropy are always stored in physical units */
            SPHP(i).temp = 1.0;
            eos_calc_egiven_v(SPHP(i).Density * All.UnitDensity_in_cgs, SPHP(i).xnuc, SPHP(i).dxnuc,
                    0, SPHP(i).Entropy, 0, &SPHP(i).temp, &SPHP(i).Pressure, &SPHP(i).dpdr);
            SPHP(i).Pressure /= All.UnitPressure_in_cgs;
#endif
        }

        SPHP(i).DtEntropy = 0;

        SPHP(i).DivVel = 0;

#ifdef REIONIZATION
        All.not_yet_reionized = 1;
#endif

    }

#if defined (CHEMISTRY) || defined (UM_CHEMISTRY)

    if(ThisTask == 0)
    {
        printf("Initial abundances (for P[1].ID=%d):\n",P[1].ID);

        printf("HI=%g, HII=%g, HeI=%g, HeII=%g, HeIII=%g \n",
                SPHP(1).HI, SPHP(1).HII, SPHP(1).HeI, SPHP(1).HeII, SPHP(1).HeIII);
        printf("HM=%g, H2I=%g, H2II=%g, elec=%g\n",
                SPHP(1).HM, SPHP(1).H2I, SPHP(1).H2II, SPHP(1).elec);

#if defined (UM_CHEMISTRY) && defined(UM_HD_COOLING)
        printf("HD=%g,  DI=%g, DII=%g\n",SPHP(1).HD,SPHP(1).DI,SPHP(1).DII);
        printf("HeHII=%g\n",SPHP(1).HeHII);
#endif

        printf("x=%g, y=%g, z=%g, vx=%g, vy=%g, vz=%g,\ndensity=%g, entropy=%g\n",
                P[N_sph - 1].Pos[0], P[N_sph - 1].Pos[1], P[N_sph - 1].Pos[2], P[N_sph - 1].Vel[0],
                P[N_sph - 1].Vel[1], P[N_sph - 1].Vel[2], SPHP(N_sph - 1).Density, SPHP(N_sph - 1).Entropy);


    }

    /* need predict the cooling time and elec_dot here */
    min_t_cool = min_t_elec = 1.0e30;
    max_t_cool = max_t_elec = -1.0e30;


    for(i = 0; i < N_sph; i++)	/* Init Chemistry: */
    {
#ifdef CHEMISTRY
        a_start = All.Time;
        a_end = All.Time + 0.001;	/*  0.001 as an arbitrary value */
        ifunc = compute_abundances(0, i, a_start, a_end);
#endif


#ifdef UM_CHEMISTRY

        double um_u;

        a_start = All.Time;
        a_end = All.Time + 0.001;	/* 0.001 as an arbitrary value */
        ifunc = compute_abundances(0, i, a_start, a_end, &um_u);

#endif

        if(fabs(SPHP(i).t_cool) < min_t_cool)
            min_t_cool = fabs(SPHP(i).t_cool);
        if(fabs(SPHP(i).t_cool) > max_t_cool)
            max_t_cool = fabs(SPHP(i).t_cool);

        if(fabs(SPHP(i).t_elec) < min_t_elec)
            min_t_elec = fabs(SPHP(i).t_elec);
        if(fabs(SPHP(i).t_elec) > max_t_elec)
            max_t_elec = fabs(SPHP(i).t_elec);
    }

    fprintf(stdout, "PE %d t_cool min= %g, max= %g in yrs \n", ThisTask, min_t_cool, max_t_cool);
    fflush(stdout);
    fprintf(stdout, "PE %d t_elec min= %g, max= %g in yrs \n", ThisTask, min_t_elec, max_t_elec);
    fflush(stdout);

#endif


    if(RestartFlag == 3)
    {
#ifdef FOF
        fof_fof(RestartSnapNum);
#endif
        endrun(0);
    }

#ifdef CHEMISTRY
    if(ThisTask == 0)
    {
        printf("Initial abundances: \n");
        printf("HI=%g, HII=%g, HeI=%g, HeII=%g, HeIII=%g \n",
                SPHP(1).HI, SPHP(1).HII, SPHP(1).HeI, SPHP(1).HeII, SPHP(1).HeIII);

        printf("HM=%g, H2I=%g, H2II=%g, elec=%g, %d\n",
                SPHP(1).HM, SPHP(1).H2I, SPHP(1).H2II, SPHP(1).elec, P[1].ID);

        printf("x=%g, y=%g, z=%g, vx=%g, vy=%g, vz=%g, density=%g, entropy=%g\n",
                P[N_sph - 1].Pos[0], P[N_sph - 1].Pos[1], P[N_sph - 1].Pos[2], P[N_sph - 1].Vel[0],
                P[N_sph - 1].Vel[1], P[N_sph - 1].Vel[2], SPHP(N_sph - 1).Density, SPHP(N_sph - 1).Entropy);
    }

    /* need predict the cooling time and elec_dot here */
    min_t_cool = min_t_elec = 1.0e30;
    max_t_cool = max_t_elec = -1.0e30;

    for(i = 0; i < N_sph; i++)
    {
        a_start = All.Time;
        a_end = All.Time + 0.001;	/* 0.001 as an arbitrary value */

        ifunc = compute_abundances(0, i, a_start, a_end);


        if(fabs(SPHP(i).t_cool) < min_t_cool)
            min_t_cool = fabs(SPHP(i).t_cool);
        if(fabs(SPHP(i).t_cool) > max_t_cool)
            max_t_cool = fabs(SPHP(i).t_cool);

        if(fabs(SPHP(i).t_elec) < min_t_elec)
            min_t_elec = fabs(SPHP(i).t_elec);
        if(fabs(SPHP(i).t_elec) > max_t_elec)
            max_t_elec = fabs(SPHP(i).t_elec);

    }

    fprintf(stdout, "PE %d t_cool min= %g, max= %g in yrs \n", ThisTask, min_t_cool, max_t_cool);
    fflush(stdout);
    fprintf(stdout, "PE %d t_elec min= %g, max= %g in yrs \n", ThisTask, min_t_elec, max_t_elec);
    fflush(stdout);

#endif


#ifdef SCFPOTENTIAL
    if(ThisTask == 0)
    { 
        printf("Init SCF...\n");
        fflush(stdout);   
    }
    SCF_init();
    if(ThisTask == 0)
    { 
        printf("Initial random seed = %ld\n", scf_seed);
        printf("done.\n");
        fflush(stdout);  
    }
#endif


}


/*! This routine computes the mass content of the box and compares it to the
 * specified value of Omega-matter.  If discrepant, the run is terminated.
 */
void check_omega(void)
{
    double mass = 0, masstot, omega;
    int i;

    for(i = 0; i < NumPart; i++)
        mass += P[i].Mass;

    MPI_Allreduce(&mass, &masstot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    omega =
        masstot / (All.BoxSize * All.BoxSize * All.BoxSize) / (3 * All.Hubble * All.Hubble / (8 * M_PI * All.G));
#ifdef TIMEDEPGRAV
    omega *= All.Gini / All.G;
#endif

#ifdef KSPACE_NEUTRINOS
    omega += All.OmegaNu;
#endif

    if(fabs(omega - All.Omega0) > 1.0e-3)
    {
        if(ThisTask == 0)
        {
            printf("\n\nI've found something odd!\n");
            printf
                ("The mass content accounts only for Omega=%g,\nbut you specified Omega=%g in the parameterfile.\n",
                 omega, All.Omega0);
            printf("\nI better stop.\n");

            fflush(stdout);
        }
        endrun(1);
    }
}



/*! This function is used to find an initial smoothing length for each SPH
 *  particle. It guarantees that the number of neighbours will be between
 *  desired_ngb-MAXDEV and desired_ngb+MAXDEV. For simplicity, a first guess
 *  of the smoothing length is provided to the function density(), which will
 *  then iterate if needed to find the right smoothing length.
 */
void setup_smoothinglengths(void)
{
    int i;

    if(RestartFlag == 0)
    {
#pragma omp parallel for
        for(i = 0; i < NumPart; i++)
        {
            int no, p;
            no = Father[i];
            /* quick hack to adjust for the baryon fraction
             * only this fraction of mass is of that type.
             * this won't work for non-dm non baryon;
             * ideally each node shall have separate count of 
             * ptypes of each type. 
             *
             * Eventually the iteration will fix this. */
            double massfactor;
            if(P[i].Type == 0) {
                massfactor = 0.04 / 0.26;
            } else {
                massfactor = 1.0 - 0.04 / 0.26;
            }
            while(10 * All.DesNumNgb * P[i].Mass > massfactor * Nodes[no].u.d.mass)
            {
                p = Nodes[no].u.d.father;

                if(p < 0)
                    break;

                no = p;
            }

#ifndef READ_HSML
#ifndef TWODIMS
#ifndef ONEDIM
            P[i].Hsml =
                pow(3.0 / (4 * M_PI) * All.DesNumNgb * P[i].Mass / (massfactor * Nodes[no].u.d.mass), 
                        1.0 / 3) * Nodes[no].len;
#else
            P[i].Hsml = All.DesNumNgb * (P[i].Mass / (massfactor * Nodes[no].u.d.mass)) * Nodes[no].len;
#endif
#else
            P[i].Hsml =
                pow(1.0 / (M_PI) * All.DesNumNgb * P[i].Mass / (massfactor * Nodes[no].u.d.mass), 1.0 / 2) * Nodes[no].len;
#endif
            if(All.SofteningTable[0] != 0 && P[i].Hsml > 500.0 * All.SofteningTable[0])
                P[i].Hsml = All.SofteningTable[0];
#endif
        }
    }

#ifdef BLACK_HOLES
    if(RestartFlag == 0 || RestartFlag == 2)
    {
        for(i = 0; i < NumPart; i++)
            if(P[i].Type == 5) {
                P[i].Hsml = All.SofteningTable[5];
                BHP(i).TimeBinLimit = -1;
            }
    }
#endif

#if defined(RADTRANSFER)
    if(RestartFlag == 0 || RestartFlag == 2)
    {
        for(i = 0; i < NumPart; i++)
            if(P[i].Type == 4)
                P[i].Hsml = All.SofteningTable[4];
    }
#endif


    density();
#ifdef VECT_POTENTIAL
    smoothed_values();
    density();
#endif

#ifdef DENSITY_INDEPENDENT_SPH
    /* for clean IC with U input only, we need to iterate to find entrpoy */
    if(RestartFlag == 0 && header.flag_entropy_instead_u == 0)
    {
        for(i = 0; i < N_sph; i++)
        {
            /* start the iteration from mass density */
            SPHP(i).EgyWtDensity = SPHP(i).Density;
        }

        double a3;
        if(All.ComovingIntegrationOn) {
            a3 = All.Time * All.Time * All.Time;
        }
        else {
            a3 = 1;
        }
        /* initialization of the entropy variable is a little trickier in this version of SPH, 
           since we need to make sure it 'talks to' the density appropriately */

        if (ThisTask == 0) {
            printf("Converint u -> entropy, with density split sph\n");
        }

        int j;
        double badness;
        double * olddensity = (double *)mymalloc("olddensity ", N_sph * sizeof(double));
        for(j=0;j<100;j++)
        {/* since ICs give energies, not entropies, need to iterate get this initialized correctly */
#pragma omp parallel for 
            for(i = 0; i < N_sph; i++)
            {
                double entropy = GAMMA_MINUS1 * SPHP(i).Entropy / pow(SPHP(i).EgyWtDensity / a3 , GAMMA_MINUS1);
                SPHP(i).EntVarPred = pow(entropy, 1/GAMMA);
                olddensity[i] = SPHP(i).EgyWtDensity;
            }
            density();
            badness = 0;

#pragma omp parallel private(i)
            {
                double mybadness = 0;
#pragma omp for
                for(i = 0; i < N_sph; i++) {
                    if(!SPHP(i).EgyWtDensity > 0) continue;
                    double value = fabs(SPHP(i).EgyWtDensity - olddensity[i]) / SPHP(i).EgyWtDensity;
                    if(value > mybadness) mybadness = value;
                }
#pragma omp critical 
                {
                    if(mybadness > badness) {
                        badness = mybadness;
                    }
                }
            }
            MPI_Allreduce(MPI_IN_PLACE, &badness, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

            if(ThisTask == 0) 
                printf("iteration %03d, max relative difference = %g \n", j, badness);

            if(badness < 1e-3) break;
        }
        myfree(olddensity);
#pragma omp parallel for
        for(i = 0; i < N_sph; i++) {
            /* EgyWtDensity stabilized, now we convert from energy to entropy*/
            SPHP(i).Entropy = GAMMA_MINUS1 * SPHP(i).Entropy / pow(SPHP(i).EgyWtDensity/a3 , GAMMA_MINUS1);
        }
    }

    /* snapshot already has Entropy and EgyWtDensity; 
     * hope it is read in correctly. (need a test
     * on this!) */
    /* regardless we initalize EntVarPred. This may be unnecessary*/
    for(i = 0; i < N_sph; i++) {
        SPHP(i).EntVarPred = pow(SPHP(i).Entropy, 1./GAMMA);
    }
    density();
#endif

#if defined(MAGNETIC) && defined(BFROMROTA)
    if(RestartFlag == 0)
    {
        if(ThisTask == 0)
            printf("Converting: Vector Potential -> Bfield\n");
        rot_a();
    }
#endif

}


void assign_unique_ids(void)
{
    int i, *numpartlist;
    MyIDType idfirst;

    numpartlist = (int *)mymalloc("numpartlist", NTask * sizeof(int));

    MPI_Allgather(&NumPart, 1, MPI_INT, numpartlist, 1, MPI_INT, MPI_COMM_WORLD);

    idfirst = 1;

    for(i = 0; i < ThisTask; i++)
        idfirst += numpartlist[i];

    for(i = 0; i < NumPart; i++)
    {
        P[i].ID = idfirst;
        idfirst++;
    }

    myfree(numpartlist);
}


static void radix_id(const void * data, void * radix, void * arg) {
    ((uint64_t *) radix)[0] = ((MyIDType*) data)[0];
}
void test_id_uniqueness(void)
{
    int i;
    double t0, t1;
    MyIDType *ids, *ids_first;

    if(ThisTask == 0)
    {
        printf("Testing ID uniqueness...\n");
        fflush(stdout);
    }

    if(NumPart == 0)
    {
        printf("need at least one particle per cpu\n");
        endrun(8);
    }

    t0 = second();

#ifndef SPH_BND_PARTICLES
    ids = (MyIDType *) mymalloc("ids", NumPart * sizeof(MyIDType));
    ids_first = (MyIDType *) mymalloc("ids_first", NTask * sizeof(MyIDType));

    for(i = 0; i < NumPart; i++)
        ids[i] = P[i].ID;

    radix_sort_mpi(ids, NumPart, sizeof(MyIDType), radix_id, 8, NULL, MPI_COMM_WORLD);

    for(i = 1; i < NumPart; i++)
        if(ids[i] == ids[i - 1])
        {
            printf("non-unique ID=%d%09d found on task=%d (i=%d NumPart=%d)\n",
                    (int) (ids[i] / 1000000000), (int) (ids[i] % 1000000000), ThisTask, i, NumPart);

            endrun(12);
        }

    MPI_Allgather(&ids[0], sizeof(MyIDType), MPI_BYTE, ids_first, sizeof(MyIDType), MPI_BYTE, MPI_COMM_WORLD);

    if(ThisTask < NTask - 1)
        if(ids[NumPart - 1] == ids_first[ThisTask + 1])
        {
            printf("non-unique ID=%d found on task=%d\n", (int) ids[NumPart - 1], ThisTask);
            endrun(13);
        }

    myfree(ids_first);
    myfree(ids);
#endif

    t1 = second();

    if(ThisTask == 0)
    {
        printf("success.  took=%g sec\n", timediff(t0, t1));
        fflush(stdout);
    }
}

int compare_IDs(const void *a, const void *b)
{
    if(*((MyIDType *) a) < *((MyIDType *) b))
        return -1;

    if(*((MyIDType *) a) > *((MyIDType *) b))
        return +1;

    return 0;
}
