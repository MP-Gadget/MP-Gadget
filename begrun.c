#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <gsl/gsl_rng.h>


#include "allvars.h"
#include "densitykernel.h"
#include "proto.h"
#include "petaio.h"

#include "config.h"

/*! \file begrun.c
 *  \brief initial set-up of a simulation run
 *
 *  This file contains various functions to initialize a simulation run. In
 *  particular, the parameterfile is read in and parsed, the initial
 *  conditions or restart files are read, and global variables are initialized
 *  to their proper values.
 */



/*! This function performs the initial set-up of the simulation. First, the
 *  parameterfile is set, then routines for setting units, reading
 *  ICs/restart-files are called, auxialiary memory is allocated, etc.
 */
void begrun(void)
{
    struct global_data_all_processes all;

    if(ThisTask == 0)
    {
        /*    printf("\nThis is P-Gadget, version `%s', svn-revision `%s'.\n", GADGETVERSION, svn_version()); */
        printf("\nThis is P-Gadget, version %s.\n", GADGETVERSION);
        printf("\nRunning on %d MPIs .\n", NTask);
        printf("\nRunning on %d Threads.\n", omp_get_max_threads());
        printf("\nCode was compiled with settings:\n %s\n", COMPILETIMESETTINGS);
        printf("\nSize of particle structure       %td  [bytes]\n",sizeof(struct particle_data));
        printf("\nSize of blackhole structure       %td  [bytes]\n",sizeof(struct bh_particle_data));
        printf("\nSize of sph particle structure   %td  [bytes]\n",sizeof(struct sph_particle_data));
    }

#if defined(X86FIX) && defined(SOFTDOUBLEDOUBLE)
    x86_fix();			/* disable 80bit treatment of internal FPU registers in favour of proper IEEE 64bit double precision arithmetic */
#endif

    read_parameter_file(ParameterFile);	/* ... read in parameters for this run */

    mymalloc_init();
    walltime_init(&All.CT);
    petaio_init();


#ifdef DEBUG
    write_pid_file();
    enable_core_dumps_and_fpu_exceptions();
#endif

#ifdef DARKENERGY
#ifdef TIMEDEPDE
    fwa_init();
#endif
#endif

    set_units();


#ifdef COOLING
    set_global_time(All.TimeBegin);
    InitCool();
#endif

#if defined(SFR)
    init_clouds();
#endif

#ifdef LIGHTCONE
    lightcone_init();
#endif

    boxSize = All.BoxSize;
    boxHalf = 0.5 * All.BoxSize;
    inverse_boxSize = 1. / boxSize;

#ifdef TIME_DEP_ART_VISC
    All.ViscSource = All.ViscSource0 / log((GAMMA + 1) / (GAMMA - 1));
    All.DecayTime = 1 / All.DecayLength * sqrt((GAMMA - 1) / 2 * GAMMA);
#endif

    random_generator = gsl_rng_alloc(gsl_rng_ranlxd1);

    gsl_rng_set(random_generator, 42);	/* start-up seed */

    if(RestartFlag != 3 && RestartFlag != 4)
        long_range_init();

#ifdef EOS_DEGENERATE
    eos_init(All.EosTable, All.EosSpecies);
#endif

#ifdef NUCLEAR_NETWORK
    network_init(All.EosSpecies, All.NetworkRates, All.NetworkPartFunc, All.NetworkMasses,
            All.NetworkWeakrates);
#endif

    All.TimeLastRestartFile = 0;

    if(RestartFlag == 0 || RestartFlag == 2 || RestartFlag == 3 || RestartFlag == 4 || RestartFlag == 5)
    {
        set_random_numbers();

        init();			/* ... read in initial model */
    }
    else
    {
        all = All;		/* save global variables. (will be read from restart file) */

        restart(RestartFlag);	/* ... read restart file. Note: This also resets 
                                   all variables in the struct `All'. 
                                   However, during the run, some variables in the parameter
                                   file are allowed to be changed, if desired. These need to 
                                   copied in the way below.
Note:  All.PartAllocFactor is treated in restart() separately.  
*/

#ifdef _OPENMP
        /* thus we will used the new NumThreads of this run */
        All.NumThreads = all.NumThreads;
#endif
        All.MinSizeTimestep = all.MinSizeTimestep;
        All.MaxSizeTimestep = all.MaxSizeTimestep;
        All.BufferSize = all.BufferSize;
        All.TimeLimitCPU = all.TimeLimitCPU;
        All.TimeBetSnapshot = all.TimeBetSnapshot;
        All.TimeBetStatistics = all.TimeBetStatistics;
        All.CpuTimeBetRestartFile = all.CpuTimeBetRestartFile;
        All.ErrTolIntAccuracy = all.ErrTolIntAccuracy;
        All.MinGasHsmlFractional = all.MinGasHsmlFractional;
        All.MaxRMSDisplacementFac = all.MaxRMSDisplacementFac;

        All.ErrTolForceAcc = all.ErrTolForceAcc;
        All.TypeOfTimestepCriterion = all.TypeOfTimestepCriterion;
        All.TypeOfOpeningCriterion = all.TypeOfOpeningCriterion;
        All.NumWritersPerSnapshot = all.NumWritersPerSnapshot;
        All.TreeDomainUpdateFrequency = all.TreeDomainUpdateFrequency;

        All.OutputListOn = all.OutputListOn;
        All.CourantFac = all.CourantFac;

        All.OutputListLength = all.OutputListLength;
        memcpy(All.OutputListTimes, all.OutputListTimes, sizeof(double) * All.OutputListLength);
        memcpy(All.OutputListFlag, all.OutputListFlag, sizeof(char) * All.OutputListLength);

#ifdef TIME_DEP_ART_VISC
        All.ViscSource = all.ViscSource;
        All.ViscSource0 = all.ViscSource0;
        All.DecayTime = all.DecayTime;
        All.DecayLength = all.DecayLength;
        All.AlphaMin = all.AlphaMin;
#endif

#ifdef DARKENERGY
        All.DarkEnergyParam = all.DarkEnergyParam;
#endif

        strcpy(All.OutputListFilename, all.OutputListFilename);
        strcpy(All.OutputDir, all.OutputDir);
        strcpy(All.RestartFile, all.RestartFile);
        strcpy(All.EnergyFile, all.EnergyFile);
        strcpy(All.InfoFile, all.InfoFile);
        strcpy(All.CpuFile, all.CpuFile);
        strcpy(All.TimingsFile, all.TimingsFile);
        strcpy(All.SnapshotFileBase, all.SnapshotFileBase);

#ifdef EOS_DEGENERATE
        strcpy(All.EosTable, all.EosTable);
        strcpy(All.EosSpecies, all.EosSpecies);
#endif

#ifdef RELAXOBJECT
        All.RelaxBaseFac = all.RelaxBaseFac;
#endif

#ifdef NUCLEAR_NETWORK
        strcpy(All.NetworkRates, all.NetworkRates);
        strcpy(All.NetworkPartFunc, all.NetworkPartFunc);
        strcpy(All.NetworkMasses, all.NetworkMasses);
        strcpy(All.NetworkWeakrates, all.NetworkWeakrates);
#endif

        if(All.TimeMax != all.TimeMax)
            readjust_timebase(All.TimeMax, all.TimeMax);

#ifdef NO_TREEDATA_IN_RESTART
        /* if this is not activated, the tree was stored in the restart-files,
           which also allocated the storage for it already */

        /* ensures that domain reconstruction will be done and new tree will be constructed */
        All.NumForcesSinceLastDomainDecomp = (int64_t) (1 + All.TotNumPart * All.TreeDomainUpdateFrequency);
#endif
    }

    open_outputfiles();

    reconstruct_timebins();

#ifdef TWODIMS
    int i;

    for(i = 0; i < NumPart; i++)
    {
        P[i].Pos[2] = 0;
        P[i].Vel[2] = 0;

        P[i].GravAccel[2] = 0;

        if(P[i].Type == 0)
        {
            SPHP(i).VelPred[2] = 0;
            SPHP(i).a.HydroAccel[2] = 0;
        }
    }
#endif


#ifdef RADTRANSFER
    if(RestartFlag == 0)
        radtransfer_set_simple_inits();

    rt_get_sigma();
#endif


    init_drift_table();

    if(RestartFlag == 2)
        All.Ti_nextoutput = find_next_outputtime(All.Ti_Current + 100);
    else
        All.Ti_nextoutput = find_next_outputtime(All.Ti_Current);


    All.TimeLastRestartFile = 0;
}




/*! Computes conversion factors between internal code units and the
 *  cgs-system.
 */
void set_units(void)
{
    double meanweight;

    All.UnitTime_in_s = All.UnitLength_in_cm / All.UnitVelocity_in_cm_per_s;
    All.UnitTime_in_Megayears = All.UnitTime_in_s / SEC_PER_MEGAYEAR;

    if(All.GravityConstantInternal == 0)
        All.G = GRAVITY / pow(All.UnitLength_in_cm, 3) * All.UnitMass_in_g * pow(All.UnitTime_in_s, 2);
    else
        All.G = All.GravityConstantInternal;
#ifdef TIMEDEPGRAV
    All.Gini = All.G;
    All.G = All.Gini * dGfak(All.TimeBegin);
#endif

    All.UnitDensity_in_cgs = All.UnitMass_in_g / pow(All.UnitLength_in_cm, 3);
    All.UnitPressure_in_cgs = All.UnitMass_in_g / All.UnitLength_in_cm / pow(All.UnitTime_in_s, 2);
    All.UnitCoolingRate_in_cgs = All.UnitPressure_in_cgs / All.UnitTime_in_s;
    All.UnitEnergy_in_cgs = All.UnitMass_in_g * pow(All.UnitLength_in_cm, 2) / pow(All.UnitTime_in_s, 2);

    /* convert some physical input parameters to internal units */

    All.Hubble = HUBBLE * All.UnitTime_in_s;

    if(ThisTask == 0)
    {
        printf("\nHubble (internal units) = %g\n", All.Hubble);
        printf("G (internal units) = %g\n", All.G);
        printf("UnitMass_in_g = %g \n", All.UnitMass_in_g);
        printf("UnitTime_in_s = %g \n", All.UnitTime_in_s);
        printf("UnitVelocity_in_cm_per_s = %g \n", All.UnitVelocity_in_cm_per_s);
        printf("UnitDensity_in_cgs = %g \n", All.UnitDensity_in_cgs);
        printf("UnitEnergy_in_cgs = %g \n", All.UnitEnergy_in_cgs);
        printf("Radiation density Omega_R = %g\n",OMEGAR);

        printf("\n");
    }

    meanweight = 4.0 / (1 + 3 * HYDROGEN_MASSFRAC);	/* note: assuming NEUTRAL GAS */

    All.MinEgySpec = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.MinGasTemp;
    All.MinEgySpec *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;


#if defined(SFR)
    set_units_sfr();
#endif


#define cm (All.HubbleParam/All.UnitLength_in_cm)
#define g  (All.HubbleParam/All.UnitMass_in_g)
#define s  (All.HubbleParam/All.UnitTime_in_s)
#define erg (g*cm*cm/(s*s))
#define keV (1.602e-9*erg)
#define deg 1.0
#define m_p (PROTONMASS * g)
#define k_B (BOLTZMANN * erg / deg)

#ifdef NAVIERSTOKES
    /* Braginskii-Spitzer shear viscosity parametrization */
    /* mu = 0.406 * m_p^0.5 * (k_b* T)^(5/2) / e^4 / logLambda  [g/cm/s] */
    /* eta = frac * mu */

    meanweight = 4.0 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));	/* assuming full ionization */

#ifdef NAVIERSTOKES_CONSTANT
    All.NavierStokes_ShearViscosity = All.FractionSpitzerViscosity * 0.406 * pow(PROTONMASS, 0.5) * pow(BOLTZMANN * All.ShearViscosityTemperature, 5. / 2.) / pow(ELECTRONCHARGE, 4) / LOG_LAMBDA;	/*in cgs units */

    if(ThisTask == 0)
        printf("Constant shear viscosity in cgs units: eta = %g\n", All.NavierStokes_ShearViscosity);

    All.NavierStokes_ShearViscosity *= All.UnitTime_in_s * All.UnitLength_in_cm / All.UnitMass_in_g / All.HubbleParam;	/* in internal code units */

    if(ThisTask == 0)
        printf("Constant shear viscosity in internal code units: eta = %g\n", All.NavierStokes_ShearViscosity);

#else
    All.NavierStokes_ShearViscosity = All.FractionSpitzerViscosity * 0.406 * pow(PROTONMASS, 0.5) * pow((meanweight * PROTONMASS * GAMMA_MINUS1), 5. / 2.) / pow(ELECTRONCHARGE, 4) / LOG_LAMBDA;	/*in cgs units */
    /*T = mu*m_p*(gamma-1)/k_b * E * UnitEnergy/UnitMass */

    All.NavierStokes_ShearViscosity *= pow((All.UnitEnergy_in_cgs / All.UnitMass_in_g), 5. / 2.);	/* now energy can be multiplied later in the internal code units */
    All.NavierStokes_ShearViscosity *= All.UnitTime_in_s * All.UnitLength_in_cm / All.UnitMass_in_g / All.HubbleParam;	/* in internal code units */

    if(ThisTask == 0)
        printf("Variable shear viscosity in internal code units: eta = %g\n", All.NavierStokes_ShearViscosity);

#endif

#ifdef NAVIERSTOKES_BULK
    if(ThisTask == 0)
        printf("Costant bulk viscosity in internal code units: zeta = %g\n", All.NavierStokes_BulkViscosity);
#endif

#ifdef VISCOSITY_SATURATION
    /* calculate ion mean free path assuming complete ionization: 
       ion mean free path for hydrogen is similar to that of helium, 
       thus we calculate only for hydrogen */
    /* l_i = 3^(3/2)*(k*T)^2 / (4*\pi^(1/2)*ni*(Z*e)^4*lnL) */

    All.IonMeanFreePath = pow(3.0, 1.5) / (4.0 * sqrt(M_PI) * pow(ELECTRONCHARGE, 4) * LOG_LAMBDA);

    All.IonMeanFreePath *= pow(meanweight * PROTONMASS * GAMMA_MINUS1, 2) * pow((All.UnitEnergy_in_cgs / All.UnitMass_in_g), 2);	/*kT -> u */

    All.IonMeanFreePath /= (HYDROGEN_MASSFRAC / PROTONMASS) *
        (All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam);
    /* n_H = rho * Hfr / mp *//* now is cgs units *///changed / to * in front of the unitdensity

    All.IonMeanFreePath *= All.HubbleParam / All.UnitLength_in_cm;
    /* in internal code units */
#endif

#endif

}



/*!  This function opens various log-files that report on the status and
 *   performance of the simulstion. On restart from restart-files
 *   (start-option 1), the code will append to these files.
 */
void open_outputfiles(void)
{
    char mode[2], buf[200];
    char dumpdir[200];
    char postfix[128];

    if(RestartFlag == 0 || RestartFlag == 2)
        strcpy(mode, "w");
    else
        strcpy(mode, "a");

    if(RestartFlag == 2) {
        sprintf(postfix, "-R%03d", RestartSnapNum);
    } else {
        sprintf(postfix, "%s", "");
    }

    /* create spliced dirs */
    int chunk = 10;
    if (NTask > 100) chunk = 100;
    if (NTask > 1000) chunk = 1000;

    sprintf(dumpdir, "%sdumpdir-%d%s/", All.OutputDir, (int)(ThisTask / chunk), postfix);
    mkdir(dumpdir, 02755);

#ifdef BLACK_HOLES
    /* Note: This is done by everyone */
    sprintf(buf, "%sblackhole_details_%d.raw", dumpdir, ThisTask);
    if(!(FdBlackHolesDetails = fopen(buf, mode)))
    {
        printf("error in opening file '%s'\n", buf);
        endrun(1);
    }
#endif
#ifdef SFR
    /* Note: This is done by everyone */
    sprintf(buf, "%ssfr_details_%d.txt", dumpdir, ThisTask);
    if(!(FdSfrDetails = fopen(buf, mode)))
    {
        printf("error in opening file '%s'\n", buf);
        endrun(1);
    }
#endif

    if(ThisTask != 0)		/* only the root processors writes to the log files */
        return;

    sprintf(buf, "%s%s%s", All.OutputDir, All.CpuFile, postfix);
    if(!(FdCPU = fopen(buf, mode)))
    {
        printf("error in opening file '%s'\n", buf);
        endrun(1);
    }

    sprintf(buf, "%s%s%s", All.OutputDir, All.InfoFile, postfix);
    if(!(FdInfo = fopen(buf, mode)))
    {
        printf("error in opening file '%s'\n", buf);
        endrun(1);
    }

    sprintf(buf, "%s%s%s", All.OutputDir, All.EnergyFile, postfix);
    if(!(FdEnergy = fopen(buf, mode)))
    {
        printf("error in opening file '%s'\n", buf);
        endrun(1);
    }

    sprintf(buf, "%s%s%s", All.OutputDir, All.TimingsFile, postfix);
    if(!(FdTimings = fopen(buf, mode)))
    {
        printf("error in opening file '%s'\n", buf);
        endrun(1);
    }

#ifdef SCFPOTENTIAL
    sprintf(buf, "%s%s%s", All.OutputDir, "scf_coeff.txt", postfix);
    if(!(FdSCF = fopen(buf, mode)))
    {
        printf("error in opening file '%s'\n", buf);
        endrun(1);
    }
#endif

#ifdef SFR
    sprintf(buf, "%s%s%s", All.OutputDir, "sfr.txt", postfix);
    if(!(FdSfr = fopen(buf, mode)))
    {
        printf("error in opening file '%s'\n", buf);
        endrun(1);
    }
#endif

#ifdef RADTRANSFER
    sprintf(buf, "%s%s%s", All.OutputDir, "radtransfer.txt", postfix);
    if(!(FdRad = fopen(buf, mode)))
    {
        printf("error in opening file '%s'\n", buf);
        endrun(1);
    }

    sprintf(buf, "%s%s%s", All.OutputDir, "radtransferNew.txt", postfix);
    if(!(FdRadNew = fopen(buf, mode)))
    {
        printf("error in opening file '%s'\n", buf);
        endrun(1);
    }
#endif

#ifdef BLACK_HOLES
    sprintf(buf, "%s%s%s", All.OutputDir, "blackholes.txt", postfix);
    if(!(FdBlackHoles = fopen(buf, mode)))
    {
        printf("error in opening file '%s'\n", buf);
        endrun(1);
    }
#endif



#ifdef FORCETEST
    if(RestartFlag == 0)
    {
        sprintf(buf, "%s%s%s", All.OutputDir, "forcetest.txt", postfix);
        if(!(FdForceTest = fopen(buf, "w")))
        {
            printf("error in opening file '%s'\n", buf);
            endrun(1);
        }
        fclose(FdForceTest);
    }
#endif

#ifdef XXLINFO
    sprintf(buf, "%s%s%s", All.OutputDir, "xxl.txt", postfix);
    if(!(FdXXL = fopen(buf, mode)))
    {
        printf("error in opening file '%s'\n", buf);
        endrun(1);
    }
    else
    {
        if(RestartFlag == 0)
        {
            fprintf(FdXXL, "nstep time ");
#ifdef TIME_DEP_ART_VISC
            fprintf(FdXXL, "<alpha> ");
#endif
            fprintf(FdXXL, "\n");
            fflush(FdXXL);
        }
    }
#endif

#ifdef DARKENERGY
    sprintf(buf, "%s%s%s", All.OutputDir, "darkenergy.txt", postfix);
    if(!(FdDE = fopen(buf, mode)))
    {
        printf("error in opening file '%s'\n", buf);
        endrun(1);
    }
    else
    {
        if(RestartFlag == 0)
        {
            fprintf(FdDE, "nstep time H(a) ");
#ifndef TIMEDEPDE
            fprintf(FdDE, "w0 Omega_L ");
#else
            fprintf(FdDE, "w(a) Omega_L ");
#endif
#ifdef TIMEDEPGRAV
            fprintf(FdDE, "dH dG ");
#endif
            fprintf(FdDE, "\n");
            fflush(FdDE);
        }
    }
#endif



}




/*!  This function closes the global log-files.
*/
void close_outputfiles(void)
{
#ifdef BLACK_HOLES
    fclose(FdBlackHolesDetails);	/* needs to be done by everyone */
#endif

#ifdef CAUSTIC_FINDER
    fclose(FdCaustics);		/* needs to be done by everyone */
#endif

    if(ThisTask != 0)		/* only the root processors writes to the log files */
        return;

    fclose(FdCPU);
    fclose(FdInfo);
    fclose(FdEnergy);
    fclose(FdTimings);

#ifdef SCFPOTENTIAL
    fclose(FdSCF);
#endif

#ifdef SFR
    fclose(FdSfr);
#endif

#ifdef RADTRANSFER
    fclose(FdRad);
    fclose(FdRadNew);
#endif

#ifdef BLACK_HOLES
    fclose(FdBlackHoles);
#endif

#ifdef XXLINFO
    fclose(FdXXL);
#endif

#ifdef DARKENERGY
    fclose(FdDE);
#endif

}



struct multichoice { char * name; int value; } ;
static int parse_multichoice(struct multichoice * table, char * strchoices) {
    int value = 0;
    struct multichoice * p = table;
    char * delim = ",;&| \t";
    char * token;
    for(token = strtok(strchoices, delim); token ; token = strtok(NULL, delim)) {
        for(p = table; p->name; p++) {
            if(strcasecmp(token, p->name) == 0) {
                value |= p->value;
                break;
            }
        }
        if(p->name == NULL) {
            /* error occured !*/
            return 0; 
        }
    }
    if(value == 0) {
        /* none is specified, use default (NULL named entry) */
        value = p->value;
    }
    return value;
}
static char * format_multichoice(struct multichoice * table, int value) {
    char buffer[2048];
    struct multichoice * p;
    char * c = buffer;
    for(p = table; p->name; p++) {
        if(HAS(value, p->value)) {
            strcpy(c, p->name);
            c += strlen(p->name);
            c[0] = '&';
            c++;
            c[0] = 0;
        }
    }
    return strdup(buffer);
}

#ifdef BLACK_HOLES
struct multichoice BlackHoleFeedbackMethodChoices [] = {
    {"mass", BH_FEEDBACK_MASS},
    {"volume", BH_FEEDBACK_VOLUME},
    {"tophat", BH_FEEDBACK_TOPHAT},
    {"spline", BH_FEEDBACK_SPLINE},
    {NULL, BH_FEEDBACK_SPLINE | BH_FEEDBACK_MASS},
};
#endif 
#ifdef SFR
struct multichoice StarformationCriterionChoices [] = {
    {"density", SFR_CRITERION_DENSITY},
    {"h2", SFR_CRITERION_MOLECULAR_H2},
    {"selfgravity", SFR_CRITERION_SELFGRAVITY},
    {"convergent", SFR_CRITERION_CONVERGENT_FLOW},
    {"continous", SFR_CRITERION_CONTINUOUS_CUTOFF},
    {NULL, SFR_CRITERION_DENSITY},
};

struct multichoice WindModelChoices [] = {
    {"subgrid", WINDS_SUBGRID},
    {"decouple", WINDS_DECOUPLE_SPH},
    {"halo", WINDS_USE_HALO},
    {"fixedefficiency", WINDS_FIXED_EFFICIENCY},
    {"sh03", WINDS_SUBGRID | WINDS_DECOUPLE_SPH | WINDS_FIXED_EFFICIENCY} ,
    {"vs08", WINDS_FIXED_EFFICIENCY},
    {"ofjt10", WINDS_USE_HALO | WINDS_DECOUPLE_SPH},
    {NULL, WINDS_SUBGRID | WINDS_DECOUPLE_SPH | WINDS_FIXED_EFFICIENCY},
};

#endif

/*! This function parses the parameterfile in a simple way.  Each paramater is
 *  defined by a keyword (`tag'), and can be either of type douple, int, or
 *  character string.  The routine makes sure that each parameter appears
 *  exactly once in the parameterfile, otherwise error messages are
 *  produced that complain about the missing parameters.
 */
void read_parameter_file(char *fname)
{
#define REAL 1
#define STRING 2
#define INT 3
#define MULTICHOICE 4
#define MAXTAGS 300

    FILE *fd, *fdout;
    char buf[200], buf1[200], buf2[200], buf3[400];
    int i, j, nt;
    int id[MAXTAGS];
    void *addr[MAXTAGS];
    struct multichoice * choices[MAXTAGS];
    char tag[MAXTAGS][50];
    int errorFlag = 0;

    All.StarformationOn = 0;	/* defaults */

    if(sizeof(int64_t) != 8)
    {
        if(ThisTask == 0)
            printf("\nType `int64_t' is not 64 bit on this platform. Stopping.\n\n");
        endrun(0);
    }

    if(sizeof(int) != 4)
    {
        if(ThisTask == 0)
            printf("\nType `int' is not 32 bit on this platform. Stopping.\n\n");
        endrun(0);
    }

    if(sizeof(float) != 4)
    {
        if(ThisTask == 0)
            printf("\nType `float' is not 32 bit on this platform. Stopping.\n\n");
        endrun(0);
    }

    if(sizeof(double) != 8)
    {
        if(ThisTask == 0)
            printf("\nType `double' is not 64 bit on this platform. Stopping.\n\n");
        endrun(0);
    }

    All.NumThreads = omp_get_max_threads();

    if(ThisTask == 0)		/* read parameter file on process 0 */
    {
        nt = 0;

        strcpy(tag[nt], "InitCondFile");
        addr[nt] = All.InitCondFile;
        id[nt++] = STRING;

        strcpy(tag[nt], "OutputDir");
        addr[nt] = All.OutputDir;
        id[nt++] = STRING;

        strcpy(tag[nt], "TreeCoolFile");
        addr[nt] = All.TreeCoolFile;
        id[nt++] = STRING;

        strcpy(tag[nt], "MetalCoolFile");
        addr[nt] = All.MetalCoolFile;
        id[nt++] = STRING;

        strcpy(tag[nt], "UVFluctuationFile");
        addr[nt] = All.UVFluctuationFile;
        id[nt++] = STRING;

        strcpy(tag[nt], "DensityKernelType");
        addr[nt] = &All.DensityKernelType;
        id[nt++] = INT;

        strcpy(tag[nt], "SnapshotFileBase");
        addr[nt] = All.SnapshotFileBase;
        id[nt++] = STRING;

        strcpy(tag[nt], "EnergyFile");
        addr[nt] = All.EnergyFile;
        id[nt++] = STRING;

        strcpy(tag[nt], "CpuFile");
        addr[nt] = All.CpuFile;
        id[nt++] = STRING;

        strcpy(tag[nt], "InfoFile");
        addr[nt] = All.InfoFile;
        id[nt++] = STRING;

        strcpy(tag[nt], "TimingsFile");
        addr[nt] = All.TimingsFile;
        id[nt++] = STRING;

        strcpy(tag[nt], "RestartFile");
        addr[nt] = All.RestartFile;
        id[nt++] = STRING;

        strcpy(tag[nt], "OutputListFilename");
        addr[nt] = All.OutputListFilename;
        id[nt++] = STRING;

        strcpy(tag[nt], "OutputListOn");
        addr[nt] = &All.OutputListOn;
        id[nt++] = INT;

        strcpy(tag[nt], "Omega0");
        addr[nt] = &All.Omega0;
        id[nt++] = REAL;

        strcpy(tag[nt], "OmegaBaryon");
        addr[nt] = &All.OmegaBaryon;
        id[nt++] = REAL;

        strcpy(tag[nt], "OmegaLambda");
        addr[nt] = &All.OmegaLambda;
        id[nt++] = REAL;

        strcpy(tag[nt], "HubbleParam");
        addr[nt] = &All.HubbleParam;
        id[nt++] = REAL;

        strcpy(tag[nt], "BoxSize");
        addr[nt] = &All.BoxSize;
        id[nt++] = REAL;

        strcpy(tag[nt], "MaxMemSizePerCore");
        addr[nt] = &All.MaxMemSizePerCore;
        id[nt++] = INT;

        strcpy(tag[nt], "TimeOfFirstSnapshot");
        addr[nt] = &All.TimeOfFirstSnapshot;
        id[nt++] = REAL;

        strcpy(tag[nt], "CpuTimeBetRestartFile");
        addr[nt] = &All.CpuTimeBetRestartFile;
        id[nt++] = REAL;

        strcpy(tag[nt], "TimeBetStatistics");
        addr[nt] = &All.TimeBetStatistics;
        id[nt++] = REAL;

        strcpy(tag[nt], "TimeBegin");
        addr[nt] = &All.TimeBegin;
        id[nt++] = REAL;

        strcpy(tag[nt], "TimeMax");
        addr[nt] = &All.TimeMax;
        id[nt++] = REAL;

        strcpy(tag[nt], "TimeBetSnapshot");
        addr[nt] = &All.TimeBetSnapshot;
        id[nt++] = REAL;

        strcpy(tag[nt], "UnitVelocity_in_cm_per_s");
        addr[nt] = &All.UnitVelocity_in_cm_per_s;
        id[nt++] = REAL;

        strcpy(tag[nt], "UnitLength_in_cm");
        addr[nt] = &All.UnitLength_in_cm;
        id[nt++] = REAL;

        strcpy(tag[nt], "UnitMass_in_g");
        addr[nt] = &All.UnitMass_in_g;
        id[nt++] = REAL;

        strcpy(tag[nt], "TreeDomainUpdateFrequency");
        addr[nt] = &All.TreeDomainUpdateFrequency;
        id[nt++] = REAL;

        strcpy(tag[nt], "ErrTolIntAccuracy");
        addr[nt] = &All.ErrTolIntAccuracy;
        id[nt++] = REAL;

        strcpy(tag[nt], "ErrTolTheta");
        addr[nt] = &All.ErrTolTheta;
        id[nt++] = REAL;

        strcpy(tag[nt], "Nmesh");
        addr[nt] = &All.Nmesh;
        id[nt++] = INT;

        strcpy(tag[nt], "ErrTolForceAcc");
        addr[nt] = &All.ErrTolForceAcc;
        id[nt++] = REAL;

        strcpy(tag[nt], "MinGasHsmlFractional");
        addr[nt] = &All.MinGasHsmlFractional;
        id[nt++] = REAL;

        strcpy(tag[nt], "MaxSizeTimestep");
        addr[nt] = &All.MaxSizeTimestep;
        id[nt++] = REAL;

        strcpy(tag[nt], "MinSizeTimestep");
        addr[nt] = &All.MinSizeTimestep;
        id[nt++] = REAL;

        strcpy(tag[nt], "MaxRMSDisplacementFac");
        addr[nt] = &All.MaxRMSDisplacementFac;
        id[nt++] = REAL;

        strcpy(tag[nt], "ArtBulkViscConst");
        addr[nt] = &All.ArtBulkViscConst;
        id[nt++] = REAL;

        strcpy(tag[nt], "CourantFac");
        addr[nt] = &All.CourantFac;
        id[nt++] = REAL;

        strcpy(tag[nt], "DensityResolutionEta");
        addr[nt] = &All.DensityResolutionEta;
        id[nt++] = REAL;

        strcpy(tag[nt], "DensityContrastLimit");
        addr[nt] = &All.DensityContrastLimit;
        id[nt++] = REAL;

#ifdef KSPACE_NEUTRINOS
        strcpy(tag[nt], "KspaceNeutrinoSeed");
        addr[nt] = &All.KspaceNeutrinoSeed;
        id[nt++] = INT;

        strcpy(tag[nt], "Nsample");
        addr[nt] = &All.Nsample;
        id[nt++] = INT;

        strcpy(tag[nt], "SphereMode");
        addr[nt] = &All.SphereMode;
        id[nt++] = INT;

        strcpy(tag[nt], "KspaceDirWithTransferfunctions");
        addr[nt] = All.KspaceDirWithTransferfunctions;
        id[nt++] = STRING;

        strcpy(tag[nt], "KspaceBaseNameTransferfunctions");
        addr[nt] = All.KspaceBaseNameTransferfunctions;
        id[nt++] = STRING;

        strcpy(tag[nt], "PrimordialIndex");
        addr[nt] = &All.PrimordialIndex;
        id[nt++] = REAL;

        strcpy(tag[nt], "Sigma8");
        addr[nt] = &All.Sigma8;
        id[nt++] = REAL;

        strcpy(tag[nt], "OmegaNu");
        addr[nt] = &All.OmegaNu;
        id[nt++] = REAL;

        strcpy(tag[nt], "InputSpectrum_UnitLength_in_cm");
        addr[nt] = &All.InputSpectrum_UnitLength_in_cm;
        id[nt++] = REAL;
#endif

        strcpy(tag[nt], "MaxNumNgbDeviation");
        addr[nt] = &All.MaxNumNgbDeviation;
        id[nt++] = REAL;

#ifdef START_WITH_EXTRA_NGBDEV
        strcpy(tag[nt], "MaxNumNgbDeviationStart");
        addr[nt] = &All.MaxNumNgbDeviationStart;
        id[nt++] = REAL;
#endif

        strcpy(tag[nt], "ICFormat");
        addr[nt] = &All.ICFormat;
        id[nt++] = INT;

        strcpy(tag[nt], "CompressionLevel");
        addr[nt] = &All.CompressionLevel;
        id[nt++] = INT;

        strcpy(tag[nt], "SnapFormat");
        addr[nt] = &All.SnapFormat;
        id[nt++] = INT;

        strcpy(tag[nt], "NumFilesPerSnapshot");
        addr[nt] = &All.NumFilesPerSnapshot;
        id[nt++] = INT;

        strcpy(tag[nt], "NumWritersPerSnapshot");
        addr[nt] = &All.NumWritersPerSnapshot;
        id[nt++] = INT;

        strcpy(tag[nt], "NumFilesPerPIG");
        addr[nt] = &All.NumFilesPerPIG;
        id[nt++] = INT;
        strcpy(tag[nt], "NumWritersPerPIG");
        addr[nt] = &All.NumWritersPerPIG;
        id[nt++] = INT;

        strcpy(tag[nt], "CoolingOn");
        addr[nt] = &All.CoolingOn;
        id[nt++] = INT;

        strcpy(tag[nt], "StarformationOn");
        addr[nt] = &All.StarformationOn;
        id[nt++] = INT;

        strcpy(tag[nt], "TypeOfTimestepCriterion");
        addr[nt] = &All.TypeOfTimestepCriterion;
        id[nt++] = INT;

        strcpy(tag[nt], "TypeOfOpeningCriterion");
        addr[nt] = &All.TypeOfOpeningCriterion;
        id[nt++] = INT;

        strcpy(tag[nt], "TimeLimitCPU");
        addr[nt] = &All.TimeLimitCPU;
        id[nt++] = REAL;

        strcpy(tag[nt], "SofteningHalo");
        addr[nt] = &All.SofteningHalo;
        id[nt++] = REAL;

        strcpy(tag[nt], "SofteningDisk");
        addr[nt] = &All.SofteningDisk;
        id[nt++] = REAL;

        strcpy(tag[nt], "SofteningBulge");
        addr[nt] = &All.SofteningBulge;
        id[nt++] = REAL;

        strcpy(tag[nt], "SofteningGas");
        addr[nt] = &All.SofteningGas;
        id[nt++] = REAL;

        strcpy(tag[nt], "SofteningStars");
        addr[nt] = &All.SofteningStars;
        id[nt++] = REAL;

        strcpy(tag[nt], "SofteningBndry");
        addr[nt] = &All.SofteningBndry;
        id[nt++] = REAL;

        strcpy(tag[nt], "SofteningHaloMaxPhys");
        addr[nt] = &All.SofteningHaloMaxPhys;
        id[nt++] = REAL;

        strcpy(tag[nt], "SofteningDiskMaxPhys");
        addr[nt] = &All.SofteningDiskMaxPhys;
        id[nt++] = REAL;

        strcpy(tag[nt], "SofteningBulgeMaxPhys");
        addr[nt] = &All.SofteningBulgeMaxPhys;
        id[nt++] = REAL;

        strcpy(tag[nt], "SofteningGasMaxPhys");
        addr[nt] = &All.SofteningGasMaxPhys;
        id[nt++] = REAL;

        strcpy(tag[nt], "SofteningStarsMaxPhys");
        addr[nt] = &All.SofteningStarsMaxPhys;
        id[nt++] = REAL;

        strcpy(tag[nt], "SofteningBndryMaxPhys");
        addr[nt] = &All.SofteningBndryMaxPhys;
        id[nt++] = REAL;

        strcpy(tag[nt], "BufferSize");
        addr[nt] = &All.BufferSize;
        id[nt++] = REAL;

        strcpy(tag[nt], "PartAllocFactor");
        addr[nt] = &All.PartAllocFactor;
        id[nt++] = REAL;

        strcpy(tag[nt], "TopNodeAllocFactor");
        addr[nt] = &All.TopNodeAllocFactor;
        id[nt++] = REAL;

        strcpy(tag[nt], "GravityConstantInternal");
        addr[nt] = &All.GravityConstantInternal;
        id[nt++] = REAL;

        strcpy(tag[nt], "InitGasTemp");
        addr[nt] = &All.InitGasTemp;
        id[nt++] = REAL;

        strcpy(tag[nt], "MinGasTemp");
        addr[nt] = &All.MinGasTemp;
        id[nt++] = REAL;

#ifdef SCALARFIELD
        strcpy(tag[nt], "ScalarBeta");
        addr[nt] = &All.ScalarBeta;
        id[nt++] = REAL;

        strcpy(tag[nt], "ScalarScreeningLength");
        addr[nt] = &All.ScalarScreeningLength;
        id[nt++] = REAL;
#endif

#ifdef OUTPUTLINEOFSIGHT
        strcpy(tag[nt], "TimeFirstLineOfSight");
        addr[nt] = &All.TimeFirstLineOfSight;
        id[nt++] = REAL;
#endif


#if defined(ADAPTIVE_GRAVSOFT_FORGAS) && !defined(ADAPTIVE_GRAVSOFT_FORGAS_HSML)
        strcpy(tag[nt], "ReferenceGasMass");
        addr[nt] = &All.ReferenceGasMass;
        id[nt++] = REAL;
#endif

#ifdef NAVIERSTOKES
        strcpy(tag[nt], "FractionSpitzerViscosity");
        addr[nt] = &All.FractionSpitzerViscosity;
        id[nt++] = REAL;
#endif

#ifdef NAVIERSTOKES_CONSTANT
        strcpy(tag[nt], "ShearViscosityTemperature");
        addr[nt] = &All.ShearViscosityTemperature;
        id[nt++] = REAL;
#endif

#ifdef NAVIERSTOKES_BULK
        strcpy(tag[nt], "NavierStokes_BulkViscosity");
        addr[nt] = &All.NavierStokes_BulkViscosity;
        id[nt++] = REAL;
#endif

#ifdef BLACK_HOLES
        strcpy(tag[nt], "TimeBetBlackHoleSearch");
        addr[nt] = &All.TimeBetBlackHoleSearch;
        id[nt++] = REAL;

        strcpy(tag[nt], "BlackHoleAccretionFactor");
        addr[nt] = &All.BlackHoleAccretionFactor;
        id[nt++] = REAL;

        strcpy(tag[nt], "BlackHoleEddingtonFactor");
        addr[nt] = &All.BlackHoleEddingtonFactor;
        id[nt++] = REAL;


        strcpy(tag[nt], "SeedBlackHoleMass");
        addr[nt] = &All.SeedBlackHoleMass;
        id[nt++] = REAL;

        strcpy(tag[nt], "MinFoFMassForNewSeed");
        addr[nt] = &All.MinFoFMassForNewSeed;
        id[nt++] = REAL;

        strcpy(tag[nt], "BlackHoleNgbFactor");
        addr[nt] = &All.BlackHoleNgbFactor;
        id[nt++] = REAL;

        strcpy(tag[nt], "BlackHoleMaxAccretionRadius");
        addr[nt] = &All.BlackHoleMaxAccretionRadius;
        id[nt++] = REAL;

        strcpy(tag[nt], "BlackHoleFeedbackFactor");
        addr[nt] = &All.BlackHoleFeedbackFactor;
        id[nt++] = REAL;

        strcpy(tag[nt], "BlackHoleFeedbackRadius");
        addr[nt] = &All.BlackHoleFeedbackRadius;
        id[nt++] = REAL;

        strcpy(tag[nt], "BlackHoleFeedbackRadiusMaxPhys");
        addr[nt] = &All.BlackHoleFeedbackRadiusMaxPhys;
        id[nt++] = REAL;

        strcpy(tag[nt], "BlackHoleFeedbackMethod");
        addr[nt] = &All.BlackHoleFeedbackMethod;
        choices[nt] = BlackHoleFeedbackMethodChoices;
        id[nt++] = MULTICHOICE;

#endif

#ifdef SFR
        strcpy(tag[nt], "StarformationCriterion");
        addr[nt] = &All.StarformationCriterion;
        choices[nt] = StarformationCriterionChoices;
        id[nt++] = MULTICHOICE;
          
        strcpy(tag[nt], "CritOverDensity");
        addr[nt] = &All.CritOverDensity;
        id[nt++] = REAL;

        strcpy(tag[nt], "CritPhysDensity");
        addr[nt] = &All.CritPhysDensity;
        id[nt++] = REAL;

        strcpy(tag[nt], "FactorSN");
        addr[nt] = &All.FactorSN;
        id[nt++] = REAL;
        strcpy(tag[nt], "FactorEVP");
        addr[nt] = &All.FactorEVP;
        id[nt++] = REAL;

        strcpy(tag[nt], "TempSupernova");
        addr[nt] = &All.TempSupernova;
        id[nt++] = REAL;

        strcpy(tag[nt], "TempClouds");
        addr[nt] = &All.TempClouds;
        id[nt++] = REAL;

        strcpy(tag[nt], "MaxSfrTimescale");
        addr[nt] = &All.MaxSfrTimescale;
        id[nt++] = REAL;

        strcpy(tag[nt], "WindModel");
        addr[nt] = &All.WindModel;
        choices[nt] = WindModelChoices;
        id[nt++] = MULTICHOICE;

        /* The following two are for VS08 and SH03*/
        strcpy(tag[nt], "WindEfficiency");
        addr[nt] = &All.WindEfficiency;
        id[nt++] = REAL;

        strcpy(tag[nt], "WindEnergyFraction");
        addr[nt] = &All.WindEnergyFraction;
        id[nt++] = REAL;

        /* The following two are for OFJT10*/
        strcpy(tag[nt], "WindSigma0");
        addr[nt] = &All.WindSigma0;
        id[nt++] = REAL;

        strcpy(tag[nt], "WindSpeedFactor");
        addr[nt] = &All.WindSpeedFactor;
        id[nt++] = REAL;

        strcpy(tag[nt], "WindFreeTravelLength");
        addr[nt] = &All.WindFreeTravelLength;
        id[nt++] = REAL;

        strcpy(tag[nt], "WindFreeTravelDensFac");
        addr[nt] = &All.WindFreeTravelDensFac;
        id[nt++] = REAL;
#endif

#if defined(SNIA_HEATING)
        strcpy(tag[nt], "SnIaHeatingRate");
        addr[nt] = &All.SnIaHeatingRate;
        id[nt++] = REAL;
#endif

#ifdef SOFTEREQS
        strcpy(tag[nt], "FactorForSofterEQS");
        addr[nt] = &All.FactorForSofterEQS;
        id[nt++] = REAL;
#endif
#ifdef DARKENERGY
#ifndef TIMEDEPDE
        strcpy(tag[nt], "DarkEnergyParam");
        addr[nt] = &All.DarkEnergyParam;
        id[nt++] = REAL;
#endif
#endif

#ifdef RESCALEVINI
        strcpy(tag[nt], "VelIniScale");
        addr[nt] = &All.VelIniScale;
        id[nt++] = REAL;
#endif

#ifdef DARKENERGY
#ifdef TIMEDEPDE
        strcpy(tag[nt], "DarkEnergyFile");
        addr[nt] = All.DarkEnergyFile;
        id[nt++] = STRING;
#endif
#endif

#ifdef TIME_DEP_ART_VISC
        strcpy(tag[nt], "ViscositySourceScaling");
        addr[nt] = &All.ViscSource0;
        id[nt++] = REAL;

        strcpy(tag[nt], "ViscosityDecayLength");
        addr[nt] = &All.DecayLength;
        id[nt++] = REAL;

        strcpy(tag[nt], "ViscosityAlphaMin");
        addr[nt] = &All.AlphaMin;
        id[nt++] = REAL;
#endif

#ifdef EOS_DEGENERATE
        strcpy(tag[nt], "EosTable");
        addr[nt] = All.EosTable;
        id[nt++] = STRING;

        strcpy(tag[nt], "EosSpecies");
        addr[nt] = All.EosSpecies;
        id[nt++] = STRING;
#endif

#ifdef NUCLEAR_NETWORK
        strcpy(tag[nt], "NetworkRates");
        addr[nt] = All.NetworkRates;
        id[nt++] = STRING;

        strcpy(tag[nt], "NetworkPartFunc");
        addr[nt] = All.NetworkPartFunc;
        id[nt++] = STRING;

        strcpy(tag[nt], "NetworkMasses");
        addr[nt] = All.NetworkMasses;
        id[nt++] = STRING;

        strcpy(tag[nt], "NetworkWeakrates");
        addr[nt] = All.NetworkWeakrates;
        id[nt++] = STRING;
#endif

#ifdef RELAXOBJECT
        strcpy(tag[nt], "RelaxBaseFac");
        addr[nt] = &All.RelaxBaseFac;
        id[nt++] = REAL;
#endif

#ifdef RADTRANSFER
        strcpy(tag[nt], "IonizingLumPerSolarMass");
        addr[nt] = &All.IonizingLumPerSolarMass;
        id[nt++] = REAL;

        strcpy(tag[nt], "IonizingLumPerSFR");
        addr[nt] = &All.IonizingLumPerSFR;
        id[nt++] = REAL;

#endif

#ifdef SINKS
        strcpy(tag[nt], "SinkHsml");
        addr[nt] = &All.SinkHsml;
        id[nt++] = REAL;

        strcpy(tag[nt], "SinkDensThresh");
        addr[nt] = &All.SinkDensThresh;
        id[nt++] = REAL;
#endif


#ifdef BP_REAL_CRs
        strcpy(tag[nt], "MinCREnergy");
        addr[nt] = &All.ecr_min;
        id[nt++] = REAL;

        strcpy(tag[nt], "MaxCREnergy");
        addr[nt] = &All.ecr_max;
        id[nt++] = REAL;
#endif

#ifdef SNAP_SET_TG
        strcpy(tag[nt], "SnapNumFac");
        addr[nt] = &All.SnapNumFac;
        id[nt++] = INT;
#endif

#ifdef END_TIME_DYN_BASED
        strcpy(tag[nt], "EndTimeDens"); 
        addr[nt] = &All.EndTimeDens;
        id[nt++] = REAL;
#endif

#ifdef GENERATE_GAS_IN_ICS
#ifdef GENERATE_GAS_TG
        strcpy(tag[nt], "GenGasRefFac");
        addr[nt] = &All.GenGasRefFac;
        id[nt++] = INT;
#endif
#endif

        if((fd = fopen(fname, "r")))
        {
            sprintf(buf, "%s%s", fname, "-usedvalues");
            if(!(fdout = fopen(buf, "w")))
            {
                printf("error opening file '%s' \n", buf);
                errorFlag = 1;
            }
            else
            {
                printf("Obtaining parameters from file '%s':\n", fname);
                while(!feof(fd))
                {
                    char *ret;

                    *buf = 0;
                    ret = fgets(buf, 200, fd);
                    if(ret == NULL)
                        continue;
                    if(sscanf(buf, "%s%s%s", buf1, buf2, buf3) < 2)
                        continue;

                    if(buf1[0] == '%')
                        continue;

                    for(i = 0, j = -1; i < nt; i++)
                        if(strcmp(buf1, tag[i]) == 0)
                        {
                            j = i;
                            tag[i][0] = 0;
                            break;
                        }

                    if(j >= 0)
                    {
                        switch (id[j])
                        {
                            case REAL:
                                *((double *) addr[j]) = atof(buf2);
                                fprintf(fdout, "%-35s%g\n", buf1, *((double *) addr[j]));
                                fprintf(stdout, "%-35s%g\n", buf1, *((double *) addr[j]));
                                break;
                            case STRING:
                                strcpy((char *) addr[j], buf2);
                                fprintf(fdout, "%-35s%s\n", buf1, buf2);
                                fprintf(stdout, "%-35s%s\n", buf1, buf2);
                                break;
                            case INT:
                                *((int *) addr[j]) = atoi(buf2);
                                fprintf(fdout, "%-35s%d\n", buf1, *((int *) addr[j]));
                                fprintf(stdout, "%-35s%d\n", buf1, *((int *) addr[j]));
                                break;
                            case MULTICHOICE:
                                {
                                    int value = parse_multichoice(choices[j], buf2);
                                    *((int *) addr[j]) = value;
                                    char * parsed = NULL;
                                    if(value == 0) {
                                        parsed = format_multichoice(choices[j], -1);
                                        fprintf(stdout, 
                                                "Error in file %s:   Tag '%s' possible choices are: %s.\n",
                                                fname, buf1, parsed);
                                        errorFlag = 1;
                                    } else {
                                        parsed = format_multichoice(choices[j], *((int *) addr[j]));
                                        fprintf(fdout, "%-35s%s\n", buf1, parsed);
                                        fprintf(stdout, "%-35s%s\n", buf1, parsed);
                                    }
                                    free(parsed);
                                }
                                break;
                        }
                    }
                    else
                    {
#ifdef ALLOWEXTRAPARAMS
                        fprintf(stdout, "WARNING from file %s:   Tag '%s' ignored !\n", fname, buf1);
#else
                        fprintf(stdout, "Error in file %s:   Tag '%s' not allowed or multiple defined.\n",
                                fname, buf1);
                        errorFlag = 1;
#endif
                    }
                }
                fclose(fd);
                fclose(fdout);
                printf("\n");

                i = strlen(All.OutputDir);
                if(i > 0)
                    if(All.OutputDir[i - 1] != '/')
                        strcat(All.OutputDir, "/");

#ifdef INVARIANCETEST
                i = strlen(All.OutputDir);
                All.OutputDir[i - 1] = 0;
                strcat(All.OutputDir, "/run0");
                i = strlen(All.OutputDir);
                All.OutputDir[i - 1] += Color;
                mkdir(All.OutputDir, 02755);
                strcat(All.OutputDir, "/");

                sprintf(buf1, "%s%s", All.OutputDir, "logfile.txt");
                printf("stdout will now appear in the file '%s'\n", buf1);
                fflush(stdout);
                freopen(buf1, "w", stdout);
#endif

            }
        }
        else
        {
            printf("Parameter file %s not found.\n", fname);
            errorFlag = 1;
        }

        for(i = 0; i < nt; i++)
        {
                if(*tag[i])
                {
                    printf("Error. I miss a value for tag '%s' in parameter file '%s'.\n", tag[i], fname);
                    errorFlag = 1;
                }
        }

        {
            if(All.DensityKernelType >= density_kernel_type_end()) {
                printf("Error. DensityKernelType can be\n");
                for(i = 0; i < density_kernel_type_end(); i++) {
                    printf("%d %s\n", i, density_kernel_name(i));
                }
                errorFlag = 1; 
            }
            printf("The Density Kernel type is %d (%s)\n", All.DensityKernelType, density_kernel_name(All.DensityKernelType));
            All.DesNumNgb = density_kernel_desnumngb(All.DensityKernelType, 
                    All.DensityResolutionEta);
            printf("The Density resolution is %g * mean separation, or %d neighbours\n",
                    All.DensityResolutionEta, All.DesNumNgb);
            int k = 0;
            for(k = 0; k < 2; k++) {
                char fn[1024];
                sprintf(fn, "density-kernel-%02d.txt", k);
                FILE * fd = fopen(fn, "w");
                double support = density_kernel_support(k);
                density_kernel_t kernel;
                density_kernel_init_with_type(&kernel, k, support); 
                double max = 1000;
                for(i = 0 ; i < max; i ++) {
                    double u = i / max;
                    double q = i / max * support;
                    fprintf(fd, "%g %g %g \n", 
                           q,
                           density_kernel_wk(&kernel, u),
                           density_kernel_dwk(&kernel, u)
                    );
                }
                fclose(fd);
            }
        }
        if(All.OutputListOn && errorFlag == 0)
            errorFlag += read_outputlist(All.OutputListFilename);
        else
            All.OutputListLength = 0;

    }

    MPI_Bcast(&errorFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(errorFlag)
    {
        fflush(stdout);
        fflush(stderr);
        MPI_Finalize();
        exit(0);
    }



    /* now communicate the relevant parameters to the other processes */
    MPI_Bcast(&All, sizeof(struct global_data_all_processes), MPI_BYTE, 0, MPI_COMM_WORLD);



    if(All.NumWritersPerSnapshot > NTask)
    {
       All.NumWritersPerSnapshot = NTask;
    }
    if(All.NumWritersPerPIG > NTask)
    {
       All.NumWritersPerPIG = NTask;
    }

#ifdef BLACK_HOLES
        /* parse blackhole feedback method string */
        {
            if(HAS(All.BlackHoleFeedbackMethod,  BH_FEEDBACK_TOPHAT)
                ==  HAS(All.BlackHoleFeedbackMethod, BH_FEEDBACK_SPLINE)){
                printf("error BlackHoleFeedbackMethod contains either tophat or spline, but both\n");
                endrun(0);
            }
            if(HAS(All.BlackHoleFeedbackMethod, BH_FEEDBACK_MASS)
                ==  HAS(All.BlackHoleFeedbackMethod, BH_FEEDBACK_VOLUME)){
                printf("error BlackHoleFeedbackMethod contains either volume or mass, but both\n");
                endrun(0);
            }
        }
#endif
#ifdef SFR
        if(!HAS(All.StarformationCriterion, SFR_CRITERION_DENSITY)) {
            printf("error: At least use SFR_CRITERION_DENSITY\n");
            endrun(0);
        }
#if ! defined SPH_GRAD_RHO || ! defined METALS
        if(HAS(All.StarformationCriterion, SFR_CRITERION_MOLECULAR_H2)) {
            printf("error: enable SPH_GRAD_RHO to use h2 criterion in sfr \n");
            endrun(0);
        }
        if(HAS(All.StarformationCriterion, SFR_CRITERION_SELFGRAVITY)) {
            printf("error: enable SPH_GRAD_RHO to use selfgravity in sfr \n");
            endrun(0);
        } 
#endif

#endif

    if(All.TypeOfTimestepCriterion >= 3)
    {
        if(ThisTask == 0)
        {
            printf("The specified timestep criterion\n");
            printf("is not valid\n");
        }
        endrun(0);
    }

#if defined(LONG_X) ||  defined(LONG_Y) || defined(LONG_Z)
#ifndef NOGRAVITY
    if(ThisTask == 0)
    {
        printf("Code was compiled with LONG_X/Y/Z, but not with NOGRAVITY.\n");
        printf("Stretched periodic boxes are not implemented for gravity yet.\n");
    }
    endrun(0);
#endif
#endif

#ifdef SFR

#ifndef MOREPARAMS
    if(ThisTask == 0)
    {
        printf("Code was compiled with SFR, but not with MOREPARAMS.\n");
        printf("This is not allowed.\n");
    }
    endrun(0);
#endif

    if(All.StarformationOn == 0)
    {
        if(ThisTask == 0)
        {
            printf("StarformationOn is disabled!\n");
        }
    }
    if(All.CoolingOn == 0)
    {
        if(ThisTask == 0)
        {
            printf("You try to use the code with star formation enabled,\n");
            printf("but you did not switch on cooling.\nThis mode is not supported.\n");
        }
        endrun(0);
    }
#else
    if(All.StarformationOn == 1)
    {
        if(ThisTask == 0)
        {
            printf("Code was compiled with star formation switched off.\n");
            printf("You must set `StarformationOn=0', or recompile the code.\n");
        }
        All.StarformationOn = 0;
    }
#endif



#ifdef METALS
#ifndef SFR
    if(ThisTask == 0)
    {
        printf("Code was compiled with METALS, but not with SFR.\n");
        printf("This is not allowed.\n");
    }
    endrun(0);
#endif
#endif

#ifndef MOREPARAMS
#ifdef TIME_DEP_ART_VISC
    if(ThisTask == 0)
    {
        fprintf(stdout, "Code was compiled with TIME_DEP_ART_VISC, but not with MOREPARAMS.\n");
        fprintf(stdout, "This is not allowed.\n");
    }
    endrun(0);
#endif

#ifdef DARKENERGY
    if(ThisTask == 0)
    {
        fprintf(stdout, "Code was compiled with DARKENERGY, but not with MOREPARAMS.\n");
        fprintf(stdout, "This is not allowed.\n");
    }
    endrun(0);
#endif

#ifdef TIMEDEPDE
    if(ThisTask == 0)
    {
        fprintf(stdout, "Code was compiled with TIMEDEPDE, but not with MOREPARAMS.\n");
        fprintf(stdout, "This is not allowed.\n");
    }
    endrun(0);
#endif
#endif

#ifdef TIMEDEPDE
#ifndef DARKENERGY
    if(ThisTask == 0)
    {
        fprintf(stdout, "Code was compiled with TIMEDEPDE, but not with DARKENERGY.\n");
        fprintf(stdout, "This is not allowed.\n");
    }
    endrun(0);
#endif
#endif

#undef REAL
#undef STRING
#undef INT
#undef MAXTAGS

}


/*! this function reads a table with a list of desired output times. The table
 *  does not have to be ordered in any way, but may not contain more than
 *  MAXLEN_OUTPUTLIST entries.
 */
int read_outputlist(char *fname)
{
    FILE *fd;
    int count, flag;
    char buf[512];

    if(!(fd = fopen(fname, "r")))
    {
        printf("can't read output list in file '%s'\n", fname);
        return 1;
    }

    All.OutputListLength = 0;

    while(1)
    {
        if(fgets(buf, 500, fd) != buf)
            break;

        count = sscanf(buf, " %lg %d ", &All.OutputListTimes[All.OutputListLength], &flag);

        if(count == 1)
            flag = 1;

        if(count == 1 || count == 2)
        {
            if(All.OutputListLength >= MAXLEN_OUTPUTLIST)
            {
                if(ThisTask == 0)
                    printf("\ntoo many entries in output-list. You should increase MAXLEN_OUTPUTLIST=%d.\n",
                            (int) MAXLEN_OUTPUTLIST);
                endrun(13);
            }

            All.OutputListFlag[All.OutputListLength] = flag;
            All.OutputListLength++;
        }
    }

    fclose(fd);

    printf("\nfound %d times in output-list.\n", All.OutputListLength);

    return 0;
}


/*! If a restart from restart-files is carried out where the TimeMax variable
 * is increased, then the integer timeline needs to be adjusted. The approach
 * taken here is to reduce the resolution of the integer timeline by factors
 * of 2 until the new final time can be reached within TIMEBASE.
 */
void readjust_timebase(double TimeMax_old, double TimeMax_new)
{
    int i;
    int64_t ti_end;

    if(sizeof(int64_t) != 8)
    {
        if(ThisTask == 0)
            printf("\nType 'int64_t' is not 64 bit on this platform\n\n");
        endrun(555);
    }

    if(ThisTask == 0)
    {
        printf("\nAll.TimeMax has been changed in the parameterfile\n");
        printf("Need to adjust integer timeline\n\n\n");
    }

    if(TimeMax_new < TimeMax_old)
    {
        if(ThisTask == 0)
            printf("\nIt is not allowed to reduce All.TimeMax\n\n");
        endrun(556);
    }

    ti_end = (int64_t) (log(TimeMax_new / All.TimeBegin) / All.Timebase_interval);

    while(ti_end > TIMEBASE)
    {
        All.Timebase_interval *= 2.0;

        ti_end /= 2;
        All.Ti_Current /= 2;

        All.PM_Ti_begstep /= 2;
        All.PM_Ti_endstep /= 2;

        for(i = 0; i < NumPart; i++)
        {
            P[i].Ti_begstep /= 2;
            P[i].Ti_current /= 2;

            if(P[i].TimeBin > 0)
            {
                P[i].TimeBin--;
                if(P[i].TimeBin <= 0)
                {
                    printf("Error in readjust_timebase(). Minimum Timebin for particle %d reached.\n", i);
                    endrun(8765);
                }
            }
        }

        All.Ti_nextlineofsight /= 2;
    }

    All.TimeMax = TimeMax_new;
}
