#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "physconst.h"
#include "cooling.h"
#include "slotsmanager.h"
#include "stats.h"
#include "walltime.h"
#include "cooling_qso_lightup.h"
#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "utils/string.h"

/* global state of system
*/
struct state_of_system
{
    double Mass;
    double EnergyKin;
    double EnergyPot;
    double EnergyInt;
    double EnergyTot;

    double Momentum[4];
    double AngMomentum[4];
    double CenterOfMass[4];
    double MassComp[6];
    /* Only Gas is used */
    double TemperatureComp[6];

    double EnergyKinComp[6];
    double EnergyPotComp[6];
    double EnergyIntComp[6];
    double EnergyTotComp[6];
    double MomentumComp[6][4];
    double AngMomentumComp[6][4];
    double CenterOfMassComp[6][4];
};

static struct stats_params
{
    /* some filenames */
    char EnergyFile[100];
    char CpuFile[100];
    /*Should we store the energy to EnergyFile on PM timesteps.*/
    int OutputEnergyDebug;
    int WriteBlackHoleDetails; /* write BH details every time step*/
    size_t MaxBlackHoleDetails; /* Max size of bh details file*/
} StatsParams;

void
set_stats_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        param_get_string2(ps, "EnergyFile", StatsParams.EnergyFile, sizeof(StatsParams.EnergyFile));
        param_get_string2(ps, "CpuFile", StatsParams.CpuFile, sizeof(StatsParams.CpuFile));
        StatsParams.OutputEnergyDebug = param_get_int(ps, "OutputEnergyDebug");
        StatsParams.WriteBlackHoleDetails = param_get_int(ps,"WriteBlackHoleDetails");
        StatsParams.MaxBlackHoleDetails = 1024L*1024L*1024L*param_get_int(ps, "MaxBlackHoleDetails");
    }
    MPI_Bcast(&StatsParams, sizeof(struct stats_params), MPI_BYTE, 0, MPI_COMM_WORLD);
}

/*!  This function opens various log-files that report on the status and
 *   performance of the simulation. On restart from restart-files
 *   (start-option 1), the code will append to these files.
 */
void
open_outputfiles(int RestartSnapNum, struct OutputFD * fds, const char * OutputDir, int BlackHoleOn, int StarformationOn)
{
    const char mode[3]="a+";
    char * buf;
    char * postfix;
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    memset(fds, 0, sizeof(struct OutputFD));
    fds->FdCPU = NULL;
    fds->FdEnergy = NULL;
    fds->FdBlackHoles = NULL;
    fds->FdSfr = NULL;
    fds->FdBlackholeDetails = NULL;
    fds->TotalBHDetailsBytesWritten = 0;
    fds->BHDetailNumber = 0;
    fds->FdHelium = NULL;

    if(RestartSnapNum != -1) {
        postfix = fastpm_strdup_printf("-R%03d", RestartSnapNum);
    } else {
        postfix = fastpm_strdup_printf("%s", "");
    }

    /* all the processors write to separate files*/
    if(BlackHoleOn && StatsParams.WriteBlackHoleDetails){
        buf = fastpm_strdup_printf("%s/%s%s/%06X", OutputDir,"BlackholeDetails",postfix,ThisTask);
        fastpm_path_ensure_dirname(buf);
        if(!(fds->FdBlackholeDetails = fopen(buf,"a")))
            endrun(1, "Failed to open blackhole detail %s\n", buf);
        myfree(buf);
    }

    /* only the root processors writes to the log files */
    if(ThisTask != 0) {
        myfree(postfix);
        return;
    }

    if(BlackHoleOn) {
        buf = fastpm_strdup_printf("%s/%s%s", OutputDir, "blackholes.txt", postfix);
        fastpm_path_ensure_dirname(buf);
        if(!(fds->FdBlackHoles = fopen(buf, mode)))
            endrun(1, "error in opening file '%s'\n", buf);
        myfree(buf);
    }

    buf = fastpm_strdup_printf("%s/%s%s", OutputDir, StatsParams.CpuFile, postfix);
    fastpm_path_ensure_dirname(buf);
    if(!(fds->FdCPU = fopen(buf, mode)))
        endrun(1, "error in opening file '%s'\n", buf);
    myfree(buf);

    if(StatsParams.OutputEnergyDebug) {
        buf = fastpm_strdup_printf("%s/%s%s", OutputDir, StatsParams.EnergyFile, postfix);
        fastpm_path_ensure_dirname(buf);
        if(!(fds->FdEnergy = fopen(buf, mode)))
            endrun(1, "error in opening file '%s'\n", buf);
        myfree(buf);
    }

    if(StarformationOn) {
        buf = fastpm_strdup_printf("%s/%s%s", OutputDir, "sfr.txt", postfix);
        fastpm_path_ensure_dirname(buf);
        if(!(fds->FdSfr = fopen(buf, mode)))
            endrun(1, "error in opening file '%s'\n", buf);
        const char * helpstr = "# SFR.txt columns are:\n"
                "# 0. Time = current scale factor,\n"
         "# 1. total_sm = expected change in stellar mass this timestep.\n"
         "# This is: sigma_i dM_* = p_* M_* = M_i (1 - exp(-sm_i / M_i))\n"
         "# 2. totsfrrate = current star formation rate in active particles in Msun/year.\n"
         "# This is: sigma_i dM_* / dt_i\n"
         "# 3. rate_in_msunperyear = expected stellar mass formation rate in Msun/year from total_sm,\n"
         "# This is sigma_i dM_* / mean(dt_i), where dt_i is over the starforming particles, and may be\n"
         "# moderately different from columns 1 and 2. This differs from Col3 in Gadget-2.\n"
         "# 4. total_sum_mass_stars = actual mass of stars formed this timestep (discretized total_sm).\n"
         "# This should be a noisier version of total_sm.\n"
         "# 5. total_sum_dtime / total_sum_part : this is the average timsetep (dt) for the currently active star particles\n"
         "# 6. total_sum_part: the number of actively star-forming particles\n"
         "# 7. tot_new stars: number of new star particles spawned or converted this timestep\n";
         fprintf(fds->FdSfr, helpstr);
        myfree(buf);
    }

    if(qso_lightup_on()) {
        buf = fastpm_strdup_printf("%s/%s%s", OutputDir, "helium.txt", postfix);
        fastpm_path_ensure_dirname(buf);
        if(!(fds->FdHelium = fopen(buf, mode)))
            endrun(1, "error in opening file '%s'\n", buf);
        myfree(buf);
    }
    myfree(postfix);
}


void
rotate_bhdetails_file(struct OutputFD * fds, const char * OutputDir, const int RestartSnapNum)
{
    if(!fds->FdBlackholeDetails)
        return;
    if(fds->TotalBHDetailsBytesWritten < StatsParams.MaxBlackHoleDetails)
        return;
    fclose(fds->FdBlackholeDetails);
    char * postfix;
    if(RestartSnapNum != -1) {
        postfix = fastpm_strdup_printf("-R%03d", RestartSnapNum);
    } else {
        postfix = fastpm_strdup_printf("%s", "");
    }
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    char * buf = fastpm_strdup_printf("%s/BlackholeDetails%s.%d/%06X", OutputDir, postfix, fds->BHDetailNumber, ThisTask);
    fastpm_path_ensure_dirname(buf);
    if(!(fds->FdBlackholeDetails = fopen(buf,"a")))
        endrun(1, "Failed to open blackhole detail %s\n", buf);
    myfree(buf);
    myfree(postfix);
    fds->TotalBHDetailsBytesWritten = 0;
    fds->BHDetailNumber++;
    message(0, "Rotating BH Details file to %d with %lu bytes written\n", fds->BHDetailNumber, StatsParams.MaxBlackHoleDetails);
}

/*!  This function closes the global log-files.
*/
void
close_outputfiles(struct OutputFD * fds)
{
    if(fds->FdCPU)
        fclose(fds->FdCPU);
    if(fds->FdEnergy)
        fclose(fds->FdEnergy);
    if(fds->FdSfr)
        fclose(fds->FdSfr);
    if(fds->FdBlackHoles)
        fclose(fds->FdBlackHoles);
    if(fds->FdBlackholeDetails)
        fclose(fds->FdBlackholeDetails);
}


void write_cpu_log(int NumCurrentTiStep, const double atime, FILE * FdCPU, double ElapsedTime)
{
    walltime_summary(0, MPI_COMM_WORLD);

    if(FdCPU)
    {
        int NTask;
        MPI_Comm_size(MPI_COMM_WORLD, &NTask);
        fprintf(FdCPU, "Step %d, Time: %g, MPIs: %d Threads: %d Elapsed: %g\n", NumCurrentTiStep, atime, NTask, omp_get_max_threads(), ElapsedTime);
        walltime_report(FdCPU, 0, MPI_COMM_WORLD);
        fflush(FdCPU);
    }
}

/* This routine computes various global properties of the particle
 * distribution and stores the result in the struct `SysState'.
 * Currently, not all the information that's computed here is
 * actually used (e.g. momentum is not really used anywhere),
 * just the energies are written to a log-file every once in a while.
 */
struct state_of_system compute_global_quantities_of_system(const double Time,  struct part_manager_type * PartManager)
{
    int i, j;
    struct state_of_system sys;
    struct state_of_system SysState;
    double a1, a2, a3;
    int ThisTask;

    a1 = Time;
    a2 = Time * Time;
    a3 = Time * Time * Time;

    double redshift = 1. / Time - 1;
    memset(&sys, 0, sizeof(sys));
    struct UVBG GlobalUVBG = get_global_UVBG(redshift);

    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)
    {
        int j;
        double entr = 0, egyspec;

        sys.MassComp[P[i].Type] += P[i].Mass;

        sys.EnergyPotComp[P[i].Type] += 0.5 * P[i].Mass * P[i].Potential / a1;

        sys.EnergyKinComp[P[i].Type] +=
            0.5 * P[i].Mass * (P[i].Vel[0] * P[i].Vel[0] + P[i].Vel[1] * P[i].Vel[1] + P[i].Vel[2] * P[i].Vel[2]) / a2;

        if(P[i].Type == 0)
        {
            double localJ21 = 0;
            double zreion = 0;
#ifdef EXCUR_REION
            localJ21 = SPHP(i).local_J21;
            zreion = SPHP(i).zreion;
#endif
            struct UVBG uvbg = get_local_UVBG(redshift, &GlobalUVBG, P[i].Pos, PartManager->CurrentParticleOffset, localJ21, zreion);
            entr = SPHP(i).Entropy;
            egyspec = entr / (GAMMA_MINUS1) * pow(SPHP(i).Density / a3, GAMMA_MINUS1);
            sys.EnergyIntComp[0] += P[i].Mass * egyspec;
            double ne = SPHP(i).Ne;
            sys.TemperatureComp[0] += P[i].Mass * get_temp(SPHP(i).Density, egyspec, (1 - HYDROGEN_MASSFRAC), &uvbg, &ne);
        }

        for(j = 0; j < 3; j++)
        {
            sys.MomentumComp[P[i].Type][j] += P[i].Mass * P[i].Vel[j];
            sys.CenterOfMassComp[P[i].Type][j] += P[i].Mass * P[i].Pos[j];
        }

        sys.AngMomentumComp[P[i].Type][0] += P[i].Mass * (P[i].Pos[1] * P[i].Vel[2] - P[i].Pos[2] * P[i].Vel[1]);
        sys.AngMomentumComp[P[i].Type][1] += P[i].Mass * (P[i].Pos[2] * P[i].Vel[0] - P[i].Pos[0] * P[i].Vel[2]);
        sys.AngMomentumComp[P[i].Type][2] += P[i].Mass * (P[i].Pos[0] * P[i].Vel[1] - P[i].Pos[1] * P[i].Vel[0]);
    }


    /* some the stuff over all processors */
    MPI_Reduce(&sys.MassComp[0], &SysState.MassComp[0], 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sys.EnergyPotComp[0], &SysState.EnergyPotComp[0], 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sys.EnergyIntComp[0], &SysState.EnergyIntComp[0], 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sys.EnergyKinComp[0], &SysState.EnergyKinComp[0], 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sys.TemperatureComp[0], &SysState.TemperatureComp[0], 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sys.MomentumComp[0][0], &SysState.MomentumComp[0][0], 6 * 4, MPI_DOUBLE, MPI_SUM, 0,
            MPI_COMM_WORLD);
    MPI_Reduce(&sys.AngMomentumComp[0][0], &SysState.AngMomentumComp[0][0], 6 * 4, MPI_DOUBLE, MPI_SUM, 0,
            MPI_COMM_WORLD);
    MPI_Reduce(&sys.CenterOfMassComp[0][0], &SysState.CenterOfMassComp[0][0], 6 * 4, MPI_DOUBLE, MPI_SUM, 0,
            MPI_COMM_WORLD);

    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0)
    {
        for(i = 0; i < 6; i++) {
            SysState.EnergyTotComp[i] = SysState.EnergyKinComp[i] +
                SysState.EnergyPotComp[i] + SysState.EnergyIntComp[i];
        }

        SysState.Mass = SysState.EnergyKin = SysState.EnergyPot = SysState.EnergyInt = SysState.EnergyTot = 0;

        for(i = 0; i < 6; i++)
        {
            if(SysState.MassComp[i] > 0) {
                SysState.TemperatureComp[i] /= SysState.MassComp[i];
            }
        }

        for(j = 0; j < 3; j++)
            SysState.Momentum[j] = SysState.AngMomentum[j] = SysState.CenterOfMass[j] = 0;

        for(i = 0; i < 6; i++)
        {
            SysState.Mass += SysState.MassComp[i];
            SysState.EnergyKin += SysState.EnergyKinComp[i];
            SysState.EnergyPot += SysState.EnergyPotComp[i];
            SysState.EnergyInt += SysState.EnergyIntComp[i];
            SysState.EnergyTot += SysState.EnergyTotComp[i];

            for(j = 0; j < 3; j++)
            {
                SysState.Momentum[j] += SysState.MomentumComp[i][j];
                SysState.AngMomentum[j] += SysState.AngMomentumComp[i][j];
                SysState.CenterOfMass[j] += SysState.CenterOfMassComp[i][j];
            }
        }

        for(i = 0; i < 6; i++)
            for(j = 0; j < 3; j++)
                if(SysState.MassComp[i] > 0)
                    SysState.CenterOfMassComp[i][j] /= SysState.MassComp[i];

        for(j = 0; j < 3; j++)
            if(SysState.Mass > 0)
                SysState.CenterOfMass[j] /= SysState.Mass;

        for(i = 0; i < 6; i++)
        {
            SysState.CenterOfMassComp[i][3] = SysState.MomentumComp[i][3] = SysState.AngMomentumComp[i][3] = 0;
            for(j = 0; j < 3; j++)
            {
                SysState.CenterOfMassComp[i][3] +=
                    SysState.CenterOfMassComp[i][j] * SysState.CenterOfMassComp[i][j];
                SysState.MomentumComp[i][3] += SysState.MomentumComp[i][j] * SysState.MomentumComp[i][j];
                SysState.AngMomentumComp[i][3] +=
                    SysState.AngMomentumComp[i][j] * SysState.AngMomentumComp[i][j];
            }
            SysState.CenterOfMassComp[i][3] = sqrt(SysState.CenterOfMassComp[i][3]);
            SysState.MomentumComp[i][3] = sqrt(SysState.MomentumComp[i][3]);
            SysState.AngMomentumComp[i][3] = sqrt(SysState.AngMomentumComp[i][3]);
        }

        SysState.CenterOfMass[3] = SysState.Momentum[3] = SysState.AngMomentum[3] = 0;

        for(j = 0; j < 3; j++)
        {
            SysState.CenterOfMass[3] += SysState.CenterOfMass[j] * SysState.CenterOfMass[j];
            SysState.Momentum[3] += SysState.Momentum[j] * SysState.Momentum[j];
            SysState.AngMomentum[3] += SysState.AngMomentum[j] * SysState.AngMomentum[j];
        }

        SysState.CenterOfMass[3] = sqrt(SysState.CenterOfMass[3]);
        SysState.Momentum[3] = sqrt(SysState.Momentum[3]);
        SysState.AngMomentum[3] = sqrt(SysState.AngMomentum[3]);
    }

    /* give everyone the result, maybe the want to do something with it */
    MPI_Bcast(&SysState, sizeof(struct state_of_system), MPI_BYTE, 0, MPI_COMM_WORLD);
    return SysState;
}

/*! This routine first calls a computation of various global
 * quantities of the particle distribution, and then writes some
 * statistics about the energies in the various particle components to
 * the file FdEnergy.
 */
void energy_statistics(FILE * FdEnergy, const double Time, struct part_manager_type * PartManager)
{
    if(!FdEnergy)
        return;

    struct state_of_system SysState = compute_global_quantities_of_system(Time, PartManager);

    message(0, "Time %g Mean Temperature of Gas %g\n",
                Time, SysState.TemperatureComp[0]);

    fprintf(FdEnergy,
            "%g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g\n",
            Time, SysState.TemperatureComp[0], SysState.EnergyInt, SysState.EnergyPot, SysState.EnergyKin, SysState.EnergyIntComp[0],
            SysState.EnergyPotComp[0], SysState.EnergyKinComp[0], SysState.EnergyIntComp[1],
            SysState.EnergyPotComp[1], SysState.EnergyKinComp[1], SysState.EnergyIntComp[2],
            SysState.EnergyPotComp[2], SysState.EnergyKinComp[2], SysState.EnergyIntComp[3],
            SysState.EnergyPotComp[3], SysState.EnergyKinComp[3], SysState.EnergyIntComp[4],
            SysState.EnergyPotComp[4], SysState.EnergyKinComp[4], SysState.EnergyIntComp[5],
            SysState.EnergyPotComp[5], SysState.EnergyKinComp[5], SysState.MassComp[0],
            SysState.MassComp[1], SysState.MassComp[2], SysState.MassComp[3], SysState.MassComp[4],
            SysState.MassComp[5]);

    fflush(FdEnergy);
}
