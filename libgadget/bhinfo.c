#include <math.h>
#include <string.h>
#include "utils/mymalloc.h"
#include "partmanager.h"
#include "slotsmanager.h"
#include "blackhole.h"
#include "bhdynfric.h"
#include "bhinfo.h"

/* Structure needs to be packed to ensure disc write is the same on all architectures and the record size is correct. */
struct __attribute__((__packed__)) BHinfo{
    /* Stores sizeof(struct BHinfo) - 2 * sizeof(size1) . Allows size of record to be stored in the struct.*/
    int size1;
    MyIDType ID;
    MyFloat Mass;
    MyFloat Mdot;
    MyFloat Density;
    int minTimeBin;
    int encounter;

    double  MinPotPos[3];
    MyFloat MinPot;
    MyFloat BH_Entropy;
    MyFloat BH_SurroundingGasVel[3];
    MyFloat BH_accreted_momentum[3];

    MyFloat BH_accreted_Mass;
    MyFloat BH_accreted_BHMass;
    MyFloat BH_FeedbackWeightSum;

    MyIDType SPH_SwallowID;
    MyIDType SwallowID;

    int CountProgs;
    int Swallowed;

    /****************************************/
    double Pos[3];
    MyFloat BH_SurroundingDensity;
    MyFloat BH_SurroundingParticles;
    MyFloat BH_SurroundingVel[3];
    MyFloat BH_SurroundingRmsVel;

    double BH_DFAccel[3];
    double BH_DragAccel[3];
    double BH_FullTreeGravAccel[3];
    double Velocity[3];
    double Mtrack;
    double Mdyn;

    double KineticFdbkEnergy;
    double NumDM;
    /* Kept for backwards compatibility, not written to*/
    double V1sumDM[3];
    double VDisp;
    double MgasEnc;
    int KEflag;

    MyDouble a;
    /* See size1 above*/
    int size2;
};


void
collect_BH_info(int * ActiveBlackHoles, int NumActiveBlackHoles, struct BHPriv *priv, struct BHDynFricPriv * dynpriv, FILE * FdBlackholeDetails)
{
    int i;

    struct BHinfo * infos = (struct BHinfo *) mymalloc("BHDetailCache", NumActiveBlackHoles * sizeof(struct BHinfo));
    report_memory_usage("BLACKHOLE");

    const int size = sizeof(struct BHinfo) - sizeof(infos[0].size1) - sizeof(infos[0].size2);

    #pragma omp parallel for
    for(i = 0; i < NumActiveBlackHoles; i++)
    {
        const int p_i = ActiveBlackHoles ? ActiveBlackHoles[i] : i;
        if(p_i < 0 || p_i > PartManager->NumPart)
            endrun(1, "Bad index %d in black hole with %d active, %ld total\n", p_i, NumActiveBlackHoles, PartManager->NumPart);
        if(P[p_i].Type != 5)
            endrun(1, "Supposed BH %d of %d has type %d\n", p_i, NumActiveBlackHoles, P[p_i].Type);
        int PI = P[p_i].PI;

        struct BHinfo *info = &infos[i];
        /* Zero the struct*/
        memset(info, 0, sizeof(struct BHinfo));
        info->size1 = size;
        info->size2 = size;
        info->ID = P[p_i].ID;
        info->Mass = BHP(p_i).Mass;
        info->Mdot = BHP(p_i).Mdot;
        info->Density = BHP(p_i).Density;
        info->minTimeBin = BHP(p_i).minTimeBin;
        info->encounter = BHP(p_i).encounter;

        info->BH_Entropy = priv->BH_Entropy[PI];
        int k;
        for(k=0; k < 3; k++) {
            info->MinPotPos[k] = BHP(p_i).MinPotPos[k] - PartManager->CurrentParticleOffset[k];
            info->BH_SurroundingGasVel[k] = priv->BH_SurroundingGasVel[PI][k];
            info->BH_accreted_momentum[k] = priv->BH_accreted_momentum[PI][k];
            info->BH_DragAccel[k] = BHP(p_i).DragAccel[k];
            info->BH_FullTreeGravAccel[k] = P[p_i].FullTreeGravAccel[k];
            info->Pos[k] = P[p_i].Pos[k] - PartManager->CurrentParticleOffset[k];
            info->Velocity[k] = P[p_i].Vel[k];
            info->BH_DFAccel[k] = BHP(p_i).DFAccel[k];
        }

        /****************************************************************************/
        /* Output some DF info for debugging */
        if(dynpriv && dynpriv->BH_SurroundingDensity) {
            info->MinPot = dynpriv->MinPot[PI];
            info->BH_SurroundingDensity = dynpriv->BH_SurroundingDensity[PI];
            info->BH_SurroundingRmsVel = dynpriv->BH_SurroundingRmsVel[PI];
            info->BH_SurroundingParticles = dynpriv->BH_SurroundingParticles[PI];
            info->BH_SurroundingVel[0] = dynpriv->BH_SurroundingVel[PI][0];
            info->BH_SurroundingVel[1] = dynpriv->BH_SurroundingVel[PI][1];
            info->BH_SurroundingVel[2] = dynpriv->BH_SurroundingVel[PI][2];
        }

        /****************************************************************************/
        info->BH_accreted_BHMass = priv->BH_accreted_BHMass[PI];
        info->BH_accreted_Mass = priv->BH_accreted_Mass[PI];
        info->BH_FeedbackWeightSum = priv->BH_FeedbackWeightSum[PI];

        info->SPH_SwallowID = priv->SPH_SwallowID[PI];
        info->SwallowID =  BHP(p_i).SwallowID;
        info->CountProgs = BHP(p_i).CountProgs;
        info->Swallowed =  P[p_i].Swallowed;
        /************************************************************************************************/
        /* When SeedBHDynMass is larger than gas particle mass, we have three mass tracer of blackhole. */
        /* BHP(p_i).Mass : intrinsic mass of BH, accreted every (active) time step.                     */
        /* P[p_i].Mass :  Dynamic mass of BH, used for gravitational interaction.                       */
        /*                Starts to accrete gas particle when BHP(p_i).Mass > SeedBHDynMass             */
        /* BHP(p_i).Mtrack: Initialized as gas particle mass, and is capped at SeedBHDynMass,           */
        /*                 it traces BHP(p_i).Mass by swallowing gas when BHP(p_i).Mass < SeedBHDynMass */
        /************************************************************************************************/
        info->Mtrack = BHP(p_i).Mtrack;
        info->Mdyn = P[p_i].Mass;

        info->KineticFdbkEnergy = BHP(p_i).KineticFdbkEnergy;
        info->NumDM = priv->NumDM[PI];
        info->VDisp = BHP(p_i).VDisp;
        info->MgasEnc = priv->MgasEnc[PI];
        info->KEflag = priv->KEflag[PI];

        info->a = priv->atime;
    }

    fwrite(infos,sizeof(struct BHinfo),NumActiveBlackHoles,FdBlackholeDetails);
    fflush(FdBlackholeDetails);
    myfree(infos);
    int64_t totalN;

    sumup_large_ints(1, &NumActiveBlackHoles, &totalN);
    message(0, "Written details of %ld blackholes in %lu bytes each.\n", totalN, sizeof(struct BHinfo));
}

void
write_blackhole_txt(FILE * FdBlackHoles, const struct UnitSystem units, const double atime)
{
    int total_bh, i;
    double total_mdoteddington;
    double total_mass_holes, total_mdot;

    double Local_BH_mass = 0;
    double Local_BH_Mdot = 0;
    double Local_BH_Medd = 0;
    int Local_BH_num = 0;
    /* Compute total mass of black holes
     * present by summing contents of black hole array*/
    #pragma omp parallel for reduction(+ : Local_BH_num) reduction(+: Local_BH_mass) reduction(+: Local_BH_Mdot) reduction(+: Local_BH_Medd)
    for(i = 0; i < SlotsManager->info[5].size; i ++)
    {
        if(BhP[i].SwallowID != (MyIDType) -1)
            continue;
        Local_BH_num++;
        Local_BH_mass += BhP[i].Mass;
        Local_BH_Mdot += BhP[i].Mdot;
        Local_BH_Medd += BhP[i].Mdot/BhP[i].Mass;
    }

    MPI_Reduce(&Local_BH_mass, &total_mass_holes, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Local_BH_Mdot, &total_mdot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Local_BH_Medd, &total_mdoteddington, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Local_BH_num, &total_bh, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(FdBlackHoles)
    {
        /* convert to solar masses per yr */
        double mdot_in_msun_per_year =
            total_mdot * (units.UnitMass_in_g / SOLAR_MASS) / (units.UnitTime_in_s / SEC_PER_YEAR);

        total_mdoteddington *= 1.0 / ((4 * M_PI * GRAVITY * LIGHTCGS * PROTONMASS /
                    (0.1 * LIGHTCGS * LIGHTCGS * THOMPSON)) * units.UnitTime_in_s);

        fprintf(FdBlackHoles, "%g %d %g %g %g %g\n",
                atime, total_bh, total_mass_holes, total_mdot, mdot_in_msun_per_year, total_mdoteddington);
        fflush(FdBlackHoles);
    }
}
