#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string.h>


#include "allvars.h"
#include "proto.h"
#if defined(SUBFIND_RESHUFFLE_CATALOGUE)
#include "subfind.h"
#endif

/* This function reads initial conditions that are in the default file format
 * of Gadget, i.e. snapshot files can be used as input files.  However, when a
 * snapshot file is used as input, not all the information in the header is
 * used: THE STARTING TIME NEEDS TO BE SET IN THE PARAMETERFILE.
 * Alternatively, the code can be started with restartflag==2, then snapshots
 * from the code can be used as initial conditions-files without having to
 * change the parameterfile.  For gas particles, only the internal energy is
 * read, the density and mean molecular weight will be recomputed by the code.
 * When InitGasTemp>0 is given, the gas temperature will be initialzed to this
 * value assuming a mean colecular weight either corresponding to complete
 * neutrality, or full ionization.
 */

#ifdef AUTO_SWAP_ENDIAN_READIC
int swap_file = 8;
#endif

#if defined(SAVE_HSML_IN_IC_ORDER) || defined(SUBFIND_RESHUFFLE_CATALOGUE)
static unsigned long FileNr;
static long long *NumPartPerFile;
#endif

void read_ic(char *fname)
{
    int i, num_files, rest_files, ngroups, gr, filenr, masterTask, lastTask, groupMaster;
    double u_init, molecular_weight;
    char buf[500];

    CPU_Step[CPU_MISC] += measure_time();

#ifdef RESCALEVINI
    if(ThisTask == 0 && RestartFlag == 0)
    {
        fprintf(stdout, "\nRescaling v_ini !\n\n");
        fflush(stdout);
    }
#endif

    NumPart = 0;
    N_sph = 0;
    N_bh = 0;
    All.TotNumPart = 0;
    num_files = find_files(fname);

#if defined(SAVE_HSML_IN_IC_ORDER) || defined(SUBFIND_RESHUFFLE_CATALOGUE)
    NumPartPerFile = (long long *) mymalloc("NumPartPerFile", num_files * sizeof(long long));

    if(ThisTask == 0)
        get_particle_numbers(fname, num_files);

    MPI_Bcast(NumPartPerFile, num_files * sizeof(long long), MPI_BYTE, 0, MPI_COMM_WORLD);
#endif

    rest_files = num_files;

    while(rest_files > NTask)
    {
        sprintf(buf, "%s.%d", fname, ThisTask + (rest_files - NTask));
        if(All.ICFormat == 3)
            sprintf(buf, "%s.%d.hdf5", fname, ThisTask + (rest_files - NTask));
#if defined(SAVE_HSML_IN_IC_ORDER) || defined(SUBFIND_RESHUFFLE_CATALOGUE)
        FileNr = ThisTask + (rest_files - NTask);
#endif

        ngroups = NTask / All.NumFilesWrittenInParallel;
        if((NTask % All.NumFilesWrittenInParallel))
            ngroups++;
        groupMaster = (ThisTask / ngroups) * ngroups;

        for(gr = 0; gr < ngroups; gr++)
        {
            if(ThisTask == (groupMaster + gr))	/* ok, it's this processor's turn */
                read_file(buf, ThisTask, ThisTask);
            MPI_Barrier(MPI_COMM_WORLD);
        }

        rest_files -= NTask;
    }


    if(rest_files > 0)
    {
        distribute_file(rest_files, 0, 0, NTask - 1, &filenr, &masterTask, &lastTask);

        if(num_files > 1)
        {
            sprintf(buf, "%s.%d", fname, filenr);
            if(All.ICFormat == 3)
                sprintf(buf, "%s.%d.hdf5", fname, filenr);
#if defined(SAVE_HSML_IN_IC_ORDER) || defined(SUBFIND_RESHUFFLE_CATALOGUE)
            FileNr = filenr;
#endif
        }
        else
        {
            sprintf(buf, "%s", fname);
            if(All.ICFormat == 3)
                sprintf(buf, "%s.hdf5", fname);
#if defined(SAVE_HSML_IN_IC_ORDER) || defined(SUBFIND_RESHUFFLE_CATALOGUE)
            FileNr = 0;
#endif
        }

        ngroups = rest_files / All.NumFilesWrittenInParallel;
        if((rest_files % All.NumFilesWrittenInParallel))
            ngroups++;

        for(gr = 0; gr < ngroups; gr++)
        {
            if((filenr / All.NumFilesWrittenInParallel) == gr)	/* ok, it's this processor's turn */
                read_file(buf, masterTask, lastTask);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

#if defined(SUBFIND_RESHUFFLE_CATALOGUE)
    subfind_reshuffle_free();
#endif

    myfree(CommBuffer);


    if(header.flag_ic_info != FLAG_SECOND_ORDER_ICS)
    {
        /* this makes sure that masses are initialized in the case that the mass-block
           is empty for this particle type */
        for(i = 0; i < NumPart; i++)
        {
            if(All.MassTable[P[i].Type] != 0)
                P[i].Mass = All.MassTable[P[i].Type];
        }
    }

#if defined(BLACK_HOLES) && defined(SWALLOWGAS)
    if(RestartFlag == 0)
    {
        All.MassTable[5] = 0;
    }
#endif

#ifdef SFR
    if(RestartFlag == 0)
    {
        if(All.MassTable[4] == 0 && All.MassTable[0] > 0)
        {
            All.MassTable[0] = 0;
            All.MassTable[4] = 0;
        }
    }
#endif

    u_init = (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.InitGasTemp;
    u_init *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;	/* unit conversion */

    if(All.InitGasTemp > 1.0e4)	/* assuming FULL ionization */
        molecular_weight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));
    else				/* assuming NEUTRAL GAS */
        molecular_weight = 4 / (1 + 3 * HYDROGEN_MASSFRAC);

    u_init /= molecular_weight;

    All.InitGasU = u_init;

#ifdef NO_UTHERM_IN_IC_FILE
    if(RestartFlag == 0)
        for(i = 0; i < N_sph; i++)
            SPHP(i).Entropy = 0;
#endif


    if(RestartFlag == 0)
    {
        if(All.InitGasTemp > 0)
        {
            for(i = 0; i < N_sph; i++)
            {
                if(ThisTask == 0 && i == 0 && SPHP(i).Entropy == 0)
                    printf("Initializing u from InitGasTemp !\n");

                if(SPHP(i).Entropy == 0)
                    SPHP(i).Entropy = All.InitGasU;

                /* Note: the coversion to entropy will be done in the function init(),
                   after the densities have been computed */
            }
        }
    }

    for(i = 0; i < N_sph; i++)
        SPHP(i).Entropy = DMAX(All.MinEgySpec, SPHP(i).Entropy);

#ifdef EOS_DEGENERATE
    for(i = 0; i < N_sph; i++)
        SPHP(i).u = 0;
#endif

    MPI_Barrier(MPI_COMM_WORLD);

    if(ThisTask == 0)
    {
        printf("reading done.\n");
        fflush(stdout);
    }

    if(ThisTask == 0)
    {
        printf("Total number of particles :  %d%09d\n\n",
                (int) (All.TotNumPart / 1000000000), (int) (All.TotNumPart % 1000000000));
        fflush(stdout);
    }

    CPU_Step[CPU_SNAPSHOT] += measure_time();
}


/*! This function reads out the buffer that was filled with particle data.
*/
void empty_read_buffer(enum iofields blocknr, int bytes_per_blockelement, int offset, int pc, int type)
{
    int n, k;
    MyIDType *ip;
    int32_t * i32p;
    float *fp_single;

#if defined(DISTORTIONTENSORPS) && defined(DISTORTION_READALL)
    int alpha, beta;
#endif

    int vt, vpb;
    char *cp;
    cp = (char *) CommBuffer;
    fp_single = (float *) CommBuffer;
    ip = (MyIDType *) CommBuffer;
    i32p = (int32_t*) CommBuffer;
    vt = get_datatype_in_block(blocknr);
    int elsize = get_elsize_in_block(blocknr, header.flag_doubleprecision);
    double tmp = 0;
#define READREAL(ptr) \
    (tmp = header.flag_doubleprecision?((double*) ptr)[0]:((float*) ptr)[0], \
     ptr += elsize, \
     tmp)

#ifdef AUTO_SWAP_ENDIAN_READIC
    vpb = get_values_per_blockelement(blocknr);
    swap_Nbyte(cp, vpb * pc, get_datatype_elsize(vt, header.flag_doubleprecision));
#endif

#ifdef COSMIC_RAYS
    int CRpop;
#endif

    switch (blocknr)
    {
        case IO_POS:		/* positions */
            for(n = 0; n < pc; n++)
                for(k = 0; k < 3; k++)
                    P[offset + n].Pos[k] = READREAL(cp);

            for(n = 0; n < pc; n++) {
                P[offset + n].Type = type;	/* initialize type here as well */
                if(type == 5) {
                    P[offset + n].PI = N_bh + n;
                }
            }
            /* and also increase the particle counts */
            NumPart += pc;
            if(type == 0) {
                N_sph += pc;
            }
            if(type == 5) {
                N_bh += pc;
            }
            break;

        case IO_VEL:		/* velocities */
            for(n = 0; n < pc; n++)
                for(k = 0; k < 3; k++)
#ifdef RESCALEVINI
                    /* scaling v to use same IC's for different cosmologies */
                    if(RestartFlag == 0)
                        P[offset + n].Vel[k] = READREAL(cp) * All.VelIniScale;
                    else
                        P[offset + n].Vel[k] = READREAL(cp);
#else
            P[offset + n].Vel[k] = READREAL(cp);
#endif
            break;

        case IO_ID:		/* particle ID */
            for(n = 0; n < pc; n++) {
                if(bytes_per_blockelement == 8) 
                    P[offset + n].ID = *ip++;
                if(bytes_per_blockelement == 4) 
                    P[offset + n].ID = *i32p++;
                if (type == 5) {
                    BhP[P[offset +n].PI].ID = P[offset + n].ID;
                }
            } 
            break;

        case IO_MASS:		/* particle mass */
            for(n = 0; n < pc; n++)
                P[offset + n].Mass = READREAL(cp);
            break;


        case IO_SHEET_ORIENTATION:	/* initial particle sheet orientation */
#ifdef DISTORTIONTENSORPS
#if !defined(COMOVING_DISTORTION) || defined(COMOVING_READIC)
            for(n = 0; n < pc; n++)
            {
                P[offset + n].V_matrix[0][0] = READREAL(cp);
                P[offset + n].V_matrix[0][1] = READREAL(cp);
                P[offset + n].V_matrix[0][2] = READREAL(cp);
                P[offset + n].V_matrix[1][0] = READREAL(cp);
                P[offset + n].V_matrix[1][1] = READREAL(cp);
                P[offset + n].V_matrix[1][2] = READREAL(cp);
                P[offset + n].V_matrix[2][0] = READREAL(cp);
                P[offset + n].V_matrix[2][1] = READREAL(cp);
                P[offset + n].V_matrix[2][2] = READREAL(cp);
            }
#endif
#endif
            break;

        case IO_INIT_DENSITY:	/* initial stream density */
#ifdef DISTORTIONTENSORPS
#if !defined(COMOVING_DISTORTION) || defined(COMOVING_READIC)
            for(n = 0; n < pc; n++)
                P[offset + n].init_density = READREAL(cp) * pow(All.TimeBegin, 3.0);
            break;
#endif
#endif

        case IO_CAUSTIC_COUNTER:	/* initial caustic counter */
#ifdef DISTORTIONTENSORPS
#if !defined(COMOVING_DISTORTION) || defined(COMOVING_READIC)
            for(n = 0; n < pc; n++)
                P[offset + n].caustic_counter = READREAL(cp);
            break;
#endif
#endif

        case IO_DISTORTIONTENSORPS:	/* phase-space distortion tensor */
#if defined(DISTORTIONTENSORPS) && defined(DISTORTION_READALL)
            for(n = 0; n < pc; n++)
            {
                for(alpha = 0; alpha < 6; alpha++)
                    for(beta = 0; beta < 6; beta++)
                        P[offset + n].distortion_tensorps[alpha][beta] = READREAL(cp);
            }

#endif
            break;

        case IO_SECONDORDERMASS:
            for(n = 0; n < pc; n++)
            {
                P[offset + n].OldAcc = P[offset + n].Mass;	/* use this to temporarily store the masses in the 2plt IC case */
                P[offset + n].Mass = READREAL(cp);
            }
            break;

        case IO_U:			/* temperature */
            for(n = 0; n < pc; n++)
                SPHP(offset + n).Entropy = READREAL(cp);
            break;

        case IO_RHO:		/* density */
            for(n = 0; n < pc; n++)
                SPHP(offset + n).d.Density = READREAL(cp);
            break;

        case IO_NE:		/* electron abundance */
#if defined(COOLING) || defined(CHEMISTRY) || defined(UM_CHEMISTRY)
            for(n = 0; n < pc; n++)
#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
                SPHP(offset + n).elec = READREAL(cp);
#else
            SPHP(offset + n).Ne = READREAL(cp);
#endif
#endif
            break;

#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
        case IO_NH:		/* neutral hydrogen abundance */
            for(n = 0; n < pc; n++)
                SPHP(offset + n).HI = READREAL(cp);
            break;

        case IO_HII:		/* ionized hydrogen abundance */
            for(n = 0; n < pc; n++)
                SPHP(offset + n).HII = READREAL(cp);
            break;

        case IO_HeI:		/* neutral Helium */
            for(n = 0; n < pc; n++)
                SPHP(offset + n).HeI = READREAL(cp);
            break;

        case IO_HeII:		/* ionized Heluum */
            for(n = 0; n < pc; n++)
                SPHP(offset + n).HeII = READREAL(cp);

        case IO_HeIII:		/* double ionised Helium */
            for(n = 0; n < pc; n++)
                SPHP(offset + n).HeIII = READREAL(cp);
            break;

        case IO_H2I:		/* H2 molecule */
            for(n = 0; n < pc; n++)
                SPHP(offset + n).H2I = READREAL(cp);
            break;

        case IO_H2II:		/* ionised H2 molecule */
            for(n = 0; n < pc; n++)
                SPHP(offset + n).H2II = READREAL(cp);

        case IO_HM:		/* H minus */
            for(n = 0; n < pc; n++)
                SPHP(offset + n).HM = READREAL(cp);
            break;

        case IO_HeHII:		/* HeH+ */
#if defined (UM_CHEMISTRY)
            for(n = 0; n < pc; n++)
                SPHP(offset + n).HeHII = READREAL(cp);
#endif
            break;

        case IO_HD:		/* HD */
#if defined (UM_CHEMISTRY) &&  defined (UM_HD_COOLING)
            for(n = 0; n < pc; n++)
                SPHP(offset + n).HD = READREAL(cp);
#endif
            break;

        case IO_DI:		/* D */
#if defined (UM_CHEMISTRY) &&  defined (UM_HD_COOLING)
            for(n = 0; n < pc; n++)
                SPHP(offset + n).DI = *fp++;
#endif
            break;

        case IO_DII:		/* D plus */
#if defined (UM_CHEMISTRY) &&  defined (UM_HD_COOLING)
            for(n = 0; n < pc; n++)
                SPHP(offset + n).DII = READREAL(cp);
#endif
            break;

#else
        case IO_NH:		/* neutral hydrogen abundance */
        case IO_HII:		/* ionized hydrogen abundance */
        case IO_HeI:		/* neutral Helium */
        case IO_HeII:		/* ionized Heluum */
        case IO_HeIII:		/* double ionised Helium */
        case IO_H2I:		/* H2 molecule */
        case IO_H2II:		/* ionised H2 molecule */
        case IO_HM:		/* H minus */
        case IO_HeHII:      /* HeH+ */
        case IO_HD:		/* HD */
        case IO_DI:		/* D */
        case IO_DII:		/* D plus  */
            break;
#endif

        case IO_HSML:		/* SPH smoothing length */
            for(n = 0; n < pc; n++)
                P[offset + n].Hsml = READREAL(cp);
            break;


        case IO_AGE:		/* Age of stars */
#ifdef STELLARAGE
            for(n = 0; n < pc; n++)
                P[offset + n].StellarAge = READREAL(cp);
#endif
            break;

        case IO_Z:			/* Gas and star metallicity */
#ifdef METALS
            for(n = 0; n < pc; n++)
                P[offset + n].Metallicity = READREAL(cp);
#endif
            break;

        case IO_EGYPROM:		/* SN Energy Reservoir */
            break;

        case IO_EGYCOLD:		/* Cold  SN Energy Reservoir */
            break;

        case IO_VTURB:	/* Turbulent Velocity */
#ifdef JD_VTURB
            for(n = 0; n < pc; n++)
                SPHP(offset + n).Vturb = READREAL(cp);
#endif
            break;

        case IO_VRMS:
#ifdef JD_VTURB
            for(n = 0; n < pc; n++)
                SPHP(offset + n).Vrms = READREAL(cp);
#endif
            break;

        case IO_VBULK:
#ifdef JD_VTURB
            for(n = 0; n < pc; n++)
                for(k = 0; k < 3; k++)
                    SPHP(offset + n).Vbulk[k] = READREAL(cp);
#endif
            break;

        case IO_VDIV:
#ifdef JD_VTURB
            for(n = 0; n < pc; n++)
                SPHP(offset+n).v.DivVel = READREAL(cp);
#endif
            break;

        case IO_VROT:
#ifdef JD_VTURB
            for(n = 0; n < pc; n++)
                SPHP(offset+n).r.CurlVel = READREAL(cp);
#endif
            break;

        case IO_TRUENGB:
#ifdef JD_VTURB
            for(n = 0; n < pc; n++)
                SPHP(offset + n).TrueNGB = READREAL(cp);
#endif
            break;

        case IO_DPP:
#ifdef JD_DPP
            for(n = 0; n < pc; n++)
                SPHP(offset + n).Dpp = READREAL(cp);
#endif
            break;

        case IO_BFLD:		/* Magnetic field */
#ifdef MAGNETIC
            for(n = 0; n < pc; n++)
                for(k = 0; k < 3; k++)
                    SPHP(offset + n).BPred[k] = READREAL(cp);
#ifdef TRACEDIVB
            SPHP(offset + n).divB = 0;
#endif
#ifdef MAGNETICSEED
            SPHP(offset + n).MagSeed = 0;
            for(k = 0; k < 3; k++)
                SPHP(offset + n).BPred[k] = 0.0;
#endif
#ifdef DIVBCLEANING_DEDNER
            SPHP(offset + n).Phi = 0;
            SPHP(offset + n).PhiPred = 0;
#endif
#endif
            break;

        case IO_CR_C0:		/* Adiabatic invariant for cosmic rays */
#ifdef COSMIC_RAYS
            for(n = 0; n < pc; n++)
                for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
                    SPHP(offset + n).CR_C0[CRpop] = READREAL(cp);
#endif
            break;

        case IO_CR_Q0:		/* Adiabatic invariant for cosmic rays */
#ifdef COSMIC_RAYS
            for(n = 0; n < pc; n++)
                for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
                    SPHP(offset + n).CR_q0[CRpop] = READREAL(cp);
#endif
            break;

        case IO_CR_P0:
            break;

        case IO_CR_E0:
#ifdef COSMIC_RAYS
            for(n = 0; n < pc; n++)
                for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
                    SPHP(offset + n).CR_E0[CRpop] = READREAL(cp);
#endif
            break;

        case IO_CR_n0:
#ifdef COSMIC_RAYS
            for(n = 0; n < pc; n++)
                for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
                    SPHP(offset + n).CR_n0[CRpop] = READREAL(cp);
#endif
            break;

        case IO_CR_ThermalizationTime:
        case IO_CR_DissipationTime:
            break;

        case IO_BHMASS:
#ifdef BLACK_HOLES
            for(n = 0; n < pc; n++)
                BHP(offset + n).Mass = READREAL(cp);
#endif
            break;

        case IO_BHMDOT:
#ifdef BLACK_HOLES
            for(n = 0; n < pc; n++)
                BHP(offset + n).Mdot = READREAL(cp);
#endif
            break;

        case IO_BHPROGS:
#ifdef BH_COUNTPROGS
            for(n = 0; n < pc; n++) {
                if(bytes_per_blockelement == 8) 
                    BHP(offset + n).CountProgs = *ip++;
                if(bytes_per_blockelement == 4) 
                    BHP(offset + n).CountProgs = *i32p++;
            }
#endif
            break;

        case IO_BHMBUB:
#ifdef BH_BUBBLES
            for(n = 0; n < pc; n++)
                BHP(offset + n).Mass_bubbles = READREAL(cp);
#endif
            break;

        case IO_BHMINI:
#ifdef BH_BUBBLES
            for(n = 0; n < pc; n++)
                BHP(offset + n).Mass_ini = READREAL(cp);
#endif
            break;

        case IO_BHMRAD:
#ifdef UNIFIED_FEEDBACK
            for(n = 0; n < pc; n++)
                BHP(offset + n).Mass_radio = READREAL(cp);
#endif
            break;

        case IO_EOSXNUC:
#ifdef EOS_DEGENERATE
            for(n = 0; n < pc; n++)
                for(k = 0; k < EOS_NSPECIES; k++)
                    SPHP(offset + n).xnuc[k] = READREAL(cp);
#endif
            break;

        case IO_Zs:
            break;

        case IO_ZAGE:
            break;

        case IO_ZAGE_LLV:
            break;

        case IO_iMass:
            break;

        case IO_CONTRIB:
            break;


            if(RestartFlag != 0)
            {
                case IO_nHII:
#ifdef RADTRANSFER
                    for(n = 0; n < pc; n++)
                    {
                        SPHP(offset + n).nHII = READREAL(cp);
                        SPHP(offset + n).nHI = 1.0 - SPHP(offset + n).nHII;
                        SPHP(offset + n).n_elec = SPHP(offset + n).nHII;
                    }
#endif
                    break;

                case IO_RADGAMMA:
#ifdef RADTRANSFER
                    for(n = 0; n < pc; n++)
                        for(k = 0; k < N_BINS; k++)
                            SPHP(offset + n).n_gamma[k] = READREAL(cp);
#endif
                    break;

                case IO_nHeII:
#ifdef RADTRANSFER
                    for(n = 0; n < pc; n++)
                        SPHP(offset + n).nHeII = READREAL(cp);
#endif
                    break;

                case IO_nHeIII:
#ifdef RADTRANSFER
                    for(n = 0; n < pc; n++)
                    {
                        SPHP(offset + n).nHeIII = READREAL(cp);
                        SPHP(offset + n).nHeI = 1.0 - SPHP(offset + n).nHeII - SPHP(offset + n).nHeIII;
                        SPHP(offset + n).n_elec +=
                            (SPHP(offset + n).nHeII + 2.0 * SPHP(offset + n).nHeIII) * (1.0 -
                                    HYDROGEN_MASSFRAC) / 4.0 /
                            HYDROGEN_MASSFRAC;
                    }
#endif
                    break;

            }

        case IO_DMHSML:
#if defined(SUBFIND_RESHUFFLE_CATALOGUE) && defined(SUBFIND)
            for(n = 0; n < pc; n++)
                P[offset + n].DM_Hsml = *fp_single++;
#endif
            break;

        case IO_DMDENSITY:
#if defined(SUBFIND_RESHUFFLE_CATALOGUE) && defined(SUBFIND)
            for(n = 0; n < pc; n++)
                P[offset + n].u.DM_Density = *fp_single++;
#endif
            break;

        case IO_DMVELDISP:
#if defined(SUBFIND_RESHUFFLE_CATALOGUE) && defined(SUBFIND)
            for(n = 0; n < pc; n++)
                P[offset + n].v.DM_VelDisp = *fp_single++;
#endif
            break;

        case IO_DMHSML_V:
#if defined(SUBFIND_RESHUFFLE_CATALOGUE_WITH_VORONOI) && defined(SUBFIND)
            for(n = 0; n < pc; n++)
                P[offset + n].DM_Hsml_V = *fp_single++;
#endif
            break;

        case IO_DMDENSITY_V:
#if defined(SUBFIND_RESHUFFLE_CATALOGUE_WITH_VORONOI) && defined(SUBFIND)
            for(n = 0; n < pc; n++)
                P[offset + n].DM_Density_V = *fp_single++;
#endif
            break;

        case IO_EULERA:
#ifdef READ_EULER
            for(n = 0; n < pc; n++)
                SPHP(offset + n).EulerA = READREAL(cp);
#endif
            break;

        case IO_EULERB:
#ifdef READ_EULER
            for(n = 0; n < pc; n++)
                SPHP(offset + n).EulerB = READREAL(cp);
#endif
            break;

        case IO_VECTA:
#ifdef READ_VECTA
            for(n = 0; n < pc; n++)
                for(k = 0; k < 3; k++)
                {
                    SPHP(offset + n).APred[k] = READREAL(cp);
                    SPHP(offset + n).SmoothA[k] = SPHP(offset + n).APred[k];
                    SPHP(offset + n).A[k] = SPHP(offset + n).APred[k];
                }
#endif
            break;

        case IO_CHEM:               /* Chemical abundances */
#ifdef CHEMCOOL
            for(n = 0; n < pc; n++)
                for(k = 0; k < TRAC_NUM; k++)
                    SPHP(offset + n).TracAbund[k] = READREAL(cp);
#endif
            break;


            /* the other input fields (if present) are not needed to define the 
               initial conditions of the code */

        case IO_SFR:
        case IO_CLDX:
        case IO_ZSMOOTH:
        case IO_POT:
        case IO_ACCEL:
        case IO_DTENTR:
        case IO_STRESSDIAG:
        case IO_STRESSOFFDIAG:
        case IO_STRESSBULK:
        case IO_SHEARCOEFF:
        case IO_TSTP:
        case IO_DBDT:
        case IO_DIVB:
        case IO_ABVC:
        case IO_COOLRATE:
        case IO_CONDRATE:
        case IO_BSMTH:
        case IO_DENN:
        case IO_MACH:
        case IO_DTENERGY:
        case IO_PRESHOCK_DENSITY:
        case IO_PRESHOCK_ENERGY:
        case IO_PRESHOCK_XCR:
        case IO_DENSITY_JUMP:
        case IO_ENERGY_JUMP:
        case IO_CRINJECT:
        case IO_AMDC:
        case IO_PHI:
        case IO_XPHI:
        case IO_GRADPHI:
        case IO_TIDALTENSORPS:
        case IO_ROTB:
        case IO_SROTB:
        case IO_FLOW_DETERMINANT:
        case IO_STREAM_DENSITY:
        case IO_PHASE_SPACE_DETERMINANT:
        case IO_ANNIHILATION_RADIATION:
        case IO_EOSTEMP:
        case IO_PRESSURE:
        case IO_PRESHOCK_CSND:
        case IO_EDDINGTON_TENSOR:
        case IO_SHELL_INFO:
        case IO_LAST_CAUSTIC:
        case IO_VALPHA:
        case IO_HTEMP:
            break;

        case IO_LASTENTRY:
            endrun(220);
            break;
    }
}



/*! This function reads a snapshot file and distributes the data it contains
 *  to tasks 'readTask' to 'lastTask'.
 */
void read_file(char *fname, int readTask, int lastTask)
{
    size_t blockmaxlen;
    int i, n_in_file, n_for_this_task[6], ntask, pc, offset = 0, task;
    int blksize1, blksize2;
    MPI_Status status;
    FILE *fd = 0;
    int nall, nread;
    int type, bnr;
    char label[4], buf[500];
    int nstart, bytes_per_blockelement, npart, nextblock, typelist[6];
    enum iofields blocknr;
    size_t bytes;

#ifdef HAVE_HDF5
    int rank, pcsum;
    hid_t hdf5_file = 0, hdf5_grp[6], hdf5_dataspace_in_file;
    hid_t hdf5_datatype = 0, hdf5_dataspace_in_memory, hdf5_dataset;
    hsize_t dims[2], count[2], start[2];
#endif

#if defined(COSMIC_RAYS) && (!defined(CR_IC))
    int CRpop;
#endif

#define SKIP  {my_fread(&blksize1,sizeof(int),1,fd);}
#define SKIP2  {my_fread(&blksize2,sizeof(int),1,fd);}

    if(ThisTask == readTask)
    {
        if(All.ICFormat == 1 || All.ICFormat == 2)
        {
            if(!(fd = fopen(fname, "r")))
            {
                printf("can't open file `%s' for reading initial conditions.\n", fname);
                endrun(123);
            }

            if(All.ICFormat == 2)
            {
                SKIP;
#ifdef AUTO_SWAP_ENDIAN_READIC
                swap_file = blksize1;
#endif
                my_fread(&label, sizeof(char), 4, fd);
                my_fread(&nextblock, sizeof(int), 1, fd);
#ifdef AUTO_SWAP_ENDIAN_READIC
                swap_Nbyte((char *) &nextblock, 1, 4);
#endif
                printf("Reading header => '%c%c%c%c' (%d byte)\n", label[0], label[1], label[2], label[3],
                        nextblock);
                SKIP2;
            }

            SKIP;
#ifdef AUTO_SWAP_ENDIAN_READIC
            if(All.ICFormat == 1)
            {
                if(blksize1 != 256)
                    swap_file = 1;
            }
#endif
            my_fread(&header, sizeof(header), 1, fd);
            SKIP2;
#ifdef AUTO_SWAP_ENDIAN_READIC
            swap_Nbyte((char *) &blksize1, 1, 4);
            swap_Nbyte((char *) &blksize2, 1, 4);
#endif

            if(blksize1 != 256 || blksize2 != 256)
            {
                printf("incorrect header format\n");
                fflush(stdout);
                endrun(890);
                /* Probable error is wrong size of fill[] in header file. Needs to be 256 bytes in total. */
            }
#ifdef AUTO_SWAP_ENDIAN_READIC
            swap_header();
#endif
        }


#ifdef HAVE_HDF5
        if(All.ICFormat == 3)
        {
            read_header_attributes_in_hdf5(fname);

            hdf5_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

            for(type = 0; type < 6; type++)
            {
                if(header.npart[type] > 0)
                {
                    sprintf(buf, "/PartType%d", type);
                    hdf5_grp[type] = H5Gopen(hdf5_file, buf);
                }
            }
        }
#endif

        for(task = readTask + 1; task <= lastTask; task++)
        {
            MPI_Ssend(&header, sizeof(header), MPI_BYTE, task, TAG_HEADER, MPI_COMM_WORLD);
#ifdef AUTO_SWAP_ENDIAN_READIC
            MPI_Ssend(&swap_file, sizeof(int), MPI_INT, task, TAG_SWAP, MPI_COMM_WORLD);
#endif
        }

    }
    else
    {
        MPI_Recv(&header, sizeof(header), MPI_BYTE, readTask, TAG_HEADER, MPI_COMM_WORLD, &status);
#ifdef AUTO_SWAP_ENDIAN_READIC
        MPI_Recv(&swap_file, sizeof(int), MPI_INT, readTask, TAG_SWAP, MPI_COMM_WORLD, &status);
#endif
    }

    if(All.TotNumPart == 0)
    {
        if(header.num_files <= 1)
            for(i = 0; i < 6; i++)
            {
                header.npartTotal[i] = header.npart[i];
#ifdef SFR
                header.npartTotalHighWord[i] = 0;
#endif
            }

        All.TotN_sph = header.npartTotal[0] + (((long long) header.npartTotalHighWord[0]) << 32);
        All.TotN_bh = header.npartTotal[5] + (((long long) header.npartTotalHighWord[5]) << 32);

        for(i = 0, All.TotNumPart = 0; i < 6; i++)
        {
            All.TotNumPart += header.npartTotal[i];
            All.TotNumPart += (((long long) header.npartTotalHighWord[i]) << 32);
        }

#ifdef NEUTRINOS
        All.TotNumNeutrinos = header.npartTotal[2] + (((long long) header.npartTotalHighWord[2]) << 32);
#endif

        for(i = 0; i < 6; i++)
            All.MassTable[i] = header.mass[i];

        All.MaxPart = (int) (All.PartAllocFactor * (All.TotNumPart / NTask));	/* sets the maximum number of particles that may */
        All.MaxPartSph = (int) (All.PartAllocFactor * (All.TotN_sph / NTask));	/* sets the maximum number of particles that may 
                                                                                   reside on a processor */
        All.MaxPartBh = (int) (0.1 * All.PartAllocFactor * (All.TotN_sph / NTask));	/* sets the maximum number of particles that may 
                                                                                   reside on a processor */
#ifdef INHOMOG_GASDISTR_HINT
        All.MaxPartSph = All.MaxPart;
#endif

        allocate_memory();

        if(!(CommBuffer = mymalloc("CommBuffer", bytes = All.BufferSize * 1024 * 1024)))
        {
            printf("failed to allocate memory for `CommBuffer' (%g MB).\n", bytes / (1024.0 * 1024.0));
            endrun(2);
        }

        if(RestartFlag >= 2)
            All.Time = All.TimeBegin = header.time;

#ifdef END_TIME_DYN_BASED
        double rho, t_ff;

        rho = All.EndTimeDens * PROTONMASS / HYDROGEN_MASSFRAC; 
        t_ff = sqrt(3.0 * M_PI / 32.0 / GRAVITY / rho); 

        if(All.ComovingIntegrationOn)
            All.TimeMax = pow(3.0 / 2.0 * HUBBLE * All.HubbleParam * sqrt(All.Omega0) * 100.0 * t_ff + pow(All.Time, 3.0 / 2.0), 2.0 / 3.0);
        else
            All.TimeMax = All.TimeBegin + 100.0 * t_ff / All.UnitTime_in_s;	    

        if(ThisTask == 0)
        {
            printf("All.Time = %g\n", All.Time);
            printf("All.TimeMax = %g\n", All.TimeMax);
        }
#endif

    }

    if(ThisTask == readTask)
    {
        for(i = 0, n_in_file = 0; i < 6; i++)
            n_in_file += header.npart[i];

        printf("\nreading file `%s' on task=%d (contains %d particles.)\n"
                "distributing this file to tasks %d-%d\n"
                "Type 0 (gas):   %8d  (tot=%6d%09d) masstab=%g\n"
                "Type 1 (halo):  %8d  (tot=%6d%09d) masstab=%g\n"
                "Type 2 (disk):  %8d  (tot=%6d%09d) masstab=%g\n"
                "Type 3 (bulge): %8d  (tot=%6d%09d) masstab=%g\n"
                "Type 4 (stars): %8d  (tot=%6d%09d) masstab=%g\n"
                "Type 5 (bndry): %8d  (tot=%6d%09d) masstab=%g\n\n", fname, ThisTask, n_in_file, readTask,
                lastTask, header.npart[0], (int) (header.npartTotal[0] / 1000000000),
                (int) (header.npartTotal[0] % 1000000000), All.MassTable[0], header.npart[1],
                (int) (header.npartTotal[1] / 1000000000), (int) (header.npartTotal[1] % 1000000000),
                All.MassTable[1], header.npart[2], (int) (header.npartTotal[2] / 1000000000),
                (int) (header.npartTotal[2] % 1000000000), All.MassTable[2], header.npart[3],
                (int) (header.npartTotal[3] / 1000000000), (int) (header.npartTotal[3] % 1000000000),
                All.MassTable[3], header.npart[4], (int) (header.npartTotal[4] / 1000000000),
                (int) (header.npartTotal[4] % 1000000000), All.MassTable[4], header.npart[5],
                (int) (header.npartTotal[5] / 1000000000), (int) (header.npartTotal[5] % 1000000000),
                All.MassTable[5]);
        fflush(stdout);
    }


    ntask = lastTask - readTask + 1;


    /* to collect the gas particles all at the beginning (in case several
       snapshot files are read on the current CPU) we move the collisionless
       particles such that a gap of the right size is created */

    for(type = 0, nall = 0; type < 6; type++)
    {
        n_in_file = header.npart[type];

        n_for_this_task[type] = n_in_file / ntask;
        if((ThisTask - readTask) < (n_in_file % ntask))
            n_for_this_task[type]++;
        nall += n_for_this_task[type];
     }
     if(N_sph + n_for_this_task[0] > All.MaxPartSph)
     {
        printf("Not enough space on task=%d for SPH particles (space for %d, need at least %d)\n",
                ThisTask, All.MaxPartSph, N_sph + n_for_this_task[0]);
        fflush(stdout);
        endrun(172);
     }
     if(N_bh + n_for_this_task[5] > All.MaxPartBh)
     {
        printf("Not enough space on task=%d for BH particles (space for %d, need at least %d)\n",
                ThisTask, All.MaxPartSph, N_sph + n_for_this_task[5]);
        fflush(stdout);
        endrun(172);
     }

    if(NumPart + nall > All.MaxPart)
    {
        printf("Not enough space on task=%d (space for %d, need at least %d)\n",
                ThisTask, All.MaxPart, NumPart + nall);
        fflush(stdout);
        endrun(173);
    }

    memmove(&P[N_sph + nall], &P[N_sph], (NumPart - N_sph) * sizeof(struct particle_data));
    nstart = N_sph;


    for(bnr = 0; bnr < 1000; bnr++)
    {
        blocknr = (enum iofields) bnr;

        if(blocknr == IO_LASTENTRY)
            break;

#ifdef NO_UTHERM_IN_IC_FILE
        if(RestartFlag == 0 && blocknr == IO_U)
            continue;
#endif

        if(RestartFlag == 5 && blocknr > IO_MASS)	/* if we only do power spectra, we don't need to read other blocks beyond the mass */
            continue;


        if(blockpresent(blocknr))
        {
#ifdef CR_IC
            if(RestartFlag == 0 && ((blocknr > IO_CR_Q0 && blocknr != IO_BFLD)
                        || (blocknr >= IO_RHO && blocknr <= IO_ACCEL)))
#else
#ifdef EOS_DEGENERATE
                if(RestartFlag == 0 && (blocknr > IO_U && blocknr != IO_EOSXNUC))
#else
#ifndef CHEMISTRY
                    if(RestartFlag == 0 && blocknr > IO_U && blocknr != IO_BFLD
#ifdef READ_HSML
                            && blocknr != IO_HSML
#endif
#ifdef READ_VECTA
                            && blocknr != IO_VECTA
#endif
#ifdef READ_EULER
                            && blocknr != IO_EULERB && blocknr != IO_EULERA
#endif
                      )
#else
                        if(RestartFlag == 0 && blocknr > IO_HM)
#endif
#endif
#endif
#ifdef DISTORTIONTENSORPS
#if !defined(COMOVING_DISTORTION) || defined(COMOVING_READIC)
                            if(RestartFlag == 0 && (blocknr > IO_U && blocknr != IO_SHEET_ORIENTATION))
                                if(RestartFlag == 0 && (blocknr > IO_U && blocknr != IO_INIT_DENSITY))
                                    if(RestartFlag == 0 && (blocknr > IO_U && blocknr != IO_CAUSTIC_COUNTER))
#ifdef DISTORTION_READALL
                                        if(RestartFlag == 0 && (blocknr > IO_U && blocknr != IO_DISTORTIONTENSORPS))
#endif
#endif
#endif
                                            continue;	/* ignore all other blocks in initial conditions */


#ifdef SUBFIND_RESHUFFLE_AND_POTENTIAL
            if(blocknr == IO_POT)
                continue;
#endif


#ifdef BINISET
            if(RestartFlag == 0 && blocknr == IO_BFLD)
                continue;
#endif

#if defined (UM_CHEMISTRY) && defined (UM_CHEMISTRY_INISET)
            if(RestartFlag == 0 && blocknr == IO_NE)
                continue;
            if(RestartFlag == 0 && blocknr == IO_NH)
                continue;
            if(RestartFlag == 0 && blocknr == IO_HII)
                continue;
            if(RestartFlag == 0 && blocknr == IO_HM)
                continue;
            if(RestartFlag == 0 && blocknr == IO_HeI)
                continue;
            if(RestartFlag == 0 && blocknr == IO_HeII)
                continue;
            if(RestartFlag == 0 && blocknr == IO_HeIII)
                continue;
            if(RestartFlag == 0 && blocknr == IO_H2I)
                continue;
            if(RestartFlag == 0 && blocknr == IO_H2II)
                continue;
            if(RestartFlag == 0 && blocknr == IO_HD)
                continue;
            if(RestartFlag == 0 && blocknr == IO_DI)
                continue;
            if(RestartFlag == 0 && blocknr == IO_DII)
                continue;
            if(RestartFlag == 0 && blocknr == IO_HeHII)
                continue;
#endif

            if(ThisTask == readTask)
            {
                get_dataset_name(blocknr, buf);
                printf("reading block %d (%s)...\n", blocknr, buf);
                fflush(stdout);
            }

            bytes_per_blockelement = get_bytes_per_blockelement(blocknr, 
                    header.flag_doubleprecision);
            blockmaxlen = (size_t) ((All.BufferSize * 1024 * 1024) / bytes_per_blockelement);

            npart = get_particles_in_block(blocknr, &typelist[0]);

            if(npart > 0)
            {
                if(blocknr != IO_DMHSML && blocknr != IO_DMDENSITY && blocknr != IO_DMVELDISP
                        && blocknr != IO_DMHSML_V && blocknr != IO_DMDENSITY_V)
                    if(ThisTask == readTask)
                    {
                        if(All.ICFormat == 2)
                        {
                            get_Tab_IO_Label(blocknr, label);
                            find_block(label, fd);
                        }

                        if(All.ICFormat == 1 || All.ICFormat == 2) {
                            SKIP;
#ifdef AUTO_SWAP_ENDIAN_READIC
                            swap_Nbyte((char *) &blksize1, 1, 4);
#endif
                            if(blocknr == IO_ID || blocknr == IO_BHPROGS) {
                                if (bytes_per_blockelement != blksize1/ npart) {
                                    printf("ID type in ic is uint32, will convert to uint64");
                                }
                                bytes_per_blockelement = blksize1 / npart;
                            }
                        }
                        for(task = readTask + 1; task <= lastTask; task++) {
                            MPI_Ssend(&bytes_per_blockelement, sizeof(bytes_per_blockelement), 
                                        MPI_BYTE, task, TAG_BYTES_PB, MPI_COMM_WORLD);
                        }
                    } else {
                        /* in case a different bytes-per_blockelement is probed by readTask */
                            MPI_Recv(&bytes_per_blockelement, sizeof(bytes_per_blockelement), 
                                    MPI_BYTE, readTask, TAG_BYTES_PB, MPI_COMM_WORLD, &status);
                    }
                    for(type = 0, offset = 0, nread = 0; type < 6; type++)
                    {
#ifdef HAVE_HDF5
                        pcsum = 0;
#endif
                        if(typelist[type] == 0)
                        {
                            offset += n_for_this_task[type];
                        }
                        else
                        {
                            for(task = readTask; task <= lastTask; task++)
                            {
                                n_in_file = header.npart[type];
                                int toread = n_in_file / ntask;
                                if((task - readTask) < (n_in_file % ntask))
                                    toread++;
                                /* to read */
                                do
                                {
                                    pc = toread;

                                    if(pc > blockmaxlen)
                                        pc = blockmaxlen;

                                    if(ThisTask == readTask)
                                    {
                                        if(All.ICFormat == 1 || All.ICFormat == 2)
                                        {
                                            if(blocknr != IO_DMHSML && blocknr != IO_DMDENSITY
                                                    && blocknr != IO_DMVELDISP && blocknr != IO_DMHSML_V
                                                    && blocknr != IO_DMDENSITY_V)
                                            {
                                                my_fread(CommBuffer, bytes_per_blockelement, pc, fd);
                                                nread += pc;
                                            }
                                            else
                                            {
#if defined(SUBFIND_RESHUFFLE_CATALOGUE) && !defined(SUBFIND_DENSITY_AND_POTENTIAL)
                                                read_hsml_files(CommBuffer, pc, blocknr,
                                                        NumPartPerFile[FileNr] + nread);
#endif
                                                nread += pc;
                                            }
                                        }

#ifdef HAVE_HDF5
                                        if(All.ICFormat == 3 && pc > 0)
                                        {
                                            get_dataset_name(blocknr, buf);
                                            hdf5_dataset = H5Dopen(hdf5_grp[type], buf);

                                            dims[0] = header.npart[type];
                                            dims[1] = get_values_per_blockelement(blocknr);
                                            if(dims[1] == 1)
                                                rank = 1;
                                            else
                                                rank = 2;

                                            hdf5_dataspace_in_file = H5Screate_simple(rank, dims, NULL);

                                            dims[0] = pc;
                                            hdf5_dataspace_in_memory = H5Screate_simple(rank, dims, NULL);

                                            start[0] = pcsum;
                                            start[1] = 0;

                                            count[0] = pc;
                                            count[1] = get_values_per_blockelement(blocknr);
                                            pcsum += pc;

                                            H5Sselect_hyperslab(hdf5_dataspace_in_file, H5S_SELECT_SET,
                                                    start, NULL, count, NULL);

                                            hdf5_datatype = get_hdf5_datatype(blocknr,
                                                    header.flag_doubleprecision);

                                            H5Dread(hdf5_dataset, hdf5_datatype, hdf5_dataspace_in_memory,
                                                    hdf5_dataspace_in_file, H5P_DEFAULT, CommBuffer);

                                            H5Tclose(hdf5_datatype);
                                            H5Sclose(hdf5_dataspace_in_memory);
                                            H5Sclose(hdf5_dataspace_in_file);
                                            H5Dclose(hdf5_dataset);
                                        }
#endif
                                    }

                                    if(ThisTask == readTask && task != readTask && pc > 0)
                                        MPI_Ssend(CommBuffer, bytes_per_blockelement * pc, MPI_BYTE, task,
                                                TAG_PDATA, MPI_COMM_WORLD);

                                    if(ThisTask != readTask && task == ThisTask && pc > 0)
                                        MPI_Recv(CommBuffer, bytes_per_blockelement * pc, MPI_BYTE, readTask,
                                                TAG_PDATA, MPI_COMM_WORLD, &status);

                                    if(ThisTask == task)
                                    {
                                        empty_read_buffer(blocknr, bytes_per_blockelement, nstart + offset, pc, type);

                                        offset += pc;
                                    }

                                    toread -= pc;
                                }
                                while(toread > 0);
                            }
                        }
                    }

                if(ThisTask == readTask)
                {
                    if(blocknr != IO_DMHSML && blocknr != IO_DMDENSITY && blocknr != IO_DMVELDISP
                            && blocknr != IO_DMHSML_V && blocknr != IO_DMDENSITY_V)
                        if(All.ICFormat == 1 || All.ICFormat == 2)
                        {
                            SKIP2;

#ifdef AUTO_SWAP_ENDIAN_READIC
                            swap_Nbyte((char *) &blksize2, 1, 4);
#endif
                            if(blksize1 != blksize2)
                            {
                                printf("incorrect block-sizes detected!\n");
                                printf("Task=%d   blocknr=%d  blksize1=%d  blksize2=%d\n", ThisTask, blocknr,
                                        blksize1, blksize2);
                                if(blocknr == IO_ID)
                                {
                                    printf
                                        ("Possible mismatch of 32bit and 64bit ID's in IC file and GADGET compilation !\n");
                                }
                                fflush(stdout);
                                endrun(1889);
                            }
                        }
                }
            }
        }
    }

    if(ThisTask == readTask)
    {
        if(All.ICFormat == 1 || All.ICFormat == 2)
            fclose(fd);
#ifdef HAVE_HDF5
        if(All.ICFormat == 3)
        {
            for(type = 5; type >= 0; type--)
                if(header.npart[type] > 0)
                    H5Gclose(hdf5_grp[type]);
            H5Fclose(hdf5_file);
        }
#endif
    }

#if defined(COSMIC_RAYS) && (!defined(CR_IC))
    for(i = 0; i < n_for_this_task; i++)
    {
        if(P[i].Type != 0)
        {
            break;
        }

        for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
        {
            SPHP(i).CR_C0[CRpop] = 0.0;
            SPHP(i).CR_q0[CRpop] = 1.0e10;
        }
    }
#endif

}



/*! This function determines on how many files a given snapshot is distributed.
*/
int find_files(char *fname)
{
    FILE *fd;
    char buf[200], buf1[200];
    int dummy;

    sprintf(buf, "%s.%d", fname, 0);
    sprintf(buf1, "%s", fname);

    if(All.ICFormat == 3)
    {
        sprintf(buf, "%s.%d.hdf5", fname, 0);
        sprintf(buf1, "%s.hdf5", fname);
    }

#ifndef  HAVE_HDF5
    if(All.ICFormat == 3)
    {
        if(ThisTask == 0)
            printf("Code wasn't compiled with HDF5 support enabled!\n");
        endrun(0);
    }
#endif

    header.num_files = 0;

    if(ThisTask == 0)
    {
        if((fd = fopen(buf, "r")))
        {
            if(All.ICFormat == 1 || All.ICFormat == 2)
            {
                if(All.ICFormat == 2)
                {
                    my_fread(&dummy, sizeof(dummy), 1, fd);
#ifdef AUTO_SWAP_ENDIAN_READIC
                    swap_file = dummy;
#endif
                    my_fread(&dummy, sizeof(dummy), 1, fd);
                    my_fread(&dummy, sizeof(dummy), 1, fd);
                    my_fread(&dummy, sizeof(dummy), 1, fd);
                }

                my_fread(&dummy, sizeof(dummy), 1, fd);
#ifdef AUTO_SWAP_ENDIAN_READIC
                if(All.ICFormat == 1)
                {
                    if(dummy == 256)
                        swap_file = 8;
                    else
                        swap_file = 1;
                }
#endif
                my_fread(&header, sizeof(header), 1, fd);
#ifdef AUTO_SWAP_ENDIAN_READIC
                swap_header();
#endif
                my_fread(&dummy, sizeof(dummy), 1, fd);
            }
            fclose(fd);

#ifdef HAVE_HDF5
            if(All.ICFormat == 3)
                read_header_attributes_in_hdf5(buf);
#endif
        }
    }

#ifdef AUTO_SWAP_ENDIAN_READIC
    MPI_Bcast(&swap_file, sizeof(int), MPI_INT, 0, MPI_COMM_WORLD);
#endif
    MPI_Bcast(&header, sizeof(header), MPI_BYTE, 0, MPI_COMM_WORLD);

    if(header.num_files > 0)
        return header.num_files;

    if(ThisTask == 0)
    {
        if((fd = fopen(buf1, "r")))
        {
            if(All.ICFormat == 1 || All.ICFormat == 2)
            {
                if(All.ICFormat == 2)
                {
                    my_fread(&dummy, sizeof(dummy), 1, fd);
#ifdef AUTO_SWAP_ENDIAN_READIC
                    swap_file = dummy;
#endif
                    my_fread(&dummy, sizeof(dummy), 1, fd);
                    my_fread(&dummy, sizeof(dummy), 1, fd);
                    my_fread(&dummy, sizeof(dummy), 1, fd);
                }

                my_fread(&dummy, sizeof(dummy), 1, fd);
#ifdef AUTO_SWAP_ENDIAN_READIC
                if(All.ICFormat == 1)
                {
                    if(dummy == 256)
                        swap_file = 8;
                    else
                        swap_file = 1;
                }
#endif
                my_fread(&header, sizeof(header), 1, fd);
#ifdef AUTO_SWAP_ENDIAN_READIC
                swap_header();
#endif
                my_fread(&dummy, sizeof(dummy), 1, fd);
            }
            fclose(fd);

#ifdef HAVE_HDF5
            if(All.ICFormat == 3)
                read_header_attributes_in_hdf5(buf1);
#endif

            header.num_files = 1;
        }
    }

#ifdef AUTO_SWAP_ENDIAN_READIC
    MPI_Bcast(&swap_file, sizeof(int), MPI_INT, 0, MPI_COMM_WORLD);
#endif
    MPI_Bcast(&header, sizeof(header), MPI_BYTE, 0, MPI_COMM_WORLD);

    if(header.num_files > 0)
        return header.num_files;

    if(ThisTask == 0)
    {
        printf("\nCan't find initial conditions file.");
        printf("neither as '%s'\nnor as '%s'\n", buf, buf1);
        fflush(stdout);
    }

    endrun(0);
    return 0;
}

#if defined(SAVE_HSML_IN_IC_ORDER) || defined(SUBFIND_RESHUFFLE_CATALOGUE)
void get_particle_numbers(char *fname, int num_files)
{
    char buf[1000];
    int blksize1, blksize2;
    char label[4];
    int nextblock;
    int i, j;

    printf("num_files=%d\n", num_files);

    for(i = 0; i < num_files; i++)
    {
        if(num_files > 1)
        {
            sprintf(buf, "%s.%d", fname, i);
            if(All.ICFormat == 3)
                sprintf(buf, "%s.%d.hdf5", fname, i);
        }
        else
        {
            sprintf(buf, "%s", fname);
            if(All.ICFormat == 3)
                sprintf(buf, "%s.hdf5", fname);
        }

#define SKIP  {my_fread(&blksize1,sizeof(int),1,fd);}
#define SKIP2  {my_fread(&blksize2,sizeof(int),1,fd);}

        if(All.ICFormat == 1 || All.ICFormat == 2)
        {
            FILE *fd;

            if(!(fd = fopen(buf, "r")))
            {
                printf("can't open file `%s' for reading initial conditions.\n", buf);
                endrun(1239);
            }

            if(All.ICFormat == 2)
            {
                SKIP;
#ifdef AUTO_SWAP_ENDIAN_READIC
                swap_file = blksize1;
#endif
                my_fread(&label, sizeof(char), 4, fd);
                my_fread(&nextblock, sizeof(int), 1, fd);
#ifdef AUTO_SWAP_ENDIAN_READIC
                swap_Nbyte((char *) &nextblock, 1, 4);
#endif
                SKIP2;
            }

            SKIP;
#ifdef AUTO_SWAP_ENDIAN_READIC
            if(All.ICFormat == 1)
            {
                if(blksize1 != 256)
                    swap_file = 1;
            }
#endif
            my_fread(&header, sizeof(header), 1, fd);
            SKIP2;
#ifdef AUTO_SWAP_ENDIAN_READIC
            swap_Nbyte((char *) &blksize1, 1, 4);
            swap_Nbyte((char *) &blksize2, 1, 4);
#endif

            if(blksize1 != 256 || blksize2 != 256)
            {
                printf("incorrect header format\n");
                fflush(stdout);
                endrun(890);
            }
#ifdef AUTO_SWAP_ENDIAN_READIC
            swap_header();
#endif
            fclose(fd);
        }

#ifdef HAVE_HDF5
        if(All.ICFormat == 3)
        {
            read_header_attributes_in_hdf5(buf);
        }
#endif

        NumPartPerFile[i] = 0;

        for(j = 0; j < 6; j++)
        {
#if defined(SUBFIND_RESHUFFLE_CATALOGUE)
            if(((1 << j) & (FOF_PRIMARY_LINK_TYPES)))
#endif
                NumPartPerFile[i] += header.npart[j];
        }

        printf("File=%4d:  NumPart= %d\n", i, (int) (NumPartPerFile[i]));
    }


    long long n, sum;

    for(i = 0, sum = 0; i < num_files; i++)
    {
        n = NumPartPerFile[i];

        NumPartPerFile[i] = sum;

        sum += n;
    }
}
#endif




/*! This function assigns a certain number of files to processors, such that
 *  each processor is exactly assigned to one file, and the number of cpus per
 *  file is as homogenous as possible. The number of files may at most be
 *  equal to the number of processors.
 */
void distribute_file(int nfiles, int firstfile, int firsttask, int lasttask, int *filenr, int *master,
        int *last)
{
    int ntask, filesleft, filesright, tasksleft, tasksright;

    if(nfiles > 1)
    {
        ntask = lasttask - firsttask + 1;

        filesleft = (int) ((((double) (ntask / 2)) / ntask) * nfiles);
        if(filesleft <= 0)
            filesleft = 1;
        if(filesleft >= nfiles)
            filesleft = nfiles - 1;

        filesright = nfiles - filesleft;

        tasksleft = ntask / 2;
        tasksright = ntask - tasksleft;

        distribute_file(filesleft, firstfile, firsttask, firsttask + tasksleft - 1, filenr, master, last);
        distribute_file(filesright, firstfile + filesleft, firsttask + tasksleft, lasttask, filenr, master,
                last);
    }
    else
    {
        if(ThisTask >= firsttask && ThisTask <= lasttask)
        {
            *filenr = firstfile;
            *master = firsttask;
            *last = lasttask;
        }
    }
}



#ifdef HAVE_HDF5
void read_header_attributes_in_hdf5(char *fname)
{
    hid_t hdf5_file, hdf5_headergrp, hdf5_attribute;
    int i;

    hdf5_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    hdf5_headergrp = H5Gopen(hdf5_file, "/Header");

    hdf5_attribute = H5Aopen_name(hdf5_headergrp, "NumPart_ThisFile");
    H5Aread(hdf5_attribute, H5T_NATIVE_INT, header.npart);
    H5Aclose(hdf5_attribute);

    hdf5_attribute = H5Aopen_name(hdf5_headergrp, "NumPart_Total");
    H5Aread(hdf5_attribute, H5T_NATIVE_UINT, header.npartTotal);
    H5Aclose(hdf5_attribute);

    hdf5_attribute = H5Aopen_name(hdf5_headergrp, "NumPart_Total_HighWord");
    H5Aread(hdf5_attribute, H5T_NATIVE_UINT, header.npartTotalHighWord);
    H5Aclose(hdf5_attribute);

    hdf5_attribute = H5Aopen_name(hdf5_headergrp, "MassTable");
    H5Aread(hdf5_attribute, H5T_NATIVE_DOUBLE, header.mass);
    H5Aclose(hdf5_attribute);

    hdf5_attribute = H5Aopen_name(hdf5_headergrp, "Time");
    H5Aread(hdf5_attribute, H5T_NATIVE_DOUBLE, &header.time);
    H5Aclose(hdf5_attribute);

    hdf5_attribute = H5Aopen_name(hdf5_headergrp, "NumFilesPerSnapshot");
    H5Aread(hdf5_attribute, H5T_NATIVE_INT, &header.num_files);
    H5Aclose(hdf5_attribute);

    hdf5_attribute = H5Aopen_name(hdf5_headergrp, "Flag_IC_Info");
    H5Aread(hdf5_attribute, H5T_NATIVE_INT, &header.flag_ic_info);
    H5Aclose(hdf5_attribute);

    hdf5_attribute = H5Aopen_name(hdf5_headergrp, "Flag_DoublePrecision");
    H5Aread(hdf5_attribute, H5T_NATIVE_INT, &header.flag_doubleprecision);
    H5Aclose(hdf5_attribute);

    H5Gclose(hdf5_headergrp);
    H5Fclose(hdf5_file);
}
#endif





#ifdef AUTO_SWAP_ENDIAN_READIC
/*-----------------------------------------------------------------------------*/
/*---------------------- Routine to swap ENDIAN -------------------------------*/
/*-------- char *data:    Pointer to the data ---------------------------------*/
/*-------- int n:         Number of elements to swap --------------------------*/
/*-------- int m:         Size of single element to swap ----------------------*/
/*--------                int,float = 4 ---------------------------------------*/
/*--------                double    = 8 ---------------------------------------*/
/*-----------------------------------------------------------------------------*/
void swap_Nbyte(char *data, int n, int m)
{
    int i, j;
    char old_data[16];

    if(swap_file != 8)
    {
        for(j = 0; j < n; j++)
        {
            memcpy(&old_data[0], &data[j * m], m);
            for(i = 0; i < m; i++)
            {
                data[j * m + i] = old_data[m - i - 1];
            }
        }
    }
}

/*------------------------------------------------------------------*/
/*----------- procedure to swap header if needed -------------------*/
/*------------------------------------------------------------------*/

void swap_header()
{
    swap_Nbyte((char *) &header.npart, 6, 4);
    swap_Nbyte((char *) &header.mass, 6, 8);
    swap_Nbyte((char *) &header.time, 1, 8);
    swap_Nbyte((char *) &header.redshift, 1, 8);
    swap_Nbyte((char *) &header.flag_sfr, 1, 4);
    swap_Nbyte((char *) &header.flag_feedback, 1, 4);
    swap_Nbyte((char *) &header.npartTotal, 6, 4);
    swap_Nbyte((char *) &header.flag_cooling, 1, 4);
    swap_Nbyte((char *) &header.num_files, 1, 4);
    swap_Nbyte((char *) &header.BoxSize, 1, 8);
    swap_Nbyte((char *) &header.Omega0, 1, 8);
    swap_Nbyte((char *) &header.OmegaLambda, 1, 8);
    swap_Nbyte((char *) &header.HubbleParam, 1, 8);
    swap_Nbyte((char *) &header.flag_stellarage, 1, 4);
    swap_Nbyte((char *) &header.flag_metals, 1, 4);
    swap_Nbyte((char *) &header.npartTotalHighWord, 6, 4);
    swap_Nbyte((char *) &header.flag_entropy_instead_u, 1, 4);
    swap_Nbyte((char *) &header.flag_doubleprecision, 1, 4);
#ifdef COSMIC_RAYS
    swap_Nbyte((char *) &header.SpectralIndex_CR_Pop, NUMCRPOP, 8);
#endif
}

#endif

/*---------------------- Routine find a block in a snapfile -------------------*/
/*-------- FILE *fd:      File handle -----------------------------------------*/
/*-------- char *label:   4 byte identifyer for block -------------------------*/
/*-------- returns length of block found, -------------------------------------*/
/*-------- the file fd points to starting point of block ----------------------*/
/*-----------------------------------------------------------------------------*/
void find_block(char *label, FILE * fd)
{
    int blocksize = 0, blksize;
    char blocklabel[5] = { "    " };

#define FBSKIP  {my_fread(&blksize,sizeof(int),1,fd);}

    rewind(fd);

    while(!feof(fd) && blocksize == 0)
    {
        FBSKIP;
#ifdef AUTO_SWAP_ENDIAN_READIC
        swap_file = blksize;
        swap_Nbyte((char *) &blksize, 1, 4);
#endif
        if(blksize != 8)
        {
            printf("Incorrect Format (blksize=%d)!\n", blksize);
            exit(1891);
        }
        else
        {
            my_fread(blocklabel, 4 * sizeof(char), 1, fd);
            my_fread(&blocksize, sizeof(int), 1, fd);
#ifdef AUTO_SWAP_ENDIAN_READIC
            swap_Nbyte((char *) &blocksize, 1, 4);
#endif
            /*
               printf("Searching <%c%c%c%c>, found Block <%s> with %d bytes\n",
               label[0],label[1],label[2],label[3],blocklabel,blocksize);
               */
            FBSKIP;
            if(strncmp(label, blocklabel, 4) != 0)
            {
                fseek(fd, blocksize, 1);
                blocksize = 0;
            }
        }
    }
    if(feof(fd))
    {
        printf("Block '%c%c%c%c' not found !\n", label[0], label[1], label[2], label[3]);
        fflush(stdout);
        endrun(1890);
    }
}
