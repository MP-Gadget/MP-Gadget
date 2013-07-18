#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/file.h>
#include <unistd.h>
#include <gsl/gsl_rng.h>

#include "allvars.h"
#include "proto.h"
#include "domain.h"

static FILE *fd;

static void in(int *x, int modus);
static void byten(void *x, int n, int modus);


/* This function reads or writes the restart files.
 * Each processor writes its own restart file, with the
 * I/O being done in parallel. To avoid congestion of the disks
 * you can tell the program to restrict the number of files
 * that are simultaneously written to NumFilesWrittenInParallel.
 *
 * If modus>0  the restart()-routine reads, 
 * if modus==0 it writes a restart file. 
 */
void restart(int modus)
{
    char buf[200], buf_bak[200], buf_mv[500];
    double save_PartAllocFactor;
    int i, nprocgroup, masterTask, groupTask, old_MaxPart, old_MaxNodes;
    struct global_data_all_processes all_task0;

#ifdef LT_STELLAREVOLUTION
    int save_NumFilesPerSnapshot, save_NumFilesWrittenInParallel;
    double save_LLv_Step_Prec, save_SnII_Step_Prec;
    unsigned int buffer;
    double save_SofteningGasMaxPhys,
           save_SofteningHaloMaxPhys,
           save_SofteningDiskMaxPhys,
           save_SofteningBulgeMaxPhys, save_SofteningStarsMaxPhys, save_SofteningBndryMaxPhys;
    double save_MinChemTimeStep, save_MinChemSpreadL, save_MaxChemSpreadL;
#endif

#if defined(SFR) || defined(BLACK_HOLES)
#ifdef NO_TREEDATA_IN_RESTART
    if(modus == 0)
    {
        rearrange_particle_sequence();
        All.NumForcesSinceLastDomainDecomp = (long long) (1 + All.TreeDomainUpdateFrequency * All.TotNumPart);	/* ensures that new tree will be constructed */
    }
#endif
#endif

    if(ThisTask == 0 && modus == 0)
    {
        sprintf(buf, "%s/restartfiles", All.OutputDir);
        mkdir(buf, 02755);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    sprintf(buf, "%s/restartfiles/%s.%d", All.OutputDir, All.RestartFile, ThisTask);
    sprintf(buf_bak, "%s/restartfiles/%s.%d.bak", All.OutputDir, All.RestartFile, ThisTask);
    sprintf(buf_mv, "mv %s %s", buf, buf_bak);

    if((NTask < All.NumFilesWrittenInParallel))
    {
        printf
            ("Fatal error.\nNumber of processors must be a smaller or equal than `NumFilesWrittenInParallel'.\n");
        endrun(2131);
    }

    nprocgroup = NTask / All.NumFilesWrittenInParallel;

    if((NTask % All.NumFilesWrittenInParallel))
    {
        nprocgroup++;
    }

    masterTask = (ThisTask / nprocgroup) * nprocgroup;

    for(groupTask = 0; groupTask < nprocgroup; groupTask++)
    {
        if(ThisTask == (masterTask + groupTask))
        {
            if(!modus)
            {
#ifndef NOCALLSOFSYSTEM
                int ret;

                ret = system(buf_mv);	/* move old restart files to .bak files */
#endif
            }
        }
    }

    for(groupTask = 0; groupTask < nprocgroup; groupTask++)
    {
        if(ThisTask == (masterTask + groupTask))	/* ok, it's this processor's turn */
        {
            if(modus)
            {
                if(!(fd = fopen(buf, "r")))
                {
                    if(!(fd = fopen(buf_bak, "r")))
                    {
                        printf("Restart file '%s' nor '%s' found.\n", buf, buf_bak);
                        endrun(7870);
                    }
                }
            }
            else
            {
                if(!(fd = fopen(buf, "w")))
                {
                    printf("Restart file '%s' cannot be opened.\n", buf);
                    endrun(7878);
                }
            }


            save_PartAllocFactor = All.PartAllocFactor;

#ifdef LT_STELLAREVOLUTION
            save_NumFilesPerSnapshot = All.NumFilesPerSnapshot;
            save_NumFilesWrittenInParallel = All.NumFilesWrittenInParallel;

            save_SofteningGasMaxPhys = All.SofteningGasMaxPhys;
            save_SofteningHaloMaxPhys = All.SofteningHaloMaxPhys;
            save_SofteningDiskMaxPhys = All.SofteningDiskMaxPhys;
            save_SofteningBulgeMaxPhys = All.SofteningBulgeMaxPhys;
            save_SofteningStarsMaxPhys = All.SofteningStarsMaxPhys;
            save_SofteningBndryMaxPhys = All.SofteningBndryMaxPhys;

            save_SnII_Step_Prec = All.SnII_Step_Prec;
            save_LLv_Step_Prec = All.LLv_Step_Prec;

            save_MinChemTimeStep = All.MinChemTimeStep;
            save_MinChemSpreadL = All.MinChemSpreadL;
            save_MaxChemSpreadL = All.MaxChemSpreadL;
            if(!modus)
                *(float *) &buffer = (float) All.Time;
            in(&buffer, modus);
            if(!modus)
                buffer = sizeof(struct global_data_all_processes);
            in(&buffer, modus);
            if(modus && buffer != sizeof(struct global_data_all_processes))
            {
                printf
                    ("in file <%s> :: sizes of the current All structure and of the stored All structure are different! (%d vs %d bytes)\n",
                     buf, (int) buffer, (int) sizeof(struct global_data_all_processes));
                endrun(23);
            }
#endif

            /* common data  */
            byten(&All, sizeof(struct global_data_all_processes), modus);

#ifdef LT_STELLAREVOLUTION
            All.NumFilesPerSnapshot = save_NumFilesPerSnapshot;
            All.NumFilesWrittenInParallel = save_NumFilesWrittenInParallel;

            All.SnII_Step_Prec = save_SnII_Step_Prec;
            All.LLv_Step_Prec = save_LLv_Step_Prec;
            All.MinChemTimeStep = save_MinChemTimeStep;

            All.SofteningGasMaxPhys = save_SofteningGasMaxPhys;
            All.SofteningHaloMaxPhys = save_SofteningHaloMaxPhys;
            All.SofteningDiskMaxPhys = save_SofteningDiskMaxPhys;
            All.SofteningBulgeMaxPhys = save_SofteningBulgeMaxPhys;
            All.SofteningStarsMaxPhys = save_SofteningStarsMaxPhys;
            All.SofteningBndryMaxPhys = save_SofteningBndryMaxPhys;

            All.MinChemSpreadL = save_MinChemSpreadL;
            All.MaxChemSpreadL = save_MaxChemSpreadL;
#endif

#ifdef VORONOI
            /* individual data  */
            byten(&Indi, sizeof(struct individual_data), modus);
#endif

            if(ThisTask == 0 && modus > 0)
                all_task0 = All;

            if(modus > 0 && groupTask == 0)	/* read */
            {
                MPI_Bcast(&all_task0, sizeof(struct global_data_all_processes), MPI_BYTE, 0, MPI_COMM_WORLD);
            }

            old_MaxPart = All.MaxPart;
            old_MaxNodes = (int) (All.TreeAllocFactor * All.MaxPart) + NTopnodes;

            if(modus)		/* read */
            {
                if(All.PartAllocFactor != save_PartAllocFactor)
                {
                    All.PartAllocFactor = save_PartAllocFactor;
                    All.MaxPart = (int) (All.PartAllocFactor * (All.TotNumPart / NTask));
                    All.MaxPartSph = (int) (All.PartAllocFactor * (All.TotN_gas / NTask));
#ifdef INHOMOG_GASDISTR_HINT
                    All.MaxPartSph = All.MaxPart;
#endif
#ifdef LT_STELLAREVOLUTION
                    if(All.TotN_star == 0)
                        All.MaxPartMet =
                            All.PartAllocFactor * (All.TotN_gas / NTask) * All.SFfactor * All.Generations;
                    else
                        All.MaxPartMet =
                            All.PartAllocFactor * (All.TotN_star / NTask +
                                    (All.TotN_gas / NTask) * All.SFfactor * All.Generations);
#endif
                    save_PartAllocFactor = -1;
                }

                if(all_task0.Time != All.Time)
                {
                    printf("The restart file on task=%d is not consistent with the one on task=0\n", ThisTask);
                    fflush(stdout);
                    endrun(16);
                }

                allocate_memory();
            }

            in(&NumPart, modus);
#ifdef LT_STELLAREVOLUTION
            if(!modus)
                buffer = sizeof(struct particle_data);
            in(&buffer, modus);
            if(modus && buffer != sizeof(struct particle_data))
            {
                printf
                    ("in file <%s> :: sizes of the current particle data and of the stored particle data are different! (%d vs %d bytes)\n",
                     buf, buffer, (int) sizeof(struct particle_data));
                endrun(23);
            }
#endif
            if(NumPart > All.MaxPart)
            {
                printf
                    ("it seems you have reduced(!) 'PartAllocFactor' below the value of %g needed to load the restart file.\n",
                     NumPart / (((double) All.TotNumPart) / NTask));
                printf("fatal error\n");
                endrun(22);
            }

            /* Particle data  */
            byten(&P[0], NumPart * sizeof(struct particle_data), modus);

            in(&N_gas, modus);
#ifdef LT_STELLAREVOLUTION
            if(!modus)
                buffer = sizeof(struct sph_particle_data);
            in(&buffer, modus);
            if(modus && buffer != sizeof(struct sph_particle_data))
            {
                printf
                    ("in file <%s> :: sizes of the current sph particle data and of the stored sph particle data are different! (%d vs %d bytes)\n",
                     buf, buffer, (int) sizeof(struct sph_particle_data));
                endrun(23);
            }
#endif
            if(N_gas > 0)
            {
                if(N_gas > All.MaxPartSph)
                {
                    printf
                        ("SPH: it seems you have reduced(!) 'PartAllocFactor' below the value of %g needed to load the restart file.\n",
                         N_gas / (((double) All.TotN_gas) / NTask));
                    printf("fatal error\n");
                    endrun(222);
                }
                /* Sph-Particle data  */
                byten(&SphP[0], N_gas * sizeof(struct sph_particle_data), modus);
            }

            /* write state of random number generator */
            byten(gsl_rng_state(random_generator), gsl_rng_size(random_generator), modus);

            /* now store relevant data for tree */
#ifdef SFR
            in(&Stars_converted, modus);
#endif

#ifdef LT_STELLAREVOLUTION
            in(&N_stars, modus);
            if(!modus)
                buffer = sizeof(struct met_particle_data);
            in(&buffer, modus);
            if(modus && buffer != sizeof(struct met_particle_data))
            {
                printf
                    ("in file <%s> :: sizes of the current (star) met particle data and of the stored (star) met particle data are different! (%d vs %d bytes)\n",
                     buf, buffer, (int) sizeof(struct met_particle_data));
                endrun(23);
            }


            if(N_stars > 0)
            {
                if(N_stars > All.MaxPartMet)
                {
                    printf
                        ("MET: it seems you have reduced(!) 'PartAllocFactor' below the value of %g needed to load the restart file.\n",
                         N_stars / (((double) All.TotN_star) / NTask));
                    printf("fatal error\n");
                    endrun(2222);
                }
                /* Sph-Particle data  */
                byten(&MetP[0], N_stars * sizeof(struct met_particle_data), modus);
            }
#endif


#if !defined(NO_TREEDATA_IN_RESTART) || defined(LT_STELLAREVOLUTION)
            /* now store relevant data for tree */
            int nmulti = MULTIPLEDOMAINS;

            in(&nmulti, modus);
            if(modus != 0 && nmulti != MULTIPLEDOMAINS)
            {
                if(ThisTask == 0)
                    printf
                        ("Looks like you changed MULTIPLEDOMAINS from %d to %d.\nWill discard tree stored in restart files and construct a new one.\n",
                         nmulti, (int) MULTIPLEDOMAINS);

                All.NumForcesSinceLastDomainDecomp = (long long) (1 + All.TreeDomainUpdateFrequency * All.TotNumPart);	/* ensures that new tree will be constructed */
            }
            else
            {
                in(&NTopleaves, modus);
                in(&NTopnodes, modus);

                if(modus)		/* read */
                {
                    domain_allocate();
                    force_treeallocate((int) (All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);
                }

                in(&Numnodestree, modus);

                if(Numnodestree > MaxNodes)
                {
                    printf
                        ("Tree storage: it seems you have reduced(!) 'PartAllocFactor' below the value needed to load the restart file (task=%d). "
                         "Numnodestree=%d  MaxNodes=%d\n", ThisTask, Numnodestree, MaxNodes);
                    endrun(221);
                }

                byten(Nodes_base, Numnodestree * sizeof(struct NODE), modus);
                byten(Extnodes_base, Numnodestree * sizeof(struct extNODE), modus);

                byten(Father, NumPart * sizeof(int), modus);

                byten(Nextnode, NumPart * sizeof(int), modus);
                byten(Nextnode + All.MaxPart, NTopnodes * sizeof(int), modus);

                byten(DomainStartList, NTask * MULTIPLEDOMAINS * sizeof(int), modus);
                byten(DomainEndList, NTask * MULTIPLEDOMAINS * sizeof(int), modus);
                byten(TopNodes, NTopnodes * sizeof(struct topnode_data), modus);
                byten(DomainTask, NTopnodes * sizeof(int), modus);
                byten(DomainNodeIndex, NTopleaves * sizeof(int), modus);

                byten(DomainCorner, 3 * sizeof(double), modus);
                byten(DomainCenter, 3 * sizeof(double), modus);
                byten(&DomainLen, sizeof(double), modus);
                byten(&DomainFac, sizeof(double), modus);

                if(modus)		/* read */
                    if(All.PartAllocFactor != save_PartAllocFactor)
                    {
                        for(i = 0; i < NumPart; i++)
                            Father[i] += (All.MaxPart - old_MaxPart);

                        for(i = 0; i < NumPart; i++)
                            if(Nextnode[i] >= old_MaxPart)
                            {
                                if(Nextnode[i] >= old_MaxPart + old_MaxNodes)
                                    Nextnode[i] += (All.MaxPart - old_MaxPart) + (MaxNodes - old_MaxPart);
                                else
                                    Nextnode[i] += (All.MaxPart - old_MaxPart);
                            }

                        for(i = 0; i < Numnodestree; i++)
                        {
                            if(Nodes_base[i].u.d.sibling >= old_MaxPart)
                            {
                                if(Nodes_base[i].u.d.sibling >= old_MaxPart + old_MaxNodes)
                                    Nodes_base[i].u.d.sibling +=
                                        (All.MaxPart - old_MaxPart) + (MaxNodes - old_MaxNodes);
                                else
                                    Nodes_base[i].u.d.sibling += (All.MaxPart - old_MaxPart);
                            }

                            if(Nodes_base[i].u.d.father >= old_MaxPart)
                            {
                                if(Nodes_base[i].u.d.father >= old_MaxPart + old_MaxNodes)
                                    Nodes_base[i].u.d.father +=
                                        (All.MaxPart - old_MaxPart) + (MaxNodes - old_MaxNodes);
                                else
                                    Nodes_base[i].u.d.father += (All.MaxPart - old_MaxPart);
                            }

                            if(Nodes_base[i].u.d.nextnode >= old_MaxPart)
                            {
                                if(Nodes_base[i].u.d.nextnode >= old_MaxPart + old_MaxNodes)
                                    Nodes_base[i].u.d.nextnode +=
                                        (All.MaxPart - old_MaxPart) + (MaxNodes - old_MaxNodes);
                                else
                                    Nodes_base[i].u.d.nextnode += (All.MaxPart - old_MaxPart);
                            }
                        }

                        for(i = 0; i < NTopnodes; i++)
                            if(Nextnode[i + All.MaxPart] >= old_MaxPart)
                            {
                                if(Nextnode[i + All.MaxPart] >= old_MaxPart + old_MaxNodes)
                                    Nextnode[i + All.MaxPart] +=
                                        (All.MaxPart - old_MaxPart) + (MaxNodes - old_MaxNodes);
                                else
                                    Nextnode[i + All.MaxPart] += (All.MaxPart - old_MaxPart);
                            }

                        for(i = 0; i < NTopnodes; i++)
                            if(DomainNodeIndex[i] >= old_MaxPart)
                            {
                                if(DomainNodeIndex[i] >= old_MaxPart + old_MaxNodes)
                                    DomainNodeIndex[i] += (All.MaxPart - old_MaxPart) + (MaxNodes - old_MaxNodes);
                                else
                                    DomainNodeIndex[i] += (All.MaxPart - old_MaxPart);
                            }
                    }
            }
#endif
            fclose(fd);
        }
        else			/* wait inside the group */
        {
            if(modus > 0 && groupTask == 0)	/* read */
            {
                MPI_Bcast(&all_task0, sizeof(struct global_data_all_processes), MPI_BYTE, 0, MPI_COMM_WORLD);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
#if defined(HEALPIX)

    //this should be readed in the parameterfile

    if(modus)
    {
        All.Nside = 32;
        //
        if(ThisTask == 0)
            printf(" Restart calculation of Healpix %i with %i \n", All.Nside, NSIDE2NPIX(All.Nside));
        // initialize the healpix array (just in case)
        All.healpixmap = (float *) malloc(NSIDE2NPIX(All.Nside) * sizeof(float));
        for(i = 0; i < NSIDE2NPIX(All.Nside); i++)
            All.healpixmap[i] = 0;
        healpix_halo_cond(All.healpixmap);
    }
#endif
}



/* reads/writes n bytes 
*/
void byten(void *x, int n, int modus)
{
    if(modus)
        my_fread(x, n, 1, fd);
    else
        my_fwrite(x, n, 1, fd);
}


/* reads/writes one int 
*/
void in(int *x, int modus)
{
    if(modus)
        my_fread(x, 1, sizeof(int), fd);
    else
        my_fwrite(x, 1, sizeof(int), fd);
}
