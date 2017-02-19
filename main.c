#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "proto.h"
#include "param.h"
#include "config.h"
#include "endrun.h"
#include "fof.h"

/*! \file main.c
 *  \brief start of the program
 */
/*!
 *  This function initializes the MPI communication packages, and sets
 *  cpu-time counters to 0. Then begrun() is called, which sets up
 *  the simulation either from IC's or from restart files.  Finally,
 *  run() is started, the main simulation loop, which iterates over
 *  the timesteps.
 */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);

    if(argc < 2)
    {
        if(ThisTask == 0)
        {
            printf("Parameters are missing.\n");
            printf("Call with <ParameterFile> [<RestartFlag>] [<RestartSnapNum>]\n");
            printf("\n");
            printf("   RestartFlag    Action\n");
            printf("       2          Restart from specified snapshot (-1 for Initial Condition) and continue simulation\n");
            printf("       3          Run FOF if enabled\n");
            printf("       99         Run Tests. \n");
            printf("\n");
        }
        goto byebye;
    }

    if(ThisTask == 0)
    {
        /*    printf("\nThis is P-Gadget, version `%s', svn-revision `%s'.\n", GADGETVERSION, svn_version()); */
        printf("This is MP-Gadget, version %s.\n", GADGETVERSION);
        printf("Running on %d MPI Ranks.\n", NTask);
        printf("           %d OpenMP Threads.\n", omp_get_max_threads());
        printf("Code was compiled with settings:\n"
               "%s\n", COMPILETIMESETTINGS);
        printf("Size of particle structure       %td  [bytes]\n",sizeof(struct particle_data));
        printf("Size of blackhole structure       %td  [bytes]\n",sizeof(struct bh_particle_data));
        printf("Size of sph particle structure   %td  [bytes]\n",sizeof(struct sph_particle_data));
    }

    read_parameter_file(argv[1]);	/* ... read in parameters for this run */

    int RestartFlag, RestartSnapNum;

    if(argc >= 3)
        RestartFlag = atoi(argv[2]);
    else
        RestartFlag = 0;

    if(argc >= 4)
        RestartSnapNum = atoi(argv[3]);
    else
        RestartSnapNum = -1;

    if(RestartFlag == 0) {
        message(1, "Restart flag of 0 is deprecated. Use 2.\n");
        RestartFlag = 2;
        RestartSnapNum = -1;
    }
    if(RestartFlag == 1) {
        endrun(1, "Restarting from restart file is no longer supported. Use a snapshot instead.\n");
    }

    if(RestartFlag == 3 && RestartSnapNum < 0) {
        endrun(1, "Need to give the snapshot number if FOF is selected for output\n");
    }

    switch(RestartFlag) {
        case 3:
            begrun(3, RestartSnapNum); 
            fof_fof(RestartSnapNum);
            break;
        case 99:
            begrun(2, RestartSnapNum);
            runtests();
            break;
        default:
            begrun(2, RestartSnapNum);
            run();			/* main simulation loop */
            break;
    }
byebye:
    MPI_Finalize();		/* clean up & finalize MPI */

    return 0;
}

