#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>

#include <libgadget/allvars.h>
#include <libgadget/slotsmanager.h>
#include <libgadget/partmanager.h>

#include <libgadget/run.h>
#include <libgadget/checkpoint.h>
#include <libgadget/config.h>

#include <libgadget/utils.h>

#include "params.h"

void gsl_handler (const char * reason, const char * file, int line, int gsl_errno)
{
    endrun(2001,"GSL_ERROR in file: %s, line %d, errno:%d, error: %s\n",file, line, gsl_errno, reason);
}

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
    int NTask;
    int thread_provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    if(thread_provided != MPI_THREAD_FUNNELED)
        message(1, "MPI_Init_thread returned %d != MPI_THREAD_FUNNELED\n", thread_provided);

    if(argc < 2)
    {
        message(0, "Parameters are missing.\n");
        message(0, "Call with <ParameterFile> [<RestartFlag>] [<RestartSnapNum>]\n\n");
        message(0, "   RestartFlag    Action\n");
        message(0, "       1          Restart from last snapshot (LastSnapNum.txt) and continue simulation\n");
        message(0, "       2          Restart from specified snapshot (-1 for Initial Condition) and continue simulation\n");
        message(0, "       3          Run FOF if enabled\n");
        message(0, "       99         Run Tests. \n\n");
        MPI_Finalize();
        return 1;
    }

    message(0, "This is MP-Gadget, version %s.\n", GADGET_VERSION);
    message(0, "Running on %d MPI Ranks.\n", NTask);
#ifdef NO_OPENMP_SPINLOCK
    message(0,"Code compiled with NO_OPENMP_SPINLOCK (no locks), so no OpenMP threads.\n");
    omp_set_num_threads(1);
#else
    message(0, "           %d OpenMP Threads.\n", omp_get_max_threads());
#endif
    message(0, "Code was compiled with settings:\n"
           "%s\n", GADGET_COMPILER_SETTINGS);
    message(0, "Size of particle structure       %td  [bytes]\n",sizeof(struct particle_data));
    message(0, "Size of blackhole structure       %td  [bytes]\n",sizeof(struct bh_particle_data));
    message(0, "Size of sph particle structure   %td  [bytes]\n",sizeof(struct sph_particle_data));
    message(0, "Size of star particle structure   %td  [bytes]\n",sizeof(struct star_particle_data));

    tamalloc_init();

    read_parameter_file(argv[1]);	/* ... read in parameters for this run */

    int RestartFlag, RestartSnapNum;

    if(argc >= 3)
        RestartFlag = atoi(argv[2]);
    else
        RestartFlag = 2;

    if(argc >= 4)
        RestartSnapNum = atoi(argv[3]);
    else
        RestartSnapNum = -1;

    if(RestartFlag == 0) {
        message(0, "Restart flag of 0 is deprecated. Use 2.\n");
        RestartFlag = 2;
        RestartSnapNum = -1;
    }
    if(RestartFlag == 3 && RestartSnapNum < 0) {
        endrun(0, "Need to give the snapshot number if FOF is selected for output\n");
    }

    if(RestartFlag == 1) {
        RestartSnapNum = find_last_snapnum();
        message(0, "Last Snapshot number is %d.\n", RestartSnapNum);
    }

    /*Set up GSL so it gives a proper MPI termination*/
    gsl_set_error_handler(gsl_handler);

    /*Initialize the memory manager*/
    mymalloc_init(All.MaxMemSizePerNode);

    /* Make sure memory has finished initialising on all ranks before doing more.
     * This may improve stability */
    MPI_Barrier(MPI_COMM_WORLD);

    init_endrun(All.ShowBacktrace);

    begrun(RestartSnapNum);

    switch(RestartFlag) {
        case 3:
            runfof(RestartSnapNum);
            break;
        case 99:
            runtests(RestartSnapNum);
            break;
        default:
            run(RestartSnapNum);        /* main simulation loop */
            break;
    }
    MPI_Finalize();		/* clean up & finalize MPI */

    return 0;
}
