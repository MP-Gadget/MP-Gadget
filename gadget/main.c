#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>

#include <libgadget/slotsmanager.h>
#include <libgadget/partmanager.h>

#include <libgadget/run.h>
#include <libgadget/checkpoint.h>
#include <libgadget/config.h>
#include <libgadget/forcetree.c>

#include <libgadget/utils.h>

#include "params.h"

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
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    if(thread_provided != MPI_THREAD_FUNNELED)
        message(1, "MPI_Init_thread returned %d != MPI_THREAD_FUNNELED\n", thread_provided);

    message(0, "This is MP-Gadget, version %s.\n", GADGET_VERSION);
    message(0, "Running on %d MPI Ranks.\n", NTask);
    message(0, "           %d OpenMP Threads.\n", omp_get_max_threads());
    message(0, "Code was compiled with settings:\n"
           "%s\n", GADGET_COMPILER_SETTINGS);
    message(0, "Size of particle structure       %td  [bytes]\n",sizeof(struct particle_data));
    message(0, "Size of blackhole structure       %td  [bytes]\n",sizeof(struct bh_particle_data));
    message(0, "Size of sph particle structure   %td  [bytes]\n",sizeof(struct sph_particle_data));
    message(0, "Size of star particle structure   %td  [bytes]\n",sizeof(struct star_particle_data));
    message(0, "Size of force tree node structure %td [bytes]\n", sizeof(struct NODE));

    if(argc < 2)
    {
        message(0, "Parameters are missing.\n");
        message(0, "Call with <ParameterFile> [<RestartFlag>] [<RestartSnapNum>]\n\n");
        message(0, "   RestartFlag    Action\n");
        message(0, "       1          Restart from last snapshot (LastSnapNum.txt) and continue simulation\n");
        message(0, "       2          Restart from specified snapshot (-1 for Initial Condition) and continue simulation\n");
        message(0, "       3          Run FOF if enabled\n");
        message(0, "       4          Generate a power spectrum and exit\n");
        message(0, "       99         Run Tests. \n\n");
        MPI_Finalize();
        return 1;
    }

    /* Avoid dumping core, except for the master process. For large jobs writing core
     * with all of main memory can be very bad. */
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask != 0) {
        struct rlimit rlim = {0};
        setrlimit(RLIMIT_CORE, &rlim);
    }
    tamalloc_init();

    int ShowBacktrace;
    double MaxMemSizePerNode;
    read_parameter_file(argv[1], &ShowBacktrace, &MaxMemSizePerNode);	/* ... read in parameters for this run */

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
    }
    if(RestartFlag == 1) {
        RestartSnapNum = find_last_snapshot();
    }
    if(RestartFlag == 3 && RestartSnapNum < 0) {
        endrun(0, "Need to give the snapshot number if FOF is selected for output\n");
    }

    /*Initialize the memory manager*/
    mymalloc_init(MaxMemSizePerNode);

    /* Make sure memory has finished initialising on all ranks before doing more.
     * This may improve stability */
    MPI_Barrier(MPI_COMM_WORLD);

    init_endrun(ShowBacktrace);

    struct header_data head = {0};
    inttime_t ti_init = begrun(RestartSnapNum, &head);

    switch(RestartFlag) {
        case 3:
            runfof(RestartSnapNum, ti_init, &head);
            break;
        case 4:
            runpower(&head);
            break;
        case 99:
            runtests(RestartSnapNum, ti_init, &head);
            break;
        default:
            run(RestartSnapNum, ti_init, &head);        /* main simulation loop */
            break;
    }
    MPI_Finalize();		/* clean up & finalize MPI */

    return 0;
}
