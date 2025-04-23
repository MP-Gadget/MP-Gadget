#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "utils/endrun.h"
#include "utils/string.h"
#include "utils/mymalloc.h"
#include "hci.h"

static double
hci_now(HCIManager * manager)
{
    if(manager->OVERRIDE_NOW) {
        return manager->_now;
    }
    manager->_now = MPI_Wtime();
    /* must be consistent between all ranks. */
    MPI_Bcast(&manager->_now, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return manager->_now;
}

void
hci_init(HCIManager * manager, char * prefix, const double WallClockTimeLimit, const double AutoCheckPointTime, const int FOFEnabled)
{
    manager->prefix = prefix;
    manager->timer_begin = hci_now(manager);
    manager->timer_query_begin = manager->timer_begin;

    manager->WallClockTimeLimit = WallClockTimeLimit;
    manager->AutoCheckPointTime = AutoCheckPointTime;
    manager->TimeLastCheckPoint = manager->timer_begin;
    manager->FOFEnabled = FOFEnabled;
    manager->LongestTimeBetweenQueries = 0;
}

void
hci_action_init(HCIAction * action)
{
    action->type = HCI_NO_ACTION;
    action->write_snapshot = 0;
    action->write_fof = 0;
    action->write_plane = 0;
}

/* override the result of hci_now; for unit testing -- we can't rely on MPI_Wtime there!
 * this function can be called before hci_init. */
void
hci_override_now(HCIManager * manager, double now)
{
    manager->_now = now;
    manager->OVERRIDE_NOW = 1;
}

static double
hci_get_elapsed_time(HCIManager * manager)
{
    return manager->timer_query_begin - manager->timer_begin;
}

static
void hci_update_query_timer(HCIManager * manager)
{
    double e = hci_now(manager);
    double g = e - manager->timer_query_begin;
    if(g > manager->LongestTimeBetweenQueries)
        manager->LongestTimeBetweenQueries = g;

    manager->timer_query_begin = e;
}

/*
 * query the filesystem for HCI commands;
 * returns the content of the file or NULL; collectively
 * */
int
hci_query_filesystem(HCIManager * manager, const char * filename, char ** request)
{
    int ThisTask;
    int NTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    int size = 0;
    char * content = NULL;
    if(ThisTask == 0) {
        char * fullname = fastpm_strdup_printf("%s/%s", manager->prefix, filename);
        content = fastpm_file_get_content(fullname);
        if(content) {
            size = strlen(content);
            remove(fullname);
        } else {
            size = -1;
        }
        myfree(fullname);
    }
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(size != -1) {
        if(ThisTask != 0) {
            content = ta_malloc("hcicontent", char, size + 1);
        }
        MPI_Bcast(content, size+1, MPI_BYTE, 0, MPI_COMM_WORLD);
    } else {
        content = NULL;
    }

    *request = content;
    return *request != NULL;
}

static int
hci_query_timeout(HCIManager * manager, char ** request)
{
    /* this function is collective because we take care to ensure manager is
     * collective */
    double now = hci_get_elapsed_time(manager);
    /*
     * factor 0.9 is a safety tolerance
     * for possible inconsistency between measured time and the true wallclock
     *
     * If there likely isn't time for a new query, then we shall timeout as well.
     * */

    *request = NULL;
    if (now + manager->LongestTimeBetweenQueries < manager->WallClockTimeLimit * 0.95) {
        return 0;
    }

    /* any freeable string would work. */
    return 1;
}

static int
hci_query_auto_checkpoint(HCIManager * manager, char ** request)
{
    /* this function is collective because we take care to ensure manager is
     * collective */
    if(manager->AutoCheckPointTime <= 0) return 0;

    /* How long since the last checkpoint? */
    double now = hci_get_elapsed_time(manager);
    if(now - manager->TimeLastCheckPoint >= manager->AutoCheckPointTime) {
        return 1;
    }
    return 0;
}

/*
 * the return value is non-zero if the mainloop shall break.
 * */
int
hci_query(HCIManager * manager, HCIAction * action)
{
    hci_action_init(action);

    /* measure time since last query */
    hci_update_query_timer(manager);

    /* Check whether we need to interrupt the run */

    char * request;

    /* Will we run out of time by the query ? highest priority.
     */
    if(hci_query_timeout(manager, &request)) {
        message(0, "HCI: Stopping due to TimeLimitCPU, dumping a CheckPoint.\n");
        action->type = HCI_TIMEOUT;
        action->write_snapshot = 1;
        if(manager->FOFEnabled)
            action->write_fof = 1;
        return 1;
    }

    if(hci_query_filesystem(manager, "reconfigure", &request))
    {
        /* FIXME: This is not implemented
         * it shall reread the configuration file and update the parameters of
         * the module listed in the request.
         * see the comment about update_IO_params
         * */
        message(0, "HCI: updating io parameters, this is not supported yet.\n");
        myfree(request);
        return 0;
    }

    if(hci_query_filesystem(manager, "checkpoint", &request))
    {
        message(0, "HCI: human controlled stop with checkpoint at next PM.\n");
        action->type = HCI_CHECKPOINT;
        /* will write checkpoint in this PM timestep */
        action->write_snapshot = 1;
        /* Write fof as well*/
        if(manager->FOFEnabled)
            action->write_fof = 1;
        myfree(request);
        manager->TimeLastCheckPoint = hci_get_elapsed_time(manager);
        return 0;
    }

    /* Is the plane-file present? If yes, ask to write a plane file. */
    if(hci_query_filesystem(manager, "plane", &request))
    {
        /* will write a lensing plane in this PM timestep, then continue.*/
        action->type = HCI_PLANE;
        action->write_plane = 1;
        myfree(request);
        return 1;
    }

    /* Is the stop-file present? If yes, interrupt the run with a snapshot. */
    if(hci_query_filesystem(manager, "stop", &request))
    {
        /* will write checkpoint in this PM timestep, then stop */
        action->type = HCI_STOP;
        action->write_snapshot = 1;
        myfree(request);
        return 1;
    }

    /* Is the terminate-file present? If yes, interrupt the run immediately. */
    if(hci_query_filesystem(manager, "terminate", &request))
    {
        message(0, "HCI: human triggered termination.\n");
        /* the caller shall take care of immediate termination.
         * This action is better than KILL as it avoids corrupt/incomplete snapshot files.*/
        action->type = HCI_TERMINATE;
        action->write_snapshot = 0;
        myfree(request);
        return 1;
    }

    /* lower priority */
    if(hci_query_auto_checkpoint(manager, &request))
    {
        message(0, "HCI: Auto checkpoint due to AutoCheckPointTime.\n");
        action->type = HCI_AUTO_CHECKPOINT;
        /* Write when the PM timestep completes*/
        action->write_snapshot = 1;
        if(manager->FOFEnabled)
            action->write_fof = 1;
        manager->TimeLastCheckPoint = hci_get_elapsed_time(manager);
        return 0;
    }

    message(0, "HCI: Nothing happened. \n");
    return 0;
}

/*
 * FIXME: rewrite update_IO_params with
 * the parser infrastructure. It probably shall occur
 * after we decentralize the initialization of the parser
 * to different modules.
 * */

#if 0
static void
update_IO_params(const char * ioctlfname)
{
    if(ThisTask == 0) {
        FILE * fd = fopen(ioctlfname, "r");
         /* there is an ioctl file, parse it and update
          * All.NumPartPerFile
          * All.NumWriters
          */
        size_t n = 0;
        char * line = NULL;
        while(-1 != getline(&line, &n, fd)) {
            sscanf(line, "BytesPerFile %lu", &All.IO.BytesPerFile);
            sscanf(line, "NumWriters %d", &All.IO.NumWriters);
        }
        myfree(line);
        fclose(fd);
    }

    MPI_Bcast(&All.IO, sizeof(All.IO), MPI_BYTE, 0, MPI_COMM_WORLD);
    message(0, "New IO parameter recieved from %s:"
               "NumPartPerfile %d"
               "NumWriters %d\n",
            ioctlfname,
            All.IO.BytesPerFile,
            All.IO.NumWriters);
}
#endif
