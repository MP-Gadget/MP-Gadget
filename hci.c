#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "endrun.h"
#include "utils-string.h"
#include "hci.h"

HCIManager HCI_DEFAULT_MANAGER[1] = {
    {.OVERRIDE_NOW = 0},
};

static double
hci_now(HCIManager * manager)
{
    double e;
    if(manager->OVERRIDE_NOW) {
        e = manager->_now;
    } else {
        e = MPI_Wtime();
    }
    /* must be consistent between all ranks. */
    MPI_Bcast(&e, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return e;
}

void
hci_init(HCIManager * manager, char * prefix, double WallClockTimeLimit, double AutoCheckPointTime)
{
    manager->prefix = strdup(prefix);
    manager->timer_begin = hci_now(manager);
    manager->timer_query_begin = manager->timer_begin;

    manager->WallClockTimeLimit = WallClockTimeLimit;
    manager->AutoCheckPointTime = AutoCheckPointTime;
    manager->LongestTimeBetweenQueries = 0;
}

/* override the result of hci_now; for unit testing -- we can't rely on MPI_Wtime there! 
 * this function can be called before hci_init. */
int
hci_override_now(HCIManager * manager, double now)
{
    manager->_now = now;
    manager->OVERRIDE_NOW = 1;
}

static double
hci_get_elapsed_time(HCIManager * manager)
{
    double e = hci_now(manager) - manager->timer_begin;
    return e;
}

static
void hci_update_query_timer(HCIManager * manager)
{
    double e = hci_now(manager);
    e = e - manager->timer_query_begin;
    if(e > manager->LongestTimeBetweenQueries)
        manager->LongestTimeBetweenQueries = e;

    manager->timer_query_begin = e;
}

/*
 * query the filesystem for HCI commands;
 * returns the content of the file or NULL; collectively
 * */
static char *
hci_query_filesystem(HCIManager * manager, char * filename)
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
        free(fullname);
    }
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(size != -1) {
        if(ThisTask != 0) {
            content = calloc(size + 1, 1);
        }
        MPI_Bcast(content, size+1, MPI_BYTE, 0, MPI_COMM_WORLD);
    } else {
        content == NULL;
    }

    return content;
}

static char *
hci_query_timeout(HCIManager * manager)
{
    double now = hci_get_elapsed_time(manager);
    /*
     * factor 0.9 is a safety tolerance
     * for possible inconsistency between measured time and the true wallclock
     *
     * If there likely isn't time for a new query, then we shall timeout as well.
     * */

    if (now + manager->LongestTimeBetweenQueries < manager->WallClockTimeLimit * 0.9) {
        return NULL;
    }

    /* any freeable string would work. */
    return calloc(32, 1);
}

static char *
hci_query_auto_checkpoint(HCIManager * manager)
{
    /*How long since the last checkpoint?*/
    if(manager->AutoCheckPointTime <= 0) return NULL;

    double now = hci_get_elapsed_time(manager);
    if(now - manager->TimeLastCheckPoint >= manager->AutoCheckPointTime) {
        /* any freeable string would work */
        return calloc(32, 1);
    }
    return NULL;
}

/*
 * the return value is non-zero if the mainloop shall break.
 * */
int
hci_query(HCIManager * manager, HCIAction * action)
{
    /* measure time since last query */
    hci_update_query_timer(manager);

    /* Check whether we need to interrupt the run */

    char * request;

    /* Will we run out of time by the query ? highest priority.
     */
    if(request = hci_query_timeout(manager)) {
        message(0, "Stopping due to TimeLimitCPU, dumping a CheckPoint.\n");
        action->type = HCI_TIMEOUT;
        action->write_snapshot = 1;
        free(request);
        return 1;
    }

    if(request = hci_query_filesystem(manager, "ioctl"))
    {
        //update_IO_params(request);
        free(request);
        return 0;
    }

    if(request = hci_query_filesystem(manager, "checkpoint"))
    {
        message(0, "human controlled stop with checkpoint at next PM.\n");
        action->type = HCI_CHECKPOINT;
        /* will write checkpoint in this PM timestep */
        action->write_snapshot = 1;
        free(request);
        manager->TimeLastCheckPoint = hci_now(manager);
        return 0;
    }

    /* Is the stop-file present? If yes, interrupt the run with a snapshot. */
    if(request = hci_query_filesystem(manager, "stop"))
    {
        /* will write checkpoint in this PM timestep, then stop */
        action->type = HCI_STOP;
        action->write_snapshot = 1;
        free(request);
        return 1;
    }

    /* Is the terminate-file present? If yes, interrupt the run immediately. */
    if(request = hci_query_filesystem(manager, "terminate"))
    {
        /* the caller shall take care of immediate termination.
         * This action is better than KILL as it avoids corrupt/incomplete snapshot files.*/
        action->type = HCI_TERMINATE;
        action->write_snapshot = 0;
        free(request);
        return 1;
    }

    /* lower priority */
    if(request = hci_query_auto_checkpoint(manager))
    {
        message(0, "Auto checkpoint due to AutoCheckPointTime.\n");
        action->type = HCI_AUTO_CHECKPOINT;
        /* Write when the PM timestep completes*/
        action->write_snapshot = 1;
        manager->TimeLastCheckPoint = hci_now(manager);
        free(request);
        return 0;
    }

    /* nothing really happened. */
    action->type = HCI_NO_ACTION;
    action->write_snapshot = 0;
    return 0;
}
