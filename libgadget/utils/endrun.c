#include <mpi.h>
#include <stdio.h>

#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <errno.h>

#include "endrun.h"
#include "system.h"

#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <execinfo.h>
#include <signal.h>

/* obtain a stacktrace with exec/fork. this is signal handler safe.
 * function based on xorg_backtrace_pstack; extracted from  xorg-server/os/backtrace.c
 *
 * an external tool is spawn to investigate the current stack of
 * the crashing process. there are different opinions about
 * calling fork in a signal handlers.
 * But we are already crashing anyways if we land in a signal.
 *
 * if no external tool is found we fallback to glibc's backtrace.
 *
 * */
#define BT_BUF_SIZE 100

static int
show_backtrace(void)
{
    pid_t kidpid;
    int pipefd[2];

    if (pipe(pipefd) != 0) {
        return -1;
    }

    kidpid = fork();

    if (kidpid == -1) {
        /* ERROR */
        return -1;
    } else if (kidpid == 0) {
        /* CHILD */
        char parent[16];
        char buf[512];
        close(STDIN_FILENO);
        close(STDOUT_FILENO);
        dup2(pipefd[1],STDOUT_FILENO);
        dup2(pipefd[1],STDERR_FILENO);

        snprintf(parent, sizeof(parent), "%d", getppid());

        /* YF: xorg didn't have the last NULL; which seems to be wrong;
         * causing random failures in execle. */

        /* We can use pstack if the elfutils stack is not available,
         * but elfutils is more powerful and we have the glibc stack trace anyway.
         * Also ptrace will sometimes hang in some cluster configurations.*/
        // execle("/usr/bin/pstack", "pstack", parent, NULL, NULL);
        execle("/usr/bin/eu-stack", "eu-stack", "-p", parent, NULL, NULL);
        write(STDERR_FILENO, buf, strlen(buf));
        exit(EXIT_FAILURE);
    } else {
        /* PARENT */
        char btline[256];
        int kidstat = 0;
        int bytesread;
        int done = 0;

        close(pipefd[1]);

        while (!done) {
            bytesread = read(pipefd[0], btline, sizeof(btline) - 1);

            if (bytesread > 0) {
                btline[bytesread] = 0;
                write(STDERR_FILENO, btline, strlen(btline));
            }
            else if ((bytesread < 0) ||
                    ((errno != EINTR) && (errno != EAGAIN)))
                done = 1;
        }
        close(pipefd[0]);
        waitpid(kidpid, &kidstat, 0);
    }
    return 0;
}

static int ShowBacktrace;

static void
OsSigHandler(int no)
{
    const char btline[] = "Task %d Killed by Signal %d. Use eu-addr2line to get function names.\n";
    char linebuf[128];
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    sprintf(linebuf, btline, ThisTask, no);
    write(STDERR_FILENO, linebuf, strlen(linebuf));
    void * buf[BT_BUF_SIZE];
    int nlines = backtrace(buf, BT_BUF_SIZE);
    backtrace_symbols_fd(buf, nlines, STDERR_FILENO);
    if(ShowBacktrace)
        show_backtrace();
    MPI_Abort(MPI_COMM_WORLD, no);
}

void
init_endrun(int backtrace)
{
    struct sigaction act, oact;

    ShowBacktrace = backtrace;
    int siglist[] = { SIGSEGV, SIGQUIT, SIGILL, SIGFPE, SIGBUS, 0};
    sigemptyset(&act.sa_mask);

    act.sa_handler = OsSigHandler;
    act.sa_flags = 0;

    int i;
    for(i = 0; siglist[i] != 0; i ++) {
        sigaction(siglist[i], &act, &oact);
    }
}

/*  This function aborts the simulation.
 *
 *  if where > 0, a stacktrace is printed per rank calling endrun.
 *  if where <= 0, the function shall be called by all ranks collectively.
 *    and only the root rank prints the error.
 *
 *  No barrier is applied.
 */
void
endrun(int where, const char * fmt, ...)
{
    va_list va;
    va_start(va, fmt);
    MPIU_Tracev(MPI_COMM_WORLD, where, 1, fmt, va);
    va_end(va);
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0 || where > 0) {
        MPI_Abort(MPI_COMM_WORLD, where);
    }
    /* This is here so the compiler knows this
     * function never returns. */
    exit(1);
}


/*  This function writes a message.
 *
 *  if where > 0, the message is uncollective.
 *  if where <= 0, the message is 'collective', only the root rank prints the message.
 *
 *  No barrier is applied.
 */

void message(int where, const char * fmt, ...)
{
    va_list va;
    va_start(va, fmt);
    MPIU_Tracev(MPI_COMM_WORLD, where, 0, fmt, va);
    va_end(va);
}

