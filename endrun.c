#include <mpi.h>
#include <stdio.h>

#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <errno.h>

#include "endrun.h"

#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <execinfo.h>

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
        seteuid(0);
        close(STDIN_FILENO);
        close(STDOUT_FILENO);
        dup2(pipefd[1],STDOUT_FILENO);
        dup2(pipefd[1],STDERR_FILENO);

        snprintf(parent, sizeof(parent), "%d", getppid());

        /* YF: xorg didn't have the last NULL; which seems to be wrong;
         * causing random failures in execle. */

        /* try a few tools in order; */
        execle("/usr/bin/pstack", "pstack", parent, NULL, NULL);
        execle("/usr/bin/eu-stack", "eu-stack", "-p", parent, NULL, NULL);

        sprintf(buf, "No tools to pretty print a stack trace for pid %d.\n"
                     "Fallback to glibc backtrace which may not contain all symbols.\n "
                     "run eu-addr2line to pretty print the output.\n", getppid());

        write(STDOUT_FILENO, buf, strlen(buf));
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
                write(STDOUT_FILENO, btline, strlen(btline));
            }
            else if ((bytesread < 0) ||
                    ((errno != EINTR) && (errno != EAGAIN)))
                done = 1;
        }
        close(pipefd[0]);

        waitpid(kidpid, &kidstat, 0);

        if (WIFEXITED(kidstat) && (WEXITSTATUS(kidstat) == EXIT_FAILURE)) {
            void * buf[100];
            backtrace_symbols_fd(buf, 100, STDOUT_FILENO);
            return -1;
        }
    }
    return 0;
}

static void
OsSigHandler(int no)
{
    const char btline[] = "Killed by Signal %d\n";
    char buf[128];
    sprintf(buf, btline, no);
    write(STDOUT_FILENO, buf, strlen(buf));

    show_backtrace();
    exit(-no);
}

static void
init_stacktrace()
{
    struct sigaction act, oact;

    int siglist[] = { SIGSEGV, SIGQUIT, SIGILL, SIGFPE, SIGBUS, 0};
    sigemptyset(&act.sa_mask);

    act.sa_handler = OsSigHandler;
    act.sa_flags = 0;

    int i;
    for(i = 0; siglist[i] != 0; i ++) {
        sigaction(siglist[i], &act, &oact);
    }
}

/* Watch out:
 *
 * On some versions of OpenMPI with CPU frequency scaling we see negative time
 * due to a bug in OpenMPI https://github.com/open-mpi/ompi/issues/3003
 *
 * But they have fixed it.
 */

static double _timestart = -1;

void
init_endrun()
{
    _timestart = MPI_Wtime();

    init_stacktrace();
}

static int
putline(const char * prefix, const char * line)
{
    const char * p, * q;
    p = q = line;
    int newline = 1;
    while(*p != 0) {
        if(newline)
            write(STDOUT_FILENO, prefix, strlen(prefix));
        if (*p == '\n') {
            write(STDOUT_FILENO, q, p - q + 1);
            q = p + 1;
            newline = 1;
            p ++;
            continue;
        }
        newline = 0;
        p++;
    }
    /* if the last line did not end with a new line, fix it here. */
    if (q != p) {
        const char * warning = "LASTMESSAGE did not end with new line: ";
        write(STDOUT_FILENO, warning, strlen(warning));
        write(STDOUT_FILENO, q, p - q);
        write(STDOUT_FILENO, "\n", 1);
    }
    return 0;
}

static void
messagev(int crash, int where, const char * fmt, va_list va)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    char prefix[128];

    char buf[4096];
    vsprintf(buf, fmt, va);

    if(where > 0) {
        sprintf(prefix, "[ %09.2f ] Task %d: ", MPI_Wtime() - _timestart, ThisTask);
    } else {
        sprintf(prefix, "[ %09.2f ] ", MPI_Wtime() - _timestart);
    }

    if(where <= 0)
        MPI_Barrier(MPI_COMM_WORLD);

    if(ThisTask == 0 || where > 0) {
        putline(prefix, buf);
        if(crash) {
            show_backtrace();
            MPI_Abort(MPI_COMM_WORLD, where);
        }
    }
}

/*  This function aborts the simulation.
 *
 *  if where > 0, the error is uncollective.
 *  if where <= 0, the error is 'collective',  only the root rank prints the error.
 */
void
endrun(int where, const char * fmt, ...)
{

    va_list va;
    va_start(va, fmt);
    va_end(va);
    messagev(1, where, fmt, va);
}


/*  This function writes a message.
 *
 *  if where > 0, the message is uncollective.
 *  if where <= 0, the message is 'collective', only the root rank prints the message. A barrier is applied.
 */

void message(int where, const char * fmt, ...)
{
    va_list va;
    va_start(va, fmt);
    va_end(va);
    messagev(0, where, fmt, va);
}

