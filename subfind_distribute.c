#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#ifdef SUBFIND

#include "allvars.h"
#include "proto.h"
#include "domain.h"
#include "fof.h"
#include "subfind.h"

void subfind_distribute_groups(void)
{
    int i, nexport = 0, nimport = 0, target, ngrp, sendTask, recvTask;
    struct group_properties *send_Group;

    /* count how many we have of each task */
    for(i = 0; i < NTask; i++)
        Send_count[i] = 0;

    for(i = 0; i < Ngroups; i++)
    {
        target = (Group[i].GrNr - 1) % NTask;
        if(target != ThisTask)
            Send_count[target]++;
    }

    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);

    for(i = 0, Recv_offset[0] = Send_offset[0] = 0; i < NTask; i++)
    {
        nimport += Recv_count[i];
        nexport += Send_count[i];

        if(i > 0)
        {
            Send_offset[i] = Send_offset[i - 1] + Send_count[i - 1];
            Recv_offset[i] = Recv_offset[i - 1] + Recv_count[i - 1];
        }
    }

    send_Group = (struct group_properties *) mymalloc("send_Group", nexport * sizeof(struct group_properties));

    for(i = 0; i < NTask; i++)
        Send_count[i] = 0;

    for(i = 0; i < Ngroups; i++)
    {
        target = (Group[i].GrNr - 1) % NTask;
        if(target != ThisTask)
        {
            send_Group[Send_offset[target] + Send_count[target]] = Group[i];
            Send_count[target]++;

            Group[i] = Group[Ngroups - 1];
            Ngroups--;
            i--;
        }
    }

    for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        sendTask = ThisTask;
        recvTask = ThisTask ^ ngrp;

        if(recvTask < NTask)
        {
            if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
            {
                /* get the group info */
                MPI_Sendrecv(&send_Group[Send_offset[recvTask]],
                        Send_count[recvTask] * sizeof(struct group_properties), MPI_BYTE,
                        recvTask, TAG_DENS_A,
                        &Group[Ngroups + Recv_offset[recvTask]],
                        Recv_count[recvTask] * sizeof(struct group_properties), MPI_BYTE,
                        recvTask, TAG_DENS_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    Ngroups += nimport;

    myfree(send_Group);
}


static int se_direction;

static int subfind_exchange_layout(int n) {
    if(se_direction == 0) {
        return P[n].targettask;
    } else {
        return P[n].origintask;
    }
}
void subfind_exchange(int direction, int particleproponly) {
    /* direction == 0 is to send 
     * direction == 1 is to recollect.
     *
     * if dmonly is 1 only exchange dm particles */
    se_direction = direction;
    domain_exchange(subfind_exchange_layout, particleproponly);
}

#endif
