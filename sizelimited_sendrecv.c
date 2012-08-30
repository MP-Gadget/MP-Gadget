#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "proto.h"

#ifdef MPISENDRECV_SIZELIMIT


#undef MPI_Sendrecv


int MPI_Sizelimited_Sendrecv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
			     int dest, int sendtag, void *recvbuf, int recvcount,
			     MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm,
			     MPI_Status * status)
{
  int iter = 0, size_sendtype, size_recvtype, send_now, recv_now;
  int count_limit;


  if(dest != source)
    endrun(3);

  MPI_Type_size(sendtype, &size_sendtype);
  MPI_Type_size(recvtype, &size_recvtype);

  if(dest == ThisTask)
    {
      memcpy(recvbuf, sendbuf, recvcount * size_recvtype);
      return 0;
    }

  count_limit = (int) ((((long long) MPISENDRECV_SIZELIMIT) * 1024 * 1024) / size_sendtype);

  while(sendcount > 0 || recvcount > 0)
    {
      if(sendcount > count_limit)
	{
	  send_now = count_limit;
	  if(iter == 0)
	    {
	      printf("imposing size limit on MPI_Sendrecv() on task=%d (send of size=%d)\n",
		     ThisTask, sendcount * size_sendtype);
	      fflush(stdout);
	    }
	  iter++;
	}
      else
	send_now = sendcount;

      if(recvcount > count_limit)
	recv_now = count_limit;
      else
	recv_now = recvcount;

      MPI_Sendrecv(sendbuf, send_now, sendtype, dest, sendtag,
		   recvbuf, recv_now, recvtype, source, recvtag, comm, status);

      sendcount -= send_now;
      recvcount -= recv_now;

      sendbuf += send_now * size_sendtype;
      recvbuf += recv_now * size_recvtype;
    }

  return 0;
}

#endif
