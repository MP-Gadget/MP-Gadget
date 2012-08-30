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




void subfind_distribute_particles(int mode)
{
  int nexport = 0, nimport = 0;
  int i, n, ngrp, target = 0;
  struct particle_data *partBuf;

  for(n = 0; n < NTask; n++)
    Send_count[n] = 0;

  for(n = 0; n < NumPart; n++)
    {
      if(mode == 1)
	{
	  if(P[n].targettask != ThisTask)
	    Send_count[P[n].targettask] += 1;
	}
      else if(mode == 2)
	{
	  if(P[n].origintask != ThisTask)
	    Send_count[P[n].origintask] += 1;
	}
    }

  MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);

  for(i = 0, Send_offset[0] = Recv_offset[0] = 0; i < NTask; i++)
    {
      nimport += Recv_count[i];
      nexport += Send_count[i];

      if(i > 0)
	{
	  Send_offset[i] = Send_offset[i - 1] + Send_count[i - 1];
	  Recv_offset[i] = Recv_offset[i - 1] + Recv_count[i - 1];
	}
    }


  if(NumPart - nexport + nimport > All.MaxPart)
    {
      printf
	("on task=%d the maximum particle number All.MaxPart=%d is reached (NumPart=%d togo=%d toget=%d)",
	 ThisTask, All.MaxPart, NumPart, nexport, nimport);
      endrun(8765);
    }

  partBuf = (struct particle_data *) mymalloc("partBuf", nexport * sizeof(struct particle_data));

  for(i = 0; i < NTask; i++)
    Send_count[i] = 0;

  for(n = 0; n < NumPart; n++)
    {
      if(mode == 0)
	{
	  if(!(P[n].GrNr > Ncollective && P[n].GrNr <= TotNgroups))	/* particle is in small group */
	    continue;

	  target = (P[n].GrNr - 1) % NTask;
	}
      else if(mode == 1)
	{
	  target = P[n].targettask;
	}
      else if(mode == 2)
	{
	  target = P[n].origintask;
	}

      if(target != ThisTask)
	{
	  partBuf[Send_offset[target] + Send_count[target]] = P[n];
	  Send_count[target]++;

	  P[n] = P[NumPart - 1];
	  NumPart--;
	  n--;
	}
    }

#ifndef NO_ISEND_IRECV_IN_DOMAIN
  int n_requests = 0;

  MPI_Request *requests = (MPI_Request *) mymalloc("requests", 10 * NTask * sizeof(MPI_Request));

  for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
      target = ThisTask ^ ngrp;

      if(target < NTask)
	{
	  if(Recv_count[target] > 0)
	    MPI_Irecv(P + NumPart + Recv_offset[target], Recv_count[target] * sizeof(struct particle_data),
		      MPI_BYTE, target, TAG_PDATA, MPI_COMM_WORLD, &requests[n_requests++]);
	}
    }


  for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
      target = ThisTask ^ ngrp;

      if(target < NTask)
	{
	  if(Send_count[target] > 0)
	    MPI_Isend(partBuf + Send_offset[target], Send_count[target] * sizeof(struct particle_data),
		      MPI_BYTE, target, TAG_PDATA, MPI_COMM_WORLD, &requests[n_requests++]);
	}
    }

  MPI_Waitall(n_requests, requests, MPI_STATUSES_IGNORE);

  myfree(requests);

#else

  double t0, t1;

  t0 = second();

  for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
      /*
         if(NTask > 512)
         {
         if(ThisTask == 0)
         {
         t1 = second();
         printf("ngrp=%d %g sec\n", ngrp, timediff(t0, t1));
         fflush(stdout);
         t0 = second();
         }
         }
       */
      target = ThisTask ^ ngrp;

      if(target < NTask)
	{
	  MPI_Sendrecv(partBuf + Send_offset[target], Send_count[target] * sizeof(struct particle_data),
		       MPI_BYTE, target, TAG_PDATA,
		       P + NumPart + Recv_offset[target], Recv_count[target] * sizeof(struct particle_data),
		       MPI_BYTE, target, TAG_PDATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
    }

#endif

  NumPart += nimport;

  myfree(partBuf);
}




#ifndef NO_ISEND_IRECV_IN_DOMAIN

void subfind_exchange(void)
{
  int count_togo = 0, count_togo_sph = 0, count_get = 0, count_get_sph = 0;
  int *count, *count_sph, *offset, *offset_sph;
  int *count_recv, *count_recv_sph, *offset_recv, *offset_recv_sph;
  int i, n, ngrp, target, n_requests;
  int *toGo, *toGoSph;
  int *local_toGo, *local_toGoSph;
  struct particle_data *partBuf;
  struct sph_particle_data *sphBuf;
  MPI_Request *requests;

#ifdef LT_STELLAREVOLUTION
  int count_togo_stars = 0, count_get_stars = 0;
  int *count_stars, *offset_stars;
  int *count_recv_stars, *offset_recv_stars;
  int *toGoStars;
  int *local_toGoStars;
  struct met_particle_data *metBuf;
#endif
  
  requests = (MPI_Request *) mymalloc("requests", 16 * NTask * sizeof(MPI_Request));
  count = (int *) mymalloc("count", NTask * sizeof(int));
  count_sph = (int *) mymalloc("count_sph", NTask * sizeof(int));
  offset = (int *) mymalloc("offset", NTask * sizeof(int));
  offset_sph = (int *) mymalloc("offset_sph", NTask * sizeof(int));
  
  count_recv = (int *) mymalloc("count_recv", NTask * sizeof(int));
  count_recv_sph = (int *) mymalloc("count_recv_sph", NTask * sizeof(int));
  offset_recv = (int *) mymalloc("offset_recv", NTask * sizeof(int));
  offset_recv_sph = (int *) mymalloc("offset_recv_sph", NTask * sizeof(int));

  toGo = (int *) mymalloc("toGo", NTask * NTask * sizeof(int));
  toGoSph = (int *) mymalloc("toGoSph", NTask * NTask * sizeof(int));
  local_toGo = (int *) mymalloc("local_toGo", NTask * sizeof(int));
  local_toGoSph = (int *) mymalloc("local_toGoSph", NTask * sizeof(int));

#ifdef LT_STELLAREVOLUTION
  count_stars = (int *) mymalloc("count_stars", NTask * sizeof(int));
  offset_stars = (int *) mymalloc("offset_stars", NTask * sizeof(int));
  count_recv_stars = (int *) mymalloc("count_recv_stars", NTask * sizeof(int));
  offset_recv_stars = (int *) mymalloc("offset_recv_stars", NTask * sizeof(int));
  toGoStars = (int *) mymalloc("toGoStars", NTask * NTask * sizeof(int));
  local_toGoStars = (int *) mymalloc("local_toGoStars", NTask * sizeof(int));
#endif

  for(n = 0; n < NTask; n++)
    {
      local_toGo[n] = 0;
      local_toGoSph[n] = 0;
#ifdef LT_STELLAREVOLUTION
      local_toGoStars[n] = 0;
#endif
    }


  for(n = 0; n < NumPart; n++)
    {
      if(P[n].targettask != ThisTask)
	{
	  local_toGo[P[n].targettask]++;
	  if(P[n].Type == 0)
	    local_toGoSph[P[n].targettask]++;
#ifdef LT_STELLAREVOLUTION
          if(P[n].Type == 4)
            local_toGoStars[P[n].targettask]++;
#endif
	}
    }

  MPI_Allgather(local_toGo, NTask, MPI_INT, toGo, NTask, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(local_toGoSph, NTask, MPI_INT, toGoSph, NTask, MPI_INT, MPI_COMM_WORLD);
#ifdef LT_STELLAREVOLUTION
  MPI_Allgather(local_toGoStars, NTask, MPI_INT, toGoStars, NTask, MPI_INT, MPI_COMM_WORLD);
#endif

  for(i = 1, offset_sph[0] = 0; i < NTask; i++)
    offset_sph[i] = offset_sph[i - 1] + toGoSph[ThisTask * NTask + i - 1];

#ifndef LT_STELLAREVOLUTION    
  offset[0] = offset_sph[NTask - 1] + toGoSph[ThisTask * NTask + NTask - 1];

  for(i = 1; i < NTask; i++)
    offset[i] = offset[i - 1] + (toGo[ThisTask * NTask + i - 1] - toGoSph[ThisTask * NTask + i - 1]);
#else
  offset_stars[0] = offset_sph[NTask - 1] + toGoSph[ThisTask * NTask + NTask - 1];
  for(i = 1; i < NTask; i++)
    offset_stars[i] = offset_stars[i - 1] + toGoStars[ThisTask * NTask + i - 1];

  offset[0] = offset_stars[NTask - 1] + toGoStars[ThisTask * NTask + NTask - 1];
  for(i = 1; i < NTask; i++)
    offset[i] = offset[i - 1] + (toGo[ThisTask * NTask + i - 1] - toGoSph[ThisTask * NTask + i - 1] - toGoStars[ThisTask * NTask + i - 1]);
#endif
  
  for(i = 0; i < NTask; i++)
    {
      count_togo += toGo[ThisTask * NTask + i];
      count_togo_sph += toGoSph[ThisTask * NTask + i];

      count_get += toGo[i * NTask + ThisTask];
      count_get_sph += toGoSph[i * NTask + ThisTask];

#ifdef LT_STELLAREVOLUTION
      count_togo_stars += toGoStars[ThisTask * NTask + i];
      count_get_stars += toGoStars[i * NTask + ThisTask];
#endif
    }

  if(NumPart + count_get - count_togo > All.MaxPart)
    {
      printf("Task=%d NumPart(%d)+count_get(%d)-count_togo(%d)=%d  All.MaxPart=%d\n", ThisTask, NumPart,
	     count_get, count_togo, NumPart + count_get - count_togo, All.MaxPart);
      endrun(12787878);
    }

  if(N_gas + count_get_sph - count_togo_sph > All.MaxPartSph)
    {
      printf("Task=%d N_gas(%d)+count_get_sph(%d)-count_togo_sph(%d)=%d All.MaxPartSph=%d\n", ThisTask,
	     N_gas, count_get_sph, count_togo_sph, N_gas + count_get_sph - count_togo_sph, All.MaxPartSph);
      endrun(712187879);
    }

#ifdef LT_STELLAREVOLUTION
  if(N_stars + count_get_stars - count_togo_stars > All.MaxPartMet)
    {
      printf("Task=%d N_stars(%d)+count_get_stars(%d)-count_togo_stars(%d)=%d All.MaxPartMet=%d\n", ThisTask,
	     N_stars, count_get_stars, count_togo_stars, N_stars + count_get_stars - count_togo_stars, All.MaxPartMet);
      endrun(712187879);
    }
#endif

  partBuf = (struct particle_data *) mymalloc("partBuf", count_togo * sizeof(struct particle_data));
  sphBuf = (struct sph_particle_data *) mymalloc("sphBuf", count_togo_sph * sizeof(struct sph_particle_data));

  for(i = 0; i < NTask; i++)
    count[i] = count_sph[i] = 0;

#ifdef LT_STELLAREVOLUTION
  metBuf =
    (struct met_particle_data *) mymalloc("metBuf", count_togo_stars * sizeof(struct met_particle_data));
  for(i = 0; i < NTask; i++)
    count_stars[i] = 0;
#endif
  
  for(n = 0; n < NumPart; n++)
    {
      if(P[n].targettask != ThisTask)
	{
	  target = P[n].targettask;

	  if(P[n].Type == 0)
	    {
	      partBuf[offset_sph[target] + count_sph[target]] = P[n];
	      sphBuf[offset_sph[target] + count_sph[target]] = SphP[n];
	      count_sph[target]++;
	    }
#ifdef LT_STELLAREVOLUTION
	  else if(P[n].Type == 4)
	    {
	      partBuf[offset_stars[target] + count_stars[target]] = P[n];
	      metBuf[offset_stars[target] - offset_stars[0] + count_stars[target]] = MetP[P[n].MetID];
	      count_stars[target]++;
	    }
#endif 
	  else
	    {
	      partBuf[offset[target] + count[target]] = P[n];
	      count[target]++;
	    }


	  if(P[n].Type == 0)
	    {
	      P[n] = P[N_gas - 1];
	      P[N_gas - 1] = P[NumPart - 1];

#ifdef LT_STELLAREVOLUTION
	      if(P[N_gas - 1].Type == 4)
		MetP[P[N_gas - 1].MetID].PID = N_gas - 1;
#endif 
	      SphP[n] = SphP[N_gas - 1];

	      NumPart--;
	      N_gas--;
	      n--;
	    }
#ifdef LT_STELLAREVOLUTION
	  else if(P[n].Type == 4)
	    {
	      MetP[P[n].MetID] = MetP[N_stars - 1];
	      P[MetP[N_stars - 1].PID].MetID = P[n].MetID;

	      if(n < NumPart - 1)
		{
		  P[n] = P[NumPart - 1];
		  if((P[n].Type & 15) == 4)
		    MetP[P[n].MetID].PID = n;
		}

	      NumPart--;
	      N_stars--;
	      n--;
	    }
#endif
	  else
	    {
	      P[n] = P[NumPart - 1];
#ifdef LT_STELLAREVOLUTION
	      if(P[n].Type == 4)
		MetP[P[n].MetID].PID = n;
#endif 
	      NumPart--;
	      n--;
	    }
	}
    }

#ifndef LT_STELLAREVOLUTION  
  if(count_get_sph)
    memmove(P + N_gas + count_get_sph, P + N_gas, (NumPart - N_gas) * sizeof(struct particle_data));
#else
  if(count_get_sph + count_get_stars)
    {
      memmove(P + N_gas + count_get_sph + count_get_stars, P + N_gas,
	      (NumPart - N_gas) * sizeof(struct particle_data));

      for(n = 0; n < N_stars; n++)
	{
	  MetP[n].PID += count_get_sph + count_get_stars;
	  if(P[MetP[n].PID].MetID != n)
	    {
	      printf("[Task %d] some serious error in adjusting the memory before particle exchange\n",
		     ThisTask);
	      fflush(stdout);
	      endrun(991000);
	    }
	}
    }
#endif
  
  for(i = 0; i < NTask; i++)
    {
      count_recv_sph[i] = toGoSph[i * NTask + ThisTask];
      count_recv[i] = toGo[i * NTask + ThisTask] - toGoSph[i * NTask + ThisTask];
#ifdef LT_STELLAREVOLUTION
      count_recv_stars[i] = toGoStars[i * NTask + ThisTask];
      count_recv[i] -= toGoStars[i * NTask + ThisTask];
#endif
    }

  for(i = 1, offset_recv_sph[0] = N_gas; i < NTask; i++)
    offset_recv_sph[i] = offset_recv_sph[i - 1] + count_recv_sph[i - 1];

#ifndef LT_STELLAREVOLUTION  
  offset_recv[0] = NumPart + count_get_sph;

  for(i = 1; i < NTask; i++)
    offset_recv[i] = offset_recv[i - 1] + count_recv[i - 1];
#else
  for(i = 1, offset_recv_stars[0] = N_gas + count_get_sph; i < NTask; i++)
    offset_recv_stars[i] = offset_recv_stars[i - 1] + count_recv_stars[i - 1];

  offset_recv[0] = NumPart + count_get_sph + count_get_stars;

  for(i = 1; i < NTask; i++)
    offset_recv[i] = offset_recv[i - 1] + count_recv[i - 1];
#endif
  
  n_requests = 0;

  for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
      target = ThisTask ^ ngrp;

      if(target < NTask)
	{
	  if(count_recv_sph[target] > 0)
	    {
	      MPI_Irecv(P + offset_recv_sph[target], count_recv_sph[target] * sizeof(struct particle_data),
			MPI_BYTE, target, TAG_PDATA_SPH, MPI_COMM_WORLD, &requests[n_requests++]);

	      MPI_Irecv(SphP + offset_recv_sph[target],
			count_recv_sph[target] * sizeof(struct sph_particle_data), MPI_BYTE, target,
			TAG_SPHDATA, MPI_COMM_WORLD, &requests[n_requests++]);
	    }

#ifdef LT_STELLAREVOLUTION
	  if(count_recv_stars[target] > 0)
	    {
	      MPI_Irecv(P + offset_recv_stars[target],
			count_recv_stars[target] * sizeof(struct particle_data), MPI_BYTE, target,
			TAG_PDATA_STARS, MPI_COMM_WORLD, &requests[n_requests++]);

	      MPI_Irecv(MetP + N_stars + offset_recv_stars[target] - offset_recv_sph[NTask - 1] -
			count_recv_sph[NTask - 1],
			count_recv_stars[target] * sizeof(struct met_particle_data), MPI_BYTE, target,
			TAG_METDATA, MPI_COMM_WORLD, &requests[n_requests++]);
	    }
#endif
          
	  if(count_recv[target] > 0)
	    {
	      MPI_Irecv(P + offset_recv[target], count_recv[target] * sizeof(struct particle_data),
			MPI_BYTE, target, TAG_PDATA, MPI_COMM_WORLD, &requests[n_requests++]);
	    }
	}
    }


  MPI_Barrier(MPI_COMM_WORLD);	/* not really necessary, but this will guarantee that all receives are
				   posted before the sends, which helps the stability of MPI on
				   bluegene, and perhaps some mpich1-clusters */

  for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
      target = ThisTask ^ ngrp;

      if(target < NTask)
	{
	  if(count_sph[target] > 0)
	    {
	      MPI_Isend(partBuf + offset_sph[target], count_sph[target] * sizeof(struct particle_data),
			MPI_BYTE, target, TAG_PDATA_SPH, MPI_COMM_WORLD, &requests[n_requests++]);

	      MPI_Isend(sphBuf + offset_sph[target], count_sph[target] * sizeof(struct sph_particle_data),
			MPI_BYTE, target, TAG_SPHDATA, MPI_COMM_WORLD, &requests[n_requests++]);
	    }

#ifdef LT_STELLAREVOLUTION
	  if(count_stars[target] > 0)
	    {
	      MPI_Isend(partBuf + offset_stars[target], count_stars[target] * sizeof(struct particle_data),
			MPI_BYTE, target, TAG_PDATA_STARS, MPI_COMM_WORLD, &requests[n_requests++]);

	      MPI_Isend(metBuf + offset_stars[target] - offset_sph[NTask - 1] - count_sph[NTask - 1],
			count_stars[target] * sizeof(struct met_particle_data), MPI_BYTE, target, TAG_METDATA,
			MPI_COMM_WORLD, &requests[n_requests++]);
	    }
#endif
          
	  if(count[target] > 0)
	    {
	      MPI_Isend(partBuf + offset[target], count[target] * sizeof(struct particle_data),
			MPI_BYTE, target, TAG_PDATA, MPI_COMM_WORLD, &requests[n_requests++]);
	    }
	}
    }

  MPI_Waitall(n_requests, requests, MPI_STATUSES_IGNORE);

  NumPart += count_get;
  N_gas += count_get_sph;
#ifdef LT_STELLAREVOLUTION
  for(i = 0; i < count_get_stars; i++)
    {
      if(P[offset_recv_stars[0] + i].Type != 4)
	printf("haloa oh!!!\n");
      P[offset_recv_stars[0] + i].MetID = N_stars + i;
      MetP[N_stars + i].PID = offset_recv_stars[0] + i;
    }
  N_stars += count_get_stars;
  for(i = 0; i < N_stars; i++)
    if(MetP[i].PID >= NumPart || P[MetP[i].PID].Type != 4)
      endrun(987654);
#endif
  
  if(NumPart > All.MaxPart)
    {
      printf("Task=%d NumPart=%d All.MaxPart=%d\n", ThisTask, NumPart, All.MaxPart);
      endrun(12787878);
    }

  if(N_gas > All.MaxPartSph)
    {
      printf("Task=%d N_gas=%d All.MaxPartSph=%d\n", ThisTask, N_gas, All.MaxPartSph);
      endrun(712187879);
    }

#ifdef LT_STELLAREVOLUTION
  if(N_stars > All.MaxPartMet)
    {
      printf("Task=%d N_stars=%d All.MaxPartMet=%d\n", ThisTask, N_stars, All.MaxPartMet);
      endrun(787880);
    }

  myfree(metBuf);
#endif
  
  myfree(sphBuf);
  myfree(partBuf);

#ifdef LT_STELLAREVOLUTION
  myfree(local_toGoStars);
  myfree(toGoStars);
  myfree(offset_recv_stars);
  myfree(count_recv_stars);
  myfree(offset_stars);
  myfree(count_stars);
#endif
  
  myfree(local_toGoSph);
  myfree(local_toGo);
  myfree(toGoSph);
  myfree(toGo);

  myfree(offset_recv_sph);
  myfree(offset_recv);
  myfree(count_recv_sph);
  myfree(count_recv);

  myfree(offset_sph);
  myfree(offset);
  myfree(count_sph);
  myfree(count);
  myfree(requests);
}

#else
void subfind_exchange(void)
{
  int count_togo = 0, count_togo_sph = 0, count_get = 0, count_get_sph = 0;
  int *count, *count_sph, *offset, *offset_sph;
  int *count_recv, *count_recv_sph, *offset_recv, *offset_recv_sph;
  int i, n, ngrp, target;
  int *toGo, *toGoSph;
  int *local_toGo, *local_toGoSph;
  struct particle_data *partBuf;
  struct sph_particle_data *sphBuf;

  count = (int *) mymalloc("count", NTask * sizeof(int));
  count_sph = (int *) mymalloc("count_sph", NTask * sizeof(int));
  offset = (int *) mymalloc("offset", NTask * sizeof(int));
  offset_sph = (int *) mymalloc("offset_sph", NTask * sizeof(int));

  count_recv = (int *) mymalloc("count_recv", NTask * sizeof(int));
  count_recv_sph = (int *) mymalloc("count_recv_sph", NTask * sizeof(int));
  offset_recv = (int *) mymalloc("offset_recv", NTask * sizeof(int));
  offset_recv_sph = (int *) mymalloc("offset_recv_sph", NTask * sizeof(int));

  toGo = (int *) mymalloc("toGo", NTask * NTask * sizeof(int));
  toGoSph = (int *) mymalloc("toGoSph", NTask * NTask * sizeof(int));
  local_toGo = (int *) mymalloc("local_toGo", NTask * sizeof(int));
  local_toGoSph = (int *) mymalloc("local_toGoSph", NTask * sizeof(int));

#ifdef LT_STELLAREVOLUTION
  count_stars = (int *) mymalloc("count_stars", NTask * sizeof(int));
  offset_stars = (int *) mymalloc("offset_stars", NTask * sizeof(int));
  count_recv_stars = (int *) mymalloc("count_recv_stars", NTask * sizeof(int));
  offset_recv_stars = (int *) mymalloc("offset_recv_stars", NTask * sizeof(int));
  toGoStars = (int *) mymalloc("toGoStars", NTask * NTask * sizeof(int));
  local_toGoStars = (int *) mymalloc("local_toGoStars", NTask * sizeof(int));
#endif
  
  for(n = 0; n < NTask; n++)
    {
      local_toGo[n] = 0;
      local_toGoSph[n] = 0;
#ifdef LT_STELLAREVOLUTION
      local_toGoStars[n] = 0;
#endif      
    }


  for(n = 0; n < NumPart; n++)
    {
      if(P[n].targettask != ThisTask)
	{
	  local_toGo[P[n].targettask]++;
	  if(P[n].Type == 0)
	    local_toGoSph[P[n].targettask]++;
#ifdef LT_STELLAREVOLUTION
          if(P[n].Type == 4)
            local_toGoStars[P[n].targettask]++;
#endif          
	}
    }

  MPI_Allgather(local_toGo, NTask, MPI_INT, toGo, NTask, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(local_toGoSph, NTask, MPI_INT, toGoSph, NTask, MPI_INT, MPI_COMM_WORLD);
#ifdef LT_STELLAREVOLUTION
  MPI_Allgather(local_toGoStars, NTask, MPI_INT, toGoStars, NTask, MPI_INT, MPI_COMM_WORLD);
#endif

  for(i = 1, offset_sph[0] = 0; i < NTask; i++)
    offset_sph[i] = offset_sph[i - 1] + toGoSph[ThisTask * NTask + i - 1];

#ifndef LT_STELLAREVOLUTION    
  offset[0] = offset_sph[NTask - 1] + toGoSph[ThisTask * NTask + NTask - 1];

  for(i = 1; i < NTask; i++)
    offset[i] = offset[i - 1] + (toGo[ThisTask * NTask + i - 1] - toGoSph[ThisTask * NTask + i - 1]);
#else
  offset_stars[0] = offset_sph[NTask - 1] + toGoSph[ThisTask * NTask + NTask - 1];
  for(i = 1; i < NTask; i++)
    offset_stars[i] = offset_stars[i - 1] + toGoStars[ThisTask * NTask + i - 1];

  offset[0] = offset_stars[NTask - 1] + toGoStars[ThisTask * NTask + NTask - 1];
  for(i = 1; i < NTask; i++)
    offset[i] = offset[i - 1] + (toGo[ThisTask * NTask + i - 1] - toGoSph[ThisTask * NTask + i - 1] - toGoStars[ThisTask * NTask + i - 1]);
#endif
  
  for(i = 0; i < NTask; i++)
    {
      count_togo += toGo[ThisTask * NTask + i];
      count_togo_sph += toGoSph[ThisTask * NTask + i];

      count_get += toGo[i * NTask + ThisTask];
      count_get_sph += toGoSph[i * NTask + ThisTask];

#ifdef LT_STELLAREVOLUTION
      count_togo_stars += toGoStars[ThisTask * NTask + i];
      count_get_stars += toGoStars[i * NTask + ThisTask];
#endif      
    }

  if(NumPart + count_get - count_togo > All.MaxPart)
    {
      printf("Task=%d NumPart(%d)+count_get(%d)-count_togo(%d)=%d  All.MaxPart=%d\n", ThisTask, NumPart,
	     count_get, count_togo, NumPart + count_get - count_togo, All.MaxPart);
      endrun(312787878);
    }

  if(N_gas + count_get_sph - count_togo_sph > All.MaxPartSph)
    {
      printf("Task=%d N_gas(%d)+count_get_sph(%d)-count_togo_sph(%d)=%d All.MaxPartSph=%d\n", ThisTask,
	     N_gas, count_get_sph, count_togo_sph, N_gas + count_get_sph - count_togo_sph, All.MaxPartSph);
      endrun(3712187879);
    }

#ifdef LT_STELLAREVOLUTION
  if(N_stars + count_get_stars - count_togo_stars > All.MaxPartMet)
    {
      printf("Task=%d N_stars(%d)+count_get_stars(%d)-count_togo_stars(%d)=%d All.MaxPartMet=%d\n", ThisTask,
	     N_stars, count_get_stars, count_togo_stars, N_stars + count_get_stars - count_togo_stars, All.MaxPartMet);
      endrun(712187879);
    }
#endif
  
  partBuf = (struct particle_data *) mymalloc("partBuf", count_togo * sizeof(struct particle_data));
  sphBuf = (struct sph_particle_data *) mymalloc("sphBuf", count_togo_sph * sizeof(struct sph_particle_data));


  for(i = 0; i < NTask; i++)
    count[i] = count_sph[i] = 0;

#ifdef LT_STELLAREVOLUTION
  metBuf =
    (struct met_particle_data *) mymalloc("metBuf", count_togo_stars * sizeof(struct met_particle_data));
  for(i = 0; i < NTask; i++)
    count_stars[i] = 0;
#endif
  
  for(n = 0; n < NumPart; n++)
    {
      if(P[n].targettask != ThisTask)
	{
	  target = P[n].targettask;

	  if(P[n].Type == 0)
	    {
	      partBuf[offset_sph[target] + count_sph[target]] = P[n];
	      sphBuf[offset_sph[target] + count_sph[target]] = SphP[n];
	      count_sph[target]++;
	    }
#ifdef LT_STELLAREVOLUTION
	  else if(P[n].Type == 4)
	    {
	      partBuf[offset_stars[target] + count_stars[target]] = P[n];
	      metBuf[offset_stars[target] - offset_stars[0] + count_stars[target]] = MetP[P[n].MetID];
	      count_stars[target]++;
	    }
#endif           
	  else
	    {
	      partBuf[offset[target] + count[target]] = P[n];
	      count[target]++;
	    }

	  if(P[n].Type == 0)
	    {
	      P[n] = P[N_gas - 1];
	      P[N_gas - 1] = P[NumPart - 1];

#ifdef LT_STELLAREVOLUTION
	      if(P[N_gas - 1].Type == 4)
		MetP[P[N_gas - 1].MetID].PID = N_gas - 1;
#endif               
	      SphP[n] = SphP[N_gas - 1];

	      NumPart--;
	      N_gas--;
	      n--;
	    }
#ifdef LT_STELLAREVOLUTION
	  else if(P[n].Type == 4)
	    {
	      MetP[P[n].MetID] = MetP[N_stars - 1];
	      P[MetP[N_stars - 1].PID].MetID = P[n].MetID;

	      if(n < NumPart - 1)
		{
		  P[n] = P[NumPart - 1];
		  if((P[n].Type & 15) == 4)
		    MetP[P[n].MetID].PID = n;
		}

	      NumPart--;
	      N_stars--;
	      n--;
	    }
#endif          
	  else
	    {
	      P[n] = P[NumPart - 1];
#ifdef LT_STELLAREVOLUTION
	      if(P[n].Type == 4)
		MetP[P[n].MetID].PID = n;
#endif               
	      NumPart--;
	      n--;
	    }
	}
    }

#ifndef LT_STELLAREVOLUTION    
  if(count_get_sph)
    memmove(P + N_gas + count_get_sph, P + N_gas, (NumPart - N_gas) * sizeof(struct particle_data));
#else
  if(count_get_sph + count_get_stars)
    {
      memmove(P + N_gas + count_get_sph + count_get_stars, P + N_gas,
	      (NumPart - N_gas) * sizeof(struct particle_data));

      for(n = 0; n < N_stars; n++)
	{
	  MetP[n].PID += count_get_sph + count_get_stars;
	  if(P[MetP[n].PID].MetID != n)
	    {
	      printf("[Task %d] some serious error in adjusting the memory before particle exchange\n",
		     ThisTask);
	      fflush(stdout);
	      endrun(991000);
	    }
	}
    }
#endif
  
  for(i = 0; i < NTask; i++)
    {
      count_recv_sph[i] = toGoSph[i * NTask + ThisTask];
      count_recv[i] = toGo[i * NTask + ThisTask] - toGoSph[i * NTask + ThisTask];
#ifdef LT_STELLAREVOLUTION
      count_recv_stars[i] = toGoStars[i * NTask + ThisTask];
      count_recv[i] -= toGoStars[i * NTask + ThisTask];
#endif
    }

  for(i = 1, offset_recv_sph[0] = N_gas; i < NTask; i++)
    offset_recv_sph[i] = offset_recv_sph[i - 1] + count_recv_sph[i - 1];

#ifndef LT_STELLAREVOLUTION    
  offset_recv[0] = NumPart + count_get_sph;

  for(i = 1; i < NTask; i++)
    offset_recv[i] = offset_recv[i - 1] + count_recv[i - 1];
#else
  for(i = 1, offset_recv_stars[0] = N_gas + count_get_sph; i < NTask; i++)
    offset_recv_stars[i] = offset_recv_stars[i - 1] + count_recv_stars[i - 1];

  offset_recv[0] = NumPart + count_get_sph + count_get_stars;

  for(i = 1; i < NTask; i++)
    offset_recv[i] = offset_recv[i - 1] + count_recv[i - 1];
#endif
  
  for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
      target = ThisTask ^ ngrp;

      if(target < NTask)
	{
	  if(count_sph[target] > 0 || count_recv_sph[target] > 0)
	    {
	      MPI_Sendrecv(partBuf + offset_sph[target], count_sph[target] * sizeof(struct particle_data),
			   MPI_BYTE, target, TAG_PDATA_SPH,
			   P + offset_recv_sph[target], count_recv_sph[target] * sizeof(struct particle_data),
			   MPI_BYTE, target, TAG_PDATA_SPH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	      MPI_Sendrecv(sphBuf + offset_sph[target], count_sph[target] * sizeof(struct sph_particle_data),
			   MPI_BYTE, target, TAG_SPHDATA,
			   SphP + offset_recv_sph[target],
			   count_recv_sph[target] * sizeof(struct sph_particle_data), MPI_BYTE, target,
			   TAG_SPHDATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    }

#ifdef LT_STELLAREVOLUTION
	  if(count_stars[target] > 0 || count_recv_stars[target] > 0)
	    {
	      MPI_Sendrecv(partBuf + offset_stars[target], count_stars[target] * sizeof(struct particle_data),
			   MPI_BYTE, target, TAG_PDATA_STARS,
			   P + offset_recv_stars[target],
			   count_recv_stars[target] * sizeof(struct particle_data), MPI_BYTE, target,
			   TAG_PDATA_STARS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	      MPI_Sendrecv(metBuf + offset_stars[target] - offset_sph[NTask - 1],
			   count_stars[target] * sizeof(struct met_particle_data), MPI_BYTE, target,
			   TAG_METDATA,
			   MetP + offset_recv_stars[target] - offset_recv_sph[NTask - 1] -
			   count_recv_sph[NTask - 1],
			   count_recv_stars[target] * sizeof(struct met_particle_data), MPI_BYTE, target,
			   TAG_SPHDATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    }
#endif
          
	  if(count[target] > 0 || count_recv[target] > 0)
	    {
	      MPI_Sendrecv(partBuf + offset[target], count[target] * sizeof(struct particle_data),
			   MPI_BYTE, target, TAG_PDATA,
			   P + offset_recv[target], count_recv[target] * sizeof(struct particle_data),
			   MPI_BYTE, target, TAG_PDATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    }
	}
    }

  NumPart += count_get;
  N_gas += count_get_sph;
#ifdef LT_STELLAREVOLUTION
  for(i = 0; i < count_get_stars; i++)
    {
      if(P[offset_recv_stars[0] + i].Type != 4)
	printf("haloa oh!!!\n");
      P[offset_recv_stars[0] + i].MetID = N_stars + i;
      MetP[N_stars + i].PID = offset_recv_stars[0] + i;
    }
  N_stars += count_get_stars;
#endif
  if(NumPart > All.MaxPart)
    {
      printf("Task=%d NumPart=%d All.MaxPart=%d\n", ThisTask, NumPart, All.MaxPart);
      endrun(112787878);
    }

  if(N_gas > All.MaxPartSph)
    {
      printf("Task=%d N_gas=%d All.MaxPartSph=%d\n", ThisTask, N_gas, All.MaxPartSph);
      endrun(1712187879);
    }

#ifdef LT_STELLAREVOLUTION
  if(N_stars > All.MaxPartMet)
    {
      printf("Task=%d N_stars=%d All.MaxPartMet=%d\n", ThisTask, N_stars, All.MaxPartMet);
      endrun(787880);
    }

  myfree(metBuf);
#endif
  
  myfree(sphBuf);
  myfree(partBuf);

#ifdef LT_STELLAREVOLUTION
  myfree(offset_recv_stars);
  myfree(count_recv_stars);
  myfree(offset_stars);
  myfree(count_stars);
#endif
  
  myfree(local_toGoSph);
  myfree(local_toGo);
  myfree(toGoSph);
  myfree(toGo);

  myfree(offset_recv_sph);
  myfree(offset_recv);
  myfree(count_recv_sph);
  myfree(count_recv);

  myfree(offset_sph);
  myfree(offset);
  myfree(count_sph);
  myfree(count);
}


#endif
























#endif
