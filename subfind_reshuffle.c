#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "allvars.h"
#include "proto.h"
#include "domain.h"

#if defined(SUBFIND) && defined(SUBFIND_RESHUFFLE_CATALOGUE)
#include "subfind.h"
#include "fof.h"

static int table_read = 0, table_read_voronoi = 0;

static int Nfiles, NfilesVoronoi;

static int64_t *NumPartPerFile;
static int64_t *NumPartPerFileVoronoi;


static struct id_list
{
  MyIDType ID;
  MyIDType GrNr;
};



void read_hsml_table(void)
{
  int i, dummy, nhsml;
  int64_t ntot;
  char fname[1000];
  FILE *fd;

  for(i = 0, Nfiles = 1; i < Nfiles; i++)
    {
      sprintf(fname, "%s/hsmldir_%03d/%s_%03d.%d", All.OutputDir, RestartSnapNum, "hsml", RestartSnapNum, i);
      if(!(fd = fopen(fname, "r")))
	{
	  printf("can't open file `%s`\n", fname);
	  endrun(113839);
	}

      my_fread(&nhsml, sizeof(int), 1, fd);
      my_fread(&dummy, sizeof(int), 1, fd);
      my_fread(&ntot, sizeof(int64_t), 1, fd);
      my_fread(&Nfiles, sizeof(int), 1, fd);

      if(i == 0)
	NumPartPerFile = mymalloc("NumPartPerFile", (Nfiles + 1) * sizeof(int64_t));

      NumPartPerFile[i] = nhsml;
    }

  int64_t n, sum;

  for(i = 0, sum = 0; i < Nfiles; i++)
    {
      n = NumPartPerFile[i];

      NumPartPerFile[i] = sum;

      sum += n;
    }

  NumPartPerFile[Nfiles] = sum;

  table_read = 1;
}

void subfind_reshuffle_free(void)
{
  if(table_read_voronoi)
    {
      myfree(NumPartPerFileVoronoi);
      table_read_voronoi = 0;
    }

  if(table_read)
    {
      myfree(NumPartPerFile);
      table_read = 0;
    }
}


void get_hsml_file(int64_t nskip, int count, int *filenr, int *n_to_read, int *n_to_skip)
{
  int i;

  for(i = 0; i < Nfiles; i++)
    if(nskip >= NumPartPerFile[i] && nskip < NumPartPerFile[i + 1])
      break;

  if(i >= Nfiles)
    endrun(1231239);

  *filenr = i;
  *n_to_skip = (int) (nskip - NumPartPerFile[i]);

  int nrest = (NumPartPerFile[i + 1] - NumPartPerFile[i] - *n_to_skip);

  if(count <= nrest)
    *n_to_read = count;
  else
    *n_to_read = nrest;
}

void read_hsml_files(float *Values, int count, enum iofields blocknr, int64_t nskip)
{
  char fname[1000];
  FILE *fd;
  float *tmp;
  int i, n_to_read, n_to_skip, filenr, dummy, ntask, nhsml;
  int64_t ntot;

  if(blocknr == IO_DMHSML_V || blocknr == IO_DMDENSITY_V)
    read_hsml_files_voronoi(Values, count, blocknr, nskip);
  else
    {
      while(count > 0)
	{
	  if(table_read == 0)
	    read_hsml_table();

	  get_hsml_file(nskip, count, &filenr, &n_to_read, &n_to_skip);

	  sprintf(fname, "%s/hsmldir_%03d/%s_%03d.%d", All.OutputDir, RestartSnapNum, "hsml", RestartSnapNum,
		  filenr);
	  if(!(fd = fopen(fname, "r")))
	    {
	      printf("can't open file `%s`\n", fname);
	      endrun(11383);
	    }

	  my_fread(&nhsml, sizeof(int), 1, fd);
	  my_fread(&dummy, sizeof(int), 1, fd);
	  my_fread(&ntot, sizeof(int64_t), 1, fd);
	  my_fread(&ntask, sizeof(int), 1, fd);

	  if(blocknr == IO_DMDENSITY)
	    fseek(fd, nhsml * sizeof(float), SEEK_CUR);	/* skip hsml */
	  if(blocknr == IO_DMVELDISP)
	    fseek(fd, 2 * nhsml * sizeof(float), SEEK_CUR);	/* skip hsml and density */

	  fseek(fd, n_to_skip * sizeof(float), SEEK_CUR);

	  tmp = mymalloc("tmp", n_to_read * sizeof(float));
	  my_fread(tmp, sizeof(float), n_to_read, fd);

	  for(i = 0; i < n_to_read; i++)
	    Values[i] = tmp[i];

	  myfree(tmp);

	  fclose(fd);

	  Values += n_to_read;
	  count -= n_to_read;
	  nskip += n_to_read;
	}
    }
}




void read_subfind_ids(void)
{
  FILE *fd;
  double t0, t1;
  char fname[500];
  int i, j;
  MyIDType *ids;
  int *list_of_nids;
  int ngrp, sendTask, recvTask;
  int Nids, nprocgroup, masterTask, groupTask;
  unsigned int64_t nid_previous;
  int fof_compare_P_SubNr(const void *a, const void *b);
  int64_t TotNids;
  int64_t *NumIdsPerFile;
  static struct id_list  *ID_list;

  if(ThisTask == 0)
    {
      printf("\nTrying to read preexisting Subhalo indices...  (presently allocated=%g MB)\n",
	     AllocatedBytes / (1024.0 * 1024.0));
      fflush(stdout);
    }


  if(ThisTask == 0)
    {
      for(i = 0, Nfiles = 1; i < Nfiles; i++)
	{
	  sprintf(fname, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, RestartSnapNum, "subhalo_ids",
		  RestartSnapNum, i);
	  if(!(fd = fopen(fname, "r")))
	    {
	      printf("can't read file `%s`\n", fname);
	      endrun(81184132);
	    }

	  my_fread(&Ngroups, sizeof(int), 1, fd);
	  my_fread(&TotNgroups, sizeof(int), 1, fd);
	  my_fread(&Nids, sizeof(int), 1, fd);
	  my_fread(&TotNids, sizeof(int64_t), 1, fd);
	  my_fread(&Nfiles, sizeof(int), 1, fd);
	  my_fread(&nid_previous, sizeof(unsigned int), 1, fd);	/* this is the number of IDs in previous files */
	  fclose(fd);

	  if(i == 0)
	    {
	      MPI_Bcast(&Nfiles, 1, MPI_INT, 0, MPI_COMM_WORLD);
	      NumIdsPerFile = mymalloc("NumIdsPerFile", (Nfiles) * sizeof(int64_t));
	    }

	  NumIdsPerFile[i] = Nids;
	}
    }
  else
    {
      MPI_Bcast(&Nfiles, 1, MPI_INT, 0, MPI_COMM_WORLD);
      NumIdsPerFile = mymalloc("NumIdsPerFile", (Nfiles) * sizeof(int64_t));
    }

  MPI_Bcast(NumIdsPerFile, Nfiles * sizeof(int64_t), MPI_BYTE, 0, MPI_COMM_WORLD);


  /* start reading of group catalogue */

  if(ThisTask == 0)
    {
      sprintf(fname, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, RestartSnapNum, "subhalo_ids",
	      RestartSnapNum, 0);
      if(!(fd = fopen(fname, "r")))
	{
	  printf("can't read file `%s`\n", fname);
	  endrun(11831);
	}

      my_fread(&Ngroups, sizeof(int), 1, fd);
      my_fread(&TotNgroups, sizeof(int), 1, fd);
      my_fread(&Nids, sizeof(int), 1, fd);
      my_fread(&TotNids, sizeof(int64_t), 1, fd);
      my_fread(&Nfiles, sizeof(int), 1, fd);
      fclose(fd);
    }

  MPI_Bcast(&Nfiles, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&TotNgroups, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&TotNids, sizeof(int64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

  t0 = second();

  if(NTask != Nfiles)
    {
      if(ThisTask == 0)
	printf
	  ("number of files (%d) in subhalo catalogues does not match MPI-Tasks, I'm working around this.\n",
	   Nfiles);

      ID_list = mymalloc("ID_list", (TotNids / NTask + NTask) * sizeof(struct id_list));

      int filenr, target, ngroups, nids, nsend, stored;

      int *nids_to_get = mymalloc("nids_to_get", NTask * sizeof(NTask));
      int *nids_obtained = mymalloc("nids_obtained", NTask * sizeof(NTask));

      for(i = 0; i < NTask; i++)
	nids_obtained[i] = 0;

      for(i = 0; i < NTask - 1; i++)
	nids_to_get[i] = (int) (TotNids / NTask);
      nids_to_get[NTask - 1] = (int) (TotNids - (NTask - 1) * (TotNids / NTask));

      Nids = nids_to_get[ThisTask];

      if(ThisTask == 0)
	{
	  /**** now ids ****/
	  for(filenr = 0; filenr < Nfiles; filenr++)
	    {
	      sprintf(fname, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, RestartSnapNum, "subhalo_ids",
		      RestartSnapNum, filenr);
	      if(!(fd = fopen(fname, "r")))
		{
		  printf("can't read file `%s`\n", fname);
		  endrun(1184132);
		}

	      printf("reading '%s'\n", fname);
	      fflush(stdout);

	      my_fread(&ngroups, sizeof(int), 1, fd);
	      my_fread(&TotNgroups, sizeof(int), 1, fd);
	      my_fread(&nids, sizeof(int), 1, fd);
	      my_fread(&TotNids, sizeof(int64_t), 1, fd);
	      my_fread(&Nfiles, sizeof(int), 1, fd);
	      my_fread(&nid_previous, sizeof(int), 1, fd);	/* this is the number of IDs in previous files */

	      /* let's now fix nid_previous in case there has been a 32-bit overflow... */
	      for(i = 0, nid_previous = 0; i < filenr; i++)
		nid_previous += NumIdsPerFile[i];


	      struct id_list *tmpID_list = mymalloc("tmpID_list", nids * sizeof(struct id_list));

	      ids = mymalloc("ids", nids * sizeof(MyIDType));

	      my_fread(ids, sizeof(MyIDType), nids, fd);

	      for(i = 0; i < nids; i++)
		{
		  tmpID_list[i].ID = ids[i];
		  tmpID_list[i].GrNr = nid_previous + i;
		}

	      myfree(ids);

	      fclose(fd);

	      target = 0;
	      stored = 0;
	      while(nids > 0)
		{
		  while(nids_to_get[target] == 0)
		    target++;

		  if(nids > nids_to_get[target])
		    nsend = nids_to_get[target];
		  else
		    nsend = nids;

		  if(target == 0)
		    memcpy(&ID_list[nids_obtained[target]], &tmpID_list[stored],
			   nsend * sizeof(struct id_list));
		  else
		    {
		      MPI_Send(&nsend, 1, MPI_INT, target, TAG_HEADER, MPI_COMM_WORLD);
		      MPI_Send(&tmpID_list[stored], nsend * sizeof(struct id_list), MPI_BYTE,
			       target, TAG_SPHDATA, MPI_COMM_WORLD);
		    }

		  nids_to_get[target] -= nsend;
		  nids_obtained[target] += nsend;
		  nids -= nsend;
		  stored += nsend;
		}

	      myfree(tmpID_list);
	    }
	}
      else
	{
	  while(nids_to_get[ThisTask])
	    {
	      MPI_Recv(&nsend, 1, MPI_INT, 0, TAG_HEADER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	      MPI_Recv(&ID_list[nids_obtained[ThisTask]], nsend * sizeof(struct id_list), MPI_BYTE,
		       0, TAG_SPHDATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	      nids_to_get[ThisTask] -= nsend;
	      nids_obtained[ThisTask] += nsend;
	    }
	}

      myfree(nids_obtained);
      myfree(nids_to_get);
    }
  else
    {
      /* read routine can constinue in parallel */

      nprocgroup = NTask / All.NumFilesWrittenInParallel;
      if((NTask % All.NumFilesWrittenInParallel))
	nprocgroup++;
      masterTask = (ThisTask / nprocgroup) * nprocgroup;
      for(groupTask = 0; groupTask < nprocgroup; groupTask++)
	{
	  if(ThisTask == (masterTask + groupTask))	/* ok, it's this processor's turn */
	    {
	      sprintf(fname, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, RestartSnapNum, "subhalo_ids",
		      RestartSnapNum, ThisTask);
	      if(!(fd = fopen(fname, "r")))
		{
		  printf("can't read file `%s`\n", fname);
		  endrun(1184132);
		}

	      printf("reading '%s'\n", fname);
	      fflush(stdout);

	      my_fread(&Ngroups, sizeof(int), 1, fd);
	      my_fread(&TotNgroups, sizeof(int), 1, fd);
	      my_fread(&Nids, sizeof(int), 1, fd);
	      my_fread(&TotNids, sizeof(int64_t), 1, fd);
	      my_fread(&Nfiles, sizeof(int), 1, fd);
	      my_fread(&nid_previous, sizeof(unsigned int), 1, fd);	/* this is the number of IDs in previous files */

	      /* let's now fix nid_previous in case there has been a 32-bit overflow... */
	      for(i = 0, nid_previous = 0; i < ThisTask; i++)
		nid_previous += NumIdsPerFile[i];

	      ID_list = mymalloc("ID_list", Nids * sizeof(struct id_list));
	      ids = mymalloc("ids", Nids * sizeof(MyIDType));

	      my_fread(ids, sizeof(MyIDType), Nids, fd);

	      for(i = 0; i < Nids; i++)
		{
		  ID_list[i].ID = ids[i];
		  ID_list[i].GrNr = nid_previous + i;
		}

	      myfree(ids);

	      fclose(fd);
	    }

	  MPI_Barrier(MPI_COMM_WORLD);	/* wait inside the group */
	}
    }

  t1 = second();
  if(ThisTask == 0)
    printf("reading  took %g sec\n", timediff(t0, t1));



  t0 = second();

  qsort(P, NumPart, sizeof(struct particle_data), io_compare_P_ID);
  qsort(ID_list, Nids, sizeof(struct id_list), subfind_reshuffle_compare_ID_list_ID);

  for(i = 0; i < NumPart; i++)
    P[i].GrNr = TotNids + 1;	/* will mark particles that are not in any group */

  t1 = second();
  if(ThisTask == 0)
    printf("sorting took %g sec\n", timediff(t0, t1));



  list_of_nids = mymalloc("list_of_nids", NTask * sizeof(int));
  MPI_Allgather(&Nids, 1, MPI_INT, list_of_nids, 1, MPI_INT, MPI_COMM_WORLD);

  static struct id_list *recv_ID_list;

  t0 = second();

  int matches = 0;

  /* exchange  data */
  for(ngrp = 0; ngrp < (1 << PTask); ngrp++)
    {
      sendTask = ThisTask;
      recvTask = ThisTask ^ ngrp;

      if(recvTask < NTask)
	{
	  if(list_of_nids[sendTask] > 0 || list_of_nids[recvTask] > 0)
	    {
	      if(ngrp == 0)
		{
		  recv_ID_list = ID_list;
		}
	      else
		{
		  recv_ID_list = mymalloc("recv_ID_list", list_of_nids[recvTask] * sizeof(struct id_list));

		  /* get the particles */
		  MPI_Sendrecv(ID_list, Nids * sizeof(struct id_list), MPI_BYTE, recvTask, TAG_DENS_A,
			       recv_ID_list, list_of_nids[recvTask] * sizeof(struct id_list), MPI_BYTE,
			       recvTask, TAG_DENS_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

	      for(i = 0, j = 0; i < list_of_nids[recvTask]; i++)
		{
		  while(j < NumPart - 1 && P[j].ID < recv_ID_list[i].ID)
		    j++;

		  if(recv_ID_list[i].ID == P[j].ID)
		    {
		      P[j].GrNr = recv_ID_list[i].GrNr;
		      matches++;
		    }
		}

	      if(ngrp != 0)
		myfree(recv_ID_list);
	    }
	}
    }

  int64_t totlen;

  sumup_large_ints(1, &matches, &totlen);
  if(totlen != TotNids)
    endrun(543);

  t1 = second();
  if(ThisTask == 0)
    printf("assigning GrNr to P[] took %g sec\n", timediff(t0, t1));

  MPI_Barrier(MPI_COMM_WORLD);

  myfree(list_of_nids);

  myfree(ID_list);
  myfree(NumIdsPerFile);
}



int subfind_reshuffle_compare_ID_list_ID(const void *a, const void *b)
{
  if(((struct id_list *) a)->ID < ((struct id_list *) b)->ID)
    return -1;

  if(((struct id_list *) a)->ID > ((struct id_list *) b)->ID)
    return +1;

  return 0;
}







void read_hsml_table_voronoi(void)
{
  int i, dummy, nhsml;
  int64_t ntot;
  char fname[1000];
  FILE *fd;

  for(i = 0, NfilesVoronoi = 1; i < NfilesVoronoi; i++)
    {
      sprintf(fname, "%s/hsmldir_%03d/%s_%03d.%d", All.OutputDir, RestartSnapNum, "voronoi_hsml",
	      RestartSnapNum, i);
      if(!(fd = fopen(fname, "r")))
	{
	  printf("can't open file `%s`\n", fname);
	  endrun(113839);
	}

      my_fread(&nhsml, sizeof(int), 1, fd);
      my_fread(&dummy, sizeof(int), 1, fd);
      my_fread(&ntot, sizeof(int64_t), 1, fd);
      my_fread(&NfilesVoronoi, sizeof(int), 1, fd);

      if(i == 0)
	NumPartPerFileVoronoi =
	  mymalloc("	NumPartPerFileVoronoi", (NfilesVoronoi + 1) * sizeof(int64_t));

      NumPartPerFileVoronoi[i] = nhsml;
    }

  int64_t n, sum;

  for(i = 0, sum = 0; i < NfilesVoronoi; i++)
    {
      n = NumPartPerFileVoronoi[i];

      NumPartPerFileVoronoi[i] = sum;

      sum += n;
    }

  NumPartPerFileVoronoi[NfilesVoronoi] = sum;

  table_read_voronoi = 1;
}


void get_hsml_file_voronoi(int64_t nskip, int count, int *filenr, int *n_to_read, int *n_to_skip)
{
  int i;

  for(i = 0; i < NfilesVoronoi; i++)
    if(nskip >= NumPartPerFileVoronoi[i] && nskip < NumPartPerFileVoronoi[i + 1])
      break;

  if(i >= NfilesVoronoi)
    endrun(1231239);

  *filenr = i;
  *n_to_skip = (int) (nskip - NumPartPerFileVoronoi[i]);

  int nrest = (NumPartPerFileVoronoi[i + 1] - NumPartPerFileVoronoi[i] - *n_to_skip);

  if(count <= nrest)
    *n_to_read = count;
  else
    *n_to_read = nrest;
}

void read_hsml_files_voronoi(float *Values, int count, enum iofields blocknr, int64_t nskip)
{
  char fname[1000];
  FILE *fd;
  float *tmp;
  int i, n_to_read, n_to_skip, filenr, dummy, ntask, nhsml;
  int64_t ntot;

  while(count > 0)
    {
      if(table_read_voronoi == 0)
	read_hsml_table_voronoi();

      get_hsml_file_voronoi(nskip, count, &filenr, &n_to_read, &n_to_skip);

      sprintf(fname, "%s/hsmldir_%03d/%s_%03d.%d", All.OutputDir, RestartSnapNum, "voronoi_hsml",
	      RestartSnapNum, filenr);
      if(!(fd = fopen(fname, "r")))
	{
	  printf("can't open file `%s`\n", fname);
	  endrun(11383);
	}

      my_fread(&nhsml, sizeof(int), 1, fd);
      my_fread(&dummy, sizeof(int), 1, fd);
      my_fread(&ntot, sizeof(int64_t), 1, fd);
      my_fread(&ntask, sizeof(int), 1, fd);

      if(blocknr == IO_DMDENSITY_V)
	fseek(fd, nhsml * sizeof(float), SEEK_CUR);	/* skip hsml */

      fseek(fd, n_to_skip * sizeof(float), SEEK_CUR);

      tmp = mymalloc("tmp", n_to_read * sizeof(float));
      my_fread(tmp, sizeof(float), n_to_read, fd);

      for(i = 0; i < n_to_read; i++)
	Values[i] = tmp[i];

      myfree(tmp);

      fclose(fd);

      Values += n_to_read;
      count -= n_to_read;
      nskip += n_to_read;
    }
}


#endif
