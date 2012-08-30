#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "allvars.h"
#include "proto.h"



#define MASK ((((long long)1)<< 32)-1)

int NTask;
int *List_NumPartGroup;
int TotGroupLen;
int Num, GrNr;

long long **Head, **Next, **Tail;
int **Len;

int LocalLen;
int *count_cand, max_candidates;

struct cand_dat
{
  long long head;
  long long rank;
  int len;
  int nsub;
  int subnr, parent;
  int bound_length;
}
**candidates;

struct sort_density_data
{
  MyFloat density;
  int ngbcount;
  long long index;		/* this will store the task in the upper word */
  long long ngb_index1, ngb_index2;
}
**sd;


int main(int argc, char **argv)
{
  int i, j;

  if(argc != 4)
    {
      fprintf(stderr, "\n  usage: Stage <parameterfile>  <SnapNum> <GrNr>\n\n");
      fprintf(stderr, "   <parameterfile>\n");
      fprintf(stderr, "   <Num>\n");
      fprintf(stderr, "   <GrNr>\n");
      exit(1);
    }

  read_parameter_file(argv[1]);

  Num = atoi(argv[2]);
  GrNr = atoi(argv[3]);

  load_sd(Num);

  Head = mymalloc(NTask * sizeof(long long *));
  Next = mymalloc(NTask * sizeof(long long *));
  Tail = mymalloc(NTask * sizeof(long long *));
  Len = mymalloc(NTask * sizeof(int *));


  for(i = 0; i < NTask; i++)
    {
      Head[i] = mymalloc(List_NumPartGroup[i] * sizeof(long long));
      Next[i] = mymalloc(List_NumPartGroup[i] * sizeof(long long));
      Tail[i] = mymalloc(List_NumPartGroup[i] * sizeof(long long));
      Len[i] = mymalloc(List_NumPartGroup[i] * sizeof(int));

      for(j = 0; j < List_NumPartGroup[i]; j++)
	{
	  Head[i][j] = Next[i][j] = Tail[i][j] = -1;
	  Len[i][j] = 0;
	}
    }

  count_cand = mymalloc(NTask * sizeof(int));
  candidates = mymalloc(NTask * sizeof(struct cand_dat *));
  
  max_candidates = (TotGroupLen / NTask / 50);
  
  for(i = 0; i < NTask; i++)
    {
      count_cand[i] = 0;
      candidates[i] = mymalloc(max_candidates * sizeof(struct cand_dat));
    }

  subfind_col_find_candidates(TotGroupLen);

  write_candidates(Num);

  return 0;
}

void load_sd(int num)
{
  FILE *fd;
  char fname[500];
  int task;

  /* start reading of sd field */

  for(task = 0, NTask = 1; task < NTask; task++)
    {
      sprintf(fname, "%s/aux_%03d/%s_%03d_%d.%d", OutputDir, num, "aux_col_sd", num, GrNr, task);
      if(!(fd = fopen(fname, "r")))
	{
	  printf("can't read file `%s`\n", fname);
	  endrun(118312);
	}

      printf("reading '%s'\n", fname);

      my_fread(&NTask, sizeof(int), 1, fd);

      if(task == 0)
	{
	  List_NumPartGroup = mymalloc(NTask * sizeof(int));
	  sd = mymalloc(NTask * sizeof(struct sort_density_data *));
	}

      my_fread(&List_NumPartGroup[task], sizeof(int), 1, fd);
      my_fread(&TotGroupLen, sizeof(int), 1, fd);
      
      printf("TotGroupLen=%d\n", TotGroupLen);

      sd[task] = mymalloc(List_NumPartGroup[task] * sizeof(struct sort_density_data));

      my_fread(sd[task], List_NumPartGroup[task], sizeof(struct sort_density_data), fd);

      fclose(fd);
    }
}

void write_candidates(int num)
{
  FILE *fd;
  char fname[500];
  int task;

  for(task = 0; task < NTask; task++)
    {
      sprintf(fname, "%s/aux_%03d/%s_%03d_%d.%d", OutputDir, num, "aux_col_list", num, GrNr, task);

      if(!(fd = fopen(fname, "w")))
	{
	  printf("can't write file `%s`\n", fname);
	  endrun(118319);
	}

      printf("writing '%s'\n", fname);

      my_fwrite(&List_NumPartGroup[task], sizeof(int), 1, fd);

      my_fwrite(Head[task], List_NumPartGroup[task], sizeof(long long), fd);
      my_fwrite(Next[task], List_NumPartGroup[task], sizeof(long long), fd);
      my_fwrite(Tail[task], List_NumPartGroup[task], sizeof(long long), fd);
      my_fwrite(Len[task], List_NumPartGroup[task], sizeof(int), fd);

      my_fwrite(&count_cand[task], 1, sizeof(int), fd);
      my_fwrite(candidates[task], count_cand[task], sizeof(struct cand_dat), fd);

      fclose(fd);
    }
}



void subfind_col_find_candidates(int totgrouplen)
{
  int ngbcount, retcode, len_attach;
  int i, k, len, task;
  long long prev, tail, tail_attach, tmp, next, index;
  long long p, ss, head, head_attach, ngb_index1, ngb_index2, rank;

  /* now find the subhalo candidates by building up link lists from high density to low density */

  for(task = 0; task < NTask; task++)
    {
      printf("begin task=%d\n", task);

      for(k = 0; k < List_NumPartGroup[task]; k++)
	{
	  /*	  if((k % 1000) == 0)
	    printf("k=%d|%d\n", k,  List_NumPartGroup[task]);
	  */

	  ngbcount = sd[task][k].ngbcount;
	  ngb_index1 = sd[task][k].ngb_index1;
	  ngb_index2 = sd[task][k].ngb_index2;

	  switch (ngbcount)	/* treat the different possible cases */
	    {
	    case 0:		/* this appears to be a lonely maximum -> new group */
	      subfind_distlinklist_set_all(sd[task][k].index, sd[task][k].index, sd[task][k].index, 1, -1);
	      break;

	    case 1:		/* the particle is attached to exactly one group */
	      head = subfind_distlinklist_get_head(ngb_index1);

	      if(head == -1)
		{
		  printf("We have a problem!  head=%d/%d for k=%d on task=%d\n",
			 (int) (head >> 32), (int) head, k, task);
		  fflush(stdout);
		  endrun(13123);
		}

	      retcode = subfind_distlinklist_get_tail_set_tail_increaselen(head, &tail, sd[task][k].index);

	      if(!(retcode & 1))
		subfind_distlinklist_set_headandnext(sd[task][k].index, head, -1);
	      if(!(retcode & 2))
		subfind_distlinklist_set_next(tail, sd[task][k].index);
	      break;

	    case 2:		/* the particle merges two groups together */
	      if((ngb_index1 >> 32) == (ngb_index2 >> 32))
		{
		  subfind_distlinklist_get_two_heads(ngb_index1, ngb_index2, &head, &head_attach);
		}
	      else
		{
		  head = subfind_distlinklist_get_head(ngb_index1);
		  head_attach = subfind_distlinklist_get_head(ngb_index2);
		}

	      if(head == -1 || head_attach == -1)
		{
		  printf("We have a problem!  head=%d/%d head_attach=%d/%d for k=%d on task=%d\n",
			 (int) (head >> 32), (int) head,
			 (int) (head_attach >> 32), (int) head_attach, k, task);
		  fflush(stdout);
		  endrun(13123);
		}

	      if(head != head_attach)
		{
		  subfind_distlinklist_get_tailandlen(head, &tail, &len);
		  subfind_distlinklist_get_tailandlen(head_attach, &tail_attach, &len_attach);

		  if(len_attach > len)	/* other group is longer, swap them */
		    {
		      tmp = head;
		      head = head_attach;
		      head_attach = tmp;
		      tmp = tail;
		      tail = tail_attach;
		      tail_attach = tmp;
		      tmp = len;
		      len = len_attach;
		      len_attach = tmp;
		    }

		  /* only in case the attached group is long enough we bother to register it 
		     as a subhalo candidate */

		  if(len_attach >= DesLinkNgb)
		    {
		      if(count_cand[task] < max_candidates)
			{
			  candidates[task][count_cand[task]].len = len_attach;
			  candidates[task][count_cand[task]].head = head_attach;
			  count_cand[task]++;
			}
		      else
			endrun(87);
		    }

		  /* now join the two groups */
		  subfind_distlinklist_set_tailandlen(head, tail_attach, len + len_attach);
		  subfind_distlinklist_set_next(tail, head_attach);

		  ss = head_attach;
		  do
		    {
		      ss = subfind_distlinklist_set_head_get_next(ss, head);
		    }
		  while(ss >= 0);
		}

	      /* finally, attach the particle to 'head' */
	      retcode = subfind_distlinklist_get_tail_set_tail_increaselen(head, &tail, sd[task][k].index);

	      if(!(retcode & 1))
		subfind_distlinklist_set_headandnext(sd[task][k].index, head, -1);
	      if(!(retcode & 2))
		subfind_distlinklist_set_next(tail, sd[task][k].index);
	      break;
	    }
	}
    }

  printf("identification of primary candidates finished\n");

  /* add the full thing as a subhalo candidate */

  for(task = 0, head = -1, prev = -1; task < NTask; task++)
    {
      for(i = 0; i < List_NumPartGroup[task]; i++)
	{
	  index = (((long long) task) << 32) + i;

	  if(Head[task][i] == index)
	    {
	      subfind_distlinklist_get_tailandlen(Head[task][i], &tail, &len);
	      next = subfind_distlinklist_get_next(tail);
	      if(next == -1)
		{
		  if(prev < 0)
		    head = index;

		  if(prev >= 0)
		    subfind_distlinklist_set_next(prev, index);

		  prev = tail;
		}
	    }
	}
    }

  if(count_cand[NTask - 1] < max_candidates)
    {
      candidates[NTask - 1][count_cand[NTask - 1]].len = totgrouplen;
      candidates[NTask - 1][count_cand[NTask - 1]].head = head;
      count_cand[NTask - 1]++;
    }
  else
    endrun(123123);

  printf("adding background as candidate finished\n");


  /* go through the whole chain once to establish a rank order. For the rank we use Len[] */

  task = (head >> 32);

  p = head;
  rank = 0;

  while(p >= 0)
    {
      p = subfind_distlinklist_setrank_and_get_next(p, &rank);
    }

  /* for each candidate, we now pull out the rank of its head */
  for(task = 0; task < NTask; task++)
    {
      for(k = 0; k < count_cand[task]; k++)
	candidates[task][k].rank = subfind_distlinklist_get_rank(candidates[task][k].head);
    }

  printf("establishing of rank order finished\n");

  if(((int) rank) != totgrouplen)
    {
      printf("mismatch\n");
      endrun(0);
    }

}





long long subfind_distlinklist_setrank_and_get_next(long long index, long long *rank)
{
  int task, i;
  long long next;

  task = (index >> 32);
  i = (index & MASK);

  Len[task][i] = *rank;
  *rank = *rank + 1;
  next = Next[task][i];

  return next;
}


long long subfind_distlinklist_set_head_get_next(long long index, long long head)
{
  int task, i;
  long long next;

  task = (index >> 32);
  i = (index & MASK);

  Head[task][i] = head;
  next = Next[task][i];

  return next;
}




void subfind_distlinklist_set_next(long long index, long long next)
{
  int task, i;

  task = (index >> 32);
  i = (index & MASK);

  Next[task][i] = next;
}


long long subfind_distlinklist_get_next(long long index)
{
  int task, i;
  long long next;

  task = (index >> 32);
  i = (index & MASK);

  next = Next[task][i];

  return next;
}

long long subfind_distlinklist_get_rank(long long index)
{
  int task, i;
  long long rank;

  task = (index >> 32);
  i = (index & MASK);

  rank = Len[task][i];

  return rank;
}



long long subfind_distlinklist_get_head(long long index)
{
  int task, i;
  long long head;

  task = (index >> 32);
  i = (index & MASK);

  head = Head[task][i];

  return head;
}

void subfind_distlinklist_get_two_heads(long long ngb_index1, long long ngb_index2,
					long long *head, long long *head_attach)
{
  int task, i1, i2;

  task = (ngb_index1 >> 32);
  i1 = (ngb_index1 & MASK);
  i2 = (ngb_index2 & MASK);

  *head = Head[task][i1];
  *head_attach = Head[task][i2];
}



void subfind_distlinklist_set_headandnext(long long index, long long head, long long next)
{
  int task, i;

  task = (index >> 32);
  i = (index & MASK);

  Head[task][i] = head;
  Next[task][i] = next;
}

int subfind_distlinklist_get_tail_set_tail_increaselen(long long index, long long *tail, long long newtail)
{
  int task, i, task_newtail, i_newtail, task_oldtail, i_oldtail, retcode;
  long long oldtail;

  task = (index >> 32);
  i = (index & MASK);

  retcode = 0;

  oldtail = Tail[task][i];
  Tail[task][i] = newtail;
  Len[task][i]++;
  *tail = oldtail;

  task_newtail = (newtail >> 32);
  i_newtail = (newtail & MASK);
  Head[task_newtail][i_newtail] = index;
  Next[task_newtail][i_newtail] = -1;
  retcode |= 1;

  task_oldtail = (oldtail >> 32);
  i_oldtail = (oldtail & MASK);
  Next[task_oldtail][i_oldtail] = newtail;
  retcode |= 2;

  return retcode;
}



void subfind_distlinklist_set_tailandlen(long long index, long long tail, int len)
{
  int task, i;

  task = (index >> 32);
  i = (index & MASK);

  Tail[task][i] = tail;
  Len[task][i] = len;
}




void subfind_distlinklist_get_tailandlen(long long index, long long *tail, int *len)
{
  int task, i;

  task = (index >> 32);
  i = (index & MASK);

  *tail = Tail[task][i];
  *len = Len[task][i];
}


void subfind_distlinklist_set_all(long long index, long long head, long long tail, int len, long long next)
{
  int task, i;

  task = (index >> 32);
  i = (index & MASK);

  Head[task][i] = head;
  Tail[task][i] = tail;
  Len[task][i] = len;
  Next[task][i] = next;
}
