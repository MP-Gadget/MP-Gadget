#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <signal.h>
#include <gsl/gsl_rng.h>

#include "allvars.h"
#include "proto.h"



static struct aux_data
{
  MyIDType ID;
  MyIDType GrNr;
  int OriginTask;
  int OriginIndex;
  int FinalTask;
}
 *Aux;


static int compare_Aux_GrNr_ID(const void *a, const void *b)
{
  if(((struct aux_data *) a)->GrNr < (((struct aux_data *) b)->GrNr))
    return -1;

  if(((struct aux_data *) a)->GrNr > (((struct aux_data *) b)->GrNr))
    return +1;

  if(((struct aux_data *) a)->ID < (((struct aux_data *) b)->ID))
    return -1;

  if(((struct aux_data *) a)->ID > (((struct aux_data *) b)->ID))
    return +1;

  return 0;
}

static int compare_Aux_OriginTask_OriginIndex(const void *a, const void *b)
{
  if(((struct aux_data *) a)->OriginTask < (((struct aux_data *) b)->OriginTask))
    return -1;

  if(((struct aux_data *) a)->OriginTask > (((struct aux_data *) b)->OriginTask))
    return +1;

  if(((struct aux_data *) a)->OriginIndex < (((struct aux_data *) b)->OriginIndex))
    return -1;

  if(((struct aux_data *) a)->OriginIndex > (((struct aux_data *) b)->OriginIndex))
    return +1;

  return 0;
}

