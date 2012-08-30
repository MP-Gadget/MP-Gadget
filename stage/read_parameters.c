#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "allvars.h"
#include "proto.h"


#define DOUBLE 1
#define STRING 2
#define INT 3
#define MAXTAGS 300


void read_parameter_file(char *fname)
{
  FILE *fd;
  char buf[400], buf1[400], buf2[400], buf3[400];
  int i, j, nt = 0;
  int id[MAXTAGS];
  void *addr[MAXTAGS];
  char tag[MAXTAGS][50];
  int errorFlag = 0;

  strcpy(tag[nt], "OutputDir");
  addr[nt] = OutputDir;
  id[nt++] = STRING;


  strcpy(tag[nt], "DesLinkNgb");
  addr[nt] = &DesLinkNgb;
  id[nt++] = INT;


  if((fd = fopen(fname, "r")))
    {
      while(!feof(fd))
	{
	  *buf = 0;
	  fgets(buf, 200, fd);
	  if(sscanf(buf, "%s%s%s", buf1, buf2, buf3) < 2)
	    continue;

	  if(buf1[0] == '%')
	    continue;

	  for(i = 0, j = -1; i < nt; i++)
	    if(strcmp(buf1, tag[i]) == 0)
	      {
		j = i;
		tag[i][0] = 0;
		break;
	      }

	  if(j >= 0)
	    {
	      switch (id[j])
		{
		case DOUBLE:
		  *((double *) addr[j]) = atof(buf2);
		  break;
		case STRING:
		  strcpy(addr[j], buf2);
		  break;
		case INT:
		  *((int *) addr[j]) = atoi(buf2);
		  break;
		}
	    }
	  else
	    {
	      printf("Error in file %s:   Tag '%s' not allowed or multiple defined.\n", fname, buf1);
	      errorFlag = 1;
	    }
	}
      fclose(fd);

      i = strlen(OutputDir);
      if(i > 0)
	if(OutputDir[i - 1] != '/')
	  strcat(OutputDir, "/");
    }
  else
    {
      printf("Parameter file %s not found.\n", fname);
      errorFlag = 1;
    }


  for(i = 0; i < nt; i++)
    {
      if(*tag[i])
	{
	  printf("Error. I miss a value for tag '%s' in parameter file '%s'.\n", tag[i], fname);
	  errorFlag = 1;
	}
    }

  if(errorFlag)
    exit(1);

  printf("Simulation parameterfile read.\n");
}



size_t my_fwrite(void *ptr, size_t size, size_t nmemb, FILE * stream)
{
  size_t nwritten;

  if(size * nmemb > 0)
    {
      if((nwritten = fwrite(ptr, size, nmemb, stream)) != nmemb)
	{
	  printf("I/O error (fwrite) has occured: %s\n", strerror(errno));
	  fflush(stdout);
	  endrun(777);
	}
    }
  else
    nwritten = 0;

  return nwritten;
}


/*! This catches I/O errors occuring for fread(). In this case we
 *  better stop.
 */
size_t my_fread(void *ptr, size_t size, size_t nmemb, FILE * stream)
{
  size_t nread;

  if((nread = fread(ptr, size, nmemb, stream)) != nmemb)
    {
      if(feof(stream))
	printf("I/O error (fread) has occured: end of file\n");
      else
	printf("I/O error (fread) has occured: %s\n", strerror(errno));
      fflush(stdout);
      endrun(778);
    }
  return nread;
}


void endrun(int nr)
{
  printf("endrun called with %d\n", nr);
  exit(1);
}
