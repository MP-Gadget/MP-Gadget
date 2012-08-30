#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "allvars.h"
#include "proto.h"


/* This is a special-purpose read-in routine for initial conditions
 * produced with Bepi Tormen's initial conditions generator ZIC.
 * Using this routine requires that you know pretty well what you're
 * doing...
 * Be aware that there are unit conversion factor `massfac', `posfac',
 * and `velfac', that have to be set appropriately.
 * Also note that there is a boundary for the intermediate resolution
 * zone set by hand below (to a value of 24000.0 in this example).
 */
void read_ic_cluster(char *fname)
{
#define BLOCKSIZE 10000
  FILE *fd = 0;
  int i, j, k, n, files, sumnumpart;
  double sqr_a;
  float *fp, dummy[BLOCKSIZE][3];
  int pc, id, pc_here;
  int type, pr, left, groupsize;
  MPI_Status status;
  int npart;
  float a0;
  char buf[100];
  int n_for_this_task, n_in_file;
  double massfac = 0, posfac = 0, velfac = 0;
  double r2;
  int blocks, subblock;
  int nhr = 0, nlr = 0;
  int counttype3, counttype2, counttype_disk, counttype_bulge;
  int nhr_blocks, nlr_blocks;
  float pmhr;

#ifdef T3E
  short int dummy4byte;		/* Note: int has 8 Bytes on the T3E ! */
#else
  int dummy4byte;
#endif

/* Felix: start */

  char head_c[120];		/* one element of the string head array */
  double head_r[64];		/* the numerical head array */
  float rcut = 0, xcm = 0, ycm = 0, zcm = 0;

/* Felix: end */

#define SKIP my_fread(&dummy4byte, sizeof(dummy4byte), 1, fd);

  /* Below, Bepi's new format is assumed !!!!!! */
  /* for the old one, the HR particle mass 'pmhr' has to be set
   * by hand! 
   */

  if(ThisTask == 0)
    {
      for(i = 0; i < 5; i++)
	{
	  All.MassTable[i] = 0;
	}


      if(!(fd = fopen(fname, "r")))
	{
	  printf("can't open file `%s'.\n", fname);
	  endrun(123);
	}

      printf("READING FILE '%s'....\n", fname);
      fflush(stdout);
/* Bepi: start */

/* read string header */
      SKIP;
      for(j = 0; j < 16; j++)
	{
	  my_fread(&head_c, 120, 1, fd);
	  printf("j, head_c[j]= %i  %s\n", j, head_c);
	}
      SKIP;

/* read numeric header */
      SKIP;
      my_fread(&head_r, sizeof(double), 64, fd);
      SKIP;

      for(j = 0; j < 64; j++)
	printf("j, head_r[j]= %i  %f\n", j, head_r[j]);

/* assign numeric header to variables */
      nhr = (int) head_r[0];
      nlr = (int) head_r[1];
      a0 = 1. / (1. + head_r[2]);
      pmhr = head_r[3];
      posfac = 1000. * head_r[4] * 3.085678e21 / All.UnitLength_in_cm;
      massfac = head_r[6] / 1.e10 * 1.989e43 / All.UnitMass_in_g;
      velfac = 100.0 * head_r[4] * 1e5 / All.UnitVelocity_in_cm_per_s;
      rcut = 1. / head_r[17] * posfac;

      printf(">>>>>>>>>>>>> Bepi: from Zic header I read: \n");
      printf("massfac = %g\n", massfac);
      printf("posfac  = %g\n", posfac);
      printf("velfac  = %g\n", velfac);
      printf("rcut    = %g\n", rcut);
      printf("nhr     = %i\n", nhr);
      printf("nlr     = %i\n", nlr);
      printf(">>>>>>>>>>>>> end Bepi \n");

/* Bepi: end */

/* Felix start 
 *
 * For twolevel resimulations the HR box is NOT in the center of the simulation
 * anymore. Therefore the automatic second shell particle searching with rcut
 * is not valid in this case. 
 *
 * xcm are set outside the ThisTask loop and are 
 */
      if(head_r[33] > 0.5)
	{
	  printf("##########################################################\n");
	  printf("##########################################################\n");
	  printf(" This run is a twolevel run! The HR box is NOT centered on\n");
	  printf(" zero anymore. Have to use values SET BY HAND IN THE CODE!\n");
	  printf(" For later runs: add center of HR box into zic.dat file!!!\n");
	  printf("##########################################################\n");
	  printf("##########################################################\n");

	  xcm = -0.1546277106E-01 * posfac;
	  ycm = -0.2029071748E-01 * posfac;
	  zcm = 0.2206191421E-02 * posfac;
	  /* rboxtwo */
	  rcut = 1. / 45.41310883 * posfac;

	  printf("##########################################################\n");
	  printf("##########################################################\n");
	  printf(" xcm = %e \n ycm = %e \n zcm = %e \n rcut = %e\n", xcm, ycm, zcm, rcut);
	  printf("##########################################################\n");
	  printf("##########################################################\n");
	}
      else
	{
#ifdef PERIODIC
	  xcm = All.BoxSize / 2;
	  ycm = All.BoxSize / 2;
	  zcm = All.BoxSize / 2;
#else
	  xcm = 0.0;
	  ycm = 0.0;
	  zcm = 0.0;
#endif
	}


      printf("##########################################################\n");
      printf("##########################################################\n");
      printf("Blocksize set to 10000000 for HR and 1000000 for LR !!!!!!\n");
      printf("##########################################################\n");
      printf("##########################################################\n");




/* Felix end */


      All.MassTable[1] = pmhr * massfac;	/* high-res particles */


      printf("All.MassTable[1]=%g\n", All.MassTable[1]);

      printf("file contains %d HR and %d LR particles.\n", nhr, nlr);

      All.TotN_gas = 0;
      All.TotNumPart = nhr + nlr;

      printf("\nN_sph: %d\nN_halo: %d\nN_disk: %d\n\n", 0, nhr, nlr);




      fclose(fd);
    }


  MPI_Bcast(&posfac, sizeof(posfac), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&velfac, sizeof(velfac), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&massfac, sizeof(massfac), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rcut, sizeof(rcut), MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&xcm, sizeof(xcm), MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&ycm, sizeof(ycm), MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&zcm, sizeof(zcm), MPI_FLOAT, 0, MPI_COMM_WORLD);


  MPI_Bcast(&All, sizeof(struct global_data_all_processes), MPI_BYTE, 0, MPI_COMM_WORLD);

  All.MaxPart = (int) (All.PartAllocFactor * (All.TotNumPart / NTask));	/* sets the maximum number of particles that may 
									   reside on a processor */

  allocate_memory();


  pc = 0;
  id = 1;


  NumPart = 0;



  for(files = 0; files <= 0; files++)	/* only one file here */
    {
      if(ThisTask == 0)
	{
	  sprintf(buf, "%s", fname);

	  if(!(fd = fopen(buf, "r")))
	    {
	      printf("can't open file `%s'.\n", buf);
	      endrun(123);
	    }

/* Bepi: start */

/* read string header */
	  SKIP;
	  for(j = 0; j < 16; j++)
	    {
	      my_fread(&head_c, 120, 1, fd);
	    }
	  SKIP;

/* read numeric header */
	  SKIP;
	  my_fread(&head_r, sizeof(double), 64, fd);
	  SKIP;

	  nhr_blocks = nhr / 1000000 + 1;
	  nlr_blocks = nlr / 1000000 + 1;
	}


      MPI_Bcast(&nhr_blocks, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&nlr_blocks, 1, MPI_INT, 0, MPI_COMM_WORLD);


      for(blocks = 0; blocks < (nhr_blocks + nlr_blocks); blocks++)
	{
	  if(ThisTask == 0)
	    {
	      SKIP;
	      my_fread(&npart, sizeof(int), 1, fd);
	      SKIP;
	      n_in_file = npart;

	      if(blocks < nhr_blocks)
		type = 1;
	      else
		type = 2;
	    }

	  MPI_Bcast(&n_in_file, 1, MPI_INT, 0, MPI_COMM_WORLD);	/* receive type of particle and total number */
	  MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);

	  n_for_this_task = n_in_file / NTask;
	  if(ThisTask < (n_in_file % NTask))
	    n_for_this_task++;



	  for(subblock = 0; subblock < 3; subblock++)
	    {
	      if(type == 1 && subblock == 2)
		continue;	/* HR part's have no mass array */

	      if(ThisTask == 0)
		{
		  SKIP;
		}

	      pc_here = pc;

	      for(pr = 0; pr < NTask; pr++)	/* go through all processes, note: pr is the receiving process */
		{
		  if(ThisTask == 0 || ThisTask == pr)
		    {
		      n = n_for_this_task;	/* number of particles for this process */

		      if(ThisTask == 0 && pr > 0)
			MPI_Recv(&n, 1, MPI_INT, pr, TAG_N, MPI_COMM_WORLD, &status);

		      if(ThisTask == pr && pr > 0)
			MPI_Send(&n, 1, MPI_INT, 0, TAG_N, MPI_COMM_WORLD);


		      left = n;

		      while(left > 0)
			{
			  if(left > BLOCKSIZE)	/* restrict transmission size to buffer length */
			    groupsize = BLOCKSIZE;
			  else
			    groupsize = left;

			  if(ThisTask == 0)
			    {
			      if(subblock < 2)
				my_fread(&dummy[0][0], sizeof(float), 3 * groupsize, fd);
			      else
				my_fread(&dummy[0][0], sizeof(float), groupsize, fd);
			    }

			  if(ThisTask == 0 && pr != 0)
			    {
			      if(subblock < 2)
				MPI_Ssend(&dummy[0][0], 3 * groupsize, MPI_FLOAT, pr, TAG_PDATA,
					  MPI_COMM_WORLD);
			      else
				MPI_Ssend(&dummy[0][0], groupsize, MPI_FLOAT, pr, TAG_PDATA, MPI_COMM_WORLD);
			    }

			  if(ThisTask != 0 && pr != 0)
			    {
			      if(subblock < 2)
				MPI_Recv(&dummy[0][0], 3 * groupsize, MPI_FLOAT, 0, TAG_PDATA, MPI_COMM_WORLD,
					 &status);
			      else
				MPI_Recv(&dummy[0][0], groupsize, MPI_FLOAT, 0, TAG_PDATA, MPI_COMM_WORLD,
					 &status);
			    }

			  if(ThisTask == pr)
			    {
			      for(i = 0, fp = &dummy[0][0]; i < groupsize; i++)
				{
				  if(subblock == 0)
				    {
				      P[pc_here].Type = type;
				      P[pc_here].ID = id;	/* now set ID */
				      for(k = 0; k < 3; k++)
					P[pc_here].Pos[k] = dummy[i][k];
				      pc_here++;
				      id++;
				    }

				  if(subblock == 1)
				    {
				      for(k = 0; k < 3; k++)
					P[pc_here].Vel[k] = dummy[i][k];
				      pc_here++;
				    }
				  if(subblock == 2)
				    {
				      P[pc_here].Mass = fp[i];
				      pc_here++;
				    }
				}

			    }

			  left -= groupsize;
			}
		    }



		  MPI_Barrier(MPI_COMM_WORLD);

		  MPI_Bcast(&id, 1, MPI_INT, pr, MPI_COMM_WORLD);
		}

	      if(ThisTask == 0)
		SKIP;
	    }

	  pc += n_for_this_task;
	  NumPart += n_for_this_task;

	}


      if(ThisTask == 0)
	{
	  fclose(fd);
	}

    }


  MPI_Barrier(MPI_COMM_WORLD);
  if(ThisTask == 0)
    {
      printf("\nreading done.\n\n");
      fflush(stdout);
    }

#ifdef RESCALEVINI
  if(ThisTask == 0)
    {
      fprintf(stdout, "\nRescaling v_ini !\n\n");
      fflush(stdout);
    }
#endif




  /* now convert the units */

  sqr_a = sqrt(All.Time);

  counttype2 = counttype3 = 0;

  for(i = 0; i < NumPart; i++)
    {
      for(j = 0; j < 3; j++)
	{
	  P[i].Pos[j] = P[i].Pos[j] * posfac;	/* here in units of kpc/h */

	  P[i].Vel[j] = P[i].Vel[j] * velfac;	/* comoving velocity xdot on km/sec */
#ifdef RESCALEVINI
	  P[i].Vel[j] *= All.VelIniScale;	/* scaling v to use same IC's for 
						   different cosmologies */
#endif
	  P[i].Vel[j] *= sqr_a;	/* transform to velocity variable u */
	}

      if(P[i].Type == 1)
	{
	  P[i].Mass = All.MassTable[1];
	}
      else
	{
	  P[i].Mass *= massfac;

	  r2 = (P[i].Pos[0] - xcm) * (P[i].Pos[0] - xcm)
	    + (P[i].Pos[1] - ycm) * (P[i].Pos[1] - ycm) + (P[i].Pos[2] - zcm) * (P[i].Pos[2] - zcm);

	  if(sqrt(r2) > rcut)	/* boundary of inner LR zone */
	    {
	      P[i].Type = 3;
	      counttype3++;
	    }
	  else
	    counttype2++;
	}
    }


  printf("Task: %d has %d particles, and %d of type 3\n", ThisTask, NumPart, counttype3);

  MPI_Reduce(&counttype2, &counttype_disk, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&counttype3, &counttype_bulge, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&counttype_disk, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&counttype_bulge, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Allreduce(&NumPart, &sumnumpart, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  All.TotNumPart = sumnumpart;

  fflush(stdout);

  if(ThisTask == 0)
    {
      printf("particles loaded: %d \n\n", (int) All.TotNumPart);
      printf("particles of type 2: %d \n", counttype_disk);
      printf("particles of type 3: %d  (together %d)  \n\n", counttype_bulge,
	     counttype_bulge + counttype_disk);

      printf("\n");
      printf("Collisionless particles   :  %d\n", (int) (All.TotNumPart - All.TotN_gas));
      printf("Baryonic particles        :  %d\n", (int) All.TotN_gas);
      printf("                             ---------\n");
      printf("Total number of particles :  %d\n\n", (int) All.TotNumPart);
    }

#undef BLOCKSIZE
}
