#include <fftw3-mpi.h>
#include <stdio.h>
#include <gsl/gsl_rng.h>
#include <math.h>
#include "allvars.h"
#include "proto.h"

typedef double fftw_real;

static ptrdiff_t Local_nx, Local_x_start;
static fftw_plan Inverse_plan;
static fftw_real        *Disp, *Workspace;
static fftw_complex     *Cdata;
static int      *Slab_to_task;
static int      *Local_nx_table;

void displacement_fields(void)
{
  MPI_Request request;
  MPI_Status status;
  gsl_rng *random_generator;
  int i, j, k, ii, jj, kk, axes;
  int n;
  int sendTask, recvTask;
  double fac, vel_prefac;
  double kvec[3], kmag, kmag2, p_of_k;
  double delta, phase, ampl, hubble_a;
  double u, v, w;
  double f1, f2, f3, f4, f5, f6, f7, f8;
  double dis, maxdisp, max_disp_glob;
  unsigned int *seedtable;

#ifdef CORRECT_CIC
  double fx, fy, fz, ff, smth;
#endif

  if(ThisTask == 0)
    {
      printf("\nstart computing displacement fields...\n");
      fflush(stdout);
    }

  hubble_a =
    Hubble * sqrt(Omega / pow(InitTime, 3) + (1 - Omega - OmegaLambda) / pow(InitTime, 2) + OmegaLambda);

  vel_prefac = InitTime * hubble_a * F_Omega(InitTime);

  vel_prefac /= sqrt(InitTime);	/* converts to Gadget velocity */

  if(ThisTask == 0)
    printf("vel_prefac= %g  hubble_a=%g fom=%g \n", vel_prefac, hubble_a, F_Omega(InitTime));

  fac = pow(2 * PI / Box, 1.5);

  maxdisp = 0;

  random_generator = gsl_rng_alloc(gsl_rng_ranlxd1);

  gsl_rng_set(random_generator, Seed);

  if(!(seedtable = malloc(Nmesh * Nmesh * sizeof(unsigned int))))
    FatalError(4);

  for(i = 0; i < Nmesh / 2; i++)
    {
      for(j = 0; j < i; j++)
	seedtable[i * Nmesh + j] = 0x7fffffff * gsl_rng_uniform(random_generator);

      for(j = 0; j < i + 1; j++)
	seedtable[j * Nmesh + i] = 0x7fffffff * gsl_rng_uniform(random_generator);

      for(j = 0; j < i; j++)
	seedtable[(Nmesh - 1 - i) * Nmesh + j] = 0x7fffffff * gsl_rng_uniform(random_generator);

      for(j = 0; j < i + 1; j++)
	seedtable[(Nmesh - 1 - j) * Nmesh + i] = 0x7fffffff * gsl_rng_uniform(random_generator);

      for(j = 0; j < i; j++)
	seedtable[i * Nmesh + (Nmesh - 1 - j)] = 0x7fffffff * gsl_rng_uniform(random_generator);

      for(j = 0; j < i + 1; j++)
	seedtable[j * Nmesh + (Nmesh - 1 - i)] = 0x7fffffff * gsl_rng_uniform(random_generator);

      for(j = 0; j < i; j++)
	seedtable[(Nmesh - 1 - i) * Nmesh + (Nmesh - 1 - j)] = 0x7fffffff * gsl_rng_uniform(random_generator);

      for(j = 0; j < i + 1; j++)
	seedtable[(Nmesh - 1 - j) * Nmesh + (Nmesh - 1 - i)] = 0x7fffffff * gsl_rng_uniform(random_generator);
    }


#if defined(MULTICOMPONENTGLASSFILE) && defined(DIFFERENT_TRANSFER_FUNC)
  for(Type = MinType; Type <= MaxType; Type++)
#endif
    {
      for(axes = 0; axes < 3; axes++)
	{
	  if(ThisTask == 0)
	    {
	      printf("\nstarting axes=%d...\n", axes);
	      fflush(stdout);
	    }

	  /* first, clean the array */
	  for(i = 0; i < Local_nx; i++)
	    for(j = 0; j < Nmesh; j++)
	      for(k = 0; k <= Nmesh / 2; k++)
		{
		  Cdata[(i * Nmesh + j) * (Nmesh / 2 + 1) + k][0] = 0;
		  Cdata[(i * Nmesh + j) * (Nmesh / 2 + 1) + k][1] = 0;
		}

	  for(i = 0; i < Nmesh; i++)
	    {
	      ii = Nmesh - i;
	      if(ii == Nmesh)
		ii = 0;
	      if((i >= Local_x_start && i < (Local_x_start + Local_nx)) ||
		 (ii >= Local_x_start && ii < (Local_x_start + Local_nx)))
		{
		  for(j = 0; j < Nmesh; j++)
		    {
		      gsl_rng_set(random_generator, seedtable[i * Nmesh + j]);

		      for(k = 0; k < Nmesh / 2; k++)
			{
			  phase = gsl_rng_uniform(random_generator) * 2 * PI;
			  do
			    ampl = gsl_rng_uniform(random_generator);
			  while(ampl == 0);

			  if(i == Nmesh / 2 || j == Nmesh / 2 || k == Nmesh / 2)
			    continue;
			  if(i == 0 && j == 0 && k == 0)
			    continue;

			  if(i < Nmesh / 2)
			    kvec[0] = i * 2 * PI / Box;
			  else
			    kvec[0] = -(Nmesh - i) * 2 * PI / Box;

			  if(j < Nmesh / 2)
			    kvec[1] = j * 2 * PI / Box;
			  else
			    kvec[1] = -(Nmesh - j) * 2 * PI / Box;

			  if(k < Nmesh / 2)
			    kvec[2] = k * 2 * PI / Box;
			  else
			    kvec[2] = -(Nmesh - k) * 2 * PI / Box;

			  kmag2 = kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2];
			  kmag = sqrt(kmag2);

			  if(SphereMode == 1)
			    {
			      if(kmag * Box / (2 * PI) > Nsample / 2)	/* select a sphere in k-space */
				continue;
			    }
			  else
			    {
			      if(fabs(kvec[0]) * Box / (2 * PI) > Nsample / 2)
				continue;
			      if(fabs(kvec[1]) * Box / (2 * PI) > Nsample / 2)
				continue;
			      if(fabs(kvec[2]) * Box / (2 * PI) > Nsample / 2)
				continue;
			    }

			  p_of_k = PowerSpec(kmag);

			  p_of_k *= -log(ampl);

			  delta = fac * sqrt(p_of_k) / Dplus;	/* scale back to starting redshift */

              printf("%d %d %d %g %g\n", i, j, k, delta * cos(phase), delta*sin(phase));
#ifdef CORRECT_CIC
			  /* do deconvolution of CIC interpolation */
			  fx = fy = fz = 1;
			  if(kvec[0] != 0)
			    {
			      fx = (kvec[0] * Box / 2) / Nmesh;
			      fx = sin(fx) / fx;
			    }
			  if(kvec[1] != 0)
			    {
			      fy = (kvec[1] * Box / 2) / Nmesh;
			      fy = sin(fy) / fy;
			    }
			  if(kvec[2] != 0)
			    {
			      fz = (kvec[2] * Box / 2) / Nmesh;
			      fz = sin(fz) / fz;
			    }
			  ff = 1 / (fx * fy * fz);
			  smth = ff * ff;

			  delta *= smth;
			  /* end deconvolution */
#endif
			  if(k > 0)
			    {
			      if(i >= Local_x_start && i < (Local_x_start + Local_nx))
				{
				  Cdata[((i - Local_x_start) * Nmesh + j) * (Nmesh / 2 + 1) + k][0] =
				    -kvec[axes] / kmag2 * delta * sin(phase);
				  Cdata[((i - Local_x_start) * Nmesh + j) * (Nmesh / 2 + 1) + k][1] =
				    kvec[axes] / kmag2 * delta * cos(phase);
				}
			    }
			  else	/* k=0 plane needs special treatment */
			    {
			      if(i == 0)
				{
				  if(j >= Nmesh / 2)
				    continue;
				  else
				    {
				      if(i >= Local_x_start && i < (Local_x_start + Local_nx))
					{
					  jj = Nmesh - j;	/* note: j!=0 surely holds at this point */

					  Cdata[((i - Local_x_start) * Nmesh + j) * (Nmesh / 2 + 1) + k][0] =
					    -kvec[axes] / kmag2 * delta * sin(phase);
					  Cdata[((i - Local_x_start) * Nmesh + j) * (Nmesh / 2 + 1) + k][1] =
					    kvec[axes] / kmag2 * delta * cos(phase);

					  Cdata[((i - Local_x_start) * Nmesh + jj) * (Nmesh / 2 + 1) + k][0] =
					    -kvec[axes] / kmag2 * delta * sin(phase);
					  Cdata[((i - Local_x_start) * Nmesh + jj) * (Nmesh / 2 + 1) + k][1] =
					    -kvec[axes] / kmag2 * delta * cos(phase);
					}
				    }
				}
			      else	/* here comes i!=0 : conjugate can be on other processor! */
				{
				  if(i >= Nmesh / 2)
				    continue;
				  else
				    {
				      ii = Nmesh - i;
				      if(ii == Nmesh)
					ii = 0;
				      jj = Nmesh - j;
				      if(jj == Nmesh)
					jj = 0;

				      if(i >= Local_x_start && i < (Local_x_start + Local_nx))
					{
					  Cdata[((i - Local_x_start) * Nmesh + j) * (Nmesh / 2 + 1) + k][0] =
					    -kvec[axes] / kmag2 * delta * sin(phase);
					  Cdata[((i - Local_x_start) * Nmesh + j) * (Nmesh / 2 + 1) + k][1] =
					    kvec[axes] / kmag2 * delta * cos(phase);
					}

				      if(ii >= Local_x_start && ii < (Local_x_start + Local_nx))
					{
					  Cdata[((ii - Local_x_start) * Nmesh + jj) * (Nmesh / 2 + 1) +
						k][0] = -kvec[axes] / kmag2 * delta * sin(phase);
					  Cdata[((ii - Local_x_start) * Nmesh + jj) * (Nmesh / 2 + 1) +
						k][1] = -kvec[axes] / kmag2 * delta * cos(phase);
					}
				    }
				}
			    }
			}
		    }
		}
	    }


      FILE * fp = fopen("CDATA", "w");
      fwrite(Cdata, Nmesh * Nmesh * (Nmesh / 2 + 1) , sizeof(double) * 2, fp);
      fclose(fp);
      abort();
	  fftw_execute(Inverse_plan);	/** FFT **/

	  /* now get the plane on the right side from neighbour on the right, 
	     and send the left plane */

	  recvTask = ThisTask;
	  do
	    {
	      recvTask--;
	      if(recvTask < 0)
		recvTask = NTask - 1;
	    }
	  while(Local_nx_table[recvTask] == 0);

	  sendTask = ThisTask;
	  do
	    {
	      sendTask++;
	      if(sendTask >= NTask)
		sendTask = 0;
	    }
	  while(Local_nx_table[sendTask] == 0);

	  /* use non-blocking send */

	  if(Local_nx > 0)
	    {
	      MPI_Isend(&Disp[0],
			sizeof(fftw_real) * Nmesh * (2 * (Nmesh / 2 + 1)),
			MPI_BYTE, recvTask, 10, MPI_COMM_WORLD, &request);

	      MPI_Recv(&Disp[(Local_nx * Nmesh) * (2 * (Nmesh / 2 + 1))],
		       sizeof(fftw_real) * Nmesh * (2 * (Nmesh / 2 + 1)),
		       MPI_BYTE, sendTask, 10, MPI_COMM_WORLD, &status);

	      MPI_Wait(&request, &status);
	    }


	  /* read-out displacements */

	  for(n = 0; n < NumPart; n++)
	    {
#if defined(MULTICOMPONENTGLASSFILE) && defined(DIFFERENT_TRANSFER_FUNC)
	      if(P[n].Type == Type)
#endif
		{
		  u = P[n].Pos[0] / Box * Nmesh;
		  v = P[n].Pos[1] / Box * Nmesh;
		  w = P[n].Pos[2] / Box * Nmesh;

		  i = (int) u;
		  j = (int) v;
		  k = (int) w;

		  if(i == (Local_x_start + Local_nx))
		    i = (Local_x_start + Local_nx) - 1;
		  if(i < Local_x_start)
		    i = Local_x_start;
		  if(j == Nmesh)
		    j = Nmesh - 1;
		  if(k == Nmesh)
		    k = Nmesh - 1;

		  u -= i;
		  v -= j;
		  w -= k;

		  i -= Local_x_start;
		  ii = i + 1;
		  jj = j + 1;
		  kk = k + 1;

		  if(jj >= Nmesh)
		    jj -= Nmesh;
		  if(kk >= Nmesh)
		    kk -= Nmesh;

		  f1 = (1 - u) * (1 - v) * (1 - w);
		  f2 = (1 - u) * (1 - v) * (w);
		  f3 = (1 - u) * (v) * (1 - w);
		  f4 = (1 - u) * (v) * (w);
		  f5 = (u) * (1 - v) * (1 - w);
		  f6 = (u) * (1 - v) * (w);
		  f7 = (u) * (v) * (1 - w);
		  f8 = (u) * (v) * (w);

		  dis = Disp[(i * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + k] * f1 +
		    Disp[(i * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + kk] * f2 +
		    Disp[(i * Nmesh + jj) * (2 * (Nmesh / 2 + 1)) + k] * f3 +
		    Disp[(i * Nmesh + jj) * (2 * (Nmesh / 2 + 1)) + kk] * f4 +
		    Disp[(ii * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + k] * f5 +
		    Disp[(ii * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + kk] * f6 +
		    Disp[(ii * Nmesh + jj) * (2 * (Nmesh / 2 + 1)) + k] * f7 +
		    Disp[(ii * Nmesh + jj) * (2 * (Nmesh / 2 + 1)) + kk] * f8;

		  P[n].Vel[axes] = dis;

		  if(dis > maxdisp)
		    maxdisp = dis;
		}
	    }
	}
    }


  /* now add displacement to Lagrangian coordinates, and multiply velocities by correct factor */
  for(n = 0; n < NumPart; n++)
    {
      for(axes = 0; axes < 3; axes++)
	{
	  P[n].Pos[axes] += P[n].Vel[axes];
	  P[n].Vel[axes] *= vel_prefac;
	  P[n].Pos[axes] = periodic_wrap(P[n].Pos[axes]);
	}
    }

  gsl_rng_free(random_generator);

  MPI_Reduce(&maxdisp, &max_disp_glob, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if(ThisTask == 0)
    {
      printf("\nMaximum displacement: %g kpc/h, in units of the part-spacing= %g\n",
	     max_disp_glob, max_disp_glob / (Box / Nmesh));
    }
}



void initialize_ffts(void)
{
  int total_size, i, additional;
  int *slab_to_task_local;
  size_t bytes;


  total_size = fftw_mpi_local_size_3d(Nmesh, Nmesh, Nmesh / 2 + 1, MPI_COMM_WORLD, 
              &Local_nx, &Local_x_start) * 2;

  Local_nx_table = malloc(sizeof(int) * NTask);
  MPI_Allgather(&Local_nx, 1, MPI_INT, Local_nx_table, 1, MPI_INT, MPI_COMM_WORLD);

  Slab_to_task = malloc(sizeof(int) * Nmesh);
  slab_to_task_local = malloc(sizeof(int) * Nmesh);

  for(i = 0; i < Nmesh; i++)
    slab_to_task_local[i] = 0;

  for(i = 0; i < Local_nx; i++)
    slab_to_task_local[Local_x_start + i] = ThisTask;

  MPI_Allreduce(slab_to_task_local, Slab_to_task, Nmesh, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  free(slab_to_task_local);

  additional = (Nmesh) * (2 * (Nmesh / 2 + 1));	/* additional plane on the right side */

  Disp = (fftw_real *) malloc(bytes = sizeof(fftw_real) * (total_size + additional));
  Workspace = (fftw_real *) malloc(bytes += sizeof(fftw_real) * total_size);

  if(Disp && Workspace)
    {
      if(ThisTask == 0)
	printf("\nallocated %g Mbyte on Task %d for FFT's\n", bytes / (1024.0 * 1024.0), ThisTask);
    }
  else
    {
      printf("failed to allocate %g Mbyte on Task %d\n", bytes / (1024.0 * 1024.0), ThisTask);
      printf("bailing out.\n");
      FatalError(1);
    }

  Cdata = (fftw_complex *) Disp;	/* transformed array */

  Inverse_plan = fftw_mpi_plan_dft_c2r_3d(Nmesh, Nmesh, Nmesh, 
        Cdata, Disp,
        MPI_COMM_WORLD, FFTW_ESTIMATE);

  setup_grid();
}



void free_ffts(void)
{
  free(Workspace);
  free(Disp);
  free(Slab_to_task);
  fftw_destroy_plan(Inverse_plan);
}




long long ijk_to_id(int i, int j, int k) {
    long long id = ((long long) i) * Nmesh * Nmesh + ((long long)j) * Nmesh + k + 1;
    return id;
}
int cmp_id(struct part_data * p1, struct part_data * p2) {
    if(p1->ID < p2->ID) return -1;
    if(p1->ID > p2->ID) return 1;
    return 0;
}

void setup_grid() {
    int slab;
    int perslabsize = Nmesh * Nmesh;
    int i;
    if(Nmesh > 45000) {
        fprintf(stderr, "per slab size is too big ( more than 2G elements)\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    NumPart = 0;
    for(slab = 0; slab < Nmesh; slab++) {
        if(Slab_to_task[slab] != ThisTask) continue;
        NumPart += perslabsize;
    }

    size_t bytes = 0;
    if(NumPart)  {
        P = (struct part_data *) malloc(bytes = sizeof(struct part_data) * NumPart);
        if(!P) {
            fprintf(stderr, "failed to allocate %g Mbyte (%d particles) on Task %d\n", bytes / (1024.0 * 1024.0), NumPart, ThisTask);
            FatalError(9891);
        } else {
        }
        if(ThisTask == 0) {
            printf( "allocate %g Mbyte (%d particles) on Task %d\n", bytes / (1024.0 * 1024.0), NumPart, ThisTask);
            fflush(stdout);
        }
    }

#ifdef  MULTICOMPONENTGLASSFILE
#error MULTICOMPONENTGLASSFILE is not supported
#endif
    int n = 0;
    for(slab = 0; slab < Nmesh; slab++) {
        if(Slab_to_task[slab] != ThisTask) continue;
        int j, k;
        for(j = 0; j < Nmesh; j ++)
        for(k = 0; k < Nmesh; k ++) {
            int i = slab;
            P[n].Pos[0] = i * Box / Nmesh;
            P[n].Pos[1] = j * Box / Nmesh;
            P[n].Pos[2] = k * Box / Nmesh;
            P[n].ID = ijk_to_id(i, j, k);
            n++;
        }
    }
    qsort(P, NumPart, sizeof(P[0]), cmp_id);    
}
