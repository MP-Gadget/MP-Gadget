#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef VORONOI

#include "allvars.h"
#include "proto.h"
#include "voronoi.h"


int Ndp;			/* number of delaunay points */
int MaxNdp;			/* maximum number of delaunay points */
point *DP;			/* delaunay points */

int Ndt;			/* number of delaunary tetrahedra/triangles */
int MaxNdt;			/* maximum number of delaunary tetrahedra/triangles */
tetra *DT;			/* Delaunay tetrahedra/triangles */

tetra_center *DTC;		/* circumcenters of delaunay tetrahedra */
char *DTF;

int Nvf;			/* number of Voronoi faces */
int MaxNvf;			/* maximum number of Voronoi faces */
face *VF;			/* Voronoi faces */

unsigned char *Edge_visited;     


int DPinfinity;
double CentralOffsetX, CentralOffsetY, CentralOffsetZ, ConversionFac;

struct list_export_data *ListExports;
struct list_P_data *List_P;
struct primexch *PrimExch;
struct grad_data *Grad, *GradExch;

int CountInSphereTests, CountInSphereTestsExact;
int CountConvexEdgeTest, CountConvexEdgeTestExact;
int Ninlist, MaxNinlist;


int CountFlips, Count_1_to_3_Flips2d, Count_2_to_4_Flips2d;
int Count_1_to_4_Flips, Count_2_to_3_Flips, Count_3_to_2_Flips, Count_4_to_4_Flips;
int Count_EdgeSplits, Count_FaceSplits;
int Count_InTetra, Count_InTetraExact;
int Largest_N_DP_Buffer;

void voronoi_mesh(void)
{
  int tlast, i, n, iter = 0, ntot, skip;
  long long totNdp;
  double timeinsert = 0, tstart, tend, t0, t1;

  CPU_Step[CPU_MISC] += measure_time();

  if(ThisTask == 0)
    printf("\ncreate delaunay mesh\n");

  initialize_and_create_first_tetra();

  CountInSphereTests = CountInSphereTestsExact = 0;
  CountConvexEdgeTest = CountConvexEdgeTestExact = 0;
  CountFlips = Count_1_to_3_Flips2d = Count_2_to_4_Flips2d = 0;
  Count_1_to_4_Flips = 0;
  Count_2_to_3_Flips = 0;
  Count_3_to_2_Flips = 0;
  Count_4_to_4_Flips = 0;
  Count_EdgeSplits = 0;
  Count_FaceSplits = 0;
  Count_InTetra = Count_InTetraExact = 0;
  Largest_N_DP_Buffer = 0;

  MaxNinlist = Indi.AllocFacNinlist;
  ListExports = mymalloc_movable(&ListExports, "ListExports", MaxNinlist * sizeof(struct list_export_data));

  List_P = mymalloc_movable(&List_P, "List_P", N_gas * sizeof(struct list_P_data));

  DTC = mymalloc_movable(&DTC, "DTC", MaxNdt * sizeof(tetra_center));
  DTF = mymalloc_movable(&DTF, "DTF", MaxNdt * sizeof(char));
  for(i = 0; i < Ndt; i++)
    DTF[i] = 0;

  Ninlist = 0;

  tlast = 0;

  t0 = second();

  do
    {
      skip = Ndp;

      if(iter == 0)
	n = voronoi_get_local_particles();
      else
	{
#ifdef ALTERNATIVE_GHOST_SEARCH
	  n = voronoi_ghost_search_alternative();
#else
	  n = voronoi_ghost_search();
#endif
	}

      MPI_Allreduce(&n, &ntot, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      if(iter == 0)
	{
	  if(ThisTask == 0)
	    printf("have obtained %d local points (%d on task 0)\n", ntot, n);
	}
      else
	{
	  if(ThisTask == 0)
	    printf("iter=%d: have obtained %d additional points (%d on task 0)\n", iter, ntot, n);
	}

      
      tstart = second();
      for(i = 0; i < n; i++)
	{
	  set_integers_for_point(skip + i);

	  tlast = insert_point(skip + i, tlast);
	}
      tend = second();
      timeinsert += timediff(tstart, tend);

      if(ThisTask == 0)
	printf("points inserted.\n");

#ifdef ALTERNATIVE_GHOST_SEARCH
      if(iter > 0)
	{
	  compute_circumcircles();

	  n = compute_max_delaunay_radius();

	  MPI_Allreduce(&n, &ntot, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	  if(ThisTask == 0)
	    printf("still no complete cell for %d particles\n", ntot);
	}
      else
	{
	  for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
	    if(P[i].Type == 0)
	      SphP[i].MaxDelaunayRadius = MAX_REAL_NUMBER;	/* too make sure that all active particles will be checked */

	  ntot = 1;
	}
#else
      compute_circumcircles();
      
      if(iter > 0)
	{
	  n = count_undecided_tetras();
	  
	  MPI_Allreduce(&n, &ntot, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	  
	  if(ThisTask == 0)
	    printf("still undecided %d tetrahedras\n", ntot);
	  
	  if(ntot)
	    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
	      if(P[i].Type == 0)
		SphP[i].Hsml *= HSML_INCREASE_FACTOR;
	}
      else
	ntot = 1;
#endif
      
      
      if(iter > MAX_VORONOI_ITERATIONS)
	terminate("too many iterations\n");

      iter++;
    }

  while(ntot > 0);

#ifndef ALTERNATIVE_GHOST_SEARCH
  compute_max_delaunay_radius();
#endif

  sumup_large_ints(1, &Ndp, &totNdp);

  t1 = second();

  if(ThisTask == 0)
    {
#ifndef TWODIMS
      printf("Delaunay tetrahedra are calculated, point insertion took=%g sec, all together took=%g\n",
	     timeinsert, timediff(t0, t1));
      printf("D-Points=%d (N_gas=%d) D-Tetrahedra=%d  InSphereTests=%d InSphereTestsExact=%d  Flips=%d\n",
	     Ndp, N_gas, Ndt, CountInSphereTests, CountInSphereTestsExact, CountFlips);
      printf
	("   1_to_4_Flips=%d  2_to_3_Flips=%d  3_to_2_Flips=%d  4_to_4_Flips=%d  FaceSplits=%d  EdgeSplits=%d\n",
	 Count_1_to_4_Flips, Count_2_to_3_Flips, Count_3_to_2_Flips, Count_4_to_4_Flips, Count_FaceSplits,
	 Count_EdgeSplits);
      printf("   InTetra=%d  InTetraExact=%d  ConvexEdgeTest=%d  ConvexEdgeTestExact=%d\n", Count_InTetra,
	     Count_InTetraExact, CountConvexEdgeTest, CountConvexEdgeTestExact);
#else
      printf("Delaunay triangles are calculated, point insertion took=%g sec\n", timeinsert);
      printf("D-Points=%d  D-Triangles=%d  InCircleTests=%d InCircleTestsExact=%d  Flips=%d\n",
	     Ndp, Ndt, CountInSphereTests, CountInSphereTestsExact, CountFlips);
      printf("   1_to_3_Flips=%d  2_to_4_Flips=%d  InTriangle=%d  InTriangleExact=%d\n",
	     Count_1_to_3_Flips2d, Count_2_to_4_Flips2d, Count_InTetra, Count_InTetraExact);
#endif

      printf("Total D-Points: %d%09d  Ratio=%g\n",
	     (int) (totNdp / 1000000000), (int) (totNdp % 1000000000), ((double) totNdp) / All.TotN_gas);
      printf("\n");
    }


  /*  dump_points(); */


  for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    if(P[i].Type == 0)
      {
	SphP[i].Volume = 0;
      }

  compute_voronoi_faces_and_volumes();

  myfree(DTF);
  myfree(DTC);
  DTC = NULL;			/* here we can free the centers of the Delaunay triangles again */


  /* check whether we can reduce allocation factors */
  while(Ndp < ALLOC_DECREASE_FACTOR * Indi.AllocFacNdp && Indi.AllocFacNdp > MIN_ALLOC_NUMBER)
    Indi.AllocFacNdp /= ALLOC_INCREASE_FACTOR;

  while(Ndt < ALLOC_DECREASE_FACTOR * Indi.AllocFacNdt && Indi.AllocFacNdt > MIN_ALLOC_NUMBER)
    Indi.AllocFacNdt /= ALLOC_INCREASE_FACTOR;

  while(Nvf < ALLOC_DECREASE_FACTOR * Indi.AllocFacNvf && Indi.AllocFacNvf > MIN_ALLOC_NUMBER)
    Indi.AllocFacNvf /= ALLOC_INCREASE_FACTOR;

  while(Ninlist < ALLOC_DECREASE_FACTOR * Indi.AllocFacNinlist && Indi.AllocFacNinlist > MIN_ALLOC_NUMBER)
    Indi.AllocFacNinlist /= ALLOC_INCREASE_FACTOR;

  while(Largest_N_DP_Buffer < ALLOC_DECREASE_FACTOR * Indi.AllocFacN_DP_Buffer
	&& Indi.AllocFacN_DP_Buffer > MIN_ALLOC_NUMBER)
    Indi.AllocFacN_DP_Buffer /= ALLOC_INCREASE_FACTOR;

  double vol, voltot;

  for(i = FirstActiveParticle, vol = 0; i >= 0; i = NextActiveParticle[i])
    if(P[i].Type == 0)
      {
	vol += SphP[i].Volume;
      }


  MPI_Allreduce(&vol, &voltot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if(ThisTask == 0)
    printf("total volume of active particles = %g\n", voltot);

  report_memory_usage(&HighMark_voronoi, "VORONOI");
}




int voronoi_get_local_particles(void)
{
  int p, count = 0;


  /* for better exploitation of the order, this version may be better */

  for(p = 0; p < N_gas; p++)
    if(P[p].Type == 0 && TimeBinActive[P[p].TimeBin])
      {
	if(Ninlist >= MaxNinlist)
	  {
	    Indi.AllocFacNinlist *= ALLOC_INCREASE_FACTOR;
	    MaxNinlist = Indi.AllocFacNinlist;
#ifdef VERBOSE
	    printf("Task=%d: increase memory allocation, MaxNinlist=%d Indi.AllocFacNinlist=%g\n",
		   ThisTask, MaxNinlist, Indi.AllocFacNinlist);
#endif
	    ListExports = myrealloc_movable(ListExports, MaxNinlist * sizeof(struct list_export_data));

	    if(Ninlist >= MaxNinlist)
	      terminate("Ninlist >= MaxNinlist");
	  }

	List_P[p].currentexport = List_P[p].firstexport = Ninlist++;
	ListExports[List_P[p].currentexport].image_bits = 1;
	ListExports[List_P[p].currentexport].nextexport = -1;
	ListExports[List_P[p].currentexport].origin = ThisTask;
	ListExports[List_P[p].currentexport].index = p;

	if(Ndp >= MaxNdp)
	  {
	    Indi.AllocFacNdp *= ALLOC_INCREASE_FACTOR;
	    MaxNdp = Indi.AllocFacNdp;
#ifdef VERBOSE
	    printf("Task=%d: increase memory allocation, MaxNdp=%d Indi.AllocFacNdp=%g\n",
		   ThisTask, MaxNdp, Indi.AllocFacNdp);
#endif
	    DP -= 5;
	    DP = myrealloc_movable(DP, (MaxNdp + 5) * sizeof(point));
	    DP += 5;

	    if(Ndp >= MaxNdp)
	      terminate("Ndp >= MaxNdp");
	  }

	DP[Ndp].x = P[p].Pos[0];
	DP[Ndp].y = P[p].Pos[1];
	DP[Ndp].z = P[p].Pos[2];
	DP[Ndp].ID = P[p].ID;
	DP[Ndp].task = ThisTask;
	DP[Ndp].index = p;
	DP[Ndp].inactiveflag = -1;
	Ndp++;
	count++;
      }
    else
      {
	List_P[p].firstexport = -1;
	List_P[p].currentexport = -1;
      }

  return count;
}



void free_voronoi_mesh(void)
{
  myfree(List_P);
  myfree(ListExports);
  myfree(DT);
  myfree(DP - 5);
  myfree(VF);
}


int compute_max_delaunay_radius(void)
{
  int i, j, count = 0;
  point *p;
  double dx, dy, dz, r;

  for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    if(P[i].Type == 0)
      SphP[i].MaxDelaunayRadius = 0;

  for(i = 0; i < Ndt; i++)
    {
      if(DT[i].t[0] < 0)	/* deleted ? */
	continue;

      dx = DP[DT[i].p[0]].x - DTC[i].cx;
      dy = DP[DT[i].p[0]].y - DTC[i].cy;
      dz = DP[DT[i].p[0]].z - DTC[i].cz;

      r = 2 * sqrt(dx * dx + dy * dy + dz * dz);

      for(j = 0; j < (DIMS + 1); j++)
	{
	  p = &DP[DT[i].p[j]];

	  if(p->task == ThisTask && p->index < N_gas && p->index >= 0)
	    if(TimeBinActive[P[p->index].TimeBin])
	      if(r > SphP[p->index].MaxDelaunayRadius)
		SphP[p->index].MaxDelaunayRadius = r;
	}
    }

#ifdef ALTERNATIVE_GHOST_SEARCH
  for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
      if(P[i].Type == 0)
	{
	  if(SphP[i].MaxDelaunayRadius >= SphP[i].Hsml)
	    {
	      count++;

	      if(SphP[i].Hsml == boxHalf_X)
		{
		  printf("too big Hsml: ID=%d vol=%g  hsml=%g MaxDelaunayRadius=%g\n", P[i].ID,
			 SphP[i].Volume, SphP[i].Hsml, SphP[i].MaxDelaunayRadius);
		  dump_points();
		  terminate("too big Hsml");
		}

	      SphP[i].Hsml *= HSML_INCREASE_FACTOR;

	      if(SphP[i].Hsml > boxHalf_X || SphP[i].Hsml > boxHalf_Y || SphP[i].Hsml > boxHalf_Z)
		{
		  SphP[i].Hsml = boxHalf_X;
		}
	    }
	}
    }
#endif

  return count;
}





void compute_voronoi_faces_and_volumes(void)
{
  int i, bit, nr;

  for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    if(P[i].Type == 0)
      {
	SphP[i].Volume = 0;
	SphP[i].Center[0] = 0;
	SphP[i].Center[1] = 0;
	SphP[i].Center[2] = 0;
	
#ifdef VORONOI_SHAPESCHEME
	SphP[i].W = 0;
#endif
      }

  Edge_visited = mymalloc_movable(&Edge_visited, "Edge_visited", Ndt * sizeof(unsigned char));

  for(i = 0; i < Ndt; i++)
    Edge_visited[i] = 0;

 for(i = 0; i < Ndt; i++)
    {
      if(DT[i].t[0] < 0)	/* deleted ? */
	continue;

      bit = 1;
      nr = 0;

      while(Edge_visited[i] != EDGE_ALL)
	{
	  if((Edge_visited[i] & bit) == 0)
	    process_edge_faces_and_volumes(i, nr);

	  bit <<= 1;
	  nr++;
	}
    }

 for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    if(P[i].Type == 0)
      if(SphP[i].Volume)
	{
	  SphP[i].Center[0] /= SphP[i].Volume;
	  SphP[i].Center[1] /= SphP[i].Volume;
	  SphP[i].Center[2] /= SphP[i].Volume;
#ifdef VORONOI_SHAPESCHEME
	  SphP[i].W /= SphP[i].Volume;
	  SphP[i].W -=
	    pow(P[i].Pos[0] - SphP[i].Center[0], 2) + pow(P[i].Pos[1] - SphP[i].Center[1],
							  2) + pow(P[i].Pos[2] - SphP[i].Center[2], 2);
#endif
	}

  myfree(Edge_visited);
}

void dump_points(void)
{
  FILE *fd;
  int i;
  float xyz[3];
  char buf[1000];

  sprintf(buf, "points_%d.dat", ThisTask);
  fd = fopen(buf, "w");
  my_fwrite(&Ndp, sizeof(int), 1, fd);
  for(i = 0; i < Ndp; i++)
    {
      xyz[0] = DP[i].x;
      xyz[1] = DP[i].y;
      xyz[2] = DP[i].z;
      my_fwrite(xyz, sizeof(float), 3, fd);
    }
  fclose(fd);
}

#endif
