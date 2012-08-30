#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef VORONOI
#include <gmp.h>
#include "allvars.h"
#include "proto.h"
#include "voronoi.h"


#if defined(TWODIMS) && !defined(ONEDIMS)	/* will only be compiled in 2D case */

#define INSIDE_EPS   1.0e-6


void initialize_and_create_first_tetra(void)
{
  point *p;

  int i, n;

  for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    if(P[i].Type == 0)
      SphP[i].Hsml = 1.01 * SphP[i].MaxDelaunayRadius;

  MaxNdp = Indi.AllocFacNdp;
  MaxNdt = Indi.AllocFacNdt;
  MaxNvf = Indi.AllocFacNvf;

  Ndp = 0;
  Nvf = 0;
  Ndt = 0;


  VF = mymalloc_movable(&VF, "VF", MaxNvf * sizeof(face));

  DP = mymalloc_movable(&DP, "DP", (MaxNdp + 5) * sizeof(point));
  DP += 5;

  DT = mymalloc_movable(&DT, "DT", MaxNdt * sizeof(tetra));


  /* construct all encompassing huge triangle */

  double box, tetra_incircle, tetra_sidelength, tetra_height;

  box = boxSize_X;
  if(box < boxSize_Y)
    box = boxSize_Y;

  box *= 1.05;

  tetra_incircle = 2.001 * (1 + sqrt(3)) / 3.0 * box;	/* to give room for ghost particles needed for periodic/reflective
							   boundary conditions, the incircle is twice as large, i.e.
							   [-0.5*box, 1.5*box,-0.5*box, 1.5*box] should be inside triangle */
  tetra_sidelength = tetra_incircle * sqrt(12);
  tetra_height = sqrt(3.0) / 2 * tetra_sidelength;

  if(ThisTask == 0)
    printf("side-length of enclosing triangle=%g tetra_height=%g box=%g\n", tetra_sidelength, tetra_height,
	   box);

  /* first, let's make the points */
  DP[-3].x = 0.5 * tetra_sidelength;
  DP[-3].y = -1.0 / 3 * tetra_height;
  DP[-3].z = 0;

  DP[-2].x = 0;
  DP[-2].y = 2.0 / 3 * tetra_height;
  DP[-2].z = 0;

  DP[-1].x = -0.5 * tetra_sidelength;
  DP[-1].y = -1.0 / 3 * tetra_height;
  DP[-1].z = 0;


  for(i = -3; i <= -1; i++)
    {
      DP[i].x += 0.5 * box;
      DP[i].y += 1.0 / 3 * tetra_height - 0.5 * box;
    }

  for(i = -3, p = &DP[-3]; i < 0; i++, p++)
    {
      p->index = -1;
      p->task = ThisTask;
    }

  /* we also define a neutral element at infinity */
  DPinfinity = -4;

  DP[DPinfinity].x = 0;
  DP[DPinfinity].y = 0;
  DP[DPinfinity].z = 0;
  DP[DPinfinity].index = -1;
  DP[DPinfinity].task = ThisTask;

  /* now let's make the big triangle */
  DT[0].p[0] = -3;
  DT[0].p[1] = -2;
  DT[0].p[2] = -1;


  /* On the outer faces, we attach tetrahedra with the neutral element as tip.
   * This way we will be able to navigate nicely within the tesselation,
   * and all tetrahedra have defined neighbouring tetrahedra.
   */

  for(i = 0; i < 3; i++)
    {
      n = i + 1;		/* tetra index */

      DT[0].t[i] = n;
      DT[0].s[i] = 2;

      DT[n].t[2] = 0;
      DT[n].s[2] = i;
      DT[n].p[2] = DPinfinity;
    }


  DT[1].p[0] = DT[0].p[2];
  DT[1].p[1] = DT[0].p[1];

  DT[2].p[0] = DT[0].p[0];
  DT[2].p[1] = DT[0].p[2];

  DT[3].p[0] = DT[0].p[1];
  DT[3].p[1] = DT[0].p[0];

  DT[1].t[0] = 3;
  DT[3].t[1] = 1;
  DT[1].s[0] = 1;
  DT[3].s[1] = 0;

  DT[1].t[1] = 2;
  DT[2].t[0] = 1;
  DT[1].s[1] = 0;
  DT[2].s[0] = 1;

  DT[2].t[1] = 3;
  DT[3].t[0] = 2;
  DT[2].s[1] = 0;
  DT[3].s[0] = 1;


  Ndt = 4;			/* we'll start out with 4 triangles */

  CentralOffsetX = 0.5 * box - 0.5000001 * tetra_sidelength;
  CentralOffsetY = -0.5000001 * box;

  ConversionFac = 1.0 / (1.001 * tetra_sidelength);

  for(i = -3; i < 0; i++)
    set_integers_for_point(i);
}




int insert_point(int pp, int ttstart)	/* returns a triangle that (currently) contains the point p */
{
  int tt0, tt1, tt2, tt3, ttetra_with_p;

  int moves, degenerate_flag;


  /* first, need to do a point location */
  tt0 = get_triangle(pp, &moves, &degenerate_flag, ttstart);

  ttetra_with_p = tt0;

  if(degenerate_flag == 1)	/* that's the normal split of a triangle into 3 */
    {
      /* we now need to split this triangle into three  */
      tt1 = Ndt++;
      tt2 = Ndt++;

      if(Ndt > MaxNdt)
	{
	  Indi.AllocFacNdt *= ALLOC_INCREASE_FACTOR;
	  MaxNdt = Indi.AllocFacNdt;
#ifdef VERBOSE
	  printf("Task=%d: increase memory allocation, MaxNdt=%d Indi.AllocFacNdt=%g\n",
		 ThisTask, MaxNdt, Indi.AllocFacNdt);
#endif
	  DT = myrealloc_movable(DT, MaxNdt * sizeof(tetra));
	  DTC = myrealloc_movable(DTC, MaxNdt * sizeof(tetra_center));
	  DTF = myrealloc_movable(DTF, MaxNdt * sizeof(char));

	  if(Ndt > MaxNdt)
	    terminate("Ndt > MaxNdt");
	}

      DT[tt1] = DT[tt0];
      DT[tt2] = DT[tt0];

      make_a_1_to_3_flip(pp, tt0, tt1, tt2);

      DTF[tt0] = 0;
      DTF[tt1] = 0;
      DTF[tt2] = 0;

      check_edge_and_flip_if_needed(pp, tt0);
      check_edge_and_flip_if_needed(pp, tt1);
      check_edge_and_flip_if_needed(pp, tt2);
    }
  else
    {
      degenerate_flag -= 10;

      tt1 = DT[tt0].t[degenerate_flag];

      /* we now need to split this into two triangles */
      tt2 = Ndt++;
      tt3 = Ndt++;

      if(Ndt > MaxNdt)
	{
	  Indi.AllocFacNdt *= ALLOC_INCREASE_FACTOR;
	  MaxNdt = Indi.AllocFacNdt;
#ifdef VERBOSE
	  printf("Task=%d: increase memory allocation, MaxNdt=%d Indi.AllocFacNdt=%g\n",
		 ThisTask, MaxNdt, Indi.AllocFacNdt);
#endif
	  DT = myrealloc_movable(DT, MaxNdt * sizeof(tetra));
	  DTC = myrealloc_movable(DTC, MaxNdt * sizeof(tetra_center));
	  DTF = myrealloc_movable(DTF, MaxNdt * sizeof(char));

	  if(Ndt > MaxNdt)
	    terminate("Ndt > MaxNdt");
	}

      DT[tt2] = DT[tt0];
      DT[tt3] = DT[tt1];

      make_a_2_to_4_flip(pp, tt0, tt1, tt2, tt3, degenerate_flag, DT[tt0].s[degenerate_flag]);

      DTF[tt0] = 0;
      DTF[tt1] = 0;
      DTF[tt2] = 0;
      DTF[tt3] = 0;

      check_edge_and_flip_if_needed(pp, tt0);
      check_edge_and_flip_if_needed(pp, tt1);
      check_edge_and_flip_if_needed(pp, tt2);
      check_edge_and_flip_if_needed(pp, tt3);
    }

  return ttetra_with_p;
}



void make_a_2_to_4_flip(int pp, int tt0, int tt1, int tt2, int tt3, int i0, int j0)
{
  tetra *t0 = &DT[tt0];

  tetra *t1 = &DT[tt1];

  tetra *t2 = &DT[tt2];

  tetra *t3 = &DT[tt3];

  int i1, i2, j1, j2;

  CountFlips++;
  Count_2_to_4_Flips2d++;


  i1 = i0 + 1;
  i2 = i0 + 2;
  j1 = j0 + 1;
  j2 = j0 + 2;

  if(i1 > 2)
    i1 -= 3;
  if(i2 > 2)
    i2 -= 3;

  if(j1 > 2)
    j1 -= 3;
  if(j2 > 2)
    j2 -= 3;

  t0->p[i1] = pp;
  t1->p[j2] = pp;
  t2->p[i2] = pp;
  t3->p[j1] = pp;

  t0->t[i0] = tt1;
  t1->t[j0] = tt0;
  t0->s[i0] = j0;
  t1->s[j0] = i0;

  t1->t[j1] = tt3;
  t3->t[j2] = tt1;
  t1->s[j1] = j2;
  t3->s[j2] = j1;

  t2->t[i1] = tt0;
  t0->t[i2] = tt2;
  t2->s[i1] = i2;
  t0->s[i2] = i1;

  t2->t[i0] = tt3;
  t3->t[j0] = tt2;
  t2->s[i0] = j0;
  t3->s[j0] = i0;

  DT[t0->t[i1]].t[t0->s[i1]] = tt0;
  DT[t1->t[j2]].t[t1->s[j2]] = tt1;
  DT[t2->t[i2]].t[t2->s[i2]] = tt2;
  DT[t3->t[j1]].t[t3->s[j1]] = tt3;
}


void make_a_1_to_3_flip(int pp, int tt0, int tt1, int tt2)
{
  tetra *t0 = &DT[tt0];

  tetra *t1 = &DT[tt1];

  tetra *t2 = &DT[tt2];

  CountFlips++;
  Count_1_to_3_Flips2d++;

  t0->p[0] = pp;
  t1->p[1] = pp;
  t2->p[2] = pp;


  t0->t[1] = tt1;
  t1->t[0] = tt0;
  t0->s[1] = 0;
  t1->s[0] = 1;

  t1->t[2] = tt2;
  t2->t[1] = tt1;
  t1->s[2] = 1;
  t2->s[1] = 2;

  t2->t[0] = tt0;
  t0->t[2] = tt2;
  t2->s[0] = 2;
  t0->s[2] = 0;

  DT[t0->t[0]].t[t0->s[0]] = tt0;
  DT[t1->t[1]].t[t1->s[1]] = tt1;
  DT[t2->t[2]].t[t2->s[2]] = tt2;
}


void check_edge_and_flip_if_needed(int ip, int it)
{
  tetra *t = &DT[it];

  int tt, pp, t0, t2;

  int pi, pi1, pi2;

  int ni, ni1, ni2;

  int st2, st0;

  if(t->p[0] == ip)
    pi = 0;
  else if(t->p[1] == ip)
    pi = 1;
  else
    pi = 2;

  /* get the point that lies accross the edge to obtain the quadriliteral */

  tt = t->t[pi];
  ni = t->s[pi];
  pp = DT[tt].p[ni];

  int ret, ret_exact;

  ret = InCircle_Errorbound(t->p[0], t->p[1], t->p[2], pp);
  CountInSphereTests++;

  if(ret != 0)
    ret_exact = ret;
  else
    {
      ret_exact = InCircle_Exact(t->p[0], t->p[1], t->p[2], pp);
      CountInSphereTestsExact++;
    }


  if(ret_exact > 0)
    {
      /* pp lies in the triangle, the edge is not Delaunay. Need to do a flip */

      CountFlips++;

      ni1 = ni + 1;
      if(ni1 > 2)
	ni1 -= 3;
      ni2 = ni + 2;
      if(ni2 > 2)
	ni2 -= 3;

      pi1 = pi + 1;
      if(pi1 > 2)
	pi1 -= 3;
      pi2 = pi + 2;
      if(pi2 > 2)
	pi2 -= 3;


      t0 = DT[tt].t[ni1];
      t2 = t->t[pi1];

      st0 = DT[tt].s[ni1];
      st2 = t->s[pi1];

      /* change the points of the triangles */
      t->p[pi2] = pp;
      DT[tt].p[ni2] = ip;

      /* change the pointers to the neighbouring triangles, and fix
         the adjency relations */

      t->t[pi1] = tt;
      DT[tt].t[ni1] = it;
      t->s[pi1] = ni1;
      DT[tt].s[ni1] = pi1;


      t->t[pi] = t0;
      DT[t0].t[st0] = it;
      t->s[pi] = st0;
      DT[t0].s[st0] = pi;


      DT[tt].t[ni] = t2;
      DT[t2].t[st2] = tt;
      DT[tt].s[ni] = st2;
      DT[t2].s[st2] = ni;

      DTF[tt] = 0;
      DTF[it] = 0;

      /* now we need to test also the two sides opposite of p */

      check_edge_and_flip_if_needed(ip, it);
      check_edge_and_flip_if_needed(ip, tt);
    }
}




int get_triangle(int pp, int *moves, int *degenerate_flag, int ttstart)
{
  int count_moves = 0;

  int ret;

  int tt, next_tetra;

  tt = ttstart;

#define MAX_COUNT_MOVES 1000000

  while((ret = FindTriangle(tt, pp, degenerate_flag, &next_tetra)) == 0)
    {
      /* we need to see in which of the three possible neighbouring triangles
         we should walk. We'll choose the one which lies along the face that
         is traversed by a line from the cm of the triangle to the point in
         question.
       */
      count_moves++;

      if(count_moves > MAX_COUNT_MOVES)
	{
	  printf("ta=%d triangle=%d\n", ThisTask, (int) (tt));

	  if(count_moves > MAX_COUNT_MOVES + 10)
	    endrun(113123);
	}

      tt = next_tetra;
    }

  *moves = count_moves;

  return tt;
}




inline void add_row_2d(double *m, int r1, int r2, double fac)
{
  int i;

  for(i = 0; i < 3; i++)
    m[r1 * 3 + i] += fac * m[r2 * 3 + i];
}


int solve_linear_equations_2d(double *m, double *res)
{
  int ix, iy;

  if(fabs(m[0]) > fabs(m[3]))
    {
      ix = 0;
      iy = 1;
    }
  else
    {
      ix = 1;
      iy = 0;
    }

  add_row_2d(m, iy, ix, -m[iy * 3] / m[ix * 3]);

  res[1] = m[iy * 3 + 2] / m[iy * 3 + 1];
  res[0] = (m[ix * 3 + 2] - res[1] * m[ix * 3 + 1]) / m[ix * 3];

  if(fabs(m[ix * 3]) < 1.0e-12)
    return -1;

  return 0;
}



/* tests whether point p lies in the triangle, on an edge, or outside. In the latter case, a nighbouring triangle is returned */
int FindTriangle(int tt, int pp, int *degnerate_flag, int *nexttetra)
{
  tetra *t = &DT[tt];

  point *p = &DP[pp];

  int pp0, pp1, pp2;

  point *p0, *p1, *p2;

  pp0 = t->p[0];
  pp1 = t->p[1];
  pp2 = t->p[2];

  p0 = &DP[pp0];
  p1 = &DP[pp1];
  p2 = &DP[pp2];

  if(pp0 == DPinfinity || pp1 == DPinfinity || pp2 == DPinfinity)
    {
      printf("we are in a triangle with an infinity point. tetra=%d  p=(%g|%g)\n", (int) (tt), p->x, p->y);
      endrun(87658);
    }

  Count_InTetra++;

  double ax = p1->xx - p0->xx;

  double ay = p1->yy - p0->yy;

  double bx = p2->xx - p0->xx;

  double by = p2->yy - p0->yy;

  double qx = p->xx - p0->xx;

  double qy = p->yy - p0->yy;

  double mv_data[] = { ax, bx, qx, ay, by, qy };
  double x[2];

  int ivol, flag2, flag1, flag0;

  int count_zeros = 0;


  int status;

  status = solve_linear_equations_2d(mv_data, x);


  if(status < 0)
    {
      ivol = Orient2d_Exact(t->p[0], t->p[1], t->p[2]);
      if(ivol <= 0)
	{
	  printf("flat or negatively triangle found (ivol=%d)\n", ivol);
	  endrun(11213192);
	}
    }

  if(status >= 0)
    {
      if(x[0] > INSIDE_EPS && x[1] > INSIDE_EPS && (1 - (x[0] + x[1])) > INSIDE_EPS)
	{
	  /* looks like we are safely inside the triangle */

	  *degnerate_flag = 1;
	  return 1;
	}


      if(x[0] < -INSIDE_EPS || x[1] < -INSIDE_EPS || (1 - (x[0] + x[1])) < -INSIDE_EPS)
	{
	  /* looks like we are clearly outside the triangle.
	     Let's look for a good neighbouring triangle to continue the search */

	  /* note: in the (a,b) basis, the center-of-mass has coordinates (1/3, 1/3) */

	  double w, u;

	  if(fabs(x[1] - (1.0 / 3)) > INSIDE_EPS)
	    {
	      w = (1.0 / 3) / ((1.0 / 3) - x[1]);
	      if(w > 0)
		{
		  u = (1.0 / 3) + w * (x[0] - (1.0 / 3));
		  if(u > -INSIDE_EPS && (1 - u) > -INSIDE_EPS)
		    {
		      *nexttetra = t->t[2];
		      return 0;
		    }
		}
	    }


	  if(fabs(x[0] - (1.0 / 3)) > INSIDE_EPS)
	    {
	      w = (1.0 / 3) / ((1.0 / 3) - x[0]);
	      if(w > 0)
		{
		  u = (1.0 / 3) + w * (x[1] - (1.0 / 3));
		  if(u > -INSIDE_EPS && (1 - u) > -INSIDE_EPS)
		    {
		      *nexttetra = t->t[1];
		      return 0;
		    }
		}
	    }

	  *nexttetra = t->t[0];
	  return 0;
	}
    }

  /* here we need to decide whether we have a degenerate case, i.e.
     whether we think the point lies on an edge of the triangle */

  Count_InTetraExact++;

  ivol = Orient2d_Exact(t->p[0], t->p[1], t->p[2]);

  if(ivol <= 0)
    {
      printf("flat or negatively oriented triangle found (ivol=%d)\n", ivol);
      endrun(1128813192);
    }

  flag0 = Orient2d_Exact(pp1, pp2, pp);
  flag1 = Orient2d_Exact(pp2, pp0, pp);
  flag2 = Orient2d_Exact(pp0, pp1, pp);

  if(flag0 == 0)
    count_zeros++;

  if(flag1 == 0)
    count_zeros++;

  if(flag2 == 0)
    count_zeros++;

  if(count_zeros >= 2)
    {
      printf("flags=%d %d %d\n", flag0, flag1, flag2);

      printf("points: %d %d %d %d\n", (int) (pp0), (int) (pp1), (int) (pp2), (int) (pp));
      printf("Ngas=%d\n", N_gas);
      printf("xyz, p=%d: (%g|%g)  index=%d task=%d ID=%d  flags\n", (int) (pp0), p0->x, p0->y, p0->index,
	     p0->task, P[p0->index % N_gas].ID);
      printf("xyz, p=%d: (%g|%g)  index=%d task=%d ID=%d  flags\n", (int) (pp1), p1->x, p1->y, p1->index,
	     p1->task, P[p1->index % N_gas].ID);
      printf("xyz, p=%d: (%g|%g)  index=%d task=%d ID=%d  flags\n", (int) (pp2), p2->x, p2->y, p2->index,
	     p2->task, P[p2->index % N_gas].ID);
      printf("xyz, p=%d: (%g|%g)  index=%d task=%d ID=%d  flags\n", (int) (pp), p->x, p->y, p->index,
	     p->task, P[p->index % N_gas].ID);


      endrun(1312399812);
    }

  if(flag0 >= 0 && flag1 >= 0 && flag2 >= 0)
    {

      /* we have a point inside the triangle, but it may still be on one of the edges */

      if(count_zeros == 0)
	{
	  /* ok, we are inside */
	  *degnerate_flag = 1;
	  return 1;
	}

      if(count_zeros == 1)	/* we lie on a face */
	{
	  if(flag2 == 0)
	    {
	      *degnerate_flag = 12;
	      return 12;	/* point lies on side A */
	    }
	  if(flag1 == 0)
	    {
	      *degnerate_flag = 11;
	      return 11;	/* point lies on side C */
	    }

	  if(flag0 == 0)
	    {
	      *degnerate_flag = 10;
	      return 10;	/* point lies on side B */
	    }
	}
    }

  /* we are clearly outside, let's select the suitable neighbour */

  if(flag0 < 0 && flag1 >= 0 && flag2 >= 0)
    {
      *nexttetra = t->t[0];
      return 0;
    }

  if(flag0 >= 0 && flag1 < 0 && flag2 >= 0)
    {
      *nexttetra = t->t[1];
      return 0;
    }

  if(flag0 >= 0 && flag1 >= 0 && flag2 < 0)
    {
      *nexttetra = t->t[2];
      return 0;
    }

  /* there are apparently two negative values. Let's pick a random one */

  int ind = -1;

  if(flag0 < 0)
    {
      if(ind < 0)
	ind = 0;
      else
	{
	  if(drand48() < 0.5)
	    ind = 0;
	}
    }

  if(flag1 < 0)
    {
      if(ind < 0)
	ind = 0;
      else
	{
	  if(drand48() < 0.5)
	    ind = 0;
	}
    }

  if(flag2 < 0)
    {
      if(ind < 0)
	ind = 0;
      else
	{
	  if(drand48() < 0.5)
	    ind = 0;
	}
    }

  *nexttetra = t->t[ind];
  return 0;
}


/* tests whether point p lies in the circumcircle around triangle p0,p1,p3 */

int InCircle_Quick(int pp0, int pp1, int pp2, int pp)
{
  point *p0 = &DP[pp0];

  point *p1 = &DP[pp1];

  point *p2 = &DP[pp2];

  point *p = &DP[pp];

  double ax, ay, bx, by, cx, cy;

  double ab, bc, ca, a2, b2, c2, x;

  if(pp0 == DPinfinity || pp1 == DPinfinity || pp2 == DPinfinity || pp == DPinfinity)
    return -1;

  ax = p0->xx - p->xx;
  ay = p0->yy - p->yy;
  bx = p1->xx - p->xx;
  by = p1->yy - p->yy;
  cx = p2->xx - p->xx;
  cy = p2->yy - p->yy;

  ab = ax * by - bx * ay;
  bc = bx * cy - cx * by;
  ca = cx * ay - ax * cy;

  a2 = ax * ax + ay * ay;
  b2 = bx * bx + by * by;
  c2 = cx * cx + cy * cy;

  x = a2 * bc + b2 * ca + c2 * ab;

  if(x < 0)
    return -1;
  if(x > 0)
    return +1;

  return 0;
}


int InCircle_Errorbound(int pp0, int pp1, int pp2, int pp)
{
  point *p0 = &DP[pp0];

  point *p1 = &DP[pp1];

  point *p2 = &DP[pp2];

  point *p = &DP[pp];

  if(pp0 == DPinfinity || pp1 == DPinfinity || pp2 == DPinfinity || pp == DPinfinity)
    return -1;

  double ax, ay, bx, by, cx, cy;

  double ab, bc, ca, a2, b2, c2, x;

  double axby, bxay, bxcy, cxby, cxay, axcy;

  ax = p0->xx - p->xx;
  ay = p0->yy - p->yy;
  bx = p1->xx - p->xx;
  by = p1->yy - p->yy;
  cx = p2->xx - p->xx;
  cy = p2->yy - p->yy;

  axby = ax * by;
  bxay = bx * ay;
  bxcy = bx * cy;
  cxby = cx * by;
  cxay = cx * ay;
  axcy = ax * cy;

  ca = cxay - axcy;
  ab = axby - bxay;
  bc = bxcy - cxby;

  a2 = ax * ax + ay * ay;
  b2 = bx * bx + by * by;
  c2 = cx * cx + cy * cy;

  x = a2 * bc + b2 * ca + c2 * ab;

  /* calculate absolute maximum size */

  double sizelimit =
    a2 * (fabs(bxcy) + fabs(cxby)) + b2 * (fabs(cxay) + fabs(axcy)) + c2 * (fabs(axby) + fabs(bxay));

  double errbound = 1.0e-14 * sizelimit;

  if(x < -errbound)
    return -1;
  else if(x > errbound)
    return +1;

  return 0;
}


int InCircle_Exact(int pp0, int pp1, int pp2, int pp)
{
  point *p0 = &DP[pp0];

  point *p1 = &DP[pp1];

  point *p2 = &DP[pp2];

  point *p = &DP[pp];

  if(pp0 == DPinfinity || pp1 == DPinfinity || pp2 == DPinfinity || pp == DPinfinity)
    return -1;

  IntegerMapType ax, ay, bx, by, cx, cy;

  ax = p0->ix - p->ix;
  ay = p0->iy - p->iy;
  bx = p1->ix - p->ix;
  by = p1->iy - p->iy;
  cx = p2->ix - p->ix;
  cy = p2->iy - p->iy;

  mpz_t axby, bxay, bxcy, cxby, cxay, axcy, tmp;

  mpz_init(tmp);

  mpz_init(axby);
  MY_mpz_set_si(tmp, ax);
  MY_mpz_mul_si(axby, tmp, by);
  mpz_init(bxay);
  MY_mpz_set_si(tmp, bx);
  MY_mpz_mul_si(bxay, tmp, ay);
  mpz_init(bxcy);
  MY_mpz_set_si(tmp, bx);
  MY_mpz_mul_si(bxcy, tmp, cy);
  mpz_init(cxby);
  MY_mpz_set_si(tmp, cx);
  MY_mpz_mul_si(cxby, tmp, by);
  mpz_init(cxay);
  MY_mpz_set_si(tmp, cx);
  MY_mpz_mul_si(cxay, tmp, ay);
  mpz_init(axcy);
  MY_mpz_set_si(tmp, ax);
  MY_mpz_mul_si(axcy, tmp, cy);

  mpz_t ca, ab, bc;

  mpz_init(ca);
  mpz_init(ab);
  mpz_init(bc);

  mpz_sub(ca, cxay, axcy);
  mpz_sub(ab, axby, bxay);
  mpz_sub(bc, bxcy, cxby);


  mpz_t AA, BB, a2, b2, c2;

  mpz_init(AA);
  mpz_init(BB);
  mpz_init(a2);
  mpz_init(b2);
  mpz_init(c2);

  MY_mpz_set_si(tmp, ax);
  MY_mpz_mul_si(AA, tmp, ax);
  MY_mpz_set_si(tmp, ay);
  MY_mpz_mul_si(BB, tmp, ay);
  mpz_add(a2, AA, BB);

  MY_mpz_set_si(tmp, bx);
  MY_mpz_mul_si(AA, tmp, bx);
  MY_mpz_set_si(tmp, by);
  MY_mpz_mul_si(BB, tmp, by);
  mpz_add(b2, AA, BB);

  MY_mpz_set_si(tmp, cx);
  MY_mpz_mul_si(AA, tmp, cx);
  MY_mpz_set_si(tmp, cy);
  MY_mpz_mul_si(BB, tmp, cy);
  mpz_add(c2, AA, BB);

  /* now calculate the final result */

  mpz_mul(AA, a2, bc);
  mpz_mul(BB, b2, ca);
  mpz_add(tmp, AA, BB);
  mpz_mul(BB, c2, ab);
  mpz_add(AA, BB, tmp);

  int sign = mpz_sgn(AA);

  mpz_clear(c2);
  mpz_clear(b2);
  mpz_clear(a2);
  mpz_clear(BB);
  mpz_clear(AA);
  mpz_clear(bc);
  mpz_clear(ab);
  mpz_clear(ca);
  mpz_clear(axcy);
  mpz_clear(cxay);
  mpz_clear(cxby);
  mpz_clear(bxcy);
  mpz_clear(bxay);
  mpz_clear(axby);
  mpz_clear(tmp);

  return sign;
}




double test_triangle_orientation(int pp0, int pp1, int pp2)
{
  point *p0 = &DP[pp0];

  point *p1 = &DP[pp1];

  point *p2 = &DP[pp2];

  return (p1->x - p0->x) * (p2->y - p0->y) - (p1->y - p0->y) * (p2->x - p0->x);
}


int Orient2d_Quick(int pp0, int pp1, int pp2)
{
  point *p0 = &DP[pp0];

  point *p1 = &DP[pp1];

  point *p2 = &DP[pp2];

  double x;

  x = (p1->xx - p0->xx) * (p2->yy - p0->yy) - (p1->yy - p0->yy) * (p2->xx - p0->xx);

  if(x < 0)
    return -1;
  if(x > 0)
    return +1;
  return 0;
}

int Orient2d_Exact(int pp0, int pp1, int pp2)
{
  point *p0 = &DP[pp0];

  point *p1 = &DP[pp1];

  point *p2 = &DP[pp2];

#if USEDBITS > 31
  IntegerMapType dx1, dy1, dx2, dy2;

  dx1 = (p1->ix - p0->ix);
  dy1 = (p1->iy - p0->iy);
  dx2 = (p2->ix - p0->ix);
  dy2 = (p2->iy - p0->iy);

  mpz_t dx1dy2, dx2dy1, tmp;

  mpz_init(tmp);
  mpz_init(dx1dy2);
  mpz_init(dx2dy1);

  MY_mpz_set_si(tmp, dx1);
  MY_mpz_mul_si(dx1dy2, tmp, dy2);

  MY_mpz_set_si(tmp, dx2);
  MY_mpz_mul_si(dx2dy1, tmp, dy1);

  mpz_sub(tmp, dx1dy2, dx2dy1);

  int sign = mpz_sgn(tmp);

  mpz_clear(dx2dy1);
  mpz_clear(dx1dy2);
  mpz_clear(tmp);

  return (sign);

#else
  signed long long dx1, dy1, dx2, dy2, x;

  dx1 = (p1->ix - p0->ix);
  dy1 = (p1->iy - p0->iy);
  dx2 = (p2->ix - p0->ix);
  dy2 = (p2->iy - p0->iy);

  x = dx1 * dy2 - dy1 * dx2;

  if(x < 0)
    return -1;
  if(x > 0)
    return +1;
  return 0;
#endif
}





const int edge_start[3] = { 1, 2, 0 };
const int edge_end[3] = { 2, 0, 1 };


void process_edge_faces_and_volumes(int tt, int nr)
{
  int i, j, qq;

  face *f;

  tetra *q;

  double nx, ny;

  double sx, sy;

  double hx, hy;

  double dvol, h;

  if(Nvf + 1 >= MaxNvf)
    {
      Indi.AllocFacNvf *= ALLOC_INCREASE_FACTOR;
      MaxNvf = Indi.AllocFacNvf;
#ifdef VERBOSE
      printf("Task=%d: increase memory allocation, MaxNvf=%d Indi.AllocFacNvf=%g\n",
	     ThisTask, MaxNvf, Indi.AllocFacNvf);
#endif
      VF = myrealloc_movable(VF, MaxNvf * sizeof(face));

      if(Nvf + 1 >= MaxNvf)
	terminate("Nvf larger than MaxNvf");
    }

  tetra *t = &DT[tt];

  i = edge_start[nr];
  j = edge_end[nr];

  point *dpi = &DP[t->p[i]];

  point *dpj = &DP[t->p[j]];

  qq = t->t[nr];
  q = &DT[qq];

  Edge_visited[tt] |= (1 << nr);
  Edge_visited[qq] |= (1 << (t->s[nr]));


  f = &VF[Nvf++];

  f->p1 = t->p[i];
  f->p2 = t->p[j];

  f->cx = 0.5 * (DTC[tt].cx + DTC[qq].cx);
  f->cy = 0.5 * (DTC[tt].cy + DTC[qq].cy);
  f->cz = 0;


  nx = DTC[tt].cx - DTC[qq].cx;
  ny = DTC[tt].cy - DTC[qq].cy;

  f->area = sqrt(nx * nx + ny * ny);

#ifdef VORONOI_SHAPESCHEME

  double ax = DTC[tt].cx - f->cx;
  double ay = DTC[tt].cy - f->cy;
  double bx = DTC[qq].cx - f->cx;
  double by = DTC[qq].cy - f->cy;

  f->T_xx = 1.0 / 3 * (ax * ax + bx * bx) + (1.0 / 6) * (ax * bx + ax * bx);
  f->T_yy = 1.0 / 3 * (ay * ay + by * by) + (1.0 / 6) * (ay * by + ay * by);
  f->T_xy = 1.0 / 3 * (ax * ay + bx * by) + (1.0 / 6) * (ax * by + ay * bx);

  f->T_zz = f->T_xz = f->T_yz = 0;
#endif


  hx = 0.5 * (dpi->x - dpj->x);
  hy = 0.5 * (dpi->y - dpj->y);

  h = sqrt(hx * hx + hy * hy);
  dvol = 0.5 * f->area * h;

  if(dpi->task == ThisTask && dpi->index >= 0 && dpi->index < N_gas)
    {
      if(TimeBinActive[P[dpi->index].TimeBin])
	{
	  SphP[dpi->index].Volume += dvol;

#ifdef VORONOI_SHAPESCHEME
	  double kx = DTC[tt].cx - dpi->x;
	  double ky = DTC[tt].cy - dpi->y;
	  double gx = DTC[qq].cx - dpi->x;
	  double gy = DTC[qq].cy - dpi->y;

	  SphP[dpi->index].W += dvol / 6.0 * (kx * kx + ky * ky + kx * gx + ky * gy + gx * gx + gy * gy);
#endif

#ifdef OUTPUT_SURFACE_AREA
	  if(f->area)
	    SphP[dpi->index].CountFaces++;
#endif

#if defined(REFINEMENT_SPLIT_CELLS)
	  if(SphP[dpi->index].MinimumEdgeDistance > h)
	    SphP[dpi->index].MinimumEdgeDistance = h;
#endif
	  /* let's now compute the center-of-mass of the pyramid at the bottom top */
	  sx = (2.0 / 3) * f->cx + (1.0 / 3) * dpi->x;
	  sy = (2.0 / 3) * f->cy + (1.0 / 3) * dpi->y;

	  SphP[dpi->index].Center[0] += dvol * sx;
	  SphP[dpi->index].Center[1] += dvol * sy;
	}
    }


  if(dpj->task == ThisTask && dpj->index >= 0 && dpj->index < N_gas)
    {
      if(TimeBinActive[P[dpj->index].TimeBin])
	{
	  SphP[dpj->index].Volume += dvol;

#ifdef VORONOI_SHAPESCHEME
	  double kx = DTC[tt].cx - dpj->x;
	  double ky = DTC[tt].cy - dpj->y;
	  double gx = DTC[qq].cx - dpj->x;
	  double gy = DTC[qq].cy - dpj->y;

	  SphP[dpj->index].W += dvol / 6.0 * (kx * kx + ky * ky + kx * gx + ky * gy + gx * gx + gy * gy);
#endif

#ifdef OUTPUT_SURFACE_AREA
	  if(f->area)
	    SphP[dpj->index].CountFaces++;
#endif

#if defined(REFINEMENT_SPLIT_CELLS)
	  if(SphP[dpj->index].MinimumEdgeDistance > h)
	    SphP[dpj->index].MinimumEdgeDistance = h;
#endif

	  /* let's now compute the center-of-mass of the pyramid on top */
	  sx = (2.0 / 3) * f->cx + (1.0 / 3) * dpj->x;
	  sy = (2.0 / 3) * f->cy + (1.0 / 3) * dpj->y;

	  SphP[dpj->index].Center[0] += dvol * sx;
	  SphP[dpj->index].Center[1] += dvol * sy;
	}
    }
}



void compute_circumcircles(void)
{
  int i;

  for(i = 0; i < Ndt; i++)
    {
      if(DTF[i] & 1)
	continue;
      DTF[i] |= 1;

      if(DT[i].p[0] == DPinfinity)
	continue;
      if(DT[i].p[1] == DPinfinity)
	continue;
      if(DT[i].p[2] == DPinfinity)
	continue;

      update_circumcircle(i);
    }
}


void update_circumcircle(int tt)
{
  tetra *t = &DT[tt];

  point *p0, *p1, *p2;

  int pp0, pp1, pp2;

  pp0 = t->p[0];
  pp1 = t->p[1];
  pp2 = t->p[2];

  p0 = &DP[pp0];
  p1 = &DP[pp1];
  p2 = &DP[pp2];

  if(t->p[0] == DPinfinity)
    return;
  if(t->p[1] == DPinfinity)
    return;
  if(t->p[2] == DPinfinity)
    return;

  double ax = p1->xx - p0->xx;

  double ay = p1->yy - p0->yy;

  double bx = p2->xx - p0->xx;

  double by = p2->yy - p0->yy;

  double aa = 0.5 * (ax * ax + ay * ay);

  double bb = 0.5 * (bx * bx + by * by);

  double mv_data[] = { ax, ay, aa, bx, by, bb };
  double x[2];

  int status = solve_linear_equations_2d(mv_data, x);

  if(status < 0)
    {
      terminate("trouble in circum-circle calculation\n");
    }
  else
    {
      x[0] += p0->xx;
      x[1] += p0->yy;

      DTC[tt].cx = (x[0] - 1.0) / ConversionFac + CentralOffsetX;
      DTC[tt].cy = (x[1] - 1.0) / ConversionFac + CentralOffsetY;
      DTC[tt].cz = 0;
    }
}




void set_integers_for_point(int pp)
{
  point *p = &DP[pp];

  p->xx = (p->x - CentralOffsetX) * ConversionFac + 1.0;
  p->yy = (p->y - CentralOffsetY) * ConversionFac + 1.0;

  if(p->xx < 1.0 || p->xx >= 2.0 || p->yy < 1.0 || p->yy >= 2.0)
    {
      printf("Task=%d xx=%g yy=%g | x=%g y=%g  point=%d\n", ThisTask,
	     p->xx, p->yy, p->x, p->y, (int) (p - DP));

      printf("pp=%d thistask=%d  DP[pp].task=%d  DP[pp].index=%d DP[pp].ID=%d  N_gas=%d\n",
	     pp, ThisTask, DP[pp].task, DP[pp].index, DP[pp].ID, N_gas);

      printf("ConversionFac=%g CentralOffsetX/Y=%g|%g", ConversionFac, CentralOffsetX, CentralOffsetY);
      terminate("point falls outside region");
    }

  p->ix = double_to_voronoiint(p->xx);
  p->iy = double_to_voronoiint(p->yy);

  p->xx = mask_voronoi_int(p->xx);
  p->yy = mask_voronoi_int(p->yy);
}






void write_voronoi_mesh(char *fname, int writeTask, int lastTask)
{
  CPU_Step[CPU_MISC] += measure_time();

  FILE *fd;

  char msg[1000];

  MPI_Status status;

  int i, j, k, MaxNel, Nel;

  int ngas_tot, nel_tot, ndt_tot, nel_before, ndt_before, task;

  int *EdgeList, *Nedges, *NedgesOffset, *whichtetra;

  int *ngas_list, *nel_list, *ndt_list, *tmp;

  float *xyz_edges;

  tetra *q, *qstart;

  DTC = mymalloc_movable(&DTC, "DTC", MaxNdt * sizeof(tetra_center));
  DTF = mymalloc_movable(&DTF, "DTF", MaxNdt * sizeof(char));
  for(i = 0; i < Ndt; i++)
    DTF[i] = 0;

  compute_circumcircles();

  MaxNel = 10 * N_gas;		/* max edge list */
  Nel = 0;			/* length of edge list */

  EdgeList = mymalloc("EdgeList", MaxNel * sizeof(int));
  Nedges = mymalloc("Nedges", N_gas * sizeof(int));
  NedgesOffset = mymalloc("NedgesOffset", N_gas * sizeof(int));
  whichtetra = mymalloc("whichtetra", N_gas * sizeof(int));
  xyz_edges = mymalloc("xyz_edges", Ndt * DIMS * sizeof(float));
  ngas_list = mymalloc("ngas_list", sizeof(int) * NTask);
  nel_list = mymalloc("nel_list", sizeof(int) * NTask);
  ndt_list = mymalloc("ndt_list", sizeof(int) * NTask);

  for(i = 0; i < Ndt; i++)
    {
      xyz_edges[i * DIMS + 0] = DTC[i].cx;
      xyz_edges[i * DIMS + 1] = DTC[i].cy;
    }

  for(i = 0; i < N_gas; i++)
    {
      Nedges[i] = 0;
      whichtetra[i] = -1;
    }

  for(i = 0; i < Ndt; i++)
    {
      for(j = 0; j < DIMS + 1; j++)
	if(DP[DT[i].p[j]].task == ThisTask && DP[DT[i].p[j]].index >= 0 && DP[DT[i].p[j]].index < N_gas)
	  whichtetra[DP[DT[i].p[j]].index] = i;
    }

  for(i = 0; i < N_gas; i++)
    {
      if(whichtetra[i] < 0)
	continue;

      qstart = q = &DT[whichtetra[i]];

      do
	{
	  Nedges[i]++;

	  if(Nel >= MaxNel)
	    terminate("Nel >= MaxNel");

	  EdgeList[Nel++] = q - DT;

	  for(j = 0; j < 3; j++)
	    if(DP[q->p[j]].task == ThisTask && DP[q->p[j]].index == i)
	      break;

	  k = j + 1;
	  if(k >= 3)
	    k -= 3;

	  q = &DT[q->t[k]];
	}
      while(q != qstart);
    }

  for(i = 1, NedgesOffset[0] = 0; i < N_gas; i++)
    NedgesOffset[i] = NedgesOffset[i - 1] + Nedges[i - 1];


  /* determine particle numbers and number of edges in file */

  if(ThisTask == writeTask)
    {
      ngas_tot = N_gas;
      nel_tot = Nel;
      ndt_tot = Ndt;

      for(task = writeTask + 1; task <= lastTask; task++)
	{
	  MPI_Recv(&ngas_list[task], 1, MPI_INT, task, TAG_LOCALN, MPI_COMM_WORLD, &status);
	  MPI_Recv(&nel_list[task], 1, MPI_INT, task, TAG_LOCALN + 1, MPI_COMM_WORLD, &status);
	  MPI_Recv(&ndt_list[task], 1, MPI_INT, task, TAG_LOCALN + 2, MPI_COMM_WORLD, &status);

	  MPI_Send(&nel_tot, 1, MPI_INT, task, TAG_N, MPI_COMM_WORLD);
	  MPI_Send(&ndt_tot, 1, MPI_INT, task, TAG_N + 1, MPI_COMM_WORLD);

	  ngas_tot += ngas_list[task];
	  nel_tot += nel_list[task];
	  ndt_tot += ndt_list[task];
	}

      if(!(fd = fopen(fname, "w")))
	{
	  sprintf(msg, "can't open file `%s' for writing snapshot.\n", fname);
	  terminate(msg);
	}

      my_fwrite(&ngas_tot, sizeof(int), 1, fd);
      my_fwrite(&nel_tot, sizeof(int), 1, fd);
      my_fwrite(&ndt_tot, sizeof(int), 1, fd);

      my_fwrite(Nedges, sizeof(int), N_gas, fd);
      for(task = writeTask + 1; task <= lastTask; task++)
	{
	  tmp = mymalloc("tmp", sizeof(int) * ngas_list[task]);
	  MPI_Recv(tmp, ngas_list[task], MPI_INT, task, TAG_N + 2, MPI_COMM_WORLD, &status);
	  my_fwrite(tmp, sizeof(int), ngas_list[task], fd);
	  myfree(tmp);
	}

      my_fwrite(NedgesOffset, sizeof(int), N_gas, fd);
      for(task = writeTask + 1; task <= lastTask; task++)
	{
	  tmp = mymalloc("tmp", sizeof(int) * ngas_list[task]);
	  MPI_Recv(tmp, ngas_list[task], MPI_INT, task, TAG_N + 3, MPI_COMM_WORLD, &status);
	  my_fwrite(tmp, sizeof(int), ngas_list[task], fd);
	  myfree(tmp);
	}

      my_fwrite(EdgeList, sizeof(int), Nel, fd);
      for(task = writeTask + 1; task <= lastTask; task++)
	{
	  tmp = mymalloc("tmp", sizeof(int) * nel_list[task]);
	  MPI_Recv(tmp, nel_list[task], MPI_INT, task, TAG_N + 4, MPI_COMM_WORLD, &status);
	  my_fwrite(tmp, sizeof(int), nel_list[task], fd);
	  myfree(tmp);
	}

      my_fwrite(xyz_edges, sizeof(float), Ndt * DIMS, fd);
      for(task = writeTask + 1; task <= lastTask; task++)
	{
	  tmp = mymalloc("tmp", sizeof(float) * DIMS * ndt_list[task]);
	  MPI_Recv(tmp, sizeof(float) * DIMS * ndt_list[task], MPI_BYTE, task, TAG_N + 5, MPI_COMM_WORLD,
		   &status);
	  my_fwrite(tmp, sizeof(float), DIMS * ndt_list[task], fd);
	  myfree(tmp);
	}

      fclose(fd);
    }
  else
    {
      MPI_Send(&N_gas, 1, MPI_INT, writeTask, TAG_LOCALN, MPI_COMM_WORLD);
      MPI_Send(&Nel, 1, MPI_INT, writeTask, TAG_LOCALN + 1, MPI_COMM_WORLD);
      MPI_Send(&Ndt, 1, MPI_INT, writeTask, TAG_LOCALN + 2, MPI_COMM_WORLD);

      MPI_Recv(&nel_before, 1, MPI_INT, writeTask, TAG_N, MPI_COMM_WORLD, &status);
      MPI_Recv(&ndt_before, 1, MPI_INT, writeTask, TAG_N + 1, MPI_COMM_WORLD, &status);

      for(i = 0; i < N_gas; i++)
	NedgesOffset[i] += nel_before;
      for(i = 0; i < Nel; i++)
	EdgeList[i] += ndt_before;

      MPI_Send(Nedges, N_gas, MPI_INT, writeTask, TAG_N + 2, MPI_COMM_WORLD);
      MPI_Send(NedgesOffset, N_gas, MPI_INT, writeTask, TAG_N + 3, MPI_COMM_WORLD);
      MPI_Send(EdgeList, Nel, MPI_INT, writeTask, TAG_N + 4, MPI_COMM_WORLD);
      MPI_Send(xyz_edges, sizeof(float) * DIMS * Ndt, MPI_BYTE, writeTask, TAG_N + 5, MPI_COMM_WORLD);
    }

  myfree(ndt_list);
  myfree(nel_list);
  myfree(ngas_list);
  myfree(xyz_edges);
  myfree(whichtetra);
  myfree(NedgesOffset);
  myfree(Nedges);
  myfree(EdgeList);

  if(ThisTask == 0)
    printf("wrote Voronoi mesh to file\n");

  myfree(DTF);
  myfree(DTC);
  DTC = NULL;
}



#ifdef VORONOI_FIELD_DUMP_PIXELS_X
void do_special_dump(int num)
{
  CPU_Step[CPU_MISC] += measure_time();

  char buf[1000];

  int pixels_x, pixels_y;

  float *dens, *denssum, *dp;

#ifdef TRACER_FIELD
  float *tracer, *tracersum, tracer_L;
#endif
  FILE *fd = 0;

  int p;

  int t0, tstart, trow;

  double l_dx, l_dy;

  int i, j, k, kmin, li, moves, ret, no, task;

  double r2, r2min, rho_L;

  peanokey key;

  sprintf(buf, "%s/density_field_%03d", All.OutputDir, num);

  pixels_x = VORONOI_FIELD_DUMP_PIXELS_X;
  pixels_y = VORONOI_FIELD_DUMP_PIXELS_Y;

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
    }

  p = Ndp;

  if(ThisTask == 0)
    {
      if(!(fd = fopen(buf, "w")))
	{
	  printf("can't open file `%s' for writing snapshot.\n", buf);
	  endrun(123);
	}

      my_fwrite(&pixels_x, sizeof(int), 1, fd);
      my_fwrite(&pixels_y, sizeof(int), 1, fd);
    }

  dens = mymalloc("dens", pixels_x * pixels_y * sizeof(float));
  denssum = mymalloc("denssum", pixels_x * pixels_y * sizeof(float));

  for(i = 0, dp = dens; i < pixels_x; i++)
    for(j = 0; j < pixels_y; j++)
      *dp++ = 0;

#ifdef TRACER_FIELD
  tracer = mymalloc(pixels_x * pixels_y * sizeof(float));
  tracersum = mymalloc(pixels_x * pixels_y * sizeof(float));

  for(i = 0, dp = tracer; i < pixels_x; i++)
    for(j = 0; j < pixels_y; j++)
      *dp++ = 0;

#endif

  trow = 0;

  for(i = 0; i < pixels_x; i++)
    {
      tstart = trow;

      for(j = 0; j < pixels_y; j++)
	{
	  DP[p].x = (i + 0.49) / pixels_x * boxSize_X;
	  DP[p].y = (j + 0.49) / pixels_y * boxSize_Y;
	  DP[p].z = 0;

#ifdef VORONOI_FIELD_COMPENSATE_VX
	  DP[p].x += All.Time * VORONOI_FIELD_COMPENSATE_VX;
	  while(DP[p].x >= boxSize_X)
	    DP[p].x -= boxSize_X;
	  while(DP[p].x < 0)
	    DP[p].x += boxSize_X;
#endif

#ifdef VORONOI_FIELD_COMPENSATE_VY
	  DP[p].y += All.Time * VORONOI_FIELD_COMPENSATE_VY;
	  while(DP[p].y >= boxSize_Y)
	    DP[p].y -= boxSize_Y;
	  while(DP[p].y < 0)
	    DP[p].y += boxSize_Y;
#endif

	  key = peano_hilbert_key((int) ((DP[p].x - DomainCorner[0]) * DomainFac),
				  (int) ((DP[p].y - DomainCorner[1]) * DomainFac),
				  (int) ((0 - DomainCorner[2]) * DomainFac), BITS_PER_DIMENSION);

	  no = 0;
	  while(TopNodes[no].Daughter >= 0)
	    no = TopNodes[no].Daughter + (key - TopNodes[no].StartKey) / (TopNodes[no].Size / 8);

	  no = TopNodes[no].Leaf;
	  task = DomainTask[no];

	  if(task == ThisTask)
	    {
	      set_integers_for_point(p);

	      t0 = get_triangle(p, &moves, &ret, tstart);

	      for(k = 0, kmin = -1, r2min = 1.0e30; k < 3; k++)
		{
		  r2 = (DP[p].x - DP[DT[t0].p[k]].x) * (DP[p].x - DP[DT[t0].p[k]].x) +
		    (DP[p].y - DP[DT[t0].p[k]].y) * (DP[p].y - DP[DT[t0].p[k]].y);
		  if(r2 < r2min)
		    {
		      r2min = r2;
		      kmin = k;
		    }
		}



	      li = DP[DT[t0].p[kmin]].index;

	      /*
	         if(ThisTask == 0 && num==1)
	         {
	         printf("x/y=%g|%g i=%d j=%d (%g|%g) (%g|%g) (%g|%g)\n",
	         DP[p].x, DP[p].y, i, j,
	         DP[DT[t0].p[0]].x, DP[DT[t0].p[0]].y,
	         DP[DT[t0].p[1]].x, DP[DT[t0].p[1]].y,
	         DP[DT[t0].p[2]].x, DP[DT[t0].p[2]].y);

	         printf("kmin=%d\n", kmin);
	         printf("DP[DT[t0].p[kmin]].task=%d  DP[DT[t0].p[kmin]].index=%d\n",
	         DP[DT[t0].p[kmin]].task, DP[DT[t0].p[kmin]].index);

	         printf("center=%g|%g\n", SphP[li].Center[0], SphP[li].Center[1]);

	         }
	       */


	      if(DP[DT[t0].p[kmin]].task == ThisTask)
		{

		  if(li >= N_gas)
		    {
		      li -= N_gas;
		    }

		  l_dx = DP[p].x - SphP[li].Center[0];
		  l_dy = DP[p].y - SphP[li].Center[1];
		}
	      else
		{
		  l_dx = DP[p].x - PrimExch[li].Center[0];
		  l_dy = DP[p].y - PrimExch[li].Center[1];
		}
#ifdef PERIODIC
#if !defined(REFLECTIVE_X)
	      if(l_dx < -boxHalf_X)
		l_dx += boxSize_X;
	      if(l_dx > boxHalf_X)
		l_dx -= boxSize_X;
#endif
#if !defined(REFLECTIVE_Y)
	      if(l_dy < -boxHalf_Y)
		l_dy += boxSize_Y;
	      if(l_dy > boxHalf_Y)
		l_dy -= boxSize_Y;
#endif
#endif

	      /*
	         if(fabs(l_dx) > 0.012 || fabs(l_dy) > 0.012)
	         {
	         printf("sqrt(r2)=%g l_dx=%g l_dy=%g\n", r2min, l_dx, l_dy);
	         //endrun(122);
	         }
	       */


	      if(DP[DT[t0].p[kmin]].task == ThisTask)
		{
		  rho_L = SphP[li].Density + SphP[li].Grad.drho[0] * l_dx + SphP[li].Grad.drho[1] * l_dy;
#ifdef TRACER_FIELD
		  tracer_L =
		    SphP[li].Tracer + SphP[li].Grad.dtracer[0] * l_dx + SphP[li].Grad.dtracer[1] * l_dy;
#endif
		}
	      else
		{
		  rho_L = PrimExch[li].Density + GradExch[li].drho[0] * l_dx + GradExch[li].drho[1] * l_dy;
#ifdef TRACER_FIELD
		  tracer_L =
		    PrimExch[li].Tracer + GradExch[li].dtracer[0] * l_dx + GradExch[li].dtracer[1] * l_dy;
#endif
		}

	      dens[i * pixels_y + j] = rho_L;

	      /*
	         if(ThisTask == 0 && num==1)
	         {

	         printf("rho_L=%g %g  i=%d j=%d\n", rho_L, dens[0], i, j);
	         endrun(1212);
	         }
	       */


#ifdef TRACER_FIELD
	      tracer[i * pixels_y + j] = tracer_L;
#endif
	      tstart = t0;

	      if(j == 0)
		trow = t0;
	    }
	}
    }


  MPI_Reduce(dens, denssum, pixels_x * pixels_y, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
#ifdef TRACER_FIELD

  MPI_Reduce(tracer, tracersum, pixels_x * pixels_y, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
#endif

  if(ThisTask == 0)
    {
      my_fwrite(denssum, sizeof(float), pixels_x * pixels_y, fd);
#ifdef TRACER_FIELD
      my_fwrite(tracersum, si zeof(float), pixels_x * pixels_y, fd);
#endif
      fclose(fd);
    }

#ifdef TRACER_FIELD
  myfree(tracersum);
  myfree(tracer);
#endif

  myfree(denssum);
  myfree(dens);

  CPU_Step[CPU_MAKEIMAGES] += measure_time();


  /*
     if(num==1)
     {
     dump_particles();
     endrun(0);
     }
   */
}
#endif




#endif


#endif
