#include <mpi.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

#include "allvars.h"
#include "proto.h"

#define TINY 1.0e-20;
#define NR_END 1
#define FREE_ARG char*

#ifdef DISTORTIONTENSORPS
/* error message */
void math_error(char error_text[])
{
  fprintf(stderr, "Phase-space run-time WARNING:\n");
  fprintf(stderr, "%s\n", error_text);
  fflush(stderr);
#ifndef GDE_DEBUG
  endrun(1012);
#endif
}

/* allocate a MyDouble vector with subscript range v[nl..nh] */
MyDouble *vector(long nl, long nh)
{
  MyDouble *v;

  v = (MyDouble *) mymalloc("v", (size_t) ((nh - nl + 1 + NR_END) * sizeof(MyDouble)));
  if(!v)
    math_error("phasespace_math.c: vector() -> memory allocation failed");
  return v - nl + NR_END;
}

/* free a MyDouble vector allocated with vector() */
void free_vector(MyDouble * v, long nl, long nh)
{
  myfree((FREE_ARG) (v + nl - NR_END));
}

/* allocate a MyDouble matrix with subscript range m[nrl..nrh][ncl..nch] */
MyDouble **matrix(long nrl, long nrh, long ncl, long nch)
{
  long i, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
  MyDouble **m;

  /* allocate pointers to rows */
  m = (MyDouble **) mymalloc("m", (size_t) ((nrow + NR_END) * sizeof(MyDouble *)));
  if(!m)
    math_error("phasespace_math.c: matrix() -> memory allocation failed (1)");
  m += NR_END;
  m -= nrl;

  /* allocate rows and set pointers to them */
  m[nrl] = (MyDouble *) mymalloc("m[nrl]", (size_t) ((nrow * ncol + NR_END) * sizeof(MyDouble)));
  if(!m[nrl])
    math_error("phasespace_math.c: matrix() -> memory allocation failed (2)");
  m[nrl] += NR_END;
  m[nrl] -= ncl;

  for(i = nrl + 1; i <= nrh; i++)
    m[i] = m[i - 1] + ncol;

  /* return pointer to array of pointers to rows */
  return m;
}

/* free a MyDouble matrix allocated by matrix() */
void free_matrix(MyDouble ** m, long nrl, long nrh, long ncl, long nch)
{
  myfree((FREE_ARG) (m[nrl] + ncl - NR_END));
  myfree((FREE_ARG) (m + nrl - NR_END));
}

/* matrix multiplication */
void mult_matrix(MyDouble ** matrix_a, MyDouble ** matrix_b, int dimension, MyDouble ** matrix_result)
{
  int counter_x, counter_y, counter;

  for(counter_x = 1; counter_x <= dimension; counter_x++)
    {
      for(counter_y = 1; counter_y <= dimension; counter_y++)
	{
	  matrix_result[counter_x][counter_y] = 0.0;
	  for(counter = 1; counter <= dimension; counter++)
	    {
	      matrix_result[counter_x][counter_y] +=
		matrix_a[counter_x][counter] * matrix_b[counter][counter_y];
	    }
	}
    }
}

/* matrix multiplication + tranposition */
void mult_matrix_transpose_A(MyDouble ** matrix_a, MyDouble ** matrix_b, int dimension,
			     MyDouble ** matrix_result)
{
  int counter_x, counter_y, counter;

  for(counter_x = 1; counter_x <= dimension; counter_x++)
    {
      for(counter_y = 1; counter_y <= dimension; counter_y++)
	{
	  matrix_result[counter_x][counter_y] = 0.0;
	  for(counter = 1; counter <= dimension; counter++)
	    {
	      matrix_result[counter_x][counter_y] +=
		matrix_a[counter][counter_x] * matrix_b[counter][counter_y];
	    }
	}
    }
}

/* LU decomposition of matrix */
void ludcmp(MyDouble ** a, int n, int *indx, MyDouble * d)
{
  int i, imax, j, k;
  MyDouble big, dum, sum, temp;
  MyDouble *vv;

  vv = vector(1, n);
  *d = 1.0;
  for(i = 1; i <= n; i++)
    {
      big = 0.0;
      for(j = 1; j <= n; j++)
	if((temp = fabsl(a[i][j])) > big)
	  big = temp;
      if(big == 0.0)
	math_error("phasespace_math.c: ludcmp() -> singular matrix in routine");
      vv[i] = 1.0 / big;
    }
  for(j = 1; j <= n; j++)
    {
      for(i = 1; i < j; i++)
	{
	  sum = a[i][j];
	  for(k = 1; k < i; k++)
	    sum -= a[i][k] * a[k][j];
	  a[i][j] = sum;
	}
      big = 0.0;
      for(i = j; i <= n; i++)
	{
	  sum = a[i][j];
	  for(k = 1; k < j; k++)
	    sum -= a[i][k] * a[k][j];
	  a[i][j] = sum;
	  if((dum = vv[i] * fabsl(sum)) >= big)
	    {
	      big = dum;
	      imax = i;
	    }
	}
      if(j != imax)
	{
	  for(k = 1; k <= n; k++)
	    {
	      dum = a[imax][k];
	      a[imax][k] = a[j][k];
	      a[j][k] = dum;
	    }
	  *d = -(*d);
	  vv[imax] = vv[j];
	}
      indx[j] = imax;
      if(a[j][j] == 0.0)
	a[j][j] = TINY;
      if(j != n)
	{
	  dum = 1.0 / (a[j][j]);
	  for(i = j + 1; i <= n; i++)
	    a[i][j] *= dum;
	}
    }
  free_vector(vv, 1, n);
}


void lubksb(MyDouble ** a, int n, int *indx, MyDouble b[])
{
  int i, ii = 0, ip, j;
  MyDouble sum;

  for(i = 1; i <= n; i++)
    {
      ip = indx[i];
      sum = b[ip];
      b[ip] = b[i];
      if(ii)
	for(j = ii; j <= i - 1; j++)
	  sum -= a[i][j] * b[j];
      else if(sum)
	ii = i;
      b[i] = sum;
    }
  for(i = n; i >= 1; i--)
    {
      sum = b[i];
      for(j = i + 1; j <= n; j++)
	sum -= a[i][j] * b[j];
      b[i] = sum / a[i][i];
    }
}

/* inverse matrix calculation */
void luinvert(MyDouble ** input_matrix, int n, MyDouble ** inverse_matrix)
{
  MyDouble **temp_matrix = matrix(1, n, 1, n);
  MyDouble *d = vector(1, n), *col = vector(1, n);
  int index[n];
  int counter_x, counter_y;

  /*save matrix */
  for(counter_x = 1; counter_x <= n; counter_x++)
    for(counter_y = 1; counter_y <= n; counter_y++)
      temp_matrix[counter_x][counter_y] = input_matrix[counter_x][counter_y];

  ludcmp(temp_matrix, n, index, d);

  for(counter_y = 1; counter_y <= n; counter_y++)
    {
      for(counter_x = 1; counter_x <= n; counter_x++)
	col[counter_x] = 0.0;
      col[counter_y] = 1.0;
      lubksb(temp_matrix, n, index, col);
      for(counter_x = 1; counter_x <= n; counter_x++)
	inverse_matrix[counter_x][counter_y] = col[counter_x];
    }

  free_vector(col, 1, n);
  free_vector(d, 1, n);
  free_matrix(temp_matrix, 1, n, 1, n);
}

#define ROTATE(a,i,j,k,l) g=a[i][j];h=a[k][l];a[i][j]=g-s*(h+g*tau);\
	a[k][l]=h+s*(g-h*tau);

/* sort eigensystem */
void eigsrt(MyDouble d[], MyDouble ** v, int n)
{
  int k, j, i;
  float p;

  for(i = 1; i < n; i++)
    {
      p = d[k = i];
      for(j = i + 1; j <= n; j++)
	if(d[j] >= p)
	  p = d[k = j];
      if(k != i)
	{
	  d[k] = d[i];
	  d[i] = p;
	  for(j = 1; j <= n; j++)
	    {
	      p = v[j][i];
	      v[j][i] = v[j][k];
	      v[j][k] = p;
	    }
	}
    }
}


/* note that the eigenvectors are normalized */
void jacobi(MyDouble ** a, int n, MyDouble d[], MyDouble ** v, int *nrot, MyIDType pindex)
{
  int j, iq, ip, i;
  MyDouble tresh, theta, tau, t, sm, s, h, g, c, *b, *z;

  b = vector(1, n);
  z = vector(1, n);
  for(ip = 1; ip <= n; ip++)
    {
      for(iq = 1; iq <= n; iq++)
	v[ip][iq] = 0.0;
      v[ip][ip] = 1.0;
    }
  for(ip = 1; ip <= n; ip++)
    {
      b[ip] = d[ip] = a[ip][ip];
      z[ip] = 0.0;
    }
  *nrot = 0;
  for(i = 1; i <= 10000; i++)
    {
      sm = 0.0;
      for(ip = 1; ip <= n - 1; ip++)
	{
	  for(iq = ip + 1; iq <= n; iq++)
	    sm += fabsl(a[ip][iq]);
	}
      if(sm == 0.0)
	{
	  free_vector(z, 1, n);
	  free_vector(b, 1, n);
	  return;
	}
      if(i < 4)
	tresh = 0.2 * sm / (n * n);
      else
	tresh = 0.0;
      for(ip = 1; ip <= n - 1; ip++)
	{
	  for(iq = ip + 1; iq <= n; iq++)
	    {
	      g = 100.0 * fabsl(a[ip][iq]);
	      if(i > 4 && (MyDouble) (fabsl(d[ip]) + g) == (MyDouble) fabsl(d[ip])
		 && (MyDouble) (fabsl(d[iq]) + g) == (MyDouble) fabsl(d[iq]))
		a[ip][iq] = 0.0;
	      else if(fabsl(a[ip][iq]) > tresh)
		{
		  h = d[iq] - d[ip];
		  if((MyDouble) (fabsl(h) + g) == (MyDouble) fabsl(h))
		    t = (a[ip][iq]) / h;
		  else
		    {
		      theta = 0.5 * h / (a[ip][iq]);
		      t = 1.0 / (fabsl(theta) + sqrtl(1.0 + theta * theta));
		      if(theta < 0.0)
			t = -t;
		    }
		  c = 1.0 / sqrtl(1 + t * t);
		  s = t * c;
		  tau = s / (1.0 + c);
		  h = t * a[ip][iq];
		  z[ip] -= h;
		  z[iq] += h;
		  d[ip] -= h;
		  d[iq] += h;
		  a[ip][iq] = 0.0;
		  for(j = 1; j <= ip - 1; j++)
		    {
		    ROTATE(a, j, ip, j, iq)}
		  for(j = ip + 1; j <= iq - 1; j++)
		    {
		    ROTATE(a, ip, j, j, iq)}
		  for(j = iq + 1; j <= n; j++)
		    {
		    ROTATE(a, ip, j, iq, j)}
		  for(j = 1; j <= n; j++)
		    {
		    ROTATE(v, j, ip, j, iq)}
		  ++(*nrot);
		}
	    }
	}
      for(ip = 1; ip <= n; ip++)
	{
	  b[ip] += z[ip];
	  d[ip] = b[ip];
	  z[ip] = 0.0;
	}
    }

  int ncount1, ncount2;

  fprintf(stderr, "Jacobi for particle ID=%d\n", P[pindex].ID);
  for(ncount1 = 0; ncount1 < n; ncount1++)
    for(ncount2 = 0; ncount2 < n; ncount2++)
      fprintf(stderr, "a[%d][%d] %g\n", ncount1, ncount2, a[ncount1][ncount2]);

  math_error("phasespace_math.c: jacobi() -> too many iterations");

}

#undef ROTATE

#endif /* DISTORTIONTENSORPS */
