#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "proto.h"
#include "allvars.h"


#ifdef SCFPOTENTIAL


#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1+(IM-1)/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)


/* Hernquist acceleration */
void sphere_acc(double x, double y, double z, double *xa, double *ya, double *za)
{
 double r = sqrt(x*x + y*y + z*z);
 *xa = -SCF_HQ_MASS/((r + SCF_HQ_A)*(r + SCF_HQ_A)) * x/r;
 *ya = -SCF_HQ_MASS/((r + SCF_HQ_A)*(r + SCF_HQ_A)) * y/r;
 *za = -SCF_HQ_MASS/((r + SCF_HQ_A)*(r + SCF_HQ_A)) * z/r;
}



void to_unit(MyDouble x, MyDouble y, MyDouble z, MyDouble *xs, MyDouble *ys, MyDouble *zs)
{
 *xs=x/SCF_HQ_A;
 *ys=y/SCF_HQ_A;
 *zs=z/SCF_HQ_A;
}

MyDouble ran1(long *idum)
{
        int j;
        long k;
        static long iy=0;
        static long iv[NTAB];
        MyDouble temp;

        if (*idum <= 0 || !iy) {
                if (-(*idum) < 1) *idum=1;
                else *idum = -(*idum);
                for (j=NTAB+7;j>=0;j--) {
                        k=(*idum)/IQ;
                        *idum=IA*(*idum-k*IQ)-IR*k;
                        if (*idum < 0) *idum += IM;
                        if (j < NTAB) iv[j] = *idum;
                }
                iy=iv[0];
        }
        k=(*idum)/IQ;
        *idum=IA*(*idum-k*IQ)-IR*k;
        if (*idum < 0) *idum += IM;
        j=iy/NDIV;
        iy=iv[j];
        iv[j] = *idum;
        if ((temp=AM*iy) > RNMX) return RNMX;
        else return temp;
}



MyDouble gasdev(long *idum)
{
	MyDouble ran1(long *idum);
	static int iset=0;
	static MyDouble gset;
	MyDouble fac,rsq,v1,v2;

	if  (iset == 0) {
		do {
			v1=2.0*ran1(idum)-1.0;
			v2=2.0*ran1(idum)-1.0;
			rsq=v1*v1+v2*v2;
		} while (rsq >= 1.0 || rsq == 0.0);
		fac=sqrt(-2.0*log(rsq)/rsq);
		gset=v1*fac;
		iset=1;
		return v2*fac;
	} else {
		iset=0;
		return gset;
	}
}

MyDouble factrl(int n)
{
    if (n == 0)
     return(1);

    return(n * factrl(n-1));
}

int nlm_all(int num, int n, int l, int m)
{
  return num*(SCF_NMAX+1)*(SCF_LMAX+1)*(SCF_LMAX+1) + n*(SCF_LMAX+1)*(SCF_LMAX+1) + l*(SCF_LMAX+1) + m;
}

int nlm(int n, int l, int m)
{
  return n*(SCF_LMAX+1)*(SCF_LMAX+1) + l*(SCF_LMAX+1) + m;
}

int nl(int n, int l)
{
  return n*(SCF_LMAX+1) + l;
}

int lm(int l, int m)
{
  return l*(SCF_LMAX+1) + m;
}

int kdelta(int a, int b)
{
  if (a==b)
    return 1;
  else
    return 0;
}

MyDouble gnlm_var(int n, int l, int m)
{
  MyDouble fac1=pow(2.,8.*l+2.)*factrl(n)/M_PI*factrl(l-m)/factrl(l+m);
  MyDouble fac2hi=(n+2.*l+1.5)*tgamma(2.*l+1.5)*tgamma(2.*l+1.5)*(2.*l+1)*(2.-kdelta(m,0))*(2.-kdelta(m,0))*(kdelta(m,0)+1.);
  MyDouble fac2lo=pow(.5*n*(n+4.*l+3.)+(l+1.)*(2.*l+1.),2.)*factrl(n+4*l+2);
 
  if (n+l+m > 0)
   return fac1*fac2hi/fac2lo;
  else
   return 0.5;
}

MyDouble hnlm_var(int n, int l, int m)
{
  if (m!=0)
    return gnlm_var(n,l,m);
  else
    return 0.;
}


#endif
