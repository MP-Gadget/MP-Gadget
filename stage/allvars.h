#ifndef ALLVARS_H
#define ALLVARS_H

#include <stdio.h>

#ifndef DOUBLEPRECISION     /* default is single-precision */
typedef float  MyFloat;
typedef float  MyDouble;
#else
#if (DOUBLEPRECISION == 2)   /* mixed precision */
typedef float   MyFloat;
typedef double  MyDouble;
#else                        /* everything double-precision */
typedef double  MyFloat;
typedef double  MyDouble;
#endif
#endif


extern int DesLinkNgb;

extern char OutputDir[500];


#endif
