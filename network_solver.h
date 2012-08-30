#ifndef NETWORK_SOLVER_H
#define NETWORK_SOLVER_H

#define min(x,y) ((x) < (y) ? (x) : (y))
#define max(x,y) ((x) > (y) ? (x) : (y))

typedef void(*func_network_getrhs)( double, double, double*, double* );
#if defined(SUPERLU) || defined(SOLVER_PARDISO)
typedef void(*func_network_getjacob)( double, double, double, double*, double*, double*, int*, int*, int* );
#else
typedef void(*func_network_getjacob)( double, double, double, double*, double*, double* );
#endif

struct network_solver_data {
	int nsteps;
	int maxstep;
	int *steps;
	int maxiter;
	int matrixsize;
	int matrixsize2;
	int nelements;
	double *aion;
	double tolerance;
	func_network_getrhs getrhs;
	func_network_getjacob getjacob;
	double *dy;
	double *ynew;
	double *yscale;
	double *rhs;
	double *jacob;
	double *d;
	double *x;
	double *err;
	double *qcol;
	double *a;
	double *alf;
#if defined(SUPERLU) || defined(SOLVER_PARDISO)
	int *columns;
	int *rowstart;
	int usedelements;
#endif
#ifdef NETWORK_OUTPUT
	FILE *fp;
#endif
} network_solver_data;

void network_solver_init( func_network_getrhs getrhs, func_network_getjacob getjacob, double tolerance, int matrixsize, int nelements, double *aion );
void network_solver_integrate( double temp, double rho, double *y, double dt );
void network_solver_calc_dy( double temp, double rho, double dt, int steps, double *y, double *yn );
void network_solver_extrapolate( int iter, double *x, double *y, double *dy, double *qcol );

#endif /* NETWORK_SOLVER_H */
