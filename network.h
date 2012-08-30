#ifndef NETWORK_JINA_H
#define NETWORK_JINA_H

#ifdef NUCLEAR_NETWORK

#define NETWORK_DIFFVAR 1e-6	/* variation for numerical derivatives */

#define min(x,y) ((x) < (y) ? (x) : (y))
#define max(x,y) ((x) > (y) ? (x) : (y))

void network_init( char *speciesfile, char *ratesfile, char *partfile, char *massesfile, char *weakratesfile );
void network_normalize(double *x, double *e);
void network_integrate( double temp, double rho, double *x, double *dx, double dt, double *dedt );
void network_getrhs( double temp, double rho, double *y, double *rhs );

void network_getjacob( double temp, double rho, double h, double *y, double *rhs, double *jacob );
#if defined(SUPERLU) || defined(SOLVER_PARDISO)
void network_getjacobLU( double temp, double rho, double h, double *y, double *rhs, double *values, int *columns, int *rowstart, int *usedelements );
#endif

void network_part( double temp );
void network_getrates( double temp, double rho, double ye );

struct network_nucdata {
	int na, nz, nn; /* atomic number, proton number, neutron number */
	char name[6]; /* full name, e.g. he4 */
	char symbol[3]; /* only atomic symbol, e.g. he */
	float part[24]; /* tabulated partition function */
	float exm;  /* mass excess */
	float spin; /* spin */
	int nrates, nweakrates; /* number of rates */
	int *rates, *weakrates;
	double **prates, **pweakrates; /* pointer to the rates */
	double *w, *wweak; /* weight factor for the rate */
} *network_nucdata;

struct network_rates {
	int type;
	int ninput, noutput; /* number of particles that are input and output of the reaction */
	int input[4], output[4]; /* element ids that are created and destroyed in the reaction */
	int isWeak, isReverse, isElectronCapture, isWeakExternal;
	float q; /* q value of the reaction */
	float data[7];
	double rate, baserate;
} *network_rates;

struct network_weakrates {
	int input, output;
	float q1, q2;
	float lambda1[143], lambda2[143]; /* 13 * 11 */
	int isReverse;
	double rate;
} *network_weakrates;

struct network_data {
	int nuc_count;
	int rate_count;
	int weakrate_count;
	float weakTemp[13], weakRhoYe[11]; /* temperature and rho*ye grid for weak reactions */
	double *gg;
	double *x;
	double *y;
	double conv; /* unit conversion factor */
	double *na;
	double *deriv;
	double oldTemp, oldRho, oldYe;
	int nElectronCaptureRates, *electronCaptureRates; /* list of electron capture rates */
	int nMatrix, nMatrix2;
	double *yrate, *yweakrate;
#if defined(SUPERLU) || defined(SOLVER_PARDISO)
    double *jacob;
#endif
} network_data;

static double network_parttemp[24] = { 1.0e8, 1.5e8, 2.0e8, 3.0e8, 4.0e8, 5.0e8, 6.0e8, 7.0e8,
                                8.0e8, 9.0e8, 1.0e9, 1.5e9, 2.0e9, 2.5e9, 3.0e9, 3.5e9,
                                4.0e9, 4.5e9, 5.0e9, 6.0e9, 7.0e9, 8.0e9, 9.0e9, 1.0e10 };

#endif /* NUCLEAR_NETWORK */

#endif /* NETWORK_JINA_H */

