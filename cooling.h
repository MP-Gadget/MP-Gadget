#ifndef INLINE_FUNC
#ifdef INLINE
#define INLINE_FUNC inline
#else
#define INLINE_FUNC
#endif
#endif


double AbundanceRatios(double u, double rho, double *ne_guess, double *nH0_pointer, double *nHeII_pointer);
double convert_u_to_temp(double u, double rho, double *ne_guess);
#ifdef LT_METAL_COOLING
double GetCoolingTime(double u_old, double rho,  double *ne_guess, double Z);
double CoolingRateFromU(double u, double rho, double *ne_guess, double Z);
double DoCooling(double u_old, double rho, double dt, double *ne_guess, double Z);
double CoolingRate(double logT, double rho, double *nelec, double Z);
double DoInstabilityCooling(double m_old, double u, double rho, double dt, double fac, double *ne_guess, double Z);
#else
double CoolingRate(double logT, double rho, double *nelec);
double CoolingRateFromU(double u, double rho, double *ne_guess);
double DoCooling(double u_old, double rho, double dt, double *ne_guess);
double GetCoolingTime(double u_old, double rho,  double *ne_guess);
double DoInstabilityCooling(double m_old, double u, double rho, double dt, double fac, double *ne_guess);
#endif

void   find_abundances_and_rates(double logT, double rho, double *ne_guess);
void   InitCool(void);
void   InitCoolMemory(void);
void   IonizeParams(void);
void   IonizeParamsFunction(void);
void   IonizeParamsTable(void);
double INLINE_FUNC LogTemp(double u, double ne);
void   MakeCoolingTable(void);
void   ReadIonizeParams(char *fname);
void   SetZeroIonization(void);
void   TestCool(void);



