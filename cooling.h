double AbundanceRatios(double u, double rho, double *ne_guess, double *nH0_pointer, double *nHeII_pointer);
double GetCoolingTime(double u_old, double rho,  double *ne_guess, double Z);
double DoCooling(double u_old, double rho, double dt, double *ne_guess, double Z);

void   InitCool(void);
void   IonizeParams(void);
void   MakeCoolingTable(void);
void   SetZeroIonization(void);



