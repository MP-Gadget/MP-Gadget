#ifndef _COOLING_H_
#define _COOLING_H_
struct UVBG {
    double J_UV;
    double gJH0;
    double gJHep;
    double gJHe0;
    double epsH0;
    double epsHep;
    double epsHe0;
} ;

void GetParticleUVBG(int i, struct UVBG * uvbg);
void GetGlobalUVBG(struct UVBG * uvbg);
double AbundanceRatios(double u, double rho, struct UVBG * uvbg, double *ne_guess, double *nH0_pointer, double *nHeII_pointer);
double GetCoolingTime(double u_old, double rho, struct UVBG * uvbg,  double *ne_guess, double Z);
double DoCooling(double u_old, double rho, double dt, struct UVBG * uvbg, double *ne_guess, double Z);
double ConvertInternalEnergy2Temperature(double u, double ne);

void   InitCool(void);
void   IonizeParams(void);
void   MakeCoolingTable(void);
void   SetZeroIonization(void);

#endif
