#ifndef AUXILIARY_FUNCTIONS_C
#define AUXILIARY_FUNCTIONS_C

#ifndef __GNUC__
#define inline
#endif



#define h  All.HubbleParam

#define cm (h/All.UnitLength_in_cm)
#define g  (h/All.UnitMass_in_g)
#define s  (h/All.UnitTime_in_s)
#define erg (g*cm*cm/(s*s))

/* #define erg (1.0/All.UnitEnergy_in_cgs) */
#define keV (1.602e-9*erg)
#define deg 1.0
#define m_p (PROTONMASS * g)
#define m_e (ELECTRONMASS * g)
#define k_B (BOLTZMANN * erg / deg)
#define LightSpeed (2.9979e10*cm/s)

#define HBAR ( 1.05457e-27 * cm * cm * g / s )

#define e2  ( HBAR * LightSpeed / 137.04 )

#define statcoul sqrt( erg * cm )

#define mpc2 ( m_p * LightSpeed * LightSpeed )
#define mec2 ( m_e * LightSpeed * LightSpeed )
#define c2   ( LightSpeed * LightSpeed )



static inline double square(double A)
{
  return (A * A);
}


static inline double cube(double A)
{
  return (A * A * A);
}



/* ====================================================================== */
/* ================== Functions for physical information ================ */
/* ====================================================================== */


static inline double IonizationGrade(SphParticle * Particle)
{
#ifdef COOLING
  return (1 + HYDROGEN_MASSFRAC) / (2 * HYDROGEN_MASSFRAC);	/* always return full ionization */
#else
  return 1.0;
#endif
}


static inline double Comoving_ElectronDensity(SphParticle * Particle)
{
  return (Particle->d.Density * HYDROGEN_MASSFRAC * IonizationGrade(Particle) / m_p);
}



static inline double Physical_ElectronDensity(SphParticle * Particle)
{
  if(All.ComovingIntegrationOn)
    {
      return Comoving_ElectronDensity(Particle) / cube(All.Time);
    }
  else
    {
      return Comoving_ElectronDensity(Particle);
    }
}


static inline double ElectronDensity(SphParticle * Particle)
     /* for compatibility with old version of code */
     /* Output: Electrons/Volume (physical) */
{
  return Physical_ElectronDensity(Particle);
}


static inline double Comoving_NumberDensity(SphParticle * Particle)
{
  return (Particle->d.Density * (3.0 * HYDROGEN_MASSFRAC + 1.0) / (4.0 * m_p));
}


static inline double Physical_NumberDensity(SphParticle * Particle)
{
  if(All.ComovingIntegrationOn)
    {
      return Comoving_NumberDensity(Particle) / cube(All.Time);
    }
  else
    {
      return Comoving_NumberDensity(Particle);
    }
}

static inline double PhysicalNumberDensity(SphParticle * Particle)
     /* for compatibility with old version of code */
{
  return Physical_NumberDensity(Particle);
}


static inline double Comoving_Density(SphParticle * Particle)
{
  return Particle->d.Density;
}


static inline double Physical_Density(SphParticle * Particle)
{
  if(All.ComovingIntegrationOn)
    {
      return Particle->d.Density / cube(All.Time);
    }
  else
    {
      return Particle->d.Density;
    }
}


static inline double Physical_InternalEnergy(SphParticle * Particle)
{
  double rPhysicalDensity;

  rPhysicalDensity = Physical_Density(Particle);

  return Particle->Entropy * pow(rPhysicalDensity, GAMMA_MINUS1) / GAMMA_MINUS1;
}

#endif
