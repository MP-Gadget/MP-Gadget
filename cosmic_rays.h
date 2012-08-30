#ifdef COSMIC_RAYS
#ifndef COSMIC_RAYS_H
#define COSMIC_RAYS_H

/* Sanity checks for compiler switches in conjunction with cosmic
 * rays
 */

#if defined(CR_SHOCK) && !defined(COOLING) 
// #error Cannot compile with CR_SHOCK but without cooling.
#endif

typedef struct sph_particle_data SphParticle;

/* ============================================================ */
/* ============ Interface functions to GADGET ================= */
/* ============================================================ */

void cosmic_ray_diffusion(void);
double cosmic_ray_diffusion_vector_multiply(double *a, double *b);
void cosmic_ray_diffusion_matrix_multiply(double *cr_E0_in, double *cr_E0_out, double *cr_n0_in, double *cr_n0_out, int CRpop);
int cosmic_ray_diffusion_evaluate(int target, int mode, 
                                  double *cr_E0_in, double *cr_E0_out, double *cr_E0_sum,
                                  double *cr_n0_in, double *cr_n0_out, double *cr_n0_sum,
                                  int *nexport, int *nsend_local, int CRpop);

int extern CR_initialize_beta_tabs( double Alpha, int CRpop );
void extern CR_free_beta_tabs ( void );
double Beta(double A, double B, double x);

int extern CR_Find_Alpha_to_InjectTo(double Alpha);
int extern CR_find_alpha_InjectTo(SphParticle * Particle);

double CR_Tab_MeanEnergy(double q, double alpha);

void extern CR_Particle_Update( SphParticle* Particle, int CRpop );
double CR_Particle_GetCoolingTimescale(double q, double rho, int CRpop);
double CR_Physical_Pressure(SphParticle * Particle, int CRpop);

double extern CR_Particle_Pressure( SphParticle* Particle, int CRpop );
double extern CR_Comoving_Pressure( SphParticle* Particle, int CRpop );
double extern CR_Physical_Pressure( SphParticle* Particle, int CRpop );

double extern CR_get_energy(SphParticle * Particle, double q0);

double extern CR_Particle_SpecificEnergy( SphParticle* Particle, int CRpop );
double extern CR_Comoving_SpecificEnergy( SphParticle* Particle );
double extern CR_Physical_SpecificEnergy( SphParticle* Particle, int CRpop );

double extern CR_Particle_SpecificNumber( SphParticle* Particle, int CRpop );

double extern CR_Particle_BaryonFraction( SphParticle* Particle, int CRpop );
double extern CR_Particle_MeanKineticEnergy( SphParticle* Particle, int CRpop );

double CR_Particle_GetCooolingTimescale(double q0, double rDensity, int CRpop);

double CR_Particle_GetQforCoolingTimeScale(SphParticle * Particle, double TimeScale, double q0);
double CR_Particle_GetCoulombTimescale(double rQ, double rDensity);

double CR_Particle_GetThermalizationTimescale(double rQ, double rDensity, int CRpop);
double CR_Particle_GetDissipationTimescale(double rQ, double rDensity, int CRpop);
double CR_Particle_GetCooolingTimescale(double rQ, double rDensity, int CRpop);

double CR_EnergyFractionAfterShiftingQ(double Q1, double Q2, double SpectralIndex);
void CR_Tab_Initialize(void);
double CR_Tab_GetThermalizationTimescale(double q, double rho, int CRpop);
double CR_Tab_GetDissipationTimescale(double q, double rho, int CRpop);
double CR_Tab_GetCoolingTimescale(double q, double rho, int CRpop);
double  CR_Find_Qinj(double rho, double TimeScale, double alpha, double qstart, int CRpop);
double CR_corresponding_q(double qinj, double alpha, int CRpop);
double CR_Tab_Beta(double q, int CRpop);


void   CR_test_routine(void);

double CR_Particle_ThermalizeAndDissipate( SphParticle *Particle,
					   double Time, int CRpop );


#ifdef SFR
double extern CR_Particle_SupernovaFeedback( SphParticle* Particle, double SpecificEnergyInput, double TimeScale );
#endif

#if defined(CR_SHOCK)
double extern CR_Particle_ShockInject( SphParticle* Particle,
				       double SpecificEnergyInput,
				       double TimeScale );
#endif

void extern CR_Particle_Inject( SphParticle* Particle, double DeltaE, double DeltaN, int CRpop );

double extern CR_q_from_mean_kinetic_energy( double T, int CRpop );
double extern CR_mean_kinetic_energy( double q, double Alpha);


double extern CR_Particle_TimeScaleQ( SphParticle *Particle, double TimeScale );

#endif
#endif




