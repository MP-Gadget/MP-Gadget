#ifdef MACHNUM
#ifndef MACHNUM_INC

double extern MachNumber( struct sph_particle_data* Particle );


#ifdef COSMIC_RAYS

int extern MachNumberCR( struct sph_particle_data *Particle, double *shock_conditions );
 
#endif

double extern DissipatedEnergy( struct sph_particle_data* Particle, struct particle_data* Properties );

#define MACHNUM_INC
#endif
#endif

