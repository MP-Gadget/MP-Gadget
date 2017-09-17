void   print_spec(void);
void   displacement_fields(void);
void   initialize_ffts(void);
void   assemble_particles(void);
void   free_ffts(void);

double periodic_wrap(double x);

double PowerSpec(double kmag);
void   initialize_powerspectrum(void);

void  write_particle_data(void);
void  read_parameterfile(char *fname);
