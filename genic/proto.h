
#include <gsl/gsl_rng.h>

void   print_spec(void);
int    FatalError(int errnum);
void   displacement_fields(void);
void   initialize_ffts(void);
void   set_units(void);
void   assemble_particles(void);
void   free_ffts(void);
double fnl(double x);

int find_files(char *fname);

void   assemble_grid(void);
void   read_power_table(void);
double periodic_wrap(double x);


double PowerSpec(double kmag);
void   initialize_powerspectrum(void);

void   combine_particle_data(void);

void  write_particle_data(void);
void  read_parameterfile(char *fname);
void  read_glass(char *fname);


size_t my_fread(void *ptr, size_t size, size_t nmemb, FILE * stream);
size_t my_fwrite(void *ptr, size_t size, size_t nmemb, FILE * stream);

void save_local_data(void);

int compare_type(const void *a, const void *b);

