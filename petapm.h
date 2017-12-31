#include <pfft.h>
typedef struct Region {
    /* represents a region in the FFT Mesh */
    ptrdiff_t offset[3];
    ptrdiff_t size[3];
    ptrdiff_t strides[3];
    size_t totalsize;
    double * buffer; 
    /* below are used mostly for investigation */
    double center[3];
    double len;
    int numpart;
    int no; /* node number for debugging */
} PetaPMRegion;
typedef void (*petapm_transfer_func)(int64_t k2, int kpos[3], pfft_complex * value);
typedef void (*petapm_readout_func)(int i, double * mesh, double weight);
typedef PetaPMRegion * (*petapm_prepare_func)(void * data, int *Nregions);

typedef struct {
    char * name;
    void (*transfer)(int64_t k2, int kpos[3], pfft_complex * value);
    void (*readout)(int i, double * mesh, double weight);
} PetaPMFunctions;

typedef struct {
    void (*global_readout)(int64_t k2, int kpos[3], pfft_complex * value);
    void (*global_analysis)(void);
    void (*global_transfer)(int64_t k2, int kpos[3], pfft_complex * value);
} PetaPMGlobalFunctions;

typedef void * (*petapm_malloc_func)(char * name, size_t * size);
typedef void * (*petapm_mfree_func)(void * ptr);

typedef struct {
    void * Parts;
    size_t elsize;
    size_t offset_pos;
    size_t offset_mass;
    size_t offset_regionind;
    int (*active) (int i);
    int NumPart;
} PetaPMParticleStruct;

void petapm_init(double BoxSize, int _Nmesh, int Nthreads);
void petapm_region_init_strides(PetaPMRegion * region);

void petapm_force(petapm_prepare_func prepare, 
        PetaPMGlobalFunctions * global_functions,
        PetaPMFunctions * functions, 
        PetaPMParticleStruct * pstruct,
        void * userdata);

void petapm_force_init(
        petapm_prepare_func prepare, 
        PetaPMParticleStruct * pstruct,
        void * userdata);
void petapm_force_r2c( 
        PetaPMGlobalFunctions * global_functions
        );
void petapm_force_c2r(
        PetaPMFunctions * functions);
void petapm_force_finish();
PetaPMRegion * petapm_get_fourier_region();
PetaPMRegion * petapm_get_real_region();
pfft_complex * petapm_get_rho_k();
int petapm_mesh_to_k(int i);
int *petapm_get_thistask2d();
int *petapm_get_ntask2d();
