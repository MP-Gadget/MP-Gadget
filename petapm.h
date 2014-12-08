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
    double hmax;
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
} petapm_functions;

void petapm_init(double BoxSize, int _Nmesh);
void petapm_region_init_strides(PetaPMRegion * region);

void petapm_force(petapm_prepare_func prepare, 
        petapm_functions * functions, 
        void * userdata);
