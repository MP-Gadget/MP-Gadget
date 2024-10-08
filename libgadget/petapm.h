#ifndef __PETAPM_H__
#define __PETAPM_H__
#include <cufftMp.h>   // NC:library change

#include "powerspectrum.h"

using int64 = long long int;

struct Box3D {
    int64 lower[3];
    int64 upper[3];
    int64 strides[3];
};

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

/* a layout is the communication object, represent
 * pencil / cells exchanged  */


// Layout determins which cells are sent to which task.
struct Layout {
    MPI_Comm comm;
    int NpExport;
    int NpImport;
    int * NpSend;
    int * NpRecv;
    int * DpSend;
    int * DpRecv;
    struct Pencil * PencilSend;
    struct Pencil * PencilRecv;

    int NcExport;
    int NcImport;
    int * NcSend;
    int * NcRecv;
    int * DcSend;
    int * DcRecv;

    double * BufSend;
    double * BufRecv;
    int * ibuffer;
};

/* Data which is private to the PetaPM structure. Don't access from outside.*/
typedef struct PetaPMPriv {
    /* These varibles are initialized by petapm_init*/

    int fftsize;
    cufftHandle plan_forw; // NC:change plan function call
    cufftHandle plan_back;
    cudaStream_t stream;
    MPI_Comm comm_cart_2d;

    /* these variables are allocated every force calculation */
    double * meshbuf;
    size_t meshbufsize;
    struct Layout layout;
    cudaLibXtDesc *desc;
} PetaPMPriv;

typedef struct PetaPM {
    /* These varibles are initialized by petapm_init*/
    MPI_Comm comm;
    PetaPMRegion real_space_region;
    PetaPMRegion fourier_space_region;
    double CellSize;
    int Nmesh;
    double Asmth;
    double BoxSize;
    double G;
    PetaPMPriv priv[1];
    int ThisTask2d[2];
    int NTask2d[2];
    int * Mesh2Task[2]; /* conversion from real space mesh to task2d,  */
    Power ps[1];
} PetaPM;

typedef struct {
    void * Parts;
    size_t elsize;
    size_t offset_pos;
    size_t offset_mass;
    int * RegionInd;
    int (*active) (int i);
    int64_t NumPart;
} PetaPMParticleStruct;

typedef void (*petapm_transfer_func)(PetaPM * pm, int64_t k2, int kpos[3], cufftComplex * value); //NC:change to complex type
typedef void (*petapm_readout_func)(PetaPM * pm, int i, double * mesh, double weight);
typedef PetaPMRegion * (*petapm_prepare_func)(PetaPM * pm, PetaPMParticleStruct * pstruct, void * data, int *Nregions);

typedef struct {
    const char * name;
    petapm_transfer_func transfer;
    petapm_readout_func readout;
} PetaPMFunctions;

/* this mixes up fourier space analysis; with transfer. Shall split them. */
typedef struct {
    /* this is a fourier space readout; need a better name */
    petapm_transfer_func global_readout;
    void (*global_analysis)(PetaPM * pm);
    petapm_transfer_func global_transfer;
} PetaPMGlobalFunctions;

/* UNUSED! */
typedef void * (*petapm_malloc_func)(char * name, size_t * size);
typedef void * (*petapm_mfree_func)(void * ptr);

void petapm_module_init(int Nthreads);

void petapm_init(PetaPM * pm, double BoxSize, double Asmth, int Nmesh, double G, MPI_Comm comm);
void petapm_destroy(PetaPM * pm);
void petapm_region_init_strides(PetaPMRegion * region);

void petapm_force(PetaPM * pm,
        petapm_prepare_func prepare,
        PetaPMGlobalFunctions * global_functions,
        PetaPMFunctions * functions,
        PetaPMParticleStruct * pstruct,
        void * userdata);

PetaPMRegion * petapm_force_init(PetaPM * pm,
        petapm_prepare_func prepare,
        PetaPMParticleStruct * pstruct,
        int * Nregions,
        void * userdata);
cufftComplex * petapm_force_r2c(PetaPM * pm,
        PetaPMGlobalFunctions * global_functions
        ); // NC: changed returned complex type
void petapm_force_c2r(PetaPM * pm,
        cufftComplex * rho_k, PetaPMRegion * regions,
        const int Nregions,
        PetaPMFunctions * functions); // NC: changed input complex type
void petapm_force_finish(PetaPM * pm);

PetaPMRegion * petapm_get_fourier_region(PetaPM * pm);
PetaPMRegion * petapm_get_real_region(PetaPM * pm);
int petapm_mesh_to_k(PetaPM * pm, int i);
int *petapm_get_thistask2d(PetaPM * pm);
int *petapm_get_ntask2d(PetaPM * pm);
cufftComplex * petapm_alloc_rhok(PetaPM * pm); // NC: changed returned complex type
#endif
