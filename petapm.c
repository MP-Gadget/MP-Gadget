#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pfft.h>

#include "allvars.h"
#include "proto.h"

static struct Region {
    /* represents a region in the FFT Mesh */
    ptrdiff_t x0[3];
    ptrdiff_t size[3];
} real_space_region, fourier_space_region;

static struct Region * regions;
static double * real;
static pfft_complex * compl;
static pfft_complex * rho_k;
static int fftsize;

static void pm_alloc();
static void pm_free();
static pfft_plan * plan_forw, * plan_back;
MPI_Comm comm_cart_2d;
void petapm_init_periodic(void) {
    All.Asmth[0] = ASMTH * All.BoxSize / PMGRID;
    All.Rcut[0] = RCUT * All.Asmth[0];
    pfft_init();
    ptrdiff_t n[3] = {PMGRID, PMGRID, PMGRID}
    ptrdiff_t np[2];
    /* try to find a square 2d decomposition */
    int i;
    for(i = sqrt(NTask) + 1; i >= 0; i ++) {
        if(NTask % i == 0) break;
    }
    /* Set up the FFTW plan files. */

    /* 
     * transform is r2c, thus the last dimension is packed!
     * See http://www.fftw.org/doc/MPI-Plan-Creation.html
     **/
    if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
        pfft_fprintf(MPI_COMM_WORLD, stderr, "Error: This test file only works with %d processes.\n", np[0]*np[1]);
        MPI_Finalize();
        return 1;
    }
    fftsize = 2 * pfft_local_size_dft_r2c_3d(n, comm_cart_2d, 
           PFFT_TRANSPOSED_OUT, 
           real_space_region.size, real_space_region.offset, 
           fourier_space_region.size, fourier_space_region.offset);

    MPI_Allreduce(&fftsize, &maxfftsize, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    pm_alloc(fftsize);
    plan_forw = pfft_plan_dft_r2c_3d(
        n, real, rho_k, comm_cart_3d, PFFT_FORWARD, 
        PFFT_TRANSPOSED_OUT | PFFT_MEASURE | PFFT_DESTROY_INPUT);    
    plan_back = pfft_plan_dft_c2r_3d(
        n, compel, real, comm_cart_3d, PFFT_FORWARD, 
        PFFT_TRANSPOSED_OUT | PFFT_MEASURE | PFFT_DESTROY_INPUT);    
    pm_alloc(free);
}

void build_region_list() {
    int i;
    for(i = 0; i < NTopnodes; i ++) {
        if(TopNodes[i].Daughter  < 0) {
            printf("TopLeave at %d\n" ,i);        
        }
    }
}
void petapm_force() {
    /* CIC */
    build_region_list();
    force_treefree();

    force_treeallocate((int) (All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);
    All.NumForcesSinceLastDomainDecomp = (int64_t) (1 + All.TotNumPart * All.TreeDomainUpdateFrequency);
}
static void pm_cic() {

}
static void pm_alloc() {
    real = (double * ) mymalloc("PMbuf1", fftsize * sizeof(double));
    compl = (pfft_complex *) mymalloc("PMbuf2", fftsize * sizeof(double));
    rho_k = (pfft_complex * ) mymalloc("PMbuf3", fftsize * sizeof(double));
}

static void pm_free() {
    myfree(rho_k);
    myfree(compl);
    myfree(real);
}

