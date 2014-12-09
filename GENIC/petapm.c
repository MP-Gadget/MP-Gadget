#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
/* do NOT use complex.h it breaks the code */
#include <pfft.h>

#include "allvars.h"
#include "proto.h"
#include "walltime.h"

static struct Region {
    /* represents a region in the FFT Mesh */
    ptrdiff_t offset[3];
    ptrdiff_t size[3];
    ptrdiff_t strides[3];
    size_t totalsize;
    double * buffer; 
} real_space_region, fourier_space_region;

static struct Region * regions = NULL;
static int Nregions = 0;
static void region_init_strides(struct Region * region);

/* a layout is the communication object, represent 
 * pencil / cells exchanged  */

struct Layout {
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

static void layout_prepare (struct Layout * L);
static void layout_build_and_exchange_pencils(struct Layout * L);
static void layout_finish(struct Layout * L);
static void layout_build_and_exchange_cells_to_pfft(struct Layout * L);
static void layout_build_and_exchange_cells_to_local(struct Layout * L);

/* cell_iterator nees to be thread safe !*/
typedef void (* cell_iterator)(double * cell_value, double * comm_buffer);
static void layout_iterate_cells(struct Layout * L, cell_iterator iter);

struct Pencil { /* a pencil starting at offset, with lenght len */
    int offset[3];
    int len;
    int first; 
    int meshbuf_first; /* first pixel in meshbuf */
    int task;
};
static int pencil_cmp_target(const struct Pencil * p1, const struct Pencil * p2);
static int pos_get_target(const int pos[2]);

static int64_t reduce_int64(int64_t input);
/* for debuggin */
static void verify_density_field();

static double * real;
static double * meshbuf;
static size_t meshbufsize;
static pfft_complex * complx;
static pfft_complex * rho_k;
static int fftsize;
static void pm_alloc();
static void pm_free();
static pfft_plan plan_forw, plan_back;
MPI_Comm comm_cart_2d;
static int * ParticleRegion;
static int ThisTask2d[2];
static int NTask2d[2];
static int * (Mesh2Task[2]); /* conversion from real space mesh to task2d,  */
static int * Mesh2K; /* convertion fourier mesh to integer frequency (or K)  */
static MPI_Datatype MPI_PENCIL;

void petapm_init_periodic(void) {

    /* define the global long / short range force cut */

    /* fac is - 4pi G     (L / 2pi) **2 / L ** 3 
     *        Gravity       k2            DFT  
     * */

    pfft_init();

    ptrdiff_t n[3] = {Nmesh, Nmesh, Nmesh};
    ptrdiff_t np[2];

    /* The following memory will never be freed */
    Mesh2Task[0] = malloc(sizeof(int) * Nmesh);
    Mesh2Task[1] = malloc(sizeof(int) * Nmesh);
    Mesh2K = malloc(sizeof(int) * Nmesh);

    /* initialize the MPI Datatype of pencil */
    MPI_Type_contiguous(sizeof(struct Pencil), MPI_BYTE, &MPI_PENCIL);
    MPI_Type_commit(&MPI_PENCIL);

    /* try to find a square 2d decomposition */
    int i;
    int k;
    for(i = sqrt(NTask) + 1; i >= 0; i --) {
        if(NTask % i == 0) break;
    }
    np[0] = i;
    np[1] = NTask / i;

    if(ThisTask == 0) {
        printf("Using 2D Task mesh %td x %td \n", np[0], np[1]);
    }
    if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
        fprintf(stderr, "Error: This test file only works with %td processes.\n", np[0]*np[1]);
        abort();
    }

    int periods_unused[2];
    MPI_Cart_get(comm_cart_2d, 2, NTask2d, periods_unused, ThisTask2d);

    if(NTask2d[0] != np[0]) abort();
    if(NTask2d[1] != np[1]) abort();

    fftsize = 2 * pfft_local_size_dft_r2c_3d(n, comm_cart_2d, 
           PFFT_TRANSPOSED_OUT, 
           real_space_region.size, real_space_region.offset, 
           fourier_space_region.size, fourier_space_region.offset);

    /*
     * In fourier space, the transposed array is ordered in
     * are in (y, z, x). The strides and sizes returned
     * from local size is in (Nx, Ny, Nz), hence we roll them once
     * so that the strides will give correct linear indexing for 
     * integer coordinates given in order of (y, z, x).
     * */

#define ROLL(a, N, j) { \
    typeof(a[0]) tmp[N]; \
    ptrdiff_t k; \
    for(k = 0; k < N; k ++) tmp[k] = a[k]; \
    for(k = 0; k < N; k ++) a[k] = tmp[(k + j)% N]; \
    }

    ROLL(fourier_space_region.offset, 3, 1);
    ROLL(fourier_space_region.size, 3, 1);

#undef ROLL

    /* calculate the strides */
    region_init_strides(&real_space_region);
    region_init_strides(&fourier_space_region); 

    /* planning the fft; need temporary arrays */

    pm_alloc();

    plan_forw = pfft_plan_dft_r2c_3d(
        n, real, rho_k, comm_cart_2d, PFFT_FORWARD, 
        PFFT_TRANSPOSED_OUT | PFFT_ESTIMATE | PFFT_DESTROY_INPUT);    
    plan_back = pfft_plan_dft_c2r_3d(
        n, complx, real, comm_cart_2d, PFFT_BACKWARD, 
        PFFT_TRANSPOSED_IN | PFFT_ESTIMATE | PFFT_DESTROY_INPUT);    

    pm_free();

    /* now lets fill up the mesh2task arrays */

    printf("ThisTask = %d (%td %td %td) - (%td %td %td)\n", ThisTask, 
            real_space_region.offset[0], 
            real_space_region.offset[1], 
            real_space_region.offset[2],
            real_space_region.size[0], 
            real_space_region.size[1], 
            real_space_region.size[2]);
    for(k = 0; k < 2; k ++) {
        int * tmp = (int*) alloca(sizeof(int) * Nmesh);
        for(i = 0; i < Nmesh; i ++) {
            tmp[i] = 0;
        }
        for(i = 0; i < real_space_region.size[k]; i ++) {
            tmp[i + real_space_region.offset[k]] = ThisTask2d[k];
        }
        /* which column / row hosts this tile? */
        /* FIXME: this is very inefficient */
        MPI_Allreduce(tmp, Mesh2Task[k], Nmesh, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        /*
        if(ThisTask == 0) {
            for(i = 0; i < Nmesh; i ++) {
                printf("Mesh2Task[%d][%d] == %d\n", k, i, Mesh2Task[k][i]);
            }
        }
        */
    }
    /* as well as the mesh2k array, we save the 2PI factor as done in
     * pm_periodic */
    for(i = 0; i < Nmesh; i ++) {
        if(i <= Nmesh / 2)
            Mesh2K[i] = i;
        else
            Mesh2K[i] = i - Nmesh;
    }

}

    
/* 
 * read out field to particle i, with value no need to be thread safe 
 * (particle i is never done by same thread)
 * */
typedef void (* pm_iterator)(int i, double * mesh, double weight);
static void pm_iterate(pm_iterator iterator);
/* apply transfer function to value, kpos array is in x, y, z order */
typedef void (*transfer_function) (int64_t k2, int kpos[3], pfft_complex * value);
static void pm_apply_transfer_function(struct Region * fourier_space_region, 
        pfft_complex * src, 
        pfft_complex * dst, transfer_function H);

static void diff_transfer(int64_t k2, pfft_complex * value);
static void density_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void disp_x_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void disp_y_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void disp_z_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void put_particle_to_mesh(int i, double * mesh, double weight);
static void readout_density(int i, double * mesh, double weight);
static void readout_potential(int i, double * mesh, double weight);
static void readout_force_x(int i, double * mesh, double weight);
static void readout_force_y(int i, double * mesh, double weight);
static void readout_force_z(int i, double * mesh, double weight);
static void gaussian_fill(struct Region * region, pfft_complex * rho_k);

void petapm_force() {
    int current_region = 0;
    int ndone = 0;
    int iterations;
    int i;
    struct Layout layout;

    walltime_measure("/Misc");
    pm_alloc();

    /* this takes care of the padding */
    memset(real, 0, sizeof(double) * fftsize);
    memset(meshbuf, 0, meshbufsize * sizeof(double));

    /* the empty cells won't be included in the communication, 
     * so we force all mesh cells to be 1 here */
    for(i = 0; i < meshbufsize; i ++) {
        meshbuf[i] = 1.0;
    }

    layout_prepare(&layout);
    walltime_measure("/Layout");
    if(ThisTask == 0) {
        printf("layouting \n");
        fflush(stdout);
    }

    /* fill in the initial gausian field */
    gaussian_fill(&fourier_space_region, rho_k);

    if(ThisTask == 0) {
        printf("filling \n");
        fflush(stdout);
    }

    walltime_measure("/Fill");
    /* Density */
    
    pm_apply_transfer_function(&fourier_space_region, rho_k, complx, density_transfer);
    pfft_execute_dft_c2r(plan_back, complx, real);
    layout_build_and_exchange_cells_to_local(&layout);
    pm_iterate(readout_density);
    if(ThisTask == 0) {
        printf("reading out density\n");
        fflush(stdout);
    }
    
    walltime_measure("/Disp/X");

    /* displacements */

    pm_apply_transfer_function(&fourier_space_region, rho_k, complx, disp_x_transfer);
    pfft_execute_dft_c2r(plan_back, complx, real);
    layout_build_and_exchange_cells_to_local(&layout);
    pm_iterate(readout_force_x);
    if(ThisTask == 0) {
        printf("reading out force x\n");
        fflush(stdout);
    }
    walltime_measure("/Disp/X");
    
    pm_apply_transfer_function(&fourier_space_region, rho_k, complx, disp_y_transfer);
    pfft_execute_dft_c2r(plan_back, complx, real);
    layout_build_and_exchange_cells_to_local(&layout);
    pm_iterate(readout_force_y);
    if(ThisTask == 0) {
        printf("reading out force y\n");
        fflush(stdout);
    }
    walltime_measure("/Disp/Y");

    pm_apply_transfer_function(&fourier_space_region, rho_k, complx, disp_z_transfer);
    pfft_execute_dft_c2r(plan_back, complx, real);
    layout_build_and_exchange_cells_to_local(&layout);
    pm_iterate(readout_force_z);
    if(ThisTask == 0) {
        printf("reading out force z\n");
        fflush(stdout);
    }
    walltime_measure("/Disp/Z");

    layout_finish(&layout);
    pm_free();

}

/* build a communication layout */

static void layout_build_pencils(struct Layout * L);
static void layout_exchange_pencils(struct Layout * L);
static void layout_prepare (struct Layout * L) {
    int r;
    int i;

    L->ibuffer = malloc(sizeof(int) * NTask * 8);

    memset(L->ibuffer, 0, sizeof(int) * NTask * 8);
    L->NpSend = &L->ibuffer[NTask * 0];
    L->NpRecv = &L->ibuffer[NTask * 1];
    L->NcSend = &L->ibuffer[NTask * 2];
    L->NcRecv = &L->ibuffer[NTask * 3];
    L->DcSend = &L->ibuffer[NTask * 4];
    L->DcRecv = &L->ibuffer[NTask * 5];
    L->DpSend = &L->ibuffer[NTask * 6];
    L->DpRecv = &L->ibuffer[NTask * 7];

    L->NpExport = 0;
    L->NcExport = 0;
    L->NpImport = 0;
    L->NcImport = 0;

    int NpAlloc = 0;
    /* count pencils until buffer would run out */
    for (r = 0; r < Nregions; r ++) {
        NpAlloc += regions[r].size[0] * regions[r].size[1];
    }

    L->PencilSend = malloc(NpAlloc * sizeof(struct Pencil));

    layout_build_pencils(L);

    /* sort the pencils by the target rank for ease of next step */
    qsort(L->PencilSend, NpAlloc, sizeof(struct Pencil), pencil_cmp_target);
    /* zero length pixels are moved to the tail */

    /* now shrink NpExport*/
    L->NpExport = NpAlloc;
    while(L->PencilSend[L->NpExport - 1].len == 0) {
        L->NpExport --;
    }

    /* count total number of cells to be exported */
    int NcExport = 0;
#pragma omp parallel for reduction(+: NcExport)
    for(i = 0; i < L->NpExport; i++) {
        NcExport += L->PencilSend[i].len;
    }
    L->NcExport = NcExport;

#pragma omp parallel for
    for(i = 0; i < L->NpExport; i ++) {
        int task = pos_get_target(L->PencilSend[i].offset);
#pragma omp atomic
        L->NpSend[task] ++;
#pragma omp atomic
        L->NcSend[task] += L->PencilSend[i].len;
    }

    MPI_Alltoall(L->NpSend, 1, MPI_INT, L->NpRecv, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Alltoall(L->NcSend, 1, MPI_INT, L->NcRecv, 1, MPI_INT, MPI_COMM_WORLD);

    /* build the displacement array; why doesn't MPI build these automatically? */
    L->DpSend[0] = 0; L->DpRecv[0] = 0;
    L->DcSend[0] = 0; L->DcRecv[0] = 0;
    for(i = 1; i < NTask; i ++) {
        L->DpSend[i] = L->NpSend[i - 1] + L->DpSend[i - 1];
        L->DpRecv[i] = L->NpRecv[i - 1] + L->DpRecv[i - 1];
        L->DcSend[i] = L->NcSend[i - 1] + L->DcSend[i - 1];
        L->DcRecv[i] = L->NcRecv[i - 1] + L->DcRecv[i - 1];
    }
    L->NpImport = L->DpRecv[NTask -1] + L->NpRecv[NTask -1];
    L->NcImport = L->DcRecv[NTask -1] + L->NcRecv[NTask -1];

    /* some checks */
    if(L->DpSend[NTask - 1] + L->NpSend[NTask -1] != L->NpExport) abort();
    if(L->DcSend[NTask - 1] + L->NcSend[NTask -1] != L->NcExport) abort();

    int64_t totNpAlloc = reduce_int64(NpAlloc);
    int64_t totNpExport = reduce_int64(L->NpExport);
    int64_t totNcExport = reduce_int64(L->NcExport);
    int64_t totNpImport = reduce_int64(L->NpImport);
    int64_t totNcImport = reduce_int64(L->NcImport);

    if(totNpExport != totNpImport) {
        abort();
    }
    if(totNcExport != totNcImport) {
        abort();
    }

    /* exchange the pencils */
    if(ThisTask == 0) {
        printf("PetaPM:  %010ld/%010ld Pencils and %010ld Cells\n", totNpExport, totNpAlloc, totNcExport);
    }
    L->PencilRecv = malloc(L->NpImport * sizeof(struct Pencil));
    layout_exchange_pencils(L);
}

static void layout_build_pencils(struct Layout * L) {
    /* now build pencils to be exported */
    int p0 = 0;
    int r;
    for (r = 0; r < Nregions; r++) {
        int ix;
#pragma omp parallel for private(ix)
        for(ix = 0; ix < regions[r].size[0]; ix++) {
            int iy;
            for(iy = 0; iy < regions[r].size[1]; iy++) {
                int poffset = ix * regions[r].size[1] + iy;
                struct Pencil * p = &L->PencilSend[p0 + poffset];

                p->offset[0] = ix + regions[r].offset[0];
                p->offset[1] = iy + regions[r].offset[1];
                p->offset[2] = regions[r].offset[2];
                p->len = regions[r].size[2];
                p->meshbuf_first = (regions[r].buffer - meshbuf) +
                    regions[r].strides[0] * ix +
                    regions[r].strides[1] * iy;
                /* now lets compress the pencil */
                while((p->len > 0) && (meshbuf[p->meshbuf_first + p->len - 1] == 0.0)) {
                    p->len --;
                }
                while((p->len > 0) && (meshbuf[p->meshbuf_first] == 0.0)) {
                    p->len --;
                    p->meshbuf_first++;
                    p->offset[2] ++;
                }

                p->task = pos_get_target(p->offset);
            }
        }
        p0 += regions[r].size[0] * regions[r].size[1];
    }

}

static void layout_exchange_pencils(struct Layout * L) {
    int r;
    int i;
    int offset;

    /* build the first pointers to refer to the correct relative buffer locations */
    /* note that the buffer hasn't bee assembled yet */
    offset = 0;
    for(i = 0; i < NTask; i ++) {
        int j;
        struct Pencil * p = &L->PencilSend[offset];

        /* do not tamper p->first if there no pencil */
        if(L->NpSend[i] == 0) continue;

        p->first = 0;
        for(j = 1; j < L->NpSend[i]; j++) {
            p[j].first = p[j - 1].first + p[j - 1].len;
        }
        offset += L->NpSend[i];
    }

    MPI_Alltoallv(
            L->PencilSend, L->NpSend, L->DpSend, MPI_PENCIL,
            L->PencilRecv, L->NpRecv, L->DpRecv, MPI_PENCIL, 
            MPI_COMM_WORLD);

    /* set first to point to absolute position in the full import cell buffer */
    offset = 0;
    for(i = 0; i < NTask; i ++) {
        struct Pencil * p = &L->PencilRecv[offset];
        int j;
        for(j = 0; j < L->NpRecv[i]; j++) {
            p[j].first += L->DcRecv[i];
        }
        offset += L->NpRecv[i];
    }
    
    /* set first to point to absolute position in the full export cell buffer */
    offset = 0;
    for(i = 0; i < NTask; i ++) {
        struct Pencil * p = &L->PencilSend[offset];
        int j;
        for(j = 0; j < L->NpSend[i]; j++) {
            p[j].first += L->DcSend[i];
        }
        offset += L->NpSend[i];
    }
}

static void layout_finish(struct Layout * L) {
    free(L->PencilRecv);
    free(L->PencilSend);
    free(L->ibuffer);
}

/* exchange cells to their pfft host, then reduce the cells to the pfft
 * array */
static void to_pfft(double * cell, double * buf) {
#pragma omp atomic
            cell[0] += buf[0];
}
static void layout_build_and_exchange_cells_to_pfft(struct Layout * L) {
    L->BufSend = malloc(L->NcExport * sizeof(double));
    L->BufRecv = malloc(L->NcImport * sizeof(double));

    int i;
    int offset;

    /* collect all cells into the send buffer */
    offset = 0;
    for(i = 0; i < L->NpExport; i ++) {
        struct Pencil * p = &L->PencilSend[i];
        memcpy(L->BufSend + offset, &meshbuf[p->meshbuf_first], 
                sizeof(double) * p->len);
        offset += p->len;
    }

    /* receive cells */
    MPI_Alltoallv(
            L->BufSend, L->NcSend, L->DcSend, MPI_DOUBLE,
            L->BufRecv, L->NcRecv, L->DcRecv, MPI_DOUBLE, 
            MPI_COMM_WORLD);

#if 0
    double massExport = 0;
    for(i = 0; i < L->NcExport; i ++) {
        massExport += L->BufSend[i];
    }

    double massImport = 0;
    for(i = 0; i < L->NcImport; i ++) {
        massImport += L->BufRecv[i];
    }
    double totmassExport;
    double totmassImport;
    MPI_Allreduce(&massExport, &totmassExport, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&massImport, &totmassImport, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(ThisTask == 0) {
        printf("totmassExport = %g totmassImport = %g\n", totmassExport, totmassImport);
    }
#endif

    layout_iterate_cells(L, to_pfft);
    free(L->BufRecv);
    free(L->BufSend);
}

/* readout cells on their pfft host, then exchange the cells to the domain 
 * host */
static void to_region(double * cell, double * region) {
    *region = *cell;
}

static void layout_build_and_exchange_cells_to_local(struct Layout * L) {
    L->BufSend = malloc(L->NcExport * sizeof(double));
    L->BufRecv = malloc(L->NcImport * sizeof(double));
    int i;
    int offset;

    layout_iterate_cells(L, to_region);

    /* exchange cells */
    /* notice the order is reversed from to_pfft */
    MPI_Alltoallv(
            L->BufRecv, L->NcRecv, L->DcRecv, MPI_DOUBLE, 
            L->BufSend, L->NcSend, L->DcSend, MPI_DOUBLE,
            MPI_COMM_WORLD);

    /* distribute BufSend to meshbuf */
    offset = 0;
    for(i = 0; i < L->NpExport; i ++) {
        struct Pencil * p = &L->PencilSend[i];
        memcpy(&meshbuf[p->meshbuf_first], 
                L->BufSend + offset, 
                sizeof(double) * p->len);
        offset += p->len;
    }
    free(L->BufRecv);
    free(L->BufSend);
}

/* iterate over the pairs of real field cells and RecvBuf cells 
 *
 * !!! iter has to be thread safe. !!!
 * */
static void layout_iterate_cells(struct Layout * L, cell_iterator iter) {
    int i;
#pragma omp parallel for
    for(i = 0; i < L->NpImport; i ++) {
        struct Pencil * p = &L->PencilRecv[i];
        int k;
        ptrdiff_t linear0 = 0;
        for(k = 0; k < 2; k ++) {
            int ix = p->offset[k];
            while(ix < 0) ix += Nmesh;
            while(ix >= Nmesh) ix -= Nmesh;
            ix -= real_space_region.offset[k];
            if(ix >= real_space_region.size[k]) {
                /* seroius problem assmpution about pfft layout was wrong*/
                abort();
            }
            linear0 += ix * real_space_region.strides[k];
        }
        int j;
        for(j = 0; j < p->len; j ++) {
            int iz = p->offset[2] + j;
            while(iz < 0) iz += Nmesh;
            while(iz >= Nmesh) iz -= Nmesh;
            if(iz >= real_space_region.size[2]) {
                /* seroius problem assmpution about pfft layout was wrong*/
                abort();
            }
            ptrdiff_t linear = iz * real_space_region.strides[2] + linear0;
            /* 
             * operate on the pencil, either modifying real or BufRecv 
             * */
            iter(&real[linear], &L->BufRecv[p->first + j]);
        }
    }
}
static void pm_alloc() {
    real = (double * ) malloc(fftsize * sizeof(double));
    complx = (pfft_complex *) malloc(fftsize * sizeof(double));
    rho_k = (pfft_complex * ) malloc(fftsize * sizeof(double));
#if 0
    /* this will set all of the values to NaNs 
     * if any of the points are not initialized it will mark everything
     * NaN.
     *
     * valgrind --malloc-fill=0xff does the same.
     * */
    memset(real , -1, sizeof(double) * fftsize);
    memset(complx, -1, sizeof(double) * fftsize);
    memset(rho_k, -1, sizeof(double) * fftsize);
#endif
    if(regions) {
        int i;
        size_t size = 0;
        for(i = 0 ; i < Nregions; i ++) {
            size += regions[i].totalsize;
        }
        meshbufsize = size;
        meshbuf = (double *) malloc(size * sizeof(double));
        size = 0;
        for(i = 0 ; i < Nregions; i ++) {
            regions[i].buffer = meshbuf + size;
            size += regions[i].totalsize;
        }
    }
}


static void pm_iterate_one(int i, pm_iterator iterator) {
    struct Region * region = &regions[0];
    int k;
    double cellsize = Box / Nmesh;
    int iCell[3];  /* integer coordinate on the regional mesh */
    double Res[3]; /* residual*/
    for(k = 0; k < 3; k++) {
        double tmp = P[i].Pos[k] / cellsize;
        iCell[k] = floor(tmp);
        Res[k] = tmp - iCell[k];
        iCell[k] -= region->offset[k];
        if(iCell[k] >= region->size[k] - 1) {
            /* seriously?! particles are supposed to be contained in cells */
            abort(); 
        }
        if(iCell[k] < 0) {
            abort();
        }
    }
    int connection = 0;
    for(connection = 0; connection < 8; connection++) {
        double weight = 1.0;
        ptrdiff_t linear = 0;
        for(k = 0; k < 3; k++) {
            int offset = (connection >> k) & 1;
            int tmp = iCell[k] + offset;
            linear += tmp * region->strides[k];
            weight *= offset?
                /* offset == 1*/ (Res[k])    :
                /* offset == 0*/ (1 - Res[k]);
        }
        if(linear >= region->totalsize) abort();
        iterator(i, &region->buffer[linear], weight);
    }
}

/* 
 * iterate over all particle / mesh pairs, call iterator 
 * function . iterator function shall be aware of thread safety.
 * no threads run on same particle same time but may 
 * access one mesh points same time.
 * */
static void pm_iterate(pm_iterator iterator) {
    int i;
#pragma omp parallel for 
    for(i = 0; i < NumPart; i ++) {
        pm_iterate_one(i, iterator); 
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

static void region_init_strides(struct Region * region) {
    int k;
    size_t rt = 1;
    for(k = 2; k >= 0; k --) {
        region->strides[k] = rt;
        rt = region->size[k] * rt;
    }
    region->totalsize = rt;
    region->buffer = NULL;
}

static int pos_get_target(const int pos[2]) {
    int k;
    int task2d[2];
    int rank;
    for(k = 0; k < 2; k ++) {
        int ix = pos[k];
        while(ix < 0) ix += Nmesh;
        while(ix >= Nmesh) ix -= Nmesh;
        task2d[k] = Mesh2Task[k][ix];
    }
    MPI_Cart_rank(comm_cart_2d, task2d, &rank);
    return rank;
}
static int pencil_cmp_target(const struct Pencil * p1, const struct Pencil * p2) {
    /* move zero length pixels to the end */
    if(p2->len == 0) return -1;
    if(p1->len == 0) return 1;
    int t1 = pos_get_target(p1->offset); 
    int t2 = pos_get_target(p2->offset); 
    return ((t2 < t1) - (t1 < t2)) * 2 +
        ((p2->meshbuf_first < p1->meshbuf_first) - (p1->meshbuf_first < p2->meshbuf_first));
}

static void pm_free() {
    if(regions) {
        free(meshbuf);
    }
    free(rho_k);
    free(complx);
    free(real);
}

static void pm_apply_transfer_function(struct Region * region, 
        pfft_complex * src, 
        pfft_complex * dst, transfer_function H
        ){
    ptrdiff_t ip = 0;

#pragma omp parallel for
    for(ip = 0; ip < region->totalsize; ip ++) {
        ptrdiff_t tmp = ip;
        int pos[3];
        int kpos[3];
        int64_t k2 = 0.0;
        int k;
        /* k goes like y, z, x */
        for(k = 0; k < 3; k ++) {
            pos[k] = tmp / region->strides[k];
            tmp -= pos[k] * region->strides[k];
            /* lets get the abs pos on the grid*/
            pos[k] += region->offset[k];
            /* check */
            if(pos[k] >= Nmesh) abort();
            kpos[k] = Mesh2K[pos[k]];
            /* Watch out the cast */
            k2 += ((int64_t)kpos[k]) * kpos[k];
        }
        /* swap 0 and 1 because fourier space was transposed */
        /* kpos is y, z, x */
        pos[0] = kpos[2];
        pos[1] = kpos[0];
        pos[2] = kpos[1];
        if(src) {
            dst[ip][0] = src[ip][0];
            dst[ip][1] = src[ip][1];
        }
        H(k2, pos, &dst[ip]);
    }

}


static int64_t reduce_int64(int64_t input) {
    int64_t result = 0;
    MPI_Allreduce(&input, &result, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    return result;
}

/** Some FFT notes
 *
 *
 * CFT = dx * iDFT (thus CFT has no 2pi factors and iCFT has, 
 *           same as wikipedia.)
 *
 * iCFT(CFT) = (2pi) ** 3, thus iCFT also has no 2pi factors.
 * **************************8*/
