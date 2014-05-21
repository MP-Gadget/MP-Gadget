#ifdef PETA_PM
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pfft.h>

#include "allvars.h"
#include "proto.h"
static size_t HighMark_petapm = 0;
static struct Region {
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
} real_space_region, fourier_space_region;

static struct Region * regions = NULL;
static int Nregions = 0;
static void region_init(struct Region * region);

/* a batch is the communication object, represent a batch of
 * pencil / cells exchanged  */

struct Batch {
    int start_region;
    int end_region;
    int NpExport;
    int NcExport;
    int NpImport;
    int NcImport;
    int * NpSend;   
    int * NpRecv;
    int * NcSend;
    int * NcRecv;
    int * DpSend;
    int * DpRecv;
    int * DcSend;
    int * DcRecv;

    struct Pencil * PencilSend;
    struct Pencil * PencilRecv;
    double * BufSend;
    double * BufRecv;
    /* internal */
    int * ibuffer;
};
static void batch_prepare (struct Batch * B, int current_region);
static void batch_build_and_exchange_pencils(struct Batch * B);
static void batch_finish(struct Batch * B);
static void batch_build_and_exchange_cells_to_pfft(struct Batch * B);
static void batch_build_and_exchange_cells_to_regions(struct Batch * B);

/* cell_iterator nees to be thread safe !*/
typedef void (* cell_iterator)(double * cell_value, double * comm_buffer);
static void batch_iterate_cells(struct Batch * B, cell_iterator iter);

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
static int maxfftsize;
static void pm_alloc();
static void pm_free();
static pfft_plan plan_forw, plan_back;
MPI_Comm comm_cart_2d;
static int * ParticleRegion;
static int ThisTask2d[2];
static int NTask2d[2];
static int * (Mesh2Task[2]); /* convertion mesh to task2d */
static MPI_Datatype MPI_PENCIL;

void petapm_init_periodic(void) {
    All.Asmth[0] = ASMTH * All.BoxSize / PMGRID;
    All.Rcut[0] = RCUT * All.Asmth[0];
    pfft_init();
    ptrdiff_t n[3] = {PMGRID, PMGRID, PMGRID};
    ptrdiff_t np[2];
    Mesh2Task[0] = malloc(sizeof(int) * PMGRID);
    Mesh2Task[1] = malloc(sizeof(int) * PMGRID);

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

    region_init(&real_space_region);
    region_init(&fourier_space_region); 

    MPI_Allreduce(&fftsize, &maxfftsize, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    /* do the planning */
    pm_alloc();
    plan_forw = pfft_plan_dft_r2c_3d(
        n, real, rho_k, comm_cart_2d, PFFT_FORWARD, 
        PFFT_TRANSPOSED_OUT | PFFT_MEASURE | PFFT_DESTROY_INPUT);    
    plan_back = pfft_plan_dft_c2r_3d(
        n, complx, real, comm_cart_2d, PFFT_FORWARD, 
        PFFT_TRANSPOSED_OUT | PFFT_MEASURE | PFFT_DESTROY_INPUT);    

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
        int * tmp = (int*) alloca(sizeof(int) * PMGRID);
        for(i = 0; i < PMGRID; i ++) {
            tmp[i] = 0;
        }
        for(i = 0; i < real_space_region.size[k]; i ++) {
            tmp[i + real_space_region.offset[k]] = ThisTask2d[k];
        }
        /* which column / row hosts this tile? */
        /* FIXME: this is very inefficient */
        MPI_Allreduce(tmp, Mesh2Task[k], PMGRID, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        /*
        if(ThisTask == 0) {
            for(i = 0; i < PMGRID; i ++) {
                printf("Mesh2Task[%d][%d] == %d\n", k, i, Mesh2Task[k][i]);
            }
        }
        */
    }
}

static int pm_mark_region_for_node(int startno, int rid);

void petapm_prepare() {
    /* build a list of regions and record which region a particle belongs to */
    int i;
    int m;
    int no;
    double cellsize = All.BoxSize / PMGRID;
    regions = malloc(sizeof(struct Region) * NTopleaves);

    int r = 0;
    int k;

    for(m = 0; m < MULTIPLEDOMAINS; m++) {
        for(i = DomainStartList[ThisTask * MULTIPLEDOMAINS + m]; i <= DomainEndList[ThisTask * MULTIPLEDOMAINS + m]; i++) {
            no = DomainNodeIndex[i];
#if 0
            printf("task = %d no = %d len = %g hmax = %g center = %g %g %g\n",
                    ThisTask, no, Nodes[no].len, Extnodes[no].hmax, 
                    Nodes[no].center[0],
                    Nodes[no].center[1],
                    Nodes[no].center[2]);
#endif
            for(k = 0; k < 3; k ++) {
                regions[r].offset[k] = (Nodes[no].center[k] - Nodes[no].len * 0.5  - Extnodes[no].hmax) / cellsize;
                int end = (int) ((Nodes[no].center[k] + Nodes[no].len * 0.5  + Extnodes[no].hmax) / cellsize) + 1;
                regions[r].size[k] = end - regions[r].offset[k] + 1;
                regions[r].center[k] = Nodes[no].center[k];
            }

            /* setup the internal data structure of the region */
            region_init(&regions[r]);

            regions[r].len  = Nodes[no].len;
            regions[r].hmax = Extnodes[no].hmax;
            /* now lets mark particles to their hosting region */
            regions[r].numpart = pm_mark_region_for_node(no, r);
            r++;
        }
    }
    Nregions = r;
    int numpart = 0;
    for(r = 0; r < Nregions; r++) {
        numpart += regions[r].numpart;
    }
    /* all particles shall have been processed just once. */
    if(numpart != NumPart) {
        abort();
    }
}

void petapm_finish() {
    free(regions);
    regions = NULL;
}
    
static int pm_local_cic();
static int pm_regions_to_pfft(int * current_region);
static int pm_pfft_to_regions(int * current_region);

void petapm_force() {
    int current_region = 0;
    int ndone = 0;
    int iterations;
    int i;

    pm_alloc();

    memset(real, 0, sizeof(double) * fftsize);
    memset(meshbuf, 0, meshbufsize * sizeof(double));

    pm_local_cic();

    iterations = 0;
    current_region = 0;
    ndone = 0;
    while(ndone < NTask) {
        int done = pm_regions_to_pfft(&current_region);
        MPI_Allreduce(&done, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if(ThisTask == 0)
            printf("regions to pfft iteration = %d\n", iterations);
        iterations++;
    }
#if 1
    verify_density_field();
#endif

    /* potential */

    /* call pfft */

    /* read out the potential */
    iterations = 0;
    current_region = 0;
    ndone = 0;
    while(ndone < NTask) {
        int done = pm_pfft_to_regions(&current_region);
        MPI_Allreduce(&done, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }

    /* forces */
    /* call pfft */
    /* read out the forces */
    ndone = 0;
    while(ndone < NTask) {
 //       int done = pm_pfft_to_regions();
        int done = 1;
        MPI_Allreduce(&done, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }

    pm_free();
}


static void pm_put_particle_to_mesh(int i);
static int pm_local_cic() {
    int i;
    /* CIC */
    if(ThisTask == 0) 
        printf("starting CIC\n");
#pragma omp parallel for 
    for(i = 0; i < NumPart; i ++) {
        pm_put_particle_to_mesh(i); 
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if(ThisTask == 0) 
        printf("done CIC\n");
    return 1;
}

static int pm_regions_to_pfft(int * current_region) {
    struct Batch B;
    batch_prepare(&B, *current_region);
    batch_build_and_exchange_pencils(&B);
    batch_build_and_exchange_cells_to_pfft(&B);
    batch_finish(&B);
    *current_region = B.end_region;
    if(*current_region == Nregions)
        return 1;
    return 0;
}

static int pm_pfft_to_regions(int * current_region) {
    struct Batch B;
    batch_prepare(&B, *current_region);
    batch_build_and_exchange_pencils(&B);
    batch_build_and_exchange_cells_to_regions(&B);
    batch_finish(&B);
    *current_region = B.end_region;
    if(*current_region == Nregions)
        return 1;
    return 0;
}



/* build a batch of regions that fill buffersize */

static void batch_prepare (struct Batch * B, int current_region) {
    int r;
    int i;

    B->ibuffer = mymalloc("PMlayout", sizeof(int) * NTask * 8);

    memset(B->ibuffer, 0, sizeof(int) * NTask * 8);
    B->NpSend = &B->ibuffer[NTask * 0];
    B->NpRecv = &B->ibuffer[NTask * 1];
    B->NcSend = &B->ibuffer[NTask * 2];
    B->NcRecv = &B->ibuffer[NTask * 3];
    B->DcSend = &B->ibuffer[NTask * 4];
    B->DcRecv = &B->ibuffer[NTask * 5];
    B->DpSend = &B->ibuffer[NTask * 6];
    B->DpRecv = &B->ibuffer[NTask * 7];

    B->NpExport = 0;
    B->NcExport = 0;
    B->NpImport = 0;
    B->NcImport = 0;

    All.BunchSize = (int)(All.BufferSize * 1024 * 1024) / sizeof(double);

    /* count pencils until buffer would run out */
    for (r = current_region; r < Nregions; r ++) {
        if(B->NcExport + regions[r].totalsize > All.BunchSize 
            /* at least send one region */
                && r > current_region) {
            break;
        }
        B->NpExport += regions[r].size[0] * regions[r].size[1];
        B->NcExport += regions[r].totalsize;
    }

    B->start_region = current_region;
    B->end_region = r;

    for (r = B->start_region; r < B->end_region; r++) {
        int ix;
#pragma omp parallel for private(ix)
        for(ix = 0; ix < regions[r].size[0]; ix++) {
            int iy;
            for(iy = 0; iy < regions[r].size[1]; iy++) {
                int pos[2];
                pos[0] = ix + regions[r].offset[0];
                pos[1] = iy + regions[r].offset[1];
                int task = pos_get_target(pos);
#pragma omp atomic
                B->NpSend[task] ++;
#pragma omp atomic
                B->NcSend[task] += regions[r].size[2];
            }
        }
    }
    MPI_Alltoall(B->NpSend, 1, MPI_INT, B->NpRecv, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Alltoall(B->NcSend, 1, MPI_INT, B->NcRecv, 1, MPI_INT, MPI_COMM_WORLD);

    B->DpSend[0] = 0; B->DpRecv[0] = 0;
    B->DcSend[0] = 0; B->DcRecv[0] = 0;
    for(i = 1; i < NTask; i ++) {
        B->DpSend[i] = B->NpSend[i - 1] + B->DpSend[i - 1];
        B->DpRecv[i] = B->NpRecv[i - 1] + B->DpRecv[i - 1];
        B->DcSend[i] = B->NcSend[i - 1] + B->DcSend[i - 1];
        B->DcRecv[i] = B->NcRecv[i - 1] + B->DcRecv[i - 1];
    }
    B->NpImport = B->DpRecv[NTask -1] + B->NpRecv[NTask -1];
    B->NcImport = B->DcRecv[NTask -1] + B->NcRecv[NTask -1];

    /* some checks */
    if(B->DpSend[NTask - 1] + B->NpSend[NTask -1] != B->NpExport) abort();
    if(B->DcSend[NTask - 1] + B->NcSend[NTask -1] != B->NcExport) abort();

    int64_t totNpExport = reduce_int64(B->NpExport);
    int64_t totNcExport = reduce_int64(B->NcExport);
    int64_t totNpImport = reduce_int64(B->NpImport);
    int64_t totNcImport = reduce_int64(B->NcImport);
    if(totNpExport != totNpImport) {
        abort();
    }
    if(totNcExport != totNcImport) {
        abort();
    }
    if(ThisTask == 0) {
        printf("Exchange of %010ld Pencils and %010ld Cells\n", totNpExport, totNcExport);
    }
    
    B->PencilSend = mymalloc("PencilSend", B->NpExport * sizeof(struct Pencil));
    B->PencilRecv = mymalloc("PencilRecv", B->NpImport * sizeof(struct Pencil));
    B->BufSend = mymalloc("PMBufSend", B->NcExport * sizeof(double));
    B->BufRecv = mymalloc("PMBufRecv", B->NcImport * sizeof(double));

}

static void batch_build_and_exchange_pencils(struct Batch * B) {
    int r;
    int i;
    int offset;

    /* now build pencils to be exported */
    int p0 = 0;
    for (r = B->start_region; r < B->end_region; r++) {
        int ix;
#pragma omp parallel for private(ix)
        for(ix = 0; ix < regions[r].size[0]; ix++) {
            int iy;
            for(iy = 0; iy < regions[r].size[1]; iy++) {
                int poffset = ix * regions[r].size[1] + iy;
                struct Pencil * p = &B->PencilSend[p0 + poffset];

                p->offset[0] = ix + regions[r].offset[0];
                p->offset[1] = iy + regions[r].offset[1];
                p->offset[2] = regions[r].offset[2];
                p->len = regions[r].size[2];
                p->meshbuf_first = (regions[r].buffer - meshbuf) +
                    regions[r].strides[0] * ix +
                    regions[r].strides[1] * iy;
                p->task = pos_get_target(p->offset);
            }
        }
        p0 += regions[r].size[0] * regions[r].size[1];
    }

    /* sort the pencils by the target rank for ease of next step */
    qsort(B->PencilSend, B->NpExport, sizeof(struct Pencil), pencil_cmp_target);

    /* build the first pointers to refer to the correct relative buffer locations */
    /* note that the buffer hasn't bee assembled yet */
    offset = 0;
    for(i = 0; i < NTask; i ++) {
        int j;
        struct Pencil * p = &B->PencilSend[offset];
        p->first = 0;
        for(j = 1; j < B->NpSend[i]; j++) {
            p[j].first = p[j - 1].first + p[j - 1].len;
        }
        offset += B->NpSend[i];
    }

    MPI_Alltoallv(
            B->PencilSend, B->NpSend, B->DpSend, MPI_PENCIL,
            B->PencilRecv, B->NpRecv, B->DpRecv, MPI_PENCIL, 
            MPI_COMM_WORLD);

    /* set first to point to absolute position in the full import cell buffer */
    offset = 0;
    for(i = 0; i < NTask; i ++) {
        struct Pencil * p = &B->PencilRecv[offset];
        int j;
        for(j = 0; j < B->NpRecv[i]; j++) {
            p[j].first += B->DcRecv[i];
        }
        offset += B->NpRecv[i];
    }
    
    /* set first to point to absolute position in the full export cell buffer */
    offset = 0;
    for(i = 0; i < NTask; i ++) {
        struct Pencil * p = &B->PencilSend[offset];
        int j;
        for(j = 0; j < B->NpSend[i]; j++) {
            p[j].first += B->DcSend[i];
        }
        offset += B->NpSend[i];
    }
}

static void batch_finish(struct Batch * B) {
    myfree(B->BufRecv);
    myfree(B->BufSend);
    myfree(B->PencilRecv);
    myfree(B->PencilSend);
    myfree(B->ibuffer);
}

/* exchange cells to their pfft host, then reduce the cells to the pfft
 * array */
static void to_pfft(double * cell, double * buf) {
#pragma omp atomic
            cell[0] += buf[0];
}
static void batch_build_and_exchange_cells_to_pfft(struct Batch * B) {
    int i;
    int offset;

    /* collect all cells into the send buffer */
    offset = 0;
    for(i = 0; i < B->NpExport; i ++) {
        struct Pencil * p = &B->PencilSend[i];
        memcpy(B->BufSend + offset, &meshbuf[p->meshbuf_first], 
                sizeof(double) * p->len);
        offset += p->len;
    }

    /* receive cells */
    MPI_Alltoallv(
            B->BufSend, B->NcSend, B->DcSend, MPI_DOUBLE,
            B->BufRecv, B->NcRecv, B->DcRecv, MPI_DOUBLE, 
            MPI_COMM_WORLD);

#if 1
    double massExport = 0;
    for(i = 0; i < B->NcExport; i ++) {
        massExport += B->BufSend[i];
    }

    double massImport = 0;
    for(i = 0; i < B->NcImport; i ++) {
        massImport += B->BufRecv[i];
    }
    double totmassExport;
    double totmassImport;
    MPI_Allreduce(&massExport, &totmassExport, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&massImport, &totmassImport, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(ThisTask == 0) {
        printf("totmassExport = %g totmassImport = %g\n", totmassExport, totmassImport);
    }
#endif

    batch_iterate_cells(B, to_pfft);
}

/* readout cells on their pfft host, then exchange the cells to the domain 
 * host */
static void to_region(double * cell, double * region) {
    *region = *cell;
}

static void batch_build_and_exchange_cells_to_regions(struct Batch * B) {
    int i;
    int offset;

    batch_iterate_cells(B, to_region);

    /* exchange cells */
    /* notice the order is reversed from to_pfft */
    MPI_Alltoallv(
            B->BufRecv, B->NcRecv, B->DcRecv, MPI_DOUBLE, 
            B->BufSend, B->NcSend, B->DcSend, MPI_DOUBLE,
            MPI_COMM_WORLD);

    /* distribute BufSend to meshbuf */
    offset = 0;
    for(i = 0; i < B->NpExport; i ++) {
        struct Pencil * p = &B->PencilSend[i];
        memcpy(&meshbuf[p->meshbuf_first], 
                B->BufSend + offset, 
                sizeof(double) * p->len);
        offset += p->len;
    }
}

/* iterate over the pairs of real field cells and RecvBuf cells 
 *
 * !!! iter has to be thread safe. !!!
 * */
static void batch_iterate_cells(struct Batch * B, cell_iterator iter) {
    int i;
#pragma omp parallel for
    for(i = 0; i < B->NpImport; i ++) {
        struct Pencil * p = &B->PencilRecv[i];
        int k;
        ptrdiff_t linear0 = 0;
        for(k = 0; k < 2; k ++) {
            int ix = p->offset[k];
            while(ix < 0) ix += PMGRID;
            while(ix >= PMGRID) ix -= PMGRID;
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
            while(iz < 0) iz += PMGRID;
            while(iz >= PMGRID) iz -= PMGRID;
            if(iz >= real_space_region.size[2]) {
                /* seroius problem assmpution about pfft layout was wrong*/
                abort();
            }
            ptrdiff_t linear = iz * real_space_region.strides[2] + linear0;
            /* most import line here, read out the real space field to the
             * pencil */
            iter(&real[linear], &B->BufRecv[p->first + j]);
        }
    }
}
static void pm_alloc() {
    real = (double * ) mymalloc("PMreal", fftsize * sizeof(double));
    complx = (pfft_complex *) mymalloc("PMcomplex", fftsize * sizeof(double));
    rho_k = (pfft_complex * ) mymalloc("PMrho_k", fftsize * sizeof(double));
    if(regions) {
        int i;
        size_t size = 0;
        for(i = 0 ; i < Nregions; i ++) {
            size += regions[i].totalsize;
        }
        meshbufsize = size;
        meshbuf = (double *) mymalloc("PMmesh", size * sizeof(double));
        report_memory_usage(&HighMark_petapm, "PetaPM");
        size = 0;
        for(i = 0 ; i < Nregions; i ++) {
            regions[i].buffer = meshbuf + size;
            size += regions[i].totalsize;
        }
    }
}

static void pm_put_particle_to_mesh(int i) {
    struct Region * region = &regions[P[i].RegionInd];
    int k;
    double cellsize = All.BoxSize / PMGRID;
    int iCell[3];  /* integer coordinate on the regional mesh */
    double Res[3]; /* residual*/
    for(k = 0; k < 3; k++) {
        double tmp = P[i].Pos[k] / cellsize;
        iCell[k] = tmp;
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
    double mass = P[i].Mass;
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
#pragma omp atomic
        region->buffer[linear] += weight * mass;
    }
}
static int pm_mark_region_for_node(int startno, int rid) {
    int numpart = 0;
    int p;
    int no = Nodes[startno].u.d.nextnode;
    while(no >= 0)
    {
        if(no < All.MaxPart)	/* single particle */
        {
            p = no;
            no = Nextnode[no];
            drift_particle(p, All.Ti_Current);
            P[p].RegionInd = rid;
            numpart ++;
        }
        else
        {
            if(no >= All.MaxPart + MaxNodes)	/* pseudo particle */
            {
                /* skip pseudo particles */
                no = Nextnode[no - MaxNodes];
                continue;
            }

            if(Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL))	
                /* we reached a top-level node again, which means that we are done with the branch */
            {
                break;
            }
            force_drift_node(no, All.Ti_Current);

            no = Nodes[no].u.d.nextnode;	/* ok, we need to open the node */
        }
    }
    return numpart;
}

static void region_init(struct Region * region) {
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
        while(ix < 0) ix += PMGRID;
        while(ix >= PMGRID) ix -= PMGRID;
        task2d[k] = Mesh2Task[k][ix];
    }
    MPI_Cart_rank(comm_cart_2d, task2d, &rank);
    return rank;
}
static int pencil_cmp_target(const struct Pencil * p1, const struct Pencil * p2) {
    int t1 = pos_get_target(p1->offset); 
    int t2 = pos_get_target(p2->offset); 
    return ((t2 < t1) - (t1 < t2)) * 2 +
        ((p2->first < p1->first) - (p1->first < p2->first));
}

static void pm_free() {
    if(regions) {
        myfree(meshbuf);
    }
    myfree(rho_k);
    myfree(complx);
    myfree(real);
}
static void verify_density_field() {
    int i;
    /* verify the density field */
    double mass_Part = 0;
#pragma omp parallel for reduction(+: mass_Part)
    for(i = 0; i < NumPart; i ++) {
        mass_Part += P[i].Mass;
    }
    double totmass_Part = 0;
    MPI_Allreduce(&mass_Part, &totmass_Part, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double mass_Region = 0;
#pragma omp parallel for reduction(+: mass_Region)
    for(i = 0; i < meshbufsize; i ++) {
        mass_Region += meshbuf[i];    
    }
    double totmass_Region = 0;
    MPI_Allreduce(&mass_Region, &totmass_Region, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double mass_CIC = 0;
#pragma omp parallel for reduction(+: mass_CIC)
    for(i = 0; i < real_space_region.totalsize; i ++) {
        mass_CIC += real[i];
    }
    double totmass_CIC = 0;
    MPI_Allreduce(&mass_CIC, &totmass_CIC, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    printf("on me Region mass = %g CIC mass = %g Particle mass = %g\n", mass_Region, mass_CIC, mass_Part);
    if(ThisTask == 0) {
        printf("total Region mass = %g CIC mass = %g Particle mass = %g\n", totmass_Region, totmass_CIC, totmass_Part);
    }
}

static int64_t reduce_int64(int64_t input) {
    int64_t result = 0;
    MPI_Allreduce(&input, &result, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    return result;
}

#endif
