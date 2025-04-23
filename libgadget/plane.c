#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include "lenstools.h"
#include "utils/string.h"
#include "utils/mymalloc.h"
#include "cosmology.h"
#include "plane.h"
#include "physconst.h"
#include "partmanager.h"
#include "timefac.h"
#include "utils/endrun.h"
#include "utils/system.h"
#include "timebinmgr.h"

static struct plane_params
{
    int64_t NormalsLength;
    int Normals[3];

    int64_t CutPointsLength;
    double CutPoints[1024];

    int Resolution;
    double Thickness; // in kpc/h
} PlaneParams;

char *
plane_get_output_fname(const int snapnum, const char * OutputDir, const int cut, const int normal)
{
    // Format the filename to include '!' to overwrite existing files
    char * fname = fastpm_strdup_printf("!%s/snap%d_potentialPlane%d_normal%d.fits", OutputDir, snapnum, cut, normal);
    return fname;
}

/* This is basically BuildOutputList but with an integer*/
int set_plane_normals(ParameterSet* ps)
{
    char * Normals = param_get_string(ps, "PlaneNormals");
    if(!Normals){
        PlaneParams.NormalsLength = 0;
        return 0;
    }
    char * strtmp = fastpm_strdup(Normals);
    char * token;
    int count;
    // print string
    //message(0,"Normals = %s\n", Normals);

    /*First parse the string to get the number of outputs*/
    for(count=0, token=strtok(strtmp,","); token; count++, token=strtok(NULL, ","))
    {}

    myfree(strtmp);

    /*Allocate enough memory*/
    PlaneParams.NormalsLength = count;
    size_t maxcount = sizeof(PlaneParams.Normals) / sizeof(PlaneParams.Normals[0]);

    if((size_t) PlaneParams.NormalsLength > maxcount) {
        message(1, "Too many entries (%ld) in the Normals, can take no more than %lu.\n", PlaneParams.NormalsLength, maxcount);
        return 1;
    }
    /*Now read in the values*/
    for(count=0,token=strtok(Normals,","); count < PlaneParams.NormalsLength && token; count++, token=strtok(NULL,","))
    {
        /* Skip a leading quote if one exists.
         * Extra characters are ignored by atof, so
         * no need to skip matching char.*/
        if(token[0] == '"')
            token+=1;

        int n = atoi(token);

        if(n != 0 && n != 1 && n != 2) {
            endrun(1, "Requesting a normal direction beyond 0, 1 and 2: %d\n", n);
        }
        PlaneParams.Normals[count] = n;
/*         message(1, "Output at: %g\n", Sync.OutputListTimes[count]); */
    }
    return 0;
}

/*Set the plane parameters*/
void
set_plane_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        // plane resolution
        PlaneParams.Resolution = param_get_int(ps, "PlaneResolution");

        // plane thickness
        PlaneParams.Thickness = param_get_double(ps, "PlaneThickness");

        // Plane normals
        set_plane_normals(ps);

        // Plane cut points
        if (!param_get_string(ps, "PlaneCutPoints")) {
            message(0, "No cut points provided, a set of default values will be set: (1/2 + i) * plane thickness (< box size, i = 0, 1, 2...)\n");
            // set the length to 0
            PlaneParams.CutPointsLength = 0;
        }
        else
            BuildOutputList(ps, "PlaneCutPoints", PlaneParams.CutPoints, &PlaneParams.CutPointsLength, 1024);
    }
    MPI_Bcast(&PlaneParams, sizeof(struct plane_params), MPI_BYTE, 0, MPI_COMM_WORLD);
}

void write_plane(int snapnum, const double atime, Cosmology * CP, const char * OutputDir, const double UnitVelocity_in_cm_per_s, const double UnitLength_in_cm) {

    double BoxSize = PartManager->BoxSize;

    /* NOTE: this is correct only for pure DM runs because this code is called on a PM step and we garbage collect after the exchange.
     * It is not generally the total number of particles*/
    int64_t num_particles_tot = 0; // number of dark matter particles
    // Use MPI_Allreduce to get the total number of particles on all ranks
    MPI_Allreduce(&PartManager->NumPart, &num_particles_tot, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    // printf("Total number of particles: %ld\n", num_particles_tot);

    // plane parameters
    int plane_resolution = PlaneParams.Resolution;
    double thickness = PlaneParams.Thickness; // in kpc/h
    if (thickness <= 0.0) {
        message(0, "No positive thickness provided, the side length of the box, %g, will be used.\n", BoxSize);
        thickness = BoxSize;
    }

    // set a set of cut points if NULL
    if (PlaneParams.CutPointsLength == 0) {
        PlaneParams.CutPointsLength = (int64_t) (BoxSize / thickness);
        for (int i = 0; i < PlaneParams.CutPointsLength; i++) {
            PlaneParams.CutPoints[i] = (.5 + i) * thickness;
        }
        // print cut points
        message(0, "Cut points set automatically:\n");
        for (int i = 0; i < PlaneParams.CutPointsLength; i++) {
            message(0,"CutPoints[%d] = %g\n", i, PlaneParams.CutPoints[i]);
        }
    }


    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    double redshift = 1./atime - 1.;
    message(0, "Computing and writing potential planes.\n");

    double *plane_result = allocate_2d_array_as_1d(plane_resolution, plane_resolution);
    double *summed_plane_result = allocate_2d_array_as_1d(plane_resolution, plane_resolution);

    double comoving_distance = compute_comoving_distance(CP, atime, 1., UnitVelocity_in_cm_per_s);

    // print comoving distance
    message(0, "Comoving distance: %g\n", comoving_distance);

    /* loop over cut points and normal directions to generate lensing potential planes */
    for (int i = 0; i < PlaneParams.CutPointsLength; i++) {
        for (int j = 0; j < PlaneParams.NormalsLength; j++) {
            message(0, "Computing for cut point %g and normal %d\n", PlaneParams.CutPoints[i], PlaneParams.Normals[j]);

            double left_corner[3] = {0, 0, 0};
            int64_t num_particles_plane = 0, num_particles_plane_tot = 0;

            /*computing lensing potential planes*/
            num_particles_plane = cutPlaneGaussianGrid(num_particles_tot,  comoving_distance, BoxSize, CP, atime, PlaneParams.Normals[j], PlaneParams.CutPoints[i], thickness, left_corner, plane_resolution, plane_result);

            /*sum up planes from all tasks*/
            MPI_Reduce(plane_result, summed_plane_result, plane_resolution * plane_resolution, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&num_particles_plane, &num_particles_plane_tot, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);

            /*saving planes*/
            if (ThisTask == 0) {
#ifdef USE_CFITSIO
                char * file_path = plane_get_output_fname(snapnum, OutputDir, i, PlaneParams.Normals[j]);
                savePotentialPlane(summed_plane_result, plane_resolution, plane_resolution, file_path, BoxSize, CP, redshift, comoving_distance, num_particles_plane_tot, UnitLength_in_cm);
                message(0, "Plane saved for cut %d and normal %d to %s\n", i, PlaneParams.Normals[j], file_path + 1); // skip the '!' in the filename
                myfree(file_path);
#endif
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    myfree(summed_plane_result);
    myfree(plane_result);


    if (ThisTask == 0) {
        double comoving_distance_Mpc  = comoving_distance * UnitLength_in_cm / CM_PER_MPC;
        char * buf = fastpm_strdup_printf("%s/info.txt", OutputDir);
        FILE * fd = fopen(buf, "a");
        fprintf(fd, "s=%d,d=%lf Mpc/h,z=%lf\n", snapnum, comoving_distance_Mpc, redshift);
        fclose(fd);
        myfree(buf);
    }
}
