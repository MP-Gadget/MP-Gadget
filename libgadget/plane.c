#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <ctype.h>
// #include <time.h>
#include "lenstools.h"
#include "utils.h"
#include "cosmology.h"
#include "plane.h"
#include "partmanager.h"
#include "petaio.h"
#include "utils.h"
#include "timefac.h"
// #include "timebinmgr.h"

static struct plane_params
{
    int64_t NormalsLength;
    int Normals[3];

    int64_t CutPointsLength;
    double CutPoints[1024];

    int Resolution;
    double Thickness; // in kpc/h
} PlaneParams;

int get_snap_number(const char *dir_path) {
    DIR *dir = opendir(dir_path);
    if (!dir) {
        // Directory cannot be opened
        endrun(0, "Output directory cannot be opened!\n");
    }

    struct dirent *entry;
    int max_number = -1;

    // Regex-like pattern to match "snap[a]_potentialPlane..."
    const char *pattern = "snap%d_potentialPlane";

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) {
            int number;
            char name[256];

            // Check if the filename matches the pattern
            if (sscanf(entry->d_name, pattern, &number) == 1) {
                if (number > max_number) {
                    max_number = number;
                }
            }
        }
    }

    closedir(dir);

    if (max_number == -1) {
        // No matching files found
        return 0;
    } else {
        // Return the next integer larger than the max found
        return max_number + 1;
    }
}

char *
plane_get_output_fname(const int snapnum, const char * OutputDir, const int cut, const int normal)
{
    char * fname = fastpm_strdup_printf("%s/snap%d_potentialPlane%d_normal%d.fits", OutputDir, snapnum, cut, normal);
    return fname;
}

int set_plane_normals(ParameterSet* ps)
{   
    char * Normals = param_get_string(ps, "PlaneNormals");
    char * strtmp = fastpm_strdup(Normals);
    char * token;
    int count;
    // print string
    message(0,"Normals = %s\n", Normals);

    /*First parse the string to get the number of outputs*/
    for(count=0, token=strtok(strtmp,","); token; count++, token=strtok(NULL, ","))
    {}

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
            endrun(1, "Requesting a normal direction beyond 0, 1 and 2: %g\n", n);
        }
        PlaneParams.Normals[count] = n;
/*         message(1, "Output at: %g\n", Sync.OutputListTimes[count]); */
    }
    myfree(strtmp);
    // print normals
    for(int i = 0; i < PlaneParams.NormalsLength; i++) {
        message(0,"Normals[%d] = %d\n", i, PlaneParams.Normals[i]);
    }
    return 0;
}

int set_plane_cuts(ParameterSet* ps)
{
    char * CutPoints = param_get_string(ps, "PlaneCutPoints");
    char * strtmp = fastpm_strdup(CutPoints);
    char * token;
    int64_t count;

    /* Note TimeInit and TimeMax not yet initialised here*/

    /*First parse the string to get the number of outputs*/
    for(count=0, token=strtok(strtmp,","); token; count++, token=strtok(NULL, ","))
    {}
/*     message(1, "Found %ld times in output list.\n", count); */

    /*Allocate enough memory*/
    PlaneParams.CutPointsLength = count;
    size_t maxcount = sizeof(PlaneParams.CutPoints) / sizeof(PlaneParams.CutPoints[0]);

    if((size_t) PlaneParams.CutPointsLength > maxcount) {
        message(1, "Too many entries (%ld) in the CutPoints, can take no more than %lu.\n", PlaneParams.CutPointsLength, maxcount);
        return 1;
    }
    /*Now read in the values*/
    for(count=0,token=strtok(CutPoints,","); count < PlaneParams.CutPointsLength && token; count++, token=strtok(NULL,","))
    {
        /* Skip a leading quote if one exists.
         * Extra characters are ignored by atof, so
         * no need to skip matching char.*/
        if(token[0] == '"')
            token+=1;

        double cut = atof(token);

        // if(cut < 0.0 ) {
        //     endrun(1, "Requesting a negative output scaling factor a = %g\n", a);
        // }
        PlaneParams.CutPoints[count] = cut;
/*         message(1, "Output at: %g\n", Sync.OutputListTimes[count]); */
    }
    myfree(strtmp);
    // print cut points
    for(int i = 0; i < PlaneParams.CutPointsLength; i++) {
        message(0,"CutPoints[%d] = %g\n", i, PlaneParams.CutPoints[i]);
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
        
        // Plane normals
        set_plane_normals(ps);

        // Plane cut points
        set_plane_cuts(ps);

        // plane resolution
        PlaneParams.Resolution = param_get_int(ps, "PlaneResolution");

        // plane thickness
        PlaneParams.Thickness = param_get_double(ps, "PlaneThickness");
        
    }
    MPI_Bcast(&PlaneParams, sizeof(struct plane_params), MPI_BYTE, 0, MPI_COMM_WORLD);
}

void write_plane(int SnapPlaneCount, const double atime, const Cosmology * CP, const char * OutputDir, const double UnitVelocity_in_cm_per_s) {

    int snapnum = get_snap_number(OutputDir);
        // simulation parameters and variables
    double BoxSize = PartManager->BoxSize;

    int64_t num_particles_tot = 0; // number of dark matter particles
    // Use MPI_Allreduce to get the total number of particles on all ranks
    MPI_Allreduce(&PartManager->NumPart, &num_particles_tot, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    // printf("Total number of particles: %ld\n", num_particles_tot);

    // plane parameters
    int plane_resolution = PlaneParams.Resolution;
    double thickness = PlaneParams.Thickness; // Example thickness in kpc/h
    // double cut_points[] = {4000, 12000, 20000};
    // int normals[] = {0, 1, 2};
    // int normals[] = {2};

    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    
    double redshift = 1./atime - 1.;
    message(0, "Computing and writing potential planes.\n");

    double **plane_result = allocate_2d_array(plane_resolution, plane_resolution);

    double **summed_plane_result = NULL;
        
    if(ThisTask == 0) {
        summed_plane_result = allocate_2d_array(plane_resolution, plane_resolution);
    }

    double comoving_distance = compute_comoving_distance(CP, atime, 1., UnitVelocity_in_cm_per_s);

    // print comoving distance
    message(0, "Comoving distance: %g\n", comoving_distance);

    /* loop over cut points and normal directions to generate lensing potential planes */
    for (int i = 0; i < PlaneParams.CutPointsLength; i++) {
        for (int j = 0; j < PlaneParams.NormalsLength; j++) {
            MPI_Barrier(MPI_COMM_WORLD);

            message(0, "Computing for cut %d and normal %d\n", atime, i, PlaneParams.Normals[j]);

            // Initialize lensing_potential with zeros
            for (int i = 0; i < plane_resolution; i++) {
                for (int j = 0; j < plane_resolution; j++) {
                    plane_result[i][j] = 0.0;  // Initially zero
                }
            }

            double left_corner[3] = {0, 0, 0};
            int64_t num_particles_plane = 0, num_particles_plane_tot = 0;
            // print input parameters

            

            /*computing lensing potential planes*/
            num_particles_plane = cutPlaneGaussianGrid(num_particles_tot,  comoving_distance, BoxSize, CP, atime, PlaneParams.Normals[j], PlaneParams.CutPoints[i], thickness, left_corner, plane_resolution, plane_result);
            
            /*sum up planes from all tasks*/
            for (int k = 0; k < plane_resolution; k++) {
            MPI_Reduce(plane_result[k], (ThisTask == 0 ? summed_plane_result[k] : NULL), plane_resolution, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            }  
            MPI_Reduce(&num_particles_plane, &num_particles_plane_tot, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);

            /*saving planes*/
            if (ThisTask == 0) {
                char * file_path;
                file_path = plane_get_output_fname(snapnum, OutputDir, i, PlaneParams.Normals[j]);
#ifdef USE_CFITSIO
                savePotentialPlane(summed_plane_result, plane_resolution, plane_resolution, file_path, BoxSize, CP, redshift, comoving_distance, num_particles_plane_tot);
#endif
            }
            message(0, "Plane saved for cut %d and normal %d\n", i, PlaneParams.Normals[j]);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    if (ThisTask == 0) {
        free_2d_array(summed_plane_result, plane_resolution);
        free_2d_array(plane_result, plane_resolution);

        char * buf = fastpm_strdup_printf("%s/info.txt", OutputDir);
        FILE * fd = fopen(buf, "a");
        fprintf(fd, "s=%d,d=%lf Mpc/h,z=%lf\n", snapnum, comoving_distance/1e3, redshift);
        fclose(fd);
        myfree(buf);
    }
}
