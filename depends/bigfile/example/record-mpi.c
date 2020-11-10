#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include <bigfile-mpi.h>

static int BH_RECORD_POSITION = 0;
static int BH_RECORD_VELOCITY = 0;
static int BH_RECORD_BHMASS = 0;
static BigRecordType BH_RECORD[1] = {{0}};

void
init_bh_record_type()
{
    big_record_type_clear(BH_RECORD);
    BH_RECORD_POSITION = big_record_type_add(BH_RECORD, "Position", "f8", 3);
    BH_RECORD_VELOCITY = big_record_type_add(BH_RECORD, "Velocity", "f4", 3);
    BH_RECORD_BHMASS = big_record_type_add(BH_RECORD, "BHMass", "f4", 1);
    big_record_type_complete(BH_RECORD);
}

int
main(int argc, char * argv[])
{
    init_bh_record_type();

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int nrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nrank);

    BigFile file[1];

    big_file_mpi_create(file, "/tmp/example", MPI_COMM_WORLD);
    big_file_mpi_create_records(file, BH_RECORD, "w+", 2, (size_t[]){100, 100}, MPI_COMM_WORLD);

    int nmemb = 200 * (rank + 1) / nrank - 200 * (rank) / nrank;
    void * bufin = calloc(nmemb, BH_RECORD->itemsize);
    void * bufout = calloc(nmemb, BH_RECORD->itemsize);

    for(int i = 0; i < nmemb; i ++) {
        double pos[3] = {i, i * 10, i * 100};
        float vel[3] = {i + 1, i * 10 + 1, i * 100 + 1};
        float bhmass = i;
        big_record_set(BH_RECORD, bufout, i, BH_RECORD_POSITION, &pos);
        big_record_set(BH_RECORD, bufout, i, BH_RECORD_VELOCITY, &vel);
        big_record_set(BH_RECORD, bufout, i, BH_RECORD_BHMASS, &bhmass);
    }

    big_file_mpi_write_records(file, BH_RECORD, 0, nmemb, bufout, nrank, MPI_COMM_WORLD);

    big_file_mpi_create_records(file, BH_RECORD, "a+", 1, (size_t[]){200}, MPI_COMM_WORLD);
    big_file_mpi_write_records(file, BH_RECORD, 200, nmemb, bufout, nrank, MPI_COMM_WORLD);

    /* verify */
    big_file_mpi_read_records(file, BH_RECORD, 0, nmemb, bufin, 1, MPI_COMM_WORLD);
    if(0 != memcmp(bufout, bufin, BH_RECORD->itemsize * nmemb)) {
        abort();
    }
    big_file_mpi_read_records(file, BH_RECORD, 200, nmemb, bufin, 1, MPI_COMM_WORLD);
    if(0 != memcmp(bufout, bufin, BH_RECORD->itemsize * nmemb)) {
        abort();
    }

    big_file_mpi_close(file, MPI_COMM_WORLD);
    big_record_type_clear(BH_RECORD);
    MPI_Finalize();
    return 0;
}
