#include "mpi_env.h"

void mpi_init(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_ENV.nb_cpu);
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_ENV.rank);

    MPI_PIXEL = register_pixel_type();
    MPI_IMG_PROP = register_img_prop_type();
    MPI_FILTER = register_filter_type();
}

void mpi_finalize() {
    MPI_Finalize();
}

