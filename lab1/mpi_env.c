#include <mpi.h>
#include "mpi_env.h"
#include "types.h"
#include "blur_filter.h"

static MpiEnv env;

void create_pixel_type(MPI_Datatype* type) {

    Pixel pixel;

    int block_len[3] = { 1, 1, 1 };
    MPI_Datatype types[3] = { MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR };
    MPI_Aint start, disp[3];

    MPI_Get_address(&pixel, &start);
    MPI_Get_address(&pixel.r, &disp[0]);
    MPI_Get_address(&pixel.g, &disp[1]);
    MPI_Get_address(&pixel.b, &disp[2]);

    disp[0] -= start;
    disp[1] -= start;
    disp[2] -= start;
    PMPI_Type_create_struct(3, block_len, disp, types, type);

    MPI_Type_commit(type);

}

void create_image_type(MPI_Datatype* type) {
    Image image;

    int block_len[2] = { 1, 1 };
    MPI_Datatype types[2] = { MPI_INT, MPI_INT };
    MPI_Aint start, disp[2];

    MPI_Get_address(&image, &start);
    MPI_Get_address(&image.width, &disp[0]);
    MPI_Get_address(&image.height, &disp[1]);

    disp[0] -= start;
    disp[1] -= start;
    PMPI_Type_create_struct(2, block_len, disp, types, type);

    MPI_Type_commit(type);
}

void create_filter_type(MPI_Datatype* type) {
    Filter filter;

    int block_len[2] = { 1, MAX_RADIUS };
    MPI_Datatype types[2] = { MPI_INT, MPI_DOUBLE };
    MPI_Aint start, disp[2];

    MPI_Get_address(&filter, &start);
    MPI_Get_address(&filter.radius, &disp[0]);
    MPI_Get_address(&filter.weights, &disp[1]);

    disp[0] -= start;
    disp[1] -= start;
    PMPI_Type_create_struct(2, block_len, disp, types, type);

    MPI_Type_commit(type);
}

void create_chunk_type(MPI_Datatype* type) {

    ImageChunk chunk;

    int block_len[3] = { 1, 1, 1 };
    MPI_Datatype types[3] = { MPI_INT, MPI_INT, MPI_INT };
    MPI_Aint start, disp[3];

    MPI_Get_address(&chunk, &start);
    MPI_Get_address(&chunk.img_idx, &disp[0]);
    MPI_Get_address(&chunk.start_offset, &disp[1]);
    MPI_Get_address(&chunk.nb_pix_to_treat, &disp[2]);

    disp[0] -= start;
    disp[1] -= start;
    disp[2] -= start;
    PMPI_Type_create_struct(3, block_len, disp, types, type);

    MPI_Type_commit(type);

}

void create_mpi_types(MpiTypes* types) {
    create_pixel_type(&types->pixel);
    create_image_type(&types->image);
    create_filter_type(&types->filter);
    create_chunk_type(&types->chunk);
}

void init_mpi(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    env.comm = MPI_COMM_WORLD;
    MPI_Comm_rank(MPI_COMM_WORLD, &env.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &env.nb_cpu);

    create_mpi_types(&env.types);

}

void close_mpi() {
    MPI_Finalize();
}

const MpiEnv* get_env() {
    return &env;
}
