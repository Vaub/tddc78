#include <blurfilter.h>
#include "types.h"

MPI_Datatype register_pixel_type() {
    pixel item;
    MPI_Datatype pixel_mpi;

    int block_lengths [] = { 1, 1, 1 };
    MPI_Datatype block_types [] = { MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR };
    MPI_Aint start, displ[3];

    MPI_Get_address(&item, &start);
    MPI_Get_address(&item.r, &displ[0]);
    MPI_Get_address(&item.g, &displ[1]);
    MPI_Get_address(&item.b, &displ[2]);

    displ[0] -= start;
    displ[1] -= start;
    displ[2] -= start;
    MPI_Type_create_struct(3, block_lengths, displ, block_types, &pixel_mpi);

    MPI_Type_commit(&pixel_mpi);

    return pixel_mpi;
}

MPI_Datatype register_img_prop_type() {
    ImageProperties item;
    MPI_Datatype img_prop_mpi;

    int block_lengths [] = { 1, 1, 1 };
    MPI_Datatype block_types [] = { MPI_INT, MPI_INT, MPI_INT };
    MPI_Aint start, displ[3];

    MPI_Get_address(&item, &start);
    MPI_Get_address(&item.width, &displ[0]);
    MPI_Get_address(&item.height, &displ[1]);
    MPI_Get_address(&item.color_depth, &displ[2]);

    displ[0] -= start;
    displ[1] -= start;
    displ[2] -= start;
    MPI_Type_create_struct(3, block_lengths, displ, block_types, &img_prop_mpi);

    MPI_Type_commit(&img_prop_mpi);

    return img_prop_mpi;
}

MPI_Datatype register_filter_type() {
    Filter item;
    MPI_Datatype img_filter_mpi;

    int block_lengths [] = { 1, MAX_RAD };
    MPI_Datatype block_types [] = { MPI_INT, MPI_DOUBLE };
    MPI_Aint start, displ[2];

    MPI_Get_address(&item, &start);
    MPI_Get_address(&item.radius, &displ[0]);
    MPI_Get_address(&item.weights, &displ[1]);

    displ[0] -= start;
    displ[1] -= start;
    MPI_Type_create_struct(2, block_lengths, displ, block_types, &img_filter_mpi);

    MPI_Type_commit(&img_filter_mpi);

    return img_filter_mpi;
}