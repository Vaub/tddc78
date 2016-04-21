#ifndef LAB1_MPI_ENV_H
#define LAB1_MPI_ENV_H

#include <mpi.h>
#include "types.h"

typedef struct ProcessEnv {
    int rank;
    int nb_cpu;
} ProcessEnv;

ProcessEnv MPI_ENV;

MPI_Datatype MPI_PIXEL;
MPI_Datatype MPI_IMG_PROP;
MPI_Datatype MPI_FILTER;

void mpi_init(int argc, char** argv);

void mpi_finalize();

#endif //LAB1_MPI_ENV_H
