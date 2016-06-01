#ifndef LAB5_MPI_ENV_H
#define LAB5_MPI_ENV_H

#include <mpi.h>
//#include "VT.h"

#define REORDER_ALLOWED 1

#define FLAG_NBR_PARTICLES 10
#define FLAG_NEW_PARTICLES 20

#define NO_NBR -1
#define ROOT_RANK 0

typedef struct mpi_types {
    MPI_Datatype particle;
} mpi_types_t;

typedef struct mpi_env {
    int nb_cpu, rank;
    int grid_size[2];
    int coords[2];

    int nbrs[3][3];

    mpi_types_t types;
    MPI_Comm grid_comm;
} mpi_env_t;



mpi_env_t init_env(void);

void quit_env(void);

#endif //LAB5_MPI_ENV_H
