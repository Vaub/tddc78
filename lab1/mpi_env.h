#ifndef LAB1_MPI_ENV_H
#define LAB1_MPI_ENV_H

typedef struct MpiTypes {
    MPI_Datatype pixel;
    MPI_Datatype image;
    MPI_Datatype filter;
    MPI_Datatype chunk;
} MpiTypes;

typedef struct MpiEnv {
    MPI_Comm comm;

    int rank;
    int nb_cpu;

    MpiTypes types;
} MpiEnv;

#define ROOT_RANK 0

void init_mpi(int argc, char** argv);
void close_mpi();

const MpiEnv* get_env();

#endif //LAB1_MPI_ENV_H
