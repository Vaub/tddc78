#include <cmath>
#include "mpi_env.h"
#include "definitions.h"

MPI_Datatype create_pcord() {
    pcord_t item;
    MPI_Datatype pcord_mpi;

    int block_length[4] = { 1,1,1,1 }; // float,float,float,float
    MPI_Datatype types[4] = { MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT };
    MPI_Aint start, displ[4];

    MPI_Get_address(&item, &start);
    MPI_Get_address(&item.x, &displ[0]);
    MPI_Get_address(&item.y, &displ[1]);
    MPI_Get_address(&item.vx, &displ[2]);
    MPI_Get_address(&item.vx, &displ[3]);

    displ[0] -= start;
    displ[1] -= start;
    displ[2] -= start;
    displ[3] -= start;

    MPI_Type_create_struct(4, block_length, displ, types, &pcord_mpi);
    MPI_Type_commit(&pcord_mpi);

    return pcord_mpi;
}

MPI_Datatype create_particle(const MPI_Datatype& pcord_mpi) {
    particle_t item;
    MPI_Datatype particle_mpi;

    int block_length[2] = { 1,1 }; // pcord_t,int
    MPI_Datatype types[2] = { pcord_mpi, MPI_INT };
    MPI_Aint start, displ[2];

    MPI_Get_address(&item, &start);
    MPI_Get_address(&item.pcord, &displ[0]);
    MPI_Get_address(&item.ptype, &displ[1]);

    displ[0] -= start;
    displ[1] -= start;

    MPI_Type_create_struct(2, block_length, displ, types, &particle_mpi);
    MPI_Type_commit(&particle_mpi);

    return particle_mpi;
}

mpi_types_t create_types(void) {
    mpi_types_t types = { create_particle(create_pcord()) };
    return types;
}


void find_nbr_rank(MPI_Comm comm, int nb_rows, int nb_cols, const int nbr_coords[2], int* nbr_rank) {
    if ((nbr_coords[0] > -1 && nbr_coords[0] < nb_rows)
        && (nbr_coords[1] > -1 && nbr_coords[1] < nb_cols)) {
        MPI_Cart_rank(comm, nbr_coords, nbr_rank);
    }
    else {
        *nbr_rank = NO_NBR;
    }
}

mpi_env_t init_env(int* argc, char** argv[]) {
    MPI_Init(argc, argv);

    int nb_cpu = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &nb_cpu);

    int factor = (int)sqrt(nb_cpu);
    int nb_rows = factor, nb_cols = ceil(nb_cpu / factor);
    int nb_procs = nb_rows * nb_cols;

    if (nb_cpu > nb_procs && nb_cpu % 2 == 0) {
	nb_rows = 2;
	nb_cols = nb_cpu / 2;

        nb_procs = nb_rows * nb_cols;
    }

    if (nb_cpu > nb_procs) {
        printf("%d process will not fit in a [%d x %d] grid!\n", nb_cpu, nb_rows, nb_cols);
        exit(4);
    }

    int dims[2] = { nb_rows, nb_cols };
    MPI_Dims_create(nb_procs, 2, dims);

    int periods[2] = { 0, 0 };

    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, REORDER_ALLOWED, &grid_comm);

    mpi_env_t env;
    env.types = create_types();

    env.nb_cpu = nb_procs;
    env.grid_comm = grid_comm;
    env.grid_size[0] = nb_rows; env.grid_size[1] = nb_cols;

    MPI_Cart_get(env.grid_comm, 2, dims, periods, env.coords);
    MPI_Cart_rank(env.grid_comm, env.coords, &env.rank);
    int x = env.coords[0], y = env.coords[1];

    if (env.rank == ROOT_RANK) {
        //printf("Init %d procs on a [%d x %d] grid\n", nb_procs, nb_rows, nb_cols);
    }

    int nbrs_coords[3][3][2];
    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            nbrs_coords[i][j][0] = x - 1 + i;
            nbrs_coords[i][j][1] = y - 1 + j;
            find_nbr_rank(env.grid_comm, nb_rows, nb_cols, nbrs_coords[i][j], &env.nbrs[i][j]);
        }
    }

    if (env.rank == ROOT_RANK) {
        //printf("Me (0,0) with rank %d\n", env.rank);

        for (auto i = 0; i < 3; ++i) {
            for (auto j = 0; j < 3; ++j) {
                int rank = env.nbrs[i][j];
                if (rank == NO_NBR || rank == ROOT_RANK) continue;
               // printf("Nbr (%d,%d) with rank %d\n", x - 1 + i, y - 1 + j, rank);
            }
        }
    }

    return env;
}

void quit_env(void) {
    MPI_Finalize();
}

