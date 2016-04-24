#include <mpi.h>
#include <memory.h>
#include <assert.h>
#include <printf.h>

#include "shared_com.h"
#include "mpi_env.h"
#include "types.h"

#define REQUEST_PIXEL 10
#define REQUEST_FINISHED 20

typedef struct receive_request {
    MPI_Request request;
    int is_waiting;
} RecvReq;

void send_pixels(const Pixel* img_buffer, const int img_size) {
    MpiEnv env = *get_env();

    MPI_Request end_requests[env.nb_cpu];
    MPI_Request pixels_requests[env.nb_cpu];

    int index_requested[env.nb_cpu];

    int tmp;
    for (int i = 1; i < env.nb_cpu; ++i) {
        MPI_Irecv(&tmp, 1, MPI_INT, i, REQUEST_FINISHED, env.comm, &end_requests[i]);
    }

    int is_started = 0;
    int is_finished = 0;
    do {
        int recv_flag = 0;

        MPI_Request send;
        MPI_Status status;

        for (int i = 1; i < env.nb_cpu && is_started; ++i) {

            MPI_Test(&pixels_requests[i], &recv_flag, &status);

            if (recv_flag) {
                MPI_Isend(&img_buffer[index_requested[i]], 1, env.types.pixel, status.MPI_SOURCE, REQUEST_PIXEL,env.comm, &send);
                MPI_Request_free(&send);
            }
            MPI_Request_free(&pixels_requests[i]);
        }
        is_started = 1;

        for (int i = 1; i < env.nb_cpu; ++i) {
            MPI_Irecv(&index_requested[i], 1, MPI_INT, i, REQUEST_PIXEL, env.comm, &pixels_requests[i]);
        }

        MPI_Testall(env.nb_cpu - 1, &end_requests[1], &is_finished, MPI_STATUS_IGNORE);
    } while (is_finished);
}

Pixel get_pixel(const int index,
                const Pixel* local_chunk, const int chunk_size) {
    Pixel pixel;
    MpiEnv env = *get_env();

    if (env.rank == ROOT_RANK) {
        return local_chunk[index];
    }

    int local_start_idx = env.rank * chunk_size;
    int in_local =
            index >= local_start_idx &&
            index < local_start_idx + chunk_size;

    if (in_local) {
        return local_chunk[index - local_start_idx];
    }

    MPI_Send(&index, 1, MPI_INT, ROOT_RANK, REQUEST_PIXEL, env.comm);
    MPI_Recv(&pixel, 1, env.types.pixel, ROOT_RANK, REQUEST_PIXEL, env.comm, MPI_STATUS_IGNORE);

    return pixel;
}

void finished_treatment() {
    int tmp = 0;
    MPI_Send(&tmp, 1, MPI_INT, ROOT_RANK, REQUEST_FINISHED, get_env()->comm);

    printf("Finished for cpu %d\n", get_env()->rank);
}