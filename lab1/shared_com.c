#include <stdbool.h>
#include "shared_com.h"

void exec_finished(const pixel* local_buffer, const int local_size) {
    MPI_Request end_requests[MPI_ENV.nb_cpu];
    for (int i = 0; i < MPI_ENV.nb_cpu; ++i) {
        if (MPI_ENV.rank == i) { continue; }

        int pack = TREATMENT_FINISHED;
        MPI_Send(&pack, 1, MPI_INT, i, PIXEL_REQ_FLAG, MPI_COMM_WORLD);
        MPI_Send(&pack, 1, MPI_INT, i, END_TREATMENT, MPI_COMM_WORLD);

        int finish;
        MPI_Irecv(&finish, 1, MPI_INT, i, END_TREATMENT, MPI_COMM_WORLD, &end_requests[i]);
    }

    int is_done = 0;
    while (!is_done) {
        process_messages(local_buffer, local_size);

        int is_waiting = 0;
        for (int i = 0; i < MPI_ENV.nb_cpu && !is_waiting; ++i) {
            if (MPI_ENV.rank == i) { continue; }

            int flag;
            MPI_Test(&end_requests[i], &flag, MPI_STATUS_IGNORE);
            is_waiting = (flag == 0);
        }

        is_done = !is_waiting;
    }

}

void process_messages(const pixel* local_buffer, const int local_size) {
    int pixel_requested;

    MPI_Status status;
    MPI_Recv(&pixel_requested, 1, MPI_INT, MPI_ANY_SOURCE, PIXEL_REQ_FLAG, MPI_COMM_WORLD, &status);

    if (pixel_requested != TREATMENT_FINISHED) {
        int local_idx = pixel_requested % local_size;
        printf("Sending\t\t%d [%d -> %d]\n", pixel_requested, MPI_ENV.rank, status.MPI_SOURCE);
        MPI_Send(&local_buffer[local_idx], 1, MPI_PIXEL, status.MPI_SOURCE, PIXEL_SENT_FLAG, MPI_COMM_WORLD);
    }
}

pixel get_pixel(int index, const pixel* local_buffer, const int local_size) {
    int cpu_to_call = index / local_size;
    if (cpu_to_call == MPI_ENV.rank) {
        return local_buffer[index % local_size];
    }

    if (cpu_to_call >= MPI_ENV.nb_cpu) {
        printf("Wut? cpu %d, for idx %d with size %d\n", MPI_ENV.rank, index, local_size);
    }

    pixel pixel;
    MPI_Request request;

    printf("Requesting\t%d [%d -> %d]\n", index, MPI_ENV.rank, cpu_to_call);
    MPI_Send(&index, 1, MPI_INT, cpu_to_call, PIXEL_REQ_FLAG, MPI_COMM_WORLD);
    MPI_Irecv(&pixel, 1, MPI_PIXEL, cpu_to_call, PIXEL_SENT_FLAG, MPI_COMM_WORLD, &request);

    process_messages(local_buffer, local_size);
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    return pixel;
}
