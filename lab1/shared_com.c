#include <stdbool.h>
#include <stdlib.h>
#include "shared_com.h"

static int recv_waiting = 0;
static int receive_buffer;
static MPI_Request receive_request;

void exec_finished(const pixel* local_buffer, const int local_size) {
    ProcessEnv env = MPI_ENV;

    int end_message = TREATMENT_FINISHED;
    MPI_Request end_requests[env.nb_cpu];
    for (int i = 0; i < env.nb_cpu; ++i) {
        MPI_Send(&end_message, 1, MPI_INT, i, PIXEL_REQ_FLAG, MPI_COMM_WORLD);
        MPI_Send(&end_message, 1, MPI_INT, i, END_TREATMENT, MPI_COMM_WORLD);
        MPI_Irecv(&end_message, 1, MPI_INT, i, END_TREATMENT, MPI_COMM_WORLD, &end_requests[i]);
    }

    int cpu_over = 0;
    do {
        process_messages(local_buffer, local_size);

        cpu_over = 0;
        int flag;
        for (int i = 0; i < env.nb_cpu; ++i) {
            MPI_Test(&end_requests[i], &flag, MPI_STATUS_IGNORE);
            cpu_over += (flag == 0) ? 0 : 1;
        }
    } while (cpu_over != env.nb_cpu);

}

void process_messages(const pixel* local_buffer, const int local_size) {
    int flag;
    MPI_Status status;

    if (recv_waiting) {
        MPI_Test(&receive_request, &flag, &status);
        recv_waiting = (flag == 0);

        if (!recv_waiting && receive_buffer != TREATMENT_FINISHED) {
            //printf("Receive req %d [%d -> %d] \n", receive_buffer, status.MPI_SOURCE, MPI_ENV.rank);
            int local_idx = receive_buffer % local_size;
            MPI_Send(&local_buffer[local_idx], 1, MPI_PIXEL, status.MPI_SOURCE, PIXEL_SENT_FLAG, MPI_COMM_WORLD);
        }

    } else {
        recv_waiting = 1;
        MPI_Irecv(&receive_buffer, 1, MPI_INT, MPI_ANY_SOURCE, PIXEL_REQ_FLAG, MPI_COMM_WORLD, &receive_request);
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

    //printf("Requesting\t%d [%d -> %d]\n", index, MPI_ENV.rank, cpu_to_call);
    MPI_Send(&index, 1, MPI_INT, cpu_to_call, PIXEL_REQ_FLAG, MPI_COMM_WORLD);
    MPI_Irecv(&pixel, 1, MPI_PIXEL, cpu_to_call, PIXEL_SENT_FLAG, MPI_COMM_WORLD, &request);

    int flag = 0;
    do {
        process_messages(local_buffer, local_size);
        MPI_Test(&request, &flag, MPI_STATUS_IGNORE);

    } while (flag != 1);

    //printf("Got pixel %d [%d -> %d]\n", index, cpu_to_call, MPI_ENV.rank);
    return pixel;
}
