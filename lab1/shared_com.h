#include <pthread.h>

#include <blurfilter.h>
#include "types.h"

#ifndef LAB1_SHARED_COM_H
#define LAB1_SHARED_COM_H

#define PIXEL_REQ_FLAG 0

typedef struct ProcessEnv {
    int rank;
    int nb_cpu;
} ProcessEnv;

pixel get_pixel(int index, const pixel* local_buffer, const int local_size, const ProcessEnv env) {
    int cpu_to_call = index / local_size;
    pixel pixel;

    MPI_Send(&index, 1, MPI_INT, cpu_to_call, PIXEL_REQ_FLAG, MPI_COMM_WORLD);
}

#endif //LAB1_SHARED_COM_H
