#include <stdio.h>

#include <blurfilter.h>
#include "types.h"
#include "mpi_env.h"

#ifndef LAB1_SHARED_COM_H
#define LAB1_SHARED_COM_H

#define PIXEL_REQ_FLAG 0
#define PIXEL_SENT_FLAG 1
#define END_TREATMENT 3

#define TREATMENT_FINISHED -1

void exec_finished(const pixel* local_buffer, const int local_size);

void process_messages(const pixel* local_buffer, const int local_size);

pixel get_pixel(int index, const pixel* local_buffer, const int local_size);

#endif //LAB1_SHARED_COM_H
