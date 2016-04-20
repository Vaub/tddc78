#ifndef LAB1_BLUR_FILTER_H
#define LAB1_BLUR_FILTER_H

#include <blurfilter.h>
#include "types.h"

typedef struct ProcessorData {
    int nb_pixels;
    int* adress;
} ProcessorData;

void find_data_location(const pixel* buffer,
                        const int local_size,
                        const int rank, const int nb_cpu,
                        ProcessorData* required_idx);

#endif //LAB1_BLUR_FILTER_H
