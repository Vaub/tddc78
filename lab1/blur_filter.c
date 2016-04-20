#include <stdlib.h>
#include "blur_filter.h"

void find_data_location(const pixel* buffer,
                        const int local_size,
                        const int rank, const int nb_cpu,
                        ProcessorData* required_idx) {
    required_idx = malloc(nb_cpu * local_size * sizeof(ProcessorData));


}
