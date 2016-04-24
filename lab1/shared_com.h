#ifndef LAB1_SHARED_COM_H
#define LAB1_SHARED_COM_H

#include "types.h"

void send_pixels(const Pixel* img_buffer, const int img_size);

Pixel get_pixel(int index,
                const Pixel* local_chunk, const int chunk_size);

void finished_treatment();

#endif //LAB1_SHARED_COM_H
