#ifndef LAB1_BLUR_FILTER_H
#define LAB1_BLUR_FILTER_H

#include "types.h"

#define MAX_RADIUS 10000
#define MAX_PIXELS (10000*10000)

typedef struct ImageChunk {
    int img_idx, start_offset, nb_pix_to_treat;
} ImageChunk;

typedef struct Image {
    int width, height;
} Image;

typedef struct Filter {
    int radius;
    double* weights;
} Filter;

int get_img_size(const Image* img);

/**
 * Blurs the pixels by row or columns depending on their layout in memory
 * for a given chunk of an image and a blur filter
 *
 * LAB ONLY : returns the number of flop done
 */
unsigned long do_blur_pass(const Pixel *buffer,
                 	   const ImageChunk *chunk, const Filter *filter,
                 	   const int row_length, Pixel *output);

#endif //LAB1_BLUR_FILTER_H
