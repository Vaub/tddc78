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

void do_x_pass(const Pixel* buffer,
               const ImageChunk* chunk,
               const Image* image, const Filter* filter,
               Pixel* output_buffer);

void do_y_pass(const Pixel* buffer,
               const ImageChunk* chunk,
               const Image* image, const Filter* filter,
               Pixel* output_buffer);

void do_blur_pass(const Pixel *buffer,
                  const ImageChunk *chunk, const Filter *filter,
                  const int row_length, Pixel *output);

#endif //LAB1_BLUR_FILTER_H
