#include "blur_filter.h"

int get_img_size(const Image* img) {
    return img->width * img->height;
}

void do_blur_pass(const Pixel *buffer,
                  const ImageChunk *chunk, const Filter *filter,
                  const int row_length, Pixel *output) {
    double r,g,b,n,w;

    int output_idx = 0;
    int last_offset = chunk->start_offset + chunk->nb_pix_to_treat;
    for (int i = chunk->start_offset; i < last_offset; ++i) {

        int pos = (chunk->img_idx + i) % row_length;
        int pos2;

        w = filter->weights[0];
        r = w * buffer[i].r;
        g = w * buffer[i].g;
        b = w * buffer[i].b;
        n = w;

        for (int wi = 1; wi < filter->radius; ++wi) {
            w = filter->weights[wi];

            pos2 = pos - wi;
            if (pos2 >= 0) {
                r += w * buffer[i - wi].r;
                g += w * buffer[i - wi].g;
                b += w * buffer[i - wi].b;
                n += w;
            }

            pos2 = pos + wi;
            if (pos2 < row_length) {
                r += w * buffer[i + wi].r;
                g += w * buffer[i + wi].g;
                b += w * buffer[i + wi].b;
                n += w;
            }

        }

        Pixel pix = { r / n, g / n, b /n };
        output[output_idx++] = pix;
    }
}