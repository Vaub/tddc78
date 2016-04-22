#include <memory.h>
#include "blur_filter.h"

int find_offset(const int width, const int x, const int y) {
    if ((width * y) + x >= 515788) {
        printf("Wtf is this with x: %d, y: %d , width is %d\n", x, y, width);
    }

    return (width * y) + x;
}

void parallel_blur(const Filter filter, const ImageProperties prop,
                   const int buffer_size, const pixel* buffer, pixel* blurred) {
    int start_idx = MPI_ENV.rank * buffer_size;
    int width = prop.width;

    printf("Working from %d with %d pixels\n", start_idx, buffer_size);
    double r,g,b,n,w;
    int x,y;

    //pixel tmp[size];
    for (int i = 0; i < buffer_size; ++i) {
        int offset = start_idx + i;
        x = (offset % width);
        y = (offset / width);

        w = filter.weights[0];
        pixel pix = get_pixel(offset,buffer,buffer_size);

        r = w * pix.r;
        g = w * pix.g;
        b = w * pix.b;
        n = w;

        for (int wi = 1; wi < filter.radius; ++wi) {
            w = filter.weights[wi];

            if (x - wi >= 0) {
                pix = get_pixel(find_offset(width, x - wi, y), buffer, buffer_size);
                r += w * pix.r;
                g += w * pix.g;
                b += w * pix.b;
                n += w;
            }

            if (x + wi < width) {
                pix = get_pixel(find_offset(width, x + wi, y), buffer, buffer_size);
                r += w * pix.r;
                g += w * pix.g;
                b += w * pix.b;
                n += w;
            }

        }

        pixel blur_pix = { r / n, g / n, b / n };
        //printf(":(\n");
        blurred[find_offset(width, x, y)] = blur_pix;
    }

    //memcpy(tmp, blurred, buffer_size);
}

void parallel_blur_v(const Filter filter, const ImageProperties prop,
                     const int buffer_size, const pixel* buffer, pixel* blurred) {
    int start_idx = MPI_ENV.rank * buffer_size;

    printf("Working from %d with %d pixels\n", start_idx, buffer_size);

    double r,g,b,n,w;
    int x,y;

    int width = prop.width;
    printf("Check with offset %d and properties w: %d, h: %d\n", start_idx, prop.width, prop.height);

    //pixel tmp[size];
    for (int i = 0; i < buffer_size; ++i) {
        int offset = start_idx + i;
        x = (offset % width);
        y = (offset / width);

        w = filter.weights[0];
        pixel pix = get_pixel(offset,buffer,buffer_size);

        r = w * pix.r;
        g = w * pix.g;
        b = w * pix.b;
        n = w;

        for (int wi = 1; wi < filter.radius; ++wi) {
            w = filter.weights[wi];

            if (y - wi >= 0) {
                pix = get_pixel(find_offset(width, x, y - wi), buffer, buffer_size);
                r += w * pix.r;
                g += w * pix.g;
                b += w * pix.b;
                n += w;
            }

            if (y + wi < prop.height) {
                pix = get_pixel(find_offset(width, x, y + wi), buffer, buffer_size);
                r += w * pix.r;
                g += w * pix.g;
                b += w * pix.b;
                n += w;
            }

        }

        pixel blur_pix = { r / n, g / n, b / n };
        blurred[find_offset(prop.width, x, y)] = blur_pix;
    }

    //memcpy(tmp, blurred, buffer_size);
}
