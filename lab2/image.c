#include "image.h"
#include "filters/ppmio.h"

#include <stdio.h>
//#include "filters/ppmio.h"

int open_image(const char* filename, int* width, int* height, Pixel* image) {

    int color_depth;
    if (read_ppm(filename, width, height, &color_depth, (char*)image) != 0) {
        return 1;
    }

    if (color_depth > 255) {
        fprintf(stderr, "Cannot have a color depth more than 255!\n");
        return 2;
    }

    return 0;
}
