#ifndef LAB2_BLUR_FILTER_H
#define LAB2_BLUR_FILTER_H

#include "image.h"

#define MAX_RADIUS 10000
#define MAX_PIXELS (10000*10000)

typedef struct Image {
    int width, height;
} Image;

typedef struct Filter {
    int radius;
    double* weights;
} Filter;

#endif 
