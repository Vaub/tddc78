#include <mpi.h>

#ifndef LAB1_TYPES_H
#define LAB1_TYPES_H

#define MAX_RAD 1000

typedef struct ImageProperties {
    int width;
    int height;
    int color_depth;
} ImageProperties;

typedef struct Filter {
    int radius;
    double weights[MAX_RAD];
} Filter;

MPI_Datatype register_pixel_type();
MPI_Datatype register_img_prop_type();
MPI_Datatype register_filter_type();

#endif //LAB1_TYPES_H
