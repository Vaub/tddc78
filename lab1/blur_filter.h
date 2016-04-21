#ifndef LAB1_BLUR_FILTER_H
#define LAB1_BLUR_FILTER_H

#include <blurfilter.h>

#include "types.h"
#include "shared_com.h"

void parallel_blur(const Filter* filter,
                   const ImageProperties* prop,
                   const int size, const pixel* buffer, pixel* blurred);

void parallel_blur_v(const Filter* filter,
                     const ImageProperties* prop,
                     const int size, const pixel* buffer, pixel* blurred);

#endif //LAB1_BLUR_FILTER_H
