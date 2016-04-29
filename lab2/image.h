#ifndef LAB2_IMAGE_H_H
#define LAB2_IMAGE_H_H

//#define MAX_PIXELS (10000*10000)

typedef struct Pixel {
    unsigned char r, g, b;
} Pixel;

/**
 * Returns 0 if all worked out
 */
int open_image(const char* filename, int* width, int* height, Pixel* image);

#endif //LAB2_IMAGE_H_H
