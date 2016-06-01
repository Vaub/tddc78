#include <pthread.h>
#include <stdlib.h>
#include <printf.h>
#include <stdio.h>
#include <time.h>

#include "gaussw.h"
#include "ppmio.h"
#include "image.h"
#include "blur_filter.h"

#define BILLION  1000000000L

typedef struct thread_data {
    int offset, nb_pix_to_treat, row_length;
    const Filter *filter;
    const Pixel *buffer;
    Pixel *output;
} ThreadData;

void read_image(int argc, char **argv, Image *image, int* radius, Pixel *buffer){

    if (argc != 5) {
        fprintf(stderr, "Blur: cpu radius file_in file_out\n");
        exit(1);
    }

    *radius = atoi(argv[2]);
    if (*radius < 1) {
        fprintf(stderr, "Radius invalid [%d]\n", *radius);
        exit(1);
    }

    if (open_image(argv[3], &image->width, &image->height, buffer) != 0) {
        exit(1);
    }
}

/**
 * Flips an image from x <-> y in memory
 * For example, for w = 3, h = 4 and layed out as row:
 *      flip_axis(w, h, 000 100 110 101, out) =>
 *          out = 0111 0010 0001
 */
void flip_axis(const int row_size, const int column_size, const Pixel *buffer, Pixel *out) {
    const int buffer_size = row_size * column_size;
    for (int i = 0; i < buffer_size; ++i) {
        int x = i % row_size, y = i / row_size;
        out[column_size * x + y] = buffer[i];
    }
}

void* do_blur_pass(void* args){
    ThreadData* data = args;

    const Pixel *buffer = data->buffer;
    const Filter *filter = data->filter;

    Pixel *output = data->output;
    int offset = data->offset;
    int nb_pix_to_treat = data->nb_pix_to_treat;
    int row_length = data->row_length;

    double r,g,b,n,w;
    int last_offset = offset + nb_pix_to_treat;

    for (int i = offset; i < last_offset; ++i) {

	 // Find out position in the memory "rows" (x or y depending on the pass)
        int pos = i % row_length;
        int pos2;

        w = filter->weights[0];
        r = w * buffer[i].r;
        g = w * buffer[i].g;
        b = w * buffer[i].b;
        n = w;

        // For this "row", apply the filter
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
        output[i] = pix;
    }
    
    pthread_exit(NULL);
}

void blur(const Filter *filter, const int nb_threads, const int image_size,
	   const int row_length, Pixel *work_buffer, Pixel *output){
	
    pthread_t threads[nb_threads];
    ThreadData data[nb_threads];
    
    const int avg_chunk_size = image_size / nb_threads;
    
    int chunk_reminder = image_size % nb_threads;
    int offset = 0;
    
    for (int i = 0; i < nb_threads; ++i) {
      
      // number of pixels that will be blurred
      int nb_pix_to_treat = avg_chunk_size + (chunk_reminder > 0 ? 1 : 0);
      
      chunk_reminder--;

      // data which is passed to the thread
      ThreadData data_to_process = { offset, nb_pix_to_treat, row_length, 
					  filter, work_buffer, output };
      data[i] = data_to_process;
      
      pthread_create(&threads[i], NULL, do_blur_pass, (void*)&data[i]);

      offset += nb_pix_to_treat;
    }
    
    // wait until all threads have finished to do the blur pass
    for (int i = 0; i < nb_threads; ++i) {
      pthread_join(threads[i], NULL);
    }
}

void blur_image(const Filter *filter, const int nb_threads, const Image *image, 
	  Pixel *work_buffer){
    
    const int image_size = image->width * image->height;
    Pixel* output = malloc(image_size * sizeof(*output));

    // Blur pass for X
    blur(filter, nb_threads, image_size, image->width, work_buffer, output);
    
    flip_axis(image->width, image->height, output, work_buffer);

    // Blur pass for Y
    blur(filter, nb_threads, image_size, image->height, work_buffer, output);

    flip_axis(image->height, image->width, output, work_buffer);
}

int main(int argc, char** argv) {
	
    Pixel* work_buffer = malloc(MAX_PIXELS * sizeof(*work_buffer));

    Image image;
    Filter filter;
    int nb_threads;

    struct timespec start, stop;
    double exec_time = 0;
    
    filter.weights = malloc(MAX_RADIUS * sizeof(*filter.weights));

    read_image(argc, argv, &image, &filter.radius, work_buffer);
    get_gauss_weights(filter.radius, filter.weights);

    nb_threads = atoi(argv[1]);
    if (nb_threads < 1) {
        fprintf(stderr, "Needs at least 1 thread\n");
        exit(1);
    }

    int size = image.width * image.height;
    
    clock_gettime(CLOCK_REALTIME, &start);
    
    blur_image(&filter, nb_threads, &image, work_buffer);
    
    clock_gettime(CLOCK_REALTIME, &stop);
    
    //Get execution time in seconds
    exec_time = (stop.tv_sec - start.tv_sec) + 
		(double)((stop.tv_nsec - start.tv_nsec) / (double)BILLION); //Add nano seconds
    
    printf("%d,%1f,%d,%d\n",
               nb_threads, exec_time, filter.radius, size);

    if(write_ppm (argv[4], image.width, image.height, (char *)work_buffer) != 0) {
        exit(1);
    }

    return 0;
}
