#include <pthread.h>
#include <stdlib.h>
#include <printf.h>
#include <stdio.h>
#include <time.h>

#include "image.h"
#include "pthread_barrier.h"
#include "filters/ppmio.h"

#define BILLION  1000000000L

static Barrier sum_barrier;
static pthread_mutex_t sum_mut = PTHREAD_MUTEX_INITIALIZER;
static unsigned int global_sum;

typedef struct ThresholdThreadData {
    int from, buffer_size, image_size;
    Pixel* buffer;
} ThreadData;

void increment_global_sum(unsigned int value) {
    pthread_mutex_lock(&sum_mut);
    global_sum += value;
    pthread_mutex_unlock(&sum_mut);
}

void* calculate_threshold(void* args) {
    ThreadData* data = args;

    int from = data->from;
    int size = data->buffer_size;
    int image_size = data->image_size;
    Pixel* buffer = data->buffer;

    unsigned int local_sum = 0;
    for (int i = from; i < from + size; ++i) {
        local_sum += (unsigned int)buffer[i].r + (unsigned int)buffer[i].g + (unsigned int)buffer[i].b;
    }

    increment_global_sum(local_sum);
    wait_for_process(&sum_barrier);

    //printf("Process gone from barrier, %d\n", from);
    unsigned int average = global_sum / image_size;

    for (int i = from; i < from + size; ++i) {
        unsigned int current_pixel = (unsigned int)buffer[i].r + (unsigned int)buffer[i].g + (unsigned int)buffer[i].b;
        buffer[i].r = buffer[i].g = buffer[i].b = (unsigned char)(average > current_pixel ? 0 : 255);
    }

    pthread_exit(NULL);
}

int main(int argc, char** argv) {

    Pixel* buffer = malloc(MAX_PIXELS * sizeof(*buffer));
    int width, height, size;
    int nb_threads;
    
    struct timespec start, stop;
    double exec_time = 0;

    if (argc != 4) {
        fprintf(stderr, "Usage: %s nb_threads infile outfile\n", argv[0]);
        exit(1);
    }

    nb_threads = atoi(argv[1]);
    if (nb_threads < 1) {
        fprintf(stderr, "Needs at least 1 thread\n");
        exit(1);
    }

    if (open_image(argv[2], &width, &height, buffer) != 0) {
        exit(1);
    }
    size = width * height;

    clock_gettime(CLOCK_REALTIME, &start);
    
    init_barrier(&sum_barrier, nb_threads);

    int avg_chunk_size = size / nb_threads;
    int reminder = size % nb_threads;

    pthread_t threads[nb_threads];
    ThreadData data[nb_threads];

    int current_offset = 0;
    for (int i = 0; i < nb_threads; ++i) {
        int chunk_size = avg_chunk_size
                         + (reminder > 0 ? 1 : 0);
        reminder--;

        ThreadData cpu_data = { current_offset, chunk_size, size, buffer };
        data[i] = cpu_data;

        pthread_create(&threads[i], NULL, calculate_threshold, (void*)&data[i]);
        current_offset += chunk_size;
    }

    for (int i = 0; i < nb_threads; ++i) {
        pthread_join(threads[i], NULL);
    }
    
    clock_gettime(CLOCK_REALTIME, &stop);
    
    //Get execution time in seconds
    exec_time = (stop.tv_sec - start.tv_sec) + 
		(double)((stop.tv_nsec - start.tv_nsec) / (double)BILLION); //Add nano seconds
		  
    
    printf("%d,%1f,%d\n",
               nb_threads, exec_time, size);

    //printf("Writing output file\n");
    if(write_ppm (argv[3], width, height, (char *)buffer) != 0) {
        exit(1);
    }

    destroy_barrier(&sum_barrier);
    return 0;
}
