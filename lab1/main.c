#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#include "filters/ppmio.h"
#include "filters/blurfilter.h"
#include "filters/gaussw.h"
#include "types.h"

struct timespec get_time() {
    struct timespec time;

#ifdef __MACH__
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    time.tv_sec = mts.tv_sec;
    time.tv_nsec = mts.tv_nsec;
#else
    clock_gettime(CLOCK_REALTIME, &time);
#endif

    return time;
}

void exit_program(int code) {
    MPI_Finalize();
    exit(code);
}

int read_img(const char* src, pixel* buffer, ImageProperties* properties) {
    buffer = malloc(MAX_PIXELS * sizeof(pixel));

    printf("Memory allocated, extracting\n");
    return read_ppm(src,
                    &properties->width, &properties->height,
                    &properties->color_depth,
                    (char*)buffer);
}

#define ROOT_RANK 0

int main(int argc, char** argv) {
    int nb_cpu, rank;

    MPI_Comm com = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(com, &nb_cpu);
    MPI_Comm_rank(com, &rank);

    MPI_Datatype pixel_mpi = register_pixel_type();
    MPI_Datatype img_prop_mpi = register_img_prop_type();
    MPI_Datatype filter_mpi = register_filter_type();

    pixel* work_buffer = NULL;

    Filter filter;
    ImageProperties img_prop;

    if (rank == ROOT_RANK) {
        if (argc != 4) {
            fprintf(stderr, "Usage %s radius infile outfile\n", argv[0]);
            exit_program(1);
        }

        printf("Starting extraction...\n");
        filter.radius = atoi(argv[1]);
        get_gauss_weights(filter.radius, filter.weights);

        printf("Reading image\n");

        work_buffer = malloc(MAX_PIXELS * sizeof(pixel));
        int read_img = read_ppm(argv[2], &img_prop.width, &img_prop.height,
                                &img_prop.color_depth, (char*)work_buffer);

        if (read_img != 0) {
            exit_program(1);
        }

        printf("Validating\n");
        if (img_prop.color_depth > 255) {
            exit_program(1);
        }
    }

    MPI_Bcast(&filter, 1, filter_mpi, ROOT_RANK, com);
    MPI_Bcast(&img_prop, 1, img_prop_mpi, ROOT_RANK, com);

    int nb_pixels = img_prop.width * img_prop.height;
    int local_size = nb_pixels / nb_cpu;
    pixel* local_buffer = malloc(local_size * sizeof(pixel));
    printf("Process %d processing %d pixels\n", rank, local_size);

    MPI_Scatter(work_buffer, local_size, pixel_mpi,
                local_buffer, local_size, pixel_mpi,
                ROOT_RANK, com);

    //int start_idx = rank * local_size;



    MPI_Finalize();
    return(0);
}