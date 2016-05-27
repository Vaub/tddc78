#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

#include "threshold_filter.h"
#include "mpi_env.h"
#include "filters/ppmio.h"
#include "blur_filter.h"

void exit_program(int code) {
    close_mpi();
    exit(code);
}

/**
 * Reads an image and parameters from params in argc and argv
 */
void open_image(int argc, char **argv, Image *image, Pixel *buffer) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s infile outfile\n", argv[0]);
        exit_program(1);
    }

    int color_depth;
    if (read_ppm(argv[1], &image->width, &image->height, &color_depth, (char *) buffer) != 0) {
        exit_program(1);
    }

    if (color_depth > 255) {
        fprintf(stderr, "Color depth invalid [%d]\n", color_depth);
        exit_program(1);
    }
}

void send_chunk_sizes(const int image_size, int *chunk_sizes, int *offsets, int *local_chunk_size) {

    MpiEnv env = *get_env();
    const int avg_chunk_size = image_size / env.nb_cpu;

    if (env.rank == ROOT_RANK) {
        int chunk_reminder = image_size % env.nb_cpu;
        int offset = 0;

        for (int cpu = 0; cpu < env.nb_cpu; ++cpu) {

            // Buffer size that will be blurred by the process [cpu]
            int nb_pix_to_treat = avg_chunk_size
                                  + (chunk_reminder > 0 ? 1 : 0);

            chunk_reminder -= 1; // Since size % nb_cpu < nb_cpu, we can add one pixel to reminder

            chunk_sizes[cpu] = nb_pix_to_treat;
            offsets[cpu] = offset;

            offset += nb_pix_to_treat;

            MPI_Request request;
            MPI_Isend(&chunk_sizes[cpu], 1, MPI_INT, cpu, 1, env.comm, &request);
        }
    }

    MPI_Recv(local_chunk_size, 1, MPI_INT, 0, 1, env.comm, MPI_STATUS_IGNORE);
}

unsigned long do_thresholding(const int image_size, Pixel *work_buffer, Pixel *output) {
    MpiEnv env = *get_env();
    int local_chunk_size;

    unsigned long flop = 0;

    // MPI Scatter/Gather counts & displacements
    int chunk_sizes[env.nb_cpu];
    int offsets[env.nb_cpu];

    send_chunk_sizes(image_size, chunk_sizes, offsets, &local_chunk_size);
    Pixel *local_chunk_buffer = malloc(local_chunk_size * sizeof(*local_chunk_buffer));

    MPI_Scatterv(work_buffer, chunk_sizes, offsets, env.types.pixel,
                 local_chunk_buffer, local_chunk_size, env.types.pixel,
                 ROOT_RANK, env.comm);

    unsigned int global_sum = 0, local_sum = 0, threshold = 0;
    Pixel *pass_output = malloc(local_chunk_size * sizeof(*pass_output));

    // Sum r,g and b values locally
    for (int i = 0; i < local_chunk_size; ++i) {
        local_sum += (unsigned int) (local_chunk_buffer[i].r + local_chunk_buffer[i].g + local_chunk_buffer[i].b);
	flop += 3;
    }

    // Combine all local sums and send the result to all processes
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_UNSIGNED, MPI_SUM, env.comm);

    threshold = global_sum / image_size;
    flop += 1;

    thresfilter(threshold, local_chunk_size, local_chunk_buffer, pass_output);
    flop += 3 * local_chunk_size;

    MPI_Gatherv(pass_output, local_chunk_size, env.types.pixel,
                output, chunk_sizes, offsets, env.types.pixel,
                ROOT_RANK, env.comm);

    return flop;
}


int main(int argc, char **argv) {
    init_mpi(argc, argv);
    MpiEnv env = *get_env();

    Image image;
    Pixel *work_buffer = NULL;

    double t_start, t_end;

    if (env.rank == ROOT_RANK) {
        work_buffer = malloc(MAX_PIXELS * sizeof(*work_buffer));
        open_image(argc, argv, &image, work_buffer);

        t_start = MPI_Wtime();
    }

    // Broadcasting of image
    MPI_Bcast(&image, 1, env.types.image, ROOT_RANK, env.comm);

    int image_size = get_img_size(&image);
    Pixel *output = malloc(image_size * sizeof(*output));

    unsigned long flop = do_thresholding(image_size, work_buffer, output);

    unsigned long global_flop = 0;
    MPI_Reduce(&flop, &global_flop, 1, MPI_UNSIGNED_LONG, MPI_SUM, ROOT_RANK, env.comm); 

    // Writing program output (filtered image)
    if (env.rank == ROOT_RANK) {
        t_end = MPI_Wtime();

        printf("%d,%1.2f,%d\n",
               env.nb_cpu, t_end - t_start, image_size);
	printf("FLOP: %ld\n",global_flop);
        if (write_ppm(argv[2], image.width, image.height, (char *) output) != 0) {
            exit_program(1);
        }
    }

    MPI_Barrier(env.comm);

    close_mpi();
    return 0;
}
