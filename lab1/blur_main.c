#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

#include "blur_filter.h"
#include "mpi_env.h"
#include "filters/gaussw.h"
#include "filters/ppmio.h"

#define min(a,b)               \
   ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
      _a <= _b ? _a : _b; })

void exit_program(int code) {
    close_mpi();
    exit(code);
}

/**
 * Flips an image from x <-> y in memory
 * For example, for w = 3, h = 4 and layed out as row:
 *      flip_axis(w, h, 000 100 110 101, out) =>
 *          out = 0111 0010 0001
 */
void flip_axis(const int row_size, const int column_size, const Pixel* buffer, Pixel* out) {
    const int buffer_size = row_size * column_size;
    for (int i = 0; i < buffer_size; ++i) {
        int x = i % row_size, y = i / row_size;
        out[column_size * x + y] = buffer[i];
    }
}

/**
 * Reads an image and blur parameters from params in argc and argv
 */
void open_image(int argc, char** argv, Image* image, Filter* filter, Pixel* buffer) {
    if (argc != 4) {
        fprintf(stderr, "Blur: radius file_in file_out\n");
        exit_program(1);
    }

    filter->radius = atoi(argv[1]);
    if (filter->radius < 1) {
        fprintf(stderr, "Radius invalid [%d]\n", filter->radius);
        exit_program(1);
    }

    int color_depth;
    if(read_ppm (argv[2], &image->width, &image->height, &color_depth, (char *)buffer) != 0) {
        exit_program(1);
    }

    if (color_depth > 255) {
        fprintf(stderr, "Color depth invalid [%d]\n", color_depth);
        exit_program(1);
    }
}

/**
 * Take an image and apply a gaussian blur on a particular axis using all available MPI process
 * If you want to do a blur pass:
 *      x coord: buffer is layed out using the rows and row_size = width
 *      y coord: buffer is layed out using the columns and row_size = height
 *
 *  A full gaussian filter will require an x and y pass to work
 */
void distribute_image(Pixel* buffer, const int buffer_size,
                      const int row_size, const Filter* filter,
                      Pixel* output) {
    MpiEnv env = *get_env();

    ImageChunk chunk;
    int local_chunk_size;

    // MPI Scatter/Gather counts & displacements
    int to_send_chunk_sizes[env.nb_cpu];
    int to_send_offsets[env.nb_cpu];
    int to_receive_chunk_sizes[env.nb_cpu];
    int to_receive_offsets[env.nb_cpu];

    const int avg_chunk_size = buffer_size / env.nb_cpu;

    if (env.rank == ROOT_RANK) {
        int chunk_reminder = buffer_size % env.nb_cpu;

        int current_idx = 0; // Current image buffer idx
        int current_to_recv_offset = 0; // For Gatherv count
        for (int cpu = 0; cpu < env.nb_cpu; ++cpu) {
            // Buffer size that will be blurred by the process [cpu]
            int nb_pix_to_treat = avg_chunk_size
                                  + (chunk_reminder > 0 ? 1 : 0);
            chunk_reminder -= 1; // Since size % nb_cpu < nb_cpu, we can add one pixel to reminder

            // Where are we in x or y (depends of the buffer orientation)
            int row_pos_start = current_idx % row_size;
            int row_pos_end = (current_idx + (row_size - 1)) % row_size;

            // Number of pixels before and after our buffer that will be blurred to cover the radius
            int pixels_before = min(filter->radius, row_pos_start),
                pixels_after  = min(filter->radius, (row_size - 1) - row_pos_end);

            chunk.img_idx = current_idx - pixels_before;
            chunk.start_offset = pixels_before;
            chunk.nb_pix_to_treat = nb_pix_to_treat;

            // Computing for Scatterv/Gatherv
            to_send_chunk_sizes[cpu] = nb_pix_to_treat + pixels_before + pixels_after;
            to_send_offsets[cpu] = chunk.img_idx;
            to_receive_chunk_sizes[cpu] = chunk.nb_pix_to_treat;
            to_receive_offsets[cpu] = current_to_recv_offset;

            current_to_recv_offset += chunk.nb_pix_to_treat;
            current_idx += chunk.nb_pix_to_treat;

            MPI_Request req_chunk, req_size;
            MPI_Isend(&chunk, 1, env.types.chunk, cpu, 0, env.comm, &req_chunk);
            MPI_Isend(&to_send_chunk_sizes[cpu], 1, MPI_INT, cpu, 1, env.comm, &req_size);
        }
    }

    MPI_Recv(&chunk, 1, env.types.chunk, 0, 0, env.comm, MPI_STATUS_IGNORE);
    MPI_Recv(&local_chunk_size, 1, MPI_INT, 0, 1, env.comm, MPI_STATUS_IGNORE);

    Pixel* local_chunk_buffer = malloc(local_chunk_size * sizeof(*local_chunk_buffer));
    MPI_Scatterv(buffer, to_send_chunk_sizes, to_send_offsets, env.types.pixel,
                 local_chunk_buffer, local_chunk_size, env.types.pixel,
                 ROOT_RANK, env.comm);

    // Parallel blur
    Pixel* pass_output = malloc(chunk.nb_pix_to_treat * sizeof(*pass_output));
    do_blur_pass(local_chunk_buffer, &chunk, filter, row_size, pass_output);

    MPI_Gatherv(pass_output, chunk.nb_pix_to_treat, env.types.pixel,
                output, to_receive_chunk_sizes, to_receive_offsets, env.types.pixel,
                ROOT_RANK, env.comm);
}

int main(int argc, char** argv) {
    init_mpi(argc, argv);

    MpiEnv env = *get_env();
    Pixel* work_buffer = NULL;

    Image image;
    Filter filter;
    filter.weights = malloc(MAX_RADIUS * sizeof(*filter.weights));

    double t_start, t_end;

    // Initialisation and image reading
    if (env.rank == ROOT_RANK) {
        work_buffer = malloc(MAX_PIXELS * sizeof(*work_buffer));
        open_image(argc, argv, &image, &filter, work_buffer);

        get_gauss_weights(filter.radius, filter.weights);
        t_start = MPI_Wtime();
    }

    // Broadcasting of image and filter metadata
    MPI_Bcast(&image, 1, env.types.image, ROOT_RANK, env.comm);
    MPI_Bcast(&filter.radius, 1, MPI_INT, ROOT_RANK, env.comm);
    MPI_Bcast(filter.weights, MAX_RADIUS, MPI_DOUBLE, ROOT_RANK, env.comm);

    int image_size = get_img_size(&image);

    // Allocating memory
    Pixel* x_pass_buffer = NULL;
    Pixel* y_flip_output = NULL;
    Pixel* y_pass_buffer = NULL;
    Pixel* x_flip_output = NULL;
    if (env.rank == ROOT_RANK) {
        x_pass_buffer = malloc(image_size * sizeof(*x_pass_buffer));
        y_flip_output = malloc(image_size * sizeof(*y_flip_output));
        y_pass_buffer = malloc(image_size * sizeof(*y_pass_buffer));
        x_flip_output = malloc(image_size * sizeof(*x_flip_output));
    }

    // Blur pass for X
    distribute_image(work_buffer, image_size, image.width, &filter, x_pass_buffer);
    if (env.rank == ROOT_RANK) {
        flip_axis(image.width, image.height, x_pass_buffer, y_flip_output);
    }

    // Blur pass for Y
    distribute_image(y_flip_output, image_size, image.height, &filter, y_pass_buffer);
    if (env.rank == ROOT_RANK) {
        flip_axis(image.height, image.width, y_pass_buffer, x_flip_output);
    }

    // Writing program output (filtered image)
    if (env.rank == ROOT_RANK) {
        t_end = MPI_Wtime();

        printf("%d,%1.2f,%d,%d\n",
               env.nb_cpu, t_end - t_start, filter.radius, image_size);
        if(write_ppm (argv[3], image.width, image.height, (char *)x_flip_output) != 0) {
            exit_program(1);
        }
    }

    MPI_Barrier(env.comm);

    close_mpi();
    return 0;
}
