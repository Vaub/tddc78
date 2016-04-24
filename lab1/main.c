#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

#include "blur_filter.h"
#include "mpi_env.h"
#include "types.h"
#include "filters/gaussw.h"
#include "filters/ppmio.h"

void exit_program(int code) {
    close_mpi();
    exit(code);
}

int main(int argc, char** argv) {
    init_mpi(argc, argv);

    MpiEnv env = *get_env();
    printf("Process %d out of %d loaded\n", env.rank, env.nb_cpu);

    Pixel* work_buffer = env.rank == ROOT_RANK ?
                         malloc(MAX_RADIUS * sizeof(*work_buffer)) : NULL;

    Image image;
    Filter filter;
    filter.weights = malloc(MAX_RADIUS * sizeof(*filter.weights));

    if (env.rank == ROOT_RANK) {
        if (argc != 4) {
            fprintf(stderr, "Blur: radius file_in file_out\n");
            exit_program(1);
        }

        filter.radius = atoi(argv[1]);
        if (filter.radius < 1) {
            fprintf(stderr, "Radius invalid [%d]\n", filter.radius);
            exit_program(1);
        }

        int color_depth;
        if(read_ppm (argv[2], &image.width, &image.height, &color_depth, (char *)work_buffer) != 0) {
            exit_program(1);
        }

        if (color_depth > 255) {
            fprintf(stderr, "Color depth invalid [%d]\n", color_depth);
            exit_program(1);
        }

        get_gauss_weights(filter.radius, filter.weights);
    }

    MPI_Bcast(&image, 1, env.types.image, ROOT_RANK, env.comm);
    MPI_Bcast(&filter.radius, 1, MPI_INT, ROOT_RANK, env.comm);
    MPI_Bcast(filter.weights, MAX_RADIUS, MPI_DOUBLE, ROOT_RANK, env.comm);

    int counts[env.nb_cpu];
    int recv_counts[env.nb_cpu];
    int displs[env.nb_cpu];
    int recv_displs[env.nb_cpu];

    int local_size = get_img_size(&image) / env.nb_cpu;
    int reminder = get_img_size(&image) % env.nb_cpu;
    int local_buffer_max = (filter.radius * 2) + (local_size + reminder);

    if (env.rank == ROOT_RANK) {
        int img_idx = 0;
        int recv_idx = 0;

        ImageChunk chunk;
        for (int i = 0; i < env.nb_cpu; ++i) {
            int send_size = reminder > 0 ? local_size + 1 : local_size;
            --reminder;

            int send_buffer_size = send_size;
            chunk.img_idx = img_idx;
            chunk.nb_pix_to_treat = send_size;

            int x_start = (img_idx % image.width);
            int x_end = ((img_idx + send_size - 1) % image.width);

            int start_radius = filter.radius > x_start ? x_start : filter.radius;
            int end_radius = filter.radius + x_end > image.width - 1 ? (image.width - 1) - x_end : filter.radius;

            chunk.img_idx -= start_radius;
            chunk.start_offset = img_idx - chunk.img_idx;
            send_buffer_size += start_radius + end_radius;

            counts[i] = send_buffer_size;
            recv_counts[i] = chunk.nb_pix_to_treat;
            displs[i] = chunk.img_idx;
            recv_displs[i] = recv_idx;

            printf("CPU %d will write %d pixels from %d\n", i, chunk.nb_pix_to_treat, recv_idx);
            recv_idx += chunk.nb_pix_to_treat;



            img_idx += send_size;
            MPI_Send(&chunk, 1, env.types.chunk, i, 0, env.comm);
        }
    }

    MPI_Barrier(env.comm);

    ImageChunk local_chunk;
    MPI_Recv(&local_chunk, 1, env.types.chunk, ROOT_RANK, 0, env.comm, MPI_STATUS_IGNORE);

    Pixel local_buffer[local_buffer_max];
    MPI_Scatterv(work_buffer, counts, displs, env.types.pixel,
                 local_buffer, local_buffer_max, env.types.pixel,
                 ROOT_RANK, env.comm);

    Pixel local_output[local_chunk.nb_pix_to_treat];
    do_x_pass(local_buffer, &local_chunk, &image, &filter, local_output);

    Pixel src[get_img_size(&image)];
    MPI_Gatherv(local_output, local_chunk.nb_pix_to_treat, env.types.pixel,
                src, recv_counts, recv_displs, env.types.pixel,
                ROOT_RANK, env.comm);

    MPI_Barrier(env.comm);

    Pixel* y_axis = malloc(get_img_size(&image) * sizeof(*y_axis));
    for (int i = 0; i < get_img_size(&image); ++i) {
        int x = i % image.width;
        int y = i / image.width;
        y_axis[image.height * x + y] = src[i];
    }

    if (env.rank == ROOT_RANK) {
        int img_idx = 0;
        int recv_idx = 0;

        ImageChunk chunk;
        for (int i = 0; i < env.nb_cpu; ++i) {
            int send_size = reminder > 0 ? local_size + 1 : local_size;
            --reminder;

            int send_buffer_size = send_size;
            chunk.img_idx = img_idx;
            chunk.nb_pix_to_treat = send_size;

            int y_start = (img_idx % image.height);
            int y_end = ((img_idx + send_size - 1) % image.height);

            int start_radius = filter.radius > y_start ? y_start : filter.radius;
            int end_radius = filter.radius + y_end > image.height - 1 ? (image.height - 1) - y_end : filter.radius;

            chunk.img_idx -= start_radius;
            chunk.start_offset = img_idx - chunk.img_idx;
            send_buffer_size += start_radius + end_radius;

            counts[i] = send_buffer_size;
            recv_counts[i] = chunk.nb_pix_to_treat;
            displs[i] = chunk.img_idx;
            recv_displs[i] = recv_idx;

            printf("CPU %d will write %d pixels from %d\n", i, chunk.nb_pix_to_treat, recv_idx);
            recv_idx += chunk.nb_pix_to_treat;



            img_idx += send_size;
            MPI_Send(&chunk, 1, env.types.chunk, i, 0, env.comm);
        }
    }

    MPI_Barrier(env.comm);
    MPI_Recv(&local_chunk, 1, env.types.chunk, ROOT_RANK, 0, env.comm, MPI_STATUS_IGNORE);

    MPI_Scatterv(y_axis, counts, displs, env.types.pixel,
                 local_buffer, local_buffer_max, env.types.pixel,
                 ROOT_RANK, env.comm);

    do_y_pass(local_buffer, &local_chunk, &image, &filter, local_output);

    MPI_Gatherv(local_output, local_chunk.nb_pix_to_treat, env.types.pixel,
                src, recv_counts, recv_displs, env.types.pixel,
                ROOT_RANK, env.comm);

    MPI_Barrier(env.comm);

    for (int i = 0; i < get_img_size(&image); ++i) {
        int x = i / image.height;
        int y = i % image.height;
        y_axis[image.width * y + x] = src[i];
    }

    printf("Hmm...\n");
    if (env.rank == ROOT_RANK) {
        printf("Writing output\n");
        if(write_ppm (argv[3], image.width, image.height, (char *)y_axis) != 0) {
            exit_program(1);
        }
    }

    MPI_Barrier(env.comm);

    close_mpi();
    return 0;
}
