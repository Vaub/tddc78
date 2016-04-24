#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

#include "blur_filter.h"
#include "mpi_env.h"
#include "types.h"
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

void distribute_image(const Pixel* buffer, const int buffer_size,
                      const int row_size, const Filter* filter,
                      Pixel* output) {
    MpiEnv env = *get_env();

    ImageChunk chunk;
    int local_chunk_size;

    int* to_send_chunk_size = malloc(env.nb_cpu * sizeof(*to_send_chunk_size));
    int* to_send_offset = malloc(env.nb_cpu * sizeof(*to_send_offset));

    int* to_receive_chunk_size = malloc(env.nb_cpu * sizeof(*to_receive_chunk_size));
    int* to_receive_offset = malloc(env.nb_cpu * sizeof(*to_receive_offset));

    const int chunk_size = buffer_size / env.nb_cpu;

    if (env.rank == ROOT_RANK) {
        int chunk_reminder = buffer_size % env.nb_cpu;

        int current_idx = 0;
        int current_to_recv_offset = 0;
        for (int cpu = 0; cpu < env.nb_cpu; ++cpu) {
            int treated_chunk_size = chunk_size
                                     + (chunk_reminder > 0 ? 1 : 0);
            chunk_reminder -= 1;

            int row_pos_start = current_idx % row_size;
            int row_pos_end = (current_idx + (row_size - 1)) % row_size;

            int pixels_before = min(filter->radius, row_pos_start),
                pixels_after  = min(filter->radius, (row_size - 1) - row_pos_end);

            chunk.img_idx = current_idx - pixels_before;
            chunk.start_offset = pixels_before;
            chunk.nb_pix_to_treat = treated_chunk_size;

            to_send_chunk_size[cpu] = treated_chunk_size + pixels_before + pixels_after;
            to_send_offset[cpu] = chunk.img_idx;

            to_receive_chunk_size[cpu] = chunk.nb_pix_to_treat;
            to_receive_offset[cpu] = current_to_recv_offset;
            current_to_recv_offset += chunk.nb_pix_to_treat;

            current_idx += chunk.nb_pix_to_treat;

            MPI_Send(&chunk, 1, env.types.chunk, cpu, 0, env.comm);
            MPI_Send(&to_send_chunk_size[cpu], 1, MPI_INT, cpu, 1, env.comm);
        }
    }

    MPI_Recv(&chunk, 1, env.types.chunk, 0, 0, env.comm, MPI_STATUS_IGNORE);
    MPI_Recv(&local_chunk_size, 1, MPI_INT, 0, 1, env.comm, MPI_STATUS_IGNORE);

    Pixel* local_chunk_buffer = malloc(local_chunk_size * sizeof(*local_chunk_buffer));

    //printf("Addr: %p, %p, %p, %p [%d]\n", buffer, local_chunk_buffer, to_send_chunk_size, to_receive_offset, env.rank);
    MPI_Scatterv(buffer, to_send_chunk_size, to_send_offset, env.types.pixel,
                 local_chunk_buffer, local_chunk_size, env.types.pixel,
                 ROOT_RANK, env.comm);

    Pixel* pass_output = malloc(chunk.nb_pix_to_treat * sizeof(*pass_output));
    do_blur_pass(local_chunk_buffer, &chunk, filter, row_size, pass_output);

    MPI_Gatherv(pass_output, chunk.nb_pix_to_treat, env.types.pixel,
                output, to_receive_chunk_size, to_receive_offset, env.types.pixel,
                ROOT_RANK, env.comm);
    MPI_Barrier(env.comm);
}

int main(int argc, char** argv) {
    init_mpi(argc, argv);

    MpiEnv env = *get_env();
    //printf("Process %d out of %d loaded\n", env.rank, env.nb_cpu);

    Pixel* work_buffer = env.rank == ROOT_RANK ?
                         malloc(MAX_RADIUS * sizeof(*work_buffer)) : NULL;

    Image image;
    Filter filter;
    filter.weights = malloc(MAX_RADIUS * sizeof(*filter.weights));

    double t_start, t_end;

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
        t_start = MPI_Wtime();
    }

    MPI_Bcast(&image, 1, env.types.image, ROOT_RANK, env.comm);
    MPI_Bcast(&filter.radius, 1, MPI_INT, ROOT_RANK, env.comm);
    MPI_Bcast(filter.weights, MAX_RADIUS, MPI_DOUBLE, ROOT_RANK, env.comm);

    int image_size = get_img_size(&image);

    // Old working
    /* int counts[env.nb_cpu];
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

            //printf("CPU %d will write %d pixels from %d\n", i, chunk.nb_pix_to_treat, recv_idx);
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

    MPI_Barrier(env.comm);*/

    Pixel* output = env.rank == ROOT_RANK ?
                    malloc(image_size * sizeof(Pixel)) : NULL;
    distribute_image(work_buffer, image_size, image.width, &filter, output);

    if (env.rank == ROOT_RANK) {
        work_buffer = malloc(image_size * sizeof(*work_buffer));
        for (int i = 0; i < image_size; ++i) {
            int x = i % image.width;
            int y = i / image.width;
            work_buffer[image.height * x + y] = output[i];
        }
    }

    output = env.rank == ROOT_RANK ?
             malloc(image_size * sizeof(Pixel)) : NULL;
    distribute_image(work_buffer, image_size, image.height, &filter, output);

    if (env.rank == ROOT_RANK) {
        work_buffer = malloc(image_size * sizeof(*work_buffer));
        for (int i = 0; i < image_size; ++i) {
            int x = i / image.height;
            int y = i % image.height;
            work_buffer[image.width * y + x] = output[i];
        }
    }

    // Old working
    /*if (env.rank == ROOT_RANK) {
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

            //printf("CPU %d will write %d pixels from %d\n", i, chunk.nb_pix_to_treat, recv_idx);
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
                ROOT_RANK, env.comm);*/

    if (env.rank == ROOT_RANK) { t_end = MPI_Wtime(); }
    MPI_Barrier(env.comm);



    //printf("Hmm...\n");
    if (env.rank == ROOT_RANK) {
        //printf("RESULTS: %d pixel in %1.2f seconds with %d cpu [radius = %d].\n",
        //       get_img_size(&image), (t_end-t_start), env.nb_cpu, filter.radius);

        //printf("Writing output\n");
        printf("%d,%1.2f,%d,%d\n",
               env.nb_cpu, t_end - t_start, filter.radius, get_img_size(&image));
        if(write_ppm (argv[3], image.width, image.height, (char *)work_buffer) != 0) {
            exit_program(1);
        }
    }

    MPI_Barrier(env.comm);

    close_mpi();
    return 0;
}
