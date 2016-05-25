#include <stdio.h>
#include <stdlib.h>
#include <list>
#include <vector>
#include <iterator>

#include "definitions.h"
#include "mpi_env.h"

#define NB_PARTICLES 100
#define NB_STEPS 1000

static mpi_env_t env;

double rand_range(void) {
    return (double) rand() / (double) RAND_MAX;
}

std::vector<particle_t> receive_from_neighbours(int flag, const mpi_env_t &env) {

    int count = 0;
    std::vector<particle_t> received(MAX_NO_PARTICLES);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            int nbr_rank = env.nbrs[i][j];
            if (nbr_rank == -1 || nbr_rank == env.rank) { continue; }

            int tmp_count;
            MPI_Status status;
            MPI_Recv(&received[count], MAX_NO_PARTICLES, env.types.particle, nbr_rank, flag,
                     env.grid_comm, &status);

            MPI_Get_count(&status, env.types.particle, &tmp_count);
            count += tmp_count;
        }
    }

    return std::vector<particle_t>(&received[0], &received[0] + count);
}

bool try_send_to_nbrs(const mpi_env_t& env, const cord_t &block,
                      const particle_t &particle, std::vector<particle_t> nbrs[3][3]) {
    const pcord_t &coords = particle.pcord;

    int shift_x = coords.x < block.x0 ? -1 :
                  coords.x > block.x1 ? +1 :
                  0;
    int shift_y = coords.y < block.y0 ? -1 :
                  coords.y > block.y1 ? +1 :
                  0;

    if (shift_x == 0 && shift_y == 0) { return false; }

    nbrs[1 + shift_x][1 + shift_y].push_back(particle);
    return true;
}

int main(int argc, char *argv[]) {

    env = init_env(&argc, &argv);

    float block_width = (float) ceil(BOX_HORIZ_SIZE / (float) env.grid_size[0]);
    float block_height = (float) ceil(BOX_VERT_SIZE / (float) env.grid_size[1]);

    cord_t block = {
            .x0 = block_width * env.coords[0],
            .y0 = block_height * env.coords[1],
            .x1 = (block_width * env.coords[0]) + block_width,
            .y1 = (block_height * env.coords[1]) + block_height
    };
    cord_t wall = {.x0 = 0, .y0 = 0, .x1 = BOX_HORIZ_SIZE, .y1 = BOX_VERT_SIZE};

    // Init particles with random positions and velocity
    std::list<particle_t> particles((size_t)(NB_PARTICLES / env.nb_cpu));
    for (auto& particle : particles) {
        double r = rand_range() * MAX_INITIAL_VELOCITY;
        double angle = rand_range() * (2*PI);
        pcord_t pcord = {
                .x = (block_width * (float)rand_range()) + block.x0,
                .y = (block_height * (float)rand_range()) + block.y0,
                .vx = (float)(r * cos(angle)), .vy = (float)(r * sin(angle))
        };

        particle.pcord = pcord;
    }

    double local_pressure = 0;
    double t_start = MPI_Wtime();
    for (auto step = 0; step < NB_STEPS; ++step) {

        // Particles sent to neighbours
        std::vector<particle_t> to_send_nbrs[3][3];

        for (auto current = particles.begin(); current != particles.end();) {
            auto& particle = *current;
            bool has_collided = false;

            // check collisions
            for (auto with = std::next(current); with != particles.end() && !has_collided;) {
                float collided_at = collide(&particle.pcord, &with->pcord);
                if (collided_at == -1) {
                    ++with;
                    continue;
                }

                interact(&particle.pcord, &with->pcord, collided_at);
                particles.erase(with);
                has_collided = true;
            }

            // move particles
            if (!has_collided) {
                feuler(&particle.pcord, 1);
            }

            local_pressure += wall_collide(&particle.pcord, wall);

            // check if particle still in region, else send to neighbour
            if (try_send_to_nbrs(env, block, particle, to_send_nbrs)) {
                current = particles.erase(current);
                continue;
            }

            current++;
        }

        int sent = 0;
        MPI_Request requests[3][3];
        for (auto i = 0; i < 3; ++i) {
            for (auto j = 0; j < 3; ++j) {
                int nbr_rank = env.nbrs[i][j];
                if (nbr_rank == env.rank || nbr_rank == -1) { continue; }
                MPI_Isend(&to_send_nbrs[i][j][0], (int)to_send_nbrs[i][j].size(), env.types.particle,
                          nbr_rank, FLAG_NEW_PARTICLES, env.grid_comm, &requests[i][j]);
                sent += to_send_nbrs[i][j].size();
            }
        }

        std::vector<particle_t> new_particles = receive_from_neighbours(FLAG_NEW_PARTICLES, env);
        particles.insert(particles.end(), new_particles.begin(), new_particles.end());

        MPI_Barrier(env.grid_comm);
        printf("[%d] \t Received %d particles, sent %d, now with %d total\n",
               env.rank, (int)new_particles.size(), sent, (int)particles.size());

    }

    MPI_Barrier(env.grid_comm);
    double pressure = 0;
    MPI_Reduce(&local_pressure, &pressure, 1, MPI_DOUBLE, MPI_SUM, ROOT_RANK, env.grid_comm);

    double t_end = MPI_Wtime();
    if(env.rank == ROOT_RANK){
        pressure /= ((2 * (BOX_HORIZ_SIZE + BOX_VERT_SIZE)));
        printf("Pressure : %f \t Time: %f\n", pressure,t_end-t_start);
    }

    quit_env();

    return 0;
}
