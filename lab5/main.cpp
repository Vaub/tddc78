#include <stdio.h>
#include <stdlib.h>
#include <list>
#include <vector>
#include <zconf.h>

#include "definitions.h"
#include "mpi_env.h"

#define NB_PARTICLES 1000
#define NB_STEPS 10

static mpi_env_t env;

double rand_range(void) {
    return (double)rand() / (double)RAND_MAX;
}

void send_to_neighbours(int flag, const mpi_env_t& env, int count, particle_t* particles) {

    MPI_Request requests[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            int nbr_rank = env.nbrs[i][j];
            if (nbr_rank == -1 || nbr_rank == env.rank) { continue; }

            MPI_Issend(particles, count, env.types.particle, nbr_rank, flag,
                       env.grid_comm, &requests[i][j]);
        }
    }

}

void receive_from_neighbours(int flag, const mpi_env_t& env, int& count, particle_t* particles) {

    count = 0;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            int nbr_rank = env.nbrs[i][j];
            if (nbr_rank == -1 || nbr_rank == env.rank) { continue; }

            int tmp_count;
            MPI_Status status;
            MPI_Recv(particles, MAX_NO_PARTICLES, env.types.particle, nbr_rank, flag,
                     env.grid_comm, &status);

            MPI_Get_count(&status, env.types.particle, &tmp_count);
            //count += tmp_count;
        }
    }

}

int main(int argc, char* argv[]) {

    env = init_env(&argc, &argv);

    float block_width = (float)ceil(BOX_HORIZ_SIZE / (float)env.grid_size[0]);
    float block_height = (float)ceil(BOX_VERT_SIZE / (float)env.grid_size[1]);

    cord_t block = {
        .x0 = block_width * env.coords[0], .y0 = block_height * env.coords[1],
        .x1 = block_width * env.coords[0] + block_width,
        .y1 = block_height * env.coords[1] + block_height
    };
    cord_t wall = { .x0 = 0, .y0 = 0, .x1 = BOX_HORIZ_SIZE, .y1 = BOX_VERT_SIZE };

    std::list<particle_t> particles(NB_PARTICLES);
    for (auto& particle : particles) {
        double vel = rand_range() * MAX_INITIAL_VELOCITY;
        double angle = rand_range() * (2*PI);

        pcord_t pcord = {
            .x = (float)(rand_range() * block_width + block.x0),
            .y = (float)(rand_range() * block_height + block.y0),
            .vx = (float)(vel * cos(angle)), .vy = (float)(vel * sin(angle))
        };

        particle.pcord = pcord;
    }

    double pressure = 0.0;

    std::vector<particle_t> treated;
    std::vector<particle_t> to_send;
    auto received_array = new particle_t[MAX_NO_PARTICLES];
    int received_count;

    to_send.reserve(particles.size());
    std::copy(particles.begin(), particles.end(), std::back_inserter(to_send));

    send_to_neighbours(FLAG_NBR_PARTICLES, env, (int)to_send.size(), &to_send[0]);
    receive_from_neighbours(FLAG_NBR_PARTICLES, env, received_count, received_array);

    std::list<particle_t> received(received_array, received_array + sizeof(received_array) / received_count);

    printf("%d \t Received %d particles from nbrs\n", env.rank, received_count);
    for (int step = 0; step < NB_STEPS; ++step) {

        for (auto current = particles.begin(); current != particles.end(); ++current) {

            bool has_interacted = false;
            for (auto i = std::next(current); i != particles.end() && !has_interacted; ++i) {
                float collided_at = collide(&current->pcord, &i->pcord);
                if (collided_at == -1) continue;

                interact(&current->pcord, &i->pcord, collided_at);

                treated.push_back(*current);
                particles.erase(i);
                has_interacted = true;
            }

            for (auto i = received.begin(); i != received.end() && !has_interacted;) {
                float collided_at = collide(&current->pcord, &i->pcord);
                if (collided_at == -1) {
                    i++;
                    continue;
                }

                interact(&current->pcord, &i->pcord, collided_at);

                treated.push_back(*current);
                i = received.erase(i);
                has_interacted = true;
            }

            if (has_interacted) {
                continue;
            }

            feuler(&current->pcord, 1);
            pressure += wall_collide(&current->pcord, wall);
        }

        particles.insert(particles.end(), treated.begin(), treated.end());
        treated.clear();

    }

    double total_pressure = pressure / (NB_STEPS + (2* BOX_VERT_SIZE + 2 * BOX_HORIZ_SIZE));
    printf("Pressure is %f\n", total_pressure);

    quit_env();

    return 0;
}
