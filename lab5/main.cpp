#include <stdio.h>
#include <stdlib.h>
#include <list>
#include <vector>
#include <iterator>

#include "definitions.h"
#include "mpi_env.h"

#include <VT.h>

#define NB_PARTICLES 100
#define NB_STEPS 1000
#define BOX_DEFAULT 10e4

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
                      particle_t *particle, std::vector<particle_t> nbrs[3][3]) {
    const pcord_t &coords = particle->pcord;

    int shift_x = coords.x < block.x0 ? -1 :
                  coords.x > block.x1 ? +1 :
                  0;
    int shift_y = coords.y < block.y0 ? -1 :
                  coords.y > block.y1 ? +1 :
                  0;

    if (shift_x == 0 && shift_y == 0) { return false; }

    //printf("(%f, %f) with square [%f -> %f]", coords.x, coords.y, block.x0, block.x1);

    nbrs[1 + shift_x][1 + shift_y].push_back(*particle);
    return true;
}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
    env = init_env();
        
    int nb_particles_cpu = argc > 1 ? atoi(argv[1]) : NB_PARTICLES;
    float box_width = BOX_DEFAULT, box_height = BOX_DEFAULT;
    if (argc == 4) {
	box_width = (float)atoi(argv[2]);
	box_height = (float)atoi(argv[3]);
    }

    int total_particles = (nb_particles_cpu * env.nb_cpu);
    if (env.rank == ROOT_RANK && total_particles > MAX_NO_PARTICLES) {
	printf("Error: cannot have more than %d particles, currently %d [%d x %d]", MAX_NO_PARTICLES, total_particles, env.nb_cpu, nb_particles_cpu);
	MPI_Finalize();
	exit(4);
    }

    int vt_error = 0;//VT_initialize(&argc, &argv);
   
    int vt_class, vt_particle_counter, vt_g_collisions, vt_g_comms;
    int class_err = VT_classdef("Simulation", &vt_class);
    //printf("VT: %d %d\n", vt_error, class_err);

    int func_err = VT_funcdef("Collisions", vt_class, &vt_g_collisions);   
    int64_t particle_bounds[2] = { 0, MAX_NO_PARTICLES };
    VT_countdef("Particles", vt_class, VT_COUNT_INTEGER64, VT_ME, particle_bounds, "p", &vt_particle_counter);    

    float block_width = (float) ceil(box_width / (float) env.grid_size[0]);
    float block_height = (float) ceil(box_height / (float) env.grid_size[1]);

    cord_t block = {
            .x0 = block_width * env.coords[0],
            .y0 = block_height * env.coords[1],
            .x1 = ((block_width * env.coords[0]) + block_width),
            .y1 = ((block_height * env.coords[1]) + block_height)
    };
    cord_t wall = {.x0 = 0, .y0 = 0, .x1 = box_width, .y1 = box_height};

    // Init particles with random positions and velocity
    std::list<particle_t> particles((size_t)nb_particles_cpu);
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

        // Particles sent to neighbours (see it as a grid around current rank at [1][1])
        std::vector<particle_t> to_send_nbrs[3][3];
        std::vector<particle_t> collided;
	//printf("[%d] first with [%f x %f]\n", env.rank, particles.begin()->pcord.x, particles.begin()->pcord.y);
	
        for (auto current = particles.begin(); current != particles.end();) {
            bool has_collided = false;

            // check collisions
            VT_enter(vt_g_collisions, VT_NOSCL);
            for (auto with = std::next(current); with != particles.end() && !has_collided;) {
                float collided_at = collide(&current->pcord, &with->pcord);
                if (collided_at == -1) {
                    ++with;
                    continue;
                }

                interact(&current->pcord, &with->pcord, collided_at);

                collided.push_back(*with);
                particles.erase(with);
                has_collided = true;
            }
	    VT_end(vt_g_collisions);

            // move particles which have not collided
            if (!has_collided) {
                feuler(&current->pcord, 1);
            }

            current++;
        }
	
	// put back the collided particles
        particles.insert(particles.end(), collided.begin(), collided.end());

	// calculate pressure and send to neighbours if needed
        for (auto i = particles.begin(); i != particles.end();) {
            local_pressure += wall_collide(&i->pcord, wall);

            if (try_send_to_nbrs(env, block, &(*i), to_send_nbrs)) {
                i = particles.erase(i);
                continue;
            }

            i++;
        }

	// Prepare the communication to neighbours
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
	
	int nb_particles = particles.size();
	VT_countval(1, &vt_particle_counter, &nb_particles);

        //printf("[%d] \t Received %d particles, sent %d, now with %d total\n",
        //       env.rank, (int)new_particles.size(), sent, (int)particles.size());

    }

    //printf("[%d \t (%d,%d)] Pressure is %.2f with %d particles\n", env.rank, env.coords[0], env.coords[1], local_pressure, (int)particles.size());

    // sum all local pressures
    double pressure = 0;
    MPI_Reduce(&local_pressure, &pressure, 1, MPI_DOUBLE, MPI_SUM, ROOT_RANK, env.grid_comm);

    double t_end = MPI_Wtime();
    if(env.rank == ROOT_RANK){
        pressure /= (NB_STEPS * (2 * (box_width + box_height)));
	printf("Procs,Time,p,V,n,R,T\n");
	
	// calculate a kind of pV=nRT to test results
	double p = pressure;
	double V = box_width * box_height;
	int n = total_particles;
	double R = 8.314; // const
	double T = (p*V)/(n*R);
	printf("%d,%f,%f,%f,%d,%f,%f\n",env.nb_cpu,t_end-t_start,p,V,n,R,T);
    }

    quit_env();
    MPI_Finalize();
    VT_finalize();

    return 0;
}
