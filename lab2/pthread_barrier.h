#ifndef LAB2_PTHREAD_BARRIER_H
#define LAB2_PTHREAD_BARRIER_H

#include <pthread.h>

typedef struct Thread_Barrier {
    int nb_thread;

    pthread_cond_t condition;
    pthread_mutex_t lock;
} Barrier;


void init_barrier(Barrier* barrier, const int nb_threads);

void wait_for_process(Barrier* barrier);

void destroy_barrier(Barrier* barrier);


#endif //LAB2_PTHREAD_BARRIER_H
