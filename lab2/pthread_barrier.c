#include "pthread_barrier.h"

void init_barrier(Barrier* barrier, const int nb_threads) {
    barrier->nb_thread = nb_threads;
    pthread_cond_init(&barrier->condition, NULL);
    pthread_mutex_init(&barrier->lock, NULL);
}

void wait_for_process(Barrier* barrier) {
    pthread_mutex_lock(&barrier->lock);
    barrier->nb_thread--;

    if (barrier->nb_thread <= 0) {
        pthread_cond_broadcast(&barrier->condition);
    } else {
        pthread_cond_wait(&barrier->condition, &barrier->lock);
    }

    pthread_mutex_unlock(&barrier->lock);
}

void destroy_barrier(Barrier* barrier) {
    pthread_cond_destroy(&barrier->condition);
    pthread_mutex_destroy(&barrier->lock);
}
