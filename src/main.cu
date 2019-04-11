#include "a_star.cuh"

__global__ void gpu_a_star(Queue *queues, Node *source, Node *destination, int k) {
    return;
}

__global__ void gpu_nodes_deduplicate() {
    return;
}

__global__ void gpu_append_expanded_nodes() {
    return;
}

int main() {
    int k = 2;
    Queue *dqueues = queues_init(k, 4);
    gpu_a_star<<<1, 1>>>(dqueues, NULL, NULL, k);
    queues_free(dqueues, k);
    return 0;
}

