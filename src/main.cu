#include "a_star.cuh"

__device__ void queue_init(Queue* queue, int capacity) {
    queue->items = (Node*)malloc(sizeof(Node)*capacity);
    queue->capacity = capacity;
    queue->items_count = 0;
}

__device__ Queue* queues_init(int k, int capacity) {
    Queue* queues;
    queues = (Queue*)malloc(sizeof(Queue)*k);
    for (int i = 0; i < k; i++) {
        queue_init(&queues[i], capacity);
    }
    return queues;
}

__device__ void queue_push(Queue *queue, Node* node) {
    return;
}

__device__ Node* queue_pop(Queue *queue) {
    return NULL;
}

__device__ void queue_free(Queue* queue) {
    free(queue->items);
}

__device__ void queues_free(Queue* queues, int k) {
    for (int i = 0; i < k; i++) {
        queue_free(&queues[i]);
    }
    free(queues);
}

__device__ int map_hash(Map* H, int j, Node *node) {
    return 7;
}

__device__ Map* map_init(int d, int size) {
    return NULL;
}

__device__ void map_deduplicate(Map* H, Node* nodes, Node* nodes_dest, int n) {}

__global__ void gpu_astar() {
    int k = QUEUE_K;
    int capacity = QUEUE_CAPACITY;

    Queue *queues = NULL;
    queues = queues_init(k, capacity);

    queues_free(queues, k);
}

int main() {
    gpu_astar<<<1, 1>>>();
    return 0;
}

