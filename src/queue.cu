#include "a_star.cuh"

void queue_init(Queue* queue, int capacity) {
    cudaMalloc((void**)queue->items, sizeof(Node)*capacity);
    queue->capacity = capacity;
    queue->items_count = 0;
}

Queue* queues_init(int k, int capacity) {
    Queue* queues;
    cudaMalloc((void **)&queues, sizeof(Queue)*k);
    for (int i = 0; i < k; i++) {
        queue_init(&queues[i], capacity);    
    }
    return queues;
}

void queue_push(Queue *queue, Node* node) {
    return;
}

Node* queue_pop(Queue *queue) {
    return NULL;
}

void queue_free(Queue* queue) {
    cudaFree(queue->items);
}

void queues_free(Queue* queues, int k) {
    for (int i = 0; i < k; i++) {
        queue_free(&queues[i]);
    }
    cudaFree(queues);
}

