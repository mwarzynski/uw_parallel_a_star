#ifndef QUEUE_H
#define QUEUE_H

#include "types.cuh"
#include "node.cuh"

typedef struct {
    Node** items;
    int capacity;
    int count;
    Node** items_pushed;
    int pushed_last_processed;
    int pushed_count;
    int pushed_length;
    int pushed_capacity;
} Queue;

__device__ void queues_init(Queue* queue, size_c memory, int k);
__device__ void queue_push(Queue* queue, Node* node);
__device__ void queue_fix(Queue* queue);
__device__ void queue_pop(Queue* queue, Node** node, int max_fg);
__device__ Node* queue_min(Queue* queue);

#endif // QUEUE_H
