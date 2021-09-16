#include "queue.cuh"

__device__ void queues_init(Queue *queues, size_c all_memory, int k) {
    size_c queues_memory = sizeof(Queue)*k;
    size_c heap_queue_memory = (all_memory - queues_memory) / k;

    size_c heap_memory = heap_queue_memory * 0.9;
    heap_memory = heap_memory - (heap_memory % 0x10);
    size_c queue_memory = heap_queue_memory - heap_memory;

    size_c tmp;

    for (int i = 0; i < k; i++) {
        // Heap memory.
        queues[i].count = 0;
        queues[i].capacity = heap_memory / sizeof(Node**);
        tmp = ((size_c)queues + queues_memory +
                (i*heap_queue_memory));
        tmp += 0x10 - (tmp % 0x10);
        queues[i].items = (Node**)tmp;
        // Queue memory.
        queues[i].pushed_count = 0;
        queues[i].pushed_length = 0;
        queues[i].pushed_last_processed = 0;
        queues[i].pushed_capacity = queue_memory / sizeof(Node**);
        tmp = ((size_c)queues + queues_memory +
                (i*heap_queue_memory) + heap_memory);
        tmp += 0x10 - (tmp % 0x10);
        queues[i].items_pushed = (Node**)tmp;
    }
}

__device__ void queue_push(Queue *queue, Node* node) {
    int length = atomicAdd(&queue->pushed_length, 1);
    assert((length) % queue->pushed_capacity != (queue->pushed_last_processed - 1)%queue->pushed_capacity);
    queue->items_pushed[length % queue->pushed_capacity] = node;
    atomicAdd(&queue->pushed_count, 1);
    // assert(queue->pushed_count < queue->pushed_capacity);
}

__device__ void queue_fix(Queue *queue) {
    int start = queue->pushed_last_processed;
    int end = queue->pushed_count;
    int p;
    Node *node;
    for (int l = start; l < end; l++) {
        node = queue->items_pushed[l % queue->pushed_capacity];
        int i = queue->count++;
        queue->items[i] = node;
        assert(queue->count < queue->capacity);
        while (i > 0) {
            p = (i-1)/2;
            if (queue->items[p]->fg < node->fg) {
                break;
            }
            if (queue->items[p]->fg == node->fg) {
                if (queue->items[p]->g > node->g) {
                    break;
                }
            }
            queue->items[i] = queue->items[p];
            i = p;
        }
        queue->items[i] = node;
    }
    queue->pushed_last_processed = end;
}

__device__ void queue_downify(Queue *queue, int i) {
    int l = 2*i+1;
    int r = l+1;
    // Determine if we need to push value down.
    int min = i;
    if (l < queue->count && queue->items[l]->fg < queue->items[min]->fg) {
        min = l;
    }
    if (r < queue->count && queue->items[r]->fg < queue->items[min]->fg) {
        min = r;
    }
    // If one of our children has a better value, bring it up.
    // We also need to make sure our subtree will have correct values.
    if (min != i) {
        Node *t = queue->items[i];
        queue->items[i] = queue->items[min];
        queue->items[min] = t;
        queue_downify(queue, min);
    }
}

__device__ void queue_pop(Queue *queue, Node **result, int max_fg) {
    if (queue->count == 0 || queue->items[0]->fg > max_fg) {
        *result = NULL;
        return;
    }
    *result = queue->items[0];
    queue->items[0] = queue->items[--queue->count];
    queue_downify(queue, 0);
}

__device__ Node* queue_min(Queue *queue) {
    if (queue->count == 0) {
        return NULL;
    }
    return queue->items[0];
}

