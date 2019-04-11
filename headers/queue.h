#include "a_star.h"

typedef struct Queue {
    int capacity;
    int items_count;
    Node[] items;
} Queue;

Queue* queue_init(capacity int);
void queue_push(Queue *queue, Node* node);
Node* queue_pop(Queue *queue);
void queue_free(Queue *queue);

