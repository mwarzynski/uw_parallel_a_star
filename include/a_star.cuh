#include <stdio.h>

#define TYPE_PUZZLE 1
#define TYPE_PATHFINDING 2

typedef struct {
    double g;
    double f;
    void *data;
} Node;

typedef struct {
    Node* nodes;
    int d;
    int size;
} Map;

int map_hash(Map* H, int j, Node *node);

Map* map_init(int d, int size);
void map_deduplicate(Map* H, Node* nodes, Node* nodes_dest, int n);

typedef struct Queue {
    int capacity;
    int items_count;
    Node* items;
} Queue;

Queue* queues_init(int k, int capacity);
void queue_push(Queue* queue, Node* node);
Node* queue_pop(Queue* queue);
void queues_free(Queue* queue, int k);

