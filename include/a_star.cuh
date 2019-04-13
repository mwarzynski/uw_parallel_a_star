#include <stdio.h>
#include <assert.h>

#define QUEUE_K 2
#define QUEUE_CAPACITY 10

#define MAP_HASHING_FUNCTIONS 10

#define PROBLEM_TYPE_PUZZLE 1
#define PROBLEM_TYPE_PATHFINDING 2

__device__ int problem_type;

typedef struct {
    int id;
} NodePuzzle;

typedef struct {
    int g;
    int f;
    void *data;
} Node;

__device__ int node_id(Node *node);

typedef struct {
    Node** nodes;
    int hs;
    int size;
} Map;

__device__ int map_hash(Map* H, int j, Node *node);

__device__ Map* map_init(int hashing_functions_count, int size);
__device__ void map_deduplicate(Map* H, Node* nodes, Node* nodes_dest, int n);
__device__ void map_free(Map* map);

typedef struct Queue {
    int capacity;
    int count;
    Node* items;
} Queue;

__device__ Queue* queues_init(int k, int capacity);
__device__ void queue_push(Queue* queue, Node* node);
__device__ void queue_pop(Queue* queue, Node *node);
__device__ void queues_free(Queue* queue, int k);

