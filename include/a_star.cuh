#include <stdio.h>
#include <assert.h>

#define QUEUE_K 2
#define QUEUE_CAPACITY 10

#define MAP_HASHING_FUNCTIONS 10

#define TYPE_PUZZLE 1
#define TYPE_PATHFINDING 2

static void HandleError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__ ))

typedef struct {
    int g;
    int f;
    void *data;
} Node;

typedef struct {
    Node* nodes;
    int hashing_functions_count;
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

