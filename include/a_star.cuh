#include <stdio.h>
#include <assert.h>

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define handleError(err) (HandleError(err, __FILE__, __LINE__))

typedef unsigned long long int size_c;

#define QUEUE_K 2
#define QUEUE_CAPACITY 10

#define MAP_HASHING_FUNCTIONS 10

#define PROBLEM_TYPE_PUZZLE 1
#define PROBLEM_TYPE_PATHFINDING 2


typedef struct {
    void *data;
    size_c allocated;
    size_c size;
} Memory;

__device__ void memory_init(Memory *mem, size_c size);
__device__ void* memory_allocate(size_c size);


typedef struct {
    int *numbers;
} NodePuzzle;

typedef struct {
} NodePathfinding;

typedef struct {
    int g;
    int f;
    void *data;
} Node;

__device__ int node_id(Node *node);


typedef struct {
    Node** nodes;
    size_c nodes_count;
    int hs;
} Map;

__device__ int map_hash(Map* H, int j, Node *node);

__device__ void map_init(Map* map, size_c map_size);
__device__ void map_deduplicate(Node* nodes, Node* nodes_dest, int n);


typedef struct {
    int capacity;
    int count;
    Node* items;
} Queue;

__device__ void queues_init(Queue* queue, size_c memory, int k);
__device__ void queue_push(Queue* queue, Node* node);
__device__ void queue_pop(Queue* queue, Node *node);


typedef struct {
    Memory* memory;
    size_c mem_size;

    Queue* queues;
    size_c queues_size;

    Map* map;
    size_c map_size;

    int k;
    int type;
} Problem;

__device__ Problem p;
