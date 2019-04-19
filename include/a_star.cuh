#include <stdio.h>
#include <stdint.h>
#include <assert.h>

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define handleError(err) (HandleError(err, __FILE__, __LINE__))

typedef unsigned long long int size_c;


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
    int numbers[];
} NodePuzzle;

typedef struct {
    int x, y;
} NodePathfinding;

typedef struct Node {
    int g;
    int f;
    Node* previous_node;
} Node;

__device__ int node_id(void *node_data);
__device__ void* node_data(Node* node);
__device__ size_t node_size();


#define MAP_HASHING_FUNCTIONS 10

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
    int n;
} Puzzle;

typedef struct {
    int dim_x, dim_y;
} Pathfinding;

typedef struct {
    Memory* memory;
    size_c mem_size;

    Queue* queues;
    size_c queues_size;

    Map* map;
    size_c map_size;

    int k;
    int type;

    Puzzle puzzle;
    Pathfinding pathfinding;
} Problem;

__device__ Problem p;
