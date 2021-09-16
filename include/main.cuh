#ifndef A_STAR_H
#define A_STAR_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <getopt.h>
#include <math.h>
#include <assert.h>
#include <limits.h>

#include "queue.cuh"
#include "memory.cuh"
#include "map.cuh"
#include "node.cuh"

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define handleError(err) (HandleError(err, __FILE__, __LINE__))

typedef struct {
    int n;
    int numbers_count;
} Sliding;

typedef struct {
    int dim_x, dim_y;
    int unusual_nodes_count;

    Map* map_unusual_nodes;
    size_c map_size;
} Pathfinding;

typedef struct {
    Memory* memory;
    size_c mem_size;

    Queue* queues;
    size_c queues_size;
    int queues_items;

    Map* map;
    size_c map_size;

    int k;
    int type;

    Node *node_start;
    Node *node_destination;

    Sliding sliding;
    Pathfinding pathfinding;

    int solutions_min_g;
    int queues_min_fg;
} Problem;

__device__ Problem p;

__device__ bool solution_found;

#endif // A_STAR_H
