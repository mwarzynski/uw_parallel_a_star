#ifndef MAP_H
#define MAP_H

#include <assert.h>
#include "node.cuh"

#define MAP_HASHING_FUNCTIONS 10

typedef struct {
    Node** nodes;
    size_t nodes_count;
    int hs;
} Map;

__device__ int map_hash(int j, int id);

__device__ void map_init(Map* map, size_t map_size);
__device__ Node* map_get(Map* map, int i);
__device__ Node* map_set(Map* map, Node *node, int i);
__device__ bool map_is_duplicate(Map* map, Node* node);

#endif // MAP_H
