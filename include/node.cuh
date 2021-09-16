#ifndef NODE_H
#define NODE_H

#include <stdint.h>
#include <assert.h>

#include "types.cuh"

typedef struct {
    int numbers[];
} NodeSliding;

typedef struct {
    int x, y;
    int weight;
} NodePathfinding;

typedef struct Node {
    unsigned int id;
    int g;
    int fg;
    Node* previous_node;
    void *data;
} Node;

__device__ unsigned int node_id(Node* node, int ptype, int sliding_n, int dim_x);
__device__ int node_f(Node *node, Node* dest, int ptype, int sliding_n);
__device__ bool node_expand(Node *node, Node* expanded, int ptype, int sliding_n, int dim_x, int dim_y);
__device__ __host__ size_t node_size(int problem_type, int sliding_n);
__device__ void node_copy(void* src, void* dst, size_t size);
__device__ bool node_compare(void *src, void *dst, int ptype, int sliding_n);

#endif // NODE_H
