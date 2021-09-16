#ifndef MEMORY_H
#define MEMORY_H

#include <assert.h>
#include "types.cuh"

typedef struct {
    void *data;
    size_c allocated;
    size_c size;
} Memory;

__device__ void memory_init(Memory *mem, size_c size);
__device__ void* memory_allocate(Memory *mem, size_c size);

#endif // MEMORY_H
