#include "memory.cuh"

__device__ void memory_init(Memory* memory, size_c mem_size) {
    memory->data = (void*)((size_c)memory + sizeof(Memory));
    memory->size = mem_size - sizeof(Memory);
    memory->allocated = 0;
}

__device__ void* memory_allocate(Memory *mem, size_c size) {
    if (size % 0x10 != 0) {
        size += (0x10 - (size % 0x10));
    }
    size_c allocated = atomicAdd(&(mem->allocated), size);
    assert(allocated + size < mem->size);
    return (void*)((size_c)mem->data + allocated);
}

