#include "a_star.cuh"

__device__ void memory_init(size_t mem_size) {
    memory->data = (void*)((size_c)memory + sizeof(Memory));
    memory->size = mem_size - sizeof(Memory);
    memory->allocated = 0;
}

__device__ void* memory_allocate(size_t size) {
    size_c allocated = atomicAdd(&memory->allocated, size);
    assert(allocated + size < memory->size);
    return (void*)((size_c)memory->data + allocated);
}

__device__ int node_id_puzzle(Node* node) {
    return 0;
}

__device__ int node_id_pathfinding(Node* node) {
    return 0;
}

__device__ int node_id(Node* node) {
    switch (problem_type) {
        case PROBLEM_TYPE_PUZZLE:
            return node_id_puzzle(node);
        case PROBLEM_TYPE_PATHFINDING:
            return node_id_pathfinding(node);
        default:
            assert(false);
            return -1;
    }
}

__device__ void queues_init(int k, size_c all_memory) {
    size_c queues_memory = sizeof(Queue)*k;
    size_c items_memory = (all_memory - queues_memory) / k;
    for (int i = 0; i < k; i++) {
        queues[i].count = 0;
        queues[i].capacity = items_memory / sizeof(Node*);
        queues[i].items = (Node*)((size_c)queues + queues_memory + (i*items_memory));
    }
}

__device__ void queue_push(Queue *queue, Node* node) {
    assert(queue->count < queue->capacity);
    queue->count++;
    int i = queue->count-1;
    int p;
    while (i > 0) {
        p = (i-1)/2;
        if (queue->items[p].f < node->f) {
            break;
        }
        queue->items[i] = queue->items[p];
        i = p;
    }
    queue->items[i] = *node;
}

__device__ void queue_downify(Queue *queue, int i) {
    int l = 2*i+1;
    int r = l+1;
    // Determine if we need to push value down.
    int min = i;
    if (l < queue->count && queue->items[l].f < queue->items[min].f) {
        min = l;
    }
    if (r < queue->count && queue->items[r].f < queue->items[min].f) {
        min = r;
    }
    // If one of our children has a better value, bring it up.
    // We also need to make sure our subtree will have correct values.
    if (min != i) {
        Node t = queue->items[i];
        queue->items[i] = queue->items[min];
        queue->items[min] = t;
        queue_downify(queue, min);
    }
}

__device__ void queue_pop(Queue *queue, Node *result) {
    assert(queue->count > 0);
    *result = queue->items[0];
    queue->items[0] = queue->items[--queue->count];
    queue_downify(queue, 0);
}

__device__ int map_hash(Map* map, int j, Node *node) {
    return 0;
}

__device__ void map_init(int k, int hashing_functions_count) {

}

__device__ void map_deduplicate(Node* nodes, Node* nodes_dest, int n) {

}

__global__ void gpu_astar_init(
        Memory *dmem,
        Queue *dqueue,
        Map *dmap,
        int k,
        int problem,
        size_c mem_size,
        size_c queues_size,
        size_c map_size) {
    memory = dmem;
    queues = dqueue;
    map = dmap;
    problem_type = problem;
    memory_init(mem_size);
    queues_init(k, queues_size);
    map_init(k, map_size);
}

__global__ void gpu_astar(int k) {
    // TODO: implement the actual algorithm.
}

int main() {
    int k = QUEUE_K;
    int problem = PROBLEM_TYPE_PUZZLE;

    Memory *dev_mem;
    Queue *dev_queues;
    Map *dev_map;

    // Initialize memory.
    size_c mem_size = 1024*1024*1024 * 9L;
    handleError(cudaMalloc((void**)&dev_mem, mem_size));
    size_c queues_size = 1024*1024 * 512L;
    handleError(cudaMalloc((void**)&dev_queues, queues_size));
    size_c map_size = 1024*1024 * 512L;
    handleError(cudaMalloc((void**)&dev_map, map_size));
    gpu_astar_init<<<1, 1>>>(dev_mem, dev_queues, dev_map, k, problem, mem_size, queues_size, map_size);
    cudaDeviceSynchronize();

    // Run algorithm.
    gpu_astar<<<100, 100>>>(k);
    cudaDeviceSynchronize();

    // Fetch results from GPU.

    // Free memory.
    handleError(cudaFree(dev_mem));
    handleError(cudaFree(dev_queues));
    handleError(cudaFree(dev_map));
    cudaDeviceSynchronize();

    return 0;
}

