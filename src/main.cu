#include "a_star.cuh"

__device__ void memory_init(Memory* memory, size_c mem_size) {
    memory->data = (void*)((size_c)memory + sizeof(Memory));
    memory->size = mem_size - sizeof(Memory);
    memory->allocated = 0;
}

__device__ void* memory_allocate(size_t size) {
    size_c allocated = atomicAdd(&(p.memory->allocated), size);
    assert(allocated + size < p.memory->size);
    return (void*)((size_c)p.memory->data + allocated);
}

__device__ int node_id_puzzle(NodePuzzle* node) {
    int count = p.puzzle.n * p.puzzle.n;
    int id = 0;
    for (int i = 0; i < count; i++) {
        id += i*node->numbers[i];
    }
    return id;
}

__device__ int node_id_pathfinding(NodePathfinding* node) {
    return node->x*p.pathfinding.dim_y + node->y;
}

__device__ int node_id(Node* node) {
    void *nd = node_data(node);
    switch (p.type) {
        case PROBLEM_TYPE_PUZZLE:
            return node_id_puzzle((NodePuzzle*)nd);
        case PROBLEM_TYPE_PATHFINDING:
            return node_id_pathfinding((NodePathfinding*)nd);
        default:
            assert(false);
            return -1;
    }
}

__device__ void* node_data(Node* node) {
    return (void*)(node + sizeof(Node));
}

__device__ size_t node_size() {
    switch (p.type) {
        case PROBLEM_TYPE_PUZZLE:
            return sizeof(Node) + sizeof(int)*p.puzzle.n*p.puzzle.n;
        case PROBLEM_TYPE_PATHFINDING:
            return sizeof(Node) + sizeof(NodePathfinding);
        default:
            assert(false);
            return 0;
    }
}

__device__ void queues_init(Queue *queues, size_c all_memory, int k) {
    size_c queues_memory = sizeof(Queue)*k;
    size_c items_memory = (all_memory - queues_memory) / k;
    for (int i = 0; i < k; i++) {
        queues[i].count = 0;
        queues[i].capacity = items_memory / sizeof(Node*);
        queues[i].items = (Node*)((size_c)queues + queues_memory +
                (i*items_memory));
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

__device__ void map_init(Map *map, size_c map_size) {
    map->hs = MAP_HASHING_FUNCTIONS;
    map->nodes = (Node**)((size_c)map + sizeof(Map));
    map->nodes_count = (map_size - sizeof(Map)) / sizeof(Node**);
}

__device__ void map_deduplicate(Node* nodes, Node* nodes_dest, int n) {

}

__global__ void gpu_astar_init(Problem hp) {
    p = hp;
    memory_init(p.memory, p.mem_size);
    queues_init(p.queues, p.queues_size, p.k);
    map_init(p.map, p.map_size);
}

__global__ void gpu_astar(int k) {

}

int main() {
    Problem p;
    p.mem_size = 1024*1024 * 9L;
    p.queues_size = 1024 * 512L;
    p.map_size = 1024 * 512L;
    p.k = 2;

    p.puzzle.n = 3;
    p.type = PROBLEM_TYPE_PUZZLE;

    // Initialize memory.
    handleError(cudaMalloc((void**)&p.memory, p.mem_size));
    handleError(cudaMalloc((void**)&p.queues, p.queues_size));
    handleError(cudaMalloc((void**)&p.map, p.map_size));
    gpu_astar_init<<<1, 1>>>(p);
    cudaDeviceSynchronize();

    // Run algorithm.
    gpu_astar<<<1, 1>>>(2);
    cudaDeviceSynchronize();

    // TODO: Fetch results from GPU.

    // Free memory.
    handleError(cudaFree(p.memory));
    handleError(cudaFree(p.queues));
    handleError(cudaFree(p.map));
    cudaDeviceSynchronize();

    return 0;
}

