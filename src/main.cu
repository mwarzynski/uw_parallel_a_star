#include "a_star.cuh"

__device__ int node_id_puzzle(Node* node) {
    NodePuzzle* p = (NodePuzzle*)node->data;
    return p->id;
}

__device__ int node_id_pathfinding(Node* node) {
    // TODO(mwarzynski): implement generating ID for pathfinding Node.
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

__device__ void queue_init(Queue* queue, int capacity) {
    queue->items = (Node*)malloc(sizeof(Node)*capacity);
    assert(queue->items != NULL);
    queue->capacity = capacity;
    queue->count = 0;
}

__device__ Queue* queues_init(int k, int capacity) {
    Queue* queues;
    queues = (Queue*)malloc(sizeof(Queue)*k);
    assert(queues != NULL);
    for (int i = 0; i < k; i++) {
        queue_init(&queues[i], capacity);
    }
    return queues;
}

__device__ void queue_push(Queue *queue, Node* node) {
    // TODO(mwarzynski): consider memory reallocation.
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

__device__ void queue_free(Queue* queue) {
    free(queue->items);
}

__device__ void queues_free(Queue* queues, int k) {
    for (int i = 0; i < k; i++) {
        queue_free(&queues[i]);
    }
    free(queues);
}

__device__ int map_hash(Map* map, int j, Node *node) {
    // TODO(mwarzynski): implement proper hashing function.
    return 7;
}

__device__ Map* map_init(int hashing_functions_count, int size) {
    Map* map = (Map*)malloc(sizeof(Map));
    assert(map != NULL);
    map->hs = hashing_functions_count;
    map->nodes = (Node**)malloc(sizeof(Node*)*size);
    assert(map->nodes != NULL);
    map->size = size;
    return map;
}

__device__ void map_deduplicate(Map* H, Node* nodes, Node* nodes_dest, int n) {
    Node* node = &nodes[0];
    // Find index 'z' to place the item.
    int z = 0;
    int id = node_id(node);
    for (int j = 0; j < H->hs; j++) {
        int i = map_hash(H, j, node);
        if (H->nodes[i] == NULL || id == node_id(H->nodes[i])) {
            z = i;
            break;
        }
    }
    // Atomic swap value with the current node.
    Node* t = (Node*)atomicExch((unsigned long long int*)H->nodes[z], (unsigned long long int)&nodes[0]);
    if (node_id(t) == node_id(node)) {
        nodes[0] = NULL;
    }
}

__device__ void map_free(Map* map) {
    free(map->nodes);
    free(map);
}

__global__ void gpu_astar(int problem) {
    problem_type = problem;
    int k = QUEUE_K;
    int capacity = QUEUE_CAPACITY;

    Queue *queues = queues_init(k, capacity);
    Map *map = map_init(MAP_HASHING_FUNCTIONS, k);

    // TODO(mwarzynski): implement the actual algorithm.

    queues_free(queues, k);
    map_free(map);
}

int main() {
    gpu_astar<<<1, 1>>>(PROBLEM_TYPE_PUZZLE);
    cudaDeviceSynchronize();
    return 0;
}

