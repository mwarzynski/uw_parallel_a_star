#include "map.cuh"


typedef unsigned long long int size_c;

__device__ int map_hash(int j, int id) {
    return (id*(j+1));
}

__device__ void map_init(Map *map, size_t map_size) {
    map->hs = MAP_HASHING_FUNCTIONS;
    map->nodes = (Node**)((size_c)map + sizeof(Map));
    map->nodes_count = (map_size - sizeof(Map)) / sizeof(Node**);
}

__device__ Node* map_get(Map *map, int id) {
    id = id % map->nodes_count;
    assert(id < map->nodes_count);
    return map->nodes[id];
}

__device__ Node* map_set(Map *map, Node *node, int i) {
    i = i % map->nodes_count;
    return (Node*)atomicExch((size_c*)&map->nodes[i], (size_c)node);
}

__device__ bool map_is_duplicate(Map* map, Node* node) {
    int z = 0;
    // Firstly, determine the first empty place (or if exists, the index for provided node).
    int nid = node->id;
    Node *tmp;
    for (int i = 0; i < map->hs; i++) {
        tmp = map_get(map, map_hash(i, nid));
        if (tmp == NULL || node->id == tmp->id) {
            z = i;
            break;
        }
    }

    // Set node in map.
    tmp = map_set(map, node, map_hash(z, nid));

    // Check if we got a duplicate.
    if (tmp != NULL && node->id == tmp->id) {
        return true;
    }
    for (int i = 0; i < map->hs; i++) {
        if (i == z) {
            continue;
        }
        tmp = map_get(map, map_hash(i, nid));
        if (tmp != NULL && node->id == tmp->id) {
            return true;
        }
    }
    return false;
}
