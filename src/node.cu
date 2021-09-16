#include "node.cuh"

__device__ int node_get_id() {
    return (threadIdx.x + blockDim.x*blockIdx.x);
}

__device__ unsigned int node_id_sliding(NodeSliding* node, int sliding_n) {
    int id = 0;
    int count = sliding_n * sliding_n;
    int base = 1;
    for (int i = 0; i < count; i++) {
        id += (node->numbers[i]*base);
        base *= 10;
        if (base > 1000000) {
            base = 1;
            id = (id * 13) % 15485863;
        }
    }
    if (id < 0)
        id *= -1;
    return id;
}

__device__ unsigned int node_id_pathfinding(NodePathfinding* node, int dim_x) {
    return node->x + dim_x*node->y;
}

__device__ unsigned int node_id(Node* node, int problem_type, int sliding_n, int dim_x) {
    switch (problem_type) {
        case PROBLEM_TYPE_SLIDING:
            return node_id_sliding((NodeSliding*)node->data, sliding_n);
        case PROBLEM_TYPE_PATHFINDING:
            return node_id_pathfinding((NodePathfinding*)node->data, dim_x);
        default:
            assert(false);
            return 0;
    }
}

__device__ void node_copy(void* src, void* dst, size_t size) {
    uint8_t* s = (uint8_t*)src;
    uint8_t* d = (uint8_t*)dst;
    for (int i = 0; i < size; i++) {
        d[i] = s[i];
    }
}

__device__ void node_sliding_find_empty_place(NodeSliding* node, int *x, int *y, int element, int sliding_n) {
    // TODO: idea: add node_dest lookup [element] -> (x,y);
    int count = sliding_n * sliding_n;
    for (int i = 0; i < count; i++) {
        if (node->numbers[i] == element) {
            *x = i / sliding_n;
            *y = i % sliding_n;
            return;
        }
    }
}

__device__ int node_f_sliding(NodeSliding* node, Node* destination, int sliding_n) {
    NodeSliding* node_destination = (NodeSliding*)destination->data;
    int f = 0;
    int x1, x2, y1, y2;
    int count = sliding_n * sliding_n;
    for (int i = 0; i < count; i++) {
        if (node->numbers[i] == 0) {
            continue;
        }
        x1 = i / sliding_n;
        y1 = i % sliding_n;
        node_sliding_find_empty_place(node_destination, &x2, &y2, node->numbers[i], sliding_n);
        f += abs(x1-x2) + abs(y1-y2);
    }
    return f;
}

__device__ int node_f_pathfinding(NodePathfinding *node, Node* destination) {
    NodePathfinding* dest = (NodePathfinding*)destination->data;
    return abs(node->x - dest->x) + abs(node->y - dest->y);
}

__device__ int node_f(Node *node, Node* destination, int ptype, int sliding_n) {
    switch (ptype) {
        case PROBLEM_TYPE_SLIDING:
            return node_f_sliding((NodeSliding*)node->data, destination, sliding_n);
        case PROBLEM_TYPE_PATHFINDING:
            return node_f_pathfinding((NodePathfinding*)node->data, destination);
        default:
            assert(false);
            return -1;
    }
}

__device__ void node_sliding_swap_elements(NodeSliding *n, int x1, int y1, int x2, int y2, int sliding_n) {
    int tmp = n->numbers[x1*sliding_n + y1];
    n->numbers[x1*sliding_n + y1] = n->numbers[x2*sliding_n + y2];
    n->numbers[x2*sliding_n + y2] = tmp;
}

__device__ bool node_expand_sliding(Node *node, Node* expanded, int ptype, int sliding_n) {
    NodeSliding *np = (NodeSliding*)node->data;
    NodeSliding *enp = (NodeSliding*)expanded->data;

    int count = sliding_n * sliding_n;
    for (int q = 0; q < count; q++) {
        enp->numbers[q] = np->numbers[q];
    }

    int id = node_get_id();
    int direction = id % 4;

    int x, y;
    node_sliding_find_empty_place(enp, &x, &y, 0, sliding_n);
    switch (direction) {
        case 0: // UP
            if (x > 0) {
                node_sliding_swap_elements(enp, x-1, y, x, y, sliding_n);
            } else {
                return false;
            }
            break;
        case 1: // LEFT
            if (y > 0) {
                node_sliding_swap_elements(enp, x, y-1, x, y, sliding_n);
            } else {
                return false;
            }
            break;
        case 2: // RIGHT
            if (y < sliding_n - 1) {
                node_sliding_swap_elements(enp, x, y+1, x, y, sliding_n);
            } else {
                return false;
            }
            break;
        case 3: // DOWN
            if (x < sliding_n - 1) {
                node_sliding_swap_elements(enp, x+1, y, x, y, sliding_n);
            } else {
                return false;
            }
            break;
        default:
            assert(false);
    }
    return true;
}

__device__ bool node_expand_pathfinding(Node* node, Node* expanded, int dim_x, int dim_y) {
    int id = node_get_id();
    int direction = id % 8;

    NodePathfinding* node_src = (NodePathfinding*)node->data;
    NodePathfinding* node_exp = (NodePathfinding*)expanded->data;

    switch (direction) {
        case 0: // UP LEFT
            if (node_src->x > 0 && node_src->y > 0)  {
                node_exp->x = node_src->x - 1;
                node_exp->y = node_src->y - 1;
            } else {
                return false;
            }
            break;
        case 1: // UP
            if (node_src->y > 0) {
                node_exp->x = node_src->x;
                node_exp->y = node_src->y - 1;
            } else {
                return false;
            }
            break;
        case 2: // UP RIGHT
            if (node_src->x < dim_x - 1 && node_src->y > 0) {
                node_exp->x = node_src->x + 1;
                node_exp->y = node_src->y - 1;
            } else {
                return false;
            }
            break;
        case 3: // RIGHT
            if (node_src->x < dim_x - 1) {
                node_exp->x = node_src->x + 1;
                node_exp->y = node_src->y;
            } else {
                return false;
            }
            break;
        case 4: // DOWN RIGHT
            if (node_src->x < dim_x - 1 && node_src-> y < dim_y - 1) {
                node_exp->x = node_src->x + 1;
                node_exp->y = node_src->y + 1;
            } else {
                return false;
            }
            break;
        case 5: // DOWN
            if (node_src-> y < dim_y - 1) {
                node_exp->x = node_src->x;
                node_exp->y = node_src->y + 1;
            } else {
                return false;
            }
            break;
        case 6: // DOWN LEFT
            if (node_src->x > 0 && node_src-> y < dim_y - 1) {
                node_exp->x = node_src->x - 1;
                node_exp->y = node_src->y + 1;
            } else {
                return false;
            }
            break;
        case 7: // LEFT
            if (node_src->x > 0) {
                node_exp->x = node_src->x - 1;
                node_exp->y = node_src->y;
            } else {
                return false;
            }
            break;
        default:
            assert(false);
            break;
    }
    return true;
}

__device__ bool node_expand(Node *node, Node* expanded, int ptype, int sliding_n, int dim_x, int dim_y) {
    switch (ptype) {
    case PROBLEM_TYPE_SLIDING:
        return node_expand_sliding(node, expanded, ptype, sliding_n);
    case PROBLEM_TYPE_PATHFINDING:
        return node_expand_pathfinding(node, expanded, dim_x, dim_y);
    default:
        assert(false);
        return false;
    }
}

__device__ __host__ size_t node_size(int ptype, int sliding_n) {
    switch (ptype) {
        case PROBLEM_TYPE_SLIDING:
            return sizeof(int)*sliding_n*sliding_n;
        case PROBLEM_TYPE_PATHFINDING:
            return sizeof(NodePathfinding);
        default:
            assert(false);
            return 0;
    }
}

__device__ bool node_compare_sliding(void *src, void *dst, int sliding_n) {
    NodeSliding *n1 = (NodeSliding*)src;
    NodeSliding *n2 = (NodeSliding*)dst;
    int count = sliding_n * sliding_n;
    for (int i = 0; i < count; i++) {
        if (n1->numbers[i] != n2->numbers[i]) {
            return false;
        }
    }
    return true;
}

__device__ bool node_compare_pathfinding(void *src, void *dst) {
    NodePathfinding *n1 = (NodePathfinding*)src;
    NodePathfinding *n2 = (NodePathfinding*)dst;
    return n1->x == n2->x && n1->y == n2->y;
}

__device__ bool node_compare(void *src, void *dst, int ptype, int sliding_n) {
    switch (ptype) {
        case PROBLEM_TYPE_PATHFINDING:
            return node_compare_pathfinding(src, dst);
        case PROBLEM_TYPE_SLIDING:
            return node_compare_sliding(src, dst, sliding_n);
        default:
            assert(false);
            return false;
    }
}
