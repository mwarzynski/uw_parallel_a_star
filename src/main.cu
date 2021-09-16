#include "main.cuh"

__device__ int get_id() {
    return (threadIdx.x + blockDim.x*blockIdx.x);
}

__global__ void gpu_astar_init(Problem hp, void *node_start, void *node_dest, void *nodes_unusual_raw) {
    p = hp;
    memory_init(p.memory, p.mem_size);
    queues_init(p.queues, p.queues_size, p.k);
    p.queues_items = 1;
    map_init(p.map, p.map_size);

    
    if (p.type == PROBLEM_TYPE_PATHFINDING) {
        map_init(p.pathfinding.map_unusual_nodes, p.pathfinding.map_size);
        NodePathfinding* nodes_unusual = (NodePathfinding*)nodes_unusual_raw;
        Node *node;
        for (int i = 0; i < p.pathfinding.unusual_nodes_count; i++) {
            node = (Node*)memory_allocate(p.memory, sizeof(Node));
            node->data = (void*)&nodes_unusual[i];
            node->id = node_id(node, PROBLEM_TYPE_PATHFINDING, 0, p.pathfinding.dim_x);
            node = map_set(p.pathfinding.map_unusual_nodes, node, node->id);
            assert(node == NULL);
        }
    }

    p.node_start = (Node*)memory_allocate(p.memory, sizeof(Node));
    p.node_start->data = node_start;
    p.node_start->id = node_id(p.node_start, p.type, p.sliding.n, p.pathfinding.dim_x);
    p.node_destination = (Node*)memory_allocate(p.memory, sizeof(Node));
    p.node_destination->data = node_dest;
    p.node_destination->id = node_id(p.node_destination, p.type, p.sliding.n, p.pathfinding.dim_x);

    p.solutions_min_g = INT_MAX;
    p.queues_min_fg = INT_MAX - 16;

    solution_found = false;
}

__global__ void gpu_astar_final(Node** solutions, void* path, int* path_count, int path_count_max) {
    Node *m = NULL;
    Node *node;
    for (int q_id = 0; q_id < p.k; q_id++) {
        node = solutions[q_id];
        if (node == NULL) {
            continue;
        }
        if (m == NULL || node->g < m->g) {
            m = node;
        }
    }
    size_t nsize = node_size(p.type, p.sliding.n);
    int i = 0;
    size_c path_node;
    while (m != NULL) {
        path_node = ((size_c)path + i*nsize);
        node_copy(m->data, (void*)path_node, nsize);
        m = m->previous_node;
        i++;
    }
    *path_count = i;
}

__device__ void gpu_astar_update_queues_min_fg() {
    bool empty = true;
    Node *q_min;
    Node *total_min = NULL;
    for (int i = 0; i < p.k; i++) {
        q_min = queue_min(&p.queues[i]);
        if (q_min == NULL) {
            continue;
        }
        empty = false;
        if (total_min == NULL || q_min->fg < total_min->fg) {
            total_min = q_min;
        }
    }
    if (empty) {
        return;
    }
    // if (p.queues_min_fg != total_min->fg) {
    //     printf("QUEUES MIN FG: %d -> %d\n", p.queues_min_fg, total_min->fg);
    // }
    p.queues_min_fg = total_min->fg;
}

__global__ void gpu_astar(Node** nodes, int nodes_count, Node** solutions) {
    int expand_factor = 4*p.type;

    int id = get_id();

    int q_id = id / expand_factor;
    assert(q_id < p.k);

    int is_main = (id % expand_factor == 0) ? 1 : 0;
    int id_main = q_id * (expand_factor+1);
    int id_node = id_main + (id % expand_factor) + 1;

    if (id < nodes_count) {
        nodes[id] = NULL;
    }
    if (id < p.k) {
        solutions[id] = NULL;
    }

    if (id == 0) {
        queue_push(&p.queues[0], p.node_start);
    }

    assert(q_id < p.k);

    Node *node;
    int queue_items_diff;
    bool queue_popped = false;
    bool force_close;
    int push_q_id;
    while (!solution_found && p.queues_items > 0) {
        // Pop the min element from queue.
        if (is_main) {
            queue_pop(&p.queues[q_id], &nodes[id_main], p.queues_min_fg + 5);
            if (nodes[id_main] != NULL) {
                queue_items_diff = -1;
                queue_popped = true;
                force_close = false;
            }
        }

        if (queue_popped) {
            if (nodes[id_main]->g >= p.solutions_min_g) {
                nodes[id_main] = NULL;
                force_close = true;
            }
        }

        // Expand the elements using one thread per each expand direction.
        if (nodes[id_main] != NULL) {
            node = nodes[id_main];
            if (is_main && node_compare(node->data, p.node_destination->data, p.type, p.sliding.n)) {
                if (solutions[q_id] == NULL || node->fg < solutions[q_id]->fg) {
                    solutions[q_id] = node;

                    if (solutions[q_id]->g < p.solutions_min_g) {
                        // printf("SOLUTIONS MIN G: %d -> %d\n", p.solutions_min_g, solutions[q_id]->g);
                        atomicExch(&p.solutions_min_g, solutions[q_id]->g);
                    }
                }
            }

            if (solutions[q_id] != NULL && solutions[q_id]->fg <= p.queues_min_fg) {
                solution_found = true;
                break;
            }

            if (nodes[id_node] == NULL) {
                nodes[id_node] = (Node*)memory_allocate(p.memory, sizeof(Node));
                nodes[id_node]->data = memory_allocate(p.memory, node_size(p.type, p.sliding.n));
            }
            nodes[id_node]->previous_node = nodes[id_main];

            bool expanded = node_expand(nodes[id_main], nodes[id_node], p.type, p.sliding.n, p.pathfinding.dim_x, p.pathfinding.dim_y);
            nodes[id_node]->id = node_id(nodes[id_node], p.type, p.sliding.n, p.pathfinding.dim_x);

            if (p.type == PROBLEM_TYPE_PATHFINDING && expanded) {
                Node *unusual = map_get(p.pathfinding.map_unusual_nodes, nodes[id_node]->id);
                if (unusual != NULL) {
                    NodePathfinding* unusualp = (NodePathfinding*)unusual->data;
                    if (unusualp->weight == -1) {
                        expanded = false;
                    } else {
                        nodes[id_node]->g = nodes[id_main]->g + unusualp->weight;
                    }
                } else {
                    nodes[id_node]->g = nodes[id_main]->g + 1;
                }
                nodes[id_node]->fg = nodes[id_node]->g + node_f(nodes[id_node], p.node_destination, PROBLEM_TYPE_PATHFINDING, 0);
            } else if (p.type == PROBLEM_TYPE_SLIDING && expanded) {
                nodes[id_node]->g = nodes[id_main]->g + 1;
                nodes[id_node]->fg = nodes[id_node]->g + node_f(nodes[id_node], p.node_destination, PROBLEM_TYPE_SLIDING, p.sliding.n);
            }

            if (expanded && nodes[id_node] != NULL) {
                if (!map_is_duplicate(p.map, nodes[id_node])) {
                    assert(nodes[id_node] != NULL);
                    push_q_id = (nodes[id_node]->id * (q_id+1));
                    if (push_q_id < 0)
                        push_q_id *= -1;
                    push_q_id = push_q_id % p.k;
                    queue_push(&p.queues[push_q_id], nodes[id_node]);
                    nodes[id_node] = NULL;
                }
            }
        }

        // Fix queues items structure (sort items).
        __syncthreads();
        if (is_main) {
            queue_fix(&p.queues[q_id]);
            if (queue_popped) {
                if (!force_close) {
                    for (int i = 0; i < expand_factor; i++) {
                        if (nodes[id_node + i] == NULL) {
                            queue_items_diff += 1;
                        }
                    }
                }
                atomicAdd(&p.queues_items, queue_items_diff);
                queue_popped = false;
            }
            nodes[id_main] = NULL;
        }

        if (id == 0) {
            gpu_astar_update_queues_min_fg();
        }
    }
}

int parse_pathfinding(Problem *problem, char *filename, void **node_start, void **node_dest, void **unusual_nodes) {
    FILE *f = fopen(filename, "r");
    if (f == NULL) {
        perror("couldn't open input-data file");
        return 1;
    }

    if (fscanf(f, "%d,%d\n", &problem->pathfinding.dim_x, &problem->pathfinding.dim_y) < 2) {
        perror("couldn't parse pathfinding dimensions");
        fclose(f);
        return 1;
    }

    NodePathfinding *start = (NodePathfinding*)malloc(sizeof(NodePathfinding));
    if (fscanf(f, "%d,%d\n", &start->x, &start->y) < 2) {
        perror("couldn't parse pathfinding start position");
        fclose(f);
        return 1;
    }
    *node_start = start;
    NodePathfinding *dest = (NodePathfinding*)malloc(sizeof(NodePathfinding));
    if (fscanf(f, "%d,%d\n", &dest->x, &dest->y) < 2) {
        perror("couldn't parse pathfinding dest position");
        fclose(f);
        return 1;
    }
    *node_dest = dest;

    int obstacles_count;
    if (fscanf(f, "%d\n", &obstacles_count) < 1) {
        perror("couldn't parse pathfinding obstacles_count number");
        fclose(f);
        return 1;
    }

    NodePathfinding* nodes_obstacles = (NodePathfinding*)malloc(obstacles_count*sizeof(NodePathfinding));
    if (nodes_obstacles == NULL) {
        perror("couldn't allocate memory for obstacles_count");
        fclose(f);
        return 1;
    }
    for (int i = 0; i < obstacles_count; i++) {
        if (fscanf(f, "%d,%d\n", &nodes_obstacles[i].x, &nodes_obstacles[i].y) < 2) {
            perror("couldn't read obstacle coordinates");
            fclose(f);
            free(nodes_obstacles);
            return 1;
        }
        nodes_obstacles[i].weight = -1;
    }

    int nodes_worse_count;
    if (fscanf(f, "%d\n", &nodes_worse_count) < 1) {
        perror("couldn't parse pathfinding nodes_worse_count number");
        fclose(f);
        free(nodes_obstacles);
        return 1;
    }
    NodePathfinding* nodes_worse = (NodePathfinding*)malloc(nodes_worse_count*sizeof(NodePathfinding));
    if (nodes_worse == NULL) {
        perror("couldn't allocate memory for nodes_worse_count");
        fclose(f);
        return 1;
    }
    for (int i = 0; i < nodes_worse_count; i++) {
        if (fscanf(f, "%d,%d,%d\n", &nodes_worse[i].x, &nodes_worse[i].y, &nodes_worse[i].weight) < 3) {
            perror("couldn't read worse node coordinates");
            fclose(f);
            free(nodes_worse);
            free(nodes_obstacles);
            return 1;
        }
    }

    problem->pathfinding.unusual_nodes_count = obstacles_count + nodes_worse_count;
    NodePathfinding* nodes = (NodePathfinding*)malloc(problem->pathfinding.unusual_nodes_count*sizeof(NodePathfinding));
    if (nodes == NULL) {
        perror("couldn't allocat ememory for unusual nodes");
        free(nodes_obstacles);
        free(nodes_worse);
        fclose(f);
        return 1;
    }
    for (int i = 0; i < obstacles_count; i++) {
        nodes[i] = nodes_obstacles[i];
    }
    for (int i = 0; i < nodes_worse_count; i++) {
        nodes[obstacles_count + i] = nodes_worse[i];
    }
    *unusual_nodes = (void*)nodes;

    free(nodes_obstacles);
    free(nodes_worse);
    fclose(f);

    return 0;
}

char* parse_sliding_read_file(char *filename, size_t max_len) {
    char *buffer = (char*)malloc(sizeof(char)*max_len);
    FILE *f;

    f = fopen(filename, "r");
    if (f == NULL) {
        perror("couldn't open input-data file");
        free(buffer);
        return NULL;
    }

    size_t l = fread(buffer, sizeof(char), max_len, f);
    if (ferror(f) != 0) {
        fputs("couldn't read input-data file", stderr);
    } else {
        buffer[l++] = '\0';
    }

    if (fclose(f)) {
        perror("couldn't close input-data file");
        free(buffer);
        return NULL;
    }
    return buffer;
}

int parse_sliding(Problem *problem, char *filename, void **node_start, void **node_dest) {
    size_t max_len = 1024*1024;
    char* buffer = parse_sliding_read_file(filename, max_len);

    int n = 0;
    for (int i = 0; i < max_len && buffer[i] != '\0'; i++) {
        if (buffer[i] == '_') {
            buffer[i] = '0';
        }
        if (buffer[i] == ',') {
            n++;
            buffer[i] = ' ';
        }
    }
    n = (n + 2)/2;

    problem->sliding.numbers_count = n;
    problem->sliding.n = sqrt(n);

    int* ns = (int*)malloc(sizeof(int)*problem->sliding.numbers_count);
    int* nd = (int*)malloc(sizeof(int)*problem->sliding.numbers_count);
    char *buf = buffer;
    int pos;
    for (int i = 0; i < problem->sliding.numbers_count; i++) {
        sscanf(buf, "%d%n", &ns[i], &pos);
        buf += pos;
    }
    for (int i = 0; i < problem->sliding.numbers_count; i++) {
        sscanf(buf, "%d%n", &nd[i], &pos);
        buf += pos;
    }
    *node_start = ns;
    *node_dest = nd;

    free(buffer);
    return 0;
}

int write_sliding(FILE *f, Problem problem, int path_count, void* path) {
    NodeSliding *np;
    size_t nsize = node_size(problem.type, problem.sliding.n);
    int chars = 0;
    for (int i = path_count - 1; i >= 0; i--) {
        np = (NodeSliding*)((size_c)path + i*nsize);
        for (int j = 0; j < problem.sliding.numbers_count-1; j++) {
            if (np->numbers[j] != 0)
                chars = fprintf(f, "%d,", np->numbers[j]);
            else
                chars = fprintf(f, "_,");
            if (chars < 0) {
                return 3;
            }
        }
        if (fprintf(f, "%d\n",np->numbers[problem.sliding.numbers_count-1]) < 0) {
            return 3;
        }
    }
    return 0;
}

int write_pathfinding(FILE *f, int path_count, void* path) {
    NodePathfinding *path_nodes = (NodePathfinding*)path;
    for (int i = path_count - 1; i >= 0; i--) {
        if (fprintf(f, "%d,%d\n", path_nodes[i].x, path_nodes[i].y) < 0) {
            return 3;
        }
    }
    return 0;
}

int write_file(char *filename, Problem problem, int path_count, void* path, float elapsedTime) {
    // Open file descriptor.
    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        perror("couldn't open output file fd");
        return 2;
    }

    // Print out solution to file.
    fprintf(f, "%.0f\n", elapsedTime);
    int status;
    switch (problem.type) {
        case PROBLEM_TYPE_SLIDING:
            status = write_sliding(f, problem, path_count, path);
            break;
        case PROBLEM_TYPE_PATHFINDING:
            status = write_pathfinding(f, path_count, path);
            break;
        default:
            assert(false);
    }

    // Close file descriptor.
    if (fclose(f)) {
        perror("couldn't close output file fd");
        return 1;
    }

    return status;
}

int parse_arguments(int argc, char **argv, int *problem_type,
    char **input_file, char **output_file) {
    char *version = NULL;
    int c;
    while (1) {
        static struct option long_options[] = {
          {"version",     required_argument, 0, 'v'},
          {"input-data",  required_argument, 0, 'i'},
          {"output-data", required_argument, 0, 'o'},
          {0, 0, 0, 0}
        };
        int option_index = 0;
        c = getopt_long (argc, argv, "v:i:o:", long_options, &option_index);
        if (c == -1) {
            break;
        }
        switch (c) {
            case 'v':
                version = optarg;
                break;
            case 'i':
                *input_file = optarg;
                break;
            case 'o':
                *output_file = optarg;
                break;
            default:
                break;
        }
    }

    if (version == NULL || *input_file == NULL || *output_file == NULL) {
        printf("Invalid arguments. Required 'version', 'input-file', 'output-file'.\n");
        return 1;
    }

    *problem_type = 0;
    if (!strcmp("pathfinding", version)) {
        *problem_type = PROBLEM_TYPE_PATHFINDING;
    }
    if (!strcmp("sliding", version)) {
        *problem_type = PROBLEM_TYPE_SLIDING;
    }
    if (*problem_type == 0) {
        printf("Invalid 'version' argument.\n");
        return 1;
    }

    return 0;
}

int main(int argc, char **argv) {
    Problem problem;
    char *input_file = NULL;
    char *output_file = NULL;

    // Parse input data.
    if (parse_arguments(argc, argv, &problem.type, &input_file, &output_file)) {
        return 1;
    }
    void *host_node_start, *host_node_dest, *host_nodes_unusual;
    void *dev_node_start, *dev_node_dest, *dev_nodes_unusual;
    switch (problem.type) {
        case PROBLEM_TYPE_PATHFINDING:
            if (parse_pathfinding(&problem, input_file, &host_node_start, &host_node_dest, &host_nodes_unusual))
                return 1;
            handleError(cudaMalloc((void**)&dev_node_start, sizeof(NodePathfinding)));
            handleError(cudaMemcpy(dev_node_start, host_node_start, sizeof(NodePathfinding), cudaMemcpyHostToDevice));
            handleError(cudaMalloc((void**)&dev_node_dest, sizeof(NodePathfinding)));
            handleError(cudaMemcpy(dev_node_dest, host_node_dest, sizeof(NodePathfinding), cudaMemcpyHostToDevice));
            handleError(cudaMalloc((void**)&dev_nodes_unusual, problem.pathfinding.unusual_nodes_count*sizeof(NodePathfinding)));
            handleError(cudaMemcpy(dev_nodes_unusual, host_nodes_unusual, problem.pathfinding.unusual_nodes_count*sizeof(NodePathfinding), cudaMemcpyHostToDevice));
            free(host_nodes_unusual);
            problem.pathfinding.map_size = 1024*1024* 1024L;
            handleError(cudaMalloc((void**)&problem.pathfinding.map_unusual_nodes, problem.pathfinding.map_size));
            break;
        case PROBLEM_TYPE_SLIDING:
            if (parse_sliding(&problem, input_file, &host_node_start, &host_node_dest))
                return 1;
            handleError(cudaMalloc((void**)&dev_node_start, problem.sliding.numbers_count*sizeof(int)));
            handleError(cudaMemcpy(dev_node_start, host_node_start, sizeof(int)*problem.sliding.numbers_count, cudaMemcpyHostToDevice));
            handleError(cudaMalloc((void**)&dev_node_dest, problem.sliding.numbers_count*sizeof(int)));
            handleError(cudaMemcpy(dev_node_dest, host_node_dest, sizeof(int)*problem.sliding.numbers_count, cudaMemcpyHostToDevice));
            break;
    }

    // Initialize problem, memory and GPU.
    int block_num = 16;
    int threads_per_block = 1024;
    problem.mem_size = 1024*1024*1024 * 7L;
    problem.queues_size = 1024*1024*1024 * 2L;
    problem.map_size = 1024*1024*1024 * 2L;
    problem.k = (block_num * threads_per_block) / (4*problem.type);
    size_t nodes_count = ((problem.type*4)+1)*problem.k;
    Node **nodes, **nodes_solutions;
    handleError(cudaMalloc((void**)&nodes, sizeof(Node*)*nodes_count));
    handleError(cudaMalloc((void**)&nodes_solutions, sizeof(Node*)*problem.k));
    handleError(cudaMalloc((void**)&problem.memory, problem.mem_size));
    handleError(cudaMalloc((void**)&problem.queues, problem.queues_size));
    handleError(cudaMalloc((void**)&problem.map, problem.map_size));
    gpu_astar_init<<<1, 1>>>(problem, dev_node_start, dev_node_dest, dev_nodes_unusual);
    cudaDeviceSynchronize();

    // Run algorithm.
    cudaEvent_t start, stop;
    float elapsedTime;
    handleError(cudaEventCreate(&start));
    handleError(cudaEventCreate(&stop));
    handleError(cudaEventRecord(start, 0));
    gpu_astar<<<block_num, threads_per_block>>>(nodes, nodes_count, nodes_solutions);
    handleError(cudaEventRecord(stop, 0));
    handleError(cudaEventSynchronize(stop));
    handleError(cudaEventElapsedTime(&elapsedTime, start, stop));
    handleError(cudaEventDestroy(start));
    handleError(cudaEventDestroy(stop));

    // Do some final staff (prepare nodes data to copy).
    handleError(cudaFree(problem.map));
    if (problem.type == PROBLEM_TYPE_PATHFINDING) {
        handleError(cudaFree(problem.pathfinding.map_unusual_nodes));
    }
    int path_count_max = problem.map_size / node_size(problem.type, problem.sliding.n);
    void *path, *dev_path;
    handleError(cudaMalloc((void**)&dev_path, problem.map_size));
    int *dev_path_count;
    handleError(cudaMalloc(&dev_path_count, sizeof(int)));
    gpu_astar_final<<<1, 1>>>(nodes_solutions, dev_path, dev_path_count, path_count_max);
    cudaDeviceSynchronize();

    // Copy results back to host memory.
    int path_count;
    cudaMemcpy(&path_count, dev_path_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Distance: %d\n", path_count-1);
    size_c path_size = node_size(problem.type, problem.sliding.n)*path_count;
    path = malloc(path_size);
    handleError(cudaMemcpy(path, dev_path, path_size, cudaMemcpyDeviceToHost));

    int status = write_file(output_file, problem, path_count, path, elapsedTime);

    // Free memory.
    handleError(cudaFree(problem.memory));
    handleError(cudaFree(problem.queues));
    handleError(cudaFree(nodes));
    handleError(cudaFree(nodes_solutions));
    handleError(cudaFree(dev_node_start));
    handleError(cudaFree(dev_node_dest));

    free(host_node_start);
    free(host_node_dest);

    return status;
}

