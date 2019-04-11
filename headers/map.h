typedef struct {
    Node[] nodes;
    int d;
    int size;
} Map;

int map_hash(Map* H, int j, Node *node);

Map* map_init(int d, int size);
void map_deduplicate(Map* H, Node[] nodes, Node[] nodes_dest, int n);

