#include <stdio.h>

#include "map.h"
#include "queue.h"

#define TYPE_PUZZLE 1
#define TYPE_PATHFINDING 2

typedef struct {
    double g;
    double f;
    void *data;
} Node;

