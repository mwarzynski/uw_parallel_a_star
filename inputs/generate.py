from random import randint

def modify(board, x, y, n, last_x, last_y):
    direction = randint(0, 3)
    modified = False
    while not modified:
        direction = (direction + 1) % 4
        if direction == 0: # UP
            if y <= 0:
                continue
            if last_y == y - 1:
                continue
            board[x*n + y], board[x*n + (y-1)] = board[x*n + (y-1)], board[x*n + y]
            y -= 1
        elif direction == 1: # RIGHT
            if x >= n - 1:
                continue
            if last_x == x + 1:
                continue
            board[x*n + y], board[(x+1)*n + y] = board[(x+1)*n + y], board[x*n + y]
            x += 1
        elif direction == 2: # DOWN
            if y >= n - 1:
                continue
            if last_y == y + 1:
                continue
            board[x*n + y], board[x*n + (y+1)] = board[x*n + (y+1)], board[x*n + y]
            y += 1
        else: # direction == 3 # LEFT
            if x <= 0:
                continue
            if last_x == x - 1:
                continue
            board[x*n + y], board[(x-1)*n + y] = board[(x-1)*n + y], board[x*n + y]
            x -= 1
        modified = True
    return x, y


if __name__ == "__main__":
    n = 5

    board = [i for i in range(0, n*n)]
    x = 0
    y = 0

    last_x = -1
    last_y = -1

    print(",".join([str(i) if i != 0 else '_' for i in board]))
    for _ in range(0, 55):
        nx, ny = modify(board, x, y, n, last_x, last_y)
        last_x = x
        last_y = y
        x = nx
        y = ny
    print(",".join([str(i) if i != 0 else '_' for i in board]))