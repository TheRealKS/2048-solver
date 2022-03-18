import numpy as np
from grid import Grid2048
from shield import criticalSectionValueAfterMove, findCriticalSection
from util import generateRandomGrid
from move import Move

def printCells(cells):
    for row in cells:
        for tile in row:
            print(str(tile) + ", ", end='')
        print("")

def arr_eq(a, b):
    for i in range(0, len(a)):
        #print(a[i])
        #print(b[i])
        if (a[i] != b[i]):
            return False

    return True

grid = np.array([[ 16,   0,  0,  0],
                [  32,   4,  0,  0],
                [  64,  16,  2,  4],
                [  512, 64,  8,  2]])

r, c = findCriticalSection(grid)
print(r,c)
print(criticalSectionValueAfterMove(grid, Move.UP, r, c))
print(criticalSectionValueAfterMove(grid, Move.RIGHT, r, c))