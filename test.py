import numpy as np
from grid import Grid2048
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

grid = np.array([[  0,  2,   4,  8],
        [  0, 2,  4,  8],
        [  2,   2, 4,  8],
        [  0,   0,   0,   0]])

g = Grid2048()
g.cells = grid
print(g.getStateScore())