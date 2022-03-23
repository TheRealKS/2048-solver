from re import I
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

grid = np.array([[ 4,    0,  0,  0],
                [  8,    4,  0,  2],
                [  32,   4,  4,  0],
                [  128,  64,  32,  16]])

a = grid > 8
b = grid <= 64
c = np.bitwise_and(a, b)
print(np.count_nonzero(np.count_nonzero(c, axis=1) == 1))