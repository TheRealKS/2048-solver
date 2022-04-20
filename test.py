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

gridc = np.array([[ 2,    2,  32,  4],
                [  0,    0,  4,  4],
                [  0,    0,  0,  0],
                [  0,   0, 0,  2]])

#gridc = np.array(list(map(lambda r: list(reversed(r)), gridc)))
#print(gridc)

grid = Grid2048()
grid.cells = gridc
print(grid.getStateScore())