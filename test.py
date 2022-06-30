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

gridc = np.array([[  8,   0,   0,   0],
 [ 16,  2,  4,   2],
 [  64,  16, 2,  2],
 [  512,   64,   32,   4]])


#gridc = np.array(list(map(lambda r: list(reversed(r)), gridc)))
#print(gridc)

grid = Grid2048()
grid.cells = gridc
print(grid.getStateScore() * 0.08)
print(grid.performActionIfPossible(Move.DOWN))
print(grid.cells)
print(grid.getStateScore() * 0.08)
print(grid.movesAvailableInDirection())
grid.cells = gridc
print(grid.performActionIfPossible(Move.RIGHT))
print(grid.cells)
print(grid.getStateScore() * 0.08)
print(grid.movesAvailableInDirection())