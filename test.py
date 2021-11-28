from grid import Grid2048
from util import generateRandomGrid
from move import Move

def printCells(cells):
    for row in cells:
        for tile in row:
            print(str(tile) + ", ", end='')
        print("")

g = Grid2048()
g.cells = [[ 2,  4,  8,  2],
            [ 8, 16,  2,  4],
            [ 4,  8, 16, 32],
            [ 2,  4,  8, 32]]

print(g.movesAvailable())