from grid import Grid2048
from util import generateRandomGrid
from move import Move

def printCells(cells):
    for row in cells:
        for tile in row:
            print(str(tile) + ", ", end='')
        print("")

g = Grid2048()
g.cells = [[ 2,  2,  4,  2],
            [ 16, 2,  32,  4],
            [ 2,  128, 2, 8],
            [ 0,  4, 32, 2]]
print(g.toIntArray())

g.performActionIfPossible(Move.UP)
g.addRandomTile()
print("")
print(g.toIntArray()) 