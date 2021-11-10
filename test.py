from grid import Grid2048
from util import generateRandomGrid
from move import Move

def printCells(cells):
    for row in cells:
        for tile in row:
            print(str(tile) + ", ", end='')
        print("")

g = Grid2048()
g.cells[0][0] = None
g.cells[1][0] = 2
g.cells[2][0] = None
g.cells[3][0] = 2
printCells(g.cells)
print("")
g.performActionIfPossible(Move.RIGHT)
printCells(g.cells)