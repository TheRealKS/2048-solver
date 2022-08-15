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

def parseMergeableTiles(tiles):
    if (len(tiles) == 0):
        return 0
    indices = tuple(zip(*tiles))
    new_set = grid.cells[indices]
    return new_set.sum()

gridc = np.array([[ 4, 0, 0, 2],
                  [  32,  8,  0,   2],
                  [  64, 32, 16,  8],
                  [  256,  4,   64,  16]])
"""
gridc = np.array([[  0,   0,   0,  0],
                  [ 0,  0,  0,  0],
                  [  0, 0, 2,  2],
                  [  8,  4,  8,  8]])
"""

grid = Grid2048()
grid.cells = gridc

print("---LEFT---")
r = grid.performActionIfPossible(Move.DOWN)
print("Reward: " + str(r))
grid.addRandomTile()
legal_moves, m = grid.movesAvailableInDirection()
newscore, hscore, vscore = grid.getStateScore()
opp_left = (hscore, parseMergeableTiles(m))

print("Legal moves: ", end='')
print(legal_moves)
print("Mergeable tiles: ", end='')
print(m, end='')
print(" (" + str(len(m) / 2) + " pairs)")
print("Mergeable score: ", end='')
print(parseMergeableTiles(m))
print("Scores: t=" + str(newscore) + ",h=" + str(hscore) + ",v=" + str(vscore))
print(grid.cells)
print("----------\n")

grid.cells = gridc

print("---RIGHT---")
r = grid.performActionIfPossible(Move.RIGHT)
print("Reward: " + str(r))
grid.addRandomTile()
legal_moves, m = grid.movesAvailableInDirection()
newscore, hscore, vscore = grid.getStateScore()
opp_right = (hscore, parseMergeableTiles(m))

print("Legal moves: ", end='')
print(legal_moves)
print("Mergeable tiles: ", end='')
print(m, end='')
print(" (" + str(len(m) / 2) + " pairs)")
print("Mergeable score: ", end='')
print(parseMergeableTiles(m))
print("Scores: t=" + str(newscore) + ",h=" + str(hscore) + ",v=" + str(vscore))
print(grid.cells)
print("-----------\n")

diff_score = opp_left[0] - opp_right[0]
if (opp_left[0] >= opp_right[0] and opp_right[1] == opp_left[1]):
    print("LEFT IS BETTER")
else:
    zero_rows = len(list(filter(lambda r: r == 0, (map(np.count_nonzero, gridc)))))
    if (opp_right[0] - zero_rows > 0 and grid.isRowLocked(3) and opp_right[1] >= opp_left[1]):
        print("RIGHT IS BETTER")
    else:
        print("LEFT IS BETTER")