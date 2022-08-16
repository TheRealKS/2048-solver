"""Non tensorflow related functions"""

from grid import Grid2048
import numpy as np

"""Generate a random grid configuration"""
def generateRandomGrid(initial_tiles = 2, strat = False):
    g = Grid2048(newtile_opt=[2,4], strategic=strat)
    for i in range (0,initial_tiles):
        g.addRandomTile()
    return g

"""Return the sum of mergeable tile values. Input should be an array of tuples of mergeable tiles, as returned by the movesAvailableInDirection method"""
def getMergeableTileValues(mergeabletiles, grid):
    realm = list(filter(lambda r: r[0] >= 0 and r[1] >= 0, mergeabletiles))
    values = [0,0,0,0,0,0]
    for tile in realm:
        val = grid[tile[0],tile[1]]
        if (val == 32):
            values[0] += 1
        if (val == 64):
            values[1] += 1
        if (val == 128):
            values[2] += 1
        if (val == 256):
            values[3] += 1
        if (val == 512):
            values[4] += 1
        if (val == 1024):
            values[5] += 1
    return np.array(values)