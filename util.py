"""Non tensorflow related functions"""

from heapq import merge
from unittest import case

from pandas import merge_asof
from grid import Grid2048
from move import Move
import numpy as np


def generateRandomGrid(initial_tiles = 2, strat = False):
    g = Grid2048(newtile_opt=[2,4], strategic=strat)
    for i in range (0,initial_tiles):
        g.addRandomTile()
    return g

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