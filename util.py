"""Non tensorflow related functions"""

from grid import Grid2048
from move import Move


def generateRandomGrid(initial_tiles = 2, strat = False):
    g = Grid2048(newtile_opt=[2,4], strategic=strat)
    for i in range (0,initial_tiles):
        g.addRandomTile()
    return g
