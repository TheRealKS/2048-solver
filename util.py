"""Non tensorflow related functions"""

from grid import Grid2048


def generateRandomGrid(initial_tiles = 2):
    g = Grid2048(newtile_opt=[2])
    for i in range (0,initial_tiles):
        g.addRandomTile()
    return g
