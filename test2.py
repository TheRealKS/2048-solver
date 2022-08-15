import itertools
import operator
import numpy as np
from grid import Grid2048


gridc = np.array([[ 0,   0,   0,  0],
                  [  0,  0,  0,   4],
                  [  0,  2, 8,  4],
                  [  0,  128,   128,   16]])


def monotone_increasing(lst):
    pairs = zip(lst, lst[1:])
    return all(itertools.starmap(operator.le, pairs))

def monotone_decreasing(lst):
    pairs = zip(lst, lst[1:])
    return all(itertools.starmap(operator.ge, pairs))


grid = Grid2048()
grid.cells = gridc

gridc += 1

print(gridc)


h = np.count_nonzero(list(map(monotone_increasing, gridc)))
v = np.count_nonzero(list(map(monotone_increasing, gridc.T)))
print(h)
print(v)