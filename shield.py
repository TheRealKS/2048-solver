import numpy as np

from grid import Grid2048
from move import Move


def findCriticalSection(state : np.ndarray):
    #Compute sums of rows and columns
    rowsum = state.sum(1)
    colsum = state.sum(0)
    #Argmax
    return np.argmax(rowsum), np.argmax(colsum)

def criticalSectionValue(state : np.ndarray, crow, ccol):
    return state[crow].sum() #+ state[:, ccol].sum() - state[crow][ccol]

def criticalSectionValueAfterMove(state : np.ndarray, move : Move, crow, ccol):
    board = Grid2048.transform_board(state, move, True)
    for i in range(board.shape[0]):
        row = board[i]
        non_zero = list(row[row != 0])
        j = 0
        while j < len(non_zero) - 1:
            if non_zero[j] == non_zero[j + 1]:
                non_zero[j] += non_zero[j + 1]
                del non_zero[j + 1]
            j += 1
        row = non_zero + [0]*(4 - len(non_zero))
        board[i, :] = row
    
    board = Grid2048.transform_board(board, move, False)

    return criticalSectionValue(board, crow, ccol)

def findOptimalMove(state : np.ndarray):
    r, c = findCriticalSection(state)
    rightval = criticalSectionValueAfterMove(state.copy(), Move.RIGHT, r, c)
    upval = criticalSectionValueAfterMove(state, Move.UP, r, c)

    if (rightval > upval):
        return Move.RIGHT
    else:
        return Move.UP