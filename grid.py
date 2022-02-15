"""Wrapper for the 2048 grid"""


from random import choice
from typing_extensions import Annotated
import numpy as np

from move import Move, toMoveEnum
from stratmove import StrategicMove

class Grid2048():
    
    def __init__(self, size=4, newtile_opt=[2,4], strategic=False):
        self._size = size
        self._newtile_opt = newtile_opt

        #Build empty grid
        self.cells = self.buildEmptyGrid()

        self.strategic = strategic
    
    """Build empy grid with all zero values - this is not the same as a random grid"""
    def buildEmptyGrid(self):
        return np.zeros((4, 4), dtype='int')
        
    """Returns true if an empty tile is available"""
    def tilesAvailable(self):
        return np.any(self.cells == 0)

    """Get the coordinates for a random available tile"""
    def randomAvailableTile(self):
        tile = np.asarray(self.cells == 0).nonzero()
        tilelist = list(zip(tile[0],tile[1]))
        tileidx = choice(tilelist)

        return tileidx

    """Returns true if a move is available. I.e., all tiles are filed, but it is possible to merge two tiles"""
    def movesAvailable(self):
        return len(self.movesAvailableInDirection())> 0

    def movesAvailableInDirection(self):
        moves = set()
        for i in range(0,self._size):
            for j in range(0,self._size):
                for r in [-1, 1]:
                    if 0 <= i+r < self._size:
                        if (self.cells[i][j] == self.cells[i+r][j]) and self.cells[i][j] > 0:
                            moves.add(("v",r))
                        elif self.cells[i+r][j] == 0 and self.cells[i][j] != 0:
                            moves.add(("v",r))
                    if 0 <= j+r < self._size:
                        if (self.cells[i][j] == self.cells[i][j+r]) and self.cells[i][j] > 0:
                            moves.add(("h",r))
                        elif self.cells[i][j+r] == 0 and self.cells[i][j] != 0:
                            moves.add(("h", r))

        return set(map(lambda m : toMoveEnum(m), moves))
        
    """Add a random tile to the grid (2 or 4)"""
    def addRandomTile(self):
        if not self.tilesAvailable():
            return False
        
        tilevalue = np.random.choice(self._newtile_opt, p=[0.9,0.1])
        tilepos = self.randomAvailableTile()
        self.cells[tilepos[0]][tilepos[1]] = tilevalue
        return True
    
    """
    Perform a move if it is possible. If move is not possible, will do nothing. If move is possible, will return the sum of all the merges made in the move
    action must be a valid member of Move enum
    """
    def performActionIfPossible(self, action, savestate = True, override=False):

        if (action.name in Move.__members__):

            if (self.strategic and (not action.name in StrategicMove.__members__) and (not override)):
                return -2.0
            else:
                action = Move.__members__[action.name]

            #Save state to compare afterwards
            prev_state = self.cells.copy()

            sum_merges = 0
            board = self.toIntArray()
            board = self.transform_board(self.cells, action, True)
            for i in range(board.shape[0]):
                row = board[i]
                non_zero = list(row[row != 0])
                j = 0
                while j < len(non_zero) - 1:
                    if non_zero[j] == non_zero[j + 1]:
                        non_zero[j] += non_zero[j + 1]
                        sum_merges += non_zero[j]
                        del non_zero[j + 1]
                    j += 1
                row = non_zero + [0]*(4 - len(non_zero))
                board[i, :] = row

            if (savestate):
                self.cells = self.transform_board(board, action, False)

            #If the move did nothing, report this
            if (np.array_equiv(self.cells, prev_state)):
                return -1.0

            return sum_merges

        else:
            raise ValueError("Action invalid")

    """
    Transform the board to merge tiles in given direction.
    """
    @staticmethod
    def transform_board(board: np.array, direction : Move, forward: bool) -> np.array:
        board = np.array(board)
        if forward:
            if (direction == Move.UP or direction == Move.DOWN):
                board = board.T
            if (direction == Move.DOWN or direction == Move.RIGHT):
                board = board[:, ::-1]
        else:
            if (direction == Move.DOWN or direction == Move.RIGHT):
                board = board[:, ::-1]
            if (direction == Move.UP or direction == Move.DOWN):
                board = board.T
        return board    

    """
    Utility methods
    """

    def getStateScore(self):
        return max(1,len(np.asarray(self.cells==0)))

    def highestTile(self):
        return self.cells.max()

    def sumOfTiles(self):
        return np.array(self.toIntArray()).sum()

    def toIntArray(self):
        return self.cells

    def toFloatArray(self):
        return self.cells.astype(np.float32)

    def shape(self):
        return self.toIntArray().shape