"""Wrapper for the 2048 grid"""


from random import choice, randint
import numpy as np

from tensorflow.python.types.core import Value
from move import Move

class Grid2048():
    
    def __init__(self, size=4, newtile_opt=[2,4]):
        self._size = size
        self._newtile_opt = newtile_opt

        #Build empty grid
        self.cells = self.buildEmptyGrid()
    
    def buildEmptyGrid(self):
        return np.zeros((4, 4), dtype='int')
        
    #Is there a tile available?
    def tilesAvailable(self):
        return np.any(self.cells == 0)

    def randomAvailableTile(self):
        tile = (-1,-1,-1)
        c = 0
        while tile[2] != 0 and c != (self._size ** 2):
            x = randint(0,self._size-1)
            y = randint(0,self._size-1)
            tile = (x,y,self.cells[x][y])
            c +=1 

        return tile
    
    def movesAvailable(self):
        for i in range(0,self._size):
            for j in range(0,self._size):
                for r in [-1, 1]:
                    if 0 <= i+r < self._size:
                        if self.cells[i][j] == self.cells[i+r][j]:
                            #print(str(i) + "," + str(j) + "=" + str(i+r) + "," + str(j))
                            return True
                    if 0 <= j+r < self._size:
                        if self.cells[i][j] == self.cells[i][j+r]:
                            #print(str(i) + "," + str(j) + "=" + str(i) + "," + str(j+r))
                            return True
        return False
        
    #Add random tile to the grid
    def addRandomTile(self):
        if not self.tilesAvailable():
            return False
        
        tilevalue = choice(self._newtile_opt)
        tilepos = self.randomAvailableTile()
        self.cells[tilepos[0]][tilepos[1]] = tilevalue
        return True
        
    def performActionIfPossible(self, action):

        if (action.name in Move.__members__):

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
            self.cells = self.transform_board(board, action, False)
            return sum_merges

        else:
            raise ValueError("Action invalid")

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

    def toIntArray(self):
        t = self.cells.copy()
        for i in range(0, self._size):
            for j in range(0, self._size):
                if (t[i][j] == None):
                    t[i][j] = 0
        
        return np.array(t)

    def toFloatArray(self):
        return self.cells.astype(np.float32)

    def toAbstraction(self):
        t = self.cells.copy()
        for i in range (0, self._size):
            for j in range(0, self._size):
                if (t[i][j] == -1):
                    t[i][j] = None
        return t

    def toFloatAbstraction(self):
        t = self.cells.copy()
        for i in range(0, self._size):
            for j in range(0, self._size):
                if (t[i][j] == None):
                    t[i][j] = -1.0
                else:
                    t[i][j] = float(t[i][j])
        return t

    def shape(self):
        return self.toIntArray().shape