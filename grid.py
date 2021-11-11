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
        cells = []
        for i in range(0,self._size):
            row = []
            for j in range(0,self._size):
                row.append(None)
            cells.append(row)
        return cells
        
    #Is there a tile available?
    def tilesAvailable(self):
        for i in range(0,self._size):
            for j in range(0,self._size):
                if (self.cells[i][j] == None):
                    return True
        return False

    def randomAvailableTile(self):
        tile = (-1,-1,-1)
        c = 0
        while tile[2] != None and c != (self._size ** 2):
            x = randint(0,self._size-1)
            y = randint(0,self._size-1)
            tile = (x,y,self.cells[x][y])
            c +=1 

        return tile

    #Add random tile to the grid
    def addRandomTile(self):
        tile = self.tilesAvailable()
        if (tile == False):
            #Game over
            return False
        
        tilevalue = choice(self._newtile_opt)
        tilepos = self.randomAvailableTile()
        self.cells[tilepos[0]][tilepos[1]] = tilevalue
        return True
        
    def performActionIfPossible(self, action):

        if (action.name in Move.__members__):
            
            #Basically slice the grid in the direction of the move and merge the slices
            newgrid = self.toAbstraction()
            score_increase = 0

            if (action == Move.UP):
                slices = self.getHorizontalSlices(newgrid)
                merge = 0
                for i in range(0, self._size):
                    sl = list(slices[i])
                    j = self._size - 1
                    while (j > 0):
                        a = sl[j]
                        b = sl[j-1]
                        if (b == None):
                            sl[j-1] = a
                            sl[j] = None
                            merge += 1
                        elif (a == b and a != None and merge < self._size / 2):
                            sl[j-1] = b * 2
                            score_increase += sl[j-1]
                            sl[j]= None
                            j = self._size - 1
                            merge += 1
                        j -= 1
                            
                    slices[i] = tuple(sl)
                self.cells = self.convertHorizontalSlicesToGrid(slices)
            elif (action == Move.DOWN):
                slices = self.getHorizontalSlices(newgrid)
                merge = 0
                for i in range(0, self._size):
                    sl = list(slices[i])
                    j = 0
                    while (j < self._size - 1):
                        a = sl[j]
                        b = sl[j+1]
                        if (b == None):
                            sl[j+1] = a
                            sl[j] = None
                            merge += 1
                        elif (a == b and a != None and merge < self._size / 2):
                            sl[j+1] = b * 2
                            score_increase += sl[j+1]
                            sl[j]= None
                            j = 0
                            merge += 1
                        j += 1
                            
                    slices[i] = tuple(sl)
                self.cells = self.convertHorizontalSlicesToGrid(slices)
            elif (action == Move.RIGHT):
                slices = newgrid
                merge = 0
                for i in range(0, self._size):
                    sl = list(slices[i])
                    j = 0
                    while (j < self._size - 1):
                        a = sl[j]
                        b = sl[j+1]
                        if (b == None):
                            sl[j+1] = a
                            sl[j] = None
                            merge += 1
                        elif (a == b and a != None and merge < self._size / 2):
                            sl[j+1] = b * 2
                            score_increase += sl[j+1]
                            sl[j]= None
                            j = 0
                            merge += 1
                        j += 1
                            
                    slices[i] = sl
                self.cells = slices
            elif (action == Move.LEFT):
                slices = newgrid
                merge = 0
                for i in range(0, self._size):
                    sl = list(slices[i])
                    j = self._size - 1
                    while (j > 0):
                        a = sl[j]
                        b = sl[j-1]
                        if (b == None):
                            sl[j-1] = a
                            sl[j] = None
                            merge += 1
                        elif (a == b and a != None and merge < self._size / 2):
                            sl[j-1] = b * 2
                            score_increase += sl[j-1]
                            sl[j]= None
                            j = self._size - 1
                            merge += 1
                        j -= 1
                            
                    slices[i] = sl
                self.cells = slices
            
            return score_increase
        else:
            raise ValueError("Action invalid")

    def placeNewTiles(self):
        for i in range(0,2):
            self.addRandomTile()

    def getHorizontalSlices(self, grid):
        return list(zip(*grid))

    def convertHorizontalSlicesToGrid(self, grid):
        return [list(a) for a in list(zip(*grid))]

    def toIntArray(self):
        t = self.cells.copy()
        for i in range(0, self._size):
            for j in range(0, self._size):
                if (t[i][j] == None):
                    t[i][j] = -1
        
        return np.array(t)

    def toAbstraction(self):
        t = self.cells.copy()
        for i in range (0, self._size):
            for j in range(0, self._size):
                if (t[i][j] == -1):
                    t[i][j] = None
        return t