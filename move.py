from enum import Enum

"""
Move enum
"""
class Move(Enum):
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3
    NOTHING = 4

def toMoveEnum(t):
    if (t[0] == 'v'):
        if (t[1] == -1):
            return Move.UP
        else:
            return Move.DOWN
    else:
        if (t[1] == -1):
            return Move.LEFT
        else:
            return Move.RIGHT