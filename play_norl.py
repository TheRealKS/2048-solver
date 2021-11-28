from grid import Grid2048
from util import generateRandomGrid
from move import Move

game = generateRandomGrid()
print(game.toIntArray())
print("\nUP")

game.performActionIfPossible(Move.UP)
game.addRandomTile()
print(game.toIntArray())
print("\nDOWN")
game.performActionIfPossible(Move.DOWN)
game.addRandomTile()
print(game.toIntArray())
print("\nRIGHT")
game.performActionIfPossible(Move.RIGHT)
game.addRandomTile()
print(game.toIntArray())
print("\nLEFT")
game.performActionIfPossible(Move.LEFT)
game.addRandomTile()
print(game.toIntArray())
print(game.movesAvailable())