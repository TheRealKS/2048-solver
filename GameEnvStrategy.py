from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from acme.tf.networks import distributional

import numpy as np

from dm_env import specs
import dm_env
from sys import maxsize
from grid import Grid2048

from util import generateRandomGrid
from move import Move

class Game2048StratEnv(dm_env.Environment):
    """The main environment in which the game is played."""

    def __init__(self, initial_state : Grid2048 = None):
        super().__init__()

        self._initial_state = initial_state
        if (initial_state == None):
            #Generate a random environment.
            self._initial_state = generateRandomGrid()
        self._initial_state_grid = self._initial_state.cells
        self._state = self._initial_state
        
        self._episode_ended = False 
    
    def action_spec(self):
        return specs.DiscreteArray(dtype=np.int32, num_values=4, name='action')
    
    def observation_spec(self): 
        return specs.BoundedArray(shape=self._state.shape(), dtype=np.float32, minimum=-1.0, maximum=float(maxsize), name='board')
    
    def reward_spec(self):
        return specs.BoundedArray(dtype=np.double, shape=(), name='reward', minimum=0, maximum=np.double(maxsize))

    def reset(self):
        self._episode_ended = False
        self._state.cells = self._initial_state_grid.copy()
        return dm_env.restart(self._state.toFloatArray())
    
    """Single step function"""
    def step(self, action):
        if self._episode_ended:
            return self.reset()

        move = Move(action)
        r = np.double(self._state.performActionIfPossible(move))
        # Check if there is an empty tile and add a random tile if that is possible
        if (self._state.addRandomTile()):
            #print("r=" + str(r))
            #Disincentivise moves that are not part of the strategy
            if (move == Move.DOWN or move == Move.LEFT):
                return dm_env.transition(reward=r, observation=self._state.toFloatArray())
            else:
                return dm_env.transition(reward=0.0, observation=self._state.toFloatArray())
        # Couldn't add a random tile; check if it is possible to merge tiles
        elif (self._state.movesAvailable()):
            #print("r=1")
            return dm_env.transition(reward=0.1, observation=self._state.toFloatArray())
        # Game over
        else:
            #print("r=0")
            self._episode_ended = True
            return dm_env.termination(reward=0.0, observation=self._state.toFloatArray())

    def render(self):
        return self._state.toIntArray()