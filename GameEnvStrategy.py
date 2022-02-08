from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dis import disco
from acme.tf.networks import distributional

import numpy as np

from dm_env import specs
import dm_env
from sys import maxsize
from grid import Grid2048
from stratmove import StrategicMove

from util import generateRandomGrid
from move import Move

class Game2048StratEnv(dm_env.DoubleEnvironment):
    """The main environment in which the game is played."""

    def __init__(self, initial_state : Grid2048 = None):
        super().__init__()

        self._initial_state = initial_state
        if (initial_state == None):
            #Generate a random environment.
            self._initial_state = generateRandomGrid(strat=True)
        self._initial_state_grid = self._initial_state.cells
        self._state = self._initial_state
        
        self._episode_ended = False 
    
    def action_spec(self):
        return specs.DiscreteArray(dtype=np.int32, num_values=2, name='action')
    
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

        #action = action.argmax()
        move = StrategicMove(action)
        r = np.double(self._state.performActionIfPossible(move))

        if (r > -1.0):
            #We are actually doing stuff thats good. We cannot add a tile if a move was not possible
            if (self._state.addRandomTile()):
                return dm_env.transition(reward=r, observation=self._state.toFloatArray())

        available_moves = self._state.movesAvailableInDirection()
        if (len(available_moves) > 0):
            if (('h', -1) in available_moves or ('v', 1) in available_moves):
                print("available")
                return dm_env.transition(reward=-5.0, observation=self._state.toFloatArray(), discount=1.0)

            savePossible = False
            for a in available_moves:
                if (a == ("h",1) or a == ("v",-1)):
                    savePossible = True
            
            if (savePossible):
                self._current_agent = 1
                return dm_env.handoff(reward=-10.0, observation=self._state.toFloatArray(), discount=1.0)
            
        # Game over
        else:
            #print("r=0")
            self._episode_ended = True
            return dm_env.termination(reward=0.0, observation=self._state.toFloatArray())

    def guardian(self, move):
        if (move == ["h",1]):
            return Move.RIGHT
        return Move.UP

    def render(self):
        return self._state.toIntArray()