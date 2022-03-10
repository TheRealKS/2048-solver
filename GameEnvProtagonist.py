from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from re import A
from acme.tf.networks import distributional

import numpy as np

from dm_env import specs
import dm_env
from sys import maxsize
from environment_spec import FeasibleEnvironment
from grid import Grid2048

from util import generateRandomGrid
from move import Move

class Game2048ProtagonistEnv(FeasibleEnvironment):
    """The main environment in which the game is played. For the protagonist"""

    def __init__(self, agent, initial_state : Grid2048 = None):
        super().__init__()

        self._initial_state = initial_state
        if (initial_state == None):
            #Generate a random environment.
            self._initial_state = generateRandomGrid()
        self._initial_state_grid = self._initial_state.cells
        self._state = self._initial_state
        
        self._episode_ended = False 
        self.prev_action = -1

        self._agent = agent
    
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

    def revert(self):
        self._state.cells = self._grid_revert.copy()

    
    """Single step function"""
    def step(self, action):
        if self._episode_ended:
            return self.reset()

        action = Move(action)

        prevstatescore = self._state.getStateScore()

        r = np.double(self._state.performActionIfPossible(action))
        newscore = self._state.getStateScore()
        if (prevstatescore > newscore):
            if (prevstatescore - newscore > 3):
                r = 0.0

        if (action == Move.DOWN or action == Move.RIGHT):
            r *= 2.0
        else:
            r = 0.0

        """         if (r == -1.0):
            #Move did nothing, so it doesn't matter. Penalise it
            print("PENALTY")
            if (self._state.addRandomTile()):
                return dm_env.transition(reward=0.0, observation=self._state.toFloatArray())
            else:
                if (self._state.movesAvailable()):
                    return dm_env.transition(reward=-1.0, observation=self._state.toFloatArray())
                print("terminate", self._state.toIntArray())
                return dm_env.termination(reward=0.0, observation=self._state.toFloatArray()) """
        #else:
        if (self._state.addRandomTile()):
            if (self._state.isWellOrdered()):
                return dm_env.transition(reward=r*2.0, observation=self._state.toFloatArray())
            else:
                return dm_env.transition(reward=r, observation=self._state.toFloatArray())
        else:
            print("termination2")
            print(self._state.toIntArray())
            return dm_env.termination(reward=0.0, observation=self._state.toFloatArray())

    def simple_step(self, action):
        r = self._state.performActionIfPossible(Move(action))
        if (self._state.addRandomTile()):
            return dm_env.transition(reward=r, observation=self._state.toFloatArray())
        else:
            return dm_env.termination(reward=r, observation=self._state.toFloatArray())

    def getState(self):
        return self._state.toFloatArray()

    @staticmethod
    def obtainFeasibleMoves(state):
        return Grid2048.movesAvailableInState(state)