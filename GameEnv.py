from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from acme.tf.networks import distributional

import numpy as np

from dm_env import specs
import dm_env
from sys import maxsize

from util import generateRandomGrid
from move import Move

class Game2048Env(dm_env.Environment):
    """The main environment in which the game is played."""

    def __init__(self, initial_state = None):
        super().__init__()

        self._initial_state = initial_state
        if (initial_state == None):
            #Generate a random environment.
            self._initial_state = generateRandomGrid()
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
        self._state = self._initial_state
        return dm_env.restart(self._state.toFloatArray())
    
    def step(self, action):
        if self._episode_ended:
            return self.reset()
        
        r = np.double(self._state.performActionIfPossible(Move(action)))
        if (self._state.addRandomTile()):
            return dm_env.transition(reward=r, observation=self._state.toFloatArray())
        else:
            self._episode_ended = True
            return dm_env.termination(reward=0.0, observation=self._state.toFloatArray())

    def render(self):
        return self._state.toIntArray()