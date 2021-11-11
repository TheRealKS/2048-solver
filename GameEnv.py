from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tensorflow.python.types.core import Value

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from grid import Grid2048

from util import generateRandomGrid
from move import Move

class Game2048Env(py_environment.PyEnvironment):
    """The main environment in which the game is played."""

    def __init__(self, initial_state = None):
        super().__init__()

        self._initial_state = initial_state
        if (initial_state == None):
            #Generate a random environment.
            self._initial_state = generateRandomGrid()
        self._state = self._initial_state
        
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(16,), dtype=np.int32, name='obervation')
        self._episode_ended = False
    
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self): 
        return self._observation_spec
    
    def _reset(self):
        self._state = self._initial_state
        print(self._state)
        print(self._initial_state)
        self._episode_ended = False
        return ts.restart(np.array(self._state.toIntArray().flatten(), dtype=np.int32))
    
    def _step(self, action):
        if self._episode_ended:
            return self._reset()
        
        r = self._state.performActionIfPossible(Move(action))
        if (self._state.tilesAvailable()):
            self._state.placeNewTiles()
            return ts.transition(np.array(self._state.toIntArray().flatten(), dtype=np.int32), reward = r, discount=0.9)
        else:
            reward = -1
            self._episode_ended = True
            return ts.termination(np.array(self._state.toIntArray().flatten(), dtype=np.int32), reward)

    def render(self, mode):
        return self._state.toIntArray()