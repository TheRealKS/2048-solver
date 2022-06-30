from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import math
from typing import Tuple
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

from grid import Grid2048
from move import Move
from shield import findOptimalMove
from shieldenvironment import ShieldedEnvironment
from util import generateRandomGrid


class Game2048PyEnv(ShieldedEnvironment):

  def __init__(self, initial_state : Grid2048 = None):
    super().__init__()

    self._initial_state = initial_state
    if (initial_state == None):
        #Generate a random environment.
        self._initial_state = generateRandomGrid()
    self._initial_state_grid = self._initial_state.cells
    self._state = self._initial_state
        
    self._episode_ended = False 

    self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
    self._observation_spec = {
        'observation': array_spec.BoundedArraySpec(shape=self._state.shape(), dtype=np.float32, minimum=-1.0, maximum=float('inf'), name='board'),
        'legal_moves': array_spec.ArraySpec(shape=(4,), dtype=np.bool_, name='legal'),
        'new_tile': array_spec.ArraySpec(shape=(2,), dtype=np.int32, name='new_tile'),
        'mergeable': array_spec.ArraySpec(shape=(), dtype=np.int32, name='mergeable'),
        'strategySwitchSuitable': array_spec.ArraySpec(shape=(), dtype=np.bool_, name='strategySwitchSuitable')
    }

    self._previous_state : Grid2048 = None

  def action_spec(self):    
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._episode_ended = False
    self._state.cells = self._initial_state_grid.copy()
    legal, _ = self._state.movesAvailableInDirection()
    legal_moves = self.parseLegalMoves(legal)
    returnspec = {
        'observation': self._state.toFloatArray(),
        'legal_moves': legal_moves,
        'new_tile': np.array([-1,-1], dtype=np.int32),
        'mergeable': np.array(0, dtype=np.int32),
        'strategySwitchSuitable': np.array(False, dtype=np.bool_)
    } 
    return ts.restart(returnspec)

  def _step(self, action, save = False):
    if self._episode_ended:
      return self.reset()

    action = Move(action)

    if (save):
      self._previous_state = self._state.copy()

    prevstatescore = self._state.getStateScore()

    r = np.double(self._state.performActionIfPossible(action))
    newscore, hscore, vscore = self._state.getStateScore()

    t, pos = self._state.addRandomTile()
    poss = [pos[0],pos[1]]
    
    legal_moves, m = self._state.movesAvailableInDirection()
    legal_moves = self.parseLegalMoves(legal_moves)
    returnspec = {
        'observation': self._state.toFloatArray(),
        'legal_moves': legal_moves,
        'new_tile': np.array(poss, dtype=np.int32),
        'mergeable': np.array(self.parseMergeableTiles(m), dtype=np.int32),
        'strategySwitchSuitable': np.array(self._state.isRowLocked(3), dtype=np.bool_)
    }
    
    if (returnspec['strategySwitchSuitable'] and action != Move.UP):
      r *= (vscore + 4)
    else:
      r *= newscore

    if (t):
        return ts.transition(returnspec, reward=r)
    else:
        #print('termination')
        #print(self._state.toIntArray())
        return ts.termination(returnspec, reward=0.0)

  def get_state(self):
    return self._state.copy()
  
  def set_state(self, state) -> None:
    g = Grid2048()
    g.cells = state
    self._state = g

  def getAbsoluteScore(self):
      return 

  def revert(self):
    self.cells = self._previous_state.cells

  def parseLegalMoves(self, legalmoves):
    new_legal_moves = np.full(4, -float('inf'))
    for move in legalmoves:
        new_legal_moves[move.value] = 0
    
    return np.logical_not(new_legal_moves)

  def parseMergeableTiles(self, tiles):
    indices = tuple(zip(*tiles))
    new_set = self._state.cells[indices]
    return new_set.sum()