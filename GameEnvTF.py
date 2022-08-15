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
        'legal_moves': array_spec.ArraySpec(shape=(4,), dtype=np.int32, name='legal'),
        'new_tile': array_spec.ArraySpec(shape=(2,), dtype=np.int32, name='new_tile'),
        'mergeable': array_spec.ArraySpec(shape=(), dtype=np.int32, name='mergeable'),
        'strategySwitchSuitable': array_spec.ArraySpec(shape=(), dtype=np.int32, name='strategySwitchSuitable')
    }

    self._previous_state : Grid2048 = None
    self.episode_ln = 0

  def action_spec(self):    
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._episode_ended = False
    self._state.cells = self._initial_state_grid.copy()
    self._episode_ln = 0
    legal, _ = self._state.movesAvailableInDirection()
    legal_moves = self.parseLegalMoves(legal)
    returnspec = {
        'observation': self._state.toFloatArray(),
        'legal_moves': legal_moves,
        'new_tile': np.array([-1,-1], dtype=np.int32),
        'mergeable': np.array(0, dtype=np.int32),
        'strategySwitchSuitable': np.array(0, dtype=np.int32)
    } 
    return ts.restart(returnspec)

  def _step(self, action):
    if self._episode_ended:
      return self.reset()

    self.episode_ln += 1
    action = Move(action)
    #Collect some baseline metrics and backup state

    legal_moves_prev = self._state.movesAvailableInState()
    prev_state = self._state.copy()

    if (action == Move.RIGHT or action == Move.UP):
      if (Move.DOWN in legal_moves_prev):
        action = Move.DOWN
      elif (Move.LEFT in legal_moves_prev):
        action = Move.LEFT
    if (action == Move.UP):
      if (Move.RIGHT in legal_moves_prev):
        action = Move.RIGHT

    #Issue naive move and collect metrics for it

    r = np.double(self._state.performActionIfPossible(action))

    t, pos = self._state.addRandomTile()
    poss = [pos[0],pos[1]]
    
    legal_moves, m = self._state.movesAvailableInDirection()
    if (Move.UP in legal_moves and len(legal_moves) > 1):
      legal_moves.remove(Move.UP)
    if (Move.RIGHT in legal_moves and len(legal_moves) > 1):
      legal_moves.remove(Move.RIGHT)
    
    parsedM = self.parseMergeableTiles(m, self._state.cells)
    newscore, hscore, vscore = self._state.getStateScore()

    new_naive_state = self._state.copy()

    #If this move led to termination, there's no use doing any work

    if (action in [Move.LEFT, Move.DOWN] and Move.RIGHT in legal_moves_prev and self.episode_ln >= 150):
      self._state = prev_state
      r_alt = np.double(self._state.performActionIfPossible(Move.RIGHT))
      t_alt, pos_alt = self._state.addRandomTile()
      poss_alt = [pos_alt[0], pos_alt[1]]
      if (t_alt and r_alt >= 0.0):
        legal_moves_alt, m_alt = self._state.movesAvailableInDirection()
        parsedM_alt = self.parseMergeableTiles(m_alt, self._state.cells)
        newscore_alt, hscore_alt, vscore_alt = self._state.getStateScore()

        #Now that we have all the metrics, we can compare the two
        if (hscore >= hscore_alt and parsedM_alt == parsedM):
          #SAFE IS BETTER
          pass
        else:
          zero_rows = self._state.getNumZeroRows()
          if (hscore_alt - zero_rows > 0 and self._state.isRowLocked(3) and parsedM_alt > parsedM):
            #RIGHT IS BETTER

            if (Move.UP in legal_moves_alt and len(legal_moves_alt) > 1):
                legal_moves_alt.remove(Move.UP)
            if (Move.RIGHT in legal_moves_alt and len(legal_moves_alt) > 1):
              legal_moves_alt.remove(Move.RIGHT)

            returnspec = {
                'observation': self._state.toFloatArray(),
                'legal_moves': self.parseLegalMoves(legal_moves_alt),
                'new_tile': np.array(poss_alt, dtype=np.int32),
                'mergeable': np.array(parsedM_alt, dtype=np.int32),
                'strategySwitchSuitable': np.array(2, dtype=np.int32)
            }

            return ts.transition(returnspec, reward=0.0, discount=0.0)

    self._state = new_naive_state

    returnspec = {
      'observation': new_naive_state.toFloatArray(),
      'legal_moves': self.parseLegalMoves(legal_moves),
      'new_tile': np.array(poss, dtype=np.int32),
      'mergeable': np.array(parsedM, dtype=np.int32),
      'strategySwitchSuitable': np.array(action.value, dtype=np.int32)
    }

    r *= newscore
    
    if (action == Move.LEFT or action == Move.DOWN):
      r *= 2.0

    if (t and len(legal_moves) > 0):
        return ts.transition(returnspec, reward=r)
    else:
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
    new_legal_moves = np.full(4, 0, dtype=np.int32)
    for move in legalmoves:
        new_legal_moves[move.value] = 1
    
    return new_legal_moves
    #return np.logical_not(new_legal_moves)

  def parseMergeableTiles(self, tiles, grid):
    if (len(tiles) == 0):
      return 0
    indices = tuple(zip(*tiles))
    new_set = grid[indices]
    return new_set.sum()

  def collect_metrics(self):
    pass