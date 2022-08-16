from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from grid import Grid2048
from move import Move
from util import generateRandomGrid


class Game2048EvalPyEnv(py_environment.PyEnvironment):

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
            'strategySwitchSuitable': np.array(0, dtype=np.int32)
        } 
        return ts.restart(returnspec)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        action = Move(action)

        r = np.double(self._state.performActionIfPossible(action))

        t, pos = self._state.addRandomTile()
        poss = [pos[0],pos[1]]
        
        legal_moves, m = self._state.movesAvailableInDirection()
        m_p = self.parseMergeableTiles(m, self._state.cells)
        legal_moves = self.parseLegalMoves(legal_moves)
        returnspec = {
            'observation': self._state.toFloatArray(),
            'legal_moves': legal_moves,
            'new_tile': np.array(poss, dtype=np.int32),
            'mergeable': np.array(m_p, dtype=np.int32),
            'strategySwitchSuitable': np.array(0, dtype=np.int32)
        }

        if (t):
            return ts.transition(returnspec, reward=r)
        else:
            return ts.termination(returnspec, reward=0.0)

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

    def get_state(self):
        return self._state.copy()
  
    def set_state(self, state):
        g = Grid2048()
        g.cells = state
        self._state = g