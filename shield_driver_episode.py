"""A Driver that steps a python environment using a python policy. A shield is provided and used to override actions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from email import policy
import random
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from tf_agents.drivers import driver, py_driver
from tf_agents.environments import py_environment
from tf_agents.policies import py_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.agents.dqn import dqn_agent


from tf_agents.typing import types
from GameEnvTF import Game2048PyEnv

from ShieldEnvTF import Game2048ShieldPyEnv
from grid import Grid2048
from shieldenvironment import ShieldedEnvironment


class ShieldDriverEpisode():
  """A driver that runs a python policy in a python environment."""

  def __init__(
      self,
      env: Game2048ShieldPyEnv,
      policy: py_policy.PyPolicy,
      observers: Sequence[Callable[[trajectory.Trajectory], Any]],
      transition_observers: Optional[Sequence[Callable[[trajectory.Transition],
                                                       Any]]] = None,
      max_steps: Optional[types.Int] = 5,
      max_episodes: Optional[types.Int] = None,
      first_actions : Optional[List] = [0,2]):

    max_steps = max_steps or 0
    max_episodes = max_episodes or 0
    if max_steps < 1 and max_episodes < 1:
      raise ValueError(
          'Either `max_steps` or `max_episodes` should be greater than 0.')

    self._max_steps = max_steps or np.inf
    self._max_episodes = max_episodes or np.inf
    self.first_actions = first_actions
    self._env = env

  def verify_trajectory(self, traj: List[trajectory.Trajectory], env : ShieldedEnvironment = None, safe_moves = [1,3]):
    if (env == None):
      state = Grid2048()
      state.cells = traj[0][1]['observation']
      env = Game2048PyEnv(state)
    
    for i in range(0, len(traj) - 3):
      transition = traj[i]
      result = traj[i+1]
      if (transition.action in safe_moves and i > 50):
        env.set_state(transition[1]['observation'])
        state_score_before = env.get_state().getStateScore()
        sum_before = result.observation['observation'].sum()
        new_action_ = 2#np.random.choice(self.first_actions, p=[0.0, 1.0])
        r1 = []
        r1.append((new_action_, env._step(new_action_)))
        cum_reward_new = r1[0][1].reward
        for j in range(1, 3):
          new_action = np.random.choice(range(1,4))
          r1.append((new_action, env._step(new_action)))
          cum_reward_new += r1[j][1].reward
        
        cum_reward_orig = transition.reward
        for j in range(1, 3):
          cum_reward_orig += traj[i+j].reward

        state_score_after = env.get_state().getStateScore()
        if (env.get_state().sumOfTiles() > sum_before and cum_reward_new > cum_reward_orig) or (r1[2][1].observation['mergeable'] >= result.observation['mergeable']):
          for j in range(0,3):
            pred = traj[i + j]
            pred = pred.replace(action = np.array(r1[j][0], dtype=np.int32))
            pred = pred.replace(reward = np.array(r1[j][1].reward))
            #pred = pred.replace(discount = np.array(0.9, dtype=np.float32))
            traj[i + j] = pred
            pred = traj[i+j+1]
            pred = pred.replace(observation=r1[j][1].observation)
            traj[i+j+1] = pred

        else:
          transition[1]['legal_moves'][new_action_] = False
          traj[i] = traj[i].replace(observation=transition[1])

    return traj
