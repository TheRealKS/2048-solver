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


class ShieldDriver():
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
    
    for i in range(0, len(traj) - 1):
      transition = traj[i]
      if (transition.action in safe_moves and i > 15):
        new_action = np.random.choice(self.first_actions, p=[0.2, 0.8])
        env.set_state(transition[1]['observation'])
        r1 = env._step(new_action)
        reward_new = r1.reward / r1.observation['observation'].mean()
        reward_old = transition.reward / transition[1]['observation'].mean()
        if (reward_new == 0 and reward_old == 0):
          if (r1.observation['mergeable'] > transition[1]['mergeable']):
            #replace
              traj[i].replace(action = np.array(new_action))
              traj[i].replace(reward = np.array(r1))
              traj[i].replace(discount = np.array(0.2))
        elif (reward_new - reward_old >= 2.0):
              traj[i].replace(action = np.array(new_action))
              traj[i].replace(reward = np.array(r1))
              traj[i].replace(discount = np.array(0.2))
        # if (r1.reward > transition.reward):
        #   traj[i].replace(action = np.array(new_action))
        #   traj[i].replace(reward = np.array(r1))
        #   traj[i].replace(discount = np.array(0.5))
        # elif (transition[1]['mergeable'] == 0 and r1.observation['mergeable'] > 0):
        #   traj[i].replace(action = np.array(new_action))
        #   traj[i].replace(reward = np.array(r1))
        #   traj[i].replace(discount = np.array(0.5))
        # elif (transition[1]['mergeable'] - r1.observation['mergeable'] >= 1):
        #   traj[i].replace(action = np.array(new_action))
        #   traj[i].replace(reward = np.array(r1))
        #   traj[i].replace(discount = np.array(0.5))

    return traj
