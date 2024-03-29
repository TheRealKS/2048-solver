"""A Driver that steps a python environment using a python policy. Safety is strictly enforced"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fcntl import F_SEAL_SEAL
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from tf_agents.drivers import driver, py_driver
from tf_agents.environments import py_environment
from tf_agents.policies import py_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.agents.dqn import dqn_agent


from tf_agents.typing import types
from move import Move

from shieldenvironment import ShieldedEnvironment


class SafeDriver(driver.Driver):
  """A driver that runs a python policy in a python environment. Safety is strictly enforced"""

  def __init__(
      self,
      env: ShieldedEnvironment,
      policy: py_policy.PyPolicy,
      observers: Sequence[Callable[[trajectory.Trajectory], Any]],
      transition_observers: Optional[Sequence[Callable[[trajectory.Transition],
                                                       Any]]] = None,
      max_steps: Optional[types.Int] = None,
      max_episodes: Optional[types.Int] = None,
      safe_moves : Optional[List] = [1,3]):
    """A driver that runs a python policy in a python environment.

    **Note** about bias when using batched environments with `max_episodes`:
    When using `max_episodes != None`, a `run` step "finishes" when
    `max_episodes` have been completely collected (hit a boundary).
    When used in conjunction with environments that have variable-length
    episodes, this skews the distribution of collected episodes' lengths:
    short episodes are seen more frequently than long ones.
    As a result, running an `env` of `N > 1` batched environments
    with `max_episodes >= 1` is not the same as running an env with `1`
    environment with `max_episodes >= 1`.

    Args:
      env: A py_environment.Base environment.
      policy: A py_policy.PyPolicy policy.
      observers: A list of observers that are notified after every step
        in the environment. Each observer is a callable(trajectory.Trajectory).
      transition_observers: A list of observers that are updated after every
        step in the environment. Each observer is a callable((TimeStep,
        PolicyStep, NextTimeStep)). The transition is shaped just as
        trajectories are for regular observers.
      max_steps: Optional maximum number of steps for each run() call. For
        batched or parallel environments, this is the maximum total number of
        steps summed across all environments. Also see below.  Default: 0.
      max_episodes: Optional maximum number of episodes for each run() call. For
        batched or parallel environments, this is the maximum total number of
        episodes summed across all environments. At least one of max_steps or
        max_episodes must be provided. If both are set, run() terminates when at
        least one of the conditions is
        satisfied.  Default: 0.

    Raises:
      ValueError: If both max_steps and max_episodes are None.
    """
    max_steps = max_steps or 0
    max_episodes = max_episodes or 0
    if max_steps < 1 and max_episodes < 1:
      raise ValueError(
          'Either `max_steps` or `max_episodes` should be greater than 0.')

    super(SafeDriver, self).__init__(env, policy, observers, transition_observers)
    self._max_steps = max_steps or np.inf
    self._max_episodes = max_episodes or np.inf
    self._env = env
    self.safe_moves = safe_moves

  def run(
      self,
      time_step: ts.TimeStep,
      policy_state: types.NestedArray = ()
  ) -> Tuple[ts.TimeStep, types.NestedArray]:
    """Run policy in environment given initial time_step and policy_state.

    Args:
      time_step: The initial time_step.
      policy_state: The initial policy_state.

    Returns:
      A tuple (final time_step, final policy_state).
    """
    num_steps = 0
    num_episodes = 0
    buffer = []
    env_state_prior = self._env.get_state()
    while num_steps < self._max_steps and num_episodes < self._max_episodes:
      # For now we reset the policy_state for non batched envs.
      if not self.env.batched and time_step.is_first() and num_episodes > 0:
        policy_state = self._policy.get_initial_state(self.env.batch_size or 1)

      action_step = self.policy.action(time_step, policy_state)

      move = Move(action_step.action)
      if (move == Move.UP or move == Move.RIGHT):
        if (time_step.observation['legal_moves'][1]):
          action_step = action_step._replace(action=np.array(1, dtype=np.int32))
        elif (time_step.observation['legal_moves'][3]):
          action_step = action_step._replace(action=np.array(3, dtype=np.int32))
      if (move == Move.UP):
        if (time_step.observation['legal_moves'][2]):
          action_step = action_step._replace(action=np.array(2, dtype=np.int32))

      next_time_step : ts.TimeStep = self.env._step(action_step.action)

      action_step_with_previous_state = action_step._replace(state=policy_state)
      traj = trajectory.from_transition(time_step, action_step_with_previous_state, next_time_step)

      for observer in self._transition_observers:
        observer((time_step, action_step_with_previous_state, next_time_step))
      for observer in self.observers:
        observer(traj)

      num_episodes += np.sum(traj.is_boundary())
      num_steps += np.sum(~traj.is_boundary())

      time_step = next_time_step
      policy_state = action_step.state

    return time_step, policy_state, num_steps
