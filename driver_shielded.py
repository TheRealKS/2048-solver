"""A Driver that steps a python environment using a python policy. A shield is provided and used to override actions."""

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

from shieldenvironment import ShieldedEnvironment


class ShieldedDriver(driver.Driver):
  """A driver that runs a python policy in a python environment."""

  def __init__(
      self,
      env: ShieldedEnvironment,
      shield: dqn_agent.DqnAgent,
      ds_iterator : Iterator,
      policy: py_policy.PyPolicy,
      observers: Sequence[Callable[[trajectory.Trajectory], Any]],
      transition_observers: Optional[Sequence[Callable[[trajectory.Transition],
                                                       Any]]] = None,
      max_steps: Optional[types.Int] = None,
      max_episodes: Optional[types.Int] = None,
      save_moves : Optional[List] = [1,3]):
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

    super(ShieldedDriver, self).__init__(env, policy, observers, transition_observers)
    self._max_steps = max_steps or np.inf
    self._max_episodes = max_episodes or np.inf
    self._env = env
    self.save_moves = save_moves
    self.shield = shield
    self.ds_iterator = ds_iterator

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
      copy_of_env = self._env.get_state().copy()
      next_time_step : ts.TimeStep = self.env._step(action_step.action, save = True)

      action_steps_to_observe = [(time_step, action_step, next_time_step)]

      #Now that we have ran the step, verify it.
      if (action_step.action in [1,3] and num_steps > 50 and False):
          self._env.set_state(copy_of_env.cells)
          new_action = 2 #np.random.choice(self.first_actions, p=[0.0, 1.0])
          r1 = self._env._step(new_action)
          r2 = self._env._step(1)

          better = self.getStateProduct(r2.observation['observation'])
          worse = self.getStateProduct(copy_of_env.cells)

          if (better <= worse and r2.observation['strategySwitchSuitable']):
              action_steps_to_observe = []
              action_step = action_step._replace(action = np.array(2, dtype=np.int32))
              action_steps_to_observe.append((time_step, action_step, r1))
              other_action_step = action_step._replace(action=np.array(1, dtype=np.int32))
              action_steps_to_observe.append((r1, other_action_step, r2))

      for (t,a,n) in action_steps_to_observe:
        action_step_with_previous_state = a._replace(state=policy_state)
        traj = trajectory.from_transition(
          t, action_step_with_previous_state, n)

        num_episodes += np.sum(traj.is_boundary())
        num_steps += np.sum(~traj.is_boundary())

        time_step = n
        policy_state = a.state

        for observer in self._transition_observers:
          observer((t, action_step_with_previous_state, n))
        for observer in self.observers:
          observer(traj)


    return time_step, policy_state

  def getStateProduct(self, state):
    flat = state.flatten()
    gzero = flat[flat > 0]
    return gzero.prod()

  def run_from_timestep(self, act, time_step: ts.TimeStep, traje : List[trajectory.Trajectory], policy_state: types.NestedArray = ()):
    #First, observe the previous timesteps
    for traj in traje:
      for observer in self.observers:
        observer(traj)

    #Observe the replaced timesteps
    action_step = self.policy.action(time_step, policy_state)
    action_step_with_previous_state = action_step._replace(state=policy_state, action=np.array(act,dtype=np.int32))
    last_observation = traje[len(traje)-1]
    prev_timestep = ts.transition(last_observation.observation, last_observation.reward)
    for observer in self.observers:
      observer(trajectory.from_transition(prev_timestep, action_step_with_previous_state, time_step))

    
    #Now, run the game from the given timestep
    self._env.set_state(time_step.observation['observation'])
    num_episodes = 0
    while num_episodes == 0:
      action_step = self.policy.action(time_step, policy_state)
      next_time_step : ts.TimeStep = self.env._step(action_step.action, save = True)

      # When using observer (for the purpose of training), only the previous
      # policy_state is useful. Therefore substitube it in the PolicyStep and
      # consume it w/ the observer.
      action_step_with_previous_state = action_step._replace(state=policy_state)
      traj = trajectory.from_transition(
          time_step, action_step_with_previous_state, next_time_step)

      num_episodes += np.sum(traj.is_boundary())

      time_step = next_time_step
      policy_state = action_step.state

      for observer in self.observers:
        observer(traj)
      
    return time_step, policy_state

  def verify_trajectory(self, traj: List[trajectory.Trajectory], safe_moves = [1,3]):
    #print("VERIFY")
    num_redone = 0
    for i in range(0, len(traj) - 1):
      #if i % 25 == 0:
        #print(str(i) + "/" + str(len(traj)))
      transition = traj[i]
      result = traj[i+1]
      if (transition.action in safe_moves and i > 50):
        self._env.set_state(transition[1]['observation'])
        new_action = 2 #np.random.choice(self.first_actions, p=[0.0, 1.0])
        r1 = self._env._step(new_action)
        r1 = self._env._step(1)

        better = r1.observation['observation'].flatten().prod()
        worse = result[1]['observation'].flatten().prod()

        if (better <= worse):
            prev_trajectory = traj[:i+1]
            self.run_from_timestep(2, trajectory.to_transition(transition[1]), prev_trajectory)
            num_redone += 1
    
    print("Replayed " + str(num_redone))
    return traj
