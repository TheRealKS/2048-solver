# python3

"""SHielded agent training loop"""

from asyncio import shield
from multiprocessing.dummy import current_process
import operator
import time
from typing import Optional

from acme import core
from acme.utils import counting
from acme.utils import loggers

import dm_env
from dm_env import specs
import numpy as np
import tree
from grid import Grid2048

from shieldenvironment import ShieldEnvironment


class ShieldEnvironmentLoop(core.Worker):
  """
  RL Loop with shielding

  This takes `Environment` and `Actor` instances and coordinates their
  interaction. Agent is updated if `should_update=True`. This can be used as:

    loop = EnvironmentLoop(environment, actor)
    loop.run(num_episodes)

  A `Counter` instance can optionally be given in order to maintain counts
  between different Acme components. If not given a local Counter will be
  created to maintain counts between calls to the `run` method.

  A `Logger` instance can also be passed in order to control the output of the
  loop. If not given a platform-specific default logger will be used as defined
  by utils.loggers.make_default_logger. A string `label` can be passed to easily
  change the label associated with the default logger; this is ignored if a
  `Logger` instance is given.
  """

  def __init__(
      self,
      play_environment: dm_env.Environment,
      shield_environment: ShieldEnvironment,
      protagonist: core.Actor,
      shield: core.Actor,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      should_update: bool = True,
      label: str = 'environment_loop',
  ):
    # Internalize agent and environment.
    self._play_environment = play_environment
    self._shield_environment = shield_environment
    self._protagonist = protagonist
    self._shield = shield
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(label)
    self._should_update = should_update
    self._current_agent = 0 # 0 = prot, 1 = antag

  def run_episode(self) -> loggers.LoggingData:
    """Run one episode.

    Each episode is a loop which interacts first with the environment to get an
    observation and then give that observation to the agent in order to retrieve
    an action.

    Returns:
      An instance of `loggers.LoggingData`.
    """
    # Reset any counts and start the environment.
    start_time = time.time()
    episode_steps = 0
    protagonist_steps = 0
    antagonist_steps = 0

    # For evaluation, this keeps track of the total undiscounted reward
    # accumulated during the episode.
    episode_return = tree.map_structure(_generate_zeros_from_spec,
                                        self._play_environment.reward_spec())
    timestep = self._play_environment.reset()

    # Protagonist has the first move
    self._protagonist.observe_first(timestep)
    #The shield also needs to be up to date
    self._shield.observe_first(timestep)

    # Run an episode.
    while not timestep.last():
        #Let the protagonist try
        prev_state = Grid2048()
        prev_state.cells = self._play_environment.getState().copy()
        action = self._protagonist.select_action(timestep.observation)
        timestep = self._play_environment.step(action)

        #Run through the shield
        shield_verdict = self._shield_environment.shieldstep(action, timestep.reward, prev_state, self._play_environment.getState())
        print(shield_verdict)
        if (shield_verdict.step_type == dm_env.StepType.HANDOVER):
            newaction = self._shield.select_action(timestep.observation)
            shield_verdict = self._shield_environment.step_newaction(shield_verdict, newaction)

        timestep = dm_env.TimeStep(shield_verdict.step_type, reward=shield_verdict.protagonist_reward, observation=shield_verdict.observation, discount=shield_verdict.discount)
        self._protagonist.observe(shield_verdict.shield_action.value, next_timestep=timestep)
        self._shield.observe(shield_verdict.shield_action.value, next_timestep=timestep)
        if (self._should_update):
            self._protagonist.update()
            self._shield.update()

        episode_return = tree.map_structure(operator.iadd, episode_return, timestep.reward)
        
        episode_steps += 1

    # Record counts.
    counts = self._counter.increment(episodes=1, steps=episode_steps)

    # Collect the results and combine with counts.
    steps_per_second = episode_steps / (time.time() - start_time)
    result = {
        'episode_length': episode_steps,
        'episode_return': episode_return,
        'steps_per_second': steps_per_second,
    }

    result.update(counts)
    return result

  def run(self,
          num_episodes: Optional[int] = None,
          num_steps: Optional[int] = None):
    """Perform the run loop.

    Run the environment loop either for `num_episodes` episodes or for at
    least `num_steps` steps (the last episode is always run until completion,
    so the total number of steps may be slightly more than `num_steps`).
    At least one of these two arguments has to be None.

    Upon termination of an episode a new episode will be started. If the number
    of episodes and the number of steps are not given then this will interact
    with the environment infinitely.

    Args:
      num_episodes: number of episodes to run the loop for.
      num_steps: minimal number of steps to run the loop for.

    Raises:
      ValueError: If both 'num_episodes' and 'num_steps' are not None.
    """

    if not (num_episodes is None or num_steps is None):
      raise ValueError('Either "num_episodes" or "num_steps" should be None.')

    def should_terminate(episode_count: int, step_count: int) -> bool:
      return ((num_episodes is not None and episode_count >= num_episodes) or
              (num_steps is not None and step_count >= num_steps))

    episode_count, step_count = 0, 0
    while not should_terminate(episode_count, step_count):
      result = self.run_episode()
      episode_count += 1
      step_count += result['episode_length']
      # Log the given results.
      self._logger.write(result)
      #print(episode_count)
      if episode_count % 5 == 0:
        print(episode_count)
        print(result, end='')
        print(result.get("episode_return") / result.get("episode_length"))


def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
  return np.zeros(spec.shape, spec.dtype)

