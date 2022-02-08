import abc
from typing import Any, NamedTuple
import dm_env
from dm_env import StepType

from move import Move

class ShieldTimeStep(NamedTuple):
  step_type: Any
  protagonist_reward: Any
  shield_reward: Any
  discount: Any
  protagonist_action: Any
  shield_action: Any
  observation: Any

  def first(self) -> bool:
    return self.step_type == StepType.FIRST
  
  def mid(self) -> bool:
    return self.step_type == StepType.MID
  
  def last(self) -> bool:
    return self.step_type == StepType.LAST

  def handover(self) -> bool:
    return self.step_type == StepType.HANDOVER


class ShieldEnvironment(dm_env.Environment):
    def __init__(self) -> None:
        super().__init__()
    
    @abc.abstractmethod
    def shieldstep(self, protagonist_action, reward, prev_state, next_state) -> ShieldTimeStep:
        pass

    @abc.abstractmethod
    def step_newaction(self, oldtimestep : ShieldTimeStep, newaction : Move) -> ShieldTimeStep:
        pass

def restart_shield(observation):
  """Returns a `TimeStep` with `step_type` set to `StepType.FIRST`."""
  return ShieldTimeStep(StepType.FIRST, None, None, None, None, None, observation)

def transition_shield(p_reward, s_reward, observation, p_action, s_action, discount=1.0):
  """Returns a `TimeStep` with `step_type` set to `StepType.MID`."""
  return ShieldTimeStep(StepType.MID, p_reward, s_reward, discount, p_action, s_action, observation)

def termination_shield(p_reward, s_reward, observation, p_action, s_action):
  """Returns a `TimeStep` with `step_type` set to `StepType.LAST`."""
  return ShieldTimeStep(StepType.MID, p_reward, s_reward, 0.0, p_action, s_action, observation)

def truncation_shield(p_reward, s_reward, observation, p_action, s_action, discount=1.0):
  """Returns a `TimeStep` with `step_type` set to `StepType.LAST`."""
  return ShieldTimeStep(StepType.MID, p_reward, s_reward, discount, p_action, s_action, observation)

def handover_shield(observation):
    return ShieldTimeStep(StepType.HANDOVER, None, None, None, None, None, observation)
