from threading import active_count
from typing import Optional

from acme import adders
from acme import core
from acme import types
# Internal imports.
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils

import dm_env
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

from infeasibilitychecker import InfeasibilityChecker

tfd = tfp.distributions


class FeedForwardInfeasibleActor(core.Actor):
  """A feed-forward actor with check for infeasibility of actions

  An actor based on a feed-forward policy which takes non-batched observations
  and outputs non-batched actions. It also allows adding experiences to replay
  and updating the weights from the policy on the learner.
  """

  def __init__(
      self,
      policy_network: snt.Module,
      adder: Optional[adders.Adder] = None,
      variable_client: Optional[tf2_variable_utils.VariableClient] = None,
      infeasibility_checker: Optional[InfeasibilityChecker] = None,
  ):
    """Initializes the actor.

    Args:
      policy_network: the policy to run.
      adder: the adder object to which allows to add experiences to a
        dataset/replay buffer.
      variable_client: object which allows to copy weights from the learner copy
        of the policy to the actor copy (in case they are separate).
    """

    # Store these for later use.
    self._adder = adder
    self._variable_client = variable_client
    self._policy_network = policy_network
    self._infeasibility_checker = infeasibility_checker

  @tf.function
  def _policy(self, observation: types.NestedTensor) -> types.NestedTensor:
    # Add a dummy batch dimension and as a side effect convert numpy to TF.
    batched_observation = tf2_utils.add_batch_dim(observation)

    # Compute the policy, conditioned on the observation.
    policy = self._policy_network(batched_observation)

    # Sample from the policy if it is stochastic.
    action = policy.sample() if isinstance(policy, tfd.Distribution) else policy

    return action

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    feasiblemoves = self._infeasibility_checker.getFeasibleMoves(observation)

    # Pass the observation through the policy network.
    action = tf2_utils.to_numpy_squeeze(self._policy(observation))
    if (len(feasiblemoves) == 0):
      return action
    
    c = 0
    while (int(action) not in feasiblemoves):
        action = tf2_utils.to_numpy_squeeze(self._policy(observation))
        c += 1
        if (c > 10):
          return np.int32(feasiblemoves[0])

    return action

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)

  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add(action, next_timestep)

  def update(self, wait: bool = False):
    if self._variable_client:
      self._variable_client.update(wait)
