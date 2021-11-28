from acme import environment_loop
from acme import specs
from acme.agents.tf import dqn
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.utils import loggers
from acme.agents.tf import d4pg
import numpy as np
import sonnet as snt
import acme

from GameEnv import Game2048Env 

env = Game2048Env()
environment_spec = specs.make_environment_spec(env)

#Create agent
num_dimensions = np.prod(environment_spec.actions.shape, dtype=np.int32)

network = snt.Sequential([
    snt.Flatten(),
    snt.nets.MLP([50, 50, environment_spec.actions.num_values])
])

# Construct the agent.
agent = dqn.DQN(
    environment_spec=environment_spec, network=network, logger=loggers.TerminalLogger(label='agent'))

def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


# Run the environment loop.
logger = loggers.TerminalLogger()
loop = acme.EnvironmentLoop(env, agent, logger=logger)
loop.run(num_episodes=10000)  # pytype: disable=attribute-error

