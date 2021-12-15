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
import tensorflow as tf

from GameEnv import Game2048Env
from move import Move 

num_episodes = 10

env = Game2048Env()
environment_spec = specs.make_environment_spec(env)

#Create agent
num_dimensions = np.prod(environment_spec.actions.shape, dtype=np.int32)

network = snt.Sequential([
  snt.Flatten(),
  snt.Linear(16),
  tf.nn.relu,
  snt.nets.MLP([50,50,4])
])

# Construct the agent.
agent = dqn.DQN(
    environment_spec=environment_spec, network=network, logger=loggers.TerminalLogger(label='agent'))

# Run the environment loop.
logger = loggers.InMemoryLogger()
loop = acme.EnvironmentLoop(env, agent, logger=logger)
loop.run(num_episodes=num_episodes)
print(logger.data)

timestep = env.reset()
f = open("run.txt", "w")
while not timestep.last():
  action = agent.select_action(timestep.observation)
  timestep = env.step(action)
  f.write(str(Move(action)) + "\n")
  f.write(str(env._state.toIntArray()) + "\n")
