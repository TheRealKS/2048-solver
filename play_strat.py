from os import environ
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
from keras import layers
from DQNalt import DQNFeasible

from GameEnvProtagonist import Game2048ProtagonistEnv
from grid import Grid2048
from infeasibilitychecker import InfeasibilityChecker
from move import Move 

num_episodes = 150

env = Game2048ProtagonistEnv(agent=[])
environment_spec = specs.make_environment_spec(env)
print(environment_spec)

#Create agent
num_dimensions = np.prod(environment_spec.actions.shape, dtype=np.int32)

network = snt.Sequential([
  snt.Flatten(),
  snt.nets.MLP([16,50]),
  snt.nets.MLP([50,4])
])

# Construct the agent.
agent = DQNFeasible(
    environment_spec=environment_spec, network=network, logger=loggers.TerminalLogger(label='agent'), checker=InfeasibilityChecker(env), target_update_period=10, epsilon=0.1, n_step=1, discount=0.6, learning_rate=0.5)
env._agent = agent

# Run the environment loop.
logger = loggers.InMemoryLogger()
loop = acme.EnvironmentLoop(env, agent, logger=logger, should_update=True)
loop.run(num_episodes=num_episodes)
print(logger.data)


print("Done, evaluating strategy")
biggest = 0
wrun = ""
for i in range(0,10):
  run = ""
  timestep = env.reset()
  while not timestep.last():
    action = agent.select_action(timestep.observation)
    timestep = env.step(action)
    run += str(Move(action)) + "\n"
    run += str(env._state.toIntArray()) + "\n"
  
  m = env._state.cells.max()
  if (m > biggest):
    biggest = m
    wrun = run

f = open("run.txt", "w")
f.write(wrun)
f.flush()
f.close()
