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

from GameEnvProtagonist import Game2048Env
from GameEnvStrategy import Game2048StratEnv
from grid import Grid2048
from move import Move 

num_episodes = 10

env = Game2048StratEnv()
environment_spec = specs.make_environment_spec(env)
print(environment_spec)

#Create agent
num_dimensions = np.prod(environment_spec.actions.shape, dtype=np.int32)

network = snt.Sequential([
  snt.Flatten(),
  snt.nets.MLP([16,100,100,100,2], activate_final=True)
])

# Construct the agent.
agent = dqn.DQN(
    environment_spec=environment_spec, network=network, logger=loggers.TerminalLogger(label='agent'), epsilon=0.1, discount=0.3, learning_rate=0.2)

# Run the environment loop.
logger = loggers.InMemoryLogger()
loop = acme.EnvironmentLoop(env, agent, logger=logger)
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
  
  m = env._state.cells.argmax()
  if (m > biggest):
    biggest = m
    wrun = run

f = open("run.txt", "w")
f.write(wrun)
f.flush()
f.close()
