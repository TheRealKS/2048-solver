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
from EnvironmentLoopShield import ShieldEnvironmentLoop

from GameEnvProtagonist import Game2048ProtagonistEnv
from GameEnvShield import Game2048ShieldEnv
from grid import Grid2048
from move import Move 

num_episodes = 600

env = Game2048ProtagonistEnv()
environment_spec = specs.make_environment_spec(env)

senv = Game2048ShieldEnv()
shield_environment_spec = specs.make_environment_spec(senv)

#Create agent
num_dimensions = np.prod(environment_spec.actions.shape, dtype=np.int32)

play_network = snt.Sequential([
  snt.Flatten(),
  snt.nets.MLP([16,50,50,4], activate_final=True)
])

shield_network = snt.Sequential([
  snt.Flatten(),
  snt.nets.MLP([16,50,50,4], activate_final=True)
])

# Construct the agent.
agent = dqn.DQN(
    environment_spec=environment_spec, network=play_network, logger=loggers.TerminalLogger(label='agent'))


shield = dqn.DQN(
  environment_spec=shield_environment_spec, network=shield_network, logger=loggers.TerminalLogger(label='shield')
)

# Run the environment loop.
logger = loggers.InMemoryLogger()
loop = ShieldEnvironmentLoop(play_environment=env, shield_environment=senv, protagonist=agent, shield=shield, logger=logger)
loop.run(num_episodes=num_episodes)
print(logger.data)

print("Done, evaluating strategy")
biggest = 0
wrun = ""
for i in range(0,100):
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
