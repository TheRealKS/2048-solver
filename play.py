from acme import environment_loop
from acme import specs
from acme.agents.tf import dqn
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.utils import loggers
import numpy as np
import sonnet as snt

from GameEnv import Game2048Env 

env = Game2048Env()
environment_spec = specs.make_environment_spec(env)

#Create agent
num_dimensions = np.prod(environment_spec.actions.shape, dtype=int)

# Create the shared observation network; here simply a state-less operation.
observation_network = tf2_utils.batch_concat

# Create the deterministic policy network.
policy_network = snt.Sequential([
    networks.LayerNormMLP((256, 256, 256), activate_final=True),
    networks.NearZeroInitializedLinear(num_dimensions),
    networks.TanhToSpec(environment_spec.actions),
])

# Create the distributional critic network.
critic_network = snt.Sequential([
    # The multiplexer concatenates the observations/actions.
    networks.CriticMultiplexer(),
    networks.LayerNormMLP((512, 512, 256), activate_final=True),
    networks.DiscreteValuedHead(vmin=-150., vmax=150., num_atoms=51),
])

agent_logger = loggers.TerminalLogger(label='agent', time_delta=10.)
env_loop_logger = loggers.TerminalLogger(label='env_loop', time_delta=10.)

agent = dqn.DQN(environment_spec=environment_spec,
  network=policy_network,
  logger=agent_logger,
  checkpoint=False)

env_loop = environment_loop.EnvironmentLoop(
    env, agent, logger=env_loop_logger)

env_loop.run(10)