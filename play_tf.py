from __future__ import absolute_import, division, print_function

import base64
from typing import Optional
import numpy as np
import PIL.Image
import reverb
from tf_agents import utils
from tf_agents.trajectories.time_step import time_step_spec
import os
from datetime import datetime

from GameEnvTF import Game2048PyEnv

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.environments import utils
from tf_agents.networks import q_network
from ShieldEnvTF import Game2048ShieldPyEnv
from driver_shielded import ShieldedDriver

from move import Move
from shield_driver import ShieldDriver

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

def splitter_fun(obs):
    return obs['observation'], obs['legal_moves']


num_iterations = 250 # @param {type:"integer"}
collect_episodes_per_iteration = 1 # @param {type:"integer"}
replay_buffer_capacity = 20000 # @param {type:"integer"}

fc_layer_params = (100,)

learning_rate = 1e-3 # @param {type:"number"}
log_interval = 25 # @param {type:"integer"}
num_eval_episodes = 10 # @param {type:"integer"}
eval_interval = 50 # @param {type:"integer"}


env = Game2048PyEnv()
utils.validate_py_environment(env, episodes=5)


train_py_env = Game2048PyEnv()
eval_py_env = Game2048PyEnv()
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

train_shield_pyenv = Game2048ShieldPyEnv()
train_shield_env = tf_py_environment.TFPyEnvironment(train_shield_pyenv)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

network_prot = q_network.QNetwork(
				env.time_step_spec().observation['observation'],
				env.action_spec(),
				fc_layer_params=fc_layer_params)

network_shield = q_network.QNetwork(
        env.time_step_spec().observation['observation'],
        env.action_spec(),
        fc_layer_params=fc_layer_params)

tf_agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=network_prot,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter,
    observation_and_action_constraint_splitter=splitter_fun)
tf_agent.initialize()

shield_agent = dqn_agent.DqnAgent(
    train_shield_env.time_step_spec(),
    train_shield_env.action_spec(),
    q_network=network_shield,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter,
    observation_and_action_constraint_splitter=splitter_fun)
shield_agent.initialize()

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

table_name = 'prot_table'
replay_buffer_signature = tensor_spec.from_spec(
      tf_agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
    replay_buffer_signature)

prot_table = reverb.Table(
    table_name,
    max_size=100000,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)
  
shield_table = reverb.Table(
    'shield_table',
    max_size=100000,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)

reverb_server = reverb.Server([prot_table, shield_table])

replay_buffer_prot = reverb_replay_buffer.ReverbReplayBuffer(
    tf_agent.collect_data_spec,
    table_name=table_name,
    sequence_length=2,
    local_server=reverb_server)
  
replay_buffer_shield = reverb_replay_buffer.ReverbReplayBuffer(
    shield_agent.collect_data_spec,
    table_name='shield_table',
    sequence_length=2,
    local_server=reverb_server)

rb_observer_prot = reverb_utils.ReverbAddTrajectoryObserver(
  replay_buffer_prot.py_client,
  table_name,
  sequence_length=2)

rb_observer_shield = reverb_utils.ReverbAddTrajectoryObserver(
  replay_buffer_shield.py_client,
  'shield_table',
  sequence_length=2
)

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
tf_agent.train = common.function(tf_agent.train)
shield_agent.train = common.function(shield_agent.train)

# Reset the train step
tf_agent.train_step_counter.assign(0)
shield_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
returns = [avg_return]

ds = replay_buffer_prot.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=64,
    num_steps=2).prefetch(3)

ds_shield = replay_buffer_shield.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=64,
    num_steps=2).prefetch(3)

iterator = iter(ds)
it_shield = iter(ds_shield)

time_step = train_py_env.reset()

policy_u = py_tf_eager_policy.PyTFEagerPolicy(
    tf_agent.collect_policy, use_tf_function=True)

shield_driver = ShieldDriver(
  train_shield_pyenv,
  policy_u,
  [rb_observer_shield],
  max_episodes = 1
)

# Create a driver to collect experience.
collect_driver = ShieldedDriver(
    train_py_env,
    shield_agent,
    shield_driver,
    it_shield,
    policy_u,
    [rb_observer_prot],
    max_episodes=1)

for _ in range(num_iterations):
  time_step = train_py_env.reset()
  # Collect a few steps and save to the replay buffer.
  time_step, s = collect_driver.run(time_step)
  
  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = tf_agent.train(experience).loss

  step = tf_agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss))

print("Done, evaluating strategy")

dirname = "run_" + datetime.now().strftime("%d-%b-%Y-(%H:%M:%S)")
os.mkdir(dirname)

biggest = 0
wrun = -1
runs = []
pol = tf_agent.post_process_policy()
for i in range(0,10):
  run = ""
  timestep = eval_env._reset()
  while not timestep.is_last():
    action_step = pol.action(timestep)
    obs = timestep.observation
    run += str(Move(action_step.action)) + "\n"
    run += str(obs['new_tile'].numpy()[0]) + "\n"
    run += str(obs['observation'][0].numpy()) + "\n"
    timestep = eval_env._step(action_step.action)

  m = timestep.observation['observation'][0].numpy().max()
  if (m > biggest):
    biggest = m
    wrun = i
  
  runs.append(run)

for i in range(0,10):
  name = "run_" + str(i) + ".txt"
  if (i == wrun):
    name = "run_best_" + str(i) + ".txt"
  f = open("./" + dirname + "/" + name, "w")
  f.write(runs[i])
  f.flush()
  f.close()
