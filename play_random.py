from __future__ import absolute_import, division, print_function
from asyncio import shield

import base64
from typing import Optional
from xmlrpc.server import resolve_dotted_attribute
import numpy as np
import reverb
import os
from tf_agents import utils
from tf_agents.trajectories.time_step import time_step_spec
from datetime import datetime
from GameEnvNTF import Game2048NPyEnv

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.policies import py_tf_eager_policy, epsilon_greedy_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.environments import utils
from tf_agents.networks import q_network, categorical_q_network
from GameEnvRTF import Game2048RPyEnv
from GameEvalEnvTF import Game2048EvalPyEnv
from driver_unshielded import UnshieldedDriver
from tf_agents.policies.policy_saver import PolicySaver

from move import Move
from policy_shield_wrapper import PolicyShieldWrapper
from util import getMergeableTileValues

def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  ln = 0
  biggesttile = 0
  allvals = []
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    vals = np.zeros((6))
    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
      ln += 1
      vals += getMergeableTileValues(time_step.observation['mergeable'][0].numpy(), time_step.observation['observation'][0].numpy())
    total_return += episode_return
    mtile = time_step.observation['observation'][0].numpy().max()
    if mtile > biggesttile:
      biggesttile = mtile
    allvals.append(vals / 2)

  avg_return = total_return / num_episodes
  avg_ln = ln / num_episodes
  avg_vals = np.array(allvals).sum(axis=0) / num_episodes
  avg_vals = np.ceil(avg_vals)
  print(avg_vals)
  return avg_return.numpy()[0], avg_ln, biggesttile

def splitter_fun(obs):
    return obs['observation'], obs['legal_moves']


num_iterations = 500 # @param {type:"integer"}
collect_episodes_per_iteration = 1 # @param {type:"integer"}
replay_buffer_capacity = 10000 # @param {type:"integer"}

fc_layer_params = (128,)

learning_rate = 1e-3 # @param {type:"number"}
log_interval = 25 # @param {type:"integer"}
num_eval_episodes = 10 # @param {type:"integer"}
eval_interval = 25 # @param {type:"integer"}


env = Game2048RPyEnv()
utils.validate_py_environment(env, episodes=5)

train_py_env = Game2048RPyEnv()
eval_py_env = Game2048EvalPyEnv()
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

network_prot = categorical_q_network.CategoricalQNetwork(
				train_env.time_step_spec().observation['observation'],
				train_env.action_spec())

tf_agent = categorical_dqn_agent.CategoricalDqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    categorical_q_network=network_prot,
    optimizer=optimizer,
    target_update_period=50,
    min_q_value=-1.0,
    max_q_value=2048,
    n_step_update=1,
    td_errors_loss_fn=common.element_wise_huber_loss,
    train_step_counter=train_step_counter,
    observation_and_action_constraint_splitter=splitter_fun,
    gamma=0.95,
    epsilon_greedy=0.1)
tf_agent.initialize()

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

table_name = 'protnaive_table'
replay_buffer_signature = tensor_spec.from_spec(
      tf_agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
    replay_buffer_signature)

prot_table = reverb.Table(
    table_name,
    max_size=replay_buffer_capacity,
    sampler=reverb.selectors.Prioritized(0.8),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)

reverb_server = reverb.Server([prot_table])

replay_buffer_prot = reverb_replay_buffer.ReverbReplayBuffer(
    tf_agent.collect_data_spec,
    table_name=table_name,
    sequence_length=2,
    local_server=reverb_server)
  
rb_observer_prot = reverb_utils.ReverbAddTrajectoryObserver(
  replay_buffer_prot.py_client,
  table_name,
  sequence_length=2)

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
tf_agent.train = common.function(tf_agent.train)

# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
returns = [avg_return]

time_step = train_py_env.reset()

policy_u = py_tf_eager_policy.PyTFEagerPolicy(
    tf_agent.collect_policy, use_tf_function=True)

# Create a driver to collect experience.
collect_driver = UnshieldedDriver(
    train_py_env,
    policy_u,
    [rb_observer_prot],
    max_episodes=1)

ds = replay_buffer_prot.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=64,
    num_steps=2).prefetch(3)

iterator = iter(ds)

losses = []

for _ in range(num_iterations):
  time_step = train_py_env.reset()
  # Collect a few steps and save to the replay buffer.
  time_step, s, ln = collect_driver.run(time_step)
  
  # Sample a batch of data from the buffer and update the agent's network.
  experience, extra = next(iterator)
  train_loss = tf_agent.train(experience)
  losses.append(train_loss.loss.numpy())

  step = tf_agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss.loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)

print("Done, evaluating strategy")

fname = "run_random_" + str(num_iterations) + "_" + datetime.now().strftime("%d-%b-%Y-(%H:%M:%S)")
os.mkdir(fname)

np_ret = np.array(returns)
np.save(fname + '/eval', np_ret)

np_loss = np.array(losses)
np.save(fname + '/losses', np_loss)

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
  f = open("./" + fname + "/" + name, "w")
  f.write(runs[i])
  f.flush()
  f.close()