from __future__ import absolute_import, division, print_function

from typing import Optional
import numpy as np
import reverb
from tf_agents import utils
import os
from datetime import datetime

from GameEnvTF import Game2048PyEnv

import tensorflow as tf

from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.policies import py_tf_eager_policy, epsilon_greedy_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.environments import utils
from tf_agents.networks import categorical_q_network
from GameEvalEnvTF import Game2048EvalPyEnv
from driver_shielded import ShieldedDriver

from move import Move
from policy_shield_wrapper import PolicyShieldWrapper

def compute_avg_return(environment, policy, num_episodes=10, record = False):
  pol = PolicyShieldWrapper(policy, environment)
  avg_return, avg_len, avg_sum, b, wrun, runs = pol.run(num_episodes, record)
  if (record):
    return avg_return.numpy()[0], avg_len, avg_sum, b, wrun, runs
  return avg_return.numpy()[0], avg_len, avg_sum, b

def splitter_fun(obs):
    return obs['observation'], obs['legal_moves']


num_iterations = 500 # @param {type:"integer"}
collect_episodes_per_iteration = 1 # @param {type:"integer"}
replay_buffer_capacity = 5000 # @param {type:"integer"}

fc_layer_params = (128,)

learning_rate = 1e-3 # @param {type:"number"}
log_interval = 25 # @param {type:"integer"}
num_eval_episodes = 10 # @param {type:"integer"}
eval_interval = 25 # @param {type:"integer"}


env = Game2048PyEnv()
utils.validate_py_environment(env, episodes=5)


train_py_env = Game2048PyEnv()
eval_py_env = Game2048EvalPyEnv()
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

network_prot = categorical_q_network.CategoricalQNetwork(
				train_env.time_step_spec().observation['observation'],
				train_env.action_spec(),
        fc_layer_params=fc_layer_params)

tf_agent = categorical_dqn_agent.CategoricalDqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    categorical_q_network=network_prot,
    optimizer=optimizer,
    target_update_period=50,
    n_step_update=2,
    min_q_value=-1.0,
    max_q_value=2048.0,
    td_errors_loss_fn=common.element_wise_huber_loss,
    train_step_counter=train_step_counter,
    observation_and_action_constraint_splitter=splitter_fun,
    gamma=0.95,
    epsilon_greedy=0.1)
tf_agent.initialize()

mask = np.array([0,1,0,1], dtype=np.int32)
tf_agent._collect_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(
          tf_agent.collect_policy.wrapped_policy, epsilon=0.1, exploration_mask=mask)

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

table_name = 'prot_table'
replay_buffer_signature = tensor_spec.from_spec(
      tf_agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
    replay_buffer_signature)

prot_table = reverb.Table(
    table_name,
    max_size=replay_buffer_capacity,
    sampler=reverb.selectors.Prioritized(priority_exponent=0.8),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)
    #max_times_sampled=3)

reverb_server = reverb.Server([prot_table])

replay_buffer_prot = reverb_replay_buffer.ReverbReplayBuffer(
    tf_agent.collect_data_spec,
    table_name=table_name,
    sequence_length=3,
    local_server=reverb_server)
  
rb_observer_prot = reverb_utils.ReverbAddTrajectoryObserver(
  replay_buffer_prot.py_client,
  table_name,
  sequence_length=3)

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
collect_driver = ShieldedDriver(
    train_py_env,
    policy_u,
    [rb_observer_prot],
    max_episodes=1)

ds = replay_buffer_prot.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=64,
    num_steps=3).prefetch(1)

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

  """   # Get the priorities and update.
  priorities = tf.abs(train_loss.extra.td_error)
  priorities = tf.cast(priorities, dtype=tf.float64)
  keys = extra.key[:,0]
  replay_buffer_prot.tf_client.update_priorities(
          'prot_table',
          keys,
          priorities) """

  step = tf_agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss.loss))

  if step == num_iterations:
    avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes, True)
    tp = (avg_return[0], avg_return[1], avg_return[2], avg_return[3])
    print('step = {0}: Average Return = {1}'.format(step, tp))
    returns.append(tp)
    wrun = avg_return[4]
    runs = avg_return[5]
  elif step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)

print("Done, evaluating strategy")

fname = "run_shield_sum_" + str(num_iterations) + "_" + datetime.now().strftime("%d-%b-%Y-(%H:%M:%S)")
os.mkdir(fname)

np_ret = np.array(returns)
np.save(fname + '/eval', np_ret)

np_loss = np.array(losses)
np.save(fname + '/losses', np_loss)

for i in range(0,10):
  name = "run_" + str(i) + ".txt"
  if (i == wrun):
    name = "run_best_" + str(i) + ".txt"
  f = open("./" + fname + "/" + name, "w")
  f.write(runs[i])
  f.flush()
  f.close()
