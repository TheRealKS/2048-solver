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
from tf_agents.policies import random_py_policy
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
from shield_driver_episode import ShieldDriverEpisode
from shield_driver_replay import ShieldDriverReplay

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


num_iterations = 100 # @param {type:"integer"}
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

print("Done, evaluating strategy")

dirname = "run_" + datetime.now().strftime("%d-%b-%Y-(%H:%M:%S)")
os.mkdir(dirname)

biggest = 0
wrun = -1
runs = []
pol = random_py_policy.RandomPyPolicy(time_step_spec=None,action_spec = env.action_spec())
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
