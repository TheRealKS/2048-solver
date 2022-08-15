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
from tf_agents.policies import py_tf_eager_policy, epsilon_greedy_policy, random_py_policy
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


env = Game2048RPyEnv()
utils.validate_py_environment(env, episodes=5)

pol = random_py_policy.RandomPyPolicy(env.time_step_spec(), env.action_spec())

print("Done, evaluating strategy")

fname = "run_randompolicy_0_" + datetime.now().strftime("%d-%b-%Y-(%H:%M:%S)")
os.mkdir(fname)

biggest = 0
wrun = -1
runs = []
rets = []
lns = []
maxt = []
for i in range(0,10):
  run = ""
  timestep = env._reset()
  reward = 0
  ln = 0
  while not timestep.is_last():
    action_step = pol.action(timestep)
    obs = timestep.observation
    run += str(Move(action_step.action)) + "\n"
    run += str(obs['new_tile']) + "\n"
    run += str(obs['observation']) + "\n"
    timestep = env._step(action_step.action)
    reward += timestep.reward
    ln += 1

  m = timestep.observation['observation'].max()
  maxt.append(m)
  if (m > biggest):
    biggest = m
    wrun = i
  
  runs.append(run)
  rets.append(reward)
  lns.append(ln)

for i in range(0,10):
  name = "run_" + str(i) + ".txt"
  if (i == wrun):
    name = "run_best_" + str(i) + ".txt"
  f = open("./" + fname + "/" + name, "w")
  f.write(runs[i])
  f.flush()
  f.close()


np_ret = np.array(rets)
np_ln = np.array(lns)
np_mt = np.array(maxt)
np.save(fname + '/eval', np.array([np_ret, np_ln, np_mt]))

np_loss = np.array(lns)
np.save(fname + '/losses', np_loss)