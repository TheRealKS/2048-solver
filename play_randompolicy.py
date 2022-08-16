from __future__ import absolute_import, division, print_function

import os
from datetime import datetime

import numpy as np
from tf_agents import utils
from tf_agents.policies import random_py_policy

from GameEnvRTF import Game2048RPyEnv
from move import Move

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
