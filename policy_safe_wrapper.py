import numpy as np
from tf_agents.policies import TFPolicy
from move import Move

from shieldenvironment import ShieldedEnvironment
from util import getMergeableTileValues


"""
Wraps a policy and makes enforces safety (adherance to strategy)
"""
class PolicySafeWrapper():
    
    def __init__(self, naivepolicy: TFPolicy, env: ShieldedEnvironment) -> None:
        self.pol = naivepolicy
        self.tfenv = env
        self.env = env._env.envs[0]

    def run(self, num_episodes, record_run=True, report_progress=False):
        total_return = 0.0
        ln = 0
        biggesttile = 0
        runs = []
        wrun = -1
        sums = 0
        allvals = []
        for i in range(0, num_episodes):
            if (report_progress):
                print(str(i + 1) + "/" + str(num_episodes))

            time_step = self.tfenv.reset()
            episode_return = 0.0
            episode_ln = 0
            run = ""
            vals = np.zeros((6))

            while not time_step.is_last():
                prev_state_legal_moves = time_step.observation['legal_moves']
                action_step = self.pol.action(time_step)
                actn = action_step.action
                if (actn == 0 or actn == 2):
                    if (prev_state_legal_moves[0][1] == 1):
                        actn = 1
                    elif (prev_state_legal_moves[0][3] == 1):
                        actn = 3
                if (actn == 0):
                    if (prev_state_legal_moves[0][2] == 1):
                        actn = 2

                next_time_step = self.tfenv.step(actn)
                if (record_run):
                    obs = time_step.observation 
                    run += str(Move(actn)) + "\n"
                    run += str(obs['new_tile'].numpy()[0]) + "\n"
                    run += str(obs['observation'][0].numpy()) + "\n"

                time_step = next_time_step
                ln += 1
                vals += getMergeableTileValues(time_step.observation['mergeable'][0].numpy(), time_step.observation['observation'][0].numpy())
                episode_return += time_step.reward
                total_return += episode_return
                episode_ln += 1
            
            mtile = time_step.observation['observation'][0].numpy().max()
            if mtile > biggesttile:
                biggesttile = mtile
                wrun = i
            runs.append(run)
            sumoftiles = time_step.observation['observation'][0].numpy().sum()
            sums += sumoftiles
            allvals.append(vals / 2)

        avg_return = total_return / num_episodes
        avg_ln = ln / num_episodes
        avg_sum = sums / num_episodes
        avg_vals = np.array(allvals).sum(axis=0) / num_episodes
        avg_vals = np.ceil(avg_vals)
        print(avg_vals)
        return avg_return, avg_ln, avg_sum, biggesttile, wrun, runs
            