from ast import Name
from tf_agents.policies import TFPolicy
from move import Move

from shieldenvironment import ShieldedEnvironment

class PolicyShieldWrapper():
    
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
        for i in range(0, num_episodes):
            if (report_progress):
                print(str(i + 1) + "/" + str(num_episodes))

            time_step = self.tfenv.reset()
            episode_return = 0.0
            episode_ln = 0
            run = ""

            while not time_step.is_last():
                prev_state = self.env.get_state()
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

                safe_next_time_step = self.tfenv.step(actn)
                next_time_step = safe_next_time_step
                safe_next_state = self.env.get_state()
                legal_moves = safe_next_time_step.observation['legal_moves']
                m = safe_next_time_step.observation['mergeable']
                newscore, hscore, vscore = self.env.get_state().getStateScore()

                replaced = False
                if ((action_step.action == 1 or action_step.action == 3) and prev_state_legal_moves[0][2] == 1 and episode_ln >= 50):
                    self.env.set_state(prev_state.cells)
                    unsafe_next_time_step = self.tfenv.step(2)
                    if (not unsafe_next_time_step.is_last() and unsafe_next_time_step.reward >= 0.0):
                        m_alt = unsafe_next_time_step.observation['mergeable']
                        newscore_alt, hscore_alt, vscore_alt = self.env.get_state().getStateScore()

                        #Now that we have all the metrics, we can compare the two
                        if (hscore >= hscore_alt and m_alt == m):
                            #SAFE IS BETTER
                            pass
                        else:   
                            zero_rows = self.env.get_state().getNumZeroRows()
                            if (hscore_alt - zero_rows > 0 and self.env.get_state().isRowLocked(3) and m_alt > m):
                                #RIGHT IS BETTER
                                actn = 2
                                replaced = True
                                next_time_step = unsafe_next_time_step

                if (not replaced):
                    self.env.set_state(safe_next_state.cells)

                if (record_run or True):
                    obs = time_step.observation 
                    run += str(Move(actn)) + ";" + str(replaced) + "\n"
                    run += str(obs['new_tile'].numpy()[0]) + "\n"
                    run += str(obs['observation'][0].numpy()) + "\n"

                time_step = next_time_step
                ln += 1
                episode_return += time_step.reward
                episode_ln += 1
            
            total_return += episode_return
            mtile = time_step.observation['observation'][0].numpy().max()
            if mtile > biggesttile:
                biggesttile = mtile
                wrun = i
            runs.append(run)

            if (biggesttile == 2048):
                name = "run_2048_.txt"
                f = open("./" + name, "w")
                f.write(run)
                f.flush()
                f.close()
        
        avg_return = total_return / num_episodes
        avg_ln = ln / num_episodes
        return avg_return, avg_ln, biggesttile, wrun, runs
            