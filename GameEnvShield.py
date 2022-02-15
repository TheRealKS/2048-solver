from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from dm_env import specs
import dm_env
from sys import maxsize
from grid import Grid2048
from shieldenvironment import ShieldEnvironment, ShieldTimeStep, handover_shield, restart_shield, termination_shield, transition_shield

from util import generateRandomGrid
from move import Move

INTERVENE_PENALTY_DIVISOR = 10

class Game2048ShieldEnv(ShieldEnvironment):
    """The main environment in which the game is played. For the shield"""

    def __init__(self, initial_state : Grid2048 = None):
        super().__init__()

        self._initial_state = initial_state
        if (initial_state == None):
            #Generate a random environment.
            self._initial_state = generateRandomGrid()
        self._initial_state_grid = self._initial_state.cells
        self._state = self._initial_state
        
        self._episode_ended = False 
        self.prev_action = -1
    
    def action_spec(self):
        return specs.DiscreteArray(dtype=np.int64, num_values=5, name='action')
    
    def observation_spec(self): 
        return specs.BoundedArray(shape=self._state.shape(), dtype=np.float32, minimum=-1.0, maximum=float(maxsize), name='observation')
    
    def reward_spec(self):
        return specs.BoundedArray(dtype=np.double, shape=(), name='reward', minimum=0, maximum=np.double(maxsize))

    def reset(self):
        self._episode_ended = False
        self._state.cells = self._initial_state_grid.copy()
        return restart_shield(self._state.toFloatArray())

    def step(self, action):
        return 0
    
    """Single step function"""
    def shieldstep(self, protagonist_action, reward, prev_state : Grid2048, next_state) -> ShieldTimeStep:
        if self._episode_ended:
            return self.reset()

        #Protagonist picks a move
        # ->
        #Shield passes this move + observation through its neural net (i.e. action spec = move + observation)
        # ->
        # Shield returns verdict on what move is best (i.e. result spec = 4 actions)
        # ->
        #Check if selected move (by protagonist) is the same. If it is, pass it on. If not, correct to what the shield said.

        #First, train the protagonist to have at least some idea of what its doing
        #Then, add the shield. Shield and protagonist will learn from each other


        #Shield learns what the consequences are when a non-strategic move is made. This means that the shield only really learns observations + Move.UP + next observation
        #Specific trajectory
        #Non strategic move is made
        #Specific trajectory
        #Did this game end in game over (when it was attributable to the non-strategic move). How do we check this?
        # - Set a number of move treshold (i.e. did the game end within 2 moves)
        # - Define some sort of 'well ordered' heuristic. i.e. did the game become 'not well ordered'?
        
        #So the training loop is:
        #1) Select action through protagonist and step
        #2) If resulting action is dangerous:
        #   1) Step the shield
        #   2) Check if there is a discrepancy
        #   3) Observe
        #3) Adjust timestep for protagonist and observe

        #Perhaps it is is a good idea to have common environment between protagonist and shield
        #This is not only more memory efficient, but can also prevent (potential) bugs relating to consistency between observations of protagonist and shield
        #This way the shield is more or less black box to the protagonist, but this needn't be bad. If we are a human player, the shield is also 'automagical'.
            
        #Evaluate the move that the protagonist made, i.e., will this move lead to game over? So the output of the shield is the probability of game over from this state? This means that the output is more or less binary?
        #Is it then a classifier? Do we need reinforcement learning? Let's try to visualise the environment that the shield is in. So it's MDP:
        #Reward for the shield is high. Double agent architecture might be a bit redundant. If the classifier will penalise the protagonist for ending up in states that end in game over. This automatically should mean that it avoids getting in that place.
        #This might require a lot of training. And does not really adress the problem of correction, just penalisastion. We could train a separate 'micro-agent' who's only objective is to stabilise the game after destabilisation. So the policy of a reinoforcement shield could be argminned.
        #Like, if the reward for doing a dangerous move in a certain state is low for the shield, this means that it might lead to game over.
        
        #The shield (on dangerous move) plays this move, and runs a few moves according to protagonists policy. If the evaluation of the resulting state is bad, the protagonist will get worse reward.


        next_state_g = Grid2048()
        next_state_g.cells = next_state
        moves = next_state_g.movesAvailableInDirection()
        if (len(moves) > 0):    
            if (reward == -1.0):
                #Protagonist move did nothing; check if there is a different move
                if (("h",1) in moves or ("v",-1) in moves):
                    #We could have done a strategic move. Perform this action and 
                    down_reward = prev_state.performActionIfPossible(Move.DOWN, savestate=False)
                    left_reward = prev_state.performActionIfPossible(Move.LEFT, savestate=False)
                    #Do what's optimal (we're trying to help)
                    if (down_reward >= left_reward):
                        prev_state.performActionIfPossible(Move.DOWN)
                        return transition_shield(p_reward=0.0, s_reward=down_reward, observation=prev_state.toFloatArray(), p_action=protagonist_action, s_action=Move.DOWN)
                    else:
                        prev_state.performActionIfPossible(Move.LEFT)
                        return transition_shield(p_reward=0.0, s_reward=down_reward, observation=prev_state.toFloatArray(), p_action=protagonist_action, s_action=Move.LEFT)
            else:
                #Protagonist move did do something; check if it is a 'dangerous' move
                if (protagonist_action == Move.DOWN or protagonist_action == Move.LEFT):
                    #These are valid strategic moves, so we don't care
                    return transition_shield(p_reward=reward, s_reward=1.0, observation=next_state, p_action=protagonist_action, s_action=protagonist_action)
                else:
                    #These are potentially dangerous moves, so we have to check if they are not stupid
                    if (("h",1) in moves or ("v",-1) in moves):
                        #We could have also made a strategic move. So correct the protagonist.
                        down_reward = prev_state.performActionIfPossible(Move.DOWN, savestate=False)
                        left_reward = prev_state.performActionIfPossible(Move.LEFT, savestate=False)
                        #Do what's optimal (we're trying to help)
                        if (down_reward >= left_reward):
                            prev_state.performActionIfPossible(Move.DOWN)
                            return transition_shield(p_reward=down_reward / INTERVENE_PENALTY_DIVISOR, s_reward=down_reward, observation=prev_state.toFloatArray(), p_action=protagonist_action, s_action=Move.DOWN)
                        else:
                            prev_state.performActionIfPossible(Move.LEFT)
                            return transition_shield(p_reward=down_reward / INTERVENE_PENALTY_DIVISOR, s_reward=down_reward, observation=prev_state.toFloatArray(), p_action=protagonist_action, s_action=Move.LEFT)
                    else:
                        #We have no choice but to make a dangerous move
                        #Refer this to the shields Q network
                        return handover_shield(prev_state)
        else:  
            #Game over
            final_reward = float(self._state.sumOfTiles())
            return termination_shield(p_reward=final_reward, s_reward=final_reward, observation=self._state.toFloatArray(), p_action=protagonist_action, s_action=Move.NOTHING)

        return transition_shield(p_reward=reward, s_reward=1.0, observation=next_state, p_action=protagonist_action, s_action=protagonist_action)


    """Step function after consulatation of shield"""
    def step_newaction(self, oldtimestep : ShieldTimeStep, newaction : Move) -> ShieldTimeStep:
        newaction = Move(newaction)
        r = oldtimestep.observation.performActionIfPossible(newaction)
        if (oldtimestep.protagonist_action == newaction):
            return transition_shield(p_reward=oldtimestep.protagonist_reward, s_reward=1.0, observation=oldtimestep.observation.toFloatArray(), p_action=newaction, s_action=newaction)
        else:
            return transition_shield(p_reward=r / INTERVENE_PENALTY_DIVISOR, s_reward=r, observation=oldtimestep.observation.toFloatArray(), p_action=oldtimestep.protagonist_reward, s_action=newaction)

    def render(self):
        return self._state.toIntArray()