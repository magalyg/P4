# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
    #This function is where we update the reward I am just confused as to what to 
    def _get_reward(s,a):
       reward =0
       next_state = s 
       if(a=0){
        

        reward = 
        }else if (a =1){
        reward = 
        }
        return reward 
  
    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.
        #phi_i represetnt the column of a matrix 
        #We need to define the states for a selected number of n not all states because it is continu0s
        #0 MEANS STAY #1 MEANS JUMP 
        alpha = 
        p_1 = [state.get('score')]
        tree = list(state.get('tree').values())
        monkey = list(state.get('monkey').values())
        phi_1 = np.hstack((np.zeros(7),p_1,tree,monkey,1))
        phi_2 = np.hstack((p_1,tree,monkey,np.zeros(7),1))
        theta =np.ones(15)
        state_i = np.hstack((p_1,tree,monkey,p_1,tree,monkey,1))
        beta = .5
        Q= np.vstack((np.multiply(theta,phi_1), np.multiply(theta,phi_2))).T
      
        #It doesnt matter which one is s' or s in any case we could jsut change this to self.reward + gamma*max(self.get_reward(s',a))
        Q_plus = self.last_reward + gamma*max((self.get_reward(s,0),self.get_reward(s,1)))
        phi_i =# this one would be defined as the maximum of any of the above 
        theta_new = theta - alpha *(Q_plus_i - Q)*phi_i
        Q_new =  np.vstack((np.multiply(theta_new,phi_1), np.multiply(theta_new,phi_2))).T
        new_action = npr.rand() < 0.1
        new_state  = state

        self.last_action = new_action
        self.last_state  = new_state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)
        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games. 
	run_games(agent, hist, 20, 10)

	# Save history. 
	np.save('hist',np.array(hist))


