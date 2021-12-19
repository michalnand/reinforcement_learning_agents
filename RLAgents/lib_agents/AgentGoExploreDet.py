import numpy
import torch 
from .GoExploreBuffer import *
import cv2
      
class AgentGoExploreDet():   
    def __init__(self, envs, config):
        self.envs = envs  

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n


        self.buffer_size            = config.buffer_size
        self.reach_threshold        = config.reach_threshold
        self.add_threshold          = config.add_threshold
        self.downsample             = config.downsample
        self.epsilon                = config.epsilon
        
 
        self.envs_count             = config.envs_count 

        self.actions                = [[] for i in range(self.envs_count)]
        self.current_actions        = [[] for i in range(self.envs_count)]
        self.actions_idx            = numpy.zeros(self.envs_count, dtype=int)

        self.episode_score_sum      = numpy.zeros(self.envs_count)

        #all agents to explore mode
        self.agent_mode             = numpy.ones(self.envs_count, dtype=int)
        


        self.goals_buffer           = GoExploreBuffer(self.envs_count, self.buffer_size, self.state_shape, self.reach_threshold, self.add_threshold, self.downsample)

        #reset envs and fill initial state
        self.states = numpy.zeros((self.envs_count, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.envs_count):
            self.states[e] = self.envs.reset(e).copy()

 
        self.enable_training()
        self.iterations         = 0 

        self.log_agent_mode     = 1.0

    
    def enable_training(self):
        self.enabled_training = True
 
    def disable_training(self):
        self.enabled_training = False


    def main(self): 
        actions     = numpy.zeros(self.envs_count, dtype=int)

        for e in range(self.envs_count):
            #go - take actions from buffers
            if self.agent_mode[e] == 0:
                actions[e] = self.actions[e][self.actions_idx[e]]

                self.actions_idx[e]+= 1 

                #switch agent to explore mode
                if self.actions_idx[e] >= len(self.actions[e]):
                    self.actions_idx[e] = 0 
                    self.agent_mode[e]  = 1
            else:
                #explore, random action
                actions[e] = numpy.random.randint(0, self.actions_count)

            if numpy.random.rand() < self.epsilon:
                actions[e] = numpy.random.randint(0, self.actions_count)

            self.current_actions[e].append(actions[e])
         
        #execute action
        states, rewards_ext, dones, infos = self.envs.step(actions)

        self.states = states.copy()

        self.episode_score_sum+= rewards_ext
 
    
        reached = self.goals_buffer.step(states, self.current_actions, self.episode_score_sum)
        
        for e in range(self.envs_count): 
            #if reached[e]:
            #    self.agent_mode[e] = 1

            if dones[e]:
                self.states[e]              = self.envs.reset(e).copy()
                self.episode_score_sum[e]   = 0
                
                _, self.actions[e], _       = self.goals_buffer.new_goal(e)
                self.actions_idx[e]         = 0

                self.agent_mode[e]          = 0

                self.current_actions[e]     = []

        self.iterations+= 1
        return rewards_ext[0], dones[0], infos[0]
    
    def save(self, save_path):
        self.goals_buffer.save(save_path + "trained/")

    def load(self, load_path):
        self.goals_buffer.load(load_path + "trained/")

    def get_log(self): 
        result = "" 

        mode        = self.agent_mode.mean()

        k                   = 0.02
        self.log_agent_mode = (1.0 - k)*self.log_agent_mode + k*mode.mean()
        goals_count         = self.goals_buffer.goals_ptr

        result+= str(round(self.log_agent_mode, 7)) + " "
        result+= str(round(goals_count, 7)) + " "

        return result 

    def render(self, env_id = 0):
        size    = 512
        state   = self.states[env_id]

        goals           = self.goals_buffer.get_goals_for_render()
        
        state_resized   = cv2.resize(state[0],  (size, size))
        goal_resized    = cv2.resize(goals,     (size, size))
        
        result_im       = numpy.zeros((size, 2*size))

        result_im[0*size:1*size, 0*size:1*size] = state_resized
        result_im[0*size:1*size, 1*size:2*size] = goal_resized
        
        text_ofs_x = 10
        text_ofs_y = size - 20

        cv2.putText(result_im, "observation",       (text_ofs_x + 0*size, text_ofs_y + 0*size), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        cv2.putText(result_im, "goals buffer  ",    (text_ofs_x + 1*size, text_ofs_y + 0*size), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

        cv2.imshow("Go explore agent", result_im)
        cv2.waitKey(1)