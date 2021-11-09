import numpy
import torch
 
from .GoalsBuffer    import *  


class AgentNirvana():   
    def __init__(self, envs, Model, config):
        self.envs = envs  


        self.steps                  = config.steps
        self.batch_size             = config.batch_size
        
        self.training_epochs        = config.training_epochs
        self.envs_count             = config.envs_count
        

        self.buffer_size                = config.buffer_size
        self.buffer_add_threshold       = config.buffer_add_threshold
        self.buffer_downsample          = config.buffer_downsample

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n


        self.goals_buffer   =  GoalsBuffer(self.buffer_size, self.state_shape, self.buffer_add_threshold, self.buffer_downsample, True)
    
        #self.model          = Model.Model(self.state_shape, self.actions_count, self.buffer_downsample)
        #self.optimizer      = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

        #reset envs and fill initial state
        self.states = numpy.zeros((self.envs_count, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.envs_count):
            self.states[e] = self.envs.reset(e).copy()

        #all agents into explore mode - nothing stored in buffer yet, so explore some new states
        self.mode = numpy.zeros(self.envs_count, dtype=bool)

        self.goals_t                = torch.zeros((self.envs_count, ) + self.goals_buffer.goal_shape)

        self.episode_ext_reward_sum = numpy.zeros(self.envs_count)
        self.episode_int_reward_sum = numpy.zeros(self.envs_count)
      
        self.enable_training()
        self.iterations                     = 0 

        self.log_goals_count    = 0.0
        self.log_mode           = 0.0
        self.log_length         = 0.0
     

    def enable_training(self):
        self.enabled_training = True
 
    def disable_training(self):
        self.enabled_training = False

    def main(self): 
        #state to tensor
        states_t    = torch.tensor(self.states, dtype=torch.float).detach()

        #obtain q values for policy
        q_values_t  = self.model_dqn(states_t, self.goals_t)

        #sample actions depending on mode and q values
        actions = self._sample_actions(q_values_t, self.modes)

        #execute action
        states, rewards_ext, dones, infos = self.envs.step(actions)

        #update actions
        self.states = states.copy()

        self.episode_ext_reward_sum+= rewards_ext
        
        #TODO, use RND ? for rewards_int
        self.episode_int_reward_sum+= 0

        #store new goal, or update existing
        self.goals_buffer.add(states_t, self.episode_ext_reward_sum, self.episode_int_reward_sum)

        #check if goal reached
        goals_reached = self.goals_buffer.goals_reached(states_t, self.goals_t)

        #switch mode to explore, if goal reached
        for e in range(self.envs_count):
            if goals_reached[e]:
                self.mode[e] = True
       
        for e in range(self.envs_count): 
            if dones[e]:
                self.states[e]      = self.envs.reset(e).copy()

                #switch agent to GoMode
                self.mode[e]  = False

                #sample new goal
                goal = self.goals_buffer.get_goal(1.0, 1.0, 0.001)

                self.goals_t[e]                 = goal.clone()
                
                #clear score
                self.episode_ext_reward_sum[e]  = 0.0
                self.episode_int_reward_sum[e]  = 0.0

   
        #collect stats
        k = 0.02
        self.log_mode     = (1.0 - k)*self.log_mode + k*100.0*self.mode.mean()

        self.iterations+= 1
        return rewards_ext[0], dones[0], infos[0]
    
    def save(self, save_path):
        pass

    def load(self, load_path):
        pass

    def get_log(self): 
        result = "" 

        result+= str(round(self.goals_buffer.goals_ptr, 7)) + " "
        result+= str(round(self.log_mode, 7)) + " "
        result+= str(round(self.log_length, 7)) + " "

      

        return result 

    
    def _sample_actions(self):
        actions = numpy.zeros(self.envs_count, dtype=int)
        for e in range(self.envs_count):
            #agent GO mode
            actions_list     = self.buffer_actions_list[e]
            action_step_idx  = self.actions_step[e]
            if self.mode[e] == False and action_step_idx < len(actions_list):
                actions[e] = actions_list[action_step_idx]
                self.actions_step[e]+= 1
            #agent Explore mode
            else:
                actions[e]      = numpy.random.randint(0, self.actions_count)

        return actions
