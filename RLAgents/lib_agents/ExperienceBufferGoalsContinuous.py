import numpy
import torch

class ExperienceBufferGoalsContinuous():
    def __init__(self, size, state_shape, goal_shape, actions_count, max_episode_length = 4096):

        self.size           = size       
        self.current_idx    = 0 
        self.initialized    = False

        self.state_shape        = state_shape
        self.goal_shape         = goal_shape
        self.actions_count      = actions_count

        self.max_episode_length = max_episode_length

    def _initialize(self):
        if self.initialized == False:
            self.state_b            = numpy.zeros((self.size, ) + self.state_shape, dtype=numpy.float32)
            self.achieved_goal_b    = numpy.zeros((self.size, ) + self.goal_shape, dtype=numpy.float32)
            self.desired_goal_b     = numpy.zeros((self.size, ) + self.goal_shape, dtype=numpy.float32)
            self.action_b       = numpy.zeros((self.size, self.actions_count), dtype=numpy.float32)
            self.reward_ext_b   = numpy.zeros((self.size, ), dtype=numpy.float32)
            self.reward_int_b   = numpy.zeros((self.size, ), dtype=numpy.float32)
            self.done_b         = numpy.zeros((self.size, ), dtype=numpy.float32)

            self.episode_length = 0 

            self.initialized    = True

            

    def add(self, state, achieved_goal, desired_goal, action, reward_ext, reward_int, done): 
        self._initialize()

        if done != 0: 
            done_ = 1.0
        else:
            done_ = 0.0

        self.state_b[self.current_idx]          = state.copy()
        self.achieved_goal_b[self.current_idx]  = achieved_goal.copy()
        self.desired_goal_b[self.current_idx]   = desired_goal.copy()
        self.action_b[self.current_idx]     = action.copy()
        self.reward_ext_b[self.current_idx] = reward_ext
        self.reward_int_b[self.current_idx] = reward_int
        self.done_b[self.current_idx]       = done_

        self.episode_length+= 1
        
        if done != 0:
            #use last episode step achieved_goal as desired goal for whole episode    
            episode_start = self.current_idx - self.episode_length

            #select random episode achived goal as desired goal
            goal_idx      = numpy.random.randint(0, self.episode_length) + episode_start

            for i in range(self.episode_length):
                idx = (i + episode_start)%self.size
                self.desired_goal_b[idx] = self.achieved_goal_b[goal_idx].copy()
        
            #set reward +1 
            self.reward_ext_b[goal_idx] = 1.0    
            self.done_b[goal_idx]       = 1.0
            
            self.episode_length = 0

        self.current_idx = (self.current_idx + 1)%self.size



    def sample(self, batch_size, device = "cpu"):
        indices         = numpy.random.randint(0, self.size, size=batch_size)
        indices_next    = (indices + 1)%self.size 


        state_t         = torch.from_numpy(numpy.take(self.state_b,         indices, axis=0)).to(device)
        achieved_goal_t = torch.from_numpy(numpy.take(self.achieved_goal_b, indices, axis=0)).to(device)
        desired_goal_t  = torch.from_numpy(numpy.take(self.desired_goal_b,  indices, axis=0)).to(device)

        state_next_t         = torch.from_numpy(numpy.take(self.state_b,         indices_next, axis=0)).to(device)
        achieved_goal_next_t = torch.from_numpy(numpy.take(self.achieved_goal_b, indices_next, axis=0)).to(device)
        desired_goal_next_t  = torch.from_numpy(numpy.take(self.desired_goal_b,  indices_next, axis=0)).to(device)


        action_t        = torch.from_numpy(numpy.take(self.action_b,        indices, axis=0)).to(device)
        reward_ext_t    = torch.from_numpy(numpy.take(self.reward_ext_b,    indices, axis=0)).to(device)
        reward_int_t    = torch.from_numpy(numpy.take(self.reward_int_b,    indices, axis=0)).to(device)
        done_t          = torch.from_numpy(numpy.take(self.done_b,          indices, axis=0)).to(device)

        return state_t, achieved_goal_t, desired_goal_t, state_next_t, achieved_goal_next_t, desired_goal_next_t, action_t, reward_ext_t, reward_int_t, done_t
