from numpy.core.fromnumeric import reshape
import torch
import numpy

class GoalsBuffer: 

    def __init__(self, envs_count, buffer_size, add_threshold, reach_threshold, downsample, state_shape):

        self.buffer_size        = buffer_size
        self.add_threshold      = add_threshold
        self.reach_threshold    = reach_threshold
        self.downsample         = downsample

        self.goal_shape         = (1, state_shape[1]//self.downsample, state_shape[2]//self.downsample)

        self.downsampled_size   = self.goal_shape[0]*self.goal_shape[1]*self.goal_shape[2]

        #init buffers
        self.goals              = torch.zeros((self.buffer_size, self.downsampled_size))
        self.active_goals       = torch.ones((envs_count, self.buffer_size))

        self.downsample         = torch.nn.AvgPool2d(downsample, downsample)
        self.upsample           = torch.nn.Upsample(scale_factor=downsample, mode='nearest')

        self.goals_ptr = 1

        self.log_used_goals     = 0.0


    def step(self, states, dones):
        batch_size  = states.shape[0]
        states      = torch.from_numpy(states)
        dones       = torch.from_numpy(dones)

        states_down, dif   = self._preprocess(states)
        
        #select only used goals from buffer
        goals_used = self.goals[0:self.goals_ptr]

        distances   = torch.cdist(states_down, goals_used)

        distances_min, distances_ids = torch.min(distances, dim=1)

        #add new goal, if add threshold reached
        self._add_goals(states_down, distances_min, dif, dones)

        rewards     = torch.zeros(batch_size)

        #returning goals
        goals_result = torch.zeros((batch_size, self.downsampled_size))

        #select only actual goals
        active_goals  = self.active_goals[range(batch_size), distances_ids]

        #reward only if goal is active and reached close distance
        reached = 1.0*(distances_min < self.reach_threshold)
        rewards = active_goals*reached

        #clear reached and active goal flag
        self.active_goals[range(batch_size), distances_ids] = active_goals*(1.0 - reached)
        
        #set new goals - closest goals to given state
        goals_result[range(batch_size)] = goals_used[distances_ids].clone()

        self.log_used_goals             = len(goals_used)

        goals_result = goals_result.reshape((batch_size, ) + self.goal_shape)
        goals_result = self.upsample(goals_result)

        #add active goals flag
        size = int(self.buffer_size**0.5)
        active_goals = self.active_goals.reshape((batch_size, 1, size, size))
        active_goals = numpy.repeat(active_goals, states.shape[2]//size, axis=2)
        active_goals = numpy.repeat(active_goals, states.shape[3]//size, axis=3)
 
        return rewards.detach().to("cpu").numpy(), goals_result.detach().to("cpu").numpy(), active_goals


    def activate_goals(self, env_idx):
        self.active_goals[env_idx] = True

    def _preprocess(self, states):
        batch_size  = states.shape[0]

        s0_down = self.downsample(states[range(batch_size), 0].unsqueeze(1))
        s0_down = torch.flatten(s0_down, start_dim=1)

        s1_down = self.downsample(states[range(batch_size), 1].unsqueeze(1))
        s1_down = torch.flatten(s1_down, start_dim=1)

        dif     = torch.abs(s0_down - s1_down).mean(dim=1)
 
        return s0_down, dif

    def _add_goals(self, goals, distances_min, dif, dones):
        #add only new goal : long distance from existing goals
        #add only interesting goal : big change value
        candidates = (distances_min > self.reach_threshold)*(dif > self.add_threshold)*(1 - dones)
         
        indices = torch.where(candidates > 0)[0]

        if len(indices) > 0 and self.goals_ptr < self.buffer_size:
            idx = indices[0]

            self.goals[self.goals_ptr] = goals[idx].clone()
            self.goals_ptr = self.goals_ptr + 1
    