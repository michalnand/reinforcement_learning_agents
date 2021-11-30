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
        self.goals          = torch.zeros((self.buffer_size, self.downsampled_size))
        self.active_goals   = torch.ones((envs_count, self.buffer_size))
        self.active_goals[range(envs_count), 0] = 0.0

        self.downsample     = torch.nn.AvgPool2d(downsample, downsample)
        self.upsample       = torch.nn.Upsample(scale_factor=downsample, mode='nearest')

        self.goals_ptr      = 1

        self.log_used_goals = 0.0

    def step(self, states):
        batch_size  = states.shape[0]
        states      = torch.from_numpy(states)

        states_down, dif   = self._preprocess(states)
        
        #select only used goals from buffer
        goals_used = self.goals[0:self.goals_ptr]

        distances   = torch.cdist(states_down, goals_used)

        distances_min, distances_ids = torch.min(distances, dim=1)

        #select only actual goals
        active_goals  = self.active_goals[range(batch_size), distances_ids]

        #reward only if goal is active and reached close distance
        reached = active_goals*(distances_min < self.reach_threshold)
        rewards = 1.0*(reached > 0.0)

        #clear reached goal to inactive state
        self.active_goals[range(batch_size), distances_ids] = active_goals*(1.0 - reached)

        #flag for already reached goals
        size            = int(self.buffer_size**0.5) 
        current_active  = 1.0 - active_goals.reshape((batch_size, 1, size, size))
        current_active  = torch.repeat_interleave(current_active, repeats=states.shape[2]//size, dim=2)
        current_active  = torch.repeat_interleave(current_active, repeats=states.shape[3]//size, dim=3)
        
        #returning goals
        goals_result = torch.zeros((batch_size, self.downsampled_size))
        goals_result[range(batch_size)] = goals_used[distances_ids].clone()

        
        goals_result = goals_result.reshape((batch_size, ) + self.goal_shape)
        goals_result = self.upsample(goals_result)

        #add new goal, if add threshold reached
        self._add_goals(states_down, distances_min, dif)
 
        return rewards.detach().to("cpu").numpy(), goals_result.detach().to("cpu").numpy(), current_active.detach().to("cpu").numpy()


    def activate_goals(self, env_idx):
        self.active_goals[env_idx]      = 1.0
        self.active_goals[env_idx][0]   = 0.0


    def get_goals_for_render(self):
        goals  = self.goals.reshape((self.buffer_size, ) + self.goal_shape)
 
        active = self.active_goals.unsqueeze(2).unsqueeze(3)
        active = torch.tile(active, (1, self.goal_shape[1], self.goal_shape[2]))

        return goals.detach().to("cpu").numpy(), active.detach().to("cpu").numpy()

    def save(self, path):
        numpy.save(path + "goals.npy", self.goals)

    def load(self, path):
        numpy.load(path + "goals.npy", self.goals)


    def _preprocess(self, states):
        batch_size  = states.shape[0]

        s0_down = self.downsample(states[range(batch_size), 0].unsqueeze(1))
        s0_down = torch.flatten(s0_down, start_dim=1)

        s1_down = self.downsample(states[range(batch_size), 1].unsqueeze(1))
        s1_down = torch.flatten(s1_down, start_dim=1)

        dif     = torch.abs(s0_down - s1_down).mean(dim=1)
 
        return s0_down, dif

    def _add_goals(self, goals, distances_min, dif):
        #add only new goal          : long distance from existing goals
        #add only interesting goal  : big value change
        candidates  = (distances_min > self.reach_threshold)*(dif > self.add_threshold)
        
        indices     = torch.where(candidates > 0)[0]

        for idx in indices:
            if self.goals_ptr < self.buffer_size:
                self.goals[self.goals_ptr] = goals[idx].clone()
                self.goals_ptr = self.goals_ptr + 1

                self.log_used_goals = self.goals_ptr
        