from numpy.core.fromnumeric import reshape
import torch
import numpy
 
class GoalsBuffer:  

    def __init__(self, envs_count, buffer_size, agent_goals_count, add_threshold, reach_threshold, downsample, state_shape):

        self.buffer_size        = buffer_size
        self.agent_goals_count  = agent_goals_count
        self.add_threshold      = add_threshold
        self.reach_threshold    = reach_threshold
        self.downsample         = downsample

        self.goal_shape         = (1, state_shape[1]//self.downsample, state_shape[2]//self.downsample)
        self.goals_shape        = (self.agent_goals_count, state_shape[1]//self.downsample, state_shape[2]//self.downsample)
        self.downsampled_size   = self.goal_shape[0]*self.goal_shape[1]*self.goal_shape[2]

        #init buffers
        self.goals          = torch.zeros((self.buffer_size, self.downsampled_size), dtype=torch.float32)
        self.active_goals   = torch.ones((envs_count, self.buffer_size), dtype=torch.float32)
        self.active_goals[range(envs_count), 0] = 0.0

        #adjacency matrix, with connections counting
        self.am             = numpy.zeros((self.buffer_size, self.buffer_size), dtype=int)

        self.closest_ids_prev   = numpy.zeros(envs_count, dtype=int)
        self.closest_ids        = numpy.zeros(envs_count, dtype=int)
        
        self.downsample     = torch.nn.AvgPool2d(downsample, downsample)
        self.upsample       = torch.nn.Upsample(scale_factor=downsample, mode='nearest')

        self.goals_ptr      = 1
        self.log_used_goals = 0


        

    def step(self, states):
        batch_size  = states.shape[0]
        states      = torch.from_numpy(states).float()

        states_down, dif   = self._preprocess(states)

        #add initial goal
        if self.goals_ptr == 1:
            self.goals[self.goals_ptr] = states_down[0].clone()
            self.goals_ptr+= 1

        #first goals (zeros), is always active
        self.active_goals[:,0] = 1.0
        
        #select only used goals from buffer
        goals_used  = self.goals[0:self.goals_ptr]

        distances   = torch.cdist(states_down, goals_used)

        self.closest_ids_prev           = self.closest_ids.copy()
        closest_distances, closest_ids  = torch.min(distances, dim=1)
        self.closest_ids                = closest_ids.detach().to("cpu").numpy()

        #update connections graph matrix
        for i in range(batch_size):
            self.am[self.closest_ids_prev[i]][self.closest_ids[i]]+= 1

        #compute reward from active goals
        #select only active goals
        active_goals  = self.active_goals[range(batch_size), self.closest_ids]

        #reward only if goal is active and reached close distance
        reached = active_goals*(closest_distances < self.reach_threshold)
        rewards = 1.0*(reached > 0.0)

        #clear reached goal to inactive state
        self.active_goals[range(batch_size), self.closest_ids] = active_goals*(1.0 - reached)

         



        #returning goals
        goals_result = torch.zeros((batch_size, self.agent_goals_count, self.downsampled_size))

        #clear diagonal, to avoid self goals
        am_cleared = self.am.copy()
        numpy.fill_diagonal(am_cleared, 0)

        #from graph matrix, select only active goals ids
        goals_counts_candidates = self.active_goals*am_cleared[self.closest_ids]

        #find indices where the highest counts presents
        #argsort returs in ascending order, so reoder it, and select N-best
        goals_ids       = numpy.argsort(goals_counts_candidates, axis=1)[:,-self.agent_goals_count:]

        goals_result    = self.goals[goals_ids, :]
        
        goals_result    = goals_result.reshape((batch_size, ) + self.goals_shape)
        goals_result    = self.upsample(goals_result)

        #add new goal, if add threshold reached
        self._add_goals(states_down, closest_distances, dif)
 

        return rewards.detach().to("cpu").numpy(), goals_result.detach().to("cpu").numpy()


    def activate_goals(self, env_idx):
        self.active_goals[env_idx]      = 1.0
        self.active_goals[env_idx][0]   = 0.0


    def get_for_render(self):
        goals  = self.goals.reshape((self.buffer_size, ) + self.goal_shape)
        active = self.active_goals[0]

        active = active.reshape((active.shape[0], 1, 1, 1))

        grid_size   = int(self.buffer_size**0.5) 
        goal_height = self.goals_shape[1]
        goal_width  = self.goals_shape[2]

        goals = goals*(1.0 + active)/2.0

        goals_result  = numpy.zeros((grid_size*goals.shape[2], grid_size*goals.shape[3]))

        for y in range(grid_size):
            for x in range(grid_size):
                y_ = y*goal_height
                x_ = x*goal_width
                goals_result[y_:y_+goal_height, x_:x_+goal_width]   = goals[y*grid_size + x][0]

        #clear diagonal, to avoid self goals
        am_cleared = self.am.copy()
        numpy.fill_diagonal(am_cleared, 0)

        am_norm = am_cleared/(numpy.max(am_cleared) + 0.0000001)
 
        return goals_result, am_norm

    def save(self, path):
        numpy.save(path + "gb_goals.npy", self.goals.detach().to("cpu").numpy())
        numpy.save(path + "gb_am.npy", self.am)

    def load(self, path):
        self.goals = torch.from_numpy(numpy.load(path + "goals.npy"))
        self.am    = numpy.load(path + "am.npy")
        
        #move goals pointer to last non-used position
        self.goals_ptr = 0
        for i in range(len(self.goals)):
            v  = self.goals[i].sum()
            self.goals_ptr+= 1
            if v < 0.001:
                break


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