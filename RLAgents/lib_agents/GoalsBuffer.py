from numpy.core.fromnumeric import reshape
import torch
import numpy
 
class GoalsBuffer:  

    def __init__(self, buffer_size, add_threshold, downsample, state_shape):

        self.buffer_size        = buffer_size
        self.add_threshold      = add_threshold
        self.downsample         = downsample

        self.goal_shape         = (1, state_shape[1]//self.downsample, state_shape[2]//self.downsample)

        self.downsampled_size   = self.goal_shape[0]*self.goal_shape[1]*self.goal_shape[2]

        #init buffers
        self.goals          = torch.zeros((self.buffer_size, self.downsampled_size), dtype=torch.float32)
        self.steps          = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.rewards        = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.rewards_sum    = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.visited_count  = torch.zeros(self.buffer_size, dtype=torch.float32)

        self.downsample     = torch.nn.AvgPool2d(downsample, downsample)
        self.upsample       = torch.nn.Upsample(scale_factor=downsample, mode='nearest')

        self.goals_ptr      = 0

    def step(self, states, current_goals, steps, rewards, rewards_sum):
        batch_size  = states.shape[0]
        states      = torch.from_numpy(states).float()

        states_down = self._preprocess(states)

        #add initial goal
        if self.goals_ptr == 0:
            self.goals[self.goals_ptr]          = states_down[0].clone()
            self.steps[self.goals_ptr]          = steps[0]
            self.rewards[self.goals_ptr]        = rewards[0]
            self.rewards_sum[self.goals_ptr]    = rewards_sum[0]
            self.visited_count[self.goals_ptr]  = 1

            self.goals_ptr+= 1
        
        #select only used goals from buffer
        goals_used  = self.goals[0:self.goals_ptr]

        distances   = torch.cdist(states_down, goals_used)

        #select closest
        distances_min, distances_ids = torch.min(distances, dim=1)


        steps_reward, bigger_reward, bigger_sum_reward, visited_reward = self._update_existing(distances_min, distances_ids, steps, rewards, rewards_sum)

        internal_motivation = steps_reward + bigger_reward + bigger_sum_reward + visited_reward

        #add new goals if any
        self._add_new(distances_min, distances_ids, states_down, steps, rewards, rewards_sum)
 
        new_goals, reached_reward = self._detect_reached(current_goals, internal_motivation)

        internal_motivation+= reached_reward

        return new_goals, internal_motivation

    def _update_existing(self, distances_min, distances_ids, steps, rewards, rewards_sum, k = 0.1):
        
        candidates      = torch.where(distances_min <= self.add_threshold)[0]

        batch_size      = distances_min.shape[0]

        steps_reward        = torch.zeros(batch_size, dtype=torch.float32)
        bigger_reward       = torch.zeros(batch_size, dtype=torch.float32)
        bigger_sum_reward   = torch.zeros(batch_size, dtype=torch.float32)
        visited_reward      = torch.zeros(batch_size, dtype=torch.float32)

        for i in range(len(candidates)):
            goal_idx    = distances_ids[i]
            source_idx  = candidates[i]

            #udpate steps count if less 
            if self.steps[goal_idx] > steps[source_idx]:
                self.steps[goal_idx] = (1.0 - k)*self.steps[goal_idx] + k*steps[source_idx]
                steps_reward[source_idx] = 1.0

            #update rewards if bigger
            if self.rewards[goal_idx] < rewards[source_idx]:
                self.rewards[goal_idx] = rewards[source_idx]
                bigger_reward[source_idx] = 1.0

            #update rewards_sum if bigger
            if self.rewards_sum[goal_idx] < rewards_sum[source_idx]:
                self.rewards_sum[goal_idx] = rewards_sum[source_idx]
                bigger_sum_reward[source_idx] = 1.0

            #increment visited
            self.visited_count[goal_idx]+= 1
            visited_reward[source_idx] = 1.0/(1.0 + (self.visited_count[goal_idx]**0.5))

        return steps_reward, bigger_reward, bigger_sum_reward, visited_reward


    def _add_new(self, distances_min, states_down, steps, rewards, rewards_sum):
        candidates = torch.where(distances_min > self.add_threshold)[0]

        for i in range(len(candidates)):
            if self.goals_ptr < self.buffer_size:
                idx = candidates[i]

                self.goals[self.goals_ptr]          = states_down[idx].clone()
                self.steps[self.goals_ptr]          = steps[idx]
                self.rewards[self.goals_ptr]        = rewards[idx]
                self.rewards_sum[self.goals_ptr]    = rewards_sum[idx]
                self.visited_count[self.goals_ptr]  = 1
                
                self.goals_ptr+= 1

    def _detect_reached(self, states_down, current_goals, internal_motivation):
        new_goals = current_goals.clone()

        goals_down = self._preprocess(current_goals)

        reached_reward = torch.zeros(current_goals.shape[0], dtype=torch.float32)



        return new_goals, reached_reward


    def get_goals_for_render(self):
        goals  = self.goals.reshape((self.buffer_size, ) + self.goal_shape)
 
        grid_size   = int(self.buffer_size**0.5) 
        goal_height = self.goal_shape[1]
        goal_width  = self.goal_shape[2]

        active  = self.active_goals.unsqueeze(2).unsqueeze(3)
        active  = torch.tile(active, (1, goal_height, goal_width))

        goals   = goals*(1.0 + 100*active)/2.0
        goals   = goals.detach().to("cpu").numpy()

        goals_result  = numpy.zeros((grid_size*goal_height, grid_size*goal_width))

        for y in range(grid_size):
            for x in range(grid_size):
                y_ = y*goal_height
                x_ = x*goal_width
                goals_result[y_:y_+goal_height, x_:x_+goal_width]   = goals[y*grid_size + x][0]

        goals_result = goals_result/(numpy.max(goals_result) + 0.00001)

        #flags for already reached goals
        size            = int(self.buffer_size**0.5) 
        current_active  = 1.0 - self.active_goals[0].reshape((size, size))

        current_active  = torch.repeat_interleave(current_active, repeats=goals_result.shape[0]//size, dim=0)
        current_active  = torch.repeat_interleave(current_active, repeats=goals_result.shape[1]//size, dim=1)
        current_active  = current_active.detach().to("cpu").numpy()

        goals_result = goals_result*(1.0 + current_active)/2.0

        return goals_result

    def save(self, path):
        numpy.save(path + "goals.npy", self.goals.detach().to("cpu").numpy())

    def load(self, path):
        self.goals = torch.from_numpy(numpy.load(path + "goals.npy"))
        
        #move goals pointer to last non-used position
        self.goals_ptr = 0
        for i in range(len(self.goals)):
            v  = self.goals[i].sum()
            self.goals_ptr+= 1
            if v < 0.001:
                break


    def _preprocess(self, states):
        batch_size  = states.shape[0]

        result = self.downsample(states[range(batch_size), 0].unsqueeze(1))
        result = torch.flatten(result, start_dim=1)
 
        return result

 