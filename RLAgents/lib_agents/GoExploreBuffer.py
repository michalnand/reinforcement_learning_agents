import torch
import numpy


class GoExploreBuffer:

    def __init__(self, envs_count, buffer_size, state_shape, reached_threshold, add_threshold, downsample):

        self.envs_count         = envs_count
        self.buffer_size        = buffer_size
        self.reached_threshold  = reached_threshold
        self.add_threshold      = add_threshold
        
        #list of actions leading to goal
        self.actions        = [[] for i in range(self.buffer_size)]

        #accumulated reward
        self.rewards_sum    = numpy.zeros(self.buffer_size)

        #visited counter
        self.visited_count  = numpy.zeros(self.buffer_size)

        #goals buffer
        self.goal_shape             = (1, state_shape[1], state_shape[2])
        self.goal_downsampled_shape = (1, state_shape[1]//downsample, state_shape[2]//downsample)
        self.downsampled_size       = numpy.prod(self.goal_downsampled_shape)


        self.current_goals_ids  = numpy.zeros(self.envs_count, dtype=int)
        self.goals_buffer       = torch.zeros((self.buffer_size, self.downsampled_size), dtype=torch.float32)


        self.downsample     = torch.nn.AvgPool2d(downsample, downsample)
        self.upsample       = torch.nn.Upsample(scale_factor=downsample, mode='nearest')

        self.goals_ptr      = 0


    def _preprocess(self, states):
        s_down   = self.downsample(states)
        s0_down  = torch.flatten(s_down[:,0], start_dim=1)
   
        return s0_down


    def step(self, states, actions, rewards_sum):
        states_t    = torch.from_numpy(states).float()

        #detect if goal reached
        states_down = self._preprocess(states_t)

        #add initial goal
        if self.goals_ptr == 0:
            self._add_goal(states_down[0], actions[0], rewards_sum[0])

        
        #select only used goals
        goals_used  = self.goals_buffer[0:self.goals_ptr]

        #each from each distance, shape (batch_size, self.goals_ptr)
        distances   = torch.cdist(states_down, goals_used)

        #find closest
        distances_min, distances_ids = torch.min(distances, dim=1)

        self._add_new_goal(distances_min, distances_ids, states_down, actions, rewards_sum)
        self._update_existing(distances_min, distances_ids, states_down, actions, rewards_sum)

        reached_goals = (distances_min < self.reached_threshold)

        return reached_goals

    def _add_new_goal(self, distances_min, distances_ids, states_down, actions, rewards_sum):
        candidates = numpy.where(distances_min > self.add_threshold)[0]

        for idx in candidates:
            self._add_goal(states_down[idx], actions[idx], rewards_sum[idx])

    def _update_existing(self, distances_min, distances_ids, states_down, actions, rewards_sum):
        batch_size = rewards_sum.shape[0]

        for b in range(batch_size):
            goal_idx = distances_ids[b]

            #udpate if shorted path found, but only if final reward not damaged
            if len(actions[b]) < len(self.actions[goal_idx]) and rewards_sum[b] >= self.rewards_sum[goal_idx]:
                self.actions[goal_idx] = actions[b].copy()

            #udpate better reward
            if rewards_sum[b] > self.rewards_sum[goal_idx]:
                self.rewards_sum[goal_idx] = rewards_sum[b]

            self.visited_count[goal_idx]+= 1
  
    #sample new random goal
    #probs depends on acumulated reward, and visited count
    def new_goal(self, env_id, k = 0.9):
        visited_reward  = 1.0/(1.0 + (self.visited_count**0.5))
        
        #compute probs
        probs_rewards_sum       = self._softmax(self.rewards_sum[0:self.goals_ptr])
        probs_visited_reward    = self._softmax(visited_reward[0:self.goals_ptr])

        #final prob
        probs =  k*probs_rewards_sum + (1.0 - k)*probs_visited_reward
        
        idx   = numpy.random.choice(range(self.goals_ptr), size=1, p = probs)[0]

        tmp   = self.goals_buffer[idx].reshape(self.goal_downsampled_shape).unsqueeze(0)

        goal  = self.upsample(tmp).squeeze(0)

        self.current_goals_ids[env_id] = idx

        return goal, self.actions[idx], idx
    
    def _softmax(self, values):
        probs = numpy.exp(values - numpy.max(values))
        probs = probs/numpy.sum(probs)

        return probs

    def _add_goal(self, state_down, actions, reward_sum):
        if self.goals_ptr < self.buffer_size:
            self.goals_buffer[self.goals_ptr] = state_down.clone()

            self.actions[self.goals_ptr]        = actions
            self.rewards_sum[self.goals_ptr]    = reward_sum
            self.visited_count[self.goals_ptr]  = 1

            self.goals_ptr+= 1

    def save(self, path):

        numpy.save(path + "gb_rewards_sum.npy", self.rewards_sum)
        numpy.save(path + "gb_visited_count.npy", self.visited_count)
        numpy.save(path + "gb_goals.npy", self.goals_buffer.detach().to("cpu").numpy())
        
        file = open("gb_actions.txt", "w")
        for j in range(self.goals_ptr):
            for value in self.actions[j]:
                file.write(int(value), " ")
            file.write("\n")
        file.close()

    def load(self, path):
        self.rewards_sum    = numpy.load(path + "gb_rewards_sum.npy")
        self.visited_count  = numpy.load(path + "gb_visited_count.npy")
        self.goals   = torch.from_numpy(numpy.load(path + "gb_goals.npy"))

        #move goals pointer to last non-used position
        self.goals_ptr = 0
        for i in range(len(self.goals)):
            v  = self.goals[i].sum()
            self.goals_ptr+= 1
            if v < 0.001:
                break

        #TODO
        '''
        file = open("gb_actions.txt", "r")
        for j in range(self.buffer_size):
            values = []
            for value in self.actions[j]:
                file.write(int(value), " ")
            file.write("\n")
        file.close()
        '''

    def get_goals_for_render(self):
        goals       = self.goals_buffer.reshape((self.buffer_size, ) + self.goal_downsampled_shape)
        goals       = goals.detach().to("cpu").numpy()

        grid_size   = int(self.buffer_size**0.5) 
        goal_height = self.goal_downsampled_shape[1]
        goal_width  = self.goal_downsampled_shape[2]

        goals_result  = numpy.zeros((grid_size*goal_height, grid_size*goal_width))

        for y in range(grid_size):
            for x in range(grid_size):
                y_ = y*goal_height
                x_ = x*goal_width
                goals_result[y_:y_+goal_height, x_:x_+goal_width]   = goals[y*grid_size + x][0]

        goals_result = goals_result/(numpy.max(goals_result) + 0.00001)

      
        return goals_result


