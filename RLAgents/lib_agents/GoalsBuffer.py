import torch
import numpy

class GoalsBuffer:
    def __init__(self, envs_count, buffer_size, reach_threshold, change_threshold, downsample, state_shape):
        
        self.buffer_size            = buffer_size
        self.reach_threshold        = reach_threshold
        self.change_threshold       = change_threshold
 
        self.goal_shape             = (1, state_shape[1], state_shape[2])
        self.goal_downsampled_shape = (1, state_shape[1]//downsample, state_shape[2]//downsample)
        self.downsampled_size       = numpy.prod(self.goal_downsampled_shape)

        
        self.current_goals  = torch.zeros((envs_count, ) + self.goal_shape,     dtype=torch.float32)
        self.goals          = torch.zeros((buffer_size, self.downsampled_size), dtype=torch.float32)

        self.rewards_sum    = numpy.zeros(buffer_size)
        self.visited_count  = numpy.zeros(buffer_size)

        self.current_goals_ids  = numpy.zeros(envs_count, dtype=int)
        self.reached_goals      = numpy.zeros((envs_count, buffer_size))

        #initial state is always reached
        self.reached_goals[range(envs_count), 0] = 1.0

        self.downsample     = torch.nn.AvgPool2d(downsample, downsample)
        self.upsample       = torch.nn.Upsample(scale_factor=downsample, mode='nearest')

        self.goals_ptr      = 0

    def step(self, states, rewards_sum):
        batch_size  = states.shape[0]
        states_t    = torch.from_numpy(states).float()

        #downsample
        states_down, dif        = self._preprocess(states_t)
        current_goals_down, _   = self._preprocess(self.current_goals)

        #initial run, add new goal
        if self.goals_ptr == 0:
            self._add_goal(states_down[0], rewards_sum[0])


        
        #select only used goals from buffer
        goals_used  = self.goals[0:self.goals_ptr]

        #each from each distance, shape (batch_size, self.goals_ptr)
        distances   = torch.cdist(states_down, goals_used)

        #find closest
        distances_min, distances_ids = torch.min(distances, dim=1)

        #update
        bigger_sum_reward, visited_reward = self._udpate_goals(distances_min, distances_ids, dif, states_down, rewards_sum)

        #find if closest goals are active (non reached)
        active_goals    = 1.0 - self.reached_goals[range(batch_size), distances_ids]

        #check if reached goal
        distances       = ((current_goals_down - states_down)**2).mean(dim=1)
        distances       = distances.detach().to("cpu").numpy()

        #reached can be only active goals
        reached_goals   = active_goals*(distances <= self.reach_threshold)


        self._set_reached_goals(reached_goals, distances_ids)


        im = bigger_sum_reward + visited_reward  + reached_goals

        '''
        if (reached_goals[0] > 0):
            print("bigger_sum_reward = ", bigger_sum_reward[0])
            print("visited_reward    = ", visited_reward[0])
            print("reached_goals     = ", reached_goals[0])
            print("\n\n")
        '''

        grid_size = int(self.buffer_size**0.5)
        
        reached = numpy.reshape(self.reached_goals, (batch_size, 1, grid_size, grid_size))
        reached = numpy.repeat(reached, self.goal_shape[1]//grid_size, axis=2)
        reached = numpy.repeat(reached, self.goal_shape[2]//grid_size, axis=3)

        return self.current_goals, reached, im, reached_goals

    #sample new goal
    def reset(self, env_id):
        self.current_goals[env_id], self.current_goals_ids[env_id] = self._new_goal(env_id)
        self.reached_goals[env_id]      = 0.0
        self.reached_goals[env_id][0]   = 1.0
    
    def _add_goal(self, state_down, reward_sum):
        if self.goals_ptr < self.buffer_size:
            self.goals[self.goals_ptr] = state_down.clone()

            self.rewards_sum[self.goals_ptr]    = reward_sum
            self.visited_count[self.goals_ptr]  = 1

            self.goals_ptr+= 1

    def _set_reached_goals(self, reached_goals, distances_ids):
        #find indices where reached goal (mostly sparse)
        indices     = numpy.where(reached_goals > 0)[0]

        for idx in indices:
            #set reached flag
            self.reached_goals[idx][distances_ids[idx]] = 1.0

            #select new goal
            self.current_goals[idx], self.current_goals_ids[idx] = self._new_goal(idx)

    def _softmax(self, values):
        probs = numpy.exp(values - numpy.max(values))
        probs = probs/numpy.sum(probs)

        return probs

    #sample new random goal
    #probs depends on reward, acumulated reward, and visited count
    def _new_goal(self, env_id):
        visited_reward  = 1.0/(1.0 + (self.visited_count**0.5))

        #mask for active goals only
        active_mask     = (1.0 - self.reached_goals[env_id])[0:self.goals_ptr]

        #compute probs
        probs_rewards_sum       = self._softmax(active_mask*self.rewards_sum[0:self.goals_ptr])
        probs_visited_reward    = self._softmax(active_mask*visited_reward[0:self.goals_ptr])

        #final prob
        probs =  (probs_rewards_sum + probs_visited_reward)/2.0
        
        idx   = numpy.random.choice(range(self.goals_ptr), size=1, p = probs)

        tmp   = self.goals[idx].reshape(self.goal_downsampled_shape).unsqueeze(0)

        goal  = self.upsample(tmp).squeeze(0)

        return goal, idx

    #add new if no close reward
    #compute intrisic reward if any
    def _udpate_goals(self, distances_min, distances_ids, dif, states_down, rewards_sum):
        batch_size = states_down.shape[0]

        #add new goal if non exist yet
        for i in range(batch_size):
            if distances_min[i] > self.reach_threshold and dif[i] > self.change_threshold:
                self._add_goal(states_down[i], rewards_sum[i])
 
        bigger_sum_reward   = numpy.zeros(batch_size)
        visited_reward      = numpy.zeros(batch_size)
        for i in range(batch_size):
            goal_idx    = distances_ids[i]

            #update rewards_sum if bigger
            if self.rewards_sum[goal_idx] < rewards_sum[i]:
                self.rewards_sum[goal_idx] = rewards_sum[i]
                bigger_sum_reward[i] = 1.0

            #increment visited
            self.visited_count[goal_idx]+= 1
            
            #visited reward
            visited_reward[i] = 1.0/(1.0 + (self.visited_count[goal_idx]**0.5))
        
        return bigger_sum_reward, visited_reward

    def _preprocess(self, states):
        s_down   = self.downsample(states)
        s0_down  = torch.flatten(s_down[:,0], start_dim=1)

        if states.shape[1] > 1:
            s1_down  = torch.flatten(s_down[:,1], start_dim=1)
        else:
            s1_down  = s0_down

        dif     = torch.abs(s0_down - s1_down).mean(dim=1)
 
        return s0_down, dif


    def save(self, path):
        numpy.save(path + "gb_goals.npy", self.goals.detach().to("cpu").numpy())
        numpy.save(path + "gb_rewards_sum.npy", self.rewards_sum)
        numpy.save(path + "gb_visited_count.npy", self.visited_count)


    def load(self, path):   
        return   
        self.goals   = torch.from_numpy(numpy.load(path + "goals.npy"))

        self.rewards_sum    = numpy.load(path + "gb_rewards_sum.npy")
        self.visited_count  = numpy.load(path + "gb_visited_count.npy")

        #move goals pointer to last non-used position
        self.goals_ptr = 0
        for i in range(len(self.goals)):
            v  = self.goals[i].sum()
            self.goals_ptr+= 1
            if v < 0.001:
                break

    def get_goals_for_render(self):
        goals       = self.goals.reshape((self.buffer_size, ) + self.goal_downsampled_shape)
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

