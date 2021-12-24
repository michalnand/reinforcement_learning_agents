import torch
import numpy

class GoalsBuffer:
    def __init__(self, envs_count, buffer_size, goals_add_threshold, reach_threshold, change_threshold, downsample, state_shape):
        self.buffer_size            = buffer_size
        self.goals_add_threshold    = goals_add_threshold
        self.reach_threshold        = reach_threshold
        self.change_threshold       = change_threshold
 
        self.goal_shape             = (1, state_shape[1], state_shape[2])
        self.goal_downsampled_shape = (1, state_shape[1]//downsample, state_shape[2]//downsample)
        self.downsampled_size       = numpy.prod(self.goal_downsampled_shape)

        #all goals stored in buffer
        self.goals_buffer       = torch.zeros((buffer_size, self.downsampled_size), dtype=torch.float32) 

        #current goals for reach
        self.active_goals       = torch.zeros((envs_count, ) + self.goal_shape,     dtype=torch.float32)
        self.active_goals_ids   = numpy.zeros(envs_count, dtype=int)

        #flags of non reached goals
        self.active_goals_flag  = numpy.ones((envs_count, buffer_size))

        #initial state is always reached, non active
        self.active_goals_flag[range(envs_count), 0] = 0.0


        self.visited_count      = numpy.zeros(buffer_size)
        

        self.goals_ids_prev     = numpy.zeros(envs_count, dtype=int)
        self.goals_ids_now      = numpy.zeros(envs_count, dtype=int)

        self.adjacency_matrix   = numpy.zeros((buffer_size, buffer_size))

        self.downsample     = torch.nn.AvgPool2d(downsample, downsample)
        self.upsample       = torch.nn.Upsample(scale_factor=downsample, mode='nearest')

        self.goals_ptr      = 0

    def step(self, states):
        batch_size  = states.shape[0]
        states_t    = torch.from_numpy(states).float()

        #downsample
        states_down, dif        = self._preprocess(states_t)
        active_goals_down, _    = self._preprocess(self.active_goals)

        #initial run, add new goal
        if self.goals_ptr == 0:
            self._add_goal(states_down[0])


        #select only used goals from buffer
        goals_used  = self.goals_buffer[0:self.goals_ptr]

        #each from each distance, shape (batch_size, self.goals_ptr)
        distances   = torch.cdist(states_down, goals_used)

        #find closest
        distances_min, distances_ids = torch.min(distances, dim=1)

        distances_ids = distances_ids.detach().to("cpu").numpy()

        self.goals_ids_prev     = self.goals_ids_now.copy()
        self.goals_ids_now      = distances_ids.copy()
      
        #add new goal if non exist yet
        #goal is big state change
        for i in range(batch_size):
            if distances_min[i] > self.goals_add_threshold and dif[i] > self.change_threshold:
                self._add_goal(states_down[i])

                #add new connection
                self.adjacency_matrix[self.goals_ids_prev[i]][self.goals_ptr-1]+= 1.0
            else:
                #update existing connection
                self.adjacency_matrix[self.goals_ids_prev[i]][self.goals_ids_now[i]]+= 1.0

        #increment visited count
        for i in range(batch_size):
            self.visited_count[distances_ids[i]]+= 1

     
        #check if reached any goal
        distances       = ((active_goals_down - states_down)**2).mean(dim=1)
        distances       = distances.detach().to("cpu").numpy()

        #reached can be only active goal
        reached_reward     = self.active_goals_flag[range(batch_size), distances_ids]*(distances <= self.reach_threshold)

        #rewards for connections
        tmp                = (self.adjacency_matrix > 0).sum(axis=1)
        connections_reward = tmp[distances_ids]
        connections_reward = connections_reward/(numpy.max(tmp) + 0.00000001)

        #final reward, combine target reaching with target importance
        rewards = reached_reward*connections_reward

        #clear flag, goal can't be reached again
        self.active_goals_flag[range(batch_size), distances_ids] = 0

        #generate new goal if ACTIVE goal reached
        reached_active = numpy.logical_and(distances <= self.reach_threshold, self.active_goals_ids == distances_ids)
        for i in range(batch_size):
            if reached_active[i]:
                self.active_goals[i], self.active_goals_ids[i] = self._new_goal()

        grid_size       = int(self.buffer_size**0.5)
        reached_flag    = 1.0 - self.active_goals_flag 
        
        #reshape to grid
        reached_flag = numpy.reshape(reached_flag, (batch_size, 1, grid_size, grid_size))
        reached_flag = numpy.repeat(reached_flag, self.goal_shape[1]//grid_size, axis=2)
        reached_flag = numpy.repeat(reached_flag, self.goal_shape[2]//grid_size, axis=3)

        return self.active_goals, reached_flag, rewards
        

    def reset(self, env_id):
        self.active_goals[env_id], self.active_goals_ids[env_id] = self._new_goal()
        
        #make all goals active, except first
        self.active_goals_flag[env_id]      = 1.0
        self.active_goals_flag[env_id][0]   = 0.0

        self.goals_ids_prev[env_id]     = 0
        self.goals_ids_now[env_id]      = 0

    def save(self, path):
        numpy.save(path + "gb_goals.npy", self.goals_buffer.detach().to("cpu").numpy())
        numpy.save(path + "gb_visited_count.npy", self.visited_count)
        numpy.save(path + "gb_adjacency_matrix.npy", self.adjacency_matrix)

    def load(self, path):
        self.goals_buffer       = torch.from_numpy(numpy.load(path + "gb_goals.npy"))
        self.visited_count      = numpy.load(path + "gb_visited_count.npy")
        self.adjacency_matrix   = numpy.load(path + "gb_adjacency_matrix.npy")

        #move goals pointer to last non-used position
        self.goals_ptr = 0
        for i in range(len(self.goals_buffer)):
            v  = self.goals_buffer[self.goals_ptr].sum()
            self.goals_ptr+= 1
            if v < 0.001:
                break


   



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

    def _add_goal(self, state_down):
        if self.goals_ptr < self.buffer_size:
            self.goals_buffer[self.goals_ptr] = state_down.clone()
 
            self.goals_ptr+= 1


    #sample new random goal
    #probs depends on connections and visited count
    def _new_goal(self):

        visited_reward      = 1.0/(1.0 + (self.visited_count**0.5))

        am_norm             = self.adjacency_matrix/(self.adjacency_matrix.sum(axis=1, keepdims=True) + 0.0000001)

        connections_reward  = (am_norm > 0.001).sum(axis=1)
 
        #goal with higher connections and fever visitings have higher prob to be goal
        probs = connections_reward*visited_reward 
        probs = probs[0:self.goals_ptr]
        probs = self._softmax(probs)

        idx   = numpy.random.choice(range(self.goals_ptr), size=1, p = probs)

        tmp   = self.goals_buffer[idx].reshape(self.goal_downsampled_shape).unsqueeze(0)

        goal  = self.upsample(tmp).squeeze(0)

        return goal, idx
    
    def _preprocess(self, states):
        s_down   = self.downsample(states)
        s0_down  = torch.flatten(s_down[:,0], start_dim=1)

        if states.shape[1] > 1:
            s1_down  = torch.flatten(s_down[:,1], start_dim=1)
        else:
            s1_down  = s0_down

        dif     = torch.abs(s0_down - s1_down).mean(dim=1)
 
        return s0_down, dif

    def _softmax(self, values):
        probs = numpy.exp(values - numpy.max(values))
        probs = probs/numpy.sum(probs)

        return probs
