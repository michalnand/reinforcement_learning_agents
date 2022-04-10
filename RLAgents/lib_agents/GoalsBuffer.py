import torch
import numpy
import cv2


class GoalsBuffer:  

    def __init__(self, buffer_size, envs_count, shape, add_threshold, goals_change_threshold, device = "cpu"):

        self.shape          = (1, shape[1], shape[2])
        self.add_threshold  = add_threshold
        self.goals_change_threshold = goals_change_threshold

        self.states         = torch.zeros((buffer_size, ) + self.shape, dtype=torch.float).to(device)
        self.steps          = torch.zeros((buffer_size, ), dtype=torch.float).to(device)

        self.reached        = torch.zeros((envs_count, buffer_size), dtype=bool).to(device)
        self.reached[:,0]   = True

        self.current_idx    = torch.ones((envs_count, ), dtype=int).to(device)

        self.count          = 1

    
    def update(self, states, dones, steps):
        batch_size      = states.shape[0]

        change          = ((states[:,0] - states[:,1])**2).mean(dim=(1,2))

        size            = self.shape[0]*self.shape[1]*self.shape[2]
        
        states_t        = states[:,0].to(self.states.device)
        states_fltn     = states_t.reshape((states_t.shape[0], size))

        states_current  = self.states[0:self.count]
        states_current  = states_current.reshape((states_current.shape[0], size))

        distances    = torch.cdist(states_fltn, states_current)/size

        #closest distances
        min_dist, min_idx = torch.min(distances, dim=1)

        #reward where goal reached : threshold and correct goal idx, reward more distant goals
        steps_reward = self.steps/(torch.max(self.steps) + 10e-10)
        steps_reward = steps_reward[min_idx]

        self.rewards = torch.logical_and( (min_dist <= self.add_threshold), (self.current_idx == min_idx) )
        self.rewards = steps_reward*self.rewards.float()
 
        #create new goal
        for i in range(batch_size):
            if self.rewards > 0:
                self.reached[i][min_idx] = True
                self.current_idx[i]      = self._get_goal(i)        
        
        #add new, if change threshold reached, and state not yet in buffer
        for i in range(batch_size):
            if change > self.goals_change_threshold and min_dist[i] > self.add_threshold and dones[i] == False:
                self._add_new(states[i], steps[i])
                break

        #update if less steps to reach
        for i in range(batch_size):
            idx = min_idx[i]
            if steps[i] < self.steps[idx]:
                self.steps[idx] = steps[i]
        
        return self.rewards, self.states[self.current_idx]

    def reset(self, env_id):
        self.current_idx[env_id]    = 1
        self.reached[env_id]        = False
        self.reached[env_id][0]     = True

    def save(self, path):
        torch.save(self.states, path + "goals_buffer_states.pt")
        torch.save(self.steps,  path + "goals_buffer_steps.pt")
        
    def load(self, path):
        self.states = torch.load(path + "goals_buffer_states.pt")
        self.steps  = torch.load(path + "goals_buffer_steps.pt")

    def render(self, env_id = 0):
        if self.rewards[env_id] > 0:
            goal_id = self.current_idx[env_id]
            print(">>> goal  ", self.current_idx[env_id], self.steps[goal_id], self.rewards[env_id])

        goals  = self.states.to("cpu").numpy()

        height_y = self.shape[1]
        height_x = self.shape[1]

        size_y = 8
        size_x = 8

        size_im = 512

        result = numpy.zeros((size_y*height_y, size_x*height_x))

        for y in range(size_y):
            for x in range(size_x):
                idx = y*size_x + x

                l = 0.5*(self.reached[env_id][idx] + 1.0)
                
                result[y*height_y:(y+1)*height_y, x*height_x:(x+1)*height_x] = l*goals[idx][0]

        result  = cv2.resize(result, (size_im, size_im))
        result  = numpy.clip(result, 0.0, 1.0)

        cv2.imshow("goals buffer", result)
        cv2.waitKey(1)

    def _add_new(self, state, steps):
        if self.count < self.states.shape[0]:
            self.states[self.count] = state[0].clone()
            self.steps[self.count]  = steps

            self.count+= 1 

    def _get_goal(self, env_id):
        result = 0
        for i in range(self.reached.shape[1]):
            if self.reached[env_id][i] == False:
                result = i
                break
        
        return result

    