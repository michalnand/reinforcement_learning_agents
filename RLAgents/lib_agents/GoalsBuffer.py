import torch
import numpy
import cv2

class GoalsBuffer:  

    def __init__(self, buffer_size, envs_count, shape, add_threshold, device = "cpu"):

        self.shape                  = shape
        self.add_threshold          = add_threshold

        self.states                 = torch.zeros((buffer_size, ) + shape, dtype=torch.float).to(device)
        self.active                 = torch.zeros((envs_count, buffer_size), dtype=bool).to(device)

        self.current_idx            = 0

        downsample                  = 2
        self.smooth                 = torch.nn.AvgPool2d(5, stride=1, padding=5//2)
        self.downsample             = torch.nn.AvgPool2d(downsample, downsample)
        self.size                   = (shape[0]*shape[1]*shape[2])//(downsample**2)


    def update(self, states):
        states_t     = torch.from_numpy(states).float().to(self.states.device)
        
        states_down_t= self.smooth(states_t) 
        states_down_t= self.downsample(states_t) 
        states_fltn  = states_down_t.reshape((states_down_t.shape[0], self.size))

        if self.current_idx == 0:
            self._add_new(states_t[0])
       
        current      = self.states[0:self.current_idx]
        current      = self.smooth(current) 
        current      = self.downsample(current)
        current_fltn = current.reshape((current.shape[0], self.size))


        distances    = torch.cdist(states_fltn, current_fltn)/self.size

        #closest distances
        min_dist, min_idx = torch.min(distances, dim=1)

        for i in range(min_dist.shape[0]):
            #add new interesting state
            if min_dist[i] > self.add_threshold:
                self._add_new(states_t[i])
                break

        result_rewards = numpy.zeros(min_dist.shape[0])
        for i in range(min_dist.shape[0]):
            if min_dist[i] <= self.add_threshold and self.active[i][min_idx[i]] == True:
                self.active[i][min_idx[i]] = False
                result_rewards[i]           = 1.0

        return result_rewards


    def reset(self, env_id):
        self.active[env_id]     = True
        self.active[env_id][0]  = False


    def render(self):
        goals  = self.states.to("cpu").numpy()

        height_y = self.shape[1]
        height_x = self.shape[1]

        size_y = 16
        size_x = 16

        size_im = 1024

        result = numpy.zeros((size_y*height_y, size_x*height_x))

        for y in range(size_y):
            for x in range(size_x):
                idx = y*size_x + x
                result[y*height_y:(y+1)*height_y, x*height_x:(x+1)*height_x] = goals[idx][0]

        result  = cv2.resize(result, (size_im, size_im))
        result  = numpy.clip(result, 0.0, 1.0)

        cv2.imshow("goals buffer", result)
        cv2.waitKey(1)


    def save(self, path):
        torch.save(self.states, path + "goals_buffer_states.pt")
        
    def load(self, path):
        self.states = torch.load(path + "goals_buffer_states.pt")

    def _add_new(self, state):
        if self.current_idx < self.states.shape[0]:
            self.states[self.current_idx]   = state.clone()
            self.current_idx+= 1
