import torch
import numpy
import cv2

class GoalsBuffer:  

    def __init__(self, buffer_size, shape, add_threshold, device = "cpu"):

        self.shape                  = shape
        self.add_threshold          = add_threshold

        self.states                 = torch.zeros((buffer_size, ) + shape).to(device)
        self.score                  = torch.zeros((buffer_size, )).to(device)
        
        self.current_idx            = 1


    def update(self, states, score_sum, rewards):

        states_t        = torch.from_numpy(states).float().to(self.states.device)
        score_sum_t     = torch.from_numpy(score_sum).float().to(self.states.device)
        rewards_t       = torch.from_numpy(rewards).float().to(self.states.device)

        size         = self.shape[0]*self.shape[1]*self.shape[2]
        
        current      = self.states[0:self.current_idx]
        current_fltn = current.reshape((current.shape[0], size))

        states_fltn  = states_t.reshape((states_t.shape[0], size))

        distances    = torch.cdist(states_fltn, current_fltn)/size

        #closest distances
        min_dist, min_idx = torch.min(distances, dim=1)

        for i in range(min_dist.shape[0]):
            #add new rewarded state
            if min_dist[i] > 0.2*self.add_threshold and rewards_t > 0.0:
                self._add_new(states_t[i], score_sum_t[i])

            #add new interesting state
            elif min_dist[i] > self.add_threshold:
                self._add_new(states_t[i], score_sum_t[i])

        #update score sum
        for i in range(min_idx.shape[0]):
            idx = min_idx[i] 
            if self.score[idx] > score_sum_t[i]:
                self.score[idx] = score_sum_t[i]


    def render(self, save = False):
        goals  = self.states.to("cpu").numpy()

        height_y = self.shape[1]
        height_x = self.shape[1]

        size_y = 8
        size_x = 8

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

        if save:
            cv2.imwrite("goals.png", result*255)


    def _add_new(self, state, score_sum):
        if self.current_idx < self.states.shape[0]:
            self.states[self.current_idx] = state.clone()
            self.score[self.current_idx]  = score_sum

            self.current_idx+= 1
