import torch
import numpy
import cv2

class GoalsBuffer:  

    def __init__(self, buffer_size, envs_count, shape, add_threshold, device = "cpu"):

        self.shape                  = shape
        self.add_threshold          = add_threshold

        self.states                 = torch.zeros((buffer_size, ) + shape).to(device)
        self.reached_goals          = torch.zeros((envs_count, buffer_size)).to(device)
        self.reached_goals[:, 0]    = 1.0

        self.goal_idx               = torch.zeros((envs_count), dtype=int).to(device)

        self.current_idx            = 1

        downsample                  = 2
        self.downsample             = torch.nn.AvgPool2d(downsample, downsample)

        self.size                   = (shape[0]*shape[1]*shape[2])//(downsample**2)

    def update(self, states):
        states_t        = torch.from_numpy(states).float().to(self.states.device)
    
        current      = self.states[0:self.current_idx]
        current      = self.downsample(current)
        current_fltn = current.reshape((current.shape[0], self.size))

        states_down_t= self.downsample(states_t) 
        states_fltn  = states_down_t.reshape((states_down_t.shape[0], self.size))

        distances    = torch.cdist(states_fltn, current_fltn)/self.size

        #closest distances
        min_dist, min_idx = torch.min(distances, dim=1)

        for i in range(min_dist.shape[0]):
            #add new interesting state
            if min_dist[i] > self.add_threshold:
                self._add_new(states_t[i])
                break

       
        rewards = numpy.zeros(min_dist.shape[0])
        #find reached goals
        for i in range(min_dist.shape[0]):
            if min_dist[i] < self.add_threshold:
                goal_idx = min_idx[i]

                if self.reached_goals[i][goal_idx] < 0.5:
                    rewards[i] = 1.0
                
                self.reached_goals[i][goal_idx] = 1.0

                self.goal_idx[i] = self._get_goal(i)

                
        #returing goals and states
        goals = self.states[self.goal_idx]


        size_y = 8
        size_x = 8

        reached_goals = self.reached_goals.unsqueeze(2).unsqueeze(3)
        reached_goals = reached_goals.reshape((reached_goals.shape[0], 1, size_y, size_x))
        reached_goals = torch.repeat_interleave(reached_goals, self.shape[1]//size_y, dim=2)
        reached_goals = torch.repeat_interleave(reached_goals, self.shape[2]//size_x, dim=3)
        
        return goals, reached_goals, rewards

        


    def reset(self, env_id):
        self.reached_goals[env_id]      = 0.0
        self.reached_goals[env_id, 0]   = 1.0
        self.goal_idx[env_id]           = self._get_goal(env_id)


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


    def _add_new(self, state):
        if self.current_idx < self.states.shape[0]:
            self.states[self.current_idx] = state.clone()

            self.current_idx+= 1

    def _get_goal(self, env_id):
        result = 0
        for i in range(self.current_idx):
            if self.reached_goals[env_id][i] < 0.5:
                result = i
                break
        
        return result