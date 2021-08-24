from numpy.core.fromnumeric import var
from numpy.core.numeric import indices
import torch
import numpy


class GoalsBuffer:
    def __init__(self, size, add_threshold, downsample, goals_ext_reward_ratio, state_shape, parallel_envs, device = "cpu"):
        self.size               = size
        self.downsample         = downsample
        self.goals_ext_reward_ratio  = goals_ext_reward_ratio
        self.add_threshold      = add_threshold
        self.state_shape        = state_shape
        self.goals_shape        = (1, state_shape[1], state_shape[2]) 
        self.device             = device
        self.alpha              = 0.1

        self.layer_downsample = torch.nn.AvgPool2d((self.downsample, self.downsample), (self.downsample, self.downsample))
        self.layer_downsample.to(self.device)

        self.layer_upsample = torch.nn.Upsample(scale_factor=self.downsample, mode="nearest")
        self.layer_upsample.to(self.device)

        self.layer_flatten = torch.nn.Flatten()
        self.layer_flatten.to(self.device)

        self.states_downsampled_shape = (1, self.state_shape[1]//self.downsample, self.state_shape[2]//self.downsample)
        self.goals_shape = (1, ) + self.state_shape[1:]
                
        self.candidate_goals_b      = torch.zeros((self.size, numpy.prod(self.states_downsampled_shape)), device=self.device)
        self.desired_goals_b        = torch.zeros((parallel_envs, ) + self.goals_shape, device=self.device)

        self.steps_b        = numpy.zeros(self.size, dtype=numpy.int)
        self.reward_ext_b   = numpy.zeros(self.size, dtype=numpy.float32)
        self.reward_int_b   = numpy.zeros(self.size, dtype=numpy.float32)

        self.total_goals  = 0


    def get(self, states_t, steps_np):        
        self.states_downsampled = self._downsmaple(states_t[:,0].unsqueeze(1))
        goals_downsampled       = self._downsmaple(self.desired_goals_b)
        self.current_goal       = self._upsample(self.states_downsampled)

        #add first goal if buffer empty
        if self.total_goals == 0:
            self.candidate_goals_b[0]   = self.states_downsampled[0].clone()
            self.steps_b[0]             = steps_np[0]
            self.reward_ext_b[0]        = 0.0
            self.reward_int_b[0]        = 0.0

            self.total_goals = 1

        #states_t distances from buffer
        distances = torch.cdist(self.states_downsampled, self.candidate_goals_b)
    
        #find closest
        self.indices   = torch.argmin(distances, dim=1)
        self.closest   = distances[range(self.states_downsampled.shape[0]), self.indices]

        
        #reward for reached goal
        goals_distances         = (((goals_downsampled - self.states_downsampled)**2.0).sum(dim=1))**0.5 
        reached_goals           = (goals_distances <= self.add_threshold).detach().to("cpu").numpy()
        reward_reached_goals    = 1.0*reached_goals

        #clear goal if reached
        for i in range(len(reward_reached_goals)):
            if reward_reached_goals[i] > 0:
                self.desired_goals_b[i] = torch.zeros(self.goals_shape, device=self.device)

        #reward for less steps to goal
        faster             = (steps_np).astype(int) < (self.steps_b[self.indices]).astype(int)
        reward_goal_steps  = numpy.tanh(0.1*faster)*reached_goals

        return self.current_goal, self.desired_goals_b, reward_reached_goals, reward_goal_steps

    def add(self, reward_ext, reward_int, steps):
        #add new item if threashold reached
        for i in range(self.states_downsampled.shape[0]):
            if self.closest[i] > self.add_threshold and self.total_goals < self.size: 
                self.candidate_goals_b[self.total_goals]       = self.states_downsampled[i].clone()
                self.steps_b[self.total_goals]        = steps[i]
                self.reward_ext_b[self.total_goals]   = reward_ext[i]
                self.reward_int_b[self.total_goals]   = reward_int[i]

                self.total_goals = (self.total_goals + 1)%self.size

        indices = self.indices.detach().to("cpu").numpy()

               
        #add higher reward
        self.reward_ext_b[indices] = numpy.maximum(self.reward_ext_b[indices], reward_ext)
    
        #smooth update reward int
        self.reward_int_b[indices] = (1.0 - self.alpha)*self.reward_int_b[indices] + self.alpha*reward_int

        #smooth update steps
        self.steps_b[indices] = (1.0 - self.alpha)*self.steps_b[indices] + self.alpha*steps
 
   
    def new_goal(self, env_idx):
        #compute target weights
        w   = self.goals_ext_reward_ratio*self.reward_ext_b + (1.0 - self.goals_ext_reward_ratio)*self.reward_int_b
        
        #select only from stored state
        w   = w[0:self.total_goals]

        #convert to probs, softmax
        probs   = numpy.exp(w - w.max())
        probs   = probs/probs.sum() 

        #get random idx, with prob given in w
        idx = numpy.random.choice(range(len(w)), 1, p=probs)[0]

        goal   = self.candidate_goals_b[idx].unsqueeze(0)
        
        goal   = self._upsample(goal)[0]

        self.desired_goals_b[env_idx] = goal.clone()

        

    #downsample and flatten
    def _downsmaple(self, states_t):
        y = self.layer_downsample(states_t)
        y = self.layer_flatten(y)
        return y 

    def _upsample(self, x):
        h = self.state_shape[1]//self.downsample
        w = self.state_shape[2]//self.downsample
        
        x = x.reshape((x.shape[0], 1, h, w))

        y = self.layer_upsample(x)

        return y
      


