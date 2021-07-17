import torch
import numpy

class GoalsMemory:
    def __init__(self, size, downsample = -1, add_prob = 0.01, alpha = 0.01, epsilon = 0.0001, device = "cpu"):
        self.size       = size
        self.downsample = downsample
        
        self.add_prob   = add_prob
        self.alpha      = alpha
        self.epsilon    = numpy.log(epsilon)

        self.device     = device

        self.buffer     = None
        self.buffer_idx = 0

        if downsample > 1:
            self.layer_downsample = torch.nn.AvgPool2d((self.downsample, self.downsample), (self.downsample, self.downsample))
            self.layer_downsample.to(self.device)

        self.layer_flatten    = torch.nn.Flatten()
        self.layer_flatten.to(self.device)


    def process(self, states_t, steps_t):
        tmp_t = self._preprocess(states_t)

        #create buffer if not created yet
        if self.buffer is None:
            self.buffer = torch.zeros((self.size, tmp_t.shape[1])).float().to(self.device)
            self.steps  = torch.ones((self.size, )).to(self.device)

        #states_t distances from buffer
        distances = torch.cdist(tmp_t, self.buffer)
        
        #find closest
        indices   = torch.argmin(distances, dim=1)
        
        #smooth update stored distances
        self.steps[indices] = (1.0 - self.alpha)*self.steps[indices] + self.alpha*(steps_t.float() + 1)

        #compute motivation
        motivation_t = torch.exp(steps_t*self.epsilon/self.steps[indices])

        #add new item on new place with add_prob probability
        for i in range(tmp_t.shape[0]):
            if numpy.random.rand() < self.add_prob:
                self.buffer[self.buffer_idx] = tmp_t[i].clone()
                self.steps[self.buffer_idx]  = steps_t.clone()

                self.buffer_idx = (self.buffer_idx + 1)%self.size

        return motivation_t

    
    #downsample and flatten
    def _preprocess(self, x):
        if self.layer_downsample is not None:
            y = self.layer_downsample(x)
        
        y = self.layer_flatten(y)
        return y 
