import torch
import numpy


 
class CountsMemory:
    def __init__(self, size = 256, add_threshold = 0.001, device = "cpu"):
        self.size               = size        
        self.add_threshold      = add_threshold
        self.device             = device

        self.states             = None
        self.counts             = None

        self.layer_flatten    = torch.nn.Flatten()
        self.layer_flatten.to(self.device)


    def process(self, states_t, attention_t):
        states_f_t      = self.layer_flatten(states_t)
        attention_f_t   = self.layer_flatten(attention_t)

        batch_size      = states_f_t.shape[0]

        #create buffer if not created yet
        if self.states is None:
            self.states      = torch.zeros((self.size, states_f_t.shape[1])).to(self.device)
            self.counts      = torch.zeros((self.size, attention_f_t.shape[1])).to(self.device)
            self.total_count = 0

        #states_t distances from buffer
        distances = torch.cdist(states_f_t, self.states)/states_f_t.shape[1]
        
        #find closest
        state_indices   = torch.argmin(distances, dim=1)

        state_closest   = distances[range(batch_size), state_indices]

        #add new item if threashold reached
        for i in range(batch_size):
            if state_closest[i] > self.add_threshold:
                self.states[self.total_count] = states_f_t[i].clone()
                state_closest[i] = self.total_count
                self.total_count = (self.total_count + 1)%self.size
                if self.total_count == 1:
                    break

        #update counts
        position_indices = torch.argmax(attention_f_t, dim=1)
        self.counts[state_indices, position_indices]+= 1

        #regularisation
        self.counts*= 0.999

        #compute motivation
        counts          = self.counts[state_indices, position_indices]
        motivation_t    = 1.0/(counts**0.5 + 0.0001)

        return motivation_t

    