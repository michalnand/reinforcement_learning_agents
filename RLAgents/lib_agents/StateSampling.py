import torch

class StateSampling:
    def __init__(self, initial_value, filter_order):
        self.filter_order = filter_order

        self.buffer = torch.zeros((self.filter_order, ) + initial_value.shape).to(initial_value.device)
        
        for i in range(self.filter_order):
            self.buffer[i] = initial_value.clone()

        self.idx = 0 

    def reset(self, env_idx, initial_value):
        for i in range(self.filter_order):
            self.buffer[i][env_idx] = initial_value.clone()

    def add(self, value):  
        self.buffer[self.idx] = value.clone()
        self.idx = (self.idx + 1)%self.filter_order  
        
        result = self.buffer.mean(dim=0) 
        
        return result
        
