import torch
import numpy


class FeaturesBuffer:  

    def __init__(self, buffer_size, envs_count, shape, device = "cpu"):
        self.buffer         = torch.zeros((buffer_size, envs_count ) + shape).to(device)
        self.current_idx    = 0

    def reset(self, env_id, initial_value = None):
        self.buffer[env_id] = 0

        if initial_value is not None:
            self.buffer[:,env_id] = initial_value.clone()

    def add(self, values_t):
        self.buffer[self.current_idx] = values_t.clone()

        self.current_idx = (self.current_idx + 1)%self.buffer.shape[1]


    def compute_entropy(self):
        std = torch.std(self.buffer, dim=0)
        std = std.mean(dim=1)
        
        return std.detach().to("cpu").numpy()
 
    def compute(self, values_t, top_n = 32):
        #difference
        dif = self.buffer - values_t

        #take last dims
        dims = tuple(range(2, len(self.buffer.shape)))

        #mean distance, shape = (buffer_size, envs_count)
        distances = (dif**2).mean(dim=dims)

        if top_n is not None:
            distances = torch.sort(distances, dim=0)[0]
            distances = distances[0:top_n,:]
 
        mean = distances.mean(dim=0)
        std  = distances.std(dim=0)

        mean = mean.detach().to("cpu").numpy()
        std  = std.detach().to("cpu").numpy()
       
        return mean, std

'''
class FeaturesBuffer:  

    def __init__(self, buffer_size, envs_count, shape, device = "cpu"):
        self.buffer         = torch.zeros((envs_count, buffer_size ) + shape).to(device)
        self.current_idx    = 0

    def reset(self, env_id, initial_value = None):
        self.buffer[env_id] = 0

        if initial_value is not None:
            self.buffer[env_id] = initial_value.clone()

    def add(self, values_t):
        for i in range(values_t.shape[0]):
            self.buffer[i][self.current_idx] = values_t[i].clone()

        self.current_idx = (self.current_idx + 1)%self.buffer.shape[1]
 
    def compute(self, values_t):
        #difference
        dif = self.buffer - values_t.unsqueeze(1)

        #take last dims
        dims = tuple(range(2, len(self.buffer.shape)))

        #mean distance, shape = (envs_count, buffer_size)
        distances = (dif**2).mean(dim=dims)
 
        mean = distances.mean(dim=1)
        std  = distances.std(dim=1)
        max  = distances.max(dim=1)[0]
        min  = distances.min(dim=1)[0]
        
        return mean, std, max, min

'''