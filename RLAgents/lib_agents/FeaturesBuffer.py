import torch
import numpy


class FeaturesBuffer:  

    def __init__(self, buffer_size, envs_count, shape, device = "cpu"):
        self.buffer         = torch.zeros((buffer_size, envs_count ) + shape).to(device)
        self.current_idx    = 0

        self.initialising   = torch.zeros(envs_count, dtype=bool).to(device)

    def reset(self, env_id, initial_value = None):
        self.buffer[:,env_id] = 0
 
        if initial_value is not None:
            self.buffer[:,env_id] = initial_value.clone()
        else:
            self.initialising[env_id] = True

    def add(self, values_t):
        self.buffer[self.current_idx] = values_t.clone()

        self.current_idx = (self.current_idx + 1)%self.buffer.shape[0]

        indices = torch.nonzero(self.initialising).squeeze(1)

        self.buffer[:,indices] = values_t[indices].clone()
        self.initialising[:] = False

    '''
    def compute_entropy(self):
        var = torch.var(self.buffer, dim=0)
        var = var.mean(dim=1)

        return var
    '''

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
       
        return mean