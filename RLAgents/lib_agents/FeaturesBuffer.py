import torch
import numpy

class FeaturesBuffer: 

    def __init__(self, buffer_size, shape, envs_count, device):
        self.buffer         = torch.zeros((envs_count, buffer_size ) + shape).to(device)
        self.current_idx    = numpy.zeros(envs_count, dtype=int)

    def reset(self, env_id, initial_value = None):
        self.buffer[env_id]         = 0
        self.current_idx[env_id]    = 0

        if initial_value is not None:
            self.buffer[env_id] = initial_value.clone()

    def add(self, values_t):
        for i in range(values_t.shape[0]):
            idx = self.current_idx[i]

            self.buffer[i][idx] = values_t[i].clone()

        self.current_idx[i] = (self.current_idx[i] + 1)%self.buffer.shape[1]

    def distances(self, values_t):
        #difference
        dif = self.buffer - values_t.unsqueeze(1)

        #take last dims
        dims = tuple(range(2, len(self.buffer.shape)))

        #mean distance, shape = (envs_count, buffer_size)
        distances = (dif**2).mean(dim=dims)

        mean = distances.mean(dim=1)
        std  = distances.std(dim=1)
        max  = distances.max(dim=1)[0]
        min  = distances.std(dim=1)[0]

        return mean, std, max, min
