import numpy
import torch

class StatesBuffer: 
    def __init__(self, size, envs, state_shape):
        self.size           = size
        self.envs           = envs
        self.state_shape    = state_shape
        self.clear()
      
    def clear(self):
        self.buffer  = numpy.zeros((self.size, self.envs, ) + self.state_shape, dtype=numpy.float32)
        self.idx     = 0
 
    def add(self, state):
        self.buffer[self.idx] = state.copy()
        self.idx = (self.idx + 1)%self.buffer.shape[0]

    def sample_batch(self, batch_size, device):
        result_a_t = torch.zeros((batch_size, ) + self.state_shape)
        result_b_t = torch.zeros((batch_size, ) + self.state_shape)
        distance_t = torch.zeros((batch_size))

        for i in range(batch_size):
            idx_a         = numpy.random.randint(0, self.size)
            idx_b         = numpy.random.randint(0, self.size)
            e             = numpy.random.randint(0, self.envs)

            result_a_t[i] = torch.from_numpy(self.buffer[idx_a][e])
            result_b_t[i] = torch.from_numpy(self.buffer[idx_b][e])
            distance_t[i] = numpy.abs(idx_a - idx_b)

        result_a_t = result_a_t.to(device)
        result_b_t = result_b_t.to(device)
        distance_t = distance_t.to(device)

        return result_a_t, result_b_t, distance_t
   