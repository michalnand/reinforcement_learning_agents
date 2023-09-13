import numpy
import torch

class ExperienceBuffer():
    def __init__(self, size, state_shape, actions_count, uint8_storage = False):

        self.size           = size       
        self.current_idx    = 0 
        self.initialized    = False

        self.state_shape        = state_shape
        self.actions_count      = actions_count
        self.uint8_storage      = uint8_storage

        if self.uint8_storage:
            self.scale  = 255
        else:
            self.scale  = 1

    def _initialize(self):
        if self.initialized == False:

            if self.uint8_storage: 
                self.state_b    = numpy.zeros((self.size, ) + self.state_shape, dtype=numpy.ubyte)
            else:
                self.state_b    = numpy.zeros((self.size, ) + self.state_shape, dtype=numpy.float32)

            self.action_b       = numpy.zeros((self.size, ), dtype=int)
            self.reward_b       = numpy.zeros((self.size, ), dtype=numpy.float32)
            self.done_b         = numpy.zeros((self.size, ), dtype=numpy.float32)
            self.reward_int_b   = numpy.zeros((self.size, ), dtype=numpy.float32)

            self.initialized    = True

    def add(self, state, action, reward, done, reward_int = 0.0): 
        self._initialize()

        if done != 0: 
            done_ = 1.0
        else:
            done_ = 0.0

        self.state_b[self.current_idx]      = state.copy()*self.scale
        self.action_b[self.current_idx]     = int(action)
        self.reward_b[self.current_idx]     = reward
        self.done_b[self.current_idx]       = done_
        self.reward_int_b[self.current_idx] = reward_int

        self.current_idx = (self.current_idx + 1)%self.size

    def sample(self, batch_size, device = "cpu"):
        indices         = numpy.random.randint(0, self.size, size=batch_size)
        indices_next    = (indices + 1)%self.size 

        state_t         = torch.from_numpy(numpy.take(self.state_b,     indices, axis=0)).to(device).float()/self.scale
        state_next_t    = torch.from_numpy(numpy.take(self.state_b,     indices_next, axis=0)).to(device).float()/self.scale
        action_t        = torch.from_numpy(numpy.take(self.action_b,    indices, axis=0)).to(device)
        reward_t        = torch.from_numpy(numpy.take(self.reward_b,    indices, axis=0)).to(device)
        done_t          = torch.from_numpy(numpy.take(self.done_b,      indices, axis=0)).to(device)
        reward_int_t    = torch.from_numpy(numpy.take(self.reward_int_b,      indices, axis=0)).to(device)

        return state_t, state_next_t, action_t, reward_t, done_t, reward_int_t

