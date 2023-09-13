import numpy
import torch

class ExperienceBufferContinuous():
    def __init__(self, size, state_shape, actions_count):

        self.size           = size       
        self.current_idx    = 0 
        self.initialized    = False

        self.state_shape        = state_shape
        self.actions_count      = actions_count

    def _initialize(self):
        if self.initialized == False:
            self.state_b        = numpy.zeros((self.size, ) + self.state_shape, dtype=numpy.float32)
            self.action_b       = numpy.zeros((self.size, self.actions_count), dtype=numpy.float32)
            self.reward_b       = numpy.zeros((self.size, ), dtype=numpy.float32)
            self.done_b         = numpy.zeros((self.size, ), dtype=numpy.float32)
            self.im_a_b         = numpy.zeros((self.size, ), dtype=numpy.float32)
            self.im_b_b         = numpy.zeros((self.size, ), dtype=numpy.float32)

            self.initialized    = True

    def add(self, state, action, reward, done, im_a = 0.0, im_b = 0.0): 
        self._initialize()

        if done != 0: 
            done_ = 1.0
        else:
            done_ = 0.0

        self.state_b[self.current_idx]          = state.copy()
        self.action_b[self.current_idx]         = action.copy()
        self.reward_b[self.current_idx]         = reward
        self.done_b[self.current_idx]           = done_
        self.im_a_b[self.current_idx]           = im_a
        self.im_b_b[self.current_idx]           = im_b

        self.current_idx = (self.current_idx + 1)%self.size

    def sample(self, batch_size, device = "cpu"):
        indices         = numpy.random.randint(0, self.size, size=batch_size)
        indices_next    = (indices + 1)%self.size

        state_t         = torch.from_numpy(numpy.take(self.state_b,     indices, axis=0)).to(device)
        state_next_t    = torch.from_numpy(numpy.take(self.state_b,     indices_next, axis=0)).to(device)
        action_t        = torch.from_numpy(numpy.take(self.action_b,    indices, axis=0)).to(device)
        reward_t        = torch.from_numpy(numpy.take(self.reward_b,    indices, axis=0)).to(device)
        im_a_t          = torch.from_numpy(numpy.take(self.im_a_b,      indices, axis=0)).to(device)
        im_b_t          = torch.from_numpy(numpy.take(self.im_b_b,      indices, axis=0)).to(device)

        done_t          = torch.from_numpy(numpy.take(self.done_b,      indices, axis=0)).to(device)

        return state_t, state_next_t, action_t, reward_t, im_a_t, im_b_t, done_t

