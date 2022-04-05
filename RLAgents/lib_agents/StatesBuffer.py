import torch
import numpy 


class StatesBuffer:

    def __init__(self, buffer_size, state_shape, actions_size, envs_count, device, uint8_storage = False):
        
        self.buffer_size    = buffer_size
        self.state_shape    = state_shape
        self.actions_size   = actions_size
        self.envs_count     = envs_count
        self.device         = device

        self.uint8_storage  = uint8_storage

        if self.uint8_storage:
            self.scale  = 255
        else:
            self.scale  = 1 
      
        self.clear()   
 
 
    def add(self, state, action, reward_ext, reward_int_a, reward_int_b, done):
        
        self.states[self.ptr]    = state.copy()*self.scale
        
        self.actions[self.ptr]   = action.copy()
        
        self.reward_ext[self.ptr]    = reward_ext.copy()
        self.reward_int_a[self.ptr]  = reward_int_a.copy()
        self.reward_int_b[self.ptr]  = reward_int_b.copy()

        self.dones[self.ptr]       = (1.0*done).copy()
        
        self.ptr = self.ptr + 1 


    def is_full(self):
        if self.ptr >= self.buffer_size:
            return True

        return False 
 

    def clear(self):
        if self.uint8_storage: 
            self.states     = numpy.zeros((self.buffer_size, self.envs_count, ) + self.state_shape, dtype=numpy.ubyte)
        else:
            self.states     = numpy.zeros((self.buffer_size, self.envs_count, ) + self.state_shape, dtype=numpy.float32)

        self.actions        = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=int)
        
        self.reward_ext     = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)
        self.reward_int_a   = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)
        self.reward_int_b   = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)

        self.dones          = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)

        self.ptr = 0  
 
