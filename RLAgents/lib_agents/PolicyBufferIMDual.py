import torch
import numpy
 
class PolicyBufferIMDual:

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
 
    def add(self, state, logits, value_ext, value_int_a, value_int_b, action, reward, internal_a, internal_b, done):
        
        self.states_b[self.ptr]         = state.copy()*self.scale
        self.logits_b[self.ptr]         = logits.copy()
        
        self.values_ext_b[self.ptr]     = value_ext.copy()
        self.values_int_a_b[self.ptr]   = value_int_a.copy()
        self.values_int_b_b[self.ptr]   = value_int_b.copy()

        self.actions_b[self.ptr]        = action.copy()
        
        self.rewards_b[self.ptr]       = reward.copy()
        self.internal_a_b[self.ptr]    = internal_a.copy()
        self.internal_b_b[self.ptr]    = internal_b.copy()

        self.dones_b[self.ptr]     = (1.0*done).copy()
        
        self.ptr = self.ptr + 1 

    def is_full(self):
        if self.ptr >= self.buffer_size:
            return True

        return False 
 
    def clear(self):
        if self.uint8_storage: 
            self.states_b           = numpy.zeros((self.buffer_size, self.envs_count, ) + self.state_shape, dtype=numpy.ubyte)
        else:
            self.states_b           = numpy.zeros((self.buffer_size, self.envs_count, ) + self.state_shape, dtype=numpy.float32)

        self.logits_b           = numpy.zeros((self.buffer_size, self.envs_count, self.actions_size), dtype=numpy.float32)

        self.values_ext_b       = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)        
        self.values_int_a_b     = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)
        self.values_int_b_b     = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)

        self.actions_b          = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=int)
        
        self.rewards_b          = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)
        self.internal_a_b       = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)
        self.internal_b_b       = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)

        self.dones_b            = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)

        self.ptr = 0 


    def compute_returns(self, gamma_ext = 0.99, gamma_int = 0.9, lam = 0.95):
        
        self.returns_ext_b,     self.advantages_ext_b   = self._gae_fast(self.rewards_b, self.values_ext_b, self.dones_b, gamma_ext, lam)
        self.returns_int_a_b,   self.advantages_int_a_b = self._gae_fast(self.internal_a_b, self.values_int_a_b, self.dones_b, gamma_int, lam)
        self.returns_int_b_b,   self.advantages_int_b_b = self._gae_fast(self.internal_b_b, self.values_int_b_b, self.dones_b, gamma_int, lam)


        #reshape buffer for faster batch sampling
        self.states_b           = self.states_b.reshape((self.buffer_size*self.envs_count, ) + self.state_shape)
        self.logits_b           = self.logits_b.reshape((self.buffer_size*self.envs_count, self.actions_size))

        self.values_ext_b       = self.values_ext_b.reshape((self.buffer_size*self.envs_count, ))        
        self.values_int_a_b     = self.values_int_a_b.reshape((self.buffer_size*self.envs_count, ))
        self.values_int_b_b     = self.values_int_b_b.reshape((self.buffer_size*self.envs_count, ))

        self.actions_b          = self.actions_b.reshape((self.buffer_size*self.envs_count, ))
        
        self.rewards_b          = self.rewards_b.reshape((self.buffer_size*self.envs_count, ))
        self.internal_a_b       = self.internal_a_b.reshape((self.buffer_size*self.envs_count, ))
        self.internal_b_b       = self.internal_b_b.reshape((self.buffer_size*self.envs_count, ))

        self.dones_b            = self.dones_b.reshape((self.buffer_size*self.envs_count, ))

        self.returns_ext_b      = self.returns_ext_b.reshape((self.buffer_size*self.envs_count, ))
        self.advantages_ext_b   = self.advantages_ext_b.reshape((self.buffer_size*self.envs_count, ))

        self.returns_int_a_b    = self.returns_int_a_b.reshape((self.buffer_size*self.envs_count, ))
        self.advantages_int_a_b = self.advantages_int_a_b.reshape((self.buffer_size*self.envs_count, ))

        self.returns_int_b_b    = self.returns_int_b_b.reshape((self.buffer_size*self.envs_count, ))
        self.advantages_int_b_b = self.advantages_int_b_b.reshape((self.buffer_size*self.envs_count, ))



    def sample_batch(self, batch_size, device):
        indices             = numpy.random.randint(0, self.buffer_size*self.envs_count, size=batch_size*self.envs_count)
        
        indices_next    = numpy.clip(indices + 1, 0, self.envs_count*self.buffer_size - 1)

        states          = torch.from_numpy(numpy.take(self.states_b, indices, axis=0)).to(device).float()/self.scale
        states_next     = torch.from_numpy(numpy.take(self.states_b, indices_next, axis=0)).to(device).float()/self.scale
        
        logits              = torch.from_numpy(numpy.take(self.logits_b, indices, axis=0)).to(device)
        
        actions             = torch.from_numpy(numpy.take(self.actions_b, indices, axis=0)).to(device)
        
        returns_ext         = torch.from_numpy(numpy.take(self.returns_ext_b, indices, axis=0)).to(device)
        returns_int_a       = torch.from_numpy(numpy.take(self.returns_int_a_b, indices, axis=0)).to(device)
        returns_int_b       = torch.from_numpy(numpy.take(self.returns_int_b_b, indices, axis=0)).to(device)

        advantages_ext      = torch.from_numpy(numpy.take(self.advantages_ext_b, indices, axis=0)).to(device)
        advantages_int_a    = torch.from_numpy(numpy.take(self.advantages_int_a_b, indices, axis=0)).to(device)
        advantages_int_b    = torch.from_numpy(numpy.take(self.advantages_int_b_b, indices, axis=0)).to(device)
    
        return states, states_next, logits, actions, returns_ext, returns_int_a, returns_int_b, advantages_ext, advantages_int_a, advantages_int_b 
 
    def _gae_fast(self, rewards, values, dones, gamma = 0.99, lam = 0.9):
        buffer_size = rewards.shape[0]
        envs_count  = rewards.shape[1]
        
        returns     = numpy.zeros((buffer_size, envs_count), dtype=numpy.float32)
        advantages  = numpy.zeros((buffer_size, envs_count), dtype=numpy.float32)

        last_gae    = numpy.zeros((envs_count), dtype=numpy.float32)
        
        for n in reversed(range(buffer_size-1)):
            delta           = rewards[n] + gamma*values[n+1]*(1.0 - dones[n]) - values[n]
            last_gae        = delta + gamma*lam*last_gae*(1.0 - dones[n])
            
            returns[n]      = last_gae + values[n]
            advantages[n]   = last_gae

        return returns, advantages