import torch
import numpy 


class PolicyBufferIMDualPolicy:

    def __init__(self, buffer_size, state_shape, actions_a_size, actions_b_size, envs_count, device, uint8_storage = False):
        

        self.buffer_size    = buffer_size
        self.state_shape    = state_shape
        self.actions_a_size = actions_a_size
        self.actions_b_size = actions_b_size
        self.envs_count     = envs_count
        self.device         = device

        self.uint8_storage  = uint8_storage

        if self.uint8_storage:
            self.scale  = 255
        else:
            self.scale  = 1 
      
        self.clear()   
 
    def add(self, state, logits_a, logits_b, value_ext, value_int, action_a, action_b, reward_ext, reward_int, done):
        
        self.states[self.ptr]    = state.copy()*self.scale
        self.logits_a[self.ptr]  = logits_a.copy()
        self.logits_b[self.ptr]  = logits_b.copy()
        
        self.values_ext[self.ptr]= value_ext.copy()
        self.values_int[self.ptr]= value_int.copy()

        self.actions_a[self.ptr]   = action_a.copy()
        self.actions_b[self.ptr]   = action_b.copy()
        
        self.reward_ext[self.ptr]  = reward_ext.copy()
        self.reward_int[self.ptr]  = reward_int.copy()

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

        self.logits_a       = numpy.zeros((self.buffer_size, self.envs_count, self.actions_a_size), dtype=numpy.float32)
        self.logits_b       = numpy.zeros((self.buffer_size, self.envs_count, self.actions_b_size), dtype=numpy.float32)

        self.values_ext     = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)        
        self.values_int     = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)

        self.actions_a      = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=int)
        self.actions_b      = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=int)
        
        self.reward_ext     = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)
        self.reward_int     = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)

        self.dones          = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)

        self.ptr = 0  
 

    def compute_returns(self, gamma_ext = 0.99, gamma_int = 0.9, lam = 0.95):
        self.returns_ext, self.advantages_ext = self._gae(self.reward_ext, self.values_ext, self.dones, gamma_ext, lam)
        self.returns_int, self.advantages_int = self._gae(self.reward_int, self.values_int, self.dones, gamma_int, lam)
        
        #reshape buffer for faster batch sampling
        self.states           = self.states.reshape((self.buffer_size*self.envs_count, ) + self.state_shape)
        self.logits_a         = self.logits_a.reshape((self.buffer_size*self.envs_count, self.actions_a_size))
        self.logits_b         = self.logits_b.reshape((self.buffer_size*self.envs_count, self.actions_b_size))

        self.values_ext       = self.values_ext.reshape((self.buffer_size*self.envs_count, ))        
        self.values_int       = self.values_int.reshape((self.buffer_size*self.envs_count, ))

        self.actions_a        = self.actions_a.reshape((self.buffer_size*self.envs_count, ))
        self.actions_b        = self.actions_b.reshape((self.buffer_size*self.envs_count, ))
        
        self.reward_ext       = self.reward_ext.reshape((self.buffer_size*self.envs_count, ))
        self.reward_int       = self.reward_int.reshape((self.buffer_size*self.envs_count, ))

        self.dones            = self.dones.reshape((self.buffer_size*self.envs_count, ))

        self.returns_ext      = self.returns_ext.reshape((self.buffer_size*self.envs_count, ))
        self.advantages_ext   = self.advantages_ext.reshape((self.buffer_size*self.envs_count, ))

        self.returns_int      = self.returns_int.reshape((self.buffer_size*self.envs_count, ))
        self.advantages_int   = self.advantages_int.reshape((self.buffer_size*self.envs_count, ))


    def sample_batch(self, batch_size, device):
        indices         = numpy.random.randint(0, self.envs_count*self.buffer_size, size=batch_size*self.envs_count)
        indices_next    = numpy.clip(indices + 1, 0, self.envs_count*self.buffer_size-1)

        states          = torch.from_numpy(numpy.take(self.states, indices, axis=0)).to(device).float()/self.scale
        states_next     = torch.from_numpy(numpy.take(self.states, indices_next, axis=0)).to(device).float()/self.scale
        logits_a        = torch.from_numpy(numpy.take(self.logits_a, indices, axis=0)).to(device)
        logits_b        = torch.from_numpy(numpy.take(self.logits_b, indices, axis=0)).to(device)
        
        actions_a       = torch.from_numpy(numpy.take(self.actions_a, indices, axis=0)).to(device)
        actions_b       = torch.from_numpy(numpy.take(self.actions_b, indices, axis=0)).to(device)

        returns_ext     = torch.from_numpy(numpy.take(self.returns_ext, indices, axis=0)).to(device)
        returns_int     = torch.from_numpy(numpy.take(self.returns_int, indices, axis=0)).to(device)

        advantages_ext  = torch.from_numpy(numpy.take(self.advantages_ext, indices, axis=0)).to(device)
        advantages_int  = torch.from_numpy(numpy.take(self.advantages_int, indices, axis=0)).to(device)


        return states, states_next, logits_a, logits_b, actions_a, actions_b, returns_ext, returns_int, advantages_ext, advantages_int 
    
  
    def _gae(self, rewards, values, dones, gamma = 0.99, lam = 0.9):
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
