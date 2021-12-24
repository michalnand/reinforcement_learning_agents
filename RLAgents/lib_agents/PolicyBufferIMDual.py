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
 
    def add(self, state, logits, value_ext, value_int_a, value_int_b, action, reward_ext, reward_int_a, reward_int_b, done):
        
        self.states[self.ptr]    = state.copy()*self.scale
        self.logits[self.ptr]    = logits.copy()
        
        self.values_ext[self.ptr]   = value_ext.copy()
        self.values_int_a[self.ptr] = value_int_a.copy()
        self.values_int_b[self.ptr] = value_int_b.copy()

        self.actions[self.ptr]   = action.copy()
        
        self.reward_ext[self.ptr]   = reward_ext.copy()
        self.reward_int_a[self.ptr] = reward_int_a.copy()
        self.reward_int_b[self.ptr] = reward_int_b.copy()

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

        self.logits         = numpy.zeros((self.buffer_size, self.envs_count, self.actions_size), dtype=numpy.float32)

        self.values_ext     = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)        
        self.values_int_a   = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)
        self.values_int_b   = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)

        self.actions        = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=int)
        
        self.reward_ext     = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)
        self.reward_int_a   = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)
        self.reward_int_b   = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)

        self.dones          = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)

        self.ptr = 0  
 

    def compute_returns(self, gamma_ext, gamma_int_a, gamma_int_b, lam = 0.95):
        self.returns_ext, self.advantages_ext       = self._gae(self.reward_ext, self.values_ext, self.dones, gamma_ext, lam)
        self.returns_int_a, self.advantages_int_a   = self._gae(self.reward_int_a, self.values_int_a, self.dones, gamma_int_a, lam)
        self.returns_int_b, self.advantages_int_b   = self._gae(self.reward_int_b, self.values_int_b, self.dones, gamma_int_b, lam)

        #reshape buffer for faster batch sampling
        self.states           = self.states.reshape((self.buffer_size*self.envs_count, ) + self.state_shape)
        self.logits           = self.logits.reshape((self.buffer_size*self.envs_count, self.actions_size))

        self.values_ext       = self.values_ext.reshape((self.buffer_size*self.envs_count, ))        
        self.values_int_a     = self.values_int_a.reshape((self.buffer_size*self.envs_count, ))
        self.values_int_b     = self.values_int_b.reshape((self.buffer_size*self.envs_count, ))

        self.actions          = self.actions.reshape((self.buffer_size*self.envs_count, ))
        
        self.reward_ext       = self.reward_ext.reshape((self.buffer_size*self.envs_count, ))
        self.reward_int_a     = self.reward_int_a.reshape((self.buffer_size*self.envs_count, ))
        self.reward_int_b     = self.reward_int_b.reshape((self.buffer_size*self.envs_count, ))

        self.dones            = self.dones.reshape((self.buffer_size*self.envs_count, ))

        self.returns_ext      = self.returns_ext.reshape((self.buffer_size*self.envs_count, ))
        self.advantages_ext   = self.advantages_ext.reshape((self.buffer_size*self.envs_count, ))

        self.returns_int_a    = self.returns_int_a.reshape((self.buffer_size*self.envs_count, ))
        self.advantages_int_a = self.advantages_int_a.reshape((self.buffer_size*self.envs_count, ))

        self.returns_int_b    = self.returns_int_b.reshape((self.buffer_size*self.envs_count, ))
        self.advantages_int_b = self.advantages_int_b.reshape((self.buffer_size*self.envs_count, ))


    def sample_batch(self, batch_size, device):
        indices         = numpy.random.randint(0, self.envs_count*self.buffer_size, size=batch_size*self.envs_count)
        indices_next    = numpy.clip(indices + 1, 0, self.envs_count*self.buffer_size-1)

        states          = torch.from_numpy(numpy.take(self.states, indices, axis=0)).to(device).float()/self.scale
        states_next     = torch.from_numpy(numpy.take(self.states, indices_next, axis=0)).to(device).float()/self.scale
        logits          = torch.from_numpy(numpy.take(self.logits, indices, axis=0)).to(device)
        
        actions         = torch.from_numpy(numpy.take(self.actions, indices, axis=0)).to(device)
        
        returns_ext     = torch.from_numpy(numpy.take(self.returns_ext, indices, axis=0)).to(device)
        returns_int_a   = torch.from_numpy(numpy.take(self.returns_int_a, indices, axis=0)).to(device)
        returns_int_b   = torch.from_numpy(numpy.take(self.returns_int_b, indices, axis=0)).to(device)

        advantages_ext  = torch.from_numpy(numpy.take(self.advantages_ext, indices, axis=0)).to(device)
        advantages_int_a= torch.from_numpy(numpy.take(self.advantages_int_a, indices, axis=0)).to(device)
        advantages_int_b= torch.from_numpy(numpy.take(self.advantages_int_b, indices, axis=0)).to(device)


        return states, states_next, logits, actions, returns_ext, returns_int_a, returns_int_b, advantages_ext, advantages_int_a, advantages_int_b 
    

    def _gae(self, rewards, values, dones, gamma, lam):
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
