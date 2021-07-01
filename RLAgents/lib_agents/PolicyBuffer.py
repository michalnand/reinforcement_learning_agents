import torch
import numpy

class PolicyBuffer:

    def __init__(self, buffer_size, state_shape, actions_size, envs_count, device):
        
        self.buffer_size    = buffer_size
        self.state_shape    = state_shape
        self.actions_size   = actions_size
        self.envs_count     = envs_count
        self.device         = device

        self.clear()

    def add(self, state, logits, value, action, reward, done):

        self.states_b[self.ptr]    = state.copy()
        self.logits_b[self.ptr]    = logits.copy()
        self.values_b[self.ptr]    = value.copy()
        self.actions_b[self.ptr]   = action.copy()
        self.rewards_b[self.ptr]   = reward.copy()
        self.dones_b[self.ptr]     = (1.0*done).copy()
        
        self.ptr = self.ptr + 1 


    def is_full(self):
        if self.ptr >= self.buffer_size:
            return True

        return False 
 
    def clear(self):
        self.states_b           = numpy.zeros((self.buffer_size, self.envs_count, ) + self.state_shape, dtype=numpy.float32)
        self.logits_b           = numpy.zeros((self.buffer_size, self.envs_count, self.actions_size), dtype=numpy.float32)
        
        self.values_b       = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)

        self.actions_b          = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=int)
        self.rewards_b          = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)
        self.dones_b            = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)
     
        self.ptr = 0 


    def compute_returns(self, gamma = 0.99, lam = 0.95):
        self.returns_b, self.advantages_b   = self._gae_fast(self.rewards_b, self.values_b, self.dones_b, gamma, lam)

        #reshape buffer for faster batch sampling
        self.states_b           = self.states_b.reshape((self.buffer_size*self.envs_count, ) + self.state_shape)
        self.logits_b           = self.logits_b.reshape((self.buffer_size*self.envs_count, self.actions_size))

        self.values_b           = self.values_b.reshape((self.buffer_size*self.envs_count, ))        

        self.actions_b          = self.actions_b.reshape((self.buffer_size*self.envs_count, ))
        self.rewards_b          = self.rewards_b.reshape((self.buffer_size*self.envs_count, ))
        self.dones_b            = self.dones_b.reshape((self.buffer_size*self.envs_count, ))

        self.returns_b          = self.returns_b.reshape((self.buffer_size*self.envs_count, ))
        self.advantages_b       = self.advantages_b.reshape((self.buffer_size*self.envs_count, ))
    

    def sample_batch(self, batch_size, device):
        indices     = numpy.random.randint(0, self.buffer_size*batch_size, size=batch_size)

        states      = torch.zeros((self.envs_count*batch_size, ) + self.state_shape, dtype=torch.float).to(self.device)
        logits      = torch.zeros((self.envs_count*batch_size, self.actions_size), dtype=torch.float).to(self.device)
        
        values      = torch.zeros((self.envs_count*batch_size, ), dtype=torch.float).to(self.device)

        actions     = torch.zeros((self.envs_count*batch_size, ), dtype=int).to(self.device)
        rewards     = torch.zeros((self.envs_count*batch_size, ), dtype=torch.float).to(self.device)
        dones       = torch.zeros((self.envs_count*batch_size, ), dtype=torch.float).to(self.device)
       
        returns     = torch.zeros((self.envs_count*batch_size, ), dtype=torch.float).to(self.device)
        advantages  = torch.zeros((self.envs_count*batch_size, ), dtype=torch.float).to(self.device)

                        
        states      = torch.from_numpy(numpy.take(self.states_b, indices, axis=0)).to(device)
        logits      = torch.from_numpy(numpy.take(self.logits_b, indices, axis=0)).to(device)
        
        values      = torch.from_numpy(numpy.take(self.values_b, indices, axis=0)).to(device)
        
        actions     = torch.from_numpy(numpy.take(self.actions_b, indices, axis=0)).to(device)
        rewards     = torch.from_numpy(numpy.take(self.rewards_b, indices, axis=0)).to(device)
        dones       = torch.from_numpy(numpy.take(self.dones_b, indices, axis=0)).to(device)

        returns     = torch.from_numpy(numpy.take(self.returns_b, indices, axis=0)).to(device)
        advantages  = torch.from_numpy(numpy.take(self.advantages_b, indices, axis=0)).to(device)

        return states, logits, values, actions, rewards, dones, returns, advantages 
    
    
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