import torch
import numpy

class PolicyBufferContinuous:

    def __init__(self, buffer_size, state_shape, actions_size, envs_count, device):
        
        self.buffer_size    = buffer_size
        self.state_shape    = state_shape
        self.actions_size   = actions_size
        self.envs_count     = envs_count
        self.device         = device

        self.clear() 

    def add(self, state, value, action, action_mu, action_var, reward, done):
        self.states_b[self.ptr]        = state.copy()
        self.values_b[self.ptr]        = value.copy()
        self.actions_b[self.ptr]       = action.copy()
        self.actions_mu_b[self.ptr]    = action_mu.copy()
        self.actions_var_b[self.ptr]   = action_var.copy()
        self.rewards_b[self.ptr]       = reward.copy()

        self.dones_b[self.ptr]         = (1.0*done).copy()
        
        self.ptr = self.ptr + 1 
        

    def is_full(self):
        if self.ptr >= self.buffer_size:
            return True

        return False 
 
    def clear(self):
        self.states_b           = numpy.zeros((self.buffer_size, self.envs_count, ) + self.state_shape, dtype=numpy.float32)        
        self.values_b           = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)

        self.actions_b          = numpy.zeros((self.buffer_size, self.envs_count, self.actions_size), dtype=numpy.float32)
        self.actions_mu_b       = numpy.zeros((self.buffer_size, self.envs_count, self.actions_size), dtype=numpy.float32)
        self.actions_var_b      = numpy.zeros((self.buffer_size, self.envs_count, self.actions_size), dtype=numpy.float32)

        self.rewards_b          = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)
        self.dones_b            = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)
       
      
        self.ptr = 0 


    def compute_returns(self, gamma = 0.99, lam = 0.95):
        self.returns_b, self.advantages_b = self._gae_fast(self.rewards_b, self.values_b, self.dones_b, gamma, lam)
        
        #reshape buffer for faster batch sampling
        self.states_b           = self.states_b.reshape((self.buffer_size*self.envs_count, ) + self.state_shape)
        self.values_b          = self.values_b.reshape((self.buffer_size*self.envs_count, ))

        self.actions_b          = self.actions_b.reshape((self.buffer_size*self.envs_count, self.actions_size))
        self.actions_mu_b       = self.actions_mu_b.reshape((self.buffer_size*self.envs_count, self.actions_size))
        self.actions_var_b      = self.actions_var_b.reshape((self.buffer_size*self.envs_count, self.actions_size))

        self.rewards_b          = self.rewards_b.reshape((self.buffer_size*self.envs_count, ))
        self.dones_b            = self.dones_b.reshape((self.buffer_size*self.envs_count, ))

        self.returns_b          = self.returns_b.reshape((self.buffer_size*self.envs_count, ))
        self.advantages_b       = self.advantages_b.reshape((self.buffer_size*self.envs_count, ))


    def sample_batch(self, batch_size, device):
        indices     = numpy.random.randint(0, self.envs_count*self.buffer_size, size=batch_size*self.envs_count)

                        
        states      = torch.from_numpy(numpy.take(self.states_b, indices, axis=0)).to(device)
        
        values      = torch.from_numpy(numpy.take(self.values_b, indices, axis=0)).to(device)
        
        actions      = torch.from_numpy(numpy.take(self.actions_b, indices, axis=0)).to(device)
        actions_mu   = torch.from_numpy(numpy.take(self.actions_mu_b, indices, axis=0)).to(device)
        actions_var  = torch.from_numpy(numpy.take(self.actions_var_b, indices, axis=0)).to(device)

        rewards     = torch.from_numpy(numpy.take(self.rewards_b, indices, axis=0)).to(device)
        dones       = torch.from_numpy(numpy.take(self.dones_b, indices, axis=0)).to(device)

        returns     = torch.from_numpy(numpy.take(self.returns_b, indices, axis=0)).to(device)
        advantages  = torch.from_numpy(numpy.take(self.advantages_b, indices, axis=0)).to(device)

       
        return states, values, actions, actions_mu, actions_var, rewards, dones, returns, advantages 

    def _gae_fast(self, rewards, values, dones, gamma, lam):
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