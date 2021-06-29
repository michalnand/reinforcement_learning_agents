import torch
import numpy

class PolicyBufferIM:

    def __init__(self, buffer_size, state_shape, actions_size, envs_count, device):
        
        self.buffer_size    = buffer_size
        self.state_shape    = state_shape
        self.actions_size   = actions_size
        self.envs_count     = envs_count
        self.device         = device
 
        self.clear() 
 
    def add(self, env, state, logits, value_ext, value_int, action, reward, internal, done):

        if done != 0:  
            done_ = 1.0
        else: 
            done_ = 0.0
        
        self.states_b[env][self.ptr]    = state
        self.logits_b[env][self.ptr]    = logits
        
        self.values_ext_b[env][self.ptr]= value_ext
        self.values_int_b[env][self.ptr]= value_int

        self.actions_b[env][self.ptr]   = action
        
        self.rewards_b[env][self.ptr]   = reward
        self.internal_b[env][self.ptr]  = internal

        self.dones_b[env][self.ptr]     = done_
        
        
        if env == self.envs_count - 1:
            self.ptr = self.ptr + 1 


    def is_full(self):
        if self.ptr >= self.buffer_size:
            return True

        return False 
 
    def clear(self):
        self.states_b           = numpy.zeros((self.envs_count, self.buffer_size, ) + self.state_shape, dtype=numpy.float32)
        self.logits_b           = numpy.zeros((self.envs_count, self.buffer_size, self.actions_size), dtype=numpy.float32)

        self.values_ext_b       = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=numpy.float32)        
        self.values_int_b       = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=numpy.float32)

        self.actions_b          = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=int)
        
        self.rewards_b          = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=numpy.float32)
        self.internal_b         = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=numpy.float32)

        self.dones_b            = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=numpy.float32)

        self.returns_int_b      = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=numpy.float32)
        self.returns_ext_b      = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=numpy.float32)
        
        self.advantages_int_b   = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=numpy.float32)
        self.advantages_ext_b   = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=numpy.float32)

        self.ptr = 0 


    def compute_returns(self, gamma_ext = 0.99, gamma_int = 0.9, lam = 0.95):
        self.returns_ext_b, self.advantages_ext_b = self._gae_fast(self, self.rewards_b, self.values_ext_b, self.dones_b, gamma_ext, lam)
        self.returns_int_b, self.advantages_int_b = self._gae_fast(self, self.internal_b, self.values_int_b, self.dones_b, gamma_int, lam)
        

    def sample_batch(self, batch_size, device):

        states           = torch.zeros((self.envs_count, batch_size, ) + self.state_shape, dtype=torch.float).to(self.device)
        states_next      = torch.zeros((self.envs_count, batch_size, ) + self.state_shape, dtype=torch.float).to(self.device)
        logits           = torch.zeros((self.envs_count, batch_size, self.actions_size), dtype=torch.float).to(self.device)
        
        actions          = torch.zeros((self.envs_count, batch_size, ), dtype=int).to(self.device)
       
        returns_ext      = torch.zeros((self.envs_count, batch_size, ), dtype=torch.float).to(self.device)
        returns_int      = torch.zeros((self.envs_count, batch_size, ), dtype=torch.float).to(self.device)

        advantages_ext   = torch.zeros((self.envs_count, batch_size, ), dtype=torch.float).to(self.device)
        advantages_int   = torch.zeros((self.envs_count, batch_size, ), dtype=torch.float).to(self.device)

        for e in range(self.envs_count):
            indices     = numpy.random.randint(0, self.buffer_size, size=batch_size)
            indices_next= numpy.clip(indices + 1, 0, self.buffer_size-1)

            states[e]       = torch.from_numpy(numpy.take(self.states_b[e], indices, axis=0)).to(device)
            states_next[e]  = torch.from_numpy(numpy.take(self.states_b[e], indices_next, axis=0)).to(device)

            logits[e]   = torch.from_numpy(numpy.take(self.logits_b[e], indices, axis=0)).to(device)
            
            actions[e]  = torch.from_numpy(numpy.take(self.actions_b[e], indices, axis=0)).to(device)
            
            returns_ext[e]      = torch.from_numpy(numpy.take(self.returns_ext_b[e], indices, axis=0)).to(device)
            returns_int[e]      = torch.from_numpy(numpy.take(self.returns_int_b[e], indices, axis=0)).to(device)

            advantages_ext[e]   = torch.from_numpy(numpy.take(self.advantages_ext_b[e], indices, axis=0)).to(device)
            advantages_int[e]   = torch.from_numpy(numpy.take(self.advantages_int_b[e], indices, axis=0)).to(device)

        states          = states.reshape((self.envs_count*batch_size, ) + self.state_shape)
        states_next     = states_next.reshape((self.envs_count*batch_size, ) + self.state_shape)
        logits          = logits.reshape((self.envs_count*batch_size, self.actions_size))
        actions         = actions.reshape((self.envs_count*batch_size, ))
        returns_ext     = returns_ext.reshape((self.envs_count*batch_size, ))
        returns_int     = returns_int.reshape((self.envs_count*batch_size, ))
        advantages_ext  = advantages_ext.reshape((self.envs_count*batch_size, ))
        advantages_int  = advantages_int.reshape((self.envs_count*batch_size, ))

        return states, states_next, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int 
 
    
    def _gae_fast(self, rewards, values, dones, gamma = 0.99, lam = 0.9):
        envs_count  = rewards.shape[0]
        buffer_size = rewards.shape[1]

        returns     = numpy.zeros((buffer_size, envs_count), dtype=numpy.float32)
        advantages  = numpy.zeros((buffer_size, envs_count), dtype=numpy.float32)

        rewards_t   = numpy.transpose(rewards)
        values_t    = numpy.transpose(values)
        dones_t     = numpy.transpose(dones)

        last_gae    = numpy.zeros((envs_count), dtype=numpy.float32)
        
        for n in reversed(range(buffer_size-1)):
            delta           = rewards_t[n] + gamma*values_t[n+1]*(1.0 - dones_t[n]) - values_t[n]
            last_gae        = delta + gamma*lam*last_gae*(1.0 - dones_t[n])
            
            returns[n]      = last_gae + values_t[n]
            advantages[n]   = last_gae

        returns     = numpy.transpose(returns)
        advantages  = numpy.transpose(advantages)

        return returns, advantages