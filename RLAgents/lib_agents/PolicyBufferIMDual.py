import torch
import numpy

class PolicyBufferIMDual:

    def __init__(self, buffer_size, state_shape, actions_size, envs_count, device):
        
        self.buffer_size    = buffer_size
        self.state_shape    = state_shape
        self.actions_size   = actions_size
        self.envs_count     = envs_count
        self.device         = device
 
        self.clear() 
 
    def add(self, env, state, logits, value_ext, value_int_a, value_int_b, action, reward, internal_a, internal_b, done):

        if done != 0:  
            done_ = 1.0
        else:
            done_ = 0.0
        
        self.states_b[env][self.ptr]    = state
        self.logits_b[env][self.ptr]    = logits
        
        self.values_ext_b[env][self.ptr]= value_ext
        self.values_int_a_b[env][self.ptr]= value_int_a
        self.values_int_b_b[env][self.ptr]= value_int_b

        self.actions_b[env][self.ptr]   = action
        
        self.rewards_b[env][self.ptr]       = reward
        self.internal_a_b[env][self.ptr]    = internal_a
        self.internal_b_b[env][self.ptr]    = internal_b

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
        self.values_int_a_b     = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=numpy.float32)
        self.values_int_b_b     = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=numpy.float32)

        self.actions_b          = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=int)
        
        self.rewards_b          = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=numpy.float32)
        self.internal_a_b       = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=numpy.float32)
        self.internal_b_b       = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=numpy.float32)

        self.dones_b            = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=numpy.float32)

        self.returns_ext_b      = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=numpy.float32)
        self.returns_int_a_b    = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=numpy.float32)
        self.returns_int_b_b    = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=numpy.float32)

        self.advantages_ext_b   = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=numpy.float32)
        self.advantages_int_a_b = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=numpy.float32)
        self.advantages_int_b_b = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=numpy.float32)

        self.ptr = 0 


    def compute_returns(self, gamma_ext = 0.99, gamma_int = 0.9, lam = 0.95):
        
        for e in range(self.envs_count):
            
            count = len(self.rewards_b[e])
            last_gae  = 0.0

            for n in reversed(range(count-1)):
            
                if self.dones_b[e][n] > 0:
                    delta       = self.rewards_b[e][n] - self.values_ext_b[e][n]
                    last_gae    = delta
                else:
                    delta       = self.rewards_b[e][n] + gamma_ext*self.values_ext_b[e][n+1] - self.values_ext_b[e][n]
                    last_gae    = delta + gamma_ext*lam*last_gae

                self.returns_ext_b[e][n]    = last_gae + self.values_ext_b[e][n]
                self.advantages_ext_b[e][n] = last_gae
        
        for e in range(self.envs_count):
            
            count = len(self.internal_a_b[e])
            last_gae  = 0.0

            for n in reversed(range(count-1)):
            
                if self.dones_b[e][n] > 0:
                    delta       = self.internal_a_b[e][n] - self.values_int_a_b[e][n]
                    last_gae    = delta
                else:
                    delta       = self.internal_a_b[e][n] + gamma_int*self.values_int_a_b[e][n+1] - self.values_int_a_b[e][n]
                    last_gae    = delta + gamma_int*lam*last_gae

                self.returns_int_a_b[e][n]    = last_gae + self.values_int_a_b[e][n]
                self.advantages_int_a_b[e][n] = last_gae

        for e in range(self.envs_count):
            
            count = len(self.internal_b_b[e])
            last_gae  = 0.0

            for n in reversed(range(count-1)):
            
                if self.dones_b[e][n] > 0:
                    delta       = self.internal_b_b[e][n] - self.values_int_b_b[e][n]
                    last_gae    = delta
                else:
                    delta       = self.internal_b_b[e][n] + gamma_int*self.values_int_b_b[e][n+1] - self.values_int_b_b[e][n]
                    last_gae    = delta + gamma_int*lam*last_gae

                self.returns_int_b_b[e][n]    = last_gae + self.values_int_b_b[e][n]
                self.advantages_int_b_b[e][n] = last_gae
    
    def sample_batch(self, batch_size, device):

        states              = torch.zeros((self.envs_count, batch_size, ) + self.state_shape, dtype=torch.float).to(self.device)
        states_next         = torch.zeros((self.envs_count, batch_size, ) + self.state_shape, dtype=torch.float).to(self.device)
        logits              = torch.zeros((self.envs_count, batch_size, self.actions_size), dtype=torch.float).to(self.device)
        
        actions             = torch.zeros((self.envs_count, batch_size, ), dtype=int).to(self.device)
       
        returns_ext         = torch.zeros((self.envs_count, batch_size, ), dtype=torch.float).to(self.device)
        returns_int_a       = torch.zeros((self.envs_count, batch_size, ), dtype=torch.float).to(self.device)
        returns_int_b       = torch.zeros((self.envs_count, batch_size, ), dtype=torch.float).to(self.device)

        advantages_ext      = torch.zeros((self.envs_count, batch_size, ), dtype=torch.float).to(self.device)
        advantages_int_a    = torch.zeros((self.envs_count, batch_size, ), dtype=torch.float).to(self.device)
        advantages_int_b    = torch.zeros((self.envs_count, batch_size, ), dtype=torch.float).to(self.device)

        for e in range(self.envs_count):
            indices     = numpy.random.randint(0, self.buffer_size, size=batch_size)
            indices_next= numpy.clip(indices+1, 0, self.buffer_size-1)

            states[e]       = torch.from_numpy(numpy.take(self.states_b[e], indices, axis=0)).to(device)
            states_next[e]  = torch.from_numpy(numpy.take(self.states_b[e], indices_next, axis=0)).to(device)

            logits[e]   = torch.from_numpy(numpy.take(self.logits_b[e], indices, axis=0)).to(device)
            
            actions[e]  = torch.from_numpy(numpy.take(self.actions_b[e], indices, axis=0)).to(device)
            
            returns_ext[e]      = torch.from_numpy(numpy.take(self.returns_ext_b[e], indices, axis=0)).to(device)
            returns_int_a[e]    = torch.from_numpy(numpy.take(self.returns_int_a_b[e], indices, axis=0)).to(device)
            returns_int_b[e]    = torch.from_numpy(numpy.take(self.returns_int_b_b[e], indices, axis=0)).to(device)

            advantages_ext[e]   = torch.from_numpy(numpy.take(self.advantages_ext_b[e], indices, axis=0)).to(device)
            advantages_int_a[e] = torch.from_numpy(numpy.take(self.advantages_int_a_b[e], indices, axis=0)).to(device)
            advantages_int_b[e] = torch.from_numpy(numpy.take(self.advantages_int_b_b[e], indices, axis=0)).to(device)

        states              = states.reshape((self.envs_count*batch_size, ) + self.state_shape)
        states_next         = states_next.reshape((self.envs_count*batch_size, ) + self.state_shape)
        logits              = logits.reshape((self.envs_count*batch_size, self.actions_size))
        actions             = actions.reshape((self.envs_count*batch_size, ))
        returns_ext         = returns_ext.reshape((self.envs_count*batch_size, ))
        returns_int_a       = returns_int_a.reshape((self.envs_count*batch_size, ))
        returns_int_b       = returns_int_b.reshape((self.envs_count*batch_size, ))
        advantages_ext      = advantages_ext.reshape((self.envs_count*batch_size, ))
        advantages_int_a    = advantages_int_a.reshape((self.envs_count*batch_size, ))
        advantages_int_b    = advantages_int_b.reshape((self.envs_count*batch_size, ))

        return states, states_next, logits, actions, returns_ext, returns_int_a, returns_int_b, advantages_ext, advantages_int_a, advantages_int_b 
 
    