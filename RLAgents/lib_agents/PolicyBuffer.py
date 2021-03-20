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

    def add(self, env, state, logits, value, action, reward, done, internal = 0.0):

        if done != 0: 
            done_ = 1.0
        else:
            done_ = 0.0
        
        self.states_b[env][self.ptr]    = state
        self.logits_b[env][self.ptr]    = logits
        self.values_b[env][self.ptr]    = value
        self.actions_b[env][self.ptr]   = action
        self.rewards_b[env][self.ptr]   = reward
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
        
        self.values_b       = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=numpy.float32)

        self.actions_b          = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=int)
        self.rewards_b          = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=numpy.float32)
        self.dones_b            = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=numpy.float32)
       
        self.returns_b          = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=numpy.float32)
        
        self.advantages_b   = numpy.zeros((self.envs_count, self.buffer_size, ), dtype=numpy.float32)

        self.ptr = 0 


    def compute_returns(self, gamma = 0.99, lam = 0.95):
        
        for e in range(self.envs_count):
            
            count = len(self.rewards_b[e])
            last_gae  = 0.0

            for n in reversed(range(count-1)):
            
                if self.dones_b[e][n] > 0:
                    delta       = self.rewards_b[e][n] - self.values_b[e][n]
                    last_gae    = delta
                else:
                    delta       = self.rewards_b[e][n] + gamma*self.values_b[e][n+1] - self.values_b[e][n]
                    last_gae    = delta + gamma*lam*last_gae

                self.returns_b[e][n]    = last_gae + self.values_b[e][n]
                self.advantages_b[e][n] = last_gae
           

    def sample_batch(self, batch_size, device):

        states           = torch.zeros((self.envs_count, batch_size, ) + self.state_shape, dtype=torch.float).to(self.device)
        logits           = torch.zeros((self.envs_count, batch_size, self.actions_size), dtype=torch.float).to(self.device)
        
        values           = torch.zeros((self.envs_count, batch_size, ), dtype=torch.float).to(self.device)

        actions          = torch.zeros((self.envs_count, batch_size, ), dtype=int).to(self.device)
        rewards          = torch.zeros((self.envs_count, batch_size, ), dtype=torch.float).to(self.device)
        dones            = torch.zeros((self.envs_count, batch_size, ), dtype=torch.float).to(self.device)
       
        returns          = torch.zeros((self.envs_count, batch_size, ), dtype=torch.float).to(self.device)
        advantages       = torch.zeros((self.envs_count, batch_size, ), dtype=torch.float).to(self.device)

        for e in range(self.envs_count):
            indices     = numpy.random.randint(0, self.buffer_size, size=batch_size)
                        
            states[e]   = torch.from_numpy(numpy.take(self.states_b[e], indices, axis=0)).to(device)
            logits[e]   = torch.from_numpy(numpy.take(self.logits_b[e], indices, axis=0)).to(device)
            
            values[e]   = torch.from_numpy(numpy.take(self.values_b[e], indices, axis=0)).to(device)
            
            actions[e]  = torch.from_numpy(numpy.take(self.actions_b[e], indices, axis=0)).to(device)
            rewards[e]  = torch.from_numpy(numpy.take(self.rewards_b[e], indices, axis=0)).to(device)
            dones[e]    = torch.from_numpy(numpy.take(self.dones_b[e], indices, axis=0)).to(device)

            returns[e]      = torch.from_numpy(numpy.take(self.returns_b[e], indices, axis=0)).to(device)
            advantages[e]   = torch.from_numpy(numpy.take(self.advantages_b[e], indices, axis=0)).to(device)

        states      = states.reshape((self.envs_count*batch_size, ) + self.state_shape)
        logits      = logits.reshape((self.envs_count*batch_size, self.actions_size))
        values      = values.reshape((self.envs_count*batch_size, ))
        actions     = actions.reshape((self.envs_count*batch_size, ))
        rewards     = rewards.reshape((self.envs_count*batch_size, ))
        dones       = dones.reshape((self.envs_count*batch_size, ))
        returns     = returns.reshape((self.envs_count*batch_size, ))
        advantages  = advantages.reshape((self.envs_count*batch_size, ))

        return states, logits, values, actions, rewards, dones, returns, advantages 
