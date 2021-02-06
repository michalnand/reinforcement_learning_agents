import torch
import numpy


class PolicyBufferContinuous:

    def __init__(self, buffer_size, state_shape, actions_size, device):
        self.buffer_size    = buffer_size
        self.state_shape    = state_shape
        self.actions_size   = actions_size
        self.device         = device

        self.clear()

    def add(self, state, value, action, action_mu, action_var, reward, done):
        
        if self.ptr < self.buffer_size:

            self.states_b[self.ptr]         = state
            self.values_b[self.ptr]         = value
            
            self.actions_b[self.ptr]        = action
            self.actions_mu_b[self.ptr]     = action_mu
            self.actions_var_b[self.ptr]    = action_var

            self.rewards_b[self.ptr]        = reward
            self.dones_b[self.ptr]          = done
        
            self.ptr = self.ptr + 1 

    def is_full(self):
        if self.ptr >= self.buffer_size:
            return True

        return False 

    def clear(self):
        self.states_b           = torch.zeros((self.buffer_size, ) + self.state_shape).to(self.device)
        self.values_b           = torch.zeros((self.buffer_size, 1)).to(self.device)

        self.actions_b          = torch.zeros((self.buffer_size, self.actions_size)).to(self.device)
        self.actions_mu_b       = torch.zeros((self.buffer_size, self.actions_size)).to(self.device)
        self.actions_var_b      = torch.zeros((self.buffer_size, self.actions_size)).to(self.device)
        
        self.rewards_b          = torch.zeros((self.buffer_size, )).to(self.device)
        self.dones_b            = torch.zeros((self.buffer_size, )).to(self.device)
       
        self.returns_b         = torch.zeros((self.buffer_size, )).to(self.device)

        self.ptr = 0 

    def cut_zeros(self):
        self.states_b           = self.states_b[0:self.ptr]
        self.values_b           = self.values_b[0:self.ptr]

        self.actions_b          = self.actions_mu_b[0:self.ptr]
        self.actions_mu_b       = self.actions_mu_b[0:self.ptr]
        self.actions_var_b      = self.actions_var_b[0:self.ptr]

        self.rewards_b          = self.rewards_b[0:self.ptr]
        self.dones_b            = self.dones_b[0:self.ptr]
       
        self.returns_b         = self.returns_b[0:self.ptr]


    def compute_returns(self, gamma, normalise = False):
        
        if normalise:
            self.rewards_b = (self.rewards_b - self.rewards_b.mean())/(self.rewards_b.std() + 0.00001)
        
        r = 0.0
        for n in reversed(range(len(self.rewards_b))):

            if self.dones_b[n]:
                gamma_ = 0.0
            else:
                gamma_ = gamma

            r = self.rewards_b[n] + gamma_*r
            self.returns_b[n] = r