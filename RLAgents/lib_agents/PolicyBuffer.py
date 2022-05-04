import torch

class PolicyBuffer:

    def __init__(self, buffer_size, state_shape, actions_size, envs_count):
        self.buffer_size    = buffer_size
        self.state_shape    = state_shape
        self.actions_size   = actions_size
        self.envs_count     = envs_count
      
        self.clear()   
 
    def add(self, state, logits, value, action, reward, done):
        
        self.states[self.ptr]    = state.clone() 
        self.logits[self.ptr]    = logits.clone()
        self.values[self.ptr]    = value.clone()
        self.actions[self.ptr]   = action.clone()
        
        self.reward[self.ptr]    = reward.clone()
        self.dones[self.ptr]     = (1.0*done).clone()
        
        self.ptr = self.ptr + 1 


    def is_full(self):
        if self.ptr >= self.buffer_size:
            return True

        return False 
 
    def clear(self):
        self.states     = torch.zeros((self.buffer_size, self.envs_count, ) + self.state_shape, dtype=torch.float32)

        self.logits     = torch.zeros((self.buffer_size, self.envs_count, self.actions_size), dtype=torch.float32)

        self.values     = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)        
     
        self.actions    = torch.zeros((self.buffer_size, self.envs_count, ), dtype=int)
        
        self.reward     = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)
        self.dones      = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)

        self.ptr = 0  
 

    def compute_returns(self, gamma, lam = 0.95):
        self.returns, self.advantages   = self._gae(self.reward, self.values, self.dones, gamma, lam)
        
        #reshape buffer for faster batch sampling
        self.states     = self.states.reshape((self.buffer_size*self.envs_count, ) + self.state_shape)
        self.logits     = self.logits.reshape((self.buffer_size*self.envs_count, self.actions_size))

        self.values     = self.values.reshape((self.buffer_size*self.envs_count, ))        
     
        self.actions    = self.actions.reshape((self.buffer_size*self.envs_count, ))
        
        self.reward     = self.reward.reshape((self.buffer_size*self.envs_count, ))
      
        self.dones      = self.dones.reshape((self.buffer_size*self.envs_count, ))

        self.returns    = self.returns.reshape((self.buffer_size*self.envs_count, ))
        self.advantages = self.advantages.reshape((self.buffer_size*self.envs_count, ))


    def sample_batch(self, batch_size, device = "cpu"):
        indices         = torch.randint(0, self.envs_count*self.buffer_size, size=(batch_size*self.envs_count, ))
        indices_next    = torch.clip(indices + 1, 0, self.envs_count*self.buffer_size - 1)

        print(indices)
        print(indices_next)
        print("\n\n\n")

        states          = torch.index_select(self.states, dim=0, index=indices).to(device)
        states_next     = torch.index_select(self.states, dim=0, index=indices_next).to(device)
        logits          = torch.index_select(self.logits, dim=0, index=indices).to(device)
        
        actions         = torch.index_select(self.actions, dim=0, index=indices).to(device)
        
        returns         = torch.index_select(self.returns, dim=0, index=indices).to(device)
        advantages      = torch.index_select(self.advantages, dim=0, index=indices).to(device)
       
        return states, states_next, logits, actions, returns, advantages
    
   
    def sample_states(self, batch_size, far_ratio = 0.5, device = "cpu"): 
        count = self.envs_count*self.buffer_size
 
        indices_a       = torch.randint(0, count, size=(batch_size, ))
        
        indices_close   = indices_a 
 
        indices_far     = torch.randint(0, count, size=(batch_size, ))

        labels          = (torch.rand(batch_size) > far_ratio)
        
        #label 0 = close states
        #label 1 = distant states
        indices_b       = torch.logical_not(labels)*indices_close + labels*indices_far

        states_a        = torch.index_select(self.states, dim=0, index=indices_a).float().to(device)
        states_b        = torch.index_select(self.states, dim=0, index=indices_b).float().to(device)
        labels_t        = labels.float().to(device) 
        
        return states_a, states_b, labels_t
    
 
    def _gae(self, rewards, values, dones, gamma, lam):
        buffer_size = rewards.shape[0]
        envs_count  = rewards.shape[1]
        
        returns     = torch.zeros((buffer_size, envs_count), dtype=torch.float32)
        advantages  = torch.zeros((buffer_size, envs_count), dtype=torch.float32)

        last_gae    = torch.zeros((envs_count), dtype=torch.float32)
        
        for n in reversed(range(buffer_size-1)):
            delta           = rewards[n] + gamma*values[n+1]*(1.0 - dones[n]) - values[n]
            last_gae        = delta + gamma*lam*last_gae*(1.0 - dones[n])
            
            returns[n]      = last_gae + values[n]
            advantages[n]   = last_gae
 
        return returns, advantages
