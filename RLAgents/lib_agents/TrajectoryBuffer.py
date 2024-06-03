import torch

class TrajectoryBuffer:

    def __init__(self, buffer_size, state_shape, actions_size, envs_count):
        self.buffer_size    = buffer_size
        self.state_shape    = state_shape
        self.actions_size   = actions_size
        self.envs_count     = envs_count

        self.hidden_state = None
      
        self.clear()   
 
    def add(self, state, logits, value, action, reward, done, hidden_state = None):  
        self.states[self.ptr]    = state.clone() 
        self.logits[self.ptr]    = logits.clone()
        self.values[self.ptr]    = value.clone()
        self.actions[self.ptr]   = action.clone()
        
        self.reward[self.ptr]    = reward.clone()
        self.dones[self.ptr]     = (1.0*done).clone()
        
        if hidden_state is not None:
            
            if self.hidden_state is None:
                self.hidden_shape   = hidden_state.shape[1:]    
                self.hidden_state   = torch.zeros((self.buffer_size, self.envs_count, ) + self.hidden_shape, dtype=torch.float32)
            
            self.hidden_state[self.ptr] = hidden_state.clone()
        
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

        self.hidden_state   = None

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

        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.reshape((self.buffer_size*self.envs_count, self.hidden_state.shape[2]))

    def sample_batch(self, batch_size, device):
        indices         = torch.randint(0, self.envs_count*self.buffer_size, size=(batch_size, ))

        states          = self.states[indices].to(device)
        logits          = self.logits[indices].to(device)
        
        actions         = self.actions[indices].to(device)
        
        returns         = self.returns[indices].to(device)
        advantages      = self.advantages[indices].to(device)
       
        return states, logits, actions, returns, advantages
    

     def sample_batch_seq(self, seq_length, batch_size, device):
        indices        = torch.randint(0, self.envs_count*(self.buffer_size - seq_length), size=(batch_size, ))
        
        states         = torch.zeros((seq_length, batch_size, ) + self.state_shape,  dtype=torch.float32, device=device)
        hidden_states  = torch.zeros((seq_length, batch_size, ) + self.hidden_shape, dtype=torch.float32, device=device)

        for n in range(seq_length):
            states[n]        = self.states[indices].to(device)
            hidden_states[n] = self.hidden_state[indices].to(device)

            if n == (seq_length-1): 
                logits     = self.logits[indices].to(device)
                
                actions    = self.actions[indices].to(device)
                
                returns    = self.returns[indices].to(device)
                advantages = self.advantages[indices].to(device)

            indices+= self.envs_count 

        return states, logits, actions, returns, advantages, hidden_states
    

    def sample_states_pairs(self, batch_size, max_distance, device):
        count           = self.buffer_size*self.envs_count
        max_distance_   = torch.randint(0, 1 + max_distance, (batch_size, ))

        indices         = torch.randint(0, count, size=(batch_size, ))
        indices_similar = torch.clip(indices + max_distance_*self.envs_count, 0, count-1)
       
        states_a  = (self.states[indices]).to(device)
        states_b  = (self.states[indices_similar]).to(device)

        return states_a, states_b
     
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
