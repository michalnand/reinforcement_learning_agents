import torch

class PolicyBufferIMNew:

    def __init__(self, buffer_size, state_shape, actions_size, envs_count):
        self.buffer_size    = buffer_size
        self.state_shape    = state_shape
        self.actions_size   = actions_size
        self.envs_count     = envs_count
      
        self.clear()     
 
    def add(self, state, logits, value_ext, value_int, action, reward_ext, reward_int, done, hidden_state = None):
        
        self.states[self.ptr]    = state.clone() 
        self.logits[self.ptr]    = logits.clone()
        
        self.values_ext[self.ptr]= value_ext.clone()
        self.values_int[self.ptr]= value_int.clone()

        self.actions[self.ptr]   = action.clone()
        
        self.reward_ext[self.ptr]  = reward_ext.clone()
        self.reward_int[self.ptr]  = reward_int.clone()

        self.dones[self.ptr]       = (1.0*done).clone()

        if hidden_state is not None:
            if self.hidden_state is None:
                self.hidden_state   = torch.zeros((self.buffer_size, self.envs_count, ) + hidden_state.shape[1:], dtype=torch.float32)
            
            self.hidden_state[self.ptr] = hidden_state.clone()

        self.ptr = self.ptr + 1 


    def is_full(self):
        if self.ptr >= self.buffer_size:
            return True

        return False 
 
    def clear(self):
        self.states         = torch.zeros((self.buffer_size, self.envs_count, ) + self.state_shape, dtype=torch.float32)

        self.logits         = torch.zeros((self.buffer_size, self.envs_count, self.actions_size), dtype=torch.float32)

        self.values_ext     = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)        
        self.values_int     = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)

        self.actions        = torch.zeros((self.buffer_size, self.envs_count, ), dtype=int)
        
        self.reward_ext     = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)
        self.reward_int     = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)

        self.dones          = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)

        self.hidden_state   = None

        self.exploration_mode = torch.zeros((self.buffer_size, self.envs_count), dtype=torch.float32)
        self.episode_steps    = torch.zeros((self.buffer_size, self.envs_count), dtype=torch.float32)

        self.ptr = 0  
 

    def compute_returns(self, gamma_ext, gamma_int, lam = 0.95):
        self.returns_ext, self.advantages_ext = self._gae(self.reward_ext, self.values_ext, self.dones, gamma_ext, lam)
        self.returns_int, self.advantages_int = self._gae(self.reward_int, self.values_int, self.dones, gamma_int, lam)
        
        #reshape buffer for faster batch sampling
        self.states           = self.states.reshape((self.buffer_size*self.envs_count, ) + self.state_shape)
        self.logits           = self.logits.reshape((self.buffer_size*self.envs_count, self.actions_size))

        self.values_ext       = self.values_ext.reshape((self.buffer_size*self.envs_count, ))        
        self.values_int       = self.values_int.reshape((self.buffer_size*self.envs_count, ))

        self.actions          = self.actions.reshape((self.buffer_size*self.envs_count, ))
        
        self.reward_ext       = self.reward_ext.reshape((self.buffer_size*self.envs_count, ))
        self.reward_int       = self.reward_int.reshape((self.buffer_size*self.envs_count, ))

        self.dones            = self.dones.reshape((self.buffer_size*self.envs_count, ))

        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.reshape((self.buffer_size*self.envs_count, self.hidden_state.shape[2]))

        self.returns_ext      = self.returns_ext.reshape((self.buffer_size*self.envs_count, ))
        self.advantages_ext   = self.advantages_ext.reshape((self.buffer_size*self.envs_count, ))

        self.returns_int      = self.returns_int.reshape((self.buffer_size*self.envs_count, ))
        self.advantages_int   = self.advantages_int.reshape((self.buffer_size*self.envs_count, ))

      
    def sample_batch(self, batch_size, device = "cpu"):
        indices         = torch.randint(0, self.envs_count*self.buffer_size, size=(batch_size, ))

        states          = (self.states[indices]).to(device)
        logits          = (self.logits[indices]).to(device)
        
        actions         = (self.actions[indices]).to(device)
         
        returns_ext     = (self.returns_ext[indices]).to(device)
        returns_int     = (self.returns_int[indices]).to(device)

        advantages_ext  = (self.advantages_ext[indices]).to(device)
        advantages_int  = (self.advantages_int[indices]).to(device)

        if self.hidden_state is not None:
            hidden_state  = (self.hidden_state[indices]).to(device)
        else:
            hidden_state  = None
 
        return states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int, hidden_state
    
   

    def sample_states_pairs(self, batch_size, max_distance = 0, device = "cpu"):
        count           = self.buffer_size*self.envs_count

        max_distance_   = torch.randint(0, 1 + max_distance, (batch_size, ))

        indices         = torch.randint(0, count, size=(batch_size, ))
        indices_similar = torch.clip(indices + max_distance_*self.envs_count, 0, count-1)
      
        states_now      = (self.states[indices]).to(device)
        states_similar  = (self.states[indices_similar]).to(device)
     
        return states_now, states_similar
    

    def sample_states_pairs_hidden(self, batch_size, max_distance = 0, device = "cpu"):
        count           = self.buffer_size*self.envs_count

        max_distance_   = torch.randint(0, 1 + max_distance, (batch_size, ))

        indices         = torch.randint(0, count, size=(batch_size, ))
        indices_similar = torch.clip(indices + max_distance_*self.envs_count, 0, count-1)
      
        states_now      = (self.states[indices]).to(device)
        states_similar  = (self.states[indices_similar]).to(device)

        hidden_now      = (self.hidden_state[indices]).to(device)
        hidden_similar  = (self.hidden_state[indices_similar]).to(device)
     
        return states_now, states_similar, hidden_now, hidden_similar
    
  
  
 
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
    