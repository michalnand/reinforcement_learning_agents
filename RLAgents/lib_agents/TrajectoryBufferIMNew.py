import torch

class TrajectoryBufferIMNew:

    def __init__(self, buffer_size, state_shape, actions_size, envs_count):
        self.buffer_size    = buffer_size
        self.state_shape    = state_shape
        self.actions_size   = actions_size 
        self.envs_count     = envs_count

        self.hidden_states = None
        
        self.clear()     
 
    def add(self, state, logits, value_ext, value_int, action, reward_ext, reward_int, done, hidden_state = None, steps = None):
        
        self.states[self.ptr]    = state.clone() 
        self.logits[self.ptr]    = logits.clone()
        
        self.values_ext[self.ptr]= value_ext.clone()
        self.values_int[self.ptr]= value_int.clone()

        self.actions[self.ptr]   = action.clone()
        
        self.reward_ext[self.ptr]  = reward_ext.clone()
        self.reward_int[self.ptr]  = reward_int.clone()

        self.dones[self.ptr]       = (1.0*done).clone()

        if hidden_state is not None:
            
            if self.hidden_states is None:
                self.hidden_shape   = hidden_state.shape[1:]    
                self.hidden_states   = torch.zeros((self.buffer_size, self.envs_count, ) + self.hidden_shape, dtype=torch.float32)
            
            self.hidden_states[self.ptr] = hidden_state.clone()

        if steps is not None:
            if self.steps is None:
                self.steps = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)
            
            self.steps[self.ptr] = steps

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

        self.hidden_states  = None
        self.steps          = None

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

        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.reshape((self.buffer_size*self.envs_count, ) + self.hidden_shape)

        if self.steps is not None:
            self.steps = self.steps.reshape((self.buffer_size*self.envs_count, ))

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


       
        return states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int
    

       
 
    def sample_batch_seq(self, seq_length, batch_size, device):
        indices        = torch.randint(0, self.envs_count*(self.buffer_size - seq_length), size=(batch_size, ))
        
        states         = torch.zeros((seq_length, batch_size, ) + self.state_shape,  dtype=torch.float32, device=device)
        hidden_states  = torch.zeros((seq_length, batch_size, ) + self.hidden_shape, dtype=torch.float32, device=device)

        for n in range(seq_length):
            states[n]        = self.states[indices].to(device)
            hidden_states[n] = self.hidden_states[indices].to(device)

            if n == (seq_length-1): 
                logits     = self.logits[indices].to(device)
                
                actions    = self.actions[indices].to(device)
                
                returns_ext    = self.returns_ext[indices].to(device)
                returns_int    = self.returns_int[indices].to(device)
                advantages_ext = self.advantages_ext[indices].to(device)
                advantages_int = self.advantages_int[indices].to(device)

            indices+= self.envs_count 

        return states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int, hidden_states
    

   
    def sample_states_pairs(self, batch_size, max_distance, stochastic_distance, device):
        count = self.buffer_size*self.envs_count

        if stochastic_distance:
            max_distance_   = torch.randint(0, 1 + max_distance, (batch_size, ))
        else:
            max_distance_   = max_distance

        indices_now  = torch.randint(0, count, size=(batch_size, ))
        indices_prev = torch.clip(indices_now - max_distance_*self.envs_count, 0, count-1)
      
        states_now   = (self.states[indices_now]).to(device)
        states_prev  = (self.states[indices_prev]).to(device)

        return states_now, states_prev
    

    def sample_states_distance_pairs(self, batch_size, max_distance, device):
        count = self.buffer_size*self.envs_count

        distance    = torch.randint(0, 1 + max_distance, (batch_size, ))

        indices_a   = torch.randint(0, count, size=(batch_size, ))
        indices_b   = torch.clip(indices_a + distance*self.envs_count, 0, count-1)
      
        xa   = (self.states[indices_a]).to(device)  
        xb   = (self.states[indices_b]).to(device)

        return xa, xb, distance.to(device)
    

    def sample_states_pairs_seq(self, seq_length, batch_size, max_distance, stochastic_distance, device):
        count = self.envs_count*(self.buffer_size - seq_length)

        if stochastic_distance: 
            max_distance_   = torch.randint(0, 1 + max_distance, (batch_size, ))
        else:
            max_distance_   = max_distance

        indices_now  = torch.randint(0, count, size=(batch_size, ))
        indices_prev = torch.clip(indices_now - max_distance_*self.envs_count, 0, count-1)

        states_now         = torch.zeros((seq_length, batch_size, ) + self.state_shape,  dtype=torch.float32, device=device)
        states_prev        = torch.zeros((seq_length, batch_size, ) + self.state_shape,  dtype=torch.float32, device=device)
        hidden_states_now  = torch.zeros((seq_length, batch_size, ) + self.hidden_shape, dtype=torch.float32, device=device)
        hidden_states_prev = torch.zeros((seq_length, batch_size, ) + self.hidden_shape, dtype=torch.float32, device=device)


        for n in range(seq_length):
            states_now[n]  = self.states[indices_now].to(device)
            states_prev[n] = self.states[indices_prev].to(device)

            hidden_states_now[n]  = self.hidden_states[indices_now].to(device)
            hidden_states_prev[n] = self.hidden_states[indices_prev].to(device)
          
            indices_now+= self.envs_count 
            indices_prev+= self.envs_count 

        return states_now, states_prev, hidden_states_now, hidden_states_prev
    
    
    def sample_states_steps(self, batch_size, device):
        count = self.buffer_size*self.envs_count

        indices  = torch.randint(0, count, size=(batch_size, ))
      
        states   = (self.states[indices]).to(device)
        steps    = (self.steps[indices]).to(device)

        return states, steps
    

    
    def sample_states_seq(self, seq_length, batch_size, device):
        count    = self.envs_count*(self.buffer_size - seq_length)

        indices  = torch.randint(0, count, size=(batch_size, ))

        states   = torch.zeros((seq_length, batch_size, ) + self.state_shape,  dtype=torch.float32, device=device)
      
        for n in range(seq_length):
            states[n] = self.states[indices]
            indices+= self.envs_count

        return states


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
    