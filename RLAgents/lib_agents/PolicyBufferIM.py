import torch

class PolicyBufferIM:

    def __init__(self, buffer_size, state_shape, actions_size, envs_count):
        self.buffer_size    = buffer_size
        self.state_shape    = state_shape
        self.actions_size   = actions_size
        self.envs_count     = envs_count
      
        self.clear()     
 
    def add(self, state, logits, value_ext, value_int, action, reward_ext, reward_int, done, hidden_state = None, exploration_mode = None, episode_steps = None):
        
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
                self.hidden_state   = torch.zeros((self.buffer_size, self.envs_count, hidden_state.shape[1]), dtype=torch.float32)
            
            self.hidden_state[self.ptr] = hidden_state.clone()

        if exploration_mode is not None:
            self.exploration_mode[self.ptr] = exploration_mode.clone()

        if episode_steps is not None:
            self.episode_steps[self.ptr] = episode_steps.clone()
        
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
        #clear external rewards if agent in exploration mode
        #this alows agent to skip greedy rewards
        self.reward_ext*= (1.0 - self.exploration_mode)

        self.returns_ext, self.advantages_ext = self._gae(self.reward_ext, self.values_ext, self.dones, gamma_ext, lam)
        self.returns_int, self.advantages_int = self._gae(self.reward_int, self.values_int, self.dones, gamma_int, lam)

        self.relations  = self._relations(self.reward_ext, self.dones)
        
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

        self.relations        = self.relations.reshape((self.buffer_size*self.envs_count, ))
        self.episode_steps    = self.episode_steps.reshape((self.buffer_size*self.envs_count, ))



    def sample_batch(self, batch_size, device = "cpu"):
        indices         = torch.randint(0, self.envs_count*self.buffer_size, size=(batch_size*self.envs_count, ))

       
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
    

    def sample_states_action_pairs(self, batch_size, device = "cpu", max_distance = 0):
        count           = self.buffer_size*self.envs_count

        max_distance_   = torch.randint(0, 1 + max_distance, (batch_size, ))

        indices         = torch.randint(0, count, size=(batch_size, ))
        indices_next    = torch.clip(indices + self.envs_count, 0, count-1)
        indices_similar = torch.clip(indices + max_distance_*self.envs_count, 0, count-1)
        indices_random  = torch.randint(0, count, size=(batch_size, )) 
      
        states_now      = (self.states[indices]).to(device)
        states_next     = (self.states[indices_next]).to(device)
        states_similar  = (self.states[indices_similar]).to(device)
        states_random   = (self.states[indices_random]).to(device)
        
        actions         = (self.actions[indices]).to(device)

        relations       = (self.relations[indices]).to(device)
        episode_steps   = (self.episode_steps[indices]).to(device)
 
     
        return states_now, states_next, states_similar, states_random, actions, relations, episode_steps
    
   
 
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
    
    def _relations(self, rewards, dones):
        buffer_size = rewards.shape[0]
        envs_count  = rewards.shape[1]
        
        relations = torch.zeros((buffer_size, envs_count), dtype=int)

        r = torch.zeros((envs_count), dtype=int)
        for n in reversed(range(buffer_size-1)):
            negative_idx   = torch.where(dones[n] > 0)[0]
            positive_idx   = torch.where(rewards[n] > 0)[0]

            r[negative_idx] = -1

            r[positive_idx] =  1

            relations[n] = r

        return relations
