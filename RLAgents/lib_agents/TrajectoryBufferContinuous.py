import torch

class TrajectoryBufferContinuous:

    def __init__(self, buffer_size, state_shape, actions_size, envs_count, device):
        
        self.buffer_size    = buffer_size
        self.state_shape    = state_shape
        self.actions_size   = actions_size
        self.envs_count     = envs_count
        self.device         = device

        self.hidden_state = None

        self.clear() 

    def add(self, state, value, action, action_mu, action_var, reward, done, hidden_state = None):
        if self.ptr >= self.buffer_size:
            return False
        
        self.states[self.ptr]        = state.clone()
        self.values[self.ptr]        = value.clone()
        self.actions[self.ptr]       = action.clone()
        self.actions_mu[self.ptr]    = action_mu.clone()
        self.actions_var[self.ptr]   = action_var.clone()
        self.rewards[self.ptr]       = reward.clone()

        self.dones[self.ptr]         = (1.0*done).clone()

        # hidden state for rnn policy
        if hidden_state is not None:
            if self.hidden_state is None:
                self.hidden_shape   = hidden_state.shape[1:]
                self.hidden_state   = torch.zeros((self.buffer_size, self.envs_count, ) + self.hidden_shape, dtype=torch.float32)
            
            self.hidden_state[self.ptr] = hidden_state.clone()
        
        
        self.ptr = self.ptr + 1 

        return True
        

    def is_full(self):
        if self.ptr >= self.buffer_size:
            return True

        return False 
 
    def clear(self):
        self.states           = torch.zeros((self.buffer_size, self.envs_count, ) + self.state_shape, dtype=torch.float32)        
        self.values           = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)

        self.actions          = torch.zeros((self.buffer_size, self.envs_count, self.actions_size), dtype=torch.float32)
        self.actions_mu       = torch.zeros((self.buffer_size, self.envs_count, self.actions_size), dtype=torch.float32)
        self.actions_var      = torch.zeros((self.buffer_size, self.envs_count, self.actions_size), dtype=torch.float32)

        self.rewards          = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)
        self.dones            = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)
       
        self.hidden_state = None
        self.ptr = 0 


    def compute_returns(self, gamma = 0.99, lam = 0.95):
        self.returns, self.advantages = self._gae(self.rewards, self.values, self.dones, gamma, lam)
        
        #reshape buffer for faster batch sampling
        self.states           = self.states.reshape((self.buffer_size*self.envs_count, ) + self.state_shape)
        self.values           = self.values.reshape((self.buffer_size*self.envs_count, ))

        self.actions          = self.actions.reshape((self.buffer_size*self.envs_count, self.actions_size))
        self.actions_mu       = self.actions_mu.reshape((self.buffer_size*self.envs_count, self.actions_size))
        self.actions_var      = self.actions_var.reshape((self.buffer_size*self.envs_count, self.actions_size))

        self.rewards          = self.rewards.reshape((self.buffer_size*self.envs_count, ))
        self.dones            = self.dones.reshape((self.buffer_size*self.envs_count, ))

        self.returns          = self.returns.reshape((self.buffer_size*self.envs_count, ))
        self.advantages       = self.advantages.reshape((self.buffer_size*self.envs_count, ))

        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.reshape((self.buffer_size*self.envs_count, ) + self.hidden_shape)


    def sample_batch(self, batch_size, device):     
        indices     = torch.randint(0, self.envs_count*self.buffer_size, size=(batch_size, ))
                        
        states      =  self.states[indices].to(device) 
        
        values      =  self.values[indices].to(device)
        
        actions      =  self.actions[indices].to(device)
        actions_mu   =  self.actions_mu[indices].to(device)
        actions_var  =  self.actions_var[indices].to(device)

        rewards     =  self.rewards[indices].to(device)
        dones       =  self.dones[indices].to(device)

        returns     =  self.returns[indices].to(device)
        advantages  =  self.advantages[indices].to(device)
 
        return states, values, actions, actions_mu, actions_var, rewards, dones, returns, advantages

    def sample_batch_seq(self, seq_length, batch_size, device):     
        indices        = torch.randint(0, self.envs_count*(self.buffer_size - seq_length), size=(batch_size, ))

        states         = torch.zeros((seq_length, batch_size, ) + self.state_shape,  dtype=torch.float32, device=device)
        hidden_states  = torch.zeros((seq_length, batch_size, ) + self.hidden_shape, dtype=torch.float32, device=device)

        for n in range(seq_length):
            states[n]        = self.states[indices].to(device)
            hidden_states[n] = self.hidden_state[indices].to(device)

            if n == (seq_length-1): 
                values      =  self.values[indices].to(device)  
                
                actions      =  self.actions[indices].to(device)
                actions_mu   =  self.actions_mu[indices].to(device)
                actions_var  =  self.actions_var[indices].to(device)

                rewards     =  self.rewards[indices].to(device)
                dones       =  self.dones[indices].to(device)

                returns     =  self.returns[indices].to(device)
                advantages  =  self.advantages[indices].to(device)

            indices+= self.envs_count   

        return states, values, actions, actions_mu, actions_var, rewards, dones, returns, advantages, hidden_states



    def sample_trajectory(self, batch_size, length, device):

        result_states = torch.zeros((length, batch_size) + self.state_shape, dtype=torch.float32, device=device)
        result_actions = torch.zeros((length, batch_size, self.actions_size), dtype=torch.float32, device=device)
        result_rewards = torch.zeros((length, batch_size), dtype=torch.float32, device=device)

        indices     = torch.randint(0, self.envs_count*(self.buffer_size - length), size=(batch_size, ))

        for n in range(length):
            indices_ = indices + n*self.envs_count
            result_states[n]  = torch.index_select(self.states,  dim=0, index=indices_).to(device)
            result_actions[n] = torch.index_select(self.actions, dim=0, index=indices_).to(device)
            result_rewards[n] = torch.index_select(self.rewards, dim=0, index=indices_).to(device)

        return result_states, result_actions, result_rewards


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