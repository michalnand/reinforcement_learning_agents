import torch

class TrajectoryBufferLP:   

    def __init__(self, buffer_size, state_shape, prompt_size, actions_count, envs_count):
        self.buffer_size    = buffer_size
        self.state_shape    = state_shape
        self.prompt_size    = prompt_size
        self.actions_count  = actions_count
        self.envs_count     = envs_count
      
        self.clear()    


    def add(self, states, prompts, task_ids, logits, values, prompt_mean, prompt_var, actions, rewards, dones):
        self.states[self.ptr]    = states.clone() 
        self.prompts[self.ptr]   = prompts.clone() 
        self.task_ids[self.ptr]  = task_ids.clone() 

        self.logits[self.ptr]      = logits.clone()
        self.values[self.ptr]      = values.clone()
        self.prompt_mean[self.ptr] = prompt_mean.clone()
        self.prompt_var[self.ptr]  = prompt_var.clone()

        self.actions[self.ptr]     = actions.clone()
        
        self.reward[self.ptr]    = rewards.clone()
        self.dones[self.ptr]     = dones.float().clone()
        
        self.ptr = self.ptr + 1 

    def is_full(self):
        if self.ptr >= self.buffer_size:
            return True

        return False 
 
    def clear(self):
        self.states     = torch.zeros((self.buffer_size, self.envs_count, ) + self.state_shape, dtype=torch.float32)
        self.prompts    = torch.zeros((self.buffer_size, self.envs_count, self.prompt_size), dtype=torch.float32)
        self.task_ids   = torch.zeros((self.buffer_size, self.envs_count, ), dtype=int)

        self.logits     = torch.zeros((self.buffer_size, self.envs_count, self.actions_count), dtype=torch.float32)
        self.values     = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)        
     
        self.prompt_mean= torch.zeros((self.buffer_size, self.envs_count, self.prompt_size), dtype=torch.float32)
        self.prompt_var = torch.zeros((self.buffer_size, self.envs_count, self.prompt_size), dtype=torch.float32)

        self.actions    = torch.zeros((self.buffer_size, self.envs_count, ), dtype=int)
        
        self.reward     = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)
        self.dones      = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)

        self.ptr = 0  
 

    def compute_returns(self, gamma, lam = 0.95):
        self.returns, self.advantages   = self._gae(self.reward, self.values, self.dones, gamma, lam)
        
        #reshape buffer for faster batch sampling
        self.states     = self.states.reshape((self.buffer_size*self.envs_count, ) + self.state_shape)
        self.prompts    = self.prompts.reshape((self.buffer_size*self.envs_count, self.prompt_size))
        self.task_ids   = self.task_ids.reshape((self.buffer_size*self.envs_count, ), dtype=int)

        self.logits     = self.logits.reshape((self.buffer_size*self.envs_count, self.actions_size))
        self.values     = self.values.reshape((self.buffer_size*self.envs_count, ))        

        self.prompt_mean= self.prompt_mean.reshape((self.buffer_size*self.envs_count, self.prompt_size))
        self.prompt_var = self.prompt_car.reshape((self.buffer_size*self.envs_count, self.prompt_size))
        
        self.actions    = self.actions.reshape((self.buffer_size*self.envs_count, ))
        
        self.reward     = self.reward.reshape((self.buffer_size*self.envs_count, ))
      
        self.dones      = self.dones.reshape((self.buffer_size*self.envs_count, ))

        self.returns    = self.returns.reshape((self.buffer_size*self.envs_count, ))
        self.advantages = self.advantages.reshape((self.buffer_size*self.envs_count, ))


    def sample_batch(self, batch_size, device):
        indices         = torch.randint(0, self.envs_count*self.buffer_size, size=(batch_size, ))

        states          = torch.index_select(self.states, dim=0, index=indices).to(device)
        prompts         = torch.index_select(self.prompts, dim=0, index=indices).to(device)
        task_ids        = torch.index_select(self.task_ids, dim=0, index=indices).to(device)

        logits          = torch.index_select(self.logits, dim=0, index=indices).to(device)

        prompt_mean     = torch.index_select(self.prompt_mean, dim=0, index=indices).to(device)
        prompt_var      = torch.index_select(self.prompt_var, dim=0, index=indices).to(device)
        
        actions         = torch.index_select(self.actions, dim=0, index=indices).to(device)
        
        returns         = torch.index_select(self.returns, dim=0, index=indices).to(device)
        advantages      = torch.index_select(self.advantages, dim=0, index=indices).to(device)

       
        return states, prompts, task_ids, logits, prompt_mean, prompt_var, actions, returns, advantages
    
    
    def sample_states(self, batch_size, device):
        indices = torch.randint(0, self.envs_count*self.buffer_size, size=(batch_size, ))

        states   = torch.index_select(self.states, dim=0, index=indices).to(device)
        prompts  = torch.index_select(self.prompts, dim=0, index=indices).to(device)
        task_ids = torch.index_select(self.task_ids, dim=0, index=indices).to(device)

        return states, prompts, task_ids
   
     
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
