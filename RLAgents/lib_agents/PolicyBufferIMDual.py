import torch
import numpy 


class PolicyBufferPartial:

    def __init__(self, buffer_size, actions_size, envs_count, device):
        
        self.buffer_size    = buffer_size
        self.actions_size   = actions_size
        self.envs_count     = envs_count
        self.device         = device
      
        self.clear()   
 
    def add(self, logits, value_ext, value_int, action, reward_ext, reward_int, done):
        self.logits[self.ptr]    = logits.copy()
        
        self.values_ext[self.ptr]= value_ext.copy()
        self.values_int[self.ptr]= value_int.copy()

        self.actions[self.ptr]   = action.copy()
        
        self.reward_ext[self.ptr]  = reward_ext.copy()
        self.reward_int[self.ptr]  = reward_int.copy()

        self.dones[self.ptr]       = (1.0*done).copy()
        
        self.ptr = self.ptr + 1 


    def is_full(self):
        if self.ptr >= self.buffer_size:
            return True

        return False 
 
    def clear(self):
        self.logits         = numpy.zeros((self.buffer_size, self.envs_count, self.actions_size), dtype=numpy.float32)

        self.values_ext     = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)        
        self.values_int     = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)

        self.actions        = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=int)
        
        self.reward_ext     = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)
        self.reward_int     = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)

        self.dones          = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)

        self.ptr = 0  
 

    def compute_returns(self, mask, gamma_ext = 0.99, gamma_int = 0.9, lam = 0.95):
        self.returns_ext, self.advantages_ext = self._gae_fast(mask, self.reward_ext, self.values_ext, self.dones, gamma_ext, lam)
        self.returns_int, self.advantages_int = self._gae_fast(mask, self.reward_int, self.values_int, self.dones, gamma_int, lam)
        
        #reshape buffer for faster batch sampling
        self.logits           = self.logits.reshape((self.buffer_size*self.envs_count, self.actions_size))

        self.values_ext       = self.values_ext.reshape((self.buffer_size*self.envs_count, ))        
        self.values_int       = self.values_int.reshape((self.buffer_size*self.envs_count, ))

        self.actions          = self.actions.reshape((self.buffer_size*self.envs_count, ))
        
        self.reward_ext       = self.reward_ext.reshape((self.buffer_size*self.envs_count, ))
        self.reward_int       = self.reward_int.reshape((self.buffer_size*self.envs_count, ))

        self.dones            = self.dones.reshape((self.buffer_size*self.envs_count, ))

        self.returns_ext      = self.returns_ext.reshape((self.buffer_size*self.envs_count, ))
        self.advantages_ext   = self.advantages_ext.reshape((self.buffer_size*self.envs_count, ))

        self.returns_int      = self.returns_int.reshape((self.buffer_size*self.envs_count, ))
        self.advantages_int   = self.advantages_int.reshape((self.buffer_size*self.envs_count, ))


    def sample_batch(self, indices): 
        logits          = torch.from_numpy(numpy.take(self.logits, indices, axis=0)).to(self.device)
        
        actions         = torch.from_numpy(numpy.take(self.actions, indices, axis=0)).to(self.device)
        
        returns_ext     = torch.from_numpy(numpy.take(self.returns_ext, indices, axis=0)).to(self.device)
        returns_int     = torch.from_numpy(numpy.take(self.returns_int, indices, axis=0)).to(self.device)

        advantages_ext  = torch.from_numpy(numpy.take(self.advantages_ext, indices, axis=0)).to(self.device)
        advantages_int  = torch.from_numpy(numpy.take(self.advantages_int, indices, axis=0)).to(self.device)


        return logits, actions, returns_ext, returns_int, advantages_ext, advantages_int 
    
    def _gae_fast(self, mask, rewards, values, dones, gamma = 0.99, lam = 0.9):
        buffer_size = rewards.shape[0]
        envs_count  = rewards.shape[1]
        
        returns     = numpy.zeros((buffer_size, envs_count), dtype=numpy.float32)
        advantages  = numpy.zeros((buffer_size, envs_count), dtype=numpy.float32)

        last_gae    = numpy.zeros((envs_count), dtype=numpy.float32)

        rewards_    = rewards*mask
        values_     = values*mask
        
        for n in reversed(range(buffer_size-1)):
            delta           = rewards_[n] + gamma*values_[n+1]*(1.0 - dones[n]) - values_[n]
            last_gae        = delta + gamma*lam*last_gae*(1.0 - dones[n])
            
            returns[n]      = last_gae + values_[n]
            advantages[n]   = last_gae

        return returns, advantages




class PolicyBufferIMDual:

    def __init__(self, buffer_size, state_shape, goal_shape, actions_size, envs_count, device, uint8_storage = False):
        self.buffer_size    = buffer_size
        self.state_shape    = state_shape
        self.goal_shape     = goal_shape
        self.envs_count     = envs_count
        self.device         = device

        self.buffer_a = PolicyBufferPartial(buffer_size, actions_size, envs_count, device)
        self.buffer_b = PolicyBufferPartial(buffer_size, actions_size, envs_count, device)

        self.uint8_storage  = uint8_storage

        if self.uint8_storage:
            self.scale  = 255
        else:
            self.scale  = 1
      
        self.clear()   


    def add(self, state, goal, mode):
        self.states[self.ptr]    = state.copy()*self.scale
        self.goals[self.ptr]     = goal.copy()*self.scale
        self.modes[self.ptr]     = mode.copy()

        self.ptr = self.ptr + 1 

    def add_a(self, logits_a, value_ext_a, value_int_a, action_a, reward_ext_a, reward_int_a, done_a):
        self.buffer_a.add(logits_a, value_ext_a, value_int_a, action_a, reward_ext_a, reward_int_a, done_a)

    def add_b(self, logits_b, value_ext_b, value_int_b, action_b, reward_ext_b, reward_int_b, done_b):
        self.buffer_b.add(logits_b, value_ext_b, value_int_b, action_b, reward_ext_b, reward_int_b, done_b)

    def clear(self):
        if self.uint8_storage: 
            self.states     = numpy.zeros((self.buffer_size, self.envs_count, ) + self.state_shape, dtype=numpy.ubyte)
        else:
            self.states     = numpy.zeros((self.buffer_size, self.envs_count, ) + self.state_shape, dtype=numpy.float32)

        if self.uint8_storage: 
            self.goals     = numpy.zeros((self.buffer_size, self.envs_count, ) + self.goal_shape, dtype=numpy.ubyte)
        else:
            self.goals     = numpy.zeros((self.buffer_size, self.envs_count, ) + self.goal_shape, dtype=numpy.float32)

        self.modes          = numpy.zeros((self.buffer_size, self.envs_count, ), dtype=numpy.float32)

        self.buffer_a.clear()
        self.buffer_b.clear()

        self.ptr = 0

    def is_full(self):
        if self.ptr >= self.buffer_size:
            return True

        return False 

    def compute_returns(self, gamma_ext = 0.99, gamma_int = 0.9, lam = 0.95):
        self.buffer_a.compute_returns(1.0 - self.modes, gamma_ext, gamma_int, lam)
        self.buffer_b.compute_returns(self.modes, gamma_ext, gamma_int, lam)

        #reshape buffer for faster batch sampling
        self.states  = self.states.reshape((self.buffer_size*self.envs_count, ) + self.state_shape)
        self.goals   = self.goals.reshape((self.buffer_size*self.envs_count, ) + self.goal_shape)
        self.modes   = self.modes.reshape((self.buffer_size*self.envs_count, ))

    def sample_batch(self, batch_size, device):

        indices = numpy.random.randint(0, self.envs_count*self.buffer_size, size=batch_size*self.envs_count)

 
        states  = torch.from_numpy(numpy.take(self.states, indices, axis=0)).to(device).float()/self.scale
        goals   = torch.from_numpy(numpy.take(self.goals, indices, axis=0)).to(device).float()/self.scale
        modes   = torch.from_numpy(numpy.take(self.modes, indices, axis=0)).to(device).float()

        res_a   = self.buffer_a.sample_batch(indices)
        res_b   = self.buffer_b.sample_batch(indices)


        return states, goals, modes, res_a, res_b