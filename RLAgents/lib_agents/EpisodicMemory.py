import torch
import numpy

'''
class EpisodicMemory:
    def __init__(self, size, initial_count = 8):
        self.size               = size
        self.initial_count      = initial_count
        self.episodic_memory    = None 
 
    def reset(self, state_t):  
        self.episodic_memory = torch.zeros((self.size , ) + state_t.shape).to(state_t.device)
        for i in range(self.size):
            self.episodic_memory[i] = state_t
        
        self.count = 0

        
    def entropy(self, state_t):  
        self._add(state_t)

        mean = self.episodic_memory.mean(axis=0)
        std  = self.episodic_memory.std(axis=0) 
        
        arg  = (state_t - mean)/(std + 10**-7)
        res  = 1.0 - torch.exp(-0.5*(arg**2)) 
        res  = res.mean() 
 
        result = res.detach().to("cpu").numpy()

        if self.count < self.initial_count:
            return 0.0
        else:
            return result


    def _add(self, state_t):
        if self.episodic_memory is None:
            self.reset(state_t)
        else:
            if self.count < self.initial_count: 
                n = self.size//self.initial_count 
                for i in range(n):
                    idx = numpy.random.randint(self.size)
                    self.episodic_memory[idx] = state_t

                self.count+= 1
            else:
                idx = numpy.random.randint(self.size)
                self.episodic_memory[idx] = state_t
'''

class EpisodicMemory:
    def __init__(self, size, initial_count = 8):
        self.size               = size
        self.initial_count      = initial_count
        self.episodic_memory    = None 

        self.mean = 0.0
        self.std  = 0.0

        self.count = 0
 
    def reset(self, state_t):  
        self.episodic_memory = torch.zeros((self.size , ) + state_t.shape).to(state_t.device)
        for i in range(self.size):
            self.episodic_memory[i] = state_t
        
        self.count = 0

        
    def entropy(self, state_t):  
        arg  = (state_t - self.mean)/(self.std + 10**-7)
        res  = 1.0 - torch.exp(-0.5*(arg**2)) 
        res  = res.mean() 
 
        result = res.detach().to("cpu").numpy()

        if self.count < self.initial_count:
            return 0.0
        else:
            return result


    def add(self, state_t):
        if self.episodic_memory is None:
            self.reset(state_t)
        
        if self.count < self.initial_count: 
            n = self.size//self.initial_count 
            for i in range(n):
                idx = numpy.random.randint(self.size)
                self.episodic_memory[idx] = state_t

            self.count+= 1
        else:
            idx = numpy.random.randint(self.size)
            self.episodic_memory[idx] = state_t

        self.mean = self.episodic_memory.mean(axis=0)
        self.std  = self.episodic_memory.std(axis=0) 
        