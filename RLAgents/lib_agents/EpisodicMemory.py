import torch
import numpy

'''
class EpisodicMemory:
    def __init__(self, size):
        self.size               = size
        self.episodic_memory    = None 

        self.mean   = 0.0
        self.std    = 0.0 

    def reset(self, state_t):
        self.episodic_memory = torch.zeros((self.size, ) + state_t.shape).to(state_t.device)
        for i in range(self.size):
            self.episodic_memory[i] = state_t

    def add(self, state_t):
        if self.episodic_memory is None:
            self.reset(state_t)

        idx = numpy.random.randint(self.size)
        self.episodic_memory[idx] = state_t

        self.mean = self.episodic_memory.mean(axis=0)
        self.std  = self.episodic_memory.std(axis=0) 
       
    def entropy(self, state_t):  
        arg  = (state_t - self.mean)/(self.std + 10**-7)
        res  = 1.0 - torch.exp(-0.5*(arg**2)) 
        res  = res.mean() 
 
        result = res.detach().to("cpu").numpy()

        return result
'''

class EpisodicMemory:
    def __init__(self, size):
        self.size   = size
        self.mean   = 0.0
        self.std    = 0.0 

        self.episodic_memory = None
        
    def reset(self, state_t):
        self.episodic_memory = torch.zeros((self.size, ) + state_t.shape).to(state_t.device)
        for i in range(self.size): 
            self.episodic_memory[i] = state_t.clone()

    def add(self, state_t):
        if self.episodic_memory is None:
            self.reset(state_t)

        idx = numpy.random.randint(self.size)
        self.episodic_memory[idx] = state_t.clone()

        self.mean = self.episodic_memory.mean(axis=0)
        self.std  = self.episodic_memory.std(axis=0) 
       
    def motivation(self, state_t):  
        arg  = (state_t - self.mean)/(self.std + 10**-7)
        res  = 1.0 - torch.exp(-0.5*(arg**2)) 
        res  = res.mean() 
 
        result = res.detach().to("cpu").numpy()

        return result
