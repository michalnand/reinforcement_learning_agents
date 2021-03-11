import torch
import numpy

class EpisodicMemory:
    def __init__(self, size, initial_count = 16):
        self.size               = size
        self.initial_count      = initial_count
        self.episodic_memory    = None 
 
    def reset(self, state_t): 
        self.episodic_memory = torch.zeros((self.size , ) + state_t.shape).to(state_t.device)
        for i in range(self.size):
            self.episodic_memory[i] = state_t
        self.count = 0

    def add(self, state_t):
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
    def entropy(self):
        mean = self.episodic_memory.mean(axis=0)
        diff = (self.episodic_memory - mean)**2
        max_ = diff.max(axis=0)[0] 
 
        result = max_.mean().detach().to("cpu").numpy()

        if self.count < self.initial_count:
            return 0.0
        else:
            return result
    '''

    def entropy(self):
        mean = self.episodic_memory.mean(axis=0)
        diff = (self.episodic_memory - mean)**2
 
        result = diff.mean().detach().to("cpu").numpy()

        if self.count < self.initial_count:
            return 0.0
        else:
            return result