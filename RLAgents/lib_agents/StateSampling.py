import numpy

class StateSampling:
    #input shape    : (batch, features)
    def __init__(self, initial_states, taps):
        self.taps       = taps
        self.taps_count = len(self.taps)
        self.idx        = 0

        self.buffer     = numpy.zeros((self.taps_count, ) + initial_states.shape, dtype=numpy.float32)
        
        for t in range(self.taps_count):
            self.buffer[t] = initial_states.copy()
        
    #input shape    : (batch, features)
    #output returns : (batch, taps, features)
    def add(self, states):
        for t in range(self.taps_count):
            if self.idx%self.taps[t] == 0:
                self.buffer[t] = states.copy()

        self.idx+= 1 

        return self.get()

    #input shape    : (features)
    def reset(self, state, batch_idx):
        for t in range(self.taps_count):
            self.buffer[t][batch_idx] = state[t].copy()

    def get(self):
        result = numpy.swapaxes(self.buffer,0, 1)
        return result