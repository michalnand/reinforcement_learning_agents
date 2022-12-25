import numpy
from numpy.core.fromnumeric import nonzero


class RunningStats:
    def __init__(self, shape, initial_value=None):
        self.count = 1
        self.eps   = 0.0000001
        self.mean  = numpy.zeros(shape)
        self.var   = numpy.ones(shape)

        if initial_value is not None:
            self.mean           = initial_value.mean(axis=0)
            self.std            = initial_value.std(axis=0)

        self.mean   = self.mean.astype(numpy.float32)
        self.var    = self.var.astype(numpy.float32)
        self.std    = (self.var**0.5) + self.eps

 
    def update(self, x): 
        
        self.count+= 1

        mean = self.mean + (x.mean(axis=0) - self.mean)/self.count
        var  = self.var  + ((x - self.mean)*(x - mean)).mean(axis=0)

        self.mean = mean
        self.var  = var

        self.std  = ((self.var/self.count)**0.5) + self.eps

    


class StateMomentum:

    def __init__(self, batch_size, shape, gammas = [0.99, 0.998]):

        self.result_shape = (shape[0] + len(gammas), shape[1], shape[2])
        self.gammas       = gammas

        self.momentum    = numpy.zeros((batch_size, len(gammas), shape[1], shape[2]), dtype=numpy.float32)

    def update(self, states): 
        for i in range(len(self.gammas)):
            g = self.gammas[i]
            self.momentum[:,i] = g*self.momentum[:,i] + (1.0 - g)*states[:,i]

        return numpy.concatenate([states, self.momentum], axis=1)

    def get(self, env_id, state):
        return numpy.concatenate([state, self.momentum[env_id]], axis=0)

    def reset(self, env_id):
        self.momentum[env_id] = 0.0