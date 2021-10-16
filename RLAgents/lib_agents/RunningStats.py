import numpy


'''
class RunningStats:
    def __init__(self, shape = (), initial_value = None):
        self.mean  = numpy.zeros(shape)
        self.mean2 = numpy.zeros(shape)

        self.std   = numpy.ones(shape) 

        if initial_value is not None:
            mean = initial_value.mean(axis=0)
            self.mean = numpy.float32(mean)
    
    def update(self, x, alpha = 0.0001):      
        mean        = x.mean(axis=0)
        mean2       = (x**2).mean(axis=0)

        self.mean   = (1.0 - alpha)*self.mean  + alpha*mean
        self.mean2  = (1.0 - alpha)*self.mean2 + alpha*mean2

        self.std    = (mean2 - (mean**2))**0.5
'''

class RunningStats:
    def __init__(self, shape, initial_value = None):
        self.n     = 0

        self.mean  = numpy.zeros(shape)
        self.var   = numpy.zeros(shape)

        if initial_value is not None:
            self.mean = initial_value.mean(axis=0)
            self.var  = initial_value.var(axis=0)

    def update(self, x):   

        x_ = x.mean(axis=0)

        self.n+= x.shape[0] 

        mean = self.mean + (x_ - self.mean)/self.n
        var  = self.var  + ((x_ - self.mean)**2)

        self.mean = mean
        self.var  = var

        self.std  = ((self.var/self.n)**0.5) + 0.000001 
