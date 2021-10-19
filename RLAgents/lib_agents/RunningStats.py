import numpy


'''
class RunningStats:
    def __init__(self, shape = (), initial_value = None):
        self.mean  = numpy.zeros(shape)
        self.std   = numpy.ones(shape) 

        if initial_value is not None:
            self.mean   = initial_value.mean(axis=0)
            self.std    = initial_value.std(axis=0)
    
    def update(self, x, alpha = 0.01):    
        mean        = x.mean(axis=0)
        std         = x.std(axis=0)  

        self.mean   = (1.0 - alpha)*self.mean   + alpha*mean
        self.std    = (1.0 - alpha)*self.std    + alpha*std
'''


class RunningStats:
    def __init__(self, shape):
        self.count = 1
        self.mean  = numpy.zeros(shape)
        self.var   = numpy.ones(shape)

        self.mean   = self.mean.astype(numpy.float64)
        self.var    = self.var.astype(numpy.float64)

        self.std    = (self.var**0.5) + 0.0000001 

    def update_initial(self, x, alpha = 0.1):
        mean        = x.mean(axis=0)
        var         = x.var(axis=0)  

        self.mean   = (1.0 - alpha)*self.mean   + alpha*mean
        self.var    = (1.0 - alpha)*self.var    + alpha*var

    def update(self, x): 

        self.count+= 1

        mean = self.mean + (x.mean(axis=0) - self.mean)/self.count
        var  = self.var  + ((x - self.mean)*(x - mean)).mean(axis=0)

        self.mean = mean
        self.var  = var

        self.std  = ((self.var/self.count)**0.5) + 0.0000001 
