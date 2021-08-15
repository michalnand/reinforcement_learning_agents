import numpy

class RunningStats:
    def __init__(self, shape = (), initial_value = None):
        self.mean  = numpy.zeros(shape)
        self.std   = numpy.ones(shape) 

        if initial_value is not None:
            mean = initial_value.mean(axis=0)
            self.mean = numpy.float32(mean)
    
    def update(self, x, alpha = 0.0001):      
        mean        = x.mean(axis=0)
        self.mean   = (1.0 - alpha)*self.mean + alpha*mean

        std         = x.std(axis=0) + 0.001
        self.std    = (1.0 - alpha)*self.std + alpha*std
   