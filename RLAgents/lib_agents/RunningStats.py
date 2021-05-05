import numpy

class RunningStats:
    def __init__(self, shape = (), initial_value = None):

        self.mean  = numpy.zeros(shape)
        self._std  = numpy.ones(shape)
        self.std  = numpy.ones(shape)

        if initial_value is not None:
            mean = initial_value.mean(axis=0)
            self.mean = numpy.float32(mean)
 
    def update(self, x, alpha = 0.001):
        if len(x.shape) > 0:  
            
            mean        = x.mean(axis=0)
            self.mean   = (1.0 - alpha)*self.mean + alpha*mean

            std         = numpy.std(x, axis=0, ddof=1) + 0.001
            self.std    = (1.0 - alpha)*self.std + alpha*std

        else:
            self.mean   = (1.0 - alpha)*self.mean + alpha*x
            self._std   = (1.0 - alpha)*self._std  + alpha*((x - self.mean)**2)

            self.std    = self._std**0.5 + 0.001

