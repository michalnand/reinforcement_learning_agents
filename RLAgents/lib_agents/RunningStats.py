import numpy

class RunningStats:
    def __init__(self, shape = (), initial_value = None):

        self.mean = numpy.zeros(shape)
        self.std  = numpy.ones(shape)

        if initial_value is not None:
            mean = initial_value.mean(axis=0)
            self.mean = mean
 
    def update(self, x, alpha = 0.0001):
        if len(x.shape) > 0:
            mean = x.mean(axis=0)
            self.mean = (1.0 - alpha)*self.mean + alpha*mean

            std = numpy.std(x, axis=0, ddof=1) + 10e-7
            self.std = (1.0 - alpha)*self.std + alpha*std

        else:
            self.mean = (1.0 - alpha)*self.mean + alpha*x
            self.std  = (1.0 - alpha)*self.std  + alpha*numpy.abs(x - self.mean) 
