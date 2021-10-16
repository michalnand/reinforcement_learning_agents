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

'''
class RunningStats:
    def __init__(self, shape, initial_value = None):
        self.count = 1
        self.mean  = numpy.zeros(shape)
        self.var   = numpy.ones(shape)

        if initial_value is not None:
            self.mean   = initial_value.mean(axis=0)
            self.var    = initial_value.var(axis=0)
            self.count  = initial_value.shape[0]

    def update(self, x):   

        x_ = x.mean(axis=0)

        self.n+= x.shape[0] 

        mean = self.mean + (x_ - self.mean)/self.n
        var  = self.var  + ((x_ - self.mean)**2)

        self.mean = mean
        self.var  = var

        self.std  = ((self.var/self.n)**0.5) + 0.000001 
'''



class RunningStats(object):
    def __init__(self, shape, initial_value = None):
        self.count = 0
        self.mean  = numpy.zeros(shape)
        self.var   = numpy.ones(shape)

        if initial_value is not None:
            self.mean = initial_value.mean(axis=0)
            self.var  = initial_value.var(axis=0)

        self.mean   = self.mean.astype(numpy.float64)
        self.var    = self.var.astype(numpy.float64)

    def update(self, x):
        batch_mean  = numpy.mean(x, axis=0)
        batch_var   = numpy.var(x, axis=0)
        batch_size  = x.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_size)

        self.std  = (self.var**0.5) + 0.000001 

    def _update_from_moments(self, batch_mean, batch_var, batch_size):
        delta       = batch_mean - self.mean
        tot_count   = self.count + batch_size

        new_mean = self.mean + delta * batch_size / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_size)
        M2 = m_a + m_b + numpy.square(delta) * self.count * batch_size / (self.count + batch_size)
        new_var = M2 / (self.count + batch_size)

        new_count = batch_size + self.count

        self.mean   = new_mean
        self.var    = new_var
        self.count  = new_count