import numpy
from numpy.core.fromnumeric import nonzero


class RunningStats:
    def __init__(self, shape, initial_value=None):
        self.count = 1
        self.eps   = 0.0000001
        self.mean  = numpy.zeros(shape)
        self.var   = numpy.ones(shape)

        if initial_value is not None:
            self.mean   = initial_value.mean(axis=0)
            self.std    = initial_value.std(axis=0)

        self.mean   = self.mean.astype(numpy.float64)
        self.var    = self.var.astype(numpy.float64)

        self.std    = (self.var**0.5) + self.eps


    def update(self, x): 
        self.count+= 1

        mean = self.mean + (x.mean(axis=0) - self.mean)/self.count
        var  = self.var  + ((x - self.mean)*(x - mean)).mean(axis=0)

        self.mean = mean
        self.var  = var

        self.std  = ((self.var/self.count)**0.5) + self.eps


class RunningStatsMultiHead:
    def __init__(self, shape, initial_value=None, heads_count=1):
        self.shape          = shape
        self.heads_count    = heads_count
        self.stats          = []

        for _ in range(self.heads_count):
            self.stats.append(RunningStats(self.shape, initial_value))

    def update(self, x, head_ids):
        for h in range(self.heads_count):
            indices = (head_ids == h).nonzero()[0]

            if len(indices) > 0:
                values  = numpy.take(x, indices, axis=0)

                self.stats[h].update(values)

    def get(self, batch_size, head_ids):
        means   = numpy.zeros((batch_size, ) + self.shape)
        stds    = numpy.zeros((batch_size, ) + self.shape)

        for i in range(batch_size):
            idx      = head_ids[i]
            means[i] = self.stats[idx].mean
            stds[i]  = self.stats[idx].std

        return means, stds

'''
class RunningStatsMultiHead:
    def __init__(self, shape, initial_value=None, heads_count=1):
        self.shape          = shape
        self.heads_count    = heads_count
        self.stats          = []

        for _ in range(self.heads_count):
            self.stats.append(RunningStats(self.shape, initial_value))

    def update(self, x, head_ids):

        for h in range(self.heads_count):
            indices = (head_ids == h).nonzero()[0]

            if len(indices) > 0:
                values  = numpy.take(x, indices, axis=0)

                self.stats[h].update(values)

    def get(self, batch_size, head_ids):
        means   = numpy.zeros((batch_size, ) + self.shape)
        stds    = numpy.zeros((batch_size, ) + self.shape)

        ax      = tuple(range(1, len(self.shape) + 1))

        for h in range(self.heads_count):
            mask    = (head_ids == h)

            mask    = numpy.expand_dims(mask, axis=ax)

            #repeat means across batch
            means_  = numpy.expand_dims(self.stats[h].mean, 0)
            stds_   = numpy.expand_dims(self.stats[h].std, 0)

            means_  = numpy.repeat(means_, batch_size, axis=0)
            stds_   = numpy.repeat(stds_, batch_size, axis=0)

            #add masked mean and std
            means+= means_*mask
            stds+=  stds_*mask

        return means, stds
'''