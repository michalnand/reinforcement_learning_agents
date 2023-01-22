import numpy

class RunningStats:

    def __init__(self, shape):
        self.mean         = numpy.zeros(shape, dtype=numpy.float32)
        self.pwr_sum_mean = numpy.zeros(shape, dtype=numpy.float32)

        self.count = 0

    def update(self, x):  
         
        self.count+= 1

        self.mean+= (x.mean(axis=0) - self.mean) / self.count

        self.pwr_sum_mean+= (x**2 - self.pwr_sum_mean).mean(axis=0) / self.count

        var = self.pwr_sum_mean - self.mean**2
        var = numpy.maximum(var, numpy.zeros_like(var) + 10**-3)
        self.std = var**0.5 

        print("running stats : ", self.mean.mean(), self.std.mean())

        return self.mean, self.std



    

'''
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
'''
    


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


'''
if __name__ == "__main__":

    shape = (4, 96, 96)

    batch = 32

    rs = RunningStats(shape)

    x_all = []

    for i in range(100):
        x = 3*numpy.random.randn(batch, shape[0], shape[1], shape[2]) + 41
        x_all.append(x)

        mean_r = numpy.mean(x_all, axis=(0, 1))
        std_r  = numpy.std(x_all, axis=(0, 1))
        mean_t, std_t = rs.add(x)

        dff_m = ((mean_r - mean_t)**2).mean()
        dff_s = ((std_r - std_t)**2).mean()

        print(dff_m, dff_s, mean_r.shape, std_r.shape)

'''