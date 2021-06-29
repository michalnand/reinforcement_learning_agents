import time
import numpy


def gae_ref(rewards, values, dones, gamma = 0.99, lam = 0.9):

    envs_count  = rewards.shape[0]
    buffer_size = rewards.shape[1]

    returns     = numpy.zeros((envs_count, buffer_size))
    advantages  = numpy.zeros((envs_count, buffer_size))

    for e in range(envs_count):
        last_gae  = 0.0

        for n in reversed(range(buffer_size-1)):
        
            if dones[e][n] > 0:
                delta       = rewards[e][n] - values[e][n]
                last_gae    = delta
            else:
                delta       = rewards[e][n] + gamma*values[e][n+1] - values[e][n]
                last_gae    = delta + gamma*lam*last_gae

            returns[e][n]    = last_gae + values[e][n]
            advantages[e][n] = last_gae

    return returns, advantages


def gae_fast(rewards, values, dones, gamma = 0.99, lam = 0.9):
    
    envs_count  = rewards.shape[0]
    buffer_size = rewards.shape[1]

    returns     = numpy.zeros((buffer_size, envs_count))
    advantages  = numpy.zeros((buffer_size, envs_count))

    rewards_t   = numpy.transpose(rewards)
    values_t    = numpy.transpose(values)
    dones_t     = numpy.transpose(dones)

    last_gae    = numpy.zeros((envs_count))
    
    for n in reversed(range(buffer_size-1)):
        delta           = rewards_t[n] + gamma*values_t[n+1]*(1.0 - dones_t[n]) - values_t[n]
        last_gae        = delta + gamma*lam*last_gae*(1.0 - dones_t[n])
        
        returns[n]      = last_gae + values_t[n]
        advantages[n]   = last_gae

    returns     = numpy.transpose(returns)
    advantages  = numpy.transpose(advantages)

    return returns, advantages


if __name__ == "__main__":
    envs_count  = 128
    buffer_size = 4096

    #some random inputs
    rewards_b   = 100.0*numpy.random.randn(envs_count, buffer_size)
    values_b    = 100.0*numpy.random.randn(envs_count, buffer_size)
    dones_b     = numpy.random.rand(envs_count, buffer_size) > 0.8
    
    #reference result
    ts = time.time()
    returns_ref, advantages_ref = gae_ref(rewards_b, values_b, dones_b)
    te = time.time()

    time_ref = te-ts

    #optimised result
    ts = time.time()
    returns_fast, advantages_fast = gae_fast(rewards_b, values_b, dones_b)
    te = time.time()

    time_fast = te-ts

    #compute errors
    returns_error    = ((returns_ref - returns_fast)**2).mean()
    advantages_error = ((advantages_ref - advantages_fast)**2).mean()

    #print results
    print("time_ref  = ", round(time_ref, 4))
    print("time_fast = ", round(time_fast, 4))
    print("speedup   = ", round(time_ref/(time_fast + 0.000000001), 4), "x")
    print("returns_error    = ", returns_error)
    print("advantages_error = ", advantages_error)
    
