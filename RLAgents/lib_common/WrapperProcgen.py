import gym
import procgen
import numpy


class ExtractState(gym.Wrapper):
    def __init__(self, env):
        super(ExtractState, self).__init__(env)

        state_shape = (3, 64, 64)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=state_shape, dtype=numpy.float32)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._get_state(obs)

        return obs, reward, done, info

    def reset(self):
        s = self.env.reset()
        s = self._get_state(s)

        return s

    def _get_state(self, s):
        s = numpy.array(s, dtype=numpy.float32)/255.0
        s = numpy.moveaxis(s, 2, 0)

        return s

class MaxStepsEnv(gym.Wrapper):
    def __init__(self, env, max_steps):
        super(MaxStepsEnv, self).__init__(env)

        self.max_steps = max_steps
        self.steps     = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        self.steps+= 1
        if self.steps >= self.max_steps:
            done = True

        return state, reward, done, info
        
    def reset(self):
        self.steps = 0
        return self.env.reset()


class RewardNormalise(gym.Wrapper):
    def __init__(self, env, gamma = 0.99, clip = 10.0):
        super(RewardNormalise, self).__init__(env)

        self.gamma      = gamma
        self.clip       = clip
        self.mean       = 0.0
        self.var        = 0.0

        self.raw_score_per_episode  = 0.0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.raw_score_per_episode+= reward

        self.mean = self.gamma*self.mean + (1.0 - self.gamma)*reward 
        self.var  = self.gamma*self.var  + (1.0 - self.gamma)*((reward - self.mean)**2)

        eps = 10**-8 

        reward_norm = numpy.clip(reward/numpy.sqrt(self.var + eps) , -self.clip, self.clip)

        return obs, reward_norm, done, info

    def reset(self):
        self.raw_score_per_episode = 0.0
        return self.env.reset()


        


 

def WrapperProcgen(env_name = "procgen:procgen-climber-v0", max_steps = 4500, render = False):
    env = gym.make(env_name, render=render, start_level = 0, num_levels = 0, use_sequential_levels=False)
    env = ExtractState(env) 
    env = MaxStepsEnv(env, max_steps)
    env = RewardNormalise(env)

    return env 

def WrapperProcgenRender(env_name = "procgen:procgen-climber-v0", max_steps = 4500):
    return WrapperProcgen(env_name, max_steps, True) 
