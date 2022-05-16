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
        
        #s = (s - s.mean())/(s.std() + 10**-10)

        return s

class MaxSteps(gym.Wrapper):
    def __init__(self, env, max_steps):
        super(MaxSteps, self).__init__(env)

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


class Score(gym.Wrapper):
    def __init__(self, env, min_score, max_score, averaging_episoded = 100):
        super(Score, self).__init__(env)

        self.min_score          = min_score
        self.max_score          = max_score
        
        self.reward_sum         = 0.0

        self.score_raw          = numpy.zeros((averaging_episoded, ), dtype=numpy.float32)
        self.score_normalised   = numpy.zeros((averaging_episoded, ), dtype=numpy.float32)

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        self.reward_sum+= reward

        if done:
            self.score_raw[self.ptr]        = self.reward_sum
            self.score_normalised[self.ptr] = self._normalise(self.reward_sum)
            
            self.reward_sum = 0.0

            self.ptr = (self.ptr+1)%self.score_raw.shape[0]

        info["raw_score"]        = round(self.score_raw.mean(), 5)
        info["normalised_score"] = round(self.score_normalised.mean(), 5)
    
        return state, reward, done, info
        
    def reset(self):
        self.reward_sum = 0.0
        return self.env.reset()

    def _normalise(self, x):
        y = (x - self.min_score)/(self.max_score - self.min_score)
        return y


def WrapperProcgen(env_name = "procgen:procgen-climber-v0", max_steps = 4500, render = False):

    r_min = 0.0
    r_max = 1.0

    if "coinrun" in env_name:
        r_min = 5.0
        r_max = 10.0
    elif "starpilot" in env_name:
        r_min = 1.5
        r_max = 35.0
    elif "caveflyer" in env_name:
        r_min = 2.0
        r_max = 13.4
    elif "dodgeball" in env_name:
        r_min = 1.5
        r_max = 19.0
    elif "fruitbot" in env_name:
        r_min = -0.5
        r_max = 27.2
    elif "chaser" in env_name:
        r_min = 0.5
        r_max = 14.2
    elif "miner" in env_name:
        r_min = 1.5
        r_max = 20.0
    elif "jumper" in env_name:
        r_min = 1.0
        r_max = 10.0
    elif "leaper" in env_name:
        r_min = 1.5
        r_max = 10.0
    elif "maze" in env_name:
        r_min = 4.0
        r_max = 10.0
    elif "bigfish" in env_name:
        r_min = 0.0
        r_max = 40.0
    elif "heist" in env_name:
        r_min = 2.0
        r_max = 10.0
    elif "climber" in env_name:
        r_min = 1.0
        r_max = 12.6
    elif "pluner" in env_name:
        r_min = 3.0
        r_max = 30.0
    elif "ninja" in env_name:
        r_min = 2.0
        r_max = 10.0
    elif "bossfight" in env_name:
        r_min = 0.5
        r_max = 13.0
    else:
        print("ERROR : unknow reward normalisation")



    env = gym.make(env_name, render=render, start_level = 0, num_levels = 0, use_sequential_levels=False)
    #env = gym.make(env_name, render=render, start_level = 0, num_levels = 1, use_sequential_levels=True)
    env = ExtractState(env)  
    env = MaxSteps(env, max_steps) 
    env = Score(env, r_min, r_max) 

    return env 

def WrapperProcgenRender(env_name = "procgen-climber-v0", max_steps = 4500):
    return WrapperProcgen(env_name, max_steps, True) 
