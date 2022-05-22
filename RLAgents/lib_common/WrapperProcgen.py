import gym
import procgen
import numpy



class StateWrapper(gym.Wrapper):
    def __init__(self, env):
        super(StateWrapper, self).__init__(env)

        state_shape = (3, 64, 64)

        self.state = numpy.zeros(state_shape, dtype=numpy.float32)

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


class FrameStacking(gym.Wrapper):
    def __init__(self, env, frame_stacking):
        super(FrameStacking, self).__init__(env)

        self.frame_stacking = frame_stacking

        state_shape = (3*self.frame_stacking, 64, 64)

        self.state = numpy.zeros(state_shape, dtype=numpy.float32)

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=state_shape, dtype=numpy.float32)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._get_state(obs)

        return obs, reward, done, info

    def reset(self):
        self.state[:,:,:] = 0.0

        s = self.env.reset()
        s = self._get_state(s) 

        return s
 
    def _get_state(self, s):
        self.state      = numpy.roll(self.state, 3, axis=0)
        self.state[0:3] = s  
        
        return self.state
        
class ScoreWrapper(gym.Wrapper):
    def __init__(self, env, min_score, max_score, averaging_episoded = 100):
        super(ScoreWrapper, self).__init__(env)

        self.min_score          = min_score
        self.max_score          = max_score
        
        self.reward_sum         = 0.0

        self.score_raw          = numpy.zeros((averaging_episoded, ), dtype=numpy.float32)
        self.score_normalised   = numpy.zeros((averaging_episoded, ), dtype=numpy.float32)

        self.ptr = 0 

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


def WrapperProcgen(env_name = "procgen-climber-v0", frame_stacking = 2, render = False):

    if "coinrun" in env_name:
        r_min = 5.0
        r_max = 10.0
    elif "starpilot" in env_name:
        r_min = 2.5
        r_max = 64.0
    elif "caveflyer" in env_name:
        r_min = 3.5
        r_max = 12.0
    elif "dodgeball" in env_name:
        r_min = 1.5
        r_max = 19.0
    elif "fruitbot" in env_name:
        r_min = -1.5
        r_max = 32.4
    elif "chaser" in env_name:
        r_min = 0.5
        r_max = 13.0
    elif "miner" in env_name:
        r_min = 1.5
        r_max = 13.0
    elif "jumper" in env_name:
        r_min = 3.0
        r_max = 10.0
    elif "leaper" in env_name:
        r_min = 3.0
        r_max = 10.0
    elif "maze" in env_name:
        r_min = 5.0
        r_max = 10.0
    elif "bigfish" in env_name:
        r_min = 1.0
        r_max = 40.0
    elif "heist" in env_name:
        r_min = 3.5
        r_max = 10.0
    elif "climber" in env_name:
        r_min = 2
        r_max = 12.6
    elif "plunder" in env_name:
        r_min = 4.5
        r_max = 30.0
    elif "ninja" in env_name:
        r_min = 3.5
        r_max = 10.0
    elif "bossfight" in env_name:
        r_min = 3.5
        r_max = 10.0
    else:
        raise ValueError("\n\nERROR : unknow reward normalisation or unsupported envname\n\n")

    #env = gym.make(env_name, render=render, start_level = 0, num_levels = 100, use_sequential_levels=False, distribution_mode="easy")
    env = gym.make(env_name, render=render, start_level = 0, num_levels = 0, use_sequential_levels=False, distribution_mode="easy")

    env = StateWrapper(env)  

    if frame_stacking > 1:
        env = FrameStacking(env, frame_stacking)  

    env = ScoreWrapper(env, r_min, r_max) 

    return env 

def WrapperProcgenRender(env_name = "procgen-climber-v0"):
    return WrapperProcgen(env_name, frame_stacking = 2, render=True) 

