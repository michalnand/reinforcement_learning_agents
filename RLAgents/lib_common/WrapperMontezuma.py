import gym
import numpy
from PIL import Image

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)

        self._obs_buffer = numpy.zeros((2,) + env.observation_space.shape, dtype=numpy.uint8)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break

        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        return self.env.reset()


class ResizeEnv(gym.ObservationWrapper):
    def __init__(self, env, height = 96, width = 96, frame_stacking = 4):
        super(ResizeEnv, self).__init__(env)
        self.height = height
        self.width  = width
        self.frame_stacking = frame_stacking

        state_shape = (self.frame_stacking, self.height, self.width)
        self.dtype = numpy.float32

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=state_shape, dtype=self.dtype)
        self.state = numpy.zeros(state_shape, dtype=self.dtype)

    def observation(self, state):
        img = Image.fromarray(state)
        img = img.convert('L')
        img = img.resize((self.height, self.width))

        for i in reversed(range(self.frame_stacking-1)):
            self.state[i+1] = self.state[i].copy()
        self.state[0] = (numpy.array(img).astype(self.dtype)/255.0).copy()

        return self.state

class RawScoreEnv(gym.Wrapper):
    def __init__(self, env, max_steps):
        gym.Wrapper.__init__(self, env)

        self.steps      = 0
        self.max_steps  = max_steps

        self.raw_episodes            = 0
        self.raw_score               = 0.0
        self.raw_score_per_episode   = 0.0
        self.raw_score_total         = 0.0  

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.steps+= 1
        if self.steps >= self.max_steps:
            self.steps = 0
            done = True

        self.raw_score+= reward
        self.raw_score_total+= reward
        if done:
            self.steps = 0
            self.raw_episodes+= 1

            k = 0.1
            self.raw_score_per_episode   = (1.0 - k)*self.raw_score_per_episode + k*self.raw_score            
            self.raw_score = 0.0

        return obs, reward, done, info

    def reset(self):
        self.steps = 0
        return self.env.reset()



def WrapperMontezuma(env, height = 96, width = 96, frame_stacking=4, frame_skipping=4, max_steps = 4500):
    env = MaxAndSkipEnv(env, frame_skipping)
    env = ResizeEnv(env, height, width, frame_stacking)
    env = RawScoreEnv(env, max_steps)
    env.observation_space.shape = (frame_stacking, height, width)

    return env


