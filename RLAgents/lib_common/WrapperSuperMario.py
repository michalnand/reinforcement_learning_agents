import gym
import numpy
from PIL import Image

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

class NopOpsEnv(gym.Wrapper):
    def __init__(self, env=None, max_count=30):
        super(NopOpsEnv, self).__init__(env)
        self.max_count = max_count

    def reset(self):
        self.env.reset()

        noops = numpy.random.randint(1, self.max_count + 1)
         
        for _ in range(noops):
            obs, _, _, _ = self.env.step(0)
            
        return obs


class SkipEnv(gym.Wrapper):
    def __init__(self, env, skip = 4):
        gym.Wrapper.__init__(self, env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            state, reward, done, info = self.env.step(action)
            total_reward+= reward
            if done:
                break

        return state, total_reward, done, info


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
        self.state[0] = numpy.array(img).astype(self.dtype)/255.0

        return self.state

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self):
        self.env.reset()
        obs, _, _, _ = self.env.step(1)
        obs, _, _, _ = self.env.step(2)

        return obs


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        
        self.was_real_done  = True

        
    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        return obs, reward, done, info

    def _reset(self, **kwargs):
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            obs, _, _, _ = self.env.step(0)
        
        return obs


class ClipRewardEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        self.raw_episodes            = 0
        self.raw_score               = 0.0
        self.raw_score_per_episode   = 0.0
        self.raw_score_total         = 0.0  


    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.raw_score+= reward
        self.raw_score_total+= reward

        if done:
            self.raw_episodes+= 1
            k = 0.1
            self.raw_score_per_episode   = (1.0 - k)*self.raw_score_per_episode + k*self.raw_score
            self.raw_score = 0.0

        reward = numpy.clip(reward/15.0, -1.0, 1.0)

        if numpy.abs(reward) < 0.2:
            reward = 0.0

        return obs, reward, done, info



def WrapperSuperMario(env, height = 96, width = 96, frame_stacking=4, frame_skipping=4):
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = NopOpsEnv(env)
    env = SkipEnv(env, frame_skipping)
    env = ResizeEnv(env, height, width, frame_stacking)
    env = FireResetEnv(env) 
    env = EpisodicLifeEnv(env)
    env = ClipRewardEnv(env)

    for i in range(5):
        env.reset()

    return env