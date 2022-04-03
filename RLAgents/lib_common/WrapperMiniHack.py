import numpy
from PIL import Image
import  gym
import  minihack


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env, height, width, frame_stacking):
        super(ResizeWrapper, self).__init__(env)
        self.height = height
        self.width  = width
        self.frame_stacking = frame_stacking

        state_shape = (frame_stacking, self.height, self.width)
        self.dtype  = numpy.float32

        self.observation_space  = gym.spaces.Box(low=0.0, high=1.0, shape=state_shape, dtype=self.dtype)
        self.state              = numpy.zeros(state_shape, dtype=self.dtype)

    def observation(self, state):

        img = Image.fromarray(state["pixel_crop"])
        img = img.convert('L')
        img = img.resize((self.height, self.width), Image.NEAREST)
         
        downsmapled = numpy.array(img).astype(self.dtype)/255.0
        downsmapled = numpy.moveaxis(downsmapled, -1, 0)

        self.state    = numpy.roll(self.state, 1, axis=0)
        self.state[0] = downsmapled.copy()

        return self.state


class ScoreWrapper(gym.Wrapper):
    def __init__(self, env, max_steps):
        gym.Wrapper.__init__(self, env)

        self.steps      = 0
        self.max_steps  = max_steps

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.steps+= 1
        if self.steps >= self.max_steps:
            self.steps = 0
            done = True 

        if done:
            self.steps = 0
 
        if reward < 0.0:
            reward = 0.0

        if reward > 1.0:
            reward = 1.0

        return obs, reward, done, info

    def reset(self):
        self.steps = 0
        obs = self.env.reset()
        return obs


def WrapperMiniHack(env_name = "MiniHack-MultiRoom-N4-Extreme-v0", height = 96, width = 96, frame_stacking = 4, max_steps = 256):
    
    env = gym.make(env_name, observation_keys=("glyphs_crop", "pixel_crop"))
    env = ResizeWrapper(env, width, height, frame_stacking)
    env = ScoreWrapper(env, max_steps)

    return env
