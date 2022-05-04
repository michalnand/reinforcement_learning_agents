import gym
import procgen
import numpy

from PIL import Image
import cv2

class VideoRecorder(gym.Wrapper):
    def __init__(self, env, file_name = "video.avi"):
        super(VideoRecorder, self).__init__(env)

        self.height  = 2*env.observation_space.shape[0]
        self.width   = 2*env.observation_space.shape[1]

        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        self.writer = cv2.VideoWriter(file_name, fourcc, 25.0, (self.width, self.height)) 
        self.frame_counter = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        
        if self.frame_counter%4 == 0:
            im_bgr = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)

            resized = cv2.resize(im_bgr, (self.width, self.height), interpolation = cv2.INTER_AREA)

            self.writer.write(resized)

        self.frame_counter+= 1

        return state, reward, done, info

    def reset(self):
        return self.env.reset()


class StateEnv(gym.ObservationWrapper):
    def __init__(self, env, frame_stacking):
        super(StateEnv, self).__init__(env)
        self.frame_stacking = frame_stacking
        state_shape = (3*self.frame_stacking, 64, 64)
        self.dtype  = numpy.float32

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=state_shape, dtype=self.dtype)
        self.state = numpy.zeros(state_shape, dtype=self.dtype)

    def observation(self, state):

        state_new = numpy.moveaxis(state, 2, 0)
        state_new = numpy.array(state_new).astype(self.dtype)/255.0

        if self.frame_stacking == 1:
            self.state = state_new
        else:
            self.state    = numpy.roll(self.state, 3, axis=0)
            self.state[0:3] = state_new[0:3]
        
        return self.state

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



def WrapperProcgen(env_name = "procgen-climber-v0", frame_stacking = 1, max_steps = 4500, render=False):

    #env = gym.make(env_name, render=render, start_level = 0, num_levels = 1, use_sequential_levels=True)
    env = gym.make(env_name, render=render, start_level = 0, num_levels = 0, use_sequential_levels=True)
    env = StateEnv(env, frame_stacking) 
    env = MaxStepsEnv(env, max_steps)
      
    return env 


def WrapperProcgenRender(env_name, frame_stacking = 1, max_steps = 4500):
    env = WrapperProcgen(env_name, frame_stacking, max_steps, True)

    return env

