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


class ResizeEnv(gym.ObservationWrapper):
    def __init__(self, env, height = 64, width = 64, frame_stacking = 4):
        super(ResizeEnv, self).__init__(env)
        self.height = height
        self.width  = width
        self.frame_stacking = frame_stacking

        state_shape = (self.frame_stacking, self.height, self.width)
        self.dtype  = numpy.float32

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=state_shape, dtype=self.dtype)
        self.state = numpy.zeros(state_shape, dtype=self.dtype)

    def observation(self, state):
        img = Image.fromarray(state)
        img = img.convert('L')
        img = img.resize((self.height, self.width))

        self.state    = numpy.roll(self.state, 1, axis=0)
        self.state[0] = (numpy.array(img).astype(self.dtype)/255.0).copy()

        return self.state

  
def WrapperProcgen(env_name = "procgen-climber-v0", height = 64, width = 64, frame_stacking = 4, render=False):

    env = gym.make(env_name, render=render, start_level = 0, num_levels = 1, use_sequential_levels=True)
    env = ResizeEnv(env, height, width, frame_stacking) 
     
    return env 

def WrapperProcgenVideo(env_name, height = 64, width = 64, frame_stacking = 4):
    env = gym.make(env_name, render=False, start_level = 0, num_levels = 1, use_sequential_levels=True)
    env = VideoRecorder(env)    
    env = WrapperProcgen(env, height, width, frame_stacking)

    return env

def WrapperProcgenRender(env_name, height = 64, width = 64, frame_stacking = 4):
    env = WrapperProcgen(env_name, height, width, frame_stacking, render=True)

    return env

