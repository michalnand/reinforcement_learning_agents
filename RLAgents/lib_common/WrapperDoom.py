import numpy
from PIL import Image


import gym
from gym import spaces

from vizdoom import DoomGame, ScreenResolution

import time


class GameIO(gym.Env):
    
    def __init__(self, game, frame_skipping = 4):
        gym.Env.__init__(self) 
        
        self.game           = game
        self.frame_skipping = frame_skipping

        self.actions_count      = self.game.get_available_buttons_size()
        self.action_space       = spaces.Discrete(self.actions_count)
        self.observation_space  = spaces.Box(low=0, high=255,  shape=(3, 480, 640), dtype=numpy.uint8)


    def step(self, action):
        action_one_hot = numpy.zeros(self.actions_count).astype(int)
        action_one_hot[action] = 1

        self.game.set_action(action_one_hot.tolist())
        self.game.advance_action(self.frame_skipping)


        done        = self.game.is_episode_finished()
        reward      = self.game.get_last_reward()/100.0
        if done:
            state   = self.reset()
        else:
            state   = self.game.get_state().screen_buffer
        
        

        return state, reward, done, None
        


    def reset(self):
        self.game.new_episode()
        return self.game.get_state().screen_buffer
  


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
        img = Image.fromarray(state, 'RGB')
        img = img.convert('L')
        img = img.resize((self.height, self.width))

        for i in reversed(range(self.frame_stacking-1)):
            self.state[i+1] = self.state[i].copy()
        self.state[0] = (numpy.array(img).astype(self.dtype)/255.0).copy()

        return self.state





def WrapperDoom(scenario, set_window_visible=False, set_sound_enabled = False, height = 96, width = 96, frame_stacking=4, frame_skipping=4):

    game = DoomGame()
    game.load_config(scenario)
    game.set_sound_enabled(set_sound_enabled)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(set_window_visible)
    game.init()
    
    env = GameIO(game, frame_skipping)
    env = ResizeEnv(env, height, width, frame_stacking)

   
    return env
    
def WrapperDoomRender(scenario):
    return WrapperDoom(scenario, set_window_visible=True, set_sound_enabled=True)