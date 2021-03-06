import gym
import numpy
from PIL import Image

class NopOpsEnv(gym.Wrapper):
    def __init__(self, env=None, max_count=30):
        super(NopOpsEnv, self).__init__(env)
        self.max_count = max_count

    def reset(self):
        self.env.reset()

        noops = numpy.random.randint(1, self.max_count + 1)
         
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)

            if done:
                self.env.reset()
                obs, _, _ ,_ = self.env.step(1)
                obs, _, _ ,_ = self.env.step(2)
           
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

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

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self):
        self.env.reset()
        
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()

        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()

        return obs

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done          = True

        self.raw_episodes           = 0
        self.raw_score              = 0.0
        self.raw_score_per_episode  = 0.0
        self.raw_score_total        = 0.0  

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done

        self.raw_score+= reward
        self.raw_score_total+= reward

        if self.was_real_done:
            k = 0.1

            self.raw_episodes+= 1
            self.raw_score_per_episode = (1.0 - k)*self.raw_score_per_episode + k*self.raw_score
            self.raw_score = 0.0


        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done    = True 
            reward  = -1.0
        if lives == 0 and self.inital_lives > 0:
            reward = -1.0 

        self.lives = lives

        reward = numpy.clip(reward, -1.0, 1.0)
        return obs, reward, done, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs = self.env.reset(**kwargs) 
        else:
            obs, _, _, _ = self.env.step(1)
            obs, _, _, _ = self.env.step(2)
            obs, _, _, _ = self.env.step(0) 

        self.lives = self.env.unwrapped.ale.lives()
        self.inital_lives = self.env.unwrapped.ale.lives()
        return obs



class NoRewardsEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
       
        self.raw_episodes           = 0
        self.raw_score              = 0.0
        self.raw_score_per_episode  = 0.0
        self.raw_score_total        = 0.0

    def step(self, action): 
        obs, _, done, info = self.env.step(action)

        self.raw_episodes           = self.env.raw_episodes
        self.raw_score              = self.env.raw_score
        self.raw_score_per_episode  = self.env.raw_score_per_episode
        self.raw_score_total        = self.env.raw_score_total

        return obs, 0.0, done, info

    def reset(self, **kwargs):
        return self.env.reset()


def WrapperAtari(env, height = 96, width = 96, frame_stacking=4, frame_skipping=4):
    env = NopOpsEnv(env)
    env = FireResetEnv(env) 
    env = MaxAndSkipEnv(env, frame_skipping)
    env = ResizeEnv(env, height, width, frame_stacking)
    env = EpisodicLifeEnv(env)

    return env
 
def WrapperAtariNoRewards(env, height = 96, width = 96, frame_stacking=4, frame_skipping=4):
    env = NopOpsEnv(env)
    env = FireResetEnv(env) 
    env = MaxAndSkipEnv(env, frame_skipping)
    env = ResizeEnv(env, height, width, frame_stacking)
    env = EpisodicLifeEnv(env)

    env = NoRewardsEnv(env) 

    return env
