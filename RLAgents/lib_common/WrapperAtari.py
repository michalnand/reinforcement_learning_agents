import gymnasium as gym
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
                obs = self.env.reset()
           
        return obs

    def step(self, action):
        return self.env.step(action)

class StickyActionEnv(gym.Wrapper):
    def __init__(self, env, p=0.25):
        super(StickyActionEnv, self).__init__(env)
        self.p = p
        self.last_action = 0

    def step(self, action):
        if numpy.random.uniform() < self.p:
            action = self.last_action

        self.last_action = action
        return self.env.step(action)

    def reset(self):
        self.last_action = 0
        return self.env.reset()
 

class RepeatActionEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.successive_frame = numpy.zeros((2,) + self.env.observation_space.shape, dtype=numpy.uint8)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        reward, done = 0, False
        for t in range(4):
            state, r, done, info = self.env.step(action)
            if t == 2:
                self.successive_frame[0] = state
            elif t == 3:
                self.successive_frame[1] = state
            reward += r
            if done:
                break

        state = self.successive_frame.max(axis=0)
        return state, reward, done, info

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
            self.steps        = 0
            self.raw_episodes+= 1
 
            k = 0.1
            self.raw_score_per_episode   = (1.0 - k)*self.raw_score_per_episode + k*self.raw_score            
            self.raw_score = 0.0

        
        lives = self.env.unwrapped.ale.lives()

        #life lost negative reward
        if lives < self.lives:
            self.lives  = lives
            reward      = -1.0
        #points reward
        else:
            reward = float(numpy.sign(reward))

        return obs, reward, done, info

    def reset(self):
        self.steps = 0
        self.lives = self.env.unwrapped.ale.lives()

        return self.env.reset()



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


def WrapperAtari(env, height = 96, width = 96, frame_stacking=4, frame_skipping=4, max_steps=4500):
    env = NopOpsEnv(env)
    env = StickyActionEnv(env)
    env = RepeatActionEnv(env) 
    env = FireResetEnv(env)

    env = ResizeEnv(env, height, width, frame_stacking)
    env = RawScoreEnv(env, max_steps)

    return env
 
def WrapperAtariNoRewards(env, height = 96, width = 96, frame_stacking=4, frame_skipping=4, max_steps=4500):
    env = NopOpsEnv(env)
    env = StickyActionEnv(env)
    env = RepeatActionEnv(env) 
    env = FireResetEnv(env)

    env = ResizeEnv(env, height, width, frame_stacking)
    env = RawScoreEnv(env, max_steps)

    env = NoRewardsEnv(env) 

    return env
