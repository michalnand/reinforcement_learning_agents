import gymnasium as gym
import numpy
from PIL import Image
import cv2

class VideoRecorder(gym.Wrapper):
    def __init__(self, env, file_name = "video.avi"):
        super(VideoRecorder, self).__init__(env)

        self.height  = env.observation_space.shape[0]
        self.width   = env.observation_space.shape[1]

        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        self.writer = cv2.VideoWriter(file_name, fourcc, 50.0, (self.width, self.height)) 
        self.frame_counter = 0 

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        
        if self.frame_counter%2 == 0:
            im_bgr = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)

            resized = cv2.resize(im_bgr, (self.width, self.height), interpolation = cv2.INTER_AREA)

            self.writer.write(resized) 

        self.frame_counter+= 1

        return state, reward, done, truncated, info

    def reset(self, seed = None, options = None):
        return self.env.reset()

class ColectStatesEnv(gym.Wrapper):

    def __init__(self, env, result_file_name = "states.npy"):
        super(ColectStatesEnv, self).__init__(env)

        self.result_file_name = result_file_name
        self.states = []

    def reset(self, seed = None, options = None):
        if len(self.states) > 0:
            print("saving states into ", self.result_file_name)
            self.states = numpy.array(self.states, dtype=numpy.float32)
            with open(self.result_file_name, 'wb') as f:
                numpy.save(f, self.states) 

            self.states = []
        return self.env.reset(seed, options)

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)

        self.states.append(state)

        return state, reward, done, truncated, info


class NopOpsEnv(gym.Wrapper):
    def __init__(self, env=None, max_count=30):
        super(NopOpsEnv, self).__init__(env)
        self.max_count = max_count

    def reset(self, seed = None, options = None):
        self.env.reset()

        noops = numpy.random.randint(1, self.max_count + 1)
         
        for _ in range(noops):
            obs, _, done, _, _ = self.env.step(0)

            if done:
                obs = self.env.reset()
           
        return obs, None

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

    def reset(self, seed = None, options = None):
        self.last_action = 0
        return self.env.reset()
 

class RepeatActionEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.successive_frame = numpy.zeros((2,) + self.env.observation_space.shape, dtype=numpy.uint8)

    def reset(self, seed = None, options = None):
        return self.env.reset()

    def step(self, action):
        reward, done = 0, False
        for t in range(4):
            state, r, done, truncated, info = self.env.step(action)
            if t == 2:
                self.successive_frame[0] = state
            elif t == 3:
                self.successive_frame[1] = state
            reward += r
            if done:
                break

        state = self.successive_frame.max(axis=0)
        return state, reward, done, truncated, info


class ResizeEnv(gym.ObservationWrapper):
    def __init__(self, env, height = 96, width = 96, frame_stacking = 4):
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



class ResizeEnvColor(gym.ObservationWrapper):
    def __init__(self, env, height = 96, width = 96, frame_stacking = 4):
        super(ResizeEnvColor, self).__init__(env)
        self.height = height
        self.width  = width
        self.frame_stacking = frame_stacking

        state_shape = (self.frame_stacking*3, self.height, self.width)
        self.dtype  = numpy.float32

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=state_shape, dtype=self.dtype)
        self.state = numpy.zeros(state_shape, dtype=self.dtype)

    def observation(self, state):
        img = Image.fromarray(state)
        img = img.resize((self.height, self.width))
        img = numpy.array(img).astype(self.dtype)/255.0
        img = numpy.rollaxis(img, 2, 0)
        img = numpy.array(img)

        self.state      = numpy.roll(self.state, 3, axis=0)
        self.state[0:3] = img

        return self.state 


class VisitedRoomsEnv(gym.Wrapper):
    '''
    room_address for games : 
    montezuma revenge : 3
    pitfall           : 1
    '''
    def __init__(self, env, room_address = 3):
        gym.Wrapper.__init__(self, env)
        self.room_address = room_address

        self.visited_rooms = {}

    def step(self, action):
        obs, reward, done, truncated, _ = self.env.step(action)

        room_id = self._get_current_room_id()

        if room_id not in self.visited_rooms:
            self.visited_rooms[room_id] = 1
        else:
            self.visited_rooms[room_id]+= 1

        info = {}
        info["room_id"]         = room_id
        info["explored_rooms"]  = len(self.visited_rooms)

        #print("room_id = ", room_id, len(self.visited_rooms))

        return obs, reward, done, truncated, info
    

    def _get_current_room_id(self):
        ram = self.env.unwrapped.ale.getRAM()
        return int(ram[self.room_address])


'''
class VisitedRoomsEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        
        self.steps              = 0
        self.rooms              = []
        self.room_id            = 0
        self.explored_rooms     = 0
        
    def step(self, action):
        obs, reward, done, truncated, _ = self.env.step(action)
        
        if self.steps%32 == 0: 
            if len(self.rooms) == 0:
                self.rooms.append(obs[0].copy())
            else:
                distance, room_id = self._distance(obs[0])
                if distance > 0.01 and len(self.rooms) < 100:
                    self.rooms.append(obs[0].copy())

                self.room_id        = room_id
                self.explored_rooms = len(self.rooms)

        self.steps+= 1

        info = {}
        info["room_id"]         = self.room_id
        info["explored_rooms"]  = self.explored_rooms
        
        return obs, reward, done, truncated, info

    def reset(self, seed = None, options = None):
        return self.env.reset()

    def _distance(self, obs):
        other       = numpy.array(self.rooms)
        distances   = ((other - obs)**2)

        shape       = (distances.shape[0], numpy.prod(distances.shape[1:]))
        distances   = distances.reshape(shape)
        distances   = distances.mean(axis=1)

        return numpy.min(distances), numpy.argmin(distances)
'''


class RawScoreEnv(gym.Wrapper):
    def __init__(self, env, max_steps):
        gym.Wrapper.__init__(self, env)

        self.steps      = 0
        self.max_steps  = max_steps

        self.raw_score               = 0.0
        self.raw_score_per_episode   = 0.0

        
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        self.steps+= 1
        if self.steps >= self.max_steps:
            self.steps = 0
            done = True 

        self.raw_score+= reward
        if done: 
            self.steps = 0
  
            k = 0.1 
            self.raw_score_per_episode = (1.0 - k)*self.raw_score_per_episode + k*self.raw_score            
            self.raw_score = 0.0

        info["raw_score"]  = self.raw_score_per_episode
        info["raw_reward"] = reward

        reward = max(0.0, float(reward)) 
        reward = numpy.sign(reward)
        
        return obs, reward, done, truncated, info

    def reset(self, seed = None, options = None):
        self.steps = 0
        return self.env.reset()




def WrapperMontezuma(env, height = 96, width = 96, frame_stacking = 4, max_steps = 4500):

    env_str = str(env)
    print("wrapper ", env_str, "MontezumaRevenge" in env_str, "Pitfall" in env_str)
    
    env = NopOpsEnv(env)
    env = StickyActionEnv(env)
    env = RepeatActionEnv(env) 
    #env = ColectStatesEnv(env)
    env = ResizeEnv(env, height, width, frame_stacking)
     
    env = VisitedRoomsEnv(env)    
    env = RawScoreEnv(env, max_steps) 

    return env



def WrapperMontezumaColor(env, height = 96, width = 96, frame_stacking = 4, max_steps = 4500):

    env = NopOpsEnv(env)
    env = StickyActionEnv(env)
    env = RepeatActionEnv(env) 
    #env = ColectStatesEnv(env)
    env = ResizeEnvColor(env, height, width, frame_stacking)
    
    env = VisitedRoomsEnv(env)    
    env = RawScoreEnv(env, max_steps) 

    return env


def WrapperMontezumaVideo(env, height = 96, width = 96, frame_stacking = 4, max_steps = 4500):
    env = VideoRecorder(env)    

    env = WrapperMontezuma(env, height, width, frame_stacking, max_steps)

    return env



def WrapperMontezumaColorVideo(env, height = 96, width = 96, frame_stacking = 4, max_steps = 4500):
    env = VideoRecorder(env)    

    env = WrapperMontezumaColor(env, height, width, frame_stacking, max_steps)

    return env
