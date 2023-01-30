import gym
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
        
        if self.frame_counter%32 == 0:
            im_bgr = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)

            resized = cv2.resize(im_bgr, (self.width, self.height), interpolation = cv2.INTER_AREA)

            self.writer.write(resized)

        self.frame_counter+= 1

        return state, reward, done, info

    def reset(self):
        return self.env.reset()


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




class VisitedRoomsEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        
        self.steps              = 0
        self.rooms              = []
        self.room_id            = 0
        self.explored_rooms     = 0
        
    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        
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
        
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()

    def _distance(self, obs):
        other       = numpy.array(self.rooms)
        distances   = ((other - obs)**2)

        shape       = (distances.shape[0], numpy.prod(distances.shape[1:]))
        distances   = distances.reshape(shape)
        distances   = distances.mean(axis=1)

        return numpy.min(distances), numpy.argmin(distances)


'''
def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env

class VisitedRoomsEnv(gym.Wrapper):
    def __init__(self, env, room_address):
        gym.Wrapper.__init__(self, env)
        self.room_address   = room_address

        self.explored_rooms = {}

        self.max_explored_rooms = 0

        self.room_id_prev   = 1
        self.room_id        = 1
        self.level_id       = 0

    def get_current_room(self):
        ram = unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.room_id_prev   = self.room_id
        self.room_id        = self.get_current_room()

        #next level
        if self.room_id == 1 and self.room_id_prev == 15:
            self.level_id+= 1

        if done:
            self.level_id = 0

        room_id = self.room_id + 24*self.level_id

        if room_id in self.explored_rooms:
            self.explored_rooms[room_id]+= 1
        else:
            self.explored_rooms[room_id] = 1

        if len(self.explored_rooms) > self.max_explored_rooms:
            self.max_explored_rooms = len(self.explored_rooms)


        info = {}
        info["room_id"]         = list(self.explored_rooms).index(room_id)
        info["explored_rooms"]  = self.max_explored_rooms

        #print("room_id = ", info["room_id"], self.max_explored_rooms, self.explored_rooms)

        return obs, reward, done, info

    def reset(self):
        return self.env.reset()
'''

class RawScoreEnv(gym.Wrapper):
    def __init__(self, env, max_steps):
        gym.Wrapper.__init__(self, env)

        self.steps      = 0
        self.max_steps  = max_steps

        self.raw_score               = 0.0
        self.raw_score_per_episode   = 0.0
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)

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

        info["raw_score"] = self.raw_score_per_episode

        #reward = max(0.0, float(numpy.sign(reward)))

        reward = max(0.0, float(reward))
        reward = numpy.sign(reward)
        #reward = numpy.log10(1.0 + reward)
        
        return obs, reward, done, info

    def reset(self):
        self.steps = 0
        return self.env.reset()




def WrapperMontezuma(env, height = 96, width = 96, frame_stacking = 4, max_steps = 4500):

    env = NopOpsEnv(env)
    env = StickyActionEnv(env)
    env = RepeatActionEnv(env) 
    env = ResizeEnv(env, height, width, frame_stacking)
    
    env = VisitedRoomsEnv(env)

    #env = VisitedRoomsEnv(env, room_address=3)
    
    env = RawScoreEnv(env, max_steps) 

    return env


def WrapperMontezumaVideo(env, height = 96, width = 96, frame_stacking = 4, max_steps = 4500):
    env = VideoRecorder(env)    

    env = WrapperMontezuma(env, height, width, frame_stacking, max_steps)

    return env

