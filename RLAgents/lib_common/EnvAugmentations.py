import gym
import numpy
from PIL import Image
import cv2
import time


class EnvAugmentations(gym.Wrapper):
    def __init__(self, env):
        super(EnvAugmentations, self).__init__(env)

    def reset(self):
        state = self.env.reset()
        self.state_shape = state.shape

        self.new_augmentation()

        state = self._augmentations(state)

        return state

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        state = self._augmentations(state)

        return state, reward, done, info

    def _augmentations(self, x):
        if self.random_backgrond or True:
            bg_indices_y, bg_indices_x = numpy.where(x.sum(axis=2) == 0)

            mask = numpy.zeros((x.shape[0], x.shape[1], 1), dtype=numpy.uint8)
            mask[bg_indices_y, bg_indices_x] = 1


            x = (1 - mask)*x + mask*self.background
        
        if self.invert_r:
            x[:, :, 0] = 255 - x[:, :, 0]

        if self.invert_g:
            x[:, :, 1] = 255 - x[:, :, 1]

        if self.invert_b:
            x[:, :, 2] = 255 - x[:, :, 2]

        if self.flip_v:
            x = numpy.flip(x, axis=0)

        if self.flip_h:
            x = numpy.flip(x, axis=1)

        x[:,:,:] = x[:,:,self.color_perm]
        

        
        return x 

    def new_augmentation(self):
        self.random_backgrond   = bool(numpy.random.randint(2))
        self.background         = self._generate_random_background(self.state_shape)

        self.invert_r = bool(numpy.random.randint(2))
        self.invert_g = bool(numpy.random.randint(2))
        self.invert_b = bool(numpy.random.randint(2))

        self.flip_v   = bool(numpy.random.randint(2))
        self.flip_h   = bool(numpy.random.randint(2))

        self.color_perm = numpy.random.permutation(3)



    def _generate_random_background(self, bg_shape):

        divs_height = self._divisors(bg_shape[0])
        divs_width  = self._divisors(bg_shape[1])

        tile_height = numpy.random.choice(divs_height)
        tile_width  = numpy.random.choice(divs_width)
        
        tile        = numpy.random.randint(0, 64, (tile_height, tile_width, 3)).astype(dtype=numpy.uint8)
        
        tile = numpy.repeat(tile, bg_shape[0]//tile_height, axis=0)
        tile = numpy.repeat(tile, bg_shape[1]//tile_width,  axis=1)

        return tile


    def _divisors(self, x):
        result = []
        for i in range(1, int(x**0.5)):
            if x%i == 0:
                result.append(i)
                result.append(x//i)

        return result




if __name__ == "__main__":

    env = gym.make("MontezumaRevengeNoFrameskip-v4")
    
    env = EnvAugmentations(env)

    env.reset()

    steps = 0

    #fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    #writer = cv2.VideoWriter("wrapper_test.avi", fourcc, 25.0, (160, 210)) 
        

    while True:
        action = numpy.random.randint(env.action_space.n)

        state, reward, done, info = env.step(action)

        if done:
            env.reset()

        #if steps%32 == 0:
        #    env.new_augmentation()

        steps+= 1

        if steps%10 == 0:
            x_ = cv2.cvtColor(state, cv2.COLOR_BGR2RGB)
            cv2.imshow("env render", x_)
            cv2.waitKey(1)


        '''
        writer.write(state)
        if steps == 256:
            break
        '''