import numpy



class AgentRandomContinuous:
    def __init__(self, env):
        self.env = env
        self.actions_count  = self.env.action_space.shape[0]

        self.iterations = 0

    def main(self):
        self.iterations+= 1
        action = 4.0*numpy.random.randn(self.actions_count)

        action = numpy.tanh(action)

        self.state, reward, done, info = self.env.step(action)

        '''
        print(state.shape)
        print(state)
        print(reward)
        print("\n\n")
        '''
        if done:
            self.state = self.env.reset()

        return reward, done
    
   
