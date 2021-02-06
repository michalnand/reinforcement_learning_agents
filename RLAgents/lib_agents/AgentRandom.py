import numpy



class AgentRandom:
    def __init__(self, env):
        self.env = env
        self.actions_count = self.env.action_space.n
        self.iterations = 0

    def main(self):
        self.iterations+= 1
        action = numpy.random.randint(self.actions_count)
        
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
    
   
