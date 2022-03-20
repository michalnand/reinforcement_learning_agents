import torch

class StatesBuffer:  

    def __init__(self, buffer_size, shape, add_threshold, downsample):
        
        self.add_threshold  = add_threshold

        shape_down          = (shape[0], shape[1]//downsample, shape[2]//downsample)

        self.featues_count  = shape_down[0]*shape_down[1]*shape_down[2]

        self.downsample     = torch.nn.AvgPool2d(downsample, downsample)

        self.states_b       = 100.0*torch.ones((buffer_size, self.featues_count))
        self.steps_b        = torch.zeros((buffer_size, ))
        self.visitings_b    = torch.zeros((buffer_size, )) 

        self.current_idx    = 0
 

    def update(self, states, steps_np):

        states_t = torch.from_numpy(states).float()
        steps_t  = torch.from_numpy(steps_np).float()

        #downsample and flatten
        print(">>>> ", states_t.shape)
        states_down     = self.downsample(states_t)
        states_down     = states_down.reshape((states_down.shape[0], self.featues_count))

        print(">>>> ", states_down.shape)

        print("\n\n")
 
        #add initial state
        if self.current_idx == 0:
            self._add_new_state(states_down[0], steps_t[0])

        used_states     = self.states_b[0:self.current_idx]

        #mean distances
        distances       = torch.cdist(states_down, used_states)/self.featues_count

        #find closest distances and indices
        closest_val, closest_ids = torch.min(distances, dim=1)

        #add new states if threshold reached
        for i in range(closest_val.shape[0]):
            if closest_val[i] > self.add_threshold:
                self._add_new_state(states_down[i], steps_t[i])
 
        #shorter path reward
        steps               = self.steps_b[closest_ids]
        less_steps_reward   = torch.round(steps_t) < torch.round(steps)

        #update with better count
        self.steps_b[closest_ids] = less_steps_reward*steps_t + torch.logical_not(less_steps_reward)*steps

        less_steps_reward = 1.0*less_steps_reward


        '''
        #update steps count with exponential moving average
        k = 0.1
        self.steps_b[closest_ids] = (1.0 - k)*steps + k*steps_t
        '''
        
        #visitings reward
        visitings = self.visitings_b[closest_ids]   

        visitings_reward  = 1.0/(1.0 + (visitings**0.5))

        #update counts
        for i in range(closest_val.shape[0]):
            self.visitings_b[closest_ids[i]]+= 1
            
        return less_steps_reward.numpy(), visitings_reward.numpy()

    def get_usage(self):
        return self.current_idx/self.states_b.shape[0]
 
    def save(self, path):
        numpy.save(self.states_b,       path + "buffer_states.pt")
        numpy.save(self.steps_b,        path + "buffer_steps.pt")
        numpy.save(self.visitings_b,    path + "buffer_visitings.pt")

    def load(self, path):
        self.states_b   = numpy.load(path + "buffer_states.pt")
        self.steps_b    = numpy.load(path + "buffer_steps.pt")
        self.visitings_b= numpy.load(path + "buffer_visitings.pt")


    def _add_new_state(self, state, steps):
        if self.current_idx < self.states_b.shape[0]:
            self.states_b[self.current_idx]     = state.clone()
            self.steps_b[self.current_idx]      = steps
            self.visitings_b[self.current_idx]  = 1.0

            self.current_idx+= 1
            

