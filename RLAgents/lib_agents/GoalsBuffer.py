import torch
import numpy

class GoalsBuffer: 

    def __init__(self, buffer_size, state_shape, add_threshold = 0.1, downsample = 8, uint8_storage = False):

        self.buffer_size        = buffer_size
        self.state_shape        = state_shape
        self.add_threshold      = add_threshold
        self.downsample         = downsample

        self.goal_shape         = (1, state_shape[1]//self.downsample, state_shape[2]//self.downsample)

        self.uint8_storage  = uint8_storage

        if self.uint8_storage:
            self.scale  = 255
        else:
            self.scale  = 1 

        count                   = self.goal_shape[0]*self.goal_shape[1]*self.goal_shape[2]

        #init buffers
        if self.uint8_storage:
            self.states             = numpy.zeros((self.buffer_size, ) + self.state_shape, dtype=numpy.ubyte)
        else:
            self.states             = numpy.zeros((self.buffer_size, ) + self.state_shape, dtype=numpy.float32)
 
        self.goals              = torch.zeros((self.buffer_size, count))
        self.actions            = [[] for _ in range(self.buffer_size)]
        self.scores_ext         = numpy.zeros((self.buffer_size))
        self.scores_int         = numpy.zeros((self.buffer_size))
        self.visited            = numpy.zeros((self.buffer_size))
 
        #downsampling model
        self.layers = [
            torch.nn.AvgPool2d(downsample, downsample),
            torch.nn.Flatten()
        ]

        self.model = torch.nn.Sequential(*self.layers)
        self.model.eval() 

        self.goals_ptr = 0

    def add(self, states_t, scores_ext, scores_int, actions):
        states_np       = states_t.detach().to("cpu").numpy()

        x               = states_t[:,0,:,:].unsqueeze(1).to("cpu")
        downsampled     = self.model(x)


        #measure distances, resulted matrix shape = (batch_size, buffer_size)
        distances       = torch.cdist(downsampled, self.goals)   

        #find closest values and indices
        #resulted shapes shape = (batch_size)
        closest_values, closest_indices  = torch.min(distances, dim=1)  

        #process whole batch
        for i in range(x.shape[0]):

            #update existing goal record
            if closest_values[i] < self.add_threshold:
                self._add_existing(i, closest_indices, actions, scores_ext, scores_int)
            
            #add new goal
            else:
                self._add_new(i, states_np, downsampled, actions, scores_ext, scores_int)

            #initial runs, add only from first batch item
            if self.goals_ptr < 128:
                break

    def goals_reached(self, states_t, goals_t):
        x               = states_t[:,0,:,:].unsqueeze(1).to("cpu")
        downsampled     = self.model(x)
        goals           = goals_t.reshape(downsampled.shape)

        distances       = torch.cdist(downsampled, goals)   

        goals_reached = numpy.zeros(x.shape[0], dtype=bool)
        for i in range(x.shape[0]): 
            if distances[i][i] < self.add_threshold:
                goals_reached[i] = True

        return goals_reached
    
    #add new goal into buffer if still space
    def _add_new(self, batch_i, states, downsampled, actions, scores_ext, scores_int):
        if self.goals_ptr < self.buffer_size:
            self.states[self.goals_ptr]     = states[batch_i].copy()
            self.goals[self.goals_ptr]      = downsampled[batch_i].clone()
            self.actions[self.goals_ptr]    = actions[batch_i].copy()
            self.scores_ext[self.goals_ptr] = scores_ext[batch_i]
            self.scores_int[self.goals_ptr] = scores_int[batch_i]
            self.visited[self.goals_ptr]    = 1

            self.goals_ptr = self.goals_ptr + 1 

    #update goal to closest stored goal
    def _add_existing(self, batch_i, closest_indices, actions, scores_ext, scores_int):

        idx = closest_indices[batch_i]

        #add shorter actions sequence leading to this target
        if len(self.actions[idx]) > len(actions[batch_i]) or len(self.actions[idx]) == 0:
            self.actions[idx] = actions[batch_i].copy()

        #add better external score
        if self.scores_ext[idx] < scores_ext[batch_i]:
            self.scores_ext[idx] = scores_ext[batch_i]

        #update internal score
        self.scores_int[idx] = scores_int[batch_i]

        #increment visited counter
        self.visited[idx]+= 1


    def sample_batch(self, batch_size, device):
        indices = numpy.random.randint(0, self.goals_ptr, size=batch_size)

        states  = torch.gather(self.states, 0, indices).to(device).float()/self.scale

        goals   = torch.gather(self.goals, 0, indices).to(device).float()
        goals   = goals.reshape((batch_size, ) + self.goal_shape)

        actions = numpy.zeros(batch_size)
        for i in range(batch_size):
            #take last action leading to goal
            actions[i] = self.actions[indices[i]][-1]

        actions = torch.from_numpy(actions).to(device).float()

        return states, goals, actions


    def get_goal(self, ext_weight, int_weight, visited_weight):

        visited = 1.0/((self.visited**0.5) + 1.0)
        values  = ext_weight*self.scores_ext + int_weight*self.scores_int + visited_weight*visited
        values  = values[0:self.goals_ptr]

        tmp     = numpy.exp(values - values.max())
        probs   = tmp/tmp.sum()

        idx     = numpy.random.choice(range(self.goals_ptr), 1, p=probs)[0]

        goal    = self.goals[idx].reshape(self.goal_shape)
        actions = self.actions[idx]
 
        return goal, actions
