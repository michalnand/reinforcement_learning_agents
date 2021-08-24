from numpy.core.fromnumeric import var
from numpy.core.numeric import indices
import torch
import numpy
import networkx
import matplotlib.pyplot as plt

#import cv2


class GoalsMemoryNovelty:
    def __init__(self, size, downsample, add_threshold = 1.0, alpha = 0.1, device = "cpu"):
        self.size               = size
        self.downsample         = downsample
        self.add_threshold      = add_threshold
        self.alpha              = alpha
        self.device             = device

        self.total_targets      = 0
        self.active_edges       = 0

        
        self.layer_downsample = torch.nn.AvgPool2d((self.downsample, self.downsample), (self.downsample, self.downsample))
        self.layer_downsample.to(self.device)

        self.layer_upsample = torch.nn.Upsample(scale_factor=self.downsample, mode="nearest")
        self.layer_upsample.to(self.device)

        self.layer_flatten = torch.nn.Flatten()
        self.layer_flatten.to(self.device)

        self.states             = None
        self.visits             = numpy.zeros((self.size, ))
        self.steps              = numpy.ones((self.size, ))
        self.reward_ext         = numpy.zeros((self.size, ))
        self.reward_int         = numpy.zeros((self.size, ))

    def step(self, states_t, goals_t, steps_np, rewards_ext_np, rewards_int_np):
        reward_visits, reward_faster = self._add(states_t, steps_np, rewards_ext_np, rewards_int_np)

        reached_goals   = self._reached_goals(states_t, goals_t)

        reward_reached  = 1.0*reached_goals

        #keep previous goals
        new_goals_t     = goals_t.clone()

        #clear goals if reached
        for i in range(len(reached_goals)):
            if reached_goals[i] == True:
                new_goals_t[i] = torch.zeros((states_t[2], states_t[3])).to(states_t.device)

        return new_goals_t, reward_visits, reward_faster, reward_reached


    def _add(self, states_t, steps_np, rewards_ext_np, rewards_int_np):
        tmp_t = self._preprocess(states_t[:,0])

        #create buffer if not created yet
        if self.states is None:
            self.state_shape    = states_t.shape[1:]
            self.states         = torch.zeros((self.size, ) + tmp_t.shape[1:]).float().to(self.device)
           
            self.states[self.total_targets] = tmp_t[0].clone()
            self.total_targets+= 1

            self.total_targets  = 0


        #states_t distances from buffer
        distances = torch.cdist(tmp_t, self.buffer)
        
        #find closest
        indices   = torch.argmin(distances, dim=1)
        closest   = distances[range(tmp_t.shape[0]), indices]

        #add new item on random place if threashold reached
        for i in range(tmp_t.shape[0]):
            if closest[i] > self.add_threshold:
                self.states[self.total_targets]     = tmp_t[i].clone()
                self.steps[self.total_targets]      = steps_np[i].clone()
                self.reward_ext[self.total_targets] = rewards_ext_np[i].clone()
                self.reward_int[self.total_targets] = rewards_int_np[i].clone()

                self.total_targets+= 1


        #states_t distances from buffer
        distances = torch.cdist(tmp_t, self.buffer)
        
        #find closest
        indices   = torch.argmin(distances, dim=1)

        #increment visits count
        self.visits[indices]+= 1


        eps = 0.0000001

        #less visiting reward
        rewards_visits = 1.0/(self.visits[indices] + eps)

        #less steps to goal reward
        faster             = (steps_np).astype(int) < (self.steps[indices]).astype(int)
        rewards_faster     = 1.0*faster

        #smooth update steps
        self.steps[indices] = (1.0 - self.alpha)*self.steps[indices] + self.alpha*steps_np

        #update if better external reward
        self.reward_ext[indices] = numpy.max(self.reward_ext[indices], rewards_ext_np)

        #smooth update internal reward
        self.reward_int[indices] = (1.0 - self.alpha)*self.reward_int[indices] + self.alpha*rewards_int_np
  

        return rewards_visits, rewards_faster

    def _reached_goals(self, states_t, goals_t):
        states_tmp  = self._preprocess(states_t[:,0])
        goals_tmp   = self._preprocess(goals_t[:,0])

        distances   = (((goals_tmp - states_tmp)**2.0).sum(dim=1))**0.5
        
        result      = (distances <= self.add_threshold).detach().to("cpu").numpy()

        return result
      

    def get_goal(self, shape, ext_reward_weight = 1.0):
        #compute target eights
        w   = ext_reward_weight*self.reward_ext + (1.0 - ext_reward_weight)*self.reward_int
        
        #select only from stored state
        w   = w[0:self.total_targets]

        #convert to probs, softmax
        probs   = numpy.exp(w - w.max())
        probs   = probs/probs.sum() 

        #get random idx, with prob given in w
        idx = numpy.random.choice(range(len(w)), 1, p=probs)[0]

        y   = self.buffer[idx]

        y = y.reshape((1, self.state_shape[1]//self.downsample, self.state_shape[2]//self.downsample))
        y = self.layer_upsample(y)
        
        return y

    def state_to_goal(self, states_t):
        y = self.layer_downsample(states_t[:, 0].unsqeeze(1))
        y = self.layer_upsample(y)
        return y
      

    #downsample and flatten
    def _preprocess(self, states_t):
        y = self.layer_downsample(states_t)
        y = self.layer_flatten(y)
        return y 





class GoalsMemoryGraph:
    def __init__(self, size, downsample = -1, add_threshold = 1.0, decay = 0.999, device = "cpu"):
        self.size               = size
        self.downsample         = downsample
        self.add_threshold      = add_threshold
        self.decay              = decay
        self.device             = device

        self.total_targets      = 0
        self.active_edges       = 0
 
        self.connections        = torch.zeros((self.size, self.size), dtype=torch.float32).to(self.device)

        self.buffer             = None
        self.indices            = None
  
        if downsample > 1:
            self.layer_downsample = torch.nn.AvgPool2d((self.downsample, self.downsample), (self.downsample, self.downsample))
            self.layer_downsample.to(self.device)

        self.layer_flatten    = torch.nn.Flatten()
        self.layer_flatten.to(self.device)


    def process(self, states_t):

        tmp_t = self._preprocess(states_t)

        #create buffer if not created yet
        if self.buffer is None:
            self.buffer = torch.zeros((self.size, tmp_t.shape[1])).float().to(self.device)
            
            self.buffer[self.total_targets] = tmp_t[0].clone()
            self.total_targets+= 1

        #states_t distances from buffer
        distances = torch.cdist(tmp_t, self.buffer)
        
        #find closest
        if self.indices is None:
            self.indices_prev  = torch.argmin(distances, dim=1)
            self.indices       = self.indices_prev.clone()
        else:
            self.indices_prev  = self.indices.clone()
            self.indices       = torch.argmin(distances, dim=1)

        #closest values
        closest   = distances[range(tmp_t.shape[0]), self.indices]
        
        self.connections[self.indices_prev, self.indices]+= 1
        self.connections[self.indices, self.indices_prev]+= 1

        eps             = 0.000001

        counts          = self.connections[self.indices]
        
        #maximum possible entropy of state
        maximum_entropy = (counts > 0.0).sum(dim=1) + eps

        #real state entropy
        counts_probs   = counts/(torch.sum(counts, dim=1).unsqueeze(1) + eps)
        entropy        = -counts_probs*torch.log2(counts_probs + eps) 
        entropy        = torch.sum(entropy, dim=1)

        #motivation, how close to maximum possible entropy, also prefer less visited states
        motivation = (1.0 - entropy/maximum_entropy)*1.0/(torch.sum(counts, dim=1) + eps)

        #add new item if threashold reached
        for i in range(tmp_t.shape[0]):
            if closest[i] > self.add_threshold:
                self.buffer[self.total_targets] = tmp_t[i].clone()
                #img = self.buffer[self.total_targets].detach().to("cpu").numpy()

                self.total_targets = (self.total_targets + 1)%self.size
                
        #regularisation
        self.connections = torch.nn.functional.hardshrink(self.connections*self.decay, 0.1)

        #count active edges
        self.active_edges = int((self.connections > 0).sum().detach().to("cpu").numpy())
        
        '''
        img = numpy.reshape(img, (12, 12))
        img = cv2.resize(img, (96, 96), interpolation = cv2.INTER_NEAREST)
        cv2.imshow("image", img)
        cv2.waitKey(1)
        '''

        return motivation


    def save(self, path = "./"):
        print("saving")
        plt.clf()

        z = self.connections.detach().to("cpu").numpy()
        G = networkx.from_numpy_matrix(numpy.array(z), create_using=networkx.MultiDiGraph())
        G.remove_nodes_from(list(networkx.isolates(G)))

        pos = networkx.spring_layout(G, seed=1)
        #pos = networkx.kamada_kawai_layout(G)

        networkx.draw_networkx_nodes(G, pos, node_size = 4)
        networkx.draw_networkx_edges(G, pos, arrows = False)

        plt.savefig(path + "graph.png", dpi=300)

        f = open(path + "graph.npy", "wb")
        numpy.save(f, z)
 
    
    #downsample and flatten
    def _preprocess(self, x):
        if self.layer_downsample is not None:
            y = self.layer_downsample(x)
        
        y = self.layer_flatten(y)
        return y 


   