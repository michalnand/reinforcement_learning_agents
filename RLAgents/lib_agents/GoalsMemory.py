from numpy.core.fromnumeric import var
from numpy.core.numeric import indices
import torch
import numpy
import networkx
import matplotlib.pyplot as plt


 
class GoalsMemoryNovelty:
    def __init__(self, size, downsample = -1, add_threshold = 0.9, alpha = 0.01, epsilon = 0.0001, device = "cpu"):
        self.size       = size
        self.downsample = downsample
        
        self.add_threshold      = add_threshold
        self.alpha              = alpha
        self.epsilon            = numpy.log(epsilon)

        self.device     = device

        self.buffer     = None
 
        if downsample > 1:
            self.layer_downsample = torch.nn.AvgPool2d((self.downsample, self.downsample), (self.downsample//2, self.downsample//2))
            self.layer_downsample.to(self.device)

        self.layer_flatten    = torch.nn.Flatten()
        self.layer_flatten.to(self.device)


    def process(self, states_t, steps_t):
        tmp_t = self._preprocess(states_t)

        #create buffer if not created yet
        if self.buffer is None:
            self.buffer = torch.zeros((self.size, tmp_t.shape[1])).float().to(self.device)
            self.steps  = torch.ones((self.size, )).to(self.device)
            self.total_targets = 0

        #states_t distances from buffer
        distances = torch.cdist(tmp_t, self.buffer)
        
        #find closest
        indices   = torch.argmin(distances, dim=1)


        closest   = distances[range(tmp_t.shape[0]), indices]

        #smooth update stored distances
        self.steps[indices]  = (1.0 - self.alpha)*self.steps[indices]   + self.alpha*(steps_t.float() + 1)
        
        #compute motivation
        motivation_t = torch.exp(steps_t*self.epsilon/self.steps[indices])

        #add new item on random place if threashold reached
        for i in range(tmp_t.shape[0]):
            if closest[i] > self.add_threshold:
                idx = numpy.random.randint(0, self.size - 1)
                self.buffer[idx] = tmp_t[i].clone()
                self.steps[idx]  = steps_t[i].clone()

                self.total_targets+= 1

        return motivation_t

    
    #downsample and flatten
    def _preprocess(self, x):
        if self.layer_downsample is not None:
            y = self.layer_downsample(x)
        
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
            self.layer_downsample = torch.nn.AvgPool2d((self.downsample, self.downsample), (self.downsample//2, self.downsample//2))
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
                self.total_targets = (self.total_targets + 1)%self.size

        #regularisation
        self.connections = torch.nn.functional.hardshrink(self.connections*self.decay, 0.1)

        #count active edges
        self.active_edges = int((self.connections > 0).sum().detach().to("cpu").numpy())
          
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


   