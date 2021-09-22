from numpy.lib.shape_base import expand_dims
import torch
import numpy
import networkx
import cv2
import matplotlib.pyplot as plt


class GoalsBuffer: 
    def __init__(self, size, add_threshold, downsample, goals_weights, state_shape, envs_count, device = "cpu"):
        self.size           = size
        self.add_threshold  = add_threshold
        self.downsample     = downsample
        self.goals_weights  = numpy.array(goals_weights)
        self.state_shape    = state_shape
        self.goals_shape    = (1, state_shape[1], state_shape[2]) 
        self.envs_count     = envs_count
        self.device         = device

        self.steps          = 0
        self.warm_up_steps  = 512
    
        self.layer_downsample = torch.nn.AvgPool2d((self.downsample, self.downsample), (self.downsample, self.downsample))
        self.layer_downsample.to(self.device)

        self.layer_upsample = torch.nn.Upsample(scale_factor=self.downsample, mode="nearest")
        self.layer_upsample.to(self.device)

        self.layer_flatten = torch.nn.Flatten()
        self.layer_flatten.to(self.device)

        
        states_downsampled_shape = (1, self.state_shape[1]//self.downsample, self.state_shape[2]//self.downsample)

        goals_shape         = numpy.prod(states_downsampled_shape)

        
        #downsampled goals
        self.goals          = torch.zeros((self.size, goals_shape), device=self.device)

        #current goals indices
        self.goals_indices  = numpy.zeros((self.envs_count, ), dtype=int)

        #flag if reached goal
        self.goals_reached  = numpy.ones((self.envs_count, ), dtype=bool)

        #graph of connections
        self.connections    = numpy.zeros((self.size, self.size), dtype=numpy.float32)

        #state score
        self.score_sum      = numpy.zeros((self.size, ), dtype=numpy.float32)

        self.indices_now    = None

        self.total_goals = 0

        self.save_idx   = 0


    def get(self, states_t):        
        self.states_downsampled = self._downsmaple(states_t[:,0].unsqueeze(1))

        #add first goal if buffer empty
        if self.total_goals == 0:
            self.goals[0]       = self.states_downsampled[0].clone()
            self.total_goals    = 1


        desired_goals_downsampled   = self.goals[self.goals_indices]

        self.current_goals          = self._upsample(self.states_downsampled)
        self.desired_goals          = self._upsample(desired_goals_downsampled)
        
        #states_t distances from buffer
        distances = torch.cdist(self.states_downsampled, self.goals)
    
       
        #find closest
        if self.indices_now is None: 
            self.indices_now    = torch.argmin(distances, dim=1).detach().to("cpu").numpy()
            self.indices_prev   = self.indices_now.copy()
        else:
            self.indices_prev   = self.indices_now.copy()
            self.indices_now    = torch.argmin(distances, dim=1).detach().to("cpu").numpy()

        self.closest_distances  = distances[range(self.states_downsampled.shape[0]), self.indices_now]

        #update graph
        #print(">>> ", self.connections.shape, self.indices_prev.shape, self.indices_now.shape)
        self.connections[self.indices_prev, self.indices_now]+= 1


        #external reward for reached goal
        goals_distances         = (((desired_goals_downsampled - self.states_downsampled)**2.0).sum(dim=1))**0.5 
        reached_goals           = (goals_distances <= self.add_threshold).detach().to("cpu").numpy()
        #clear if already reached
        reward_ext              = (1.0 - self.goals_reached)*reached_goals

        #internal reward, different motivations   
        reward_int = self._reward_int(self.indices_now)
        #clear if already reached
        reward_int = (1.0 - self.goals_reached)*reward_int

        #set reached goal flag
        self.goals_reached      = numpy.logical_or(self.goals_reached, reached_goals)

      
        #clear already reached goals
        for e in range(self.envs_count):
            if self.goals_reached[e]:
                self.desired_goals[e] = torch.zeros(self.goals_shape, device=self.device)


        #self._visualise(states_t[0], self.desired_goals[0], self.goals_indices[0], reward_ext[0], reward_int[0])

        return self.desired_goals, reward_ext, reward_int

    def add(self, rewards_sum):
        #add new item if threashold reached
        for e in range(self.envs_count):
            if self.steps > self.warm_up_steps or e == 0:
                if self.closest_distances[e] > self.add_threshold and self.total_goals < self.size:
                    self.goals[self.total_goals]                = self.states_downsampled[e].clone()
                    self.total_goals = self.total_goals + 1
                
                #update state score sum of bigger
                idx = self.indices_now[e]
                if self.score_sum[idx] < rewards_sum[e]:
                    self.score_sum[idx] = rewards_sum[e]

        self.steps+= 1
            


    def new_goal(self, env_idx):
        #compute target weights using internal reward
        #select only from stored states
        w  = self._reward_int(range(self.total_goals))

        #convert weights to probs, use softmax
        w       = 10.0*w
        w       = w - w.max()
        probs   = numpy.exp(w - w.max())
        probs   = probs/probs.sum() 

        #get random idx, with prob given in w
        idx = numpy.random.choice(range(len(w)), 1, p=probs)[0]

        self.goals_indices[env_idx] = idx
        self.goals_reached[env_idx] = False

        return idx 

    def zero_goal(self, env_idx):
        self.goals_indices[env_idx] = 0
        self.goals_reached[env_idx] = True


    def save(self, path = "./"):
        plt.clf()

        z = self.connections
        G = networkx.from_numpy_matrix(numpy.array(z), create_using=networkx.MultiDiGraph())
        G.remove_nodes_from(list(networkx.isolates(G)))

        pos = networkx.spring_layout(G, seed=1)
        #pos = networkx.kamada_kawai_layout(G)

        networkx.draw_networkx_nodes(G, pos, node_size = 4)
        networkx.draw_networkx_edges(G, pos, arrows = False)

        plt.savefig(path + "result/graph/" + "graph.png", dpi=300)

        f = open(path + "result/graph/" + "graph_" + str(self.save_idx) + ".npy", "wb")
        numpy.save(f, z)  
 
        reward_int = self._reward_int(range(self.size))
        f = open(path + "result/graph/" + "reward_int_" + str(self.save_idx) + ".npy", "wb")
        numpy.save(f, reward_int) 

        visited_count = self.connections.sum(axis=1)
        f = open(path + "result/graph/" + "visited_count_" + str(self.save_idx) + ".npy", "wb")
        numpy.save(f, visited_count) 

        self.save_idx+= 1
 

    #downsample and flatten
    def _downsmaple(self, states_t, quant_levels = 8):
        y = self.layer_downsample(states_t)
        y = self.layer_flatten(y)
        y = torch.round(y*quant_levels)/quant_levels
        return y 

    #upsample and reshape
    def _upsample(self, x):
        h = self.state_shape[1]//self.downsample
        w = self.state_shape[2]//self.downsample
        
        x = x.reshape((x.shape[0], 1, h, w))

        y = self.layer_upsample(x)

        return y

    def _reward_int(self, indices): 
        reward_int_score    = self.goals_weights[0]*self._reward_int_score()[indices]
        reward_int_visited  = self.goals_weights[1]*self._reward_int_visited()[indices]
        reward_int_entropy  = self.goals_weights[2]*self._reward_int_entropy()[indices]
        
        reward_int = reward_int_score + reward_int_visited + reward_int_entropy
        
        return reward_int

    #score count rewards, max value = 1, for highest reward state
    def _reward_int_score(self):
        eps  = 0.000001
        return self.score_sum/(self.score_sum.max() + eps)
    
    #visited count rewards, max value = 1, for zero visited state
    def _reward_int_visited(self):
        eps             = 0.000001
        visited_counts  = self.connections.sum(axis=1)

        return 1.0 - visited_counts/(visited_counts.max() + eps)

    #state entropy rewards, max value = 1, for state with most unbalanced connections
    def _reward_int_entropy(self):
        eps             = 0.000001
        counts          = self.connections
        
        #maximum possible entropy of state
        maximum_entropy = (counts > 0.0).sum(axis=1) + eps

        #real state entropy 
        counts_probs   = counts/(numpy.expand_dims(numpy.sum(counts, axis=1), 1) + eps)
        entropy        = -counts_probs*numpy.log2(counts_probs + eps) 
        entropy        = numpy.sum(entropy, axis=1)

        #motivation, how close to maximum possible entropy, also prefer less visited states
        return 1.0 - entropy/maximum_entropy

    #TODO
    #reward for shortest graph path distance
    def _reward_int_distance(self):
        #distances = networkx.algorithms.shortest_paths.generic.shortest_path(G, source = , target = )
        pass

    
    def _visualise(self, state, goal, goal_id, reward_ext, reward_int):
        state_np  = state[0].detach().to("cpu").numpy()
        goal_np   = goal[0].detach().to("cpu").numpy()

        goal_id_str    = str(goal_id)
        reward_ext_str = str(round(reward_ext, 3))
        reward_int_str = str(round(reward_int, 3))

        size = 256

        state_img   = cv2.resize(state_np, (size, size), interpolation      = cv2.INTER_NEAREST)
        goal_img    = cv2.resize(goal_np, (size, size), interpolation     = cv2.INTER_NEAREST)

        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        cv2.putText(goal_img, goal_id_str,   (30, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(goal_img, reward_ext_str,(30, 60), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(goal_img, reward_int_str,(30, 90), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        img = numpy.hstack((state_img, goal_img))

        cv2.imshow("image", img)
        cv2.waitKey(1)

        print(">>> total goals ", self.total_goals)

        w  = self._reward_int(range(self.total_goals))
        print(">>> goals weights = ", w)


