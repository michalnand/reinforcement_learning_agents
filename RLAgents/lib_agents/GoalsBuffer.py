from numpy.lib.shape_base import expand_dims
import torch
import numpy
import networkx
import cv2
import matplotlib.pyplot as plt


class GoalsBuffer:
    def __init__(self, size, add_threshold, downsample, reached_coeff, visited_coeff, state_shape, envs_count, device = "cpu"):
        self.size           = size
        self.downsample     = downsample
        self.add_threshold  = add_threshold
        self.state_shape    = state_shape
        self.goals_shape    = (1, state_shape[1], state_shape[2]) 
        self.envs_count     = envs_count
        self.device         = device

        self.steps          = 0
        self.warm_up_steps  = 512

        self.reached_coeff  = reached_coeff
        self.visited_coeff  = visited_coeff

    
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

        #visiting count
        self.goals_counter  = numpy.zeros((self.size, ))

        #current goals indices
        self.goals_indices  = numpy.zeros((self.envs_count, ), dtype=int)

        #flag if reached goal
        self.goals_reached  = numpy.ones((self.envs_count, ), dtype=bool)

     
        self.total_goals = 0


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
        self.indices_now     = torch.argmin(distances, dim=1).detach().to("cpu").numpy()
        self.closest_distances  = distances[range(self.states_downsampled.shape[0]), self.indices_now]

        #reward for reached goal
        goals_distances         = (((desired_goals_downsampled - self.states_downsampled)**2.0).sum(dim=1))**0.5 
        
        reached_goals           = (goals_distances <= self.add_threshold).detach().to("cpu").numpy()
        reward_reached_goals    = (1.0 - self.goals_reached)*reached_goals
        reward_visited_goals    = reward_reached_goals*self._visited_rewards()[self.indices_now]

        #reward   = self.goals_ext_reward_ratio*reward_reached_goals + (1.0 - self.goals_ext_reward_ratio)*reward_visited_goals
        reward   = self.reached_coeff*reward_reached_goals + self.visited_coeff*reward_visited_goals

        self.goals_reached      = numpy.logical_or(self.goals_reached, reached_goals)

        '''
        if reward_reached_goals[0] > 0.0:
            print("goal reached = ", reward, "\n\n")
            print(self.goals_counter[0:self.total_goals])
            print(self.goals_rewards[0:self.total_goals])

        if reward_reached_goals[0] > 0.0:
            idx = self.goals_indices[0]
            self._visualise(states_t[0], self.current_goals[0], self.desired_goals[0], self.goals_rewards[idx])
        '''        

        for e in range(self.envs_count):
            if self.goals_reached[e]:
                self.current_goals[e] = torch.zeros(self.goals_shape, device=self.device)
                self.desired_goals[e] = torch.zeros(self.goals_shape, device=self.device)
       

        return self.current_goals, self.desired_goals, reward

    def add(self):
        #add new item if threashold reached
        for i in range(self.envs_count):
            if self.steps > self.warm_up_steps or i == 0:
                if self.closest_distances[i] > self.add_threshold and self.total_goals < self.size:
                    self.goals[self.total_goals]                = self.states_downsampled[i].clone()
                    self.goals_counter[self.total_goals]    = 1

                    self.total_goals = self.total_goals + 1
  
        #update visited counter
        for e in range(self.envs_count):
            self.goals_counter[self.indices_now[e]]+= 1

        self.steps+= 1
            


    def new_goal(self, env_idx):
        #compute target weights
        w   = self._visited_rewards()

        #select only from stored state
        w   = w[0:self.total_goals]

        #convert weights to probs, softmax
        w       = 10.0*w
        w       = w - w.max()
        probs   = numpy.exp(w - w.max())
        probs   = probs/probs.sum() 

        #get random idx, with prob given in w
        idx = numpy.random.choice(range(len(w)), 1, p=probs)[0]

        self.goals_indices[env_idx] = idx
        self.goals_reached[env_idx] = False

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

    def _visited_rewards(self):
        return 1.0 - self.goals_counter/(numpy.max(self.goals_counter) + 0.000000001)

    def _visualise_goal(self, idx):
        
        goal_np = self._upsample(self.goals[idx].unsqueeze(0)).squeeze(0)

        goal_np = goal_np[0].detach().to("cpu").numpy()

        img   = cv2.resize(goal_np, (256, 256), interpolation      = cv2.INTER_NEAREST)
        cv2.imshow("image", img)
        cv2.waitKey(1)

    def _visualise(self, state, current, target, reward):
        state_np    = state[0].detach().to("cpu").numpy()
        current_np  = current[0].detach().to("cpu").numpy()
        target_np   = target[0].detach().to("cpu").numpy()

        reward_str = str(round(reward, 3))

        size = 256

        state_img   = cv2.resize(state_np, (size, size), interpolation      = cv2.INTER_NEAREST)
        current_img = cv2.resize(current_np, (size, size), interpolation    = cv2.INTER_NEAREST)
        target_img  = cv2.resize(target_np, (size, size), interpolation     = cv2.INTER_NEAREST)

        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        cv2.putText(target_img, reward_str,(30, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        img = numpy.hstack((state_img, current_img, target_img))

        cv2.imshow("image", img)
        cv2.waitKey(1)







class GoalsBufferGraph:
    def __init__(self, size, add_threshold, downsample, reached_coeff, visited_coeff, entropy_coeff, state_shape, envs_count, device = "cpu"):
        self.size           = size
        self.downsample     = downsample
        self.add_threshold  = add_threshold
        self.state_shape    = state_shape
        self.goals_shape    = (1, state_shape[1], state_shape[2]) 
        self.envs_count     = envs_count
        self.device         = device

        self.steps          = 0
        self.warm_up_steps  = 512

        self.reached_coeff  = reached_coeff
        self.visited_coeff  = visited_coeff
        self.entropy_coeff  = entropy_coeff

    
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

        #visiting count
        self.goals_counter  = numpy.zeros((self.size, ))

        #current goals indices
        self.goals_indices  = numpy.zeros((self.envs_count, ), dtype=int)

        #flag if reached goal
        self.goals_reached  = numpy.ones((self.envs_count, ), dtype=bool)

        #graph of connections
        self.connections    = numpy.zeros((self.size, self.size), dtype=numpy.float32)

        self.indices_now    = None

        self.total_goals = 0


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


        reward_entropy = self._entropy_rewards()[self.indices_now]
      

        #reward for reached goal
        goals_distances         = (((desired_goals_downsampled - self.states_downsampled)**2.0).sum(dim=1))**0.5 
        
        reached_goals           = (goals_distances <= self.add_threshold).detach().to("cpu").numpy()
        reward_reached_goals    = (1.0 - self.goals_reached)*reached_goals
        reward_visited_goals    = reward_reached_goals*self._visited_rewards()[self.indices_now]

        #reward   = self.goals_ext_reward_ratio*reward_reached_goals + (1.0 - self.goals_ext_reward_ratio)*reward_visited_goals
        reward   = self.reached_coeff*reward_reached_goals + self.visited_coeff*reward_visited_goals + self.entropy_coeff*reward_entropy

        self.goals_reached      = numpy.logical_or(self.goals_reached, reached_goals)

        '''
        if reward_reached_goals[0] > 0.0:
            print("goal reached = ", reward, "\n\n")
            print(self.goals_counter[0:self.total_goals])
            print(self.goals_rewards[0:self.total_goals])

        if reward_reached_goals[0] > 0.0:
            idx = self.goals_indices[0]
            self._visualise(states_t[0], self.current_goals[0], self.desired_goals[0], self.goals_rewards[idx])
        '''        

        for e in range(self.envs_count):
            if self.goals_reached[e]:
                self.current_goals[e] = torch.zeros(self.goals_shape, device=self.device)
                self.desired_goals[e] = torch.zeros(self.goals_shape, device=self.device)
       

        return self.current_goals, self.desired_goals, reward

    def add(self):
        #add new item if threashold reached
        for i in range(self.envs_count):
            if self.steps > self.warm_up_steps or i == 0:
                if self.closest_distances[i] > self.add_threshold and self.total_goals < self.size:
                    self.goals[self.total_goals]                = self.states_downsampled[i].clone()
                    self.goals_counter[self.total_goals]    = 1

                    self.total_goals = self.total_goals + 1
  
        #update visited counter
        for e in range(self.envs_count):
            self.goals_counter[self.indices_now[e]]+= 1

        self.steps+= 1
            


    def new_goal(self, env_idx):
        #compute target weights
        w   = 0*self._visited_rewards() + self._entropy_rewards()


        #select only from stored state
        w   = w[0:self.total_goals]

        #convert weights to probs, softmax
        w       = 10.0*w
        w       = w - w.max()
        probs   = numpy.exp(w - w.max())
        probs   = probs/probs.sum() 

        #get random idx, with prob given in w
        idx = numpy.random.choice(range(len(w)), 1, p=probs)[0]

        self.goals_indices[env_idx] = idx
        self.goals_reached[env_idx] = False


    def save(self, path = "./"):
        print("saving")
        plt.clf()

        z = self.connections
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

    def _visited_rewards(self):
        return 1.0 - self.goals_counter/(numpy.max(self.goals_counter) + 0.000000001)


    def _entropy_rewards(self):
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


    def _visualise_goal(self, idx):
        
        goal_np = self._upsample(self.goals[idx].unsqueeze(0)).squeeze(0)

        goal_np = goal_np[0].detach().to("cpu").numpy()

        img   = cv2.resize(goal_np, (256, 256), interpolation      = cv2.INTER_NEAREST)
        cv2.imshow("image", img)
        cv2.waitKey(1)

    def _visualise(self, state, current, target, reward):
        state_np    = state[0].detach().to("cpu").numpy()
        current_np  = current[0].detach().to("cpu").numpy()
        target_np   = target[0].detach().to("cpu").numpy()

        reward_str = str(round(reward, 3))

        size = 256

        state_img   = cv2.resize(state_np, (size, size), interpolation      = cv2.INTER_NEAREST)
        current_img = cv2.resize(current_np, (size, size), interpolation    = cv2.INTER_NEAREST)
        target_img  = cv2.resize(target_np, (size, size), interpolation     = cv2.INTER_NEAREST)

        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        cv2.putText(target_img, reward_str,(30, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        img = numpy.hstack((state_img, current_img, target_img))

        cv2.imshow("image", img)
        cv2.waitKey(1)

