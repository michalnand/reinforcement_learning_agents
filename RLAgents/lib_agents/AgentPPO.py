import numpy
import torch
import time

from .ValuesLogger      import *
from .TrajectoryBuffer  import *
from .SelfSupervised    import * 
from .Augmentations     import *
  
class AgentPPO():
    def __init__(self, envs, Model, config):
        self.envs = envs
 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma              = config.gamma
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip

        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        
        self.training_epochs    = config.training_epochs
        self.envs_count         = config.envs_count
 
        self.rnn_policy         = config.rnn_policy

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n
 
        if config.self_supervised_loss is not None:
            self.self_supervised_loss       = config.self_supervised_loss
            self.max_similar_state_distance = config.max_similar_state_distance
            self.augmentations              = config.augmentations
        else:
            self.self_supervised_loss   = None

        self.model = Model.Model(self.state_shape, self.actions_count)
        self.model.to(self.device)
        print(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
 
        self.policy_buffer = TrajectoryBuffer(self.steps, self.state_shape, self.actions_count, self.envs_count)
 
        
        if self.rnn_policy:
            self.hidden_state = torch.zeros((self.envs_count, self.model.rnn_size), dtype=torch.float32, device=self.device)
        else:
            self.hidden_state = torch.zeros((self.envs_count, 128), dtype=torch.float32, device=self.device)


        print("gamma                    = ", self.gamma)
        print("entropy_beta             = ", self.entropy_beta)
        print("learning_rate            = ", config.learning_rate)
        print("batch_size               = ", self.batch_size)
        print("rnn_policy               = ", self.rnn_policy)
        print("self_supervised_loss     = ", self.self_supervised_loss)
        if self.self_supervised_loss is not None:
            print("max_similar_state_distance = ", self.max_similar_state_distance)
            print("augmentations              = ", self.augmentations)
        print("\n\n")


        self.states_t = torch.zeros((self.envs_count, ) + self.state_shape, dtype=torch.float32)
        self.logits_t = torch.zeros((self.envs_count, self.actions_count), dtype=torch.float32)
        self.values_t = torch.zeros((self.envs_count, ) , dtype=torch.float32)
        self.actions_t = torch.zeros((self.envs_count, ) , dtype=int)
        self.rewards_t = torch.zeros((self.envs_count, ) , dtype=torch.float32)
        self.dones_t   = torch.zeros((self.envs_count, ) , dtype=torch.float32)

        if self.rnn_policy:
            self.hidden_state_t = torch.zeros((self.envs_count, self.model.rnn_size) , dtype=torch.float32)
        else:
            self.hidden_state_t = torch.zeros((self.envs_count,128) , dtype=torch.float32)

      
        self.iterations = 0   

        self.values_logger  = ValuesLogger()
        self.values_logger.add("loss_actor", 0.0)
        self.values_logger.add("loss_critic", 0.0)
        
    def round_start(self): 
        pass

    def round_finish(self): 
        pass
        
    def episode_done(self, env_idx):
        pass

    def step(self, states, training_enabled, legal_actions_mask):        
        states_t  = torch.tensor(states, dtype=torch.float).detach().to(self.device)

        if self.rnn_policy: 
            logits_t, values_t, hidden_state_new  = self.model.forward(states_t, self.hidden_state)
        else:
            logits_t, values_t  = self.model.forward(states_t)

        actions = self._sample_actions(logits_t, legal_actions_mask)

        states_new, rewards, dones, _, infos = self.envs.step(actions)

        
        #put into policy buffer
        if training_enabled:
            states_t        = states_t.detach().to("cpu")
            logits_t        = logits_t.detach().to("cpu")
            values_t        = values_t.squeeze(1).detach().to("cpu") 
            actions         = torch.from_numpy(actions).to("cpu")
            rewards_t       = torch.from_numpy(rewards).to("cpu")
            dones           = torch.from_numpy(dones).to("cpu")
            hidden_state_t  = self.hidden_state_t.detach().to("cpu")

            self.policy_buffer.add(states_t, logits_t, values_t, actions, rewards_t, dones, hidden_state_t)

            if self.policy_buffer.is_full():
                self.train()

        #udpate rnn hiddens tate
        if self.rnn_policy:
            self.hidden_state = hidden_state_new.detach().clone()

            dones_idx = numpy.where(dones)

            #clear rnn hidden state if done
            for e in dones_idx:
                self.hidden_state[e] = torch.zeros(self.hidden_state.shape[1], dtype=torch.float32, device=self.device)
        
       

        self.iterations+= 1
        return states_new, rewards, dones, infos
    
    def save(self, save_path):
        torch.save(self.model.state_dict(), save_path + "trained/model.pt")

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path + "trained/model.pt", map_location = self.device))


    def get_log(self):
        return self.values_logger.get_str()
    
    def _sample_actions(self, logits, legal_actions_mask):
        if legal_actions_mask is not None:
            legal_actions_mask_t  = torch.from_numpy(legal_actions_mask).to(self.device).float()
        else:
            legal_actions_mask_t = 1.0

        action_probs_t        = torch.nn.functional.softmax(logits, dim = 1)
        
        #keep only legal actions probs, and renormalise probs
        action_probs_t        = action_probs_t*legal_actions_mask_t
        action_probs_t        = action_probs_t/action_probs_t.sum(dim=-1).unsqueeze(1)

        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        
        action_t              = action_distribution_t.sample()
        actions               = action_t.detach().to("cpu").numpy()
        
        return actions
    
    def train(self): 
        self.policy_buffer.compute_returns(self.gamma)

        samples_count = self.steps*self.envs_count
        batch_count = samples_count//self.batch_size

        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                states, logits, actions, returns, advantages, hidden_state = self.policy_buffer.sample_batch(self.batch_size, self.device)

                loss_ppo = self._loss_ppo(states, logits, actions, returns, advantages, hidden_state)

                if self.self_supervised_loss is not None:
                    states_a, states_b = self.policy_buffer.sample_states_action_pairs(self.batch_size//self.training_epochs, self.device, self.max_similar_state_distance)
                    loss_self_supervised = self._loss_self_supervised(states_a, states_b)
                else:
                    loss_self_supervised = 0    

                loss = loss_ppo + loss_self_supervised

                self.optimizer.zero_grad()        
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step() 

        self.policy_buffer.clear()   

    
    def _loss_ppo(self, states, logits, actions, returns, advantages, hidden_state):
        log_probs_old = torch.nn.functional.log_softmax(logits, dim = 1).detach()

        if self.rnn_policy: 
            logits_new, values_new, _ = self.model.forward(states, hidden_state)
        else:
            logits_new, values_new    = self.model.forward(states)

        probs_new     = torch.nn.functional.softmax(logits_new,     dim = 1)
        log_probs_new = torch.nn.functional.log_softmax(logits_new, dim = 1)


        '''
        compute critic loss, as MSE
        L = (T - V(s))^2
        '''
        values_new = values_new.squeeze(1)
        loss_value = (returns.detach() - values_new)**2
        loss_value = loss_value.mean()

        ''' 
        compute actor loss, surrogate loss
        '''
        advantages       = advantages.detach() 
        #this normalisation has no effect
        #advantages  = (advantages - torch.mean(advantages))/(torch.std(advantages) + 1e-10)

        log_probs_new_  = log_probs_new[range(len(log_probs_new)), actions]
        log_probs_old_  = log_probs_old[range(len(log_probs_old)), actions]
                        
        ratio       = torch.exp(log_probs_new_ - log_probs_old_)
        p1          = ratio*advantages
        p2          = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages
        loss_policy = -torch.min(p1, p2)  
        loss_policy = loss_policy.mean()  
    
        '''
        compute entropy loss, to avoid greedy strategy
        L = beta*H(pi(s)) = beta*pi(s)*log(pi(s))
        '''
        loss_entropy = (probs_new*log_probs_new).sum(dim = 1)
        loss_entropy = self.entropy_beta*loss_entropy.mean()

        loss = loss_value + loss_policy + loss_entropy

        self.values_logger.add("loss_actor",  loss_policy.detach().to("cpu").numpy())
        self.values_logger.add("loss_critic", loss_value.detach().to("cpu").numpy())

        return loss

    def _loss_self_supervised(self, states_now, states_similar):
        states_a = self._augmentations(states_now)
        states_b = self._augmentations(states_similar)

        za = self.model.forward_self_supervised(states_a)
        zb = self.model.forward_self_supervised(states_b)

        return loss_vicreg_direct(za, zb) 

    
    def _augmentations(self, x, p = 0.5): 
        if "random_filter" in self.augmentations:
            x = aug_random_apply(x, p, aug_conv)

        if "noise" in self.augmentations:
            x = aug_random_apply(x, p, aug_noise)
        
        if "random_tiles" in self.augmentations:
            x = aug_random_apply(x, p, aug_random_tiles)

        if "inverse" in self.augmentations:
            x = aug_random_apply(x, p, aug_inverse)

        if "permutation" in self.augmentations:
            x = aug_random_apply(x, p, aug_permutation)

        return x.detach()  
    
    