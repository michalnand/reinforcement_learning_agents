import numpy
import torch
import time

from torch.distributions import Categorical
 
from .PolicyBufferIM    import *  
from .StateSampling     import *

from .EpisodicMemory    import *

import cv2

class AgentPPOHierarchyEntropy():   
    def __init__(self, envs, ModelPPO, config):
        self.envs = envs  
   
        self.gamma_ext          = config.gamma_ext
        self.gamma_int          = config.gamma_int
           
        self.ext_adv_coeff      = config.ext_adv_coeff
        self.int_adv_coeff      = config.int_adv_coeff
    
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip 
    
        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        
        self.training_epochs    = config.training_epochs
        self.actors             = config.actors 
        self.entropy_coeff      = config.entropy_coeff

        self.stages_count       = len(config.sampling_indices)

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        self.model_ppo      = ModelPPO.Model(self.state_shape, self.actions_count)
        self.optimizer_ppo  = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)
 

        self.policy_buffer  = PolicyBufferIM(self.steps, (self.stages_count, ) + self.state_shape, self.actions_count, self.actors, self.model_ppo.device)
 
        self.states = numpy.zeros((self.actors, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.actors):
            self.states[e] = self.envs.reset(e).copy()

        self.state_sampling = []
        for e in range(self.stages_count):
            self.state_sampling.append(StateSampling(torch.from_numpy(self.states).to(self.model_ppo.device), config.sampling_indices[e]))


        self.episodic_memory = [] 
        for e in range(self.actors): 
            self.episodic_memory.append(EpisodicMemory(config.episodic_memory_size, 8))
 
        self.enable_training()
        self.iterations                 = 0 

        self.log_entropy                = 0.0
        self.log_advantages             = 0.0
        self.log_entropy_advatages      = 0.0


    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False

    def main(self):
        #state to tensor
        states_t            = torch.from_numpy(self.states).to(self.model_ppo.device)

        #sample states byt filtering
        states_sampled_t    = torch.zeros((self.stages_count, self.actors) + self.state_shape).to(self.model_ppo.device)
        for i in range(self.stages_count):
            states_sampled_t[i] = self.state_sampling[i].add(states_t)

        #compute model output
        logits_t, values_ext_t, values_int_t  = self.model_ppo.forward(states_sampled_t)
        
        logits_np           = logits_t.detach().to("cpu").numpy()
        values_ext_np       = values_ext_t.squeeze(1).detach().to("cpu").numpy()
        values_int_np       = values_int_t.squeeze(1).detach().to("cpu").numpy()

        #collect actions
        actions = self._sample_actions(logits_t)
        
        #execute action
        states, rewards, dones, infos = self.envs.step(actions)

        
        #entropy motivation
        entropy_np      = self.entropy_coeff*self._entropy(states_t)
        entropy_np      = numpy.clip(entropy_np, -1.0, 1.0)
        
        self.states = states.copy()

        #put into policy buffer
        if self.enabled_training:
            states_sampled_tr = torch.transpose(states_sampled_t, 0, 1).detach().to("cpu").numpy()

            self.policy_buffer.add(states_sampled_tr, logits_np, values_ext_np, values_int_np, actions, rewards, entropy_np, dones)

            if self.policy_buffer.is_full():
                self.train()
        
        for e in range(self.actors): 
            if dones[e]:
                self.states[e] = self.envs.reset(e).copy()

                s_new = torch.from_numpy(self.states[e]).to(self.model_ppo.device)
                for i in range(self.stages_count):
                    self.state_sampling[i].reset(e, s_new)

                self.episodic_memory[e].reset(s_new[0])


        #collect stats
        k = 0.02
        self.log_entropy = (1.0 - k)*self.log_entropy + k*entropy_np.mean()

        self.iterations+= 1
        return rewards[0], dones[0], infos[0]
    
    def save(self, save_path):
        self.model_ppo.save(save_path + "trained/")

    def load(self, load_path):
        self.model_ppo.load(load_path + "trained/")

    def get_log(self): 
        result = "" 
        result+= str(round(self.log_entropy, 7)) + " "
        result+= str(round(self.log_advantages, 7)) + " "
        result+= str(round(self.log_entropy_advatages, 7)) + " "
        return result 
    
    def _sample_actions(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits, dim = 1)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()
        actions               = action_t.detach().to("cpu").numpy()
        return actions
    
    def train(self): 
        self.policy_buffer.compute_returns(self.gamma_ext, self.gamma_int)

        batch_count = self.steps//self.batch_size

        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int = self.policy_buffer.sample_batch(self.batch_size, self.model_ppo.device)

                states = torch.transpose(states, 0, 1)

                #train PPO model
                loss = self._compute_loss(states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int)

                self.optimizer_ppo.zero_grad()        
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_ppo.parameters(), max_norm=0.5)
                self.optimizer_ppo.step()

        self.policy_buffer.clear() 

    
    def _compute_loss(self, states, logits, actions,  returns_ext, returns_int, advantages_ext, advantages_int):
        log_probs_old = torch.nn.functional.log_softmax(logits, dim = 1).detach()

        logits_new, values_ext_new, values_int_new  = self.model_ppo.forward(states)

        probs_new     = torch.nn.functional.softmax(logits_new,     dim = 1)
        log_probs_new = torch.nn.functional.log_softmax(logits_new, dim = 1)

        ''' 
        compute external critic loss, as MSE
        L = (T - V(s))^2
        '''
        values_ext_new  = values_ext_new.squeeze(1)
        loss_ext_value  = (returns_ext.detach() - values_ext_new)**2
        loss_ext_value  = loss_ext_value.mean()

        '''
        compute internal critic loss, as MSE
        L = (T - V(s))^2
        '''
        values_int_new  = values_int_new.squeeze(1)
        loss_int_value  = (returns_int.detach() - values_int_new)**2
        loss_int_value  = loss_int_value.mean()
         
        loss_critic     = loss_ext_value + loss_int_value
 
        ''' 
        compute actor loss, surrogate loss
        '''
        advantages      = self.ext_adv_coeff*advantages_ext + self.int_adv_coeff*advantages_int
        advantages      = advantages.detach() 
        
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

        loss = 0.5*loss_critic + loss_policy + loss_entropy

        k = 0.02
        self.log_advantages             = (1.0 - k)*self.log_advantages + k*advantages_ext.mean().detach().to("cpu").numpy()
        self.log_entropy_advatages      = (1.0 - k)*self.log_entropy_advatages + k*advantages_int.mean().detach().to("cpu").numpy()

        return loss 

    def _action_one_hot(self, action_idx_t):
        batch_size = action_idx_t.shape[0]

        action_one_hot_t = torch.zeros((batch_size, self.actions_count))
        action_one_hot_t[range(batch_size), action_idx_t] = 1.0  
        action_one_hot_t = action_one_hot_t.to(self.model_ppo.device)

        return action_one_hot_t

    def _curiosity(self, state_t):
        state_norm_t    = self._norm_state(state_t)

        features_predicted_t, features_target_t = self.model_rnd(state_norm_t)

        curiosity_t     = (features_target_t - features_predicted_t)**2
        
        curiosity_t     = curiosity_t.sum(dim=1)/2.0
        
        return curiosity_t.detach().to("cpu").numpy()

    def _norm_state(self, state_t):
        mean = torch.from_numpy(self.states_running_stats.mean).to(state_t.device).float()

        state_norm_t = state_t - mean

        return state_norm_t

    def _entropy(self, states):
        result = numpy.zeros(states.shape[0])

        for e in range(states.shape[0]):
            result[e] = self.episodic_memory[e].add(states[e][0])

        return result
 