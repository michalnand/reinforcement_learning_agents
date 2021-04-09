import numpy
import torch
import time

from torch.distributions import Categorical

from .PolicyBufferIME   import *
from .RunningStats      import * 
from .EpisodicMemory    import * 
   
class AgentPPOEntropy:
    def __init__(self, envs, ModelPPO, ModelAutoencoder, Config):
        self.envs = envs

        config = Config.Config() 
   
        self.gamma_ext          = config.gamma_ext
        self.gamma_int          = config.gamma_int
         
        self.ext_adv_coeff                  = config.ext_adv_coeff
        self.int_global_novelty_adv_coeff    = config.int_global_novelty_adv_coeff
        self.int_episodic_novelty_adv_coeff  = config.int_episodic_novelty_adv_coeff
   
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip

        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        
        self.training_epochs    = config.training_epochs
        self.actors             = config.actors

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        self.model_ppo      = ModelPPO.Model(self.state_shape, self.actions_count)
        self.optimizer_ppo  = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)
 
        self.policy_buffer  = PolicyBufferIME(self.steps, self.state_shape, self.actions_count, self.actors, self.model_ppo.device)

        self.global_novelty_memory = EpisodicMemory(config.global_novelty_memory_size)

        self.states = [] 
        self.episodic_novelty_memory = [] 
        for e in range(self.actors):
            self.states.append(self.envs.reset(e))
            self.episodic_novelty_memory.append(EpisodicMemory(config.episodic_novelty_memory_size))

        self.model_autoencoder      = ModelAutoencoder.Model(self.state_shape)
        self.optimizer_autoencoder  = torch.optim.Adam(self.model_autoencoder.parameters(), lr=config.learning_rate_autoencoder)

        self.states_running_stats   = RunningStats(self.state_shape, numpy.array(self.states))
        
        self.enable_training()
        self.iterations = 0

        self.log_loss_autoencoder               = 0.0
        self.log_global_novelty                 = 0.0
        self.log_episodic_novelty               = 0.0
        self.log_advantages                     = 0.0
        self.log_global_novelty_advatages       = 0.0
        self.log_episodic_novelty_advatages     = 0.0


    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False

    def main(self):
        #state to tensor
        states_t            = torch.tensor(self.states, dtype=torch.float).detach().to(self.model_ppo.device)

        #compute model output
        logits_t, values_ext_t, values_global_novelty_t, values_episodic_novelty  = self.model_ppo.forward(states_t)

        states_np                   = states_t.detach().to("cpu").numpy()
        logits_np                   = logits_t.detach().to("cpu").numpy()
        values_ext_np               = values_ext_t.detach().to("cpu").numpy()
        values_global_novelty_np    = values_global_novelty_t.detach().to("cpu").numpy()
        values_episodic_novelty_np  = values_episodic_novelty.detach().to("cpu").numpy()
        
        #collect actions
        actions = []
        for e in range(self.actors):
            actions.append(self._sample_action(logits_t[e]))

        #update long term state mean and variance
        self.states_running_stats.update(states_np)

        #execute action
        states, rewards, dones, _ = self.envs.step(actions)

        state_norm_t    = states_t - torch.from_numpy(self.states_running_stats.mean).to(self.model_autoencoder.device)
        features_t      = self.model_autoencoder.eval_features(state_norm_t).detach()

        #global novelty motivation
        global_novelty_np         = self._global_novelty(features_t, dones)
        global_novelty_np         = numpy.clip(global_novelty_np, -1.0, 1.0)

        #episodic novelty motivation 
        episodic_novelty_np          = self._episodic_novelty(features_t)
        episodic_novelty_np          = numpy.clip(episodic_novelty_np, -1.0, 1.0)

        #put into policy buffer
        for e in range(self.actors):            
            if self.enabled_training:
                self.policy_buffer.add(e, states_np[e], logits_np[e], values_ext_np[e], values_global_novelty_np[e], values_episodic_novelty_np[e], actions[e], rewards[e], global_novelty_np[e], episodic_novelty_np[e], dones[e])

                if self.policy_buffer.is_full():
                    self.train()

            if dones[e]:
                self.states[e] = self.envs.reset(e)
                self._reset_episodic_memory(e, self.states[e])
            else:
                self.states[e] = states[e].copy()

        #collect stats
        k = 0.02
        self.log_global_novelty   = (1.0 - k)*self.log_global_novelty           + k*global_novelty_np.mean()
        self.log_episodic_novelty = (1.0 - k)*self.log_episodic_novelty     + k*episodic_novelty_np.mean()
      
        self.iterations+= 1
        return rewards[0], dones[0]

    
    def save(self, save_path):
        self.model_ppo.save(save_path + "trained/")
        self.model_autoencoder.save(save_path + "trained/")

    def load(self, load_path):
        self.model_ppo.load(load_path + "trained/")
        self.model_autoencoder.load(load_path + "trained/")
 
    def get_log(self):
        result = ""  
        
        result+= str(round(self.log_loss_autoencoder, 7)) + " "      
        result+= str(round(self.log_global_novelty, 7)) + " "        
        result+= str(round(self.log_episodic_novelty, 7)) + " "           
        result+= str(round(self.log_advantages, 7)) + " "         
        result+= str(round(self.log_global_novelty_advatages, 7)) + " "
        result+= str(round(self.log_episodic_novelty_advatages, 7)) + " "  

        return result 
    
    
    def _sample_action(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits, dim = 0)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()

        return action_t.item()  
    
    def train(self): 
        self.policy_buffer.compute_returns(self.gamma_ext, self.gamma_int)

        batch_count = self.steps//self.batch_size

        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                states, logits, values_ext, values_cur, values_ent, actions, rewards, dones, returns_ext, returns_cur, returns_ent, advantages_ext, advantages_cur, advantages_ent = self.policy_buffer.sample_batch(self.batch_size, self.model_ppo.device)

                loss = self._compute_loss(states, logits, actions, returns_ext, returns_cur, returns_ent, advantages_ext, advantages_cur, advantages_ent)

                self.optimizer_ppo.zero_grad()        
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_ppo.parameters(), max_norm=0.5)
                self.optimizer_ppo.step()
                
                if e == 0: 
                    #train autoencoder model, MSE loss
                    state_norm_t            = states - torch.from_numpy(self.states_running_stats.mean).to(self.model_autoencoder.device)
                    state_norm_t            = state_norm_t.detach()
                    state_predicted_t, _    = self.model_autoencoder(state_norm_t)

                    #reconstruction loss
                    loss_autoencoder    = (state_norm_t - state_predicted_t)**2
                    loss_autoencoder    = loss_autoencoder.mean()

                    self.optimizer_autoencoder.zero_grad()
                    loss_autoencoder.backward()
                    self.optimizer_autoencoder.step()

                    k = 0.02
                    self.log_loss_autoencoder   = (1.0 - k)*self.log_loss_autoencoder + k*loss_autoencoder.detach().to("cpu").numpy()


        self.policy_buffer.clear() 

    
    def _compute_loss(self, states, logits, actions,  returns_ext, returns_cur, returns_ent, advantages_ext, advantages_cur, advantages_ent):
        probs_old     = torch.nn.functional.softmax(logits, dim = 1).detach()
        log_probs_old = torch.nn.functional.log_softmax(logits, dim = 1).detach()

        logits_new, values_ext_new, values_int_cur_new, values_int_ent_new  = self.model_ppo.forward(states)

        probs_new     = torch.nn.functional.softmax(logits_new, dim = 1)
        log_probs_new = torch.nn.functional.log_softmax(logits_new, dim = 1)

        '''
        compute external critic loss, as MSE
        L = (T - V(s))^2
        '''
        values_ext_new  = values_ext_new.squeeze(1)
        loss_ext_value  = (returns_ext.detach() - values_ext_new)**2
        loss_ext_value  = loss_ext_value.mean()


        '''
        compute internal global_novelty critic loss, as MSE
        L = (T - V(s))^2
        '''
        values_int_cur_new  = values_int_cur_new.squeeze(1)
        loss_int_cur_value  = (returns_cur.detach() - values_int_cur_new)**2
        loss_int_cur_value  = loss_int_cur_value.mean()


        '''
        compute internal entropy critic loss, as MSE
        L = (T - V(s))^2
        '''
        values_int_ent_new  = values_int_ent_new.squeeze(1)
        loss_int_ent_value  = (returns_ent.detach() - values_int_ent_new)**2
        loss_int_ent_value  = loss_int_ent_value.mean()


        ''' 
        compute actor loss, surrogate loss
        '''
        advantages      = self.ext_adv_coeff*advantages_ext + self.int_global_novelty_adv_coeff*advantages_cur + self.int_episodic_novelty_adv_coeff*advantages_ent
        advantages      = advantages.detach()
        advantages_norm = (advantages - torch.mean(advantages))/(torch.std(advantages) + 1e-10)
        

        log_probs_new_  = log_probs_new[range(len(log_probs_new)), actions]
        log_probs_old_  = log_probs_old[range(len(log_probs_old)), actions]
                        
        ratio       = torch.exp(log_probs_new_ - log_probs_old_)
        p1          = ratio*advantages_norm
        p2          = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages_norm
        loss_policy = -torch.min(p1, p2)  
        loss_policy = loss_policy.mean()
    
        '''
        compute entropy loss, to avoid greedy strategy
        L = beta*H(pi(s)) = beta*pi(s)*log(pi(s))
        '''
        loss_entropy = (probs_new*log_probs_new).sum(dim = 1)
        loss_entropy = self.entropy_beta*loss_entropy.mean()

        loss = loss_ext_value + loss_int_cur_value + loss_int_ent_value + loss_policy + loss_entropy

        k = 0.02
        self.log_advantages             = (1.0 - k)*self.log_advantages             + k*advantages_ext.mean().detach().to("cpu").numpy()
        self.log_global_novelty_advatages    = (1.0 - k)*self.log_global_novelty_advatages    + k*advantages_cur.mean().detach().to("cpu").numpy()
        self.log_episodic_novelty_advatages      = (1.0 - k)*self.log_episodic_novelty_advatages      + k*advantages_ent.mean().detach().to("cpu").numpy()
    
        return loss

    def _action_one_hot(self, action_idx_t):
        batch_size = action_idx_t.shape[0]

        action_one_hot_t = torch.zeros((batch_size, self.actions_count))
        action_one_hot_t[range(batch_size), action_idx_t] = 1.0  
        action_one_hot_t = action_one_hot_t.to(self.model_ppo.device)

        return action_one_hot_t


    def _global_novelty(self, features_t, dones): 
        result = numpy.zeros(self.actors)

        for e in range(self.actors):
            result[e] = self.global_novelty_memory.motivation(features_t[e])

        for e in range(self.actors):
            if numpy.random.rand() < 1.0/self.actors:
                self.global_novelty_memory.add(features_t[e])
        
        return result

    def _reset_episodic_memory(self, env_idx, state_np):
        state_t       = torch.from_numpy(state_np).unsqueeze(0).to(self.model_autoencoder.device).float()

        state_norm_t  = state_t - torch.from_numpy(self.states_running_stats.mean).to(self.model_autoencoder.device)

        features_t    = self.model_autoencoder.eval_features(state_norm_t)
        features_t    = features_t.squeeze(0).detach()
 
        self.episodic_novelty_memory[env_idx].reset(features_t) 
              
    def _episodic_novelty(self, features_t): 
        result = numpy.zeros(self.actors)

        for e in range(self.actors):
            result[e] = self.episodic_novelty_memory[e].motivation(features_t[e])

        for e in range(self.actors):
            self.episodic_novelty_memory[e].add(features_t[e])
        
        return result

    '''
    def visualise(self, states_t):
        state_norm_t  = states_t - torch.from_numpy(self.states_running_stats.mean).to(self.model_autoencoder.device)
    ''' 
