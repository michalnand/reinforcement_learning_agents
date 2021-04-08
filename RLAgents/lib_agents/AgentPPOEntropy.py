import numpy
import torch
import time

from torch.distributions import Categorical

from .PolicyBufferIME   import *
from .RunningStats      import * 
from .EpisodicMemory    import * 
  
class AgentPPOEntropy():
    def __init__(self, envs, ModelPPO, ModelForward, ModelForwardTarget, ModelAutoencoder, Config):
        self.envs = envs

        config = Config.Config() 
   
        self.gamma_ext          = config.gamma_ext
        self.gamma_int          = config.gamma_int
         
        self.ext_adv_coeff      = config.ext_adv_coeff
        self.int_curiosity_adv_coeff = config.int_curiosity_adv_coeff
        self.int_entropy_adv_coeff   = config.int_entropy_adv_coeff
   
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip

        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        
        self.training_epochs    = config.training_epochs
        self.actors             = config.actors

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        self.model_ppo          = ModelPPO.Model(self.state_shape, self.actions_count)
        self.optimizer_ppo      = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)
 
        self.model_forward          = ModelForward.Model(self.state_shape)
        self.optimizer_forward      = torch.optim.Adam(self.model_forward.parameters(), lr=config.learning_rate_forward)

        self.model_forward_target   = ModelForwardTarget.Model(self.state_shape)

        self.policy_buffer = PolicyBufferIME(self.steps, self.state_shape, self.actions_count, self.actors, self.model_ppo.device)

        self.states = [] 
        for e in range(self.actors):
            self.states.append(self.envs.reset(e))
        
        self.episodic_memory    = EpisodicMemory(config.episodic_memory_size)

        self.model_autoencoder       = ModelAutoencoder.Model(self.state_shape)
        self.optimizer_autoencoder   = torch.optim.Adam(self.model_autoencoder.parameters(), lr=config.learning_rate_autoencoder)

        self.states_running_stats       = RunningStats(self.state_shape, numpy.array(self.states))
        
        self.enable_training()
        self.iterations = 0

        
        self.log_loss_forward           = 0.0
        self.log_loss_autoencoder       = 0.0
        self.log_curiosity              = 0.0
        self.log_entropy                = 0.0
        self.log_advantages             = 0.0
        self.log_curiosity_advatages    = 0.0
        self.log_entropy_advatages      = 0.0


    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False

    def main(self):
        #state to tensor
        states_t            = torch.tensor(self.states, dtype=torch.float).detach().to(self.model_ppo.device)

        #compute model output
        logits_t, values_ext_t, values_curiosity_t, values_entropy_t  = self.model_ppo.forward(states_t)

        states_np       = states_t.detach().to("cpu").numpy()
        logits_np       = logits_t.detach().to("cpu").numpy()
        values_ext_np   = values_ext_t.detach().to("cpu").numpy()
        values_curiosity_np     = values_curiosity_t.detach().to("cpu").numpy()
        values_entropy_np       = values_entropy_t.detach().to("cpu").numpy()
        

        #collect actions
        actions = []
        for e in range(self.actors):
            actions.append(self._sample_action(logits_t[e]))

        #update long term state mean and variance
        self.states_running_stats.update(states_np)

        #execute action
        states, rewards, dones, _ = self.envs.step(actions)

        #curiosity motivation
        curiosity_np         = self._curiosity(states_t).detach().to("cpu").numpy()        
        curiosity_np         = numpy.clip(curiosity_np, -1.0, 1.0)

        #entropy motivation 
        entropy_np          = self._entropy(states_t, dones, self.iterations)
        entropy_np          = numpy.clip(entropy_np, -1.0, 1.0)

        #put into policy buffer
        for e in range(self.actors):            
            if self.enabled_training:
                self.policy_buffer.add(e, states_np[e], logits_np[e], values_ext_np[e], values_curiosity_np[e], values_entropy_np[e], actions[e], rewards[e], curiosity_np[e], entropy_np[e], dones[e])

                if self.policy_buffer.is_full():
                    self.train()

            if dones[e]:
                self.states[e] = self.envs.reset(e)
            else:
                self.states[e] = states[e].copy()

        #collect stats
        k = 0.02
        self.log_curiosity   = (1.0 - k)*self.log_curiosity + k*curiosity_np.mean()
        self.log_entropy     = (1.0 - k)*self.log_entropy   + k*entropy_np.mean()
      
        self.iterations+= 1
        return rewards[0], dones[0]


    
    def save(self, save_path):
        self.model_ppo.save(save_path + "trained/")
        self.model_forward.save(save_path + "trained/")
        self.model_forward_target.save(save_path + "trained/")
        self.model_autoencoder.save(save_path + "trained/")

    def load(self, load_path):
        self.model_ppo.load(load_path + "trained/")
        self.model_forward.load(load_path + "trained/")
        self.model_forward_target.load(load_path + "trained/")
        self.model_autoencoder.load(load_path + "trained/")
 
    def get_log(self):
        result = ""  
        
        result+= str(round(self.log_loss_forward, 7)) + " "
        result+= str(round(self.log_loss_autoencoder, 7)) + " "      
        result+= str(round(self.log_curiosity, 7)) + " "        
        result+= str(round(self.log_entropy, 7)) + " "           
        result+= str(round(self.log_advantages, 7)) + " "         
        result+= str(round(self.log_curiosity_advatages, 7)) + " "
        result+= str(round(self.log_entropy_advatages, 7)) + " "  

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
                    #train forward model, MSE loss
                    curiosity_t         = self._curiosity(states)

                    loss_forward = curiosity_t.mean()
                    self.optimizer_forward.zero_grad()
                    loss_forward.backward()
                    self.optimizer_forward.step()

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
                    self.log_loss_forward       = (1.0 - k)*self.log_loss_forward     + k*loss_forward.detach().to("cpu").numpy()
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
        compute internal curiosity critic loss, as MSE
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
        advantages      = self.ext_adv_coeff*advantages_ext + self.int_curiosity_adv_coeff*advantages_cur + self.int_entropy_adv_coeff*advantages_ent
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
        self.log_curiosity_advatages    = (1.0 - k)*self.log_curiosity_advatages    + k*advantages_cur.mean().detach().to("cpu").numpy()
        self.log_entropy_advatages      = (1.0 - k)*self.log_entropy_advatages      + k*advantages_ent.mean().detach().to("cpu").numpy()
    
        return loss

    def _action_one_hot(self, action_idx_t):
        batch_size = action_idx_t.shape[0]

        action_one_hot_t = torch.zeros((batch_size, self.actions_count))
        action_one_hot_t[range(batch_size), action_idx_t] = 1.0  
        action_one_hot_t = action_one_hot_t.to(self.model_ppo.device)

        return action_one_hot_t

    def _curiosity(self, state_t):
        state_norm_t            = state_t - torch.from_numpy(self.states_running_stats.mean).to(self.model_forward.device)

        features_predicted_t    = self.model_forward(state_norm_t)
        features_target_t       = self.model_forward_target(state_norm_t)

        curiosity_t    = (features_target_t.detach() - features_predicted_t)**2
        curiosity_t    = curiosity_t.mean(dim=1)

        return curiosity_t

    def _reset_episodic_memory(self, env_idx, state_np):
        state_t       = torch.from_numpy(state_np).unsqueeze(0).to(self.model_autoencoder.device).float()

        state_norm_t  = state_t - torch.from_numpy(self.states_running_stats.mean).to(self.model_autoencoder.device)

        features_t    = self.model_autoencoder.eval_features(state_norm_t)
        features_t    = features_t.squeeze(0).detach()
 
        self.episodic_memory[env_idx].reset(features_t) 
              
    def _entropy(self, states_t, dones, iterations): 
        state_norm_t  = states_t - torch.from_numpy(self.states_running_stats.mean).to(self.model_autoencoder.device)
        
        features_t    = self.model_autoencoder.eval_features(state_norm_t)
        features_t    = features_t.squeeze(0).detach()
 
        entropy       = numpy.zeros(self.actors)
        for e in range(self.actors):
            entropy[e] = self.episodic_memory.entropy(features_t[e])

        for e in range(self.actors):
            if dones[e] or self.iterations%self.actors == 0:
                self.episodic_memory.add(features_t[e])
        
        return entropy

    '''
    def visualise(self, states_t):
        state_norm_t  = states_t - torch.from_numpy(self.states_running_stats.mean).to(self.model_autoencoder.device)
    ''' 
