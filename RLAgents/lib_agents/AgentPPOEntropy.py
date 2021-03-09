import numpy
import torch
import time

from torch.distributions import Categorical

from .PolicyBufferIM    import *
from .RunningStats      import *
from .EpisodicMemory    import *
  
class AgentPPOEntropy():
    def __init__(self, envs, ModelPPO, ModelForward, ModelForwardTarget, ModelAutoencoder, Config):
        self.envs = envs

        config = Config.Config()

        self.gamma_ext          = config.gamma_ext
        self.gamma_int          = config.gamma_int
        
        self.ext_adv_coeff      = config.ext_adv_coeff
        self.int_adv_coeff      = config.int_adv_coeff
        self.alpha              = 0.0001
        self.beta               = config.beta

        self.normalize_internal_motivation = config.normalize_internal_motivation

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

        self.policy_buffer = PolicyBufferIM(self.steps, self.state_shape, self.actions_count, self.actors, self.model_ppo.device)

        self.states = []
        self.episodic_memory = [] 
        for e in range(self.actors):
            self.states.append(self.envs.reset(e))
            self.episodic_memory.append(EpisodicMemory(config.episodic_memory_size))
            
        self.model_autoencoder       = ModelAutoencoder.Model(self.state_shape)
        self.optimizer_autoencoder   = torch.optim.Adam(self.model_autoencoder.parameters(), lr=config.learning_rate_autoencoder)

        self.states_running_stats                   = RunningStats(self.state_shape, numpy.array(self.states))
        self.int_curiosity_reward_running_stats     = RunningStats()
        self.int_entropy_reward_running_stats       = RunningStats()
  

        self.enable_training()
        self.iterations = 0

        
        self.loss_forward           = 0.0
        self.curiosity_motivation   = 0.0

        self.loss_autoencoder_mse   = 0.0
        self.loss_autoencoder_ortho = 0.0
        self.entropy_motivation     = 0.0


    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False

    def main(self):
        states_t            = torch.tensor(self.states, dtype=torch.float).detach().to(self.model_ppo.device)
 
        logits_t, values_ext_t, values_int_t  = self.model_ppo.forward(states_t)

        states_np       = states_t.detach().to("cpu").numpy()
        logits_np       = logits_t.detach().to("cpu").numpy()
        values_ext_np   = values_ext_t.detach().to("cpu").numpy()
        values_int_np   = values_int_t.detach().to("cpu").numpy()

        actions = []
        for e in range(self.actors):
            actions.append(self._sample_action(logits_t[e]))

        
        curiosity_np         = self._curiosity(states_t).detach().to("cpu").numpy()
        self.int_curiosity_reward_running_stats.update(curiosity_np)
        if normalize_internal_motivation:
            curiosity_np        = (curiosity_np - self.int_curiosity_reward_running_stats.mean)/self.int_curiosity_reward_running_stats.std

        self._add_episodic_memory(states_t)

        entropy_np          = self._entropy()
        self.int_entropy_reward_running_stats.update(entropy_np)
        if self.normalize_internal_motivation:
            entropy_np          = (entropy_np - self.int_entropy_reward_running_stats.mean)/self.int_entropy_reward_running_stats.std
 
        internal_motivation_np = self.beta*curiosity_np + (1.0 - self.beta)*entropy_np

        states, rewards, dones, _ = self.envs.step(actions)

        for e in range(self.actors):            
            if self.enabled_training:
                self.policy_buffer.add(e, states_np[e], logits_np[e], values_ext_np[e], values_int_np[e], actions[e], rewards[e], internal_motivation_np[e], dones[e])

                if self.policy_buffer.is_full():
                    self.train()

            if dones[e]:
                self.states[e] = self.envs.reset(e)
                self._reset_episodic_memory(e, self.states[e])
            else:
                self.states[e] = states[e].copy()

        k = 0.02
        self.curiosity_motivation   = (1.0 - k)*self.curiosity_motivation + k*curiosity_np.mean()
        self.entropy_motivation     = (1.0 - k)*self.entropy_motivation + k*entropy_np.mean()
      
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
        result+= str(round(self.loss_forward, 7)) + " "
        result+= str(round(self.curiosity_motivation, 7)) + " "
        result+= str(round(self.loss_autoencoder_mse, 7)) + " "
        result+= str(round(self.loss_autoencoder_ortho, 7)) + " "
        result+= str(round(self.entropy_motivation, 7)) + " "
    
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
                states, logits, values_ext, values_int, actions, rewards, dones, returns_ext, returns_int, advantages_ext, advantages_int = self.policy_buffer.sample_batch(self.batch_size, self.model_ppo.device)

                loss = self._compute_loss(states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int)

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
                    state_norm_t  = states - torch.from_numpy(self.states_running_stats.mean).to(self.model_autoencoder.device)

                    state_predicted_t, z_t  = self.model_autoencoder(state_norm_t.detach())

                    #reconstruction loss
                    loss_ae_mse    = (state_norm_t.detach() - state_predicted_t)**2
                    loss_ae_mse    = loss_ae_mse.mean()

                    #orthogonality loss
                    #normalise length to 1
                    #compute dot product
                    z_norm          = z_t/(((torch.sum(z_t**2, dim=1))**0.5) + 0.0000001).unsqueeze(1)
                    z_dot           = torch.mm(z_norm, z_norm.permute(1, 0))
                    loss_ae_ortho   = self.alpha*(z_dot**2).mean()
 
                    #L1 regularisation, dirty but fast
                    #z_t             = torch.abs(z_t)
                    #loss_ae_ortho   = self.alpha*z_t.mean()

                    loss_autoencoder = loss_ae_mse + loss_ae_ortho

                    self.optimizer_autoencoder.zero_grad()
                    loss_autoencoder.backward()
                    self.optimizer_autoencoder.step()

                    k = 0.02
                    self.loss_forward               = (1.0 - k)*self.loss_forward + k*loss_forward.detach().to("cpu").numpy()
                    self.loss_autoencoder_mse       = (1.0 - k)*self.loss_autoencoder_mse + k*loss_ae_mse.detach().to("cpu").numpy()
                    self.loss_autoencoder_ortho     = (1.0 - k)*self.loss_autoencoder_ortho + k*loss_ae_ortho.detach().to("cpu").numpy()

        self.policy_buffer.clear() 

    
    def _compute_loss(self, states, logits, actions,  returns_ext, returns_int, advantages_ext, advantages_int):
        probs_old     = torch.nn.functional.softmax(logits, dim = 1).detach()
        log_probs_old = torch.nn.functional.log_softmax(logits, dim = 1).detach()

        logits_new, values_ext_new, values_int_new  = self.model_ppo.forward(states)

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
        compute internal critic loss, as MSE
        L = (T - V(s))^2
        '''
        values_int_new  = values_int_new.squeeze(1)
        loss_int_value  = (returns_int.detach() - values_int_new)**2
        loss_int_value  = loss_int_value.mean()

        ''' 
        compute actor loss, surrogate loss
        '''
        advantages      = self.ext_adv_coeff*advantages_ext + self.int_adv_coeff*advantages_int

        log_probs_new_  = log_probs_new[range(len(log_probs_new)), actions]
        log_probs_old_  = log_probs_old[range(len(log_probs_old)), actions]
                        
        ratio       = torch.exp(log_probs_new_ - log_probs_old_)
        p1          = ratio*advantages.detach()
        p2          = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages.detach()
        loss_policy = -torch.min(p1, p2)  
        loss_policy = loss_policy.mean()
    
        '''
        compute entropy loss, to avoid greedy strategy
        L = beta*H(pi(s)) = beta*pi(s)*log(pi(s))
        '''
        loss_entropy = (probs_new*log_probs_new).sum(dim = 1)
        loss_entropy = self.entropy_beta*loss_entropy.mean()

        loss = loss_ext_value + loss_int_value + loss_policy + loss_entropy

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

   
    def _add_episodic_memory(self, states_t):
        state_norm_t  = states_t - torch.from_numpy(self.states_running_stats.mean).to(self.model_autoencoder.device)
 
        features_t    = self.model_autoencoder.eval_features(state_norm_t)
        features_t    = features_t.squeeze(0).detach()
 
        for e in range(len(self.episodic_memory)):
            self.episodic_memory[e].add(features_t[e]) 

    def _reset_episodic_memory(self, env_idx, state_np):
        state_t       = torch.from_numpy(state_np).unsqueeze(0).to(self.model_autoencoder.device).float()

        state_norm_t  = state_t - torch.from_numpy(self.states_running_stats.mean).to(self.model_autoencoder.device)

        features_t    = self.model_autoencoder.eval_features(state_norm_t)
        features_t    = features_t.squeeze(0).detach()
 
        self.episodic_memory[env_idx].reset(features_t) 
              
    def _entropy(self): 
        entropy       = numpy.zeros(self.actors)
        for e in range(len(self.episodic_memory)):
            entropy[e] = self.episodic_memory[e].entropy()
        
        return entropy
