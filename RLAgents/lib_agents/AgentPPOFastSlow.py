import numpy
import torch 
from .PolicyBufferIM    import *  
from .RunningStats      import *  
      
class AgentPPOFastSlow():   
    def __init__(self, envs, ModelPPO, ModelAE, config):
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
        self.envs_count         = config.envs_count 

        self.im_alpha               = config.im_alpha
        self.im_beta                = config.im_beta
        self.im_alpha_downsample    = config.im_alpha_downsample
        self.im_beta_downsample     = config.im_beta_downsample

        self.im_noise_level    = config.im_noise_level

        self.normalise_im_mean      = config.normalise_im_mean
        self.normalise_im_std       = config.normalise_im_std

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n


        self.model_ppo      = ModelPPO.Model(self.state_shape, self.actions_count)
        self.optimizer_ppo  = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)
 
        self.model_ae_a     = ModelAE.Model(self.state_shape, self.im_alpha_downsample)
        self.optimizer_ae_a = torch.optim.Adam(self.model_ae_a.parameters(), lr=config.learning_rate_ae)

        self.model_ae_b     = ModelAE.Model(self.state_shape, self.im_beta_downsample)
        self.optimizer_ae_b = torch.optim.Adam(self.model_ae_b.parameters(), lr=config.learning_rate_ae)
 
        self.policy_buffer = PolicyBufferIM(self.steps, self.state_shape, self.actions_count, self.envs_count, self.model_ppo.device, True)

       
        self.rewards_int_running_stats_a    = RunningStats((1, ))
        self.rewards_int_running_stats_b    = RunningStats((1, ))


        #reset envs and fill initial state
        self.states = numpy.zeros((self.envs_count, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.envs_count):
            self.states[e] = self.envs.reset(e).copy()

 
        self.enable_training()
        self.iterations                     = 0 

        self.log_loss_ae_a                  = 0.0
        self.log_loss_ae_b                  = 0.0
        self.log_loss_actor                 = 0.0
        self.log_loss_critic                = 0.0

        self.log_internal_motivation_mean_a   = 0.0
        self.log_internal_motivation_std_a    = 0.0
        
        self.log_internal_motivation_mean_b   = 0.0
        self.log_internal_motivation_std_b    = 0.0
        
        self.log_internal_motivation_mean   = 0.0
        self.log_internal_motivation_std    = 0.0
      

    def enable_training(self):
        self.enabled_training = True
 
    def disable_training(self):
        self.enabled_training = False

    def main(self): 
        #state to tensor
        states_t        = torch.tensor(self.states, dtype=torch.float).detach().to(self.model_ppo.device)

        #compute model output
        logits_t, values_ext_t, values_int_t  = self.model_ppo.forward(states_t)
        
        states_np       = states_t.detach().to("cpu").numpy()
        logits_np       = logits_t.detach().to("cpu").numpy()
        values_ext_np   = values_ext_t.squeeze(1).detach().to("cpu").numpy()
        values_int_np   = values_int_t.squeeze(1).detach().to("cpu").numpy()

        #collect actions
        actions = self._sample_actions(logits_t)
         
        #execute action
        states, rewards_ext, dones, infos = self.envs.step(actions)

        self.states = states.copy()
 
        #curiosity motivation
        rewards_int_a, rewards_int_b  = self._curiosity(states_t)
        
        self.rewards_int_running_stats_a.update(rewards_int_a)
        self.rewards_int_running_stats_b.update(rewards_int_b)

        #normalise internal motivation
        if self.normalise_im_mean:
            rewards_int_a  = rewards_int_a - self.rewards_int_running_stats_a.mean
            rewards_int_b  = rewards_int_b - self.rewards_int_running_stats_b.mean

        if self.normalise_im_std:
            rewards_int_a   = rewards_int_a/self.rewards_int_running_stats_a.std
            rewards_int_b   = rewards_int_b/self.rewards_int_running_stats_b.std


        rewards_int    = self.im_alpha*rewards_int_a + self.im_beta*rewards_int_b

        rewards_int    = numpy.clip(rewards_int, 0.0, 1.0)
        
        #put into policy buffer
        if self.enabled_training:
            self.policy_buffer.add(states_np, logits_np, values_ext_np, values_int_np, actions, rewards_ext, rewards_int, dones)

            if self.policy_buffer.is_full():
                self.train()
        
        for e in range(self.envs_count): 
            if dones[e]:
                self.states[e] = self.envs.reset(e).copy()

        #collect stats
        k = 0.02
        self.log_internal_motivation_mean_a   = (1.0 - k)*self.log_internal_motivation_mean_a + k*rewards_int_a.mean()
        self.log_internal_motivation_std_a    = (1.0 - k)*self.log_internal_motivation_std_a  + k*rewards_int_a.std()

        self.log_internal_motivation_mean_b   = (1.0 - k)*self.log_internal_motivation_mean_b + k*rewards_int_b.mean()
        self.log_internal_motivation_std_b    = (1.0 - k)*self.log_internal_motivation_std_b  + k*rewards_int_b.std()
        
        self.log_internal_motivation_mean   = (1.0 - k)*self.log_internal_motivation_mean + k*rewards_int.mean()
        self.log_internal_motivation_std    = (1.0 - k)*self.log_internal_motivation_std  + k*rewards_int.std()

        self.iterations+= 1
        return rewards_ext[0], dones[0], infos[0]
    
    def save(self, save_path):
        self.model_ppo.save(save_path + "trained/")
        self.model_ae_a.save(save_path + "trained/")
        self.model_ae_b.save(save_path + "trained/")

    def load(self, load_path):
        self.model_ppo.load(load_path + "trained/")
        self.model_ae_a.load(load_path + "trained/")
        self.model_ae_b.load(load_path + "trained/")

    def get_log(self): 
        result = "" 

        result+= str(round(self.log_loss_actor, 7)) + " "
        result+= str(round(self.log_loss_critic, 7)) + " "

        result+= str(round(self.log_loss_ae_a, 7)) + " "
        result+= str(round(self.log_loss_ae_b, 7)) + " "

        result+= str(round(self.log_internal_motivation_mean_a, 7)) + " "
        result+= str(round(self.log_internal_motivation_std_a, 7)) + " "

        result+= str(round(self.log_internal_motivation_mean_b, 7)) + " "
        result+= str(round(self.log_internal_motivation_std_b, 7)) + " "

        result+= str(round(self.log_internal_motivation_mean, 7)) + " "
        result+= str(round(self.log_internal_motivation_std, 7)) + " "

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
                states, _, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int = self.policy_buffer.sample_batch(self.batch_size, self.model_ppo.device)

                #train PPO model
                loss_ppo = self._compute_loss_ppo(states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int)

                self.optimizer_ppo.zero_grad()        
                loss_ppo.backward()
                torch.nn.utils.clip_grad_norm_(self.model_ppo.parameters(), max_norm=0.5)
                self.optimizer_ppo.step()

        k    = 0.02
        step = 0

        ae_batch_size = 32

        batch_count = self.steps//ae_batch_size
        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):

                states = self.policy_buffer.sample_states(ae_batch_size, self.model_ppo.device)
         
                if step%self.im_alpha_downsample == 0:
                    #train ae model A, MSE loss
                    prediction_a_t, _, = self.model_ae_a(states, self.im_noise_level)
                    loss_ae_a = ((states.detach() - prediction_a_t)**2).mean()
                    
                    self.optimizer_ae_a.zero_grad() 
                    loss_ae_a.backward()
                    self.optimizer_ae_a.step()

                    self.log_loss_ae_a  = (1.0 - k)*self.log_loss_ae_a + k*loss_ae_a.detach().to("cpu").numpy()

                if step%self.im_beta_downsample == 0:
                    #train ae model B, MSE loss
                    prediction_b_t, _, = self.model_ae_b(states, self.im_noise_level)
                    loss_ae_b = ((states.detach() - prediction_b_t)**2).mean()
                    
                    self.optimizer_ae_b.zero_grad() 
                    loss_ae_b.backward()
                    self.optimizer_ae_b.step()

                    self.log_loss_ae_b  = (1.0 - k)*self.log_loss_ae_b + k*loss_ae_b.detach().to("cpu").numpy()

                step+= 1

        self.policy_buffer.clear() 

    
    def _compute_loss_ppo(self, states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int):
        logits_new, values_ext_new, values_int_new  = self.model_ppo.forward(states)

        #critic loss
        loss_critic = self._compute_critic_loss(values_ext_new, returns_ext, values_int_new, returns_int)

        #actor loss        
        advantages  = self.ext_adv_coeff*advantages_ext + self.int_adv_coeff*advantages_int
        advantages  = advantages.detach() 
        loss_policy, loss_entropy  = self._compute_actor_loss(logits, logits_new, advantages, actions)

        loss_actor = loss_policy + loss_entropy
        
        #total loss
        loss = 0.5*loss_critic + loss_actor

        #store to log
        k = 0.02
        self.log_loss_actor     = (1.0 - k)*self.log_loss_actor  + k*loss_actor.mean().detach().to("cpu").numpy()
        self.log_loss_critic    = (1.0 - k)*self.log_loss_critic + k*loss_critic.mean().detach().to("cpu").numpy()

        return loss 

    #MSE critic loss
    def _compute_critic_loss(self, values_ext_new, returns_ext, values_int_new, returns_int):
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
        return loss_critic

    #PPO actor loss
    def _compute_actor_loss(self, logits, logits_new, advantages, actions):
        log_probs_old = torch.nn.functional.log_softmax(logits, dim = 1).detach()

        probs_new     = torch.nn.functional.softmax(logits_new, dim = 1)
        log_probs_new = torch.nn.functional.log_softmax(logits_new, dim = 1)

        ''' 
        compute actor loss, surrogate loss
        '''
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

        return loss_policy, loss_entropy

    
    #compute internal motivation
    def _curiosity(self, state_t):
        _, im_a_t   = self.model_ae_a(state_t, self.im_noise_level)
        _, im_b_t   = self.model_ae_b(state_t, self.im_noise_level)

        return im_a_t.detach().to("cpu").numpy(), im_b_t.detach().to("cpu").numpy()

  
  