import numpy
import torch
import time

from torch.distributions import Categorical
 
from .PolicyBufferIM    import * 
from .CABuffer          import * 

from .RunningStats      import * 

 
class AgentPPOCSA():   
    def __init__(self, envs, ModelPPO, ModelRND, ModelCA, config):
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
        self.envs_count      = config.envs_count 

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        self.model_ppo      = ModelPPO.Model(self.state_shape, self.actions_count)
        self.optimizer_ppo  = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)

        self.model_rnd      = ModelRND.Model(self.state_shape)
        self.optimizer_rnd  = torch.optim.Adam(self.model_rnd.parameters(), lr=config.learning_rate_rnd)
 
        self.model_ca      = ModelCA.Model(self.state_shape, self.actions_count)
        self.optimizer_ca  = torch.optim.Adam(self.model_ca.parameters(), lr=config.learning_rate_ca)
 
        self.policy_buffer = PolicyBufferIM(self.steps, self.state_shape, self.actions_count, self.envs_count, self.model_ppo.device, True)
        self.ca_buffer     = CABuffer(config.ca_buffer_size, self.state_shape, config.ca_add_threshold, config.ca_downsample, self.model_ppo.device)

        self.states = numpy.zeros((self.envs_count, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.envs_count):
            self.states[e] = self.envs.reset(e).copy()

        self.states_running_stats       = RunningStats(self.state_shape, self.states)

        self.enable_training()
        self.iterations                 = 0 

        self.log_internal_motivation_rnd    = 0.0
        self.log_internal_motivation_ca     = 0.0
        self.log_ca_buffer_usage            = 0.0
        self.log_action_prediction          = 0.0

        self.log_loss_rnd                   = 0.0
        self.log_loss_ca                    = 0.0
        self.log_advantages_ext             = 0.0
        self.log_advantages_int             = 0.0

    def enable_training(self):
        self.enabled_training = True
 
    def disable_training(self):
        self.enabled_training = False

    def main(self): 
        #state to tensor
        states_t            = torch.tensor(self.states, dtype=torch.float).detach().to(self.model_ppo.device)

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

        #update long term states mean and variance
        self.states_running_stats.update(states_np)

        #internal RND motivation
        rewards_int_rnd    = self._curiosity(states_t)
        rewards_int_rnd    = numpy.clip(rewards_int_rnd, 0.0, 1.0)

        #internal CA motivation
        attention_t     = self.model_ca.forward(states_t)
        rewards_int_ca  = self.ca_buffer.add(states_t, attention_t)
        
        #combine internal rewards
        rewards_int = rewards_int_rnd + rewards_int_ca

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
        self.log_internal_motivation_rnd    = (1.0 - k)*self.log_internal_motivation_rnd + k*rewards_int_rnd.mean()
        self.log_internal_motivation_ca     = (1.0 - k)*self.log_internal_motivation_ca  + k*rewards_int_ca.mean()
        self.log_ca_buffer_usage            = (1.0 - k)*self.log_ca_buffer_usage         + k*self.ca_buffer.current_idx

        self.iterations+= 1
        return rewards_ext[0], dones[0], infos[0]
    
    def save(self, save_path):
        self.model_ppo.save(save_path + "trained/")
        self.model_rnd.save(save_path + "trained/")
        self.model_ca.save(save_path + "trained/")

    def load(self, load_path):
        self.model_ppo.load(load_path + "trained/")
        self.model_rnd.load(load_path + "trained/")
        self.model_ca.load(load_path + "trained/")

    def get_log(self): 
        result = "" 
        result+= str(round(self.log_internal_motivation_rnd, 7)) + " "
        result+= str(round(self.log_internal_motivation_ca, 7)) + " "

        result+= str(round(self.log_ca_buffer_usage, 7)) + " "

        result+= str(round(self.log_loss_rnd, 7)) + " "
        result+= str(round(self.log_loss_ca, 7)) + " "

        result+= str(round(self.log_action_prediction, 7)) + " "
        result+= str(round(self.log_advantages_ext, 7)) + " "
        result+= str(round(self.log_advantages_int, 7)) + " "

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
                states, states_next, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int = self.policy_buffer.sample_batch(self.batch_size, self.model_ppo.device)

                #train PPO model
                loss_ppo = self._compute_loss_ppo(states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int)

                self.optimizer_ppo.zero_grad()        
                loss_ppo.backward()
                torch.nn.utils.clip_grad_norm_(self.model_ppo.parameters(), max_norm=0.5)
                self.optimizer_ppo.step()

                #train RND model, MSE loss
                loss_rnd = self._compute_loss_rnd(states)

                self.optimizer_rnd.zero_grad() 
                loss_rnd.backward()
                self.optimizer_rnd.step()

                #train CA model for action prediction
                loss_ca,  acc_ca = self._compute_loss_ca(states, states_next, actions)

                self.optimizer_ca.zero_grad() 
                loss_ca.backward()
                self.optimizer_ca.step()

                #update logs
                k = 0.02
                self.log_loss_rnd           = (1.0 - k)*self.log_loss_rnd   + k*loss_rnd.detach().to("cpu").numpy()
                self.log_loss_ca            = (1.0 - k)*self.log_loss_ca    + k*loss_ca.detach().to("cpu").numpy()
                self.log_action_prediction  = (1.0 - k)*self.log_action_prediction + k*acc_ca

        self.policy_buffer.clear() 

    
    def _compute_loss_ppo(self, states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int):
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
        self.log_advantages_ext     = (1.0 - k)*self.log_advantages_ext + k*advantages_ext.mean().detach().to("cpu").numpy()
        self.log_advantages_int     = (1.0 - k)*self.log_advantages_int + k*advantages_int.mean().detach().to("cpu").numpy()

        return loss 

    def _compute_loss_rnd(self, states):
        #MSE loss for RND model
        state_norm_t    = self._norm_state(states).detach()

        features_predicted_t, features_target_t  = self.model_rnd(state_norm_t)

        loss_rnd        = (features_target_t - features_predicted_t)**2
        
        #75% regularisation
        random_mask     = torch.rand(loss_rnd.shape).to(loss_rnd.device)
        random_mask     = 1.0*(random_mask < 0.25)
        loss_rnd        = (loss_rnd*random_mask).sum() / (random_mask.sum() + 0.00000001)

        return loss_rnd

    def _compute_loss_ca(self, states, states_next, actions):
        #one hot actions encoding
        actions_target_t = torch.zeros((states.shape[0], self.actions_count)).to(self.model_ca.device)
        actions_target_t[range(states.shape[0]), actions] = 1.0

        action_pred_t    = self.model_ca.forward_inverse(states, states_next)

        loss_ca = ((actions_target_t - action_pred_t)**2).mean()

        #accuracy stats
        target_indices = numpy.argmax(actions_target_t.detach().to("cpu").numpy(), axis=1)
        pred_indices   = numpy.argmax(action_pred_t.detach().to("cpu").numpy(), axis=1)
        
        hit     = (target_indices == pred_indices).sum()
        miss    = (target_indices != pred_indices).sum()

        acc     = 100.0*hit/(hit + miss)

        return loss_ca, acc


    def _curiosity(self, state_t):
        state_norm_t            = self._norm_state(state_t)

        features_predicted_t, features_target_t  = self.model_rnd(state_norm_t)

        curiosity_t    = (features_target_t - features_predicted_t)**2
        
        curiosity_t    = curiosity_t.sum(dim=1)/2.0
        
        return curiosity_t.detach().to("cpu").numpy()

    def _norm_state(self, state_t):
        mean = torch.from_numpy(self.states_running_stats.mean).to(state_t.device).float()
        state_norm_t = state_t - mean 
        
        return state_norm_t
