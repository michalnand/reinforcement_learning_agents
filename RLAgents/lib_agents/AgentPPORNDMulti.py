import numpy
import torch
import time

from torch.distributions import Categorical
 
from .PolicyBufferIMMulti   import *  
from .RunningStats          import * 
    
class AgentPPORNDMulti():   
    def __init__(self, envs, ModelPPO, ModelRND, config):
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

        self.rnd_heads      = config.rnd_heads
        self.rnd_steps      = config.rnd_steps

        self.model_ppo      = ModelPPO.Model(self.state_shape, self.actions_count)
        self.optimizer_ppo  = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)
 
        self.model_rnd      = ModelRND.Model(self.state_shape, self.rnd_heads)
        self.optimizer_rnd  = torch.optim.Adam(self.model_rnd.parameters(), lr=config.learning_rate_rnd)
 
        self.policy_buffer = PolicyBufferIMMulti(self.steps, self.state_shape, self.actions_count, self.envs_count, self.model_ppo.device, True)
 
        self.states = numpy.zeros((self.envs_count, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.envs_count):
            self.states[e] = self.envs.reset(e).copy()

        self.states_running_stats       = RunningStats(self.state_shape, self.states)

        self.episode_score_sum  = numpy.zeros(self.envs_count)
 
        self.enable_training()
        self.iterations                 = 0 

        self.log_loss_rnd               = 0.0
        self.log_internal_motivation    = 0.0
        self.log_advantages_ext         = 0.0
        self.log_advantages_int         = 0.0
        self.log_heads_usage            = numpy.zeros(self.rnd_heads)

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

        self.episode_score_sum+= rewards_ext

        rnd_head_ids = self._rnd_heads_ids(self.episode_score_sum)

        self.states = states.copy()
 
        #update long term states mean and variance
        self.states_running_stats.update(states_np)

        #curiosity motivation
        rewards_int     = self._curiosity(states_t, rnd_head_ids)
        rewards_int     = numpy.clip(rewards_int, -1.0, 1.0)
         
        #put into policy buffer
        if self.enabled_training:
            self.policy_buffer.add(states_np, logits_np, values_ext_np, values_int_np, actions, rewards_ext, rewards_int, dones, rnd_head_ids)

            if self.policy_buffer.is_full():
                self.train()
        
        for e in range(self.envs_count): 
            if dones[e]:
                self.states[e]              = self.envs.reset(e).copy()
                self.episode_score_sum[e]   = 0

        #collect stats
        k = 0.02
        self.log_internal_motivation = (1.0 - k)*self.log_internal_motivation + k*rewards_int.mean()

        heads_usage = numpy.zeros(self.rnd_heads)
        for h in range(self.rnd_heads):
            heads_usage[h] = (rnd_head_ids == h).sum()
        heads_usage = heads_usage/heads_usage.sum()

        self.log_heads_usage = (1.0 - k)*self.log_heads_usage + k*heads_usage

        self.iterations+= 1
        return rewards_ext[0], dones[0], infos[0]
    
    def save(self, save_path):
        self.model_ppo.save(save_path + "trained/")
        self.model_rnd.save(save_path + "trained/")

    def load(self, load_path):
        self.model_ppo.load(load_path + "trained/")
        self.model_rnd.load(load_path + "trained/")

    def get_log(self): 
        result = "" 
        result+= str(round(self.log_loss_rnd, 7)) + " "
        result+= str(round(self.log_internal_motivation, 7)) + " "
        result+= str(round(self.log_advantages_ext, 7)) + " "
        result+= str(round(self.log_advantages_int, 7)) + " "

        for h in range(self.rnd_heads):
            result+= str(round(self.log_heads_usage[h], 3)) + " "

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
                states, _, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int, rnd_head_ids = self.policy_buffer.sample_batch(self.batch_size, self.model_ppo.device)

                #train PPO model
                loss_ppo = self._compute_loss_ppo(states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int)

                self.optimizer_ppo.zero_grad()        
                loss_ppo.backward()
                torch.nn.utils.clip_grad_norm_(self.model_ppo.parameters(), max_norm=0.5)
                self.optimizer_ppo.step()

                #train RND model, MSE loss
                loss_rnd = self._compute_loss_rnd(states, rnd_head_ids)

                self.optimizer_rnd.zero_grad() 
                loss_rnd.backward()
                self.optimizer_rnd.step()

                k = 0.02
                self.log_loss_rnd  = (1.0 - k)*self.log_loss_rnd + k*loss_rnd.detach().to("cpu").numpy()

        self.policy_buffer.clear() 

    
    def _compute_loss_ppo(self, states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int):
        log_probs_old = torch.nn.functional.log_softmax(logits, dim = 1).detach()

        logits_new, values_ext_new, values_int_new  = self.model_ppo.forward(states)

        #critic loss
        loss_critic = self._compute_critic_loss(values_ext_new, returns_ext, values_int_new, returns_int)

        #actor loss
        advantages  = self.ext_adv_coeff*advantages_ext + self.int_adv_coeff*advantages_int
        advantages  = advantages.detach() 
        loss_policy, loss_entropy  = self._compute_actor_loss(log_probs_old, logits_new, advantages, actions)
        
        loss = 0.5*loss_critic + loss_policy + loss_entropy

        k = 0.02
        self.log_advantages_ext     = (1.0 - k)*self.log_advantages_ext + k*advantages_ext.mean().detach().to("cpu").numpy()
        self.log_advantages_int     = (1.0 - k)*self.log_advantages_int + k*advantages_int.mean().detach().to("cpu").numpy()

        return loss 

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


    def _compute_actor_loss(self, log_probs_old, logits_new, advantages, actions):
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


    def _compute_loss_rnd(self, states, heads_ids):
        #MSE loss for RND model
        state_norm_t    = self._norm_state(states).detach()

        features_predicted_t, features_target_t  = self.model_rnd(state_norm_t, heads_ids)

        loss_rnd        = (features_target_t - features_predicted_t)**2 
        
        #regularisation
        prob            = 32.0/self.envs_count
        random_mask     = torch.rand(loss_rnd.shape).to(loss_rnd.device)
        random_mask     = 1.0*(random_mask < prob)
        loss_rnd        = (loss_rnd*random_mask).sum() / (random_mask.sum() + 0.00000001)

        return loss_rnd

    def _rnd_heads_ids(self, episode_score_sum):
        tmp = numpy.floor(episode_score_sum/self.rnd_steps)
        tmp = tmp.astype(int)%self.rnd_heads
        return tmp
        
    def _curiosity(self, state_t, heads_ids):
        state_norm_t    = self._norm_state(state_t)

        head_ids_t    = torch.from_numpy(heads_ids).to(state_norm_t.device)

        features_predicted_t, features_target_t  = self.model_rnd(state_norm_t, head_ids_t)

        curiosity_t    = (features_target_t - features_predicted_t)**2
        curiosity_t    = curiosity_t.sum(dim=1)/2.0
        
        return curiosity_t.detach().to("cpu").numpy()

    def _norm_state(self, state_t):
        mean = torch.from_numpy(self.states_running_stats.mean).to(state_t.device).float()

        state_norm_t = state_t - mean 
        return state_norm_t
